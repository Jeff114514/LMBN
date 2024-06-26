#import data_v1
import data_v2
from loss import make_loss
from model import make_model
from optim import make_optimizer, make_scheduler

# import engine_v1
# import engine_v2
import engine_v3
import os.path as osp
from option import args
import utils.utility as utility
from utils.model_complexity import compute_model_complexity
from torch.utils.collect_env import get_pretty_env_info
import yaml
import torch
import os
import cv2

def get_data_by_indexList(loader, ind):
    size = args.batchtest
    datas, ids = {}, {}
    for i, data in enumerate(loader):
        for j in range(size):
            lable = i * size + j
            if lable in ind:
                datas[lable] = data[0][j]
                ids[lable] = data[1][j]
    img, id = [], []
    for i in ind:
        img.append(datas[i])
        id.append(ids[i])
    return img, id


def main():

    if args.config != "":
        with open(args.config, "r") as f:
            config = yaml.full_load(f)
        for op in config:
            setattr(args, op, config[op])
    torch.backends.cudnn.benchmark = True

    # loader = data.Data(args)
    ckpt = utility.checkpoint(args)
    loader = data_v2.ImageDataManager(args)
    model = make_model(args, ckpt)
    optimzer = make_optimizer(args, model)
    loss = make_loss(args, ckpt) if not args.test_only else None

    start = -1
    if args.load != "":
        start, model, optimizer = ckpt.resume_from_checkpoint(
            osp.join(ckpt.dir, "model-latest.pth"), model, optimzer
        )
        start = start - 1
    if args.pre_train != "":
        ckpt.load_pretrained_weights(model, args.pre_train)

    scheduler = make_scheduler(args, optimzer, start)

    # print('[INFO] System infomation: \n {}'.format(get_pretty_env_info()))
    ckpt.write_log(
        "[INFO] Model parameters: {com[0]} flops: {com[1]}".format(
            com=compute_model_complexity(model, (1, 3, args.height, args.width))
        )
    )

    engine = engine_v3.Engine(args, model, optimzer, scheduler, loss, loader, ckpt)
    # engine = engine.Engine(args, model, loss, loader, ckpt)


    simMat = engine.test_show()
    #print(simMat)

    query_loader = loader.query_loader
    gallary_loader = loader.test_loader

    for i in range(5):
        qImg, qId = get_data_by_indexList(query_loader, [i])
        gImgList, gIdList = get_data_by_indexList(gallary_loader, simMat[i])

        cv2.imshow('query', qImg[0].permute(1, 2, 0).numpy())
        cv2.imshow('gImg1', gImgList[0].permute(1, 2, 0).numpy())
        cv2.imshow('gImg2', gImgList[1].permute(1, 2, 0).numpy())
        cv2.imshow('gImg3', gImgList[2].permute(1, 2, 0).numpy())
        cv2.imshow('gImg4', gImgList[3].permute(1, 2, 0).numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        






if __name__ == "__main__":
    
    args.nThread = 8
    args.nGPU = 1
    curPath = os.path.abspath(os.path.dirname(__file__))
    dataPath = os.path.dirname(curPath)
    dataPath = os.path.join(dataPath, 'ReIDataset')

    #test
    args.test_only = True
    testPath = curPath #os.path.join(curPath, 'experiment/demo')
    args.config = os.path.join(testPath, 'cfg_lmbn_n_market.yaml')
    args.pre_train = os.path.join(testPath, 'lmbn_n_market.pth')

    if args.config != "":
        with open(args.config, "r") as f:
            config = yaml.full_load(f)
        for op in config:
            setattr(args, op, config[op])
    torch.backends.cudnn.benchmark = True

    # loader = data_v2.ImageDataManager(args)
    # query_loader = loader.query_loader
    # gallary_loader = loader.test_loader
    # img, ids = get_data_by_indexList(query_loader, [1])
    # print(img[0].shape)
    # cv2.imshow('query', img[0].permute(1, 2, 0).numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    main()
