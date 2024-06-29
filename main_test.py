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

    n = start + 1
    while not engine.terminate():
        n += 1
        engine.train()
        if args.test_every != 0 and n % args.test_every == 0:
            engine.test()
        elif n == args.epochs:
            engine.test()

#print(args)
#  default args = Namespace(nThread=4, cpu=False, nGPU=1, config='',
#  datadir='Market-1501-v15.09.15', data_train='Market1501',
#  data_test='Market1501', cuhk03_labeled=False, epochs=80,
#  test_every=20, batchid=16, batchimage=4, batchtest=32,
#  test_only=False, sampler=True, model='LMBN_n', loss='1*CrossEntropy+1*Triplet',
#  if_labelsmooth=False, bnneck=False, feat_inference='after',
#  drop_block=False, w_ratio=1.0, h_ratio=0.3, act='relu', pool='avg',
#  feats=512, height=384, width=128, num_classes=751, T=1,
#  num_anchors=2, lr=0.0006, optimizer='ADAM', momentum=0.9,
#  dampening=0, nesterov=False, beta1=0.9, beta2=0.999, amsgrad=False,
#  epsilon=1e-08, gamma=0.1, weight_decay=0.0005, decay_type='step',
#  lr_decay=60, warmup='constant', pcb_different_lr=True, cosine_annealing=False,
#  w_cosine_annealing=False, parts=6, margin=1.2, re_rank=False,
#  cutout=False, random_erasing=False, probability=0.5, save='test',
#  load='', pre_train='', activation_map=False, nep_token='',
#  nep_id='', nep_name='x.ji/mcmp', reset=False, wandb=False, wandb_name='')

#  train
#  datadir = path, 
#  data_train = data_set, 
#  data_test = data_set,
#  batchid = num,
#  batchimage = num,
#  batchtest = num,
#  test_every = num.
#  loss = LossFn  #e.g. 0.5*CrossEntropy+0.5*MSLoss
#  nGPU =  1,
#  lr = 6e-4
#  epochs = 100
#  optimizer = optimizer,
#  save = 'path'
#  random_erasing if_labelsmooth w_cosine_annealing = True

#  test
#  test_only = True
#  config = 'path'
#  pre_train = 'path'


if __name__ == "__main__":
    
    args.nThread = 8
    args.nGPU = 1
    curPath = os.path.abspath(os.path.dirname(__file__))
    dataPath = os.path.dirname(curPath)
    dataPath = os.path.join(dataPath, 'ReIDataset')

    #train
    args.datadir = dataPath
    args.data_train = 'MyData'
    args.data_test = 'MyData'
    args.batchid = 4
    args.batchimage = 6
    #6G vRAM
    args.batchtest = 32
    args.test_every = 64
    args.epochs = 5   # 120
    args.save = 'demo'
    args.model = 'LMBN_n'
    args.num_classes = 589
    args.random_erasing = True
    args.if_labelsmooth = True
    args.w_cosine_annealing = True

    #test
    # args.test_only = True
    # testPath = curPath #os.path.join(curPath, 'experiment/demo')
    # args.config = os.path.join(testPath, 'cfg_lmbn_n_market.yaml')
    # args.pre_train = os.path.join(testPath, 'lmbn_n_market.pth')

    main()

