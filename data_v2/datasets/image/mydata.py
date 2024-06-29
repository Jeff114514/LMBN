from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings


from .. import ImageDataset

class MyData(ImageDataset):

    def __init__(self, **kwargs):
        self.cur_path = osp.dirname(osp.abspath(__file__))
        self.data_path = osp.join(self.cur_path, '..', '..', '..', 'MyData', 'satimg_train')
        train_list = os.listdir(self.data_path)
        train_data = []
        for lable in train_list:
            #匹配__之后的数字
            pid = int(re.findall(r"__(.*)", lable)[0])
            for img in glob.glob(os.path.join(self.data_path, lable, '*.jpeg')):
                train_data.append([img, pid, 1])
            for img in glob.glob(os.path.join(self.data_path, lable, '*.png')):
                train_data.append([img, pid, 1])
        #按照pid排序
        train_data = sorted(train_data, key=lambda x: x[1])
        #从0开始重新编号
        train_data_refreash = []
        num = 0
        for i, data in enumerate(train_data):
            if i != 0 and train_data[i][1] != train_data[i-1][1]:
                num += 1
            train_data_refreash.append((data[0], num, data[2]))

        
        #just test
        query_data = []
        self.query_path = osp.join(self.cur_path, '..', '..', '..', 'MyData', 'satellite')
        query_lables = os.listdir(self.query_path)
        query_ids = [re.findall(r"__(.*)", lable)[0] for lable in query_lables]
        #去除ids结尾的.jpg并转为int
        query_ids = [int(i.split('.')[0]) for i in query_ids]

        for i, name in enumerate(glob.glob(os.path.join(self.query_path, '*.jpg'))):
            query_data.append([name, query_ids[i], 1])

        query_data = sorted(query_data, key=lambda x: x[1])
        query_data_refreash = []
        num = 0
        for i, data in enumerate(query_data):
            if i != 0 and query_data[i][1] != query_data[i-1][1]:
                num += 1
            query_data_refreash.append((data[0], num, data[2]))
        
        gallery_data = train_data_refreash
        super(MyData, self).__init__(train_data_refreash, query_data_refreash, gallery_data, **kwargs)
        

# if __name__ == '__main__':
#     cur_path = osp.dirname(osp.abspath(__file__))
#     data_path = osp.join(cur_path, '..', '..', '..', 'MyData', 'satellite')
#     #读取文件名称
#     data_names = os.listdir(data_path)
#     for i, names in enumerate(glob.glob(os.path.join(data_path, '*.jpg'))):
#         print(names, data_names[i])
