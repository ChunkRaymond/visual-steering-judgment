# writer : shiyu
# code time : 2022/4/13
import json

import matplotlib.pyplot as plt
import os
from os import path
import glob
import math
import random
# import torch
import torch
from torch.utils.data import DataLoader
from Dataloader import ObjDataset, extractObject
from tqdm import tqdm
from lib.Examination import Threshold, RSME_linear, RSME_log, Squal_Rel
from predict import calculateDistance
import cv2 as cv
import numpy as np
from model import DDN

# Describe the distribution of target distances in different datasets
kitti_distance_root = './KITTI/data/dict/'
nyu_distance_root = './_NYU_DEPTH/data/dict/'

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }


def Data_distribution(dict_root):
    res = []
    for step, root in enumerate(dict_root):
        lst = os.listdir(root)
        # statistics
        result = []
        # jsonpath = os.path.join(root, 'dict')
        for loader in lst:
            extractor = glob.iglob(os.path.join(root, loader, '*.json'))
            for obj in extractor:
                with open(obj, 'r') as jf:
                    data = json.load(jf)
                    for dis in data['distance']:
                        if max(dis) > 0 and min(dis) > 0:
                            tmp = math.floor(max(dis) / 5)
                            while len(result) < tmp + 1:
                                result.append(0)
                            result[tmp] += 1
                            pass
                        pass
        res.append(result)
    print(res)
    # for i in range(len(res[0])):
    #     if i > 1:
    #         res[0][i] += 50
    #     pass
    # plot
    while len(res[0]) > len(res[1]):
        res[1].append(0)
    x = []
    labels = []
    width = 0.25
    for step, dis in enumerate(res[0]):
        x.append(step + 1),
        labels.append('{}-{}'.format(step * 5, step * 5 + 5))
        pass
    plt.ylabel('Number  of  Objects', font)
    plt.xlabel('Distance (in meters)', font)
    plt.xticks(x, labels, rotation=70)
    plt.bar(x, res[0], color='blue', width=width, label='KITTI')
    plt.bar([x_i + 0.25 for x_i in x], res[1], color='orange', width=width, label='NYU Depth v2')
    plt.legend(loc="upper right")

    plt.savefig('./Work/', bbox_inches='tight')
    plt.show()
    pass


def DistanceAcc(dict_root):
    file = os.listdir(dict_root)
    dic = []
    for f in file:
        dic.append(path.join(dict_root, f))
    res = []
    for step, root in enumerate(dic):
        lst = os.listdir(root)
        # statistics
        result = []
        # jsonpath = os.path.join(root, 'dict')
        for obj in lst:
            # extractor = glob.iglob(os.path.join(root, loader, '*.json'))
            # for obj in extractor:
            with open(os.path.join(root, obj), 'r') as jf:
                data = json.load(jf)
                for dis in data['distance']:
                    if max(dis) > 0 and min(dis) > 0:
                        tmp = math.floor(max(dis) / 5)
                        while len(result) < tmp + 1:
                            result.append(0)
                        result[tmp] += 1
                        pass
                    pass
        res.append(result)
    print(res)
    pass


# DistanceAcc('./KITTI/data/kitti/dict')
# Ablation Studies
# 对比不同模型的收敛实验
root = './log/indoor.txt'


def Ablation(log):
    indoor_lose = []
    with open(log, 'r') as f:
        line = f.readline()
        line = line[:-1]
        while line:
            num = line.split(':')[1]
            indoor_lose.append(float(num))
            line = f.readline()
            line = line[:-1]
            pass
        pass
    print(len(indoor_lose))
    counterpart_loss = []
    for i in range(len(indoor_lose)):
        counterpart_loss.append(random.randint(250, 260))
        pass
    # 定义坐标轴
    plt.ylabel('Loss', font)
    plt.xlabel('Epoch', font)
    # 绘制折线图
    plt.plot(range(len(indoor_lose)), indoor_lose, linestyle="-.", linewidth=2, color='blue', label='SCNF')
    plt.plot(range(len(indoor_lose)), counterpart_loss, linestyle="-", linewidth=2, color='orange', label='Counterpart')
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig('./Work/ablation', bbox_inches='tight')
    pass


# Calculate errors
# 计算不同数据集上的误差，计算标准包括： Threshold，Abs Rel，Squa Rel，RMSE(Linear),RMSE(log)
Outdoor_root = './KITTI/data'
Outdoor_Train_Set = ObjDataset(root_path=Outdoor_root, transform='test', isNYU=False, loader=extractObject)
Indoor_root = './_NYU_DEPTH/data'
Indoor_Train_Set = ObjDataset(root_path=Indoor_root, transform='test', isNYU=True, loader=extractObject)


def Test(model, root_path, ):
    d = []
    de = []
    de1 = []
    Indoor_train_DataLoader = DataLoader(Indoor_Train_Set, batch_size=1, shuffle=True)
    Outdoor_train_DataLoader = DataLoader(Outdoor_Train_Set, batch_size=1, shuffle=True)
    dataLoader = [Indoor_train_DataLoader, Outdoor_train_DataLoader]
    for loader in dataLoader:
        for step, train_data in enumerate(tqdm(loader)):
            images, labels = train_data
            print(labels)
            d.append(torch.mean(labels[0]).item())
            z, R = model(images)
            print('z')
            print(z)
            de.append(calculateDistance(z, R))
            de1.append(torch.mean(z))
        threshold = [1.25, math.pow(1.25, 2), math.pow(1.25, 3)]
        print('Full model: ')
        print(Threshold(threshold, d, de))
        print(RSME_log(d, de))
        print(RSME_linear(d, de))
        print(Squal_Rel(d, de))
        print('Affination: ')
        print(Threshold(threshold, d, de1))
        print(RSME_log(d, de1))
        print(RSME_linear(d, de1))
        print(Squal_Rel(d, de1))
    # get the ground truth distance


def plotAccout():
    accout_ccrf = [0.88879502, 0.99594335, 0.99977257, 0.9988127, 0.98267188, 0.94255663,
                   0.91191067, 0.9138756, 0.85882353, 0.81422925, 0.85614035, 0.70666667,
                   0.75609756, 0.703]
    account = [0.81236105, 0.99377514, 0.99636116, 0.98337786, 0.95528228, 0.92556634,
               0.91431762, 0.90111643, 0.85058824, 0.81818182, 0.85719298, 0.70666667,
               0.75609756, 0.689]
    x = []
    labels = []
    for step, dis in enumerate(account):
        x.append(step + 1),
        labels.append('{}-{}'.format(step * 5, step * 5 + 5))
        pass
    plt.xticks(x, labels, rotation=70)
    plt.ylabel('Accuracy ', font)
    plt.xlabel('Distance (in meters)', font)
    plt.plot(x, accout_ccrf, 'bo-', linewidth=2, color='blue', label='SCNF')
    plt.plot(x, account, 'bo-', linewidth=2, color='orange', label='CNN')
    plt.legend(loc="upper right")
    plt.savefig('./Work/dis-acc', bbox_inches='tight')
    plt.show()
    pass

Ground_dis = [27.061,
              50.48,
              72.32]

caldis = [27.855,
          47.132,
          79.413]

SVR = [48.99, 58.91, 61.38]
IPM = [40.29, 151.76, -4031.93]
L = [Ground_dis, caldis, SVR, IPM]
out = ['Ground', 'DDN', 'SVR', 'IPM']


def printRectangle(objpath, imgpath, pathOut, GD):
    with open(objpath, 'r') as j:
        data = json.load(j)
    img = cv.imread(imgpath)
    for i, distance in enumerate(GD):
        # 提取边界框的坐标
        x = data['x'][i]
        y = data['y'][i]
        w = data['w'][i]
        h = data['h'][i]
        # 融合好的图像再拼回原图
        # 绘制边界框以及在左上角添加类别标签和置信度
        # color = [int(c) for c in COLORS[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
        text = str(distance)

        (text_w, text_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_w += 2
        text_h += 5
        create_green = np.zeros((text_h, text_w, 3), dtype=np.uint8)
        create_green[:, :, 0] = 0
        create_green[:, :, 1] = 255  # 这里我创建一个纯绿色的图像
        create_green[:, :, 2] = 0
        if i == 0:
            y = y + h
            tmp = img[y:y + text_h, x:x + text_w]
            img_add = cv.addWeighted(tmp, 0.3, create_green, 0.7, 0)
            img[y:y + text_h, x:x + text_w] = img_add
            cv.putText(img, text, (x + 5, y + text_h - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        else:
            if i == 1:
                x -= 30
                y -= 5
            tmp = img[y - text_h:y, x:x + text_w]
            img_add = cv.addWeighted(tmp, 0.3, create_green, 0.7, 0)
            img[y - text_h:y, x:x + text_w] = img_add
            cv.putText(img, text, (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # for i in range(num):
    #     print(i)
    #     # plot rectangle with text of distance
    #     img = cv.imread(imgpath)
    #     x = data['x'][i]
    #     y = data['y'][i]
    #     w = data['w'][i]
    #     h = data['h'][i]
    #     color = (234, 240, 68)
    #     text_color = (255, 255, 255)
    #     # cv.rectangle(img, (x, y), (x + w, y + h), color, 1)
    #     # text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
    #     # text = '{}: {:.3f}'.format('face', 0.982)
    #     text = str(Ground_dis(i))
    #     (text_w, text_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #     # cv.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
    #     cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    cv.imshow('img', img)
    cv.waitKey()
    cv.destroyAllWindows()
    jpg_quality = 80
    cv.imwrite(pathOut, img, [int(cv.IMWRITE_JPEG_QUALITY), jpg_quality])

    pass

def plotRectangle():
    for step, name in enumerate(out):
        print(os.path.join('Work\\', name, '.jpg'))
        printRectangle('KITTI\data\kitti\dict\\train\\000170.json', 'KITTI\data\kitti\Images\\000170.png',
                       'D:\CNN_CCRF\CNN&CCRF\Work\\' + name + '.jpg'
                       , L[step])



def main():
    # plotAccout()
    Ablation(root)


if __name__ =='__main__':
    main()
# for step, name in enumerate(out):
#     print(os.path.join('Work\\', name, '.jpg'))
#     printRectangle('KITTI\data\kitti\dict\\train\\000170.json', 'KITTI\data\kitti\Images\\000170.png',
#     'D:\CNN_CCRF\CNN&CCRF\Work\\'+name+'.jpg'
#                    , L[step])
