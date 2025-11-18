#coding=utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_map(datapath):
    #这个函数接受一个参数 datapath，它是指向包含图像数据的目录的路径。
    print(datapath)
    for name in os.listdir(datapath+'/PH2 Dataset images'):
        #os.listdir 列出指定目录（datapath/mask）中的所有文件和文件夹名。
        mask = cv2.imread(datapath+'/PH2 Dataset images/'+name+'/'+name+'_lesion/'+name+'_lesion.bmp',0)
        #使用 OpenCV 的 imread 函数读取掩码图像。0 表示图像以灰度模式读取。
        body = cv2.blur(mask, ksize=(5,5))
        #使用 cv2.blur 函数对图像进行模糊处理，ksize=(5,5) 指定了模糊操作的核大小。
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        #cv2.distanceTransform 计算每个非零像素点到最近零像素点的距离。distanceType=cv2.DIST_L2 指定使用欧氏距离，maskSize=5 指定掩码大小。
        #然后对结果取平方根，这可能是为了调整距离的度量或使其更符合某些后续处理的需求。

        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(datapath+'/body-origin/'):
            os.makedirs(datapath+'/body-origin/')
        cv2.imwrite(datapath+'/body-origin/'+name+'.bmp', body)

        if not os.path.exists(datapath+'/detail-origin/'):
            os.makedirs(datapath+'/detail-origin/')
        cv2.imwrite(datapath+'/detail-origin/'+name+'.bmp', mask-body)


if __name__=='__main__':
    split_map('data/ddw_data/skin-image/PH2Dataset')

