import cv2 as cv2
import numpy as np
import time

# 本程序可以去除图片边缘的零星点
# 本程序需在该程序的母文件夹中打开

def xuantu():
    image_name = input("请输入图片文件名：")
    image = cv2.imread(image_name + ".png", cv2.IMREAD_GRAYSCALE) # 读取为灰度图
    return image_name, image

def jinghua(image_name, image):
    grids = image.shape[0]
    radius = 5 #这是一个经验半径
    # radius = int(input("请输入遮罩半径："))
    threshold = 400 #这是一个经验阈值
    # threshold = int(input("请输入阈值："))
    for i in range(grids):
        for j in range(grids):
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (j, i), radius, 1, -1)
            local_intensity = np.sum(image[mask == 1])
            if local_intensity < threshold: # 去掉一切半径5范围内总亮度小于阈值的点
                image[i][j] = 0
    cv2.imwrite(image_name + "-modified.png", image)

def main():
    image_name, image = xuantu()
    jinghua(image_name, image)

main()

