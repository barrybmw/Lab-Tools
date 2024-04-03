import cv2 as cv2
import numpy as np
import time

# 此程序应在该程序的母文件夹中打开

def shuru():
    image_1_path = input("请输入图1文件名：")
    image_2_path = input("请输入图2文件名：")
    image_1 = cv2.imread(image_1_path + ".png", cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(image_2_path + ".png", cv2.IMREAD_GRAYSCALE)

    # 设定半径与网格行列数
    radius = int(input("请输入计算半径："))  # 示例值，需要根据具体情况调整
    grids = int(input("请输入网格行列数："))
    return image_1, image_2, radius, grids

def yanzheng(image_1, image_2):# 验证图像尺寸是否一致
    assert image_1.shape == image_2.shape, "Images must have the same dimensions"

def jisuan(image_1, image_2, radius, grids):
    # 计算整个图像的亮度总和，以优化计算过程
    total_intensity_1 = np.sum(image_1)
    total_intensity_2 = np.sum(image_2)

    # 准备一个坐标网格，间隔为图像尺寸除以输入的网格行列数
    grid_size = (image_1.shape[0] // grids, image_1.shape[1] // grids)

    # 初始化元素分布差异值
    element_distribution_difference = 0

    # 遍历所有个位置
    for i in range(grids):
        for j in range(grids):
            # 计算每个位置的坐标
            center_x = int(grid_size[1] / 2 + j * grid_size[1])
            center_y = int(grid_size[0] / 2 + i * grid_size[0])

            # 使用OpenCV的圆形遮罩计算局部亮度总和
            mask = np.zeros(image_1.shape, dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            
            local_intensity_1 = np.sum(image_1[mask == 1])
            local_intensity_2 = np.sum(image_2[mask == 1])

            # 计算相对元素密度
            relative_density_1 = local_intensity_1 / total_intensity_1 if total_intensity_1 > 0 else 0
            relative_density_2 = local_intensity_2 / total_intensity_2 if total_intensity_2 > 0 else 0

            # 计算两图在该位置的相对元素密度差的平方
            density_difference = (relative_density_1 - relative_density_2) ** 2

            # 累加到元素分布差异值
            element_distribution_difference += density_difference

    # 输出元素分布差异值
    result = np.round(element_distribution_difference*10**5,4)
    return result

def main():
    image_1, image_2, radius, grids = shuru()

    a = time.time()
    yanzheng(image_1, image_2)
    result = jisuan(image_1, image_2, radius, grids)
    b = time.time()
    time_cost = b-a

    print("元素分布差异值为" + str(result) + ".")
    print("计算历时" + str(time_cost) + "s.")

def test():
    image_1, image_2 = shuru()
    print(image_1.shape[0], image_1.shape[1])

main()
#test()