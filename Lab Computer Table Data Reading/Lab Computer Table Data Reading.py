import cv2
import pytesseract
import pygetwindow as gw
import pyautogui
from PIL import ImageGrab
import numpy as np
import pandas as pd
import re
import easyocr
from matplotlib import pyplot as plt
import time

# 配置Tesseract的路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 根据实际路径修改

def capture_screen(region=None):
    """
    截取屏幕的指定区域
    :param region: 区域坐标(x, y, width, height)
    :return: 截取的图像
    """
    screen = ImageGrab.grab(bbox=region)
    screen_np = np.array(screen)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
    return screen_np

def extract_table_data(image):
    """
    从图像中提取表格数据
    :param image: 输入图像
    :return: 识别的文本
    """
    # 对图像进行处理，如需要可以添加滤波、二值化等
    processed_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 使用pytesseract识别文本
    text = pytesseract.image_to_string(processed_img)
    return text

def rowcol(img):
    rows, cols = img.shape
    scale = 20
    # 自适应获取核值
    # 识别横线:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(img, kernel, iterations=1)
    dilated_col = cv2.dilate(eroded, kernel, iterations=1)
    cv2.imwrite('dilated_col.png', dilated_col)

    # 识别竖线：
    scale = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(img, kernel, iterations=1)
    dilated_row = cv2.dilate(eroded, kernel, iterations=1)
    cv2.imwrite('dilated_row.png', dilated_row)
    return dilated_col, dilated_row

def merge(dilated_col, dilated_row, img):
    # 将识别出来的横竖线合起来
    bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
    cv2.imwrite('bitwise_and.png', bitwise_and)

    # 标识表格轮廓
    merge = cv2.add(dilated_col, dilated_row)

    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(img, merge)

    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erode_image = cv2.morphologyEx(merge2, cv2.MORPH_OPEN, new_kernel)
    merge3 = cv2.add(erode_image, bitwise_and)

    # 将焦点标识取出来
    ys, xs = np.where(bitwise_and)
    #print(ys)
    #print(xs)
    return ys, xs, merge2

def sort(ys, xs):
    # 横纵坐标数组
    y_point_arr = []
    x_point_arr = []
    # 通过排序，排除掉相近的像素点，只取相近值的最后一点
    # 这个10就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    sort_x_point = np.sort(xs)
    for i in range(len(sort_x_point) - 1):
        if sort_x_point[i + 1] - sort_x_point[i] > 3:
            x_point_arr.append(sort_x_point[i])
        i = i + 1
    # 要将最后一个点加入
    x_point_arr.append(sort_x_point[i])

    i = 0
    sort_y_point = np.sort(ys)
    #print(np.sort(ys))
    for i in range(len(sort_y_point) - 1):
        if (sort_y_point[i + 1] - sort_y_point[i] > 3):
            y_point_arr.append(sort_y_point[i])
        i = i + 1
    y_point_arr.append(sort_y_point[i])
    #print(y_point_arr, x_point_arr)
    return y_point_arr, x_point_arr

def recognize(y_point_arr, x_point_arr, merge):
    # 循环y坐标，x坐标分割表格
    data = [[] for i in range(len(y_point_arr) - 1)]
    for i in range(len(y_point_arr) - 1):
        for j in range(len(x_point_arr) - 1):
            # 在分割时，第一个参数为y坐标，第二个参数为x坐标
            cell = merge[y_point_arr[i]:y_point_arr[i + 1], x_point_arr[j]:x_point_arr[j + 1]]
            #plt.imshow(cell)
            #plt.show()

            # 读取文字，此为默认英文
            # pytesseract.pytesseract.tesseract_cmd = 'E:/Tesseract-OCR/tesseract.exe'
            #text1 = pytesseract.image_to_string(cell, lang="chi_sim+eng")
            #reader = easyocr.Reader(['en'])
            #text1 = reader.readtext(cell)
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=E0123456789B.-'
            text1 = pytesseract.image_to_string(cell, config=custom_config, lang='eng')
            #print(text1)

            cleaned_text = text1.replace('\n', '')
            data[i].append(cleaned_text)
            
            j = j + 1
        i = i + 1
    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=False, header=False)
    df.to_excel('output.xlsx', index=False, header=False)

def main():
    a = time.time()
    '''
    print("请将鼠标移到屏幕的左上角位置，然后按回车键。")
    input()
    x1, y1 = pyautogui.position()

    print("请将鼠标移到屏幕的右下角位置，然后按回车键。")
    input()
    x2, y2 = pyautogui.position()

    region = (x1, y1, x2, y2)
    screen_img = capture_screen(region)
    '''
    screen_img = cv2.imread('img.PNG')
    processed_img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2GRAY)
    height, width = processed_img.shape
    scale_factor = 5
    scaled_img = cv2.resize(processed_img, (width*scale_factor, height*scale_factor))
    ret, img0 = cv2.threshold(scaled_img,180,255,cv2.THRESH_BINARY)
    img = 255 - img0
    #plt.imshow(img)
    #plt.show()
    cv2.imwrite('img1.png', img) 

    dilated_col, dilated_row = rowcol(img)

    ys, xs, merge2 = merge(dilated_col, dilated_row, img)

    y_point_arr, x_point_arr = sort(ys, xs)

    recognize(y_point_arr, x_point_arr, merge2)

    #recognize(ys, xs, merge2)
    
    '''
    # 提取表格数据
    data = extract_table_data(screen_img)
    print("提取到的数据：")
    print(data)

    # 将数据分割成行
    data_lines = data.split('\n')

    # 将每行数据根据空格分割
    data_list = [line.split(' ') for line in data_lines if line.strip() != '']

    # 创建DataFrame
    df = pd.DataFrame(data_list)

    # 保存到Excel文件
    df.to_excel('output.xlsx', index=False)
    '''

    b = time.time()

    print(b-a)

if __name__ == "__main__":
    main()
