import os
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('pictures/序列 01.00_01_14_00.Still065.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cut = gray[86:1311, 800:1760]
print(cut.shape)

# cut "cut" to 4 parts averagely, each part is 1/4 of the graph, 4 * 1
for i in range(4):
    for j in range(3):
        cell = cut[i * 306 : (i + 1) * 306, j * 350 : (j + 1) * 350]
        #cv2.imwrite('pictures/example_cut/cell' + str(i) + '_' + str(j) + '.jpg', cell)
        print(str(i * 3 + j) +',' + str(np.sum(cell)))

print(cut.shape)
