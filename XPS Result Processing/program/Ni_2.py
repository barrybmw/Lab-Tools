import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

a = time.time()

# Load the data from the Excel file
file_path = 'Ni_2.xlsx'
ni_data = pd.read_excel(file_path)

# Convert the data to a numpy array for visualization
ni_matrix = ni_data.to_numpy()
ni_matrix[ni_matrix != 0] -= 3
ni_matrix[ni_matrix <0] = 0

# 假设 ni_matrix 是您的数据数组
x, y = np.nonzero(ni_matrix)  # 获取非零值的坐标
values = ni_matrix[x, y]      # 获取对应的数值

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=values, cmap='hot', s=values)  # 使用数值作为点的大小
plt.colorbar(label='Ni Concentration')
plt.title('Ni Element Concentration with Emphasized Points')
plt.axis('off')
b = time.time()
print(b-a)
plt.show()