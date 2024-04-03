import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.colors as mcolors

a = time.time()

# Load the data from the Excel file
file_path = 'Ni_3.xlsx'
ni_data = pd.read_excel(file_path)
A = ni_data.to_numpy()

num_splits = 5

# Function to split the matrix into smaller matrices
def split_matrix_equally(matrix, num_splits):
    rows, cols = matrix.shape
    row_split = [rows // num_splits + (1 if i < rows % num_splits else 0) for i in range(num_splits)]
    col_split = [cols // num_splits + (1 if i < cols % num_splits else 0) for i in range(num_splits)]

    current_row = 0
    small_matrices = []
    for r in row_split:
        current_col = 0
        for c in col_split:
            small_matrix = matrix[current_row:current_row + r, current_col:current_col + c]
            small_matrices.append(small_matrix)
            current_col += c
        current_row += r

    return small_matrices

# 将矩阵 A 分割成25个小矩阵
small_matrices = split_matrix_equally(A, 5)

# Calculate the number of non-zero elements in the middle matrix
a10 = small_matrices[10]
a11 = small_matrices[11]
a12 = small_matrices[12]
a13 = small_matrices[13]
a14 = small_matrices[14]

a12_max = np.max(a12)

a10_999 = np.percentile(a10,99.996)
a11_999 = np.percentile(a11,99.996)
a12_999 = np.percentile(a12,99.996)
a13_999 = np.percentile(a13,99.996)
a14_999 = np.percentile(a14,99.996)

a12_99 = np.percentile(a12,99)
a12_90 = np.percentile(a12,90)

a_999_max = np.max([a10_999,a11_999,a12_999,a13_999,a14_999])
print(a_999_max)

#A[A < a_999_max] = 0


x = int(np.round(1.05*np.count_nonzero(a12)))
y = int(np.round(x*25))


# Set the smallest x non-zero elements in each small matrix to zero
for sm in small_matrices:
    non_zero_elements = sm[sm != 0]  # Extract non-zero elements
    if len(non_zero_elements) > x:
        threshold_value = np.partition(non_zero_elements, x-1)[x-1]
        sm[(sm != 0) & (sm <= threshold_value)] = 0
    else:
        sm[sm != 0] = 0  # If non-zero elements are less than x, set all to zero

# 重新拼接矩阵
def reconstruct_matrix(small_matrices, num_splits):
    rows = [np.hstack(small_matrices[i*num_splits:(i+1)*num_splits]) for i in range(num_splits)]
    return np.vstack(rows)

# 重新拼接为大矩阵 B
B = reconstruct_matrix(small_matrices, 5)


#non_zero_elements_A = A[A != 0]
#threshold_value = np.partition(non_zero_elements_A, y-1)[y-1]
#A[(A != 0) & (A <= threshold_value)] = 0


# 创建一个自定义的颜色映射
cmap = plt.cm.hot
colors = cmap(np.arange(cmap.N))
colors[0, :3] = np.array([1, 1, 1])  # 将最低值（即0）设置为白色
white_hot = mcolors.ListedColormap(colors)


def main1(A):
    # Plotting the data
    plt.figure(figsize=(10, 10))
    plt.imshow(A, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Ni Concentration')
    plt.axis('off')  # Turn off axis since they are not relevant in this context
    b = time.time()
    print(b-a)
    plt.savefig('Ni_Concentration_Map_1.png', format='png', dpi=2000)
    plt.show()

def main2(A):
    # 假设 A 是您的数据数组
    x, y = np.nonzero(A)  # 获取非零值的坐标
    values = A[x, y]      # 获取对应的数值

    plt.figure(figsize=(10, 10))
    plt.scatter(y, A.shape[1]-x, c=values, cmap='hot', s=values)  # 使用数值作为点的大小
    plt.colorbar(label='Ni Concentration')
    plt.axis('off')
    plt.savefig('Ni_Concentration_Map_2.png', format='png', dpi=2000)
    plt.show()

main1(A)
main2(A)