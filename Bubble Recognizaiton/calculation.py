import os
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def shudian():
    folder_path = 'pictures'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # calculate the toal number of files
    total_files = len(files)
    # create a empty matrix
    total_data = np.zeros((12, 2, total_files))
    k = 0
    #background = np.zeros([12,1])

    for file in files:
        file_path = os.path.join(folder_path, file)
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # calculate time based on "file" name, e,g, "序列 01.00_00_35_00" corresponds to 35 seconds
        time = int(file.split('_')[1]) * 60 + int(file.split('_')[2]) + int(file.split('_')[3].split('.')[0])/30

        # cut the graph to 12 parts, each part is 1/12 of the graph, 4 * 3, create 12 dictionarys
        cut = gray[86:1311, 800:1760]
        for i in range(4):
            for j in range(3):
                cell = cut[i * 306 : (i + 1) * 306, j * 350 : (j + 1) * 350]
                # cv2.imwrite('pictures/example_cut/cell' + str(i) + '_' + str(j) + '.jpg', cell)
                cell_value = np.sum(cell)
                total_data[i * 3 + j][0][k] = time
                total_data[i * 3 + j][1][k] = max(cell_value - total_data[i * 3 + j][1][0], 0)
        
        k += 1
    
    # for each one of 12 matrix in total_data, rank the time and value
    for i in range(12):
        total_data[i] = total_data[i][:, np.argsort(total_data[i][0])]

        # save all data together in one excel
        np.savetxt('results/table/total_data' + str(i+1) + '.csv', total_data[i], delimiter=',')
    
    # merge the csv files into one excel
    os.system('cat results/table/total_data*.csv > results/table/total_data.csv')

    # plot all the curves together in one graph
    plt.figure(figsize=(16,9))

    for i in range(12):
        f = interp1d(total_data[i][0], total_data[i][1], kind = 'slinear')
        xnew = np.linspace(total_data[i][0][0], total_data[i][0][-1], 300)
        ynew = f(xnew)
        plt.plot(xnew, ynew, label='catalyst ' + str(i+1))
    
    plt.xlabel('Time/s')
    plt.ylabel('Bubble Quantity')
    plt.legend()
    plt.savefig('results/total_data.png', dpi = 300, bbox_inches='tight')

shudian()



