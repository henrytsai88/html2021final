import numpy as np
from matplotlib.pylab import plt
from data import data_N6_158 as raw_data
from ClientData import ClientData

data = ClientData(raw_data=raw_data)

def plotPoints(data_x, data_y, label_x, label_y, show=True):
    title = label_y + '-' + label_x

    plt.figure(num=title)
    data_x_1 = []
    data_y_1 = []
    i = 0
    for i in data.category[0]: 
        data_x_1.append(data_x[i])
        data_y_1.append(data_y[i])
    plt.plot(data_x, data_y, '.', color='b')
    plt.plot(data_x_1, data_y_1, '.', color='r')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    if show:
        plt.show()
