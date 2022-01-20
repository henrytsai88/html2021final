import numpy as np
from matplotlib.pyplot import plt

def plot_points(data_x, data_y, label_x, predict_y=None, folder_name="", show=True, size=[7,3.5]):
    print(label_x)
    title = label_x
    classes = ['No Churn', 'Competitor', 'Dissatisfaction', 'Attitude', 'Price', 'Other']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'gray']
    data_x_sort = np.array([])
    predict_y_sort = np.array([])
    count = [0]
    total = 0
    if predict_y is None:
        predict_y = data_y
    for y in classes:
        data_x_sort = np.append(data_x_sort, data_x[data_y == y])
        predict_y_sort = np.append(predict_y_sort, predict_y[data_y == y])
        total += data_y[data_y == y].shape[0]
        count.append(total)
    id = np.array([i for i in range(len(data_x_sort))])
    data_x_points = []
    data_y_points = []
    for y in classes:
        data_x_points.append(data_x_sort[predict_y_sort == y])
        data_y_points.append(id[predict_y_sort == y])

    # Colored by ground truth class
    plt.figure(num=title, figsize=size)
    for i in range(len(classes)):
        plt.plot([y for y in range(count[i], count[i+1])], data_x_sort[count[i]:count[i+1]], '.', color=colors[i])
    plt.xlabel("Samples")
    ## Hide numbers on x-axis
    # plt.tick_params(
    #   axis='x',          # changes apply to the x-axis
    #   which='both',      # both major and minor ticks are affected
    #   bottom=False,      # ticks along the bottom edge are off
    #   top=False,         # ticks along the top edge are off
    #   labelbottom=False) # labels along the bottom edge are off
    plt.ylabel(label_x)
    plt.legend(classes)
    plt.title(title)
    if show:
        plt.show()

    # Colored by predicted class
    plt.figure(num=title, figsize=size)
    for i in range(len(classes)):
        plt.plot(data_y_points[i], data_x_points[i], '.', color=colors[i])
    plt.xlabel("Samples")
    ## Hide numbers on x-axis
    # plt.tick_params(
    #   axis='x',          # changes apply to the x-axis
    #   which='both',      # both major and minor ticks are affected
    #   bottom=False,      # ticks along the bottom edge are off
    #   top=False,         # ticks along the top edge are off
    #   labelbottom=False) # labels along the bottom edge are off
    plt.ylabel(label_x)
    plt.legend(classes)
    plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig(f"{folder_name}/{title}.png")