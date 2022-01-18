from matplotlib.pylab import plt

def plot_points(data_x, data_y, label_x, folder_name="", show=True):
    title = label_x

    plt.figure(num=title, figsize=[40, 10])
    classes = ['No Churn', 'Competitor', 'Dissatisfaction', 'Attitude', 'Price', 'Other']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'gray']
    data_x_points = []
    count = [0]
    total = 0
    for y in classes:
        data_x_points.append(data_x[data_y == y])
        total += data_y[data_y == y].shape[0]
        count.append(total)
    for i in range(len(classes)):
        plt.plot([y for y in range(count[i], count[i+1])], data_x_points[i], '.', color=colors[i])

    plt.legend(classes)
    plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig(f"{folder_name}/{title}.png")