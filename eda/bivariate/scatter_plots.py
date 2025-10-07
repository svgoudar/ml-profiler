def create_scatter_plot(x, y, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def save_scatter_plot(x, y, filename='scatter_plot.png', title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()