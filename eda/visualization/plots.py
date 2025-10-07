def create_line_plot(data, x_column, y_column, title='Line Plot', xlabel='X-axis', ylabel='Y-axis'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_column], data[y_column], marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def create_bar_plot(data, x_column, y_column, title='Bar Plot', xlabel='Categories', ylabel='Values'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_column], data[y_column], color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

def create_histogram(data, column, title='Histogram', xlabel='Values', ylabel='Frequency', bins=10):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, color='lightgreen', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.show()

def create_scatter_plot(data, x_column, y_column, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def create_box_plot(data, column, title='Box Plot', ylabel='Values'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data[column])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.show()