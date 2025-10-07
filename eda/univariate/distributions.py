def plot_distribution(data, column, bins=30):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def describe_distribution(data, column):
    description = data[column].describe()
    return description

def check_normality(data, column):
    from scipy import stats

    k2, p = stats.normaltest(data[column])
    alpha = 0.05
    if p < alpha:
        return f'The distribution of {column} is not normal (p-value = {p})'
    else:
        return f'The distribution of {column} is normal (p-value = {p})'