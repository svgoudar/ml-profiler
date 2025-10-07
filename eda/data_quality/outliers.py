def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

def detect_outliers_zscore(data, column, threshold=3):
    from scipy import stats
    z_scores = stats.zscore(data[column])
    return data[abs(z_scores) > threshold]

def handle_outliers(data, column, method='remove'):
    outliers = detect_outliers_iqr(data, column)
    if method == 'remove':
        return data[~data.index.isin(outliers.index)]
    elif method == 'replace':
        median = data[column].median()
        data[column] = data[column].where(~data.index.isin(outliers.index), median)
        return data
    else:
        raise ValueError("Method must be 'remove' or 'replace'")