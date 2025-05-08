import matplotlib.dates as md
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
##IQR technique

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset





##Local Outlier Factor

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


##plot outliers

def plot_binary_outliers(data_table, col, outlier_col):
    """
    Plot outliers in case of a binary outlier score.

    Parameters:
    - data_table: DataFrame containing the data
    - col: Name of the column to plot
    - outlier_col: Column with binary (bool/int) outlier flags

    If the index is datetime, it will be formatted as time. Otherwise, a numeric index is used.
    """


    # Drop missing values
    data_table = data_table.dropna(subset=[col, outlier_col])
    data_table[outlier_col] = data_table[outlier_col].astype(bool)

    # Create the plot with constrained_layout to allow legend outside
    fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)

    # Check if the index is datetime type
    if pd.api.types.is_datetime64_any_dtype(data_table.index):
        ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        ax.set_xlabel('Time')
    else:
        ax.set_xlabel('Index')

    ax.set_ylabel(col)

    # Plot outliers in red
    ax.plot(data_table.index[data_table[outlier_col]], 
            data_table[col][data_table[outlier_col]], 
            'r+', label=f'Outlier {col}')
    
    # Plot normal values in blue
    ax.plot(data_table.index[~data_table[outlier_col]], 
            data_table[col][~data_table[outlier_col]], 
            'b+', label=f'No Outlier {col}')

    # Move legend above plot
    ax.legend(numpoints=1, fontsize='small', loc='lower center',
              bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True)



