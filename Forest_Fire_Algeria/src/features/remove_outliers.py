import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Visualize_outliers
##read dataset
df= pd.read_pickle(f'../../data/processed/Forest_fire_processed_data')

##boxplot
plt.figure(figsize=(12,20))
sns.boxplot(df)
plt.show()

##IQR (distribution based method to detect outliers)


numeric_feat = list(df.select_dtypes(include=['number']).columns)

for i in range(0,len(numeric_feat)):
    df_IQR = Visualize_outliers.mark_outliers_iqr(df,numeric_feat[i])
    Visualize_outliers.plot_binary_outliers(df_IQR, numeric_feat[i], numeric_feat[i]+'_outlier')
    plt.savefig(f'../../reports/figures/IQR/IQR_for_{numeric_feat[i]}.png')
    plt.show()
    plt.close()
    

##Local Outlier Factor (distance/density based method)
df_lof,outliers, X_scores = Visualize_outliers.mark_outliers_lof(df,numeric_feat)
for i in range(0,len(numeric_feat)):
        Visualize_outliers.plot_binary_outliers(df_lof, numeric_feat[i],'outlier_lof')
        plt.savefig(f'../../reports/figures/LOF/LOF_for_{numeric_feat[i]}.png')
        plt.show()
        plt.close()

###############



df_outliers_removed = df.copy()
for col in numeric_feat:
    df_outliers_removed = Visualize_outliers.mark_outliers_iqr(df_outliers_removed,col)
    df_outliers_removed.loc[df_outliers_removed[col+'_outlier'],col] = np.nan 
    n_outliers = len(df) - len(df_outliers_removed[col].dropna())
    print(f'number of outliers from column {col} = {n_outliers}')



df_outliers_removed.drop(['Temperature_outlier', 'RH_outlier',
       'Ws_outlier', 'Rain_outlier', 'FFMC_outlier', 'DMC_outlier',
       'DC_outlier', 'ISI_outlier', 'BUI_outlier', 'FWI_outlier'],axis=1,inplace=True)

## dealing with missing values 
for col in numeric_feat:
    df_outliers_removed[col].fillna(df_outliers_removed[col].median(),inplace=True)




##Export the new dataframe
df_outliers_removed.to_pickle('../../data/interim/outliers_removed_IQR.pkl')