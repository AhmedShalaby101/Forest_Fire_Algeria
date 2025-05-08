import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Principal_Component_Analysis import PrincipalComponentAnalysis
## read dataset
df = pd.read_pickle('../../data/interim/outliers_removed_IQR.pkl')

numeric_feat = list(df.select_dtypes(include=['number']).columns)



## Principal Component Analysis


df_pca = df.copy()
PCA = PrincipalComponentAnalysis()
pca_values = PCA.determine_pc_explained_variance(df,numeric_feat)

plt.figure(figsize=(10,10))
plt.plot(range(1, len(numeric_feat) + 1),
         pca_values, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.savefig(r'../../reports/figures/elbow_figure_PCA.png')
plt.show()
plt.close()
df_pca = PCA.apply_pca(df_pca,numeric_feat,4)

## Export dataset 

df_pca.to_pickle('../../data/external/build_features_pca.pkl')
