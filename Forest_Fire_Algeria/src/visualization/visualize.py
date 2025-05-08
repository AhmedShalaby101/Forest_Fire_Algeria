#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_settings import apply_plot_style

# apply plot settings
apply_plot_style()


#read dataset
df = pd.read_pickle('../../data/processed/Forest_fire_processed_data')

# distribution for dataset
numeric_feat= []
for col in df.columns:
    if col != 'Date' and col != 'Region' and col != 'Classes':
        numeric_feat.append(col)
        plt.figure()
        sns.kdeplot(df[col],fill=True,color='b')
        plt.title(f'{col} Distribution')
        plt.savefig(f'../../reports/figures/Distributions/{col}_distribution.png')
        plt.show()
        plt.close()
"""
most of the dstibutions are right tailed or left tailed(skewed)
only 3 features we could only consider them normally distributed
Temperature,RH,Ws.
"""   
skew_values = df[numeric_feat].skew()
skew_df = pd.DataFrame(skew_values).T  
# Transpose to make it 2D (1 row, multiple columns)
sns.heatmap(skew_df, annot=True,cbar=True)
plt.savefig(f'../../reports/figures/skewness.png')
"""
Rain feature is extremely right skewed
Skewness is more of a problem when we use linear models
so we should work with care when use linear models
even apply Log transformation df['col'] = np.log1p(df['col'])
 """
 
#Check linear corr
plt.figure(figsize=(15,15))
sns.heatmap(df[numeric_feat].corr(),annot=True)
plt.savefig(f'../../reports/figures/corr_matrix.png')
plt.show()
plt.close()
"""
Since there is strong linear collinearity among some features,
we will apply PCA to remove redundancy for models
that are sensitive to multicollinearity

However, we retain the basic features for tree-based models 
like Random Forest and XGBoost
as they are not affected by multicollinearity
"""


# plot a pie chart for fire percentage
plt.pie(df['Classes'].value_counts(),labels=['fire','not fire']
        ,autopct='%1.1f%%',colors=['red','blue']
        ,labeldistance=0.8,explode=[0.05,0.05])
plt.title('Fire percentage in data')
plt.savefig(f'../../reports/figures/fire_pie_chart.png')
plt.show()
plt.close()


#plot the fire and not fire by Region
sns.countplot(data=df,x='Region',hue='Classes')
plt.title("Fire Vs Not Fire counts by Region")
plt.savefig(f'../../reports/figures/fire_count_by_Region.png')
plt.show()
plt.close()

##check the patterns in data
plt.figure(figsize=(15,20))
plt.suptitle('scatter plots for numerical features')
for i in range(0,len(numeric_feat)):
    plt.subplot(5,2,i+1)
    plt.scatter(y=numeric_feat[i],x=df.index,data=df,color='r')
    plt.ylabel(numeric_feat[i])
    plt.tight_layout()
plt.savefig(f'../../reports/figures/patterns_in_data.png')
plt.show()
plt.close()


