#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
df = pd.read_csv("../../data/raw/Algerian_forest_fires_dataset_UPDATE.csv",header=1)

#reading each region data 
df_bejaia=df.iloc[0:122,:]
df_sidi_bel = df.iloc[125:,:]

#labeling with new column
df_bejaia['Region'] = 'Bejaia'
df_sidi_bel['Region'] = 'Sidi Bel'

df=pd.concat((df_bejaia,df_sidi_bel),axis=0).reset_index()

#remove spaces from cols names
df.columns = df.columns.str.strip()

#remove extra index column
df.drop('index',axis=1,inplace=True)

#cleaning Classes column
df['Classes'].value_counts()
df['Classes'].unique()
df['Classes']=df['Classes'].str.strip()
df['Classes'].unique()

#add date feature instead of 3 features
df['Date']=pd.to_datetime(df[['day','month','year']])
df.drop(['day','month','year'],axis=1,inplace=True)

##readjust the types of data 
df.info()

df['Temperature'].unique()
df['RH'].unique()
df['Ws'].unique()
df['Rain'].unique()
df[['Temperature','RH','Ws']] = df[['Temperature','RH','Ws']].astype('int')

for col in df.columns:
    if df[col].dtype == 'object' and col != 'Classes' and col !='Region' :
        df[col] = df[col].astype('float64')

#check for the missing values
df.isnull().sum() # one null value in the classes column
df.dropna(inplace=True)

##Export datset
df.to_pickle('../../data/processed/Forest_fire_processed_data')


