#Overview of Project 

#import the necessary package 

import pandas as pd 
import numpy as np 

#load Traning Data 
Train_data=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
print(Train_data.head())
print(Train_data.columns)
#Load Test Data
Test_data=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(pd.Series(Train_data.columns).value_counts().sum())
Train_data=Train_data.drop(columns=['PoolQC','MiscFeature','Alley','Fence','GarageYrBlt','GarageCond','BsmtFinType2'],axis=1)
Test_data=Test_data.drop(columns=['PoolQC','MiscFeature','Alley','Fence','GarageYrBlt','GarageCond','BsmtFinType2'],axis=1)
print(Train_data.head())
print(pd.Series(Train_data.columns).value_counts().sum())

print(Train_data.dtypes[Train_data.dtypes != 'object'])
#print(Train_data.describe())
#print(Train_data.info())
print(Train_data.duplicated())

import matplotlib.pyplot as plt
#print(Train_data.query('LotFrontage >200'))
Train_data=Train_data.drop(Train_data[Train_data['LotFrontage']>200.00].index)

print("="*50)
#print(Train_data.query('LotArea >100000'))
Train_data=Train_data.drop(Train_data[Train_data['LotArea']>100000].index)
#print(Train_data.query('OverallCond <3 & SalePrice >390000'))
Train_data=Train_data.drop(Train_data[(Train_data['OverallCond']==2) & (Train_data['SalePrice']>390000)].index)
#print(Train_data.query('MasVnrArea >1300 or SalePrice >700000'))
Train_data=Train_data.drop(Train_data[(Train_data['MasVnrArea']>1300) | (Train_data['SalePrice']>700000)].index)
Train_data=Train_data.drop(Train_data[Train_data['BedroomAbvGr']>7].index)
Train_data=Train_data.drop(Train_data[(Train_data['KitchenAbvGr']<0.2) | (Train_data['KitchenAbvGr']>2.7)].index)
Train_data=Train_data.drop(Train_data[Train_data['PoolArea']>450].index)
Train_data=Train_data.drop(Train_data[Train_data['MiscVal']>8000].index)

plt.scatter(x='MoSold',y='SalePrice',data= Train_data)
plt.show()

print(Train_data.info())

Train_data=Train_data[Train_data["Electrical"].notna()]
print(Train_data.info())

import seaborn as sns

#print(Train_data['BsmtQual'].unique())
Train_data['BsmtQual'].fillna('Ex',inplace=True)
Train_data['BsmtQual'].fillna('Ex',inplace=True)

print(Train_data['BsmtCond'].unique())
Train_data['BsmtCond'].fillna('TA',inplace=True)
Train_data['BsmtCond'].fillna('TA',inplace=True)

print(Train_data['BsmtExposure'].unique())
Train_data['BsmtExposure'].fillna('Gd',inplace=True)
Train_data['BsmtExposure'].fillna('Gd',inplace=True)

print(Train_data['BsmtFinType1'].unique())
Train_data['BsmtFinType1'].fillna('Unf',inplace=True)
Train_data['BsmtFinType1'].fillna('Unf',inplace=True)

print(Train_data['GarageType'].unique())
Train_data['GarageType'].fillna('No',inplace=True)
Train_data['GarageType'].fillna('No',inplace=True)

print(Train_data['GarageFinish'].unique())
Train_data['GarageFinish'].fillna('Unf',inplace=True)
Train_data['GarageFinish'].fillna('Unf',inplace=True)

print(Train_data['GarageQual'].unique())
Train_data['GarageQual'].fillna('No',inplace=True)
Train_data['GarageQual'].fillna('No',inplace=True)

print("="*50)

#print(Train_data['FireplaceQu'].unique())
Train_data['FireplaceQu'].fillna('No',inplace=True)
Train_data['FireplaceQu'].fillna('No',inplace=True)
#sns.catplot(data=Train_data, x='FireplaceQu', y='SalePrice', kind='box')

#print(Train_data['MasVnrType'].unique())
Train_data['MasVnrType'].fillna('No',inplace=True)
Train_data['MasVnrType'].fillna('No',inplace=True)
#sns.catplot(data=Train_data, x='MasVnrType', y='SalePrice', kind='box')

print(Train_data['GarageQual'].unique())
Train_data['GarageQual'].fillna('No',inplace=True)
Train_data['GarageQual'].fillna('No',inplace=True)
sns.catplot(data=Train_data, x='GarageQual', y='SalePrice', kind='box')