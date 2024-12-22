import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

InputFilePath=r"C:\Users\z004fpbu\Desktop\jay\MyCodes\Projects\ML_Project(HousePrice)\Inputs\Housing.csv"
df=pd.read_csv(InputFilePath)
print(df.shape)
print(df.info())

#performing EDA by using 3 party tool sweetviz (Explotery data analysis).
#import sweetviz as sv
#report=sv.analyze(df)
#report.show_html("./report.html")

# data preprocessing 
#1). checking of null values
print(df.isna().sum())

#2).Drop Independent Variables(not related to analysis Output)

print(df.info())
df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']] = df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']].replace({'yes': 1, 'no': 0}).astype(int)

Newdf=df
print(Newdf)

plt.scatter(x='area',y='price',data=Newdf)
plt.xlabel('Area per sq')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs. Area per sq')
plt.show()
Newdf = Newdf.drop(Newdf[Newdf['area']>14000].index)
print(Newdf['area'].sort_values())

plt.scatter(x='parking',y='price',data=Newdf)
plt.xlabel('Number of parking')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs. Number of parking')
#plt.show()
Newdf = Newdf.drop(Newdf[(Newdf['parking']==3.0)&(Newdf['price']>12000000)].index)
print(Newdf['parking'].sort_values())

plt.scatter(x='bedrooms',y='price',data=Newdf)
plt.xlabel('Number of bedrooms')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs. Number of bedrooms')
#plt.show()
Newdf = Newdf.drop(Newdf[Newdf['bedrooms']==6].index)
print(Newdf['bedrooms'].sort_values())

plt.scatter(x='bathrooms',y='price',data=Newdf)
plt.xlabel('Number of bathrooms')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs. Number of bathrooms')
#plt.show()
Newdf = Newdf.drop(Newdf[(Newdf['bedrooms']==2.00)&(Newdf['price']>12000000)].index)
print(Newdf['bedrooms'].sort_values())

plt.scatter(x='stories',y='price',data=Newdf)
plt.xlabel('Number of stories')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs. Number of stories')
#plt.show()
Newdf = Newdf.drop(Newdf[(Newdf['stories']==3.0)&(Newdf['price']>12000000)].index)
print(Newdf['stories'].sort_values())

plt.scatter(x='mainroad',y='price',data=Newdf)
plt.xlabel('Number of mainroad')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs. Number of mainroad')
#plt.show()
Newdf = Newdf.drop(Newdf[(Newdf['mainroad']==3.0)&(Newdf['price']>12000000)].index)
print(Newdf['mainroad'].sort_values())

print(Newdf.describe())
plt.scatter(x='area',y='price',data=Newdf)
plt.xlabel('area')
plt.ylabel('Price')
plt.title('Check data')
plt.show()


# Binding Models 
from sklearn.preprocessing import LabelEncoder
Newdf['furnishingstatus']=LabelEncoder().fit_transform(Newdf['furnishingstatus'])
Newdf=Newdf.astype('int64')
print(Newdf.info())
x=Newdf.drop(columns=['price'])
y=Newdf['price']

from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics import mean_absolute_error,r2_score

X_tain,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.2,random_state=42)
print(X_tain.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#applying linear regression 
#col_trans=make_column_transformer((OneHotEncoder(sparse_output=False),['furnishingstatus']),remainder='passthrough',n_jobs=-1)

Scaler=StandardScaler()
lr=LinearRegression()
pipe=make_pipeline(X_tain,Scaler,lr)
pipe.fit(X_tain,Y_train)
y_pred_lr=pipe.predict(X_test)
print(mean_absolute_error(Y_test,y_pred_lr))
print(r2_score(Y_test,y_pred_lr))

#laso
lasso=Lasso()
pipe=make_pipeline(X_tain,Scaler,lasso)
pipe.fit(X_tain,Y_train)
Pipeline(steps=[( 'StandardScaling',StandardScaler()),('Lasso',Lasso())])
y_pred_lasso=pipe.predict(X_test)
print(r2_score(Y_test,y_pred_lasso))                







print("end")












