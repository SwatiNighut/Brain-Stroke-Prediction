import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import pickle

df= pd.read_csv("healthcare-dataset-stroke-data.csv")

bmean= df['bmi'].mean()
df['bmi'].fillna(bmean,inplace = True)


from sklearn.preprocessing import LabelEncoder

df_cat= df.select_dtypes(object)
df_num = df.select_dtypes(["int64","float64"])

for col in df_cat:
    le= LabelEncoder()
    df_cat[col]= le.fit_transform(df_cat[col])

df= pd.concat([df_cat,df_num],axis=1)


x= df.iloc[:,1:-1].values
y= df.iloc[:,-1].values

from sklearn.utils import check_array

x=check_array(x, dtype='float64',accept_sparse='csr')



xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=1,stratify=y)

ros = RandomOverSampler(random_state=1)

xsample,ysample = ros.fit_resample(xtrain,ytrain)

gbc= GradientBoostingClassifier(n_estimators=100)
gbc.fit(xsample,ysample)


pickle_out= open("mlmodel.pkl","wb")
pickle.dump(gbc,pickle_out)
pickle_out.close()
