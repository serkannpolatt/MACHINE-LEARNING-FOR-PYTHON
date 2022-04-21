from json import load
from django.template import VariableDoesNotExist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris=load_iris()

data =iris.data
feature_names=iris.feature_names
y=iris.target

df=pd.DataFrame(data,columns=feature_names)
df["sinif"]=y
#print(df.head(2))


#Temel Bileşenlerden 2 tanesini ele alalım
from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=True) #whitten = normalize
pca.fit(data)

x_pca=pca.transform(data)

#print("variance ratio:",pca.explained_variance_ratio_)
#print("sum:",sum(pca.explained_variance_ratio_))


#Temel Bileşenleri Görselleştirelim

from sklearn.feature_selection import VarianceThreshold
X=[[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]
#print(X)

sel=VarianceThreshold(threshold=(0.8*(1-0.8)))
#print(sel.fit_transform(X))
# ilk sütunun elenmesini beklıyoruz
# zira orada 0 değeri olma olasılığı 5/6>0.8

## İSTATİKSEL MODEL SİÇİMİ

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X, y =load_iris(return_X_y=True)
#print(X.shape)

X_new=SelectKBest(chi=2,k=2).fit_transform(X, y)
#print(X_new.shape)

#MODEL TEMELLİ OZNİTELİK 
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

X,y=load_iris(return_X_y=True)
# print(X.shape)

