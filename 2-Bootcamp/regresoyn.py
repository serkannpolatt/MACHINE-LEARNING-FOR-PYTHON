from matplotlib.cbook import print_cycles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data#özellik
y=iris.target#sınıf

#normalizasyon
x=(x-np.min(x))/(np.max(x)-np.min(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#çapraz doğrulama ile model eğitimi

#en yakın komşu
# from sklearn.neighbors import KNeighborsClassifier 
# knn=KNeighborsClassifier(n_neighbors=3)
# # print(knn)

# #10 fold çapraz doğrulama yapalım
# from sklearn.model_selection import cross_val_score
# fold_sayisi=10
# dogruluklar=cross_val_score(estimator=knn,X=x_train,y=y_train,cv=fold_sayisi)
# print("ortalama doğrılık:",np.mean(dogruluklar))
# print("doğrulukların standart sapması:",np.std(dogruluklar))

#IZGARA ARAMASI ÇAPRAZ DOĞRULAMA
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3) 

from sklearn.model_selection import GridSearchCV
grid={"n_neighbors":np.arange(1,50)}
knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid,cv=10)
# print(knn_cv.fit(x,y))

# print("En İyi K değeri:",knn_cv.best_params_)
# print("En İyi K değerine göre en iyi doğruluk değeri:",knn_cv.best_score_)



#lojistik regresyonla bulalım

# from sklearn.linear_model import LogisticRegression
# grid={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} #l1=lasso l2 =ridege

# logreg=LogisticRegression()
# logreg_cv=GridSearchCV(logreg,grid,cv=10)
# logreg_cv.fit(x,y)

# print("En İyi Hiper parametreler:",logreg_cv.best_params_)
# print("En İyi Hiper Parametreler göre en iyi doğruluk değeri:",logreg_cv.best_score_)