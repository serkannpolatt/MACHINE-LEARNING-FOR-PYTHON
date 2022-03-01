from unicodedata import numeric
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

data=pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")
#print(data.head())

sns.scatterplot(data=data,x="lumbar_lordosis_angle",y="pelvic_tilt numeric",hue="class")
plt.xlabel("lomber lordoz aşısı")
plt.ylabel("pelvik eğim")
plt.legend()
#plt.show()


data["class"]=[1 if each =="Abnormal" else 0 for each in data ["class"]]
#print(data.head(2))


y=data["class"].values # sınıfları y değişkenin içerisine koyalım
x_data= data.drop(["class"],axis=1) # özellikler x_Data değişklenin içerisine atalım


#normalizasyon
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


#eğitim test bolunmesi 
from sklearn.model_selection import train_test_split
#%15 test , %85 eğitim
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=1)


#K-EN YAKIN KOMŞU ALGORTIMASI EĞİTİM VE TESTİ
from sklearn.neighbors import KNeighborsClassifier
komsu_sayisi=4
knn=KNeighborsClassifier(n_neighbors=komsu_sayisi)
knn.fit(x_train,y_train)

prediction=knn.predict(x_test)
#print("{} En yakın komşu modeli test doğruluk:{}".format(komsu_sayisi,knn.score(x_test,y_test)))

score_list=[]
for each in range(1,50):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,50),score_list)
plt.xlabel("k değerleri")
plt.ylabel("Doğruluk")
plt.title("En iyi K değerinin bulunması")
#plt.show()


# destek vektör makinesi algoritmasının eğitimi ve testi
from sklearn.svm import SVC
#EĞİTİMİ
svm=SVC(random_state=1)
svm.fit(x_train,y_train)
#testi
#print("Destek Vektör Makinesi Modeli Test Doğruluğu :{}".format(svm.score(x_train,y_train)))


#karar ağacı
from sklearn.tree import DecisionTreeClassifier
#karar ağaç eğitimi
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
#print("Karar Ağacı Modeli Eğitim Testi Doğruluğu : {}".format(dt.score(x_train,y_train)))


#RASTGELE ORMAN ALGORİTMASI EĞİTİM TESTİ
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
#print("Rastgele Orman Modeli Test Dığruluk :{}".format(dt.score(x_test,y_test)))


#confisıon matrix
from sklearn.metrics import confusion_matrix
y_pred=rf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
#print(cm)


#sıcaklık haritası
f, ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="white",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
#plt.show()

