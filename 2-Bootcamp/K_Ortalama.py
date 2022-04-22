#K-ortalama kümele algoritmasını eğitebilmek için gerekli olan veriyi oluşturcaz

from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#küme1
x1=np.random.normal(25,5,1000)
y1=np.random.normal(25,5,1000)

#küme2
x2=np.random.normal(55,5,1000)
y2=np.random.normal(60,5,1000)

#küme3
x3=np.random.normal(55,5,1000)
y3=np.random.normal(15,5,1000)

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((x1,x2,x3),axis=0)

dictionary={"x":x,"y":y}

data=pd.DataFrame(dictionary)
# print(data.head())

# plt.figure()
# plt.scatter(x1,y1)
# plt.scatter(x2,y2)
# plt.scatter(x3,y3)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("K-Ortalama Kümeleme Yönetimi İçin Oluşturulan Veri Seti")
# plt.show()

#k ortalama algortıması veriyi bole gorecek
# plt.figure()
# plt.scatter(x1,y1,color="Black")
# plt.scatter(x2,y2,color="Black")
# plt.scatter(x3,y3,color="Black")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("K-Ortalama Kümeleme Yönetimi İçin Oluşturulan Veri Seti")
# plt.show()



#dirsek yöntemi
from sklearn.cluster import KMeans
wcss=[]

for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# plt.figure()
# plt.plot(range(1,15),wcss)
# plt.xticks(range(1,15))
# plt.xlabel("Küme Sayısı(K)")
# plt.ylabel("wcss")
# plt.show()

#dirsekde hangi şeyler içeriye giriyor

k_ortalama=KMeans(n_clusters=3)
kumeler=k_ortalama.fit_predict(data)

data["label"]=kumeler

# plt.figure()
# plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red",label="Küme1")
# plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green",label="küme2")
# plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue",label="Küme3")
# plt.scatter(k_ortalama.cluster_centers_[:,0],k_ortalama.cluster_centers_[:,1],color="yellow")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("3-Ortalama Kümeleme Sonucu")
# plt.show()

#dendogram gosterimi
from scipy.cluster.hierarchy import linkage, dendrogram

# merg=linkage(data,method="ward")
# dendrogram(merg,leaf_rotation=90)
# plt.xlabel("Veri Noktası")
# plt.ylabel=("Öklid Mesafesi")
# plt.show()


#hierarşik kümeleme
from sklearn.cluster import AgglomerativeClustering

hiyerarsi_kume=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
kume=hiyerarsi_kume.fit_predict(data)
data["label"]=kume

# plt.figure()
# plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red",label="Küme 1")
# plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green",label="Küme 2")
# plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue",label="Küme 3")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("3-Ortalama Kümeleme Sonucu")
# plt.show()


