from turtle import shape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

veri=pd.read_csv("egitim.csv")
print("Verinin boyutu:",veri.shape)
print(veri.head())

label_filtre0=0
label_filtre1=1

veri=pd.concat([veri[veri["label"]==label_filtre0],veri[veri["label"]==label_filtre1]],axis=0)
print(veri.head())


Y_veri=veri["label"]

X_veri=veri.drop(["label"],axis=1)
print("X veri:",X_veri.shape)
print("Y veri:",Y_veri.shape)

resim_boyutu=int(np.sqrt(X_veri.shape[1]))
print(resim_boyutu)

resim1=X_veri.iloc[900].values
resim1=resim1.reshape((resim_boyutu,resim_boyutu))
plt.imshow(resim1,cmap="gray")
plt.axis("off")
plt.show()


resim2=X_veri.iloc[8000].values
resim2=resim2.reshape((resim_boyutu,resim_boyutu))
plt.imshow(resim2,cmap="gray")
plt.axis("off")
plt.show()


#eğitim test
X_egitim,X_test,Y_egitim,Y_test=train_test_split(X_veri,Y_veri,test_size=0.15,random_state=42)
print("X egitim:",X_egitim.shape)
print("X test:",X_test.shape)


x_egitim=X_egitim.T
x_test=X_test.T
y_egitim=Y_egitim.values.reshape(-1,1).T
y_test=Y_test.values.reshape(-1,1).T
print("x eegitim:",x_egitim.shape)
print("xtest:",x_test.shape)
print("y egitim:",y_test.shape)
print("y test:",y_test.shape)
