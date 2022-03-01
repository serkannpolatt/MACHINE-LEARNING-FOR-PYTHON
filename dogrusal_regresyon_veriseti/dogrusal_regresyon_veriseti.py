import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df=pd.read_csv("dogrusal_regresyon_veriseti.csv",sep=";")
#print(df.head())

plt.scatter(df.deneyim,df.maas)
plt.xlabel("Deneyim(Yıl)")
plt.ylabel("Maaş(TL)")
plt.title("Deneyim Maaş İlişkisi")
plt.grid(True)
# plt.show()



#DOĞRUSAL REGRESYON MODEL EĞİTİMİ

linear_reg=LinearRegression()
#data içerisinde bulunan maas ve deneyim sütunlarını numpy arraye çevir)

x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

#doğrusal regresoyon eğitiimi

linear_reg.fit(x,y)

#y eksenini kestiği nokta intercept bulunması
y_ekseni_kesisim=np.array([0]).reshape(1,-1)
b0=linear_reg.predict(y_ekseni_kesisim)
# print("b0:",b0)

#y eksenini kestğiğ nokta intercept
b0_=linear_reg.intercept_
# print("b0_:",b0_)

#egim(slope) bulunması
b1=linear_reg.coef_
# print("b1:",b1)

deneyim=11

maas=1663+1138*deneyim
#y eksenini kestiği nokta ve eğime göre doğrusal model oluşturulur


#11 yıllık deneyime sahip birini maaşı tahmin edilir
maas_yeni=1663+1138*deneyim
# print(maas_yeni)

#11 yıllık deneyime sahip birinin maaşı predict metodu ile tahmin edilir
sonuc=linear_reg.predict(np.array([deneyim]).reshape(1,-1))
# print("11 yılık deneyime sahip birinin maaaşı:{} TL".format(sonuc[0]))



#doğrusal regresyon modeli ile test/tahmin/görselleştirme

array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.figure()
plt.scatter(x,y)

#0-15 yılları arasında deneyime sahip insanların maaşı tahnin edilir
y_head=linear_reg.predict(array) #y_head=maaş

plt.plot(array,y_head,color="red") #deneyim,maaş
plt.xlabel("Deneyim(Yıl)")
plt.ylabel=("Maaş (TL)")
plt.title("Deneyim Maaş İlişkisi")
plt.grid(True)
# plt.show()


