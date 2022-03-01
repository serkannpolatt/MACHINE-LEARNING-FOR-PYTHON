import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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
linear_reg=LinearRegression()


#doğrusal regresyon eğitini
linear_reg.fit(x,y)


#polinosal regresyon modeli eğitimni
polinom_regresyon=PolynomialFeatures()

x_polinom=polinom_regresyon.fit_transform(x)


#polinomsal regresyon eğitebilmek için polinomsal ozellikler ile
poly_reg=LinearRegression()
poly_reg.fit(x_polinom,y)


#test
y_tahmin_linear=linear_reg.predict(x)
y_tahmin_poly=poly_reg.predict(x_polinom)


plt.scatter(df.deneyim,df.maas)
plt.plot(x,y_tahmin_linear,color="red",label="Doğrusal")
plt.scatter(x,y_tahmin_poly,color="Blue",label="Polinomsal")
plt.xlabel("Dneneyim Yılı")
plt.ylabel("Maaş(TL)")
plt.title("Deneyim Maaş İlişkisi")
plt.grid(True)
# plt.show()

#print("Doğrusal Regresyon R Kare :{}",r2_score(y,y_tahmin_linear))
#print("Polinomsal Regreson R Kare :{}",r2_score(y,y_tahmin_poly))





