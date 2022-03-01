import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #doğrusal
from sklearn.preprocessing import PolynomialFeatures 


df=pd.read_csv("polinomsal_regresyon_veriseti.csv",sep=";")
# print(df.head())

y=df.araba_max_hiz.values.reshape(-1,1)
x=df.araba_fiyat.values.reshape(-1,1)

# plt.scatter(x,y)
# plt.ylabel("Araba Maksimum Hızı")
# plt.xlabel("Araba Fiyatı")
# plt.title("Araba Hız ve Fiyat İlişkisi")
# plt.grid(True)
# plt.show()

#doğrusal regresyon model eğitimi
lr=LinearRegression()
#doğrusal regresoyn eğitimi
lr.fit(x,y)

#tahmin
y_tahmin=lr.predict(x)

# plt.scatter(x,y)
# plt.plot(x,y_tahmin,color="red")
# plt.ylabel("Araba Maksimum Hızı")
# plt.xlabel("Araba Fiyatı")
# plt.title("Araba Hız ve Fiyat İlişkisi")
# plt.grid(True)
# plt.show()

araba_fiyatı=10000
# print("10 ymilyon tllik araba hizi tahmimni:",lr.predict((np.array([araba_fiyatı]).reshape(1,-1))))


#polinomsal regresoyon = y=b0+b*1x+b2*x^2....

#polinomsal ozellikler

polinom_regresyon=PolynomialFeatures(degree=4) #4.derecedenx

x_polinom=polinom_regresyon.fit_transform(x)
# print(x_polinom)


#polinomsal regresyon eğitebilmek için polinomsal ozellikler ile

lr2=LinearRegression()
lr2.fit(x_polinom,y)

#tahmin
y_tahmin2=lr2.predict(x_polinom)

# plt.scatter(x,y)
# plt.plot(x,y_tahmin,color="red",label="Doğrusal")
# plt.plot(x,y_tahmin2,color="green",label="Polinomsal")
# plt.legend()
# plt.ylabel("Araba Maksimum Hızı")
# plt.xlabel("Araba Fiyatı")
# plt.title("Araba HIZ fiyat İlşKİİS")
# plt.grid(True)
# plt.show()


