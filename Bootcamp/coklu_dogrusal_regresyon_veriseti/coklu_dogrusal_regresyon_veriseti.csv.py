from lib2to3.pgen2.literals import test
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df=pd.read_csv("coklu_dogrusal_regresyon_veriseti.csv",sep=";")
# print(df.head())

x=df.iloc[:,[0,2]].values #iloc index tabanlı lokasyon belırler -- deneyim ve yaşı bağımsız değişkenler olarak alır
# print(x)
y=df.maas.values.reshape(-1,1)# maaşı bağımlı değişken olarak alalım
# print(y)

#Çoklu Doğrusal regresyon modeli eğitimi

coklu_dogrusal_regresyon=LinearRegression()

#dogrusal regresyon eğitimi
coklu_dogrusal_regresyon.fit(x,y)
 
#test 1

test_verisi1=np.array([[10,35]]) #deneyim,=10 yaş=35
test_sonucu1=coklu_dogrusal_regresyon.predict(test_verisi1)
# print("10 yıllık deneyim ve 35 yaş sonucu çıkan maaş:{}TL".format(test_sonucu1[0]))


#test 2 

test_verisi2=np.array([[5,35]])
test_sonucu2=coklu_dogrusal_regresyon.predict(test_verisi2)
# print("5 yıllık deneyim ve 35 yaş sonucu çıkan maaş:{}".format(test_sonucu2[0]))


