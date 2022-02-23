#KÜTÜPHANELERİ İMPORT EDELİM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#UYARILARI KAPATALIM
import warnings
warnings.filterwarnings("ignore")



# VERİYİ İÇE AKTARALIM #

veri= pd.read_csv("olimpiyat.csv")
result=veri.head()
#print(result)
result=veri.info()
#print(result)

# COLUMN ADLARINI DEĞİŞTİRME #

veri.rename(columns={"ID" : "id",
                    "Name": "isim",
                  "Gender": "cinsiyet",
                   "Age"  : "yas", 
                  "Height": "boy",
                  "Weight": "kilo",
                    "Team": "takim",
                     "NOC": "uok",
                   "Games": "oyunlar",
                   "Year" : "yil",
                 "Season" : "sezon",
                   "City" : "sehir",
                  "Sport" : "spor",
                   "Event": "etkinlik",
                   "Medal": "madalya"}, inplace=True )
# result=veri.head() 


# DROP FONKSİYONUYLA İD VE OYUNLAR SÜTUNLARINI ÇIKARALIM #

veri=veri.drop(["id","oyunlar"],axis=1)
result=veri.head(2)                    

essiz_etkinlik=pd.unique(veri.etkinlik)
#print("Eşsiz etkinlik sayısı:{}".format(len(essiz_etkinlik)))
essiz_etkinlik[:10]




# HER BİR ETKİNLİK İTERATİF OLARAK DOLAŞ #
# ETKİNLİK ÜZERİNDE BOY-KİLO ORTALAMALARINI HESAPLA #
# TKİNLİK ÜZERİNDE BOY-KİLO DEĞERLERİNİ ETKİNLİK ORTALAMLARINA EŞİTLE #

veri_gecici=veri.copy() # gerçek veriyi bozmamak için kopyasını oluşturalım #
boy_kilo_liste=["boy","kilo"]

for e in essiz_etkinlik: # liste içerisinde dolaş #
     # etkinlik filtresi oluşturalım #
     etkinlik_filtre =veri_gecici.etkinlik==e
     # veriyi etkinliğe göre filtreyelim #
     veri_filtreli=veri_gecici[etkinlik_filtre]

     # boy kilo için etkinlik özelinde ortalmlarını hesaplayalım #
     for s in boy_kilo_liste:
         ortalama=np.round(np.mean(veri_filtreli[s]),2)
         if ~np.isnan(ortalama): # eğer etkinlik özelinde aram varsa #
             veri_filtreli[s]=veri_filtreli[s].fillna(ortalama)
         else: # eğer etkinlik özelinde ortalama varsa ortalamayı hesapla #
             tum_veri_ortalamasi =np.round(np.mean(veri[s]),2)
             veri_filtreli[s]=veri_filtreli[s].fillna(ortalama)
     #etkinlik özelinde kayıp değerleri doldurulmuş olan veriyi,veri geciciye #
     veri_gecici[etkinlik_filtre]=veri_filtreli

# KAYIP DEĞERLERİ GİDERİLMİŞ OLAN GEÇİCİ VERİYİ GERÇEK VERİYE EŞİTLEYELİM #
veri=veri_gecici.copy()
veri.info() # boy kilo sütunlarında kayıp değer sayısna bakalım #



# YAŞ SÜTUNUNA KAYIP VERİ DOLDURMA #

yas_ortalamasi=np.round(np.mean(veri.yas),2)
#print("Yaş Ortalaması:{}".format(yas_ortalamasi))
veri["yas"] =veri["yas"].fillna(yas_ortalamasi)
#veri.info()



# MADALYA ALMAYAN SPORCULARI VERİ SETİNDEN ÇIKARALIM #

madalya_degiskeni=veri["madalya"]
pd.isnull(madalya_degiskeni).sum()

madalya_degiskeni_filtresi =~pd.isnull(madalya_degiskeni)
result=veri.head(5)
result=veri.info()




# VERİYİ KAYDEDELİM #

result=veri.to_csv("olimpiyat_temiz.csv",index=False)

print(result)





# TEK DEĞİŞKENLİ VERİ ANALİZİ #

def plotHistogram(degisken):
     """
           GİRDİ : Değişken/sütun ismi
           ÇIKTI : İlgili değişken histogramı
     """      
     plt.figure()
     plt.hist(veri[degisken],bins=85,color ="blue")
     plt.xlabel(degisken)
     plt.ylabel("Frekans")
     plt.title("Veri Sıklığı - {}".format(degisken))
    #plt.show()


# TÜM SAYISAL DEĞER İÇİN HİSTOGRAM ÇİZELİM #

sayısal_degisken=["yas","boy","kilo","yil"]
for i in sayısal_degisken:
    plotHistogram(i)


plt.boxplot(veri.yas)
plt.title("Yaş Değişkeni İçin Kutu Grafği")
plt.xlabel("yas")
plt.ylabel("Değer")
# plt.show()



# ÖNCELİKLE ÇUBUK GRAFİĞİNİ ÇİZDİRCEZ MODELİ YAZALIM #

def plotBar(degisken,n=5):
     """
        Girdi:Değişken/sütun ismi
              n=Gösterilecek eşsiz değer sayısı
        Çıktı: Çubuk Grafiği
     """
     veri_=veri[degisken]
     veri_sayma=veri_.value_counts()
     veri_sayma=veri_sayma[:n] 
     plt.figure()
     plt.bar(veri_sayma.index,veri_sayma,color="orange")        
     plt.xticks(veri_sayma.index,veri_sayma.index.values)
     plt.xticks(rotation=45)
     plt.ylabel("Frekans")
     plt.title("Veri Sıklığı - {}".format(degisken))
#    plt.show()
#    print("{}:\n {}".format(degisken,veri_sayma))


kategorik_degisken=["isim","cinsiyet","takim","uok","sezon","sehir","spor","madalya"]
for i in kategorik_degisken:
      plotBar(i)                      
#     print(plotBar(i))





# CİNSİYETE GÖRE BOY-KİLO KARŞILAŞTIRMA #

erkek=veri[veri.cinsiyet=="M"]


kadin=veri[veri.cinsiyet=="F"]

plt.figure()
plt.scatter(kadin.boy,kadin.kilo,alpha=0.4,label="Kadın",color="orange")
plt.scatter(erkek.boy,erkek.kilo,alpha=0.4,label="Erkek",color="blue")
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Boy Kilo ilişkisi")
plt.legend()
# plt.show()


# SAYISAL SÜTUNLAR ARASINDAKI İLİŞKİ #

result=veri.loc[:,["yas","boy","kilo",]].corr() #korelasyon tablosu
#print(result)





# MADALYA YAŞ İLİŞKİSİ #

# SPORCULARI MADALYALARINA GÖRE AYARLAMA #
veri_gecici=veri.copy()
veri_gecici=pd.get_dummies(veri_gecici,columns=["madalya"])
# print(veri_gecici.head(2))

result=veri_gecici.loc[:,["yas","madalya_Bronze","madalya_Gold","madalya_Silver"]].corr()
# print(result)



# TAKIMLARIN KAZANDIKLARI ALTIN-GÜMÜŞ-BRONZ MADALYA SAYILARI #

#print(veri_gecici[["takim","madalya_Bronze","madalya_Gold","madalya_Silver",]].groupby(["takim"], as_index=False).sum().sort_values(by="madalya_Gold",ascending=False)[:10])







# ÇOK DEĞİŞKENLİ VERİ ANALİZİ #

# PİVOT TABLOSU #
veri_pivot=veri.pivot_table(index="madalya",columns="cinsiyet",
                             values=["boy","kilo","yas"],
                             aggfunc={"boy":np.mean,"kilo":np.mean,"yas":[min,max,np.std]})

# print(veri_pivot.head())







# ANOMALİ TESPİTİ #

def anomaliTespiti(df,ozellik):
     outlier_indices=[]

     for c in ozellik:
         #1.çeyrek
         Q1=np.percentile(df[c],25)
         #3.çeyrek
         Q3=np.percentile(df[c],75)
         #IQR=Inter Quartile Range
         IQR=Q3-Q1
         #aykırı değer için ek adım miktarı
         outlier_step=1.5*IQR
         #aykırı değeri ve de bulunduğu indeksi tespit edelim
         outlier_list_col=df[(df[c]<Q1 - outlier_step)|(df[c]>Q3 +outlier_step)].index
         #tespit edilen indeksler depolayalım
         outlier_indices.extend(outlier_list_col) 
        
     #eşsiz aykırı değerleri bulalım
     outlier_indices=Counter(outlier_indices)
     #eğer bir ornek v adet sutunda farklı değer ise bunu aykırı kabul edelim 
     multiple_outliers=list(i for i,v in outlier_indices.items() if v >1)
     return multiple_outliers


veri_anomali=veri.loc[anomaliTespiti(veri,["yas","kilo","boy"])]
# print(veri_anomali.spor.value_counts())


plt.figure()
plt.bar(veri_anomali.spor.value_counts().index, veri_anomali.spor.value_counts().values)
plt.xticks(rotation=30)
plt.title("Anomaliye rastalnana spor branşları")
plt.ylabel("Frekans")
plt.grid(True,alpha=0.5)
# plt.show()


veri_gym=veri_anomali[veri_anomali.spor=="Gynmastics"]
# print(veri_gym)


# print(veri_gym.etkinlik.value_counts())


# ZAMAN SERİLERİ VERİ ANALİZİ #

veri_zaman=veri.copy()
veri_zaman.head(3)

essiz_yillar=veri_zaman.yil.unique()
# print(essiz_yillar)


# OLİMPİYATLARI YAPTIKLARI YILLARA GÖRE SIRALAYALIM #

dizili_array=np.sort(veri_zaman.yil.unique())
print(dizili_array)

plt.figure()
plt.scatter(range(len(dizili_array)),dizili_array)
plt.grid(True)
plt.ylabel("Yıllar")
plt.title("Olimpiyatlar Çift Yıllarda Düzenlenir")
# plt.show()



# VERİ İÇERİSİNDE BULUNAN YIL DEĞERLERİNİ datetime VERİ TİPİNE DÖNÜŞTÜRELİM #
tarih_saat_nesnesi=pd.to_datetime(veri_zaman["yil"],format="%Y")
# print(tarih_saat_nesnesi.head(3))
veri_zaman["tarih_saat"]=tarih_saat_nesnesi
# print(veri_zaman.head(3))


#veri_Zaman DEĞİŞKENİNİ ANA İNDEKSİNİ, datetime OLAN  tarih_saat DEĞERİNE #

veri_zaman=veri_zaman.set_index("tarih_saat")
veri_zaman.drop(["yil"],axis=1,inplace=True)
# print(veri_zaman)


periyodik_veri=veri_zaman.resample("2A").mean() # 2yıllık periyotlar halinde #
# print(periyodik_veri.head())

periyodik_veri.dropna(axis=0,inplace=True)
# print(periyodik_veri.head(3))

plt.figure()
periyodik_veri.plot()
plt.title("Yıllara Göre Ortalama Yaş, Boy ve Ağırlık Değişimi")
plt.xlabel("Yıl")
plt.grid(True)
# plt.show()

# KAYIP VERİLERİ ÇIKARALIM #
periyodik_veri.dropna(axis=0,inplace=True)
# print(periyodik_veri.head())




## YILLARA GÖRE MADALYA SAYILARI #

veri_zaman=pd.get_dummies(veri_zaman,columns=["madalya"])
# print(veri_zaman.head(2))
periyodik_veri=veri_zaman.resample("2A").sum()
# print(periyodik_veri.head())


# KAYIP VERİLERİ ÇIKARMA #

periyodik_veri=periyodik_veri[~(periyodik_veri==0).any(axis=1)]
# print(periyodik_veri.tail())

plt.figure()
periyodik_veri.loc[:,["madalya_Bronze","madalya_Gold","madalya_Silver"]].plot()
plt.title("Yıllara Gore Madalya Sayıları")
plt.ylabel("Sayı")
plt.xlabel("Yıl")
plt.grid("True")
# plt.show()

# YILLARA VE SEZONLARA GORE MADALYA SAYILARI #

yaz=veri_zaman[veri_zaman.sezon=="Summer"]
kis=veri_zaman[veri_zaman.sezon=="Winter"]
# print(kis.head())


periyodik_veri_yaz=yaz.resample("A").sum()
periyodik_veri_yaz=periyodik_veri_yaz[~(periyodik_veri_yaz==0).any(axis=1)]
# print(periyodik_veri_yaz.head(2))

plt.figure()
periyodik_veri_yaz.loc[:,["madalya_Bronze","madalya_Gold","madalya_Silver"]]
plt.title("Yıllara Göre Madalya Sayıları - Yaz Sezonu")
plt.ylabel("Sayı")
plt.xlabel("Yıl")
plt.grid(True)
# plt.show()