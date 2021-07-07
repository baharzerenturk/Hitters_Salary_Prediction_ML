#Salary Prediction with Machine Learning

######################################
#İŞ PROBLEMİ
######################################

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

######################################
#VERİ SETİ HİKAYESİ
######################################

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir
# parçasıdır. Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.

######################################
#DEĞİŞKENLER
######################################

# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı +
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı +
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı +
# Years: Oyuncunun major liginde oynama süresi (sene) +
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı +
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı +
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör


from datetime import date
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("WEEK_07/datasets_07/hitters.csv")
df.head()

df.describe()

##########################################################
#OUTLIERS:
##########################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

##########################################################
#MISSING VALUES:
##########################################################
df['Years_Range'] = pd.qcut(x=df['Years'], q=4,labels = ["Beginner", "Intermediate", "UpperIntermediate", "Advanced"])

for col in df.columns:
    if col in num_cols:
        df[col] = df[col].fillna(df.groupby("Years_Range")[col].transform("median"))

missing_values_table(df)

#############################################
for i in num_cols:
    fig, axes = plt.subplots(1, 2, figsize = (17,4))
    df.hist(str(i), bins=10, ax=axes[0])
    df.boxplot(str(i), ax=axes[1], vert=False);

    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[0].set_title(i + " | Histogram")
    axes[1].set_title(i + " | Boxplot")
    plt.show()
# FEATURE ENGINEERING

df['HmRun_Ranges'] = pd.qcut(x=df['HmRun'], q=4 ,labels = ["D_HmRun", "C_HmRun", "B_HmRun", "A_HmRun"])
df["RBI_Ranges"] = pd.qcut(x=df["RBI"], q=4, labels=["D_RBI","C_RBI","B_RBI","A_RBI"])

#HmRun_Ranges için Anova:

#Normallik Testi:
for group in list(df["HmRun_Ranges"].unique()):
    pvalue = shapiro(df.loc[df["HmRun_Ranges"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
#H0 reddedilir, dağılımlar normal değil

# parametrik anova testi:
pvalue = kruskal(df.loc[df["HmRun_Ranges"] == "D_HmRun", "Salary"],
         df.loc[df["HmRun_Ranges"] == "C_HmRun", "Salary"],
         df.loc[df["HmRun_Ranges"] == "B_HmRun", "Salary"],
         df.loc[df["HmRun_Ranges"] == "A_HmRun", "Salary"])[1]
print("p-value: %.4f" % pvalue)

#H0 red

#RBI_Ranges için Anova:

#Normallik Testi:
for group in list(df["RBI_Ranges"].unique()):
    pvalue = shapiro(df.loc[df["RBI_Ranges"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
#H0 reddedilir, dağılımlar normal değil

# parametrik anova testi:
pvalue = kruskal(df.loc[df["RBI_Ranges"] == "D_RBI", "Salary"],
         df.loc[df["RBI_Ranges"] == "C_RBI", "Salary"],
         df.loc[df["RBI_Ranges"] == "B_RBI", "Salary"],
         df.loc[df["RBI_Ranges"] == "A_RBI", "Salary"])[1]
print("p-value: %.4f" % pvalue)


#####################################################
#FEATURE 1 : RBIXHMRUN

scaler = MinMaxScaler(feature_range=(1,4))
df["RBI_Scaled"]= scaler.fit_transform(df[["RBI"]])
df["HmRun_Scaled"]= scaler.fit_transform(df[["HmRun"]])

df["RBIXHmRun"] = df["RBI_Scaled"] * df["HmRun_Scaled"]

df["RBIXHmRun_Cat"] = pd.qcut(x=df["RBIXHmRun"], q=4, labels=["D", "C", "B", "A"])
df["RBIXHmRun_Cat"].value_counts()

#Normallik Testi:
for group in list(df["RBIXHmRun_Cat"].unique()):
    pvalue = shapiro(df.loc[df["RBIXHmRun_Cat"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
#H0 reddedilir, dağılımlar normal değil.

pvalue = kruskal(df.loc[df["RBIXHmRun_Cat"] == "D", "Salary"],
         df.loc[df["RBIXHmRun_Cat"] == "C", "Salary"],
         df.loc[df["RBIXHmRun_Cat"] == "B", "Salary"],
         df.loc[df["RBIXHmRun_Cat"] == "A", "Salary"])[1]
print("p-value: %.4f" % pvalue)
#H0 reddedilir. Oluşturulan featureın sınıflarının maaşa etkisi arasında istatistiksel olarak anlamlı bir farklılık vardır.

#####################################################
#FEATURE 2 : RUNSXWALKS

scaler = MinMaxScaler(feature_range=(1,4))
df["Runs_Scaled"]= scaler.fit_transform(df[["Runs"]])
df["Hits_Scaled"]= scaler.fit_transform(df[["Hits"]])

df["RunsXHits"] = df["Runs_Scaled"] * df["Hits_Scaled"]
df["RunsXHits_Cat"] = pd.qcut(x=df["RunsXHits"], q=4, labels=["D", "C", "B", "A"])
df["RunsXHits"].describe()

#Normallik Testi:
for group in list(df["RunsXHits_Cat"].unique()):
    pvalue = shapiro(df.loc[df["RunsXHits_Cat"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
#H0 reddedilir, dağılımlar normal değil.

pvalue = kruskal(df.loc[df["RunsXHits_Cat"] == "D", "Salary"],
         df.loc[df["RunsXHits_Cat"] == "C", "Salary"],
         df.loc[df["RunsXHits_Cat"] == "B", "Salary"],
         df.loc[df["RunsXHits_Cat"] == "A", "Salary"])[1]
print("p-value: %.4f" % pvalue)
#H0 reddedilir. Oluşturulan featureın sınıflarının maaşa etkisi arasında istatistiksel olarak anlamlı bir farklılık vardır.

#####################################################
#FEATURE 3 : YEARSXCHITS

df["Years_Scaled"]= scaler.fit_transform(df[["Years"]])
df["Chits_Scaled"]= scaler.fit_transform(df[["CHits"]])

df["YearsXChits"] = df["Years_Scaled"] * df["Chits_Scaled"]

df["YearsXChits_Cat"] = pd.qcut(x=df["YearsXChits"], q=4, labels=["D", "C", "B", "A"])
df["YearsXChits_Cat"].value_counts()

for group in list(df['YearsXChits_Cat'].unique()):
    pvalue = shapiro(df.loc[df['YearsXChits_Cat'] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)

pvalue = kruskal(df.loc[df['YearsXChits_Cat'] == "D", "Salary"],
                 df.loc[df['YearsXChits_Cat'] == "C", "Salary"],
                 df.loc[df['YearsXChits_Cat'] == "B", "Salary"],
                 df.loc[df['YearsXChits_Cat'] == "A", "Salary"])[1]
print("p-value: %.4f" % pvalue)
#H0 reddedilir. Oluşturulan featureın sınıflarının maaşa etkisi arasında istatistiksel olarak anlamlı bir farklılık vardır.

df.head()

#FEATURE 4:

df['At/CAt'] = df['AtBat'] / df['CAtBat']
df['Hits/CHits'] = df['Hits'] / df['CHits']
df['Runs/CRuns'] = df['Runs'] / df['CRuns']


#ONE HOT ENCODING PART

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)


# RARE ANALYSER
#Ratio'su 0.01 altında olan bir sınıf olmadığı için, bir değişiklik olmadı.
# rare_analyser(df, "Salary", cat_cols)
# df = rare_encoder(df, 0.01)
# rare_analyser(df, "Salary", cat_cols)

#STANDARD SCALER PART

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])

from helpers.eda import check_df
check_df(df)

#MODELLING

dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1)
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_   # -0.028081843510618307

# coefficients (w - weights)
reg_model.coef_

# Train RMSE
y_pred = reg_model.predict(X_train) #model nesnesine train bağımsız değişkenlerini göndererek bağımlı değişkeni tahmin ettirdik.buradan train seti için tahmin edilen değerler gelecek. y_tarini şu anda saklıyoruz.
np.sqrt(mean_squared_error(y_train, y_pred))  #0.3360589814387444. #train seti için tahmin edilen değerlerle train seti için gerçek y değerlerini kıyaslıyoruz.

# TRAIN RKARE
reg_model.score(X_train, y_train) #0.7359743757704422. #açıklanabilirlik yüzde 74 geldi. yani veri setindeki bağımsız değişkenler, bağımlı değişkendeki değişimin yüzde 74unu açıklıyor denilebilir.

#Test RMSE
y_pred = reg_model.predict(X_test)  # elimde göstermediğim bağımsız değişkenleri tahmin etme
np.sqrt(mean_squared_error(y_test, y_pred)) #0.43348868547778613

# Test RKARE
reg_model.score(X_test, y_test)  #0.6936060914169069

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error"))) #0.42820527440738737

#neg_mean_squared_error, negatif skorlar ürettiği için fonksiyonu - ile kullandık.
#10 katlı çapraz doğrulama ile 10 farklı error'ın mean sqrt ulaşacağız. Bunun bütün verinin kullanılarak yapıldığına dikkat edilmeli. Çünkü bu verisetinde gözlemler çok az.














