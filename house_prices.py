import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import xgboost as xgb

from scipy import stats
from scipy.stats import norm

plt.rcParams['figure.figsize'] = (10.0, 8.0)
#Meload dataset
train = pd.read_excel("/data/train.xls")
test = pd.read_excel("/data/test.xls")

#Mengamati data
train.head()
test.head()

#Menampilkan informasi data set
train.shape[0],train.shape[1]
test.shape[0],test.shape[1]

#Menampilkan informasi statistik dari SalePrice
train.SalePrice.describe()

#Menampilkan semua feature
train.info()
test.info()

#Menampilkan kolom yang kosong/data hilang/tidak sesuai
train.columns[train.isnull().any()]
#Menghitung persehtase kolom yang hilang
#missing = train.isnull().sum()/len(train)*100
missing = train.isnull().sum()/len(train)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing
#Visualisasi data yang hilang
missing = missing.to_frame()
missing.columns = ['count']
missing.index.names = ['Name']
missing['Name'] = missing.index
#Plot chart
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=missing)
plt.xticks(rotation = 90)
plt.show()

#Menampilkan Skenwess
train['SalePrice'].skew()
#Menampilkan grafik skewness
sns.distplot(train['SalePrice'])
#Transformasi skewmess
target = np.log(train['SalePrice'])
print ('Skewness is', target.skew())
sns.distplot(target)

#Memisahkan variable numeric dengan categorical
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
numeric_data.shape[1]
cat_data.shape[1]

#Menghapus kolom Id
del numeric_data['Id']
numeric_data.shape[1]
#Menampilkan korelasi
corr = numeric_data.corr()
sns.heatmap(corr)
#Menampilkan 15 highest correlation
print(corr['SalePrice'].sort_values(ascending=False)[:15], '\n')
#Menampilkan 5 lowest correlation
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

#Menampilkan pivot table
train['OverallQual'].unique()
pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(pivot)
pivot.plot(kind='bar', color='red')
sns.jointplot(x=train['OverallQual'], y=train['SalePrice'])
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'])
sns.jointplot(x=train['GarageArea'], y=train['SalePrice'])

#Mengamati apakah ada kolom kososng pada variable numeric
numeric_data.columns[numeric_data.isnull().any()]

numeric_data.describe()

#Menampilkan informasi categorical variable
print(cat_data.describe())
cat_data.columns[cat_data.isnull().any()]
#Mnemapilkan median dari SaleCondition
sp_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
sp_pivot
sp_pivot.plot(kind='bar',color='red')

#Mendefinisikan p-value untuk ANOVA test
cat = [f for f in train.columns if train.dtypes[f] == 'object']
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['SalePrice'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

cat_data['SalePrice'] = train.SalePrice.values
k = anova(cat_data)
k['disparity'] = np.log(1./k['pval'].values)
sns.barplot(data=k, x = 'features', y='disparity')
plt.xticks(rotation=90)
plt

#Menampilkan histogram
#create numeric plots
num = [f for f in train.columns if train.dtypes[f] != 'object']
num.remove('Id')
nd = pd.melt(train, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
n1

#Menampilkan boxplot
def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)

cat = [f for f in train.columns if train.dtypes[f] == 'object']

p = pd.melt(train, id_vars='SalePrice', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, height=5)
g = g.map(boxplot, 'value','SalePrice')
g
#####################Data Preprocessing#####################
#Hapus aoulier pada GrLivArea untuk data di atas 4000
train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)
train.shape
#Hapus aoulier pada Garage untuk data di atas 1200
train.drop(train[train['GarageArea'] > 1200].index, inplace=True)
train.shape

null=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
print (null)

print ("Unique values are:", train.MiscFeature.unique())
data=train.select_dtypes(include=[np.number]).interpolate().dropna()

print (sum(data.isnull().sum()))
categoricals=train.select_dtypes(exclude=[np.number])
categoricals.describe()

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

condition_pivot=train.pivot_table(index="SaleCondition",values="SalePrice",aggfunc=np.median)
condition_pivot.plot(kind="bar",color="blue")
plt.show()

def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

print (train.LotShape.unique())
pt=train.pivot_table(index="LotShape",values="SalePrice",aggfunc=np.median)
pt.plot(kind="bar",color="blue")
plt.show()

LotShape_D=pd.get_dummies(train.LotShape)
train=pd.concat([train,LotShape_D],axis=1)
test=pd.concat([test,LotShape_D],axis=1)

################### MODEL TRAINING & EVALUATION #################
y=np.log(train.SalePrice)
x=data.drop(["SalePrice","Id"], axis=1)
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=.33)

#A. Linear Regression
lr=linear_model.LinearRegression()
model=lr.fit(X_train,y_train)
print("Training - R squared is: \n", model.score(X_test,y_test))
predictions=model.predict(X_test)
print("Training - RMSE is: \n", mean_squared_error(y_test,predictions))
actual_values=y_test
plt.scatter(predictions,actual_values,alpha=.75,color="b")
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.title("Linear Regression Model")
plt.show()

#B. Ridge Regularization
for i in range(-2,3):
 alpha=10**i
 rm=linear_model.Ridge(alpha=alpha)
 ridge_model= rm.fit(X_train,y_train)
 print("Training - R squared is: \n", ridge_model.score(X_test,y_test))
 preds_ridge=ridge_model.predict(X_test)
 print("Training - RMSE is: \n", mean_squared_error(y_test,preds_ridge))
 plt.scatter(preds_ridge,actual_values,alpha=.75,color="b")
 plt.xlabel("Predicted Price")
 plt.ylabel("Actual Price")
 plt.title("Ridge Regularization with alpha = {}".format(alpha))
 overlay="R squared is: {}\n RMSE is: {}".format(
 ridge_model.score(X_test,y_test),
 mean_squared_error(y_test,preds_ridge))
 plt.annotate(s=overlay,xy=(12.1,10.6),size="x-large")
 plt.show()

#C. Lasso Regression
alpha = 0.00099
lasso = Lasso(alpha=alpha, max_iter=50000)
lasso_model= lasso.fit(X_train,y_train)
print("Training - R squared is: \n", lasso_model.score(X_test,y_test))
preds_lasso=lasso_model.predict(X_test)
print("Training - RMSE is: \n", mean_squared_error(y_test,preds_lasso))
plt.scatter(preds_lasso,actual_values,alpha=.75,color="b")
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.title("Lasso with alpha = {}".format(alpha))
overlay="R squared is: {}\n RMSE is: {}".format(
lasso_model.score(X_test,y_test),
mean_squared_error(y_test,preds_lasso))
plt.annotate(s=overlay,xy=(12.1,10.6),size="x-large")
plt.show()

#D. XGBoost
regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

xgb_model= regr.fit(X_train,y_train)
print("Training - R squared is: \n", xgb_model.score(X_test,y_test))
preds_xgb=xgb_model.predict(X_test)
print("Training - RMSE is: \n", mean_squared_error(y_test,preds_xgb))
plt.scatter(preds_xgb,actual_values,alpha=.75,color="b")
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.title("XGBoost with alpha = {}".format(alpha))
overlay="R squared is: {}\n RMSE is: {}".format(
xgb_model.score(X_test,y_test),
mean_squared_error(y_test,preds_xgb))
plt.annotate(s=overlay,xy=(12.1,10.6),size="x-large")
plt.show()

pd.DataFrame(preds_xgb.flatten())
train['SalePrice']

