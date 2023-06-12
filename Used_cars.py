import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import models
import time
from sklearnex import patch_sklearn
patch_sklearn()

df = pd.read_csv("true_car_listings.csv")
#%%
df_mod = df.drop(columns=['City', 'Vin'], axis=1)
df_dum = pd.get_dummies(df_mod)

X = df_dum.drop('Price', axis=1)
y = df_dum.Price.values
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

#%%
# models we will use
t0 = time.time()
xg_model = models.xgboost_reg(X_train, y_train, XGBRegressor)
print("XGBoost model finished")
print("Training time:", time.time()-t0)
# time to train = 156 sec
# time to train tune = 251
xg_model.best_iteration # 999
xg_model.best_score
#%%
t0 = time.time()
lin_model = models.lin_reg(X_train, y_train, LinearRegression)
print("Linear model finished")
print("Training time:", time.time()-t0)
# time to train = 1515 sec
#%%
t0 = time.time()
las_model = models.lasso_reg(X_train, y_train, Lasso)
print("Lasso model finished")
print("Training time:", time.time()-t0)
# time to train = 1111 sec
#%%
t0 = time.time()
rf_model = models.rf_reg(X_train, y_train, RandomForestRegressor)
print("Random Forest model finished")
print("Training time:", time.time()-t0)
# time to train = 7477 sec

#%%
mae = mean_absolute_error(y_test, rf_model.predict(X_test))
print(mae)
# xg_model = $3823
# xg_tune = $ 2930
# lin_model = $2891
# las_model = $3337
# rf_model = $2310

#cv_score = np.mean(cross_val_score(lin_model, X_train, y_train, scoring='neg_mean_absolute_error', n_jobs=-1))
#print(cv_score)
#%%
'''

#elast_model = models.elast_reg(X_train, y_train, ElasticNet)
#mods.append(elast_model)

'''
#%%
print(rf_model.score(X_train, y_train))
# xg_tune training score = 89%
# lin_model training score = 86%
# las_model training score = 82%
# rf_model training score = 98%
print(rf_model.score(X_test, y_test))
# xg_tune testing score = 88%
# lin_model testing score = 86%
# las_model testing score = 81%
# rf_model testing score = 90%
#%%
len(rf_model.estimators_) #100 trees
df.info()
#%%