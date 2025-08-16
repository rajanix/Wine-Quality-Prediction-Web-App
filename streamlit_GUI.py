import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import math

def split_and_normalise(df):
  X,y = df.drop("quality",axis = 1).values,df["quality"].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
  X_train, X_val, y_train,y_val = train_test_split(X_train,y_train,test_size=0.1/0.9,random_state=42)
  # scaler = MinMaxScaler() # not used
  # scaler = StandardScaler()
  # X_train_norm = scaler.fit_transform(X_train)
  # X_val_norm = scaler.fit_transform(X_val)
  # X_test_norm = scaler.transform(X_test)
  # return scaler,X_train_norm,y_train,X_val_norm,y_val,X_test_norm,y_test
  return X_train,y_train,X_val,y_val,X_test,y_test

def random_forest_model(X_train,y_train,X_val,y_val,X_test,y_test):
  n_estimators = [100,110,120]
  mse_loss = np.inf
  best_estimator = None
  for n in n_estimators:
    rf = RandomForestRegressor(n_estimators = n)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_val)
    y_pred = np.round(y_pred).astype(int)
    mse = mean_squared_error(y_val,y_pred)
    if mse < mse_loss:
      mse_loss =  mse
      best_estimator = rf
  y_pred = best_estimator.predict(X_test)
  y_pred = np.round(y_pred).astype(int)
  mse = mean_squared_error(y_test,y_pred)
#   print(f"mean squared test error : {mse}")
  return best_estimator

red_df = pd.read_csv("wine+quality/winequality-red.csv",delimiter=";")
white_df = pd.read_csv("wine+quality/winequality-white.csv",delimiter=";")

df = {"Red":red_df,"White":white_df}

# print(red_df.describe())
# print(white_df.describe())

X_train,y_train,X_val,y_val,X_test,y_test = split_and_normalise(white_df)
RX_train,Ry_train,RX_val,Ry_val,RX_test,Ry_test = split_and_normalise(red_df)


rf_model = random_forest_model(X_train,y_train,X_val,y_val,X_test,y_test)
Rrf_model = random_forest_model(RX_train,Ry_train,RX_val,Ry_val,RX_test,Ry_test)


model = {"Red":Rrf_model,"White":rf_model}
st.title('Wine Quality')

option = st.radio("Wine Colour:",("Red","White"))
cols = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

value = {}
for col in cols:
    mini = float(math.floor(df[option][col].min()))
    maxi = float(math.ceil(df[option][col].max()))
    value[col] = st.slider(col,min_value = mini,max_value = maxi,value=mini,step= (maxi-mini)/100)

if st.button("Test"):

    test_df = pd.DataFrame(value,index=[0])
    test = test_df.values 
    y_pred = model[option].predict(test)
    y_pred = np.round(y_pred).astype("int")
    st.write(f"Quality of Wine is {y_pred[0]}")
# else :
#     value1 = st.slider("fixed acidity",min_value=,max_value =,value=,step=)
#     value2 = st.slider("volatile acidity",min_value=,max_value =,value=,step=)
#     value3 = st.slider("citric acid",min_value=,max_value =,value=,step=)
#     value4 = st.slider("residual sugar",min_value=,max_value =,value=,step=)
#     value5 = st.slider("chlorides",min_value=,max_value =,value=,step=)
#     value6 = st.slider("free sulfur dioxide",min_value=,max_value =,value=,step=)
#     value7 = st.slider("density",min_value=,max_value =,value=,step=)
#     value8 = st.slider("pH",min_value=,max_value =,value=,step=)
#     value9 = st.slider("sulphates",min_value=,max_value =,value=,step=)
#     value10 = st.slider("alcohol",min_value=,max_value =,value=,step=)


