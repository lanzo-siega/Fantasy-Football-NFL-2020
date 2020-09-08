#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from etl import qb, rb, wr, te, k, dst

from sklearn import metrics
from sklearn.model_selection import train_test_split # train-test split
from sklearn.linear_model import LinearRegression #linear regression
from sklearn.ensemble import RandomForestRegressor #random forest
from sklearn.svm import SVR #support vector regression
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBRegressor

import pickle

def trainer(df):
    global X_train, X_test, y_train, y_test, lrs, y_pred
    
    if df is not dst:
        df2 = df[df.columns.difference(['NAME', 'TEAM', 'Rank','Team','Games','Avg'])]
    elif df is dst:
        df2 = df[df.columns.difference(['Rank_x', 'TEAM', 'TOP','Bye Week','Rank_y','Games'])]
    else:
        pass
        
    X = df2[df2.columns.difference(['Points'])]
    y = df2['Points']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    def lr(X_train, X_test, y_train, y_test):
        
        lrr = LinearRegression()
        lrr.fit(X_train, y_train)
        
        return lrr
    
    def rf(X_train, X_test, y_train, y_test):
        
        sc = StandardScaler()
        model = RandomForestRegressor(n_jobs=-1, random_state=20)
        estimators = np.arange(1, 200, 10)
        scores = []
        estim = []
        for n in estimators:
            model.set_params(n_estimators=n)
            model.fit(X_train, y_train) #random_state to keep stable results
            scores.append(model.score(X_test, y_test))
            estim.append(n)

        rfdf = pd.DataFrame({'Estimator':estim, 'Score':scores})
        best = next((x for x in rfdf['Estimator'][rfdf['Score'] == max(rfdf['Score'])]), None)
        
        regr = RandomForestRegressor(n_estimators=best, random_state=0)
        regr.fit(X_train, y_train)
        
        return regr
    
    def xgb(X_train, X_test, y_train, y_test):
        
        mod = XGBRegressor(learning_rate=0.2, objective='reg:squarederror')
        estimators = np.arange(1, 200, 10)
        scores = []
        estim = []

        for n in estimators:
            mod.set_params(n_estimators=n)
            mod.fit(X_train, y_train)
            scores.append(mod.score(X_test, y_test))
            estim.append(n)

        xdf = pd.DataFrame({'Estimator':estim, 'Score':scores})
        best = next((x for x in xdf['Estimator'][xdf['Score'] == max(xdf['Score'])]), None)

        xgbr = XGBRegressor(n_estimators=best, learning_rate=0.2, objective='reg:squarederror')
        xgbr.fit(X_train, y_train)
        
        return xgbr
    
    lrs = lr(X_train, X_test, y_train, y_test)
    rfs = rf(X_train, X_test, y_train, y_test)
    xgs = xgb(X_train, X_test, y_train, y_test)
    
    acc = [lrs.score(X_test, y_test), rfs.score(X_test, y_test), xgs.score(X_test, y_test)]
    model = ['Linear Regression', 'Random Forest Regression', 'XGBoost Regression']
    
    dff = pd.DataFrame({'Model': model, 'Accuracy': acc})
    
    winner = next((x for x in dff['Model'][dff['Accuracy'] == max(dff['Accuracy'])]), None)
    
    filename='final_model.sav'
    
    if winner == 'Linear Regression':
        pickle.dump(lrs, open(filename, 'wb'))
    elif winner == 'Random Forest Regression':
        pickle.dump(rfs, open(filename, 'wb'))
    elif winner == 'XGBoost Regression':
        pickle.dump(xgs, open(filename, 'wb'))
    else:
        pass

def preddy(df):
    trainer(df)
    
    df3 = df.copy(deep=True)
    
    if df is not dst:
        df2 = df3[df3.columns.difference(['NAME', 'TEAM', 'Rank','Team','Games','Avg'])]
    elif df is dst:
        df2 = df3[df3.columns.difference(['Rank_x', 'TEAM', 'TOP','Bye Week','Rank_y','Games'])]
    else:
        pass
        
    X = df2[df2.columns.difference(['Points'])]
    y = df2['Points']
    
    load_mod = pickle.load(open('final_model.sav', 'rb'))
    
    result = load_mod.predict(X)
    
    df3['Predicted'] = result
    
    return df3

