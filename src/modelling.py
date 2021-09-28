import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model_hg = model.fit(X_train, y_train["FTHG"])
    model_ag = model.fit(X_train, y_train["FTAG"])
    y_pred_hg = model_hg.predict(X_test)
    y_pred_ag = model_ag.predict(X_test)

    return y_pred_hg, y_pred_ag

def forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model_hg = model.fit(X_train, y_train["FTHG"])
    model_ag = model.fit(X_train, y_train["FTAG"])
    y_pred_hg = model_hg.predict(X_test)
    y_pred_ag = model_ag.predict(X_test)

    return y_pred_hg, y_pred_ag

def gradient_model(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier()
    model_hg = model.fit(X_train, y_train["FTHG"])
    model_ag = model.fit(X_train, y_train["FTAG"])
    y_pred_hg = model_hg.predict(X_test)
    y_pred_ag = model_ag.predict(X_test)

    return y_pred_hg, y_pred_ag