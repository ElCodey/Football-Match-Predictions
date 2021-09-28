import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def df_concat():
    df_20_21 = pd.read_csv("E0.csv")
    df_19_20 = pd.read_csv("season_19_20.csv")
    df_18_19 = pd.read_csv("season_18_19csv.csv")
    df_18_17 = pd.read_csv("season-1718_csv.csv")

    df = pd.concat([df_18_17, df_18_19, df_19_20, df_20_21], join="inner")

    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
        "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HR", "AR"]]
        
    return df

def x_y_split(df):
    X = df.drop(["FTHG", "FTAG"], axis=1)
    y = df[["FTHG", "FTAG"]]
    X_dummy = pd.get_dummies(X)
    X = X.drop(["HomeTeam", "AwayTeam"], axis=1)

    return X, X_dummy, y

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=1)

    return X_train, X_test, y_train, y_test


def main():
    df = df_concat()
    X, X_dummy, y = x_y_split(df)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train_dum, X_test_dum, y_train_dum, y_test_dum = split(X_dummy, y)
    

    return X_train, X_test, y_train, y_test, X_train_dum, X_test_dum, y_train_dum, y_test_dum

X_train, X_test, y_train, y_test, X_train_dum, X_test_dum, y_train_dum, y_test_dum = main()
print(X_train)
