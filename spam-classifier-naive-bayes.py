import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score



def read_data():
    df = pd.read_csv('emails.csv')
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y=shuffled_df['Prediction']
    X=shuffled_df.drop(['Email No.','Prediction'], axis=1)
    return X,y

def split_data(X,y,test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return  X_train,X_test,y_train,y_test

def train_model(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

def __main__():
    X, y = read_data()
    X_train, X_test, y_train, y_test = split_data(X,y,0.3)
    print(y_test)
    model=train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f1)

__main__()