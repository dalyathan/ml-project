import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo 



def read_data():
    df = fetch_ucirepo(id=468)

    X = df.data.features 
    y = df.data.targets 
    X= prepare_data(X)
    X_std=X#(X - X.mean()) / X.std()
    pca = PCA(n_components=6)
    pca.fit(X_std)
    transformed_data = X_std#pca.transform(X_std)
    return transformed_data,y

def prepare_data(X):
    X = X.dropna()
    categorical_features=['Month', 'VisitorType']
    X = pd.get_dummies(X[categorical_features])
    return X

def split_data(X,y,test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return  X_train,X_test,y_train,y_test

def train_gaussian_naive_bias_model(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

def train_logistic_regression_model(X_train, y_train):
   model = LogisticRegression()
   model.fit(X_train, y_train)
   return model

def draw_plot(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Spam")
    plt.ylabel("Actual Spam")
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.show()

def main():
    X, y = read_data()
    X_train, X_test, y_train, y_test = split_data(X,y,0.3)
    model=train_gaussian_naive_bias_model(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("f1",  "{:.3f}".format(f1))
    cm = confusion_matrix(y_test, y_pred)
    draw_plot(cm)
    print(f1)

main()