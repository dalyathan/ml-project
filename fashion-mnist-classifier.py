import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def train_test_split():
    X_train_raw, X_test_raw, y_train, y_test = read_data2()
    X_train_raw= preprocess_data(X_train_raw)
    X_test_raw= preprocess_data(X_test_raw)
    return X_train_raw,X_test_raw,y_train, y_test

def read_data2():
    fashion_mnist_test = pd.read_csv('./fashion-mnist_test.csv')
    fashion_mnist_train = pd.read_csv('./fashion-mnist_train.csv')
    y_test = fashion_mnist_test['label'].values;
    X_test = fashion_mnist_test.drop(columns=['label']).values;
    y_train = fashion_mnist_train['label'].values;
    X_train = fashion_mnist_train.drop(columns=['label']).values;
    return X_train, X_test, y_train, y_test

def preprocess_data(X):
    return (X - X.mean()) / X.std()

def show_sample_images(X):
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        ax[i // 5, i % 5].imshow(X[i].reshape(28, 28), cmap='gray')
        ax[i // 5, i % 5].axis('off')
    plt.show()

def apply_pca(X, n_components):
    pca = PCA(n_components)
    return pca.fit_transform(X)

def train_gaussian_naive_bias_model(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

def train_logistic_regression_model(X_train, y_train, penalty, C, class_weight, solver, l1_ratio=None):
    model = LogisticRegression(C=C, class_weight=class_weight, solver=solver, penalty=penalty, l1_ratio=0.5)
    model.fit(X_train, y_train)
    return model

def enumerate_lr_models(X_train, y_train, C):
    model1= train_logistic_regression_model(X_train, y_train, 'l2', C, 'balanced', 'lbfgs')
    model2= train_logistic_regression_model(X_train, y_train, 'l2', C, None, 'lbfgs')

    model3= train_logistic_regression_model(X_train, y_train, 'l2', C, 'balanced', 'newton-cg')
    model4= train_logistic_regression_model(X_train, y_train, 'l2', C, None, 'newton-cg')

    model5= train_logistic_regression_model(X_train, y_train, 'l2', C, 'balanced', 'sag')
    model6= train_logistic_regression_model(X_train, y_train, 'l2', C, None, 'sag')

    model7= train_logistic_regression_model(X_train, y_train, 'l2', C, 'balanced', 'saga')
    model8= train_logistic_regression_model(X_train, y_train, 'l2', C, None, 'saga')

    model9= train_logistic_regression_model(X_train, y_train, 'elasticnet', C, 'balanced', 'saga', 0.5)
    model10= train_logistic_regression_model(X_train, y_train, 'elasticnet', C, None, 'saga', 0.5)

    model11= train_logistic_regression_model(X_train, y_train, 'l1', C, 'balanced', 'saga')
    model12= train_logistic_regression_model(X_train, y_train, 'l1', C, None, 'saga')

    return ['logistic_regression_l2_balanced_lbfgs', 'logistic_regression_l2_lbfgs', 'logistic_regression_l2_balanced_newton-cg', 'logistic_regression_l2_newton-cg', 'logistic_regression_l2_balanced_sag', 'logistic_regression_l2_sag', 'logistic_regression_l2_balanced_saga', 'logistic_regression_l2_saga', 'logistic_regression_elasticnet_balanced_saga', 'logistic_regression_elasticnet_saga', 'logistic_regression_l1_balanced_sag', 'logistic_regression_l1_sag'],[model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]

def train_svc(X_train, y_train, kernel, C, degree, probability, gamma):
    model = svm.SVC(kernel=kernel, C=C, degree=degree, probability=probability, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def train_KNN(X_train, y_train, neighbours):
    model = KNeighborsClassifier(neighbours=neighbours)
    model.fit(X_train, y_train)
    return model

def save_cm(model_type,model_name,cm):
    # plt.figure(figsize=(940, 611))
    plt.figure(figsize=(9.2, 6))
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {model_name}", )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.savefig(f"./confusion matrix/{model_type}/{model_name}.png")
    #plt.show()

def run_knn():
    X_train_raw, X_test_raw, y_train, y_test = train_test_split()
    for neighbours in range(2,20):
        for n_components in range(10, 550, 10):
            model_name= f'knn_{neighbours}_{n_components}'
            X_train= apply_pca(X_train_raw, n_components)
            X_test= apply_pca(X_test_raw, n_components)
            model= train_KNN(X_train, y_train, neighbours)
            y_pred = model.predict(X_test)
            pred= model.predict(X_train)
            f1_train = round(f1_score(y_train, pred, average='macro'),3)
            accuracy_train= round(accuracy_score(y_train, pred),3)
            f1_test = round(f1_score(y_test, y_pred, average='macro'),3)
            accuracy_test= round(accuracy_score(y_test, y_pred),3)
            cm = confusion_matrix(y_test, y_pred)
            difference=round(accuracy_train-accuracy_test,3)
            save_cm('knn',model_name, cm)
            with open('./training_results/knn/knn_ex_2.csv', mode='a', newline='', encoding='utf-8') as file:
                file.write('\n')
                data=[neighbours,n_components,accuracy_train,f1_train,accuracy_test,f1_test,difference]
                line = ','.join(map(str, data))
                file.write(line)

def run_lr():
    X_train_raw, X_test_raw, y_train, y_test = train_test_split()
    for C in range(1,30,5):
        C=C/10
        for n_components in range(10, 550, 50):
            X_train= apply_pca(X_train_raw, n_components)
            X_test= apply_pca(X_test_raw, n_components)
            model_names, models=enumerate_lr_models(X_train, y_train, C)
            for model_index in range(len(models)):
                model= models[model_index]
                model_name= model_names[model_index]
                y_pred = model.predict(X_test)
                pred= model.predict(X_train)
                f1_train = round(f1_score(y_train, pred, average='macro'),3)
                accuracy_train= round(accuracy_score(y_train, pred),3)
                f1_test = round(f1_score(y_test, y_pred, average='macro'),3)
                accuracy_test= round(accuracy_score(y_test, y_pred),3)
                cm = confusion_matrix(y_test, y_pred)
                difference=round(accuracy_train-accuracy_test,3)
                save_cm('logistic_regression',model_name, cm)
                with open('./training_results/logistic_regression/logistic_regression_ex_2.csv', mode='a', newline='', encoding='utf-8') as file:
                    file.write('\n')
                    data=[C,n_components,model.get_params()['penalty'],model.get_params()['class_weight'],model.get_params()['solver'],accuracy_train,f1_train,accuracy_test,f1_test,difference]
                    line = ','.join(map(str, data))
                    file.write(line)

def run_gnn():
    X_train_raw, X_test_raw, y_train, y_test = train_test_split()
    for n_components in range(10, 550, 10):
        model_name= f'gnn_{n_components}'
        X_train= apply_pca(X_train_raw, n_components)
        X_test= apply_pca(X_test_raw, n_components)
        model= train_gaussian_naive_bias_model(X_train, y_train)
        y_pred = model.predict(X_test)
        pred= model.predict(X_train)
        f1_train = round(f1_score(y_train, pred, average='macro'),3)
        accuracy_train= round(accuracy_score(y_train, pred),3)
        f1_test = round(f1_score(y_test, y_pred, average='macro'),3)
        accuracy_test= round(accuracy_score(y_test, y_pred),3)
        cm = confusion_matrix(y_test, y_pred)
        difference=round(accuracy_train-accuracy_test,3)
        save_cm('gnn',model_name, cm)
        with open('./training_results/gnn/gnn_ex_2.csv', mode='a', newline='', encoding='utf-8') as file:
            file.write('\n')
            data=[n_components,accuracy_train,f1_train,accuracy_test,f1_test,difference]
            line = ','.join(map(str, data))
            file.write(line)

def run_svc():
    kernel='rbf';degree=3;probability=False
    X_train_raw, X_test_raw, y_train, y_test = train_test_split()
    for c in range(5,100,20):
        for n_components in range(100, 550, 50):
            X_train= apply_pca(X_train_raw, n_components)
            X_test= apply_pca(X_test_raw, n_components)
            for gamma in ['auto', 'scale']:
                model= train_svc(X_train, y_train, kernel, c, degree, probability, gamma)
                if kernel == 'poly':
                    model_name= f'svc_{kernel}_{degree}_{C}_{n_components}_{gamma}'
                else:
                    model_name= f'svc_{kernel}__{c}_{n_components}_{gamma}'
                if probability:
                    model_name+='_with_probablity'
                print(model_name)
                y_pred = model.predict(X_test)
                pred= model.predict(X_train)
                f1_train = round(f1_score(y_train, pred, average='macro'),3)
                accuracy_train= round(accuracy_score(y_train, pred),3)
                f1_test = round(f1_score(y_test, y_pred, average='macro'),3)
                accuracy_test= round(accuracy_score(y_test, y_pred),3)
                print("f1_train",  "{:.3f}".format(f1_train))
                print("accuracy_train",  "{:.3f}".format(accuracy_train))
                print("f1_test",  "{:.3f}".format(f1_test))
                print("accuracy_test",  "{:.3f}".format(accuracy_test))
                cm = confusion_matrix(y_test, y_pred)
                difference=round(accuracy_train-accuracy_test,3)
                save_cm('svc',model_name, cm)
                with open('./training_results/svc/svc_rbf_ex_10.csv', mode='a', newline='', encoding='utf-8') as file:
                    file.write('\n')
                    data=[c,n_components,gamma,probability,accuracy_train,f1_train,accuracy_test,f1_test,difference]
                    line = ','.join(map(str, data))
                    file.write(line)