# Fashion MNIST Classifier

This project implements machine learning models to classify images from the **Fashion MNIST** dataset. 
The dataset consists of **70,000 grayscale images** (28×28 pixels) across **10 clothing categories**, making it a useful benchmark for classification tasks.

## Implemented Models:
- **Gaussian Naïve Bayes**
- **Logistic Regression**
- **Support Vector Classifier**

## Project Structure:
```
📂 fashion-mnist-classifier/  
├── 📄 fashion-mnist-classifier.py 
├── 📄 fashion-mnist_train.csv      
├── 📄 fashion-mnist_test.csv       
├── 📂 confusion_matrix/            
│   ├── ...
├── 📂 training_results/           
│   ├── ...
```

The results are stored in dedicated folders for confusion matrices and training performance.

## Getting Started
1. Extract the training Fasion MNIST training dataset
```cmd
tar --use-compress-program='zstd' -xvf fashion-mnist_train.tar.zst
```
2. Extract the testing Fasion MNIST training dataset
```cmd
tar --use-compress-program='zstd' -xvf fashion-mnist_test.tar.zst
```
3. Extract the confusion matrices folder by running
```cmd
tar --use-compress-program='zstd' -xvf confusion-matrices.tar.zst
```
4. Run any of the functions `run_gnn()`, `run_svc()`, `run_lr()` respectively for running `Gaussian Naive Biase`, `Support Vector Classifier`, and `Logistic Regression Model` models found inside `./fashion-mnist-classifier.py`.
