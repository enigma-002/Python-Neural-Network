

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv('mnist_test.csv')
#print(data.head())

data  = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] /255.0

data_train = data[1000:m].T 
Y_train = data_train[0]
X_train = data_train[1:n] /255.0

def init_params():
    W1 = np.random.rand(10,784) - 0.5 #creates rnadom weights between -0.5 and 0.5
    b1 = np.random.rand(10,1) - 0.5 # creates random bias between -0.5 and 0.5
    W2 = np.random.rand(10,10) - 0.5 
    b2 = np.random.rand(10,1) - 0.5
    W3 = np.random.rand(10,10) - 0.5
    b3 = np.random.rand(10,1) - 0.5


    return W1,b1,W2,b2,W3,b3

def ReLU(Z):
    return np.maximum(0,Z) #returns the maximum of 0 and Z, which is the ReLU activation function

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True) #returns the softmax activation function, which is used for multi-class classification

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)  
    return Z1, A1, Z2, A2, Z3, A3


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,Y.max()+1)) #assumes that Y contains integer labels starting from 0 to max(Y) correctly sized matrix
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0 


def back_prop(Z1,A1,Z2,A2,Z3,A3,W3,W2,W1,X,Y):
    one_hot_Y = one_hot(Y)
    
    dZ3 = A3 - one_hot_Y
    dW3 = 1 /X.shape[1] * dZ3.dot(A2.T)
    db3 = 1 /X.shape[1] * np.sum(dZ3,1,keepdims=True)
    
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1 /X.shape[1] * dZ2.dot(A1.T)
    db2 = 1 /X.shape[1]  * np.sum(dZ2,1,keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 /X.shape[1] * dZ1.dot(X.T)
    db1 = 1 /X.shape[1] * np.sum(dZ1,1,keepdims=True)


    return dW1, db1, dW2, db2, dW3, db3

def update_params(dW1,db1,dW2,db2,dW3,db3,W1,b1,W2,b2,W3,b3,learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    return W1,b1,W2,b2,W3,b3




def predictions(A3):
    return np.argmax(A3,0) 

def accuracy(prediction,Y):
    print(prediction,Y)
    return np.sum(prediction == Y) / Y.size 


def gradient_descent(X,Y,iterations,alpha):
    W1,b1,W2,b2,W3,b3 = init_params()
    accuracies = []


    for i in range(iterations):
        Z1,A1,Z2,A2,Z3,A3 = forward_prop(W1,b1,W2,b2,W3,b3,X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1,A1,Z2,A2,Z3,A3,W3,W2,W1,X,Y)
        W1,b1,W2,b2,W3,b3 = update_params(dW1,db1,dW2,db2,dW3,db3,W1,b1,W2,b2,W3,b3,alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", accuracy(predictions(A3),Y))
            accuracies.append(accuracy(predictions(A3),Y))

    return W1,b1,W2,b2,W3,b3, accuracies



def show_visuals(accuracies, X, Y, W1, b1, W2, b2, W3, b3):
    
    # --- Plot 1: Training Accuracy Curve ---
    plt.figure(figsize=(10, 4))
    plt.plot(accuracies, color='steelblue')
    plt.title('Training Accuracy Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Sample Predictions Grid ---
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    preds = predictions(A3)

    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = X[:, i].reshape(28, 28)  # reshape flat pixel vector to 28x28
        ax.imshow(img, cmap='gray')
        color = 'green' if preds[i] == Y[i] else 'red'
        ax.set_title(f'P:{preds[i]} A:{int(Y[i])}', color=color, fontsize=8)
        ax.axis('off')
    
    plt.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=12)
    plt.tight_layout()
    plt.show()


W1,b1,W2,b2,W3,b3, accuracies = gradient_descent(X_train,Y_train,500,0.1)
show_visuals(accuracies, X_train, Y_train, W1, b1, W2, b2, W3, b3)
