import numpy as np
import h5py
from sklearn.model_selection import train_test_split

PATH = "D:\Downloads\MNISTdata.hdf5"
f = h5py.File(PATH, 'r')

x_test = np.array(f['x_test'])
x_train = np.array(f['x_train'])
y_test = np.array(f['y_test'])
y_train = np.array(f['y_train'])

def convolve(f,image):
    
    feature_map = np.zeros(shape = (f.shape[0],image.shape[0] - f.shape[1]+1, image.shape[1] - f.shape[2]+1))
    
    for i in range(image.shape[0] - f.shape[1]+1):
        for j in range(image.shape[1] - f.shape[2]+1):
            feature_map[:,i,j] = np.multiply(image[i:i+f.shape[1],j:j+f.shape[2]], f).sum(axis=(1,2))
            
    return feature_map


def ReLu(x):
    return np.where(x > 0, x, 0)


def grad_ReLu(x):
    return np.where(x > 0, 1, 0)


def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def forward_pass(image,f,W,b):
    Z = convolve(f,image)
    H = ReLu(Z)
    U = np.multiply(W,H).sum(axis = (1,2,3)) + b
    return(stable_softmax(U), H, Z)




def true_soft(label):
    arr = np.zeros(shape = (10,1))
    arr[label] = 1
    return arr




def get_alpha(i):
    if i <= 1:
        return 0.1
    elif i <= 10:
        return 0.01
    else:
        return 0.001

    
    

def predict(image,f,W,b):
    keep, discard1, discard2 = forward_pass(image,f,W,b)
    return np.argmax(keep)



def batch_predict(x,f,W,b):
    pred_label = np.zeros(shape = (len(x)))
    for i in range(len(x)):
        image = np.reshape(x[i], (28,28))
        pred_label[i] = predict(image,f,W,b)
    return pred_label



def cross_entropy_loss(pred,true):
    return -np.log(pred[true])



def accuracy(pred, true):
    return 100*sum(np.where(pred == np.reshape(true,len(true)), 1,0))/len(pred)

x_training, x_validate, y_training, y_validate = train_test_split(x_train, y_train, test_size = 0.10)

channels = 5
filter_x = 3
filter_y = 3

f = np.random.randn(channels, filter_x , filter_y)

W = np.random.randn(10 , channels  ,  28 - filter_x + 1 ,   28 - filter_y + 1)/np.sqrt(28 - filter_x + 1)

b = np.zeros(shape = 10)

"""
 Training Algorithm

"""
l = len(x_training)
epochs = int(input("Enter the Number of Epochs you want to train for\n"))


print("============Training Conv Net===================================================\n")

for j in range(epochs):
    alpha = get_alpha(j)
    loss = 0
    for i in range(l):
        #stochastic gradient descent
        image = np.reshape(x_training[i], (28,28))
        pred,H,Z = forward_pass(image,f,W,b)
        true = true_soft(y_training[i])

        diff_outer = -(true.transpose()[0] - pred)
        
        #print(diff_outer)
        grad_W = np.array([H * i for i in list(diff_outer)])
        grad_b = diff_outer
        delta = sum([i*j for i,j in zip(diff_outer,W)])
        grad_f = convolve(np.multiply(grad_ReLu(Z), delta),image)
        
        b = b - alpha*grad_b
        W = W - alpha*grad_W
        f = f - alpha*grad_f
        
        loss = cross_entropy_loss(pred,y_training[i])
        
    loss = loss/l
    predicted_labels_validation = batch_predict(x_validate,f,W,b)
    print("For epoch {} accuracy is: {} %, average loss : {}\n".format(j, accuracy(predicted_labels_validation,y_validate),loss[0]))
    
print("=======================FINISHED TRAINING============================================\n\n\n\n")

"""
Prediction
"""
predicted_labels_test = batch_predict(x_test,f,W,b)
print("Accuracy on Test set is {}".format(accuracy(predicted_labels_test,y_test)))