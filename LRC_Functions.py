#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with a Neural Network mindset
# * This is a Practical Programming Assignment . I'll build a Logistic Regression Classifier to Recognize Cats
# 

# # 1 - Packages
# * First, let's run the cell below to import all the packages that I need during the assignment.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt # to plot graphs
import h5py              # to interact with a dataset that is stored on a H5 file
import scipy             # to test the model at the end
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset

#get_ipython().run_line_magic('matplotlib', 'inline')


# * Loading the dataset ("data.h5")

# In[7]:


def load_dataset():
    with h5py.File(r"C:\\Users\\SherifMounir\\Desktop\\PythonScripts\\train_catvnoncat.h5", "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File(r"C:\\Users\\SherifMounir\\Desktop\\PythonScripts\\test_catvnoncat.h5", "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[8]:


# loading the data (cat/non-cat)
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()


# In[11]:


# Example of a picture
index = 7
#plt.imshow(train_set_x_orig[index])
#print("y = " + str(train_set_y_orig[:,index]) + ", it's '" +  classes[np.squeeze(train_set_y_orig[:,index])].decode("utf-8") + "' Picture")


# # 2 - Pre-Processing

# * After we load the Dataset , now we're going to keep matrix/vector dimensions straight as :
# *   m_train = (number of training examples)
# *   m_test = (number of test examples)
# *   num_px = (= height = width of a training image)

# In[15]:


m_train = train_set_x_orig.shape[0]
m_test  = test_set_x_orig.shape[0]
num_px  = train_set_x_orig.shape[1]
'''
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + "," + str(num_px) + ", 3)" )
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y_orig.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y_orig.shape))
'''

# * Now reshape images of shape (num_px , num_px , 3) in a numpy-array of shape (num_px * num_px * 3 , 1).After this , our     training(and test) dataset is a numpy-array where each column represents a flattened image .

# In[16]:


# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0] , -1)).T
test_set_x_flatten  = test_set_x_orig.reshape((test_set_x_orig.shape[0] , -1)).T
'''
print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y_orig.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y_orig.shape))
'''


# * One commom pre-processing step in machine learning is to center and standardize your dataset . for picture datasets , it's simpler just divide every row of the dataset by 255 (the maximum value of a pixel channel) .
# 

# In[17]:


train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


# # 3 - Building Our Algorithm
# 

# * Helper functions -- Sigmoid Activation Function

# In[18]:

'''
def sigmoid(z): # z = w*x + b
    s = 1/(1 + np.exp(-z))
    return s

'''
# In[20]:


#print("sigmoid([0 , 2]) = " + str(sigmoid(np.array([0,2]))))


# * Initlializing parameters

# In[21]:


def initialize_with_zeros(dim): # dim is the size of w vector we want
    w = np.zeros((dim , 1))
    b = 0
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b


# In[22]:

'''
dim = 2
w , b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))

'''
# * Forward and Backward propagation

# In[25]:


def propagate(w , b , X , Y):
    m = X.shape[1]
    # Forward propagation
    A = sigmoid(np.dot(w.T,X) + b)  # compute Activation
    logA = np.log(A)
    Y_multi_logA = np.dot(Y , logA.T)
    logA2 = np.log(1 - A)
    Y_multi_logA2 = np.dot((1 - Y) , logA2.T)
    cost = (-1/m)*(Y_multi_logA + Y_multi_logA2) # compute cost function
    ###############################################
    # Back propagation
    dz = A - Y
    dw = (1/m)*(np.dot(X , dz.T))
    db_hat = (1/m)*dz
    db = np.sum(db_hat , axis = 1 ,keepdims = True)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw":dw ,
             "db":db }
    return grads , cost
    


# In[29]:

'''
w , b , X , Y = np.array([[1.] , [2.]]) , 2. , np.array([[1. , 2. , -1.] , [3. , 4. , -3.2]]) , np.array([1 , 0 , 1])
grads , cost = propagate(w , b , X , Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))
'''

# * Optimization -- update the parameters w , b using Gradient Descent
# 

# In[36]:


def optimize(w , b , X , Y , num_iterations , learning_rate , print_cost = False):
    costs = [] # list of all costs computed during the optimization ,the will be used to plot the learning curve
    for i in range(num_iterations):
        grads , cost = propagate(w , b , X , Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost) 
        if print_cost and i % 100 == 0 :
            print("Cost after iteration %i : %f" %(i , cost))
    params = {"w":w ,
              "b":b}   
    grads = {"dw":dw ,
             "db":db}
    return params , grads , costs


# In[38]:

'''
params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate=0.009 , print_cost = False)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
'''

# * Prediction

# In[41]:

'''
def predict(w , b , X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0] , 1)
    A = sigmoid(np.dot(w.T , X) + b)
    for i in range(A.shape[1]):
        if A[0,i] <= 0.5 :
            Y_prediction[0 , i] = 0
        else:
            Y_prediction[0 , i] = 1
    assert(Y_prediction.shape == (1 , m))        
    return Y_prediction
            
'''

# In[42]:

'''
w = np.array([[0.1124579] , [0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2] , [1.2 ,2. ,0.1]])
print("predictions = " + str(predict(w , b , X)))

'''
# # 4 - Merge all functions into a Model

# In[64]:


def model(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
    # builds the logistic regression model by calling the previously implemented functions
    w , b = initialize_with_zeros(X_train.shape[0])
    parameters , grads , costs = optimize(w , b , X_train , Y_train , num_iterations = 2000 , learning_rate=0.5 , print_cost = False)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test  = predict(w , b , X_test)
    Y_prediction_train = predict(w , b , X_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
   
    d = {"costs": costs ,
         "Y_prediction_test": Y_prediction_test ,
         "Y_prediction_train": Y_prediction_train ,
         "w" : w ,
         "b" : b ,
         "learning_rate" : learning_rate ,
         "num_iterations" : num_iterations
        }
    return d


# In[65]:


#d = model(train_set_x , train_set_y_orig  , test_set_x  , test_set_y_orig , num_iterations = 2000 , learning_rate = 0.005 , print_cost = True)


# # Comment
# * Training accuracy is close 100%. This is a good sanity check: the model is working and has high enough capacity to fit the training data. Test accuracy is 72%. It's actually not bad for this simple model , given the small dataset we used and that logistic regression is a linear classifier. Also, we see that the model is clearly overfitting the training data . Using Regularization for example will reduce the overfitting. But no worries , I'll build an even better classifier in later assignment.

# In[ ]:

def sigmoid(Z):   
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache


def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     



def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y):   
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1, m),dtype=int)
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)


    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: %s" % str(np.sum(p == y)/float(m)))
        
    return p

def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


