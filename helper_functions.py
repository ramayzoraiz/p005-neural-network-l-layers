import numpy as np
import matplotlib.pyplot as plt


def layer_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = np.matmul(W,A_prev) + b
        A = 1/(1+np.exp(-Z))
    elif activation == "relu":
        Z = np.matmul(W,A_prev) + b
        A = np.maximum(0,Z)
    elif activation == "tanh":
        Z = np.matmul(W,A_prev) + b
        A = np.tanh(Z)    
    cache = ((A_prev,W,b), Z)  # linear_cache, activation_cache
    return A, cache


def layer_backward(dA, cache, activation):
    # linear_cache, activation_cache = cache
    (A_prev,W,b), Z = cache
    m = A_prev.shape[1]
    if activation == "relu":
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        dW = 1/m*np.matmul(dZ,A_prev.T)
        db = 1/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T,dZ)
    elif activation == "sigmoid":
        a = 1/(1+np.exp(-Z))
        dZ = dA * a * (1-a)
        dW = 1/m*np.matmul(dZ,A_prev.T)
        db = 1/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T,dZ)
    elif activation == "tanh":
        a = np.tanh(Z)
        dZ = dA * ( 1- np.square(a) )
        dW = 1/m*np.matmul(dZ,A_prev.T)
        db = 1/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T,dZ)
    return dA_prev, dW, db


def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def predict(X, y, parameters, activation_L_1="relu", activation_L="sigmoid"):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m),dtype=np.int8)
    
    # Forward propagation
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = layer_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = activation_L_1)
        caches.append(cache)
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = layer_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = activation_L)
    caches.append(cache)
       
    # convert probas to 0/1 predictions
    for i in range(0, AL.shape[1]):
        if AL[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))  
    return p


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
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


        