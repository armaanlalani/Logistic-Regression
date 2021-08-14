import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def loadData():
    with np.load('notMNIST.npz') as dataset:
        Data, Target = dataset['images'], dataset['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss(W, b, x, y, reg):
    y_hat = sigmoid(np.matmul(W.T,x)+b)
    log_1 = np.log(y_hat+0.00001)
    log_2 = np.log(1-(y_hat-0.00001))
    loss = -np.dot(log_1,y) - np.dot(log_2,1-y)
    loss = loss/y_hat.shape[1] + reg/2*(np.linalg.norm(W)**2)
    return loss[0,0]

def grad_loss(W, b, x, y, reg):
    y_hat = sigmoid(np.matmul(W.T,x)+b)
    grad = np.matmul(x,y_hat.T-y)
    b_grad = np.sum(y_hat.T-y)/x.shape[1]
    grad = grad/x.shape[0] + reg*W
    return grad, b_grad

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, x_valid, y_valid):
    loss_array = []
    accuracy_array = []
    loss_array_valid = []
    accuracy_array_valid = []
    for i in range(0,epochs):
        prediction = sigmoid(np.matmul(W.T,x))
        loss_value = loss(W, b, x, y, reg)
        grad_loss_value, b_grad_value = grad_loss(W, b, x, y, reg)
        accuracy_value = accuracy((np.matmul(W.T,x)+b)[0],y,b)
        if np.linalg.norm(W-(W-grad_loss_value*alpha)) < error_tol:
            break
        W = W - grad_loss_value * alpha
        b = b - b_grad_value * alpha
        loss_array_valid.append(loss(W,b,x_valid,y_valid,reg))
        accuracy_array_valid.append(accuracy((np.matmul(W.T,x_valid)+b)[0],y_valid,b))
        loss_array.append(loss_value)
        accuracy_array.append(accuracy_value)
    plt.plot(loss_array, label='Training')
    plt.plot(loss_array_valid, label='Validation')
    plt.legend()
    plt.title('Loss vs Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(accuracy_array, label='Training')
    plt.plot(accuracy_array_valid, label='Validation')
    plt.legend()
    plt.title('Accuracy vs Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    return W, b

def accuracy(prediction, label, b):
    prediction = np.ndarray.tolist(prediction)
    label = np.ndarray.tolist(label)
    correct = 0
    for i in range(0,len(prediction)):
        if prediction[i] > b and label[i][0] == 1:
            correct += 1
        if prediction[i] < b and label[i][0] == 0:
            correct += 1
    return correct/i

def buildGraph(alpha, reg, beta1=0.9, beta2=0.999, epsilon=1e-8):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    tf.compat.v1.set_random_seed(0)
    W = tf.Variable(tf.random.truncated_normal(shape=(1,trainData.shape[1]*trainData.shape[2]),mean=0,stddev=0.5,dtype=tf.float32))
    b = tf.Variable(tf.zeros(1))
    
    x = tf.compat.v1.placeholder(tf.float32,shape=(None,trainData.shape[1]*trainData.shape[2]))
    y = tf.compat.v1.placeholder(tf.float32,shape=(None,trainTarget.shape[1]))
    reg = tf.constant(reg,tf.float32)

    prediction = tf.matmul(x,tf.transpose(W)) + b
    regularization = tf.nn.l2_loss(W)
    
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction)) + reg*regularization
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    optimizer = optimizer.minimize(loss=loss)

    return W, b, prediction, x, y, loss, optimizer

def training(epochs, X, Y, X_valid, Y_valid, batch_size, alpha, reg):
    W, b, prediction, x, y, loss_t, optimizer = buildGraph(alpha, reg)
    variables = tf.compat.v1.global_variables_initializer()

    X = X.reshape(np.shape(X)[0],-1)
    X_valid = X_valid.reshape(np.shape(X_valid)[0],-1)
    sess = tf.compat.v1.InteractiveSession()
    sess.run(variables)

    loss_array = []
    accuracy_array = []
    valid_accuracy_array = []
    valid_loss_array = []
    for epoch in range(0,epochs):
        instance = np.shape(X)[0]
        batches = int(instance/batch_size)
        idx = np.random.permutation(X.shape[0])
        X_shuffle = X[idx]
        Y_shuffle = Y[idx]
        num = 0
        accuracy_value = 0
        loss_value = 0
        for batch in range(0,batches):
            X_batch = X_shuffle[num:(num+batch_size),:]
            Y_batch = Y_shuffle[num:(num+batch_size),:]
            _, loss_val, W_new, b_new, y_hat = sess.run([optimizer,loss_t,W,b,prediction],feed_dict={x:X_batch,y:Y_batch})
            num += batch_size
            accuracy_value += accuracy(y_hat,Y_batch,b_new[0])
            loss_value += loss_val
        prediction_valid = np.matmul(W_new,X_valid.T) + b_new[0]
        valid_accuracy_array.append(accuracy(prediction_valid[0],Y_valid,b_new[0]))
        valid_loss_array.append(loss(W_new.T,b_new[0],X_valid.T,Y_valid,reg))
        loss_array.append(loss_value/Y.shape[0]/(batch+1))
        accuracy_array.append(accuracy_value/(batch+1))
    plt.plot(loss_array, label='Training')
    plt.plot(valid_loss_array, label='Validation')
    plt.legend()
    plt.title('Loss vs Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(accuracy_array, label='Training')
    plt.plot(valid_accuracy_array, label='Validation')
    plt.legend()
    plt.title('Accuracy vs Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    return W_new, b_new


if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W = np.random.normal(0,1,(trainData.shape[1]*trainData.shape[2],1))
    tensorflow = True

    if not tensorflow:
        trainData = (trainData.reshape(trainData.shape[0],trainData.shape[1]*trainData.shape[2])).T
        validData = (validData.reshape(validData.shape[0],validData.shape[1]*validData.shape[2])).T
        testData = (testData.reshape(testData.shape[0],testData.shape[1]*testData.shape[2])).T
        print("X shape: " + str(trainData.shape))
        print("y shape: " + str(trainTarget.shape))
        print("W shape: " + str(W.shape))

        reg = 0.5
        W, b = grad_descent(W, 0, trainData, trainTarget, 0.005, 5000, reg, 10e-7, validData, validTarget)
        test_acc = accuracy((np.matmul(W.T,testData)+b)[0],testTarget,b)
        test_loss = loss(W,b,testData,testTarget,reg)
        print('Test Accuracy: ' + str(test_acc))
        print('Test Loss: ' + str(test_loss))

    elif tensorflow:
        reg = 0
        W, b = training(700,trainData,trainTarget,validData,validTarget,500,0.001,reg)
        testData = testData.reshape(np.shape(testData)[0],-1)
        test_acc = accuracy((np.matmul(W,testData.T)+b)[0],testTarget,b[0])
        test_loss = loss(W.T,b,testData.T,testTarget,reg)
        print('Test Accuracy: ' + str(test_acc))
        print('Test Loss: ' + str(test_loss))