
import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

#build model
#compile
#train
#evaluate
#make predictions
def generate_dataset(num_samples, test_size):
    x= np.array([[random() for _ in range(2)] for _ in range(num_samples)])
    y= np.array([[i[0]+i[1]] for i in x])
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size)
    return x_train,x_test,y_train,y_test

#keras makes tf easy
if __name__=="__main__":
    x_train,x_test,y_train,y_test = generate_dataset(5000,0.3)
    #print("x_train: ",x_train)
    #build model 2->5 ->1 sequential network- left to right 
    model =  tf.keras.Sequential([
        tf.keras.layers.Dense(5,activation='sigmoid',input_dim=2),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
#compile - sgd- stochastic gradient descent
    optimiser =tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimiser,loss='MSE')

    #train
    model.fit(x_train,y_train,epochs=100)
    print("/n Model evaluation: ")
    model.evaluate(x_test,y_test,verbose =1)

    #make predictions
    data =np.array([[0.1,0.2],[0.2,0.2]])
    predictions =model.predict(data)
    print("Predictions: ")
    for d,p in zip(data,predictions):
        print("Input: ",d,"Prediction: ",p)