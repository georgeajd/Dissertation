import os
import datetime
import csv
import statistics as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers



#import file
df = pd.read_csv("C:/Users/George/Documents/Programs/Time Series Forcasting/sources/TSLA_daily.csv", sep=',')

#retrieve closing prices
df = df["BidClose"].pct_change().tolist() #(T-Tminus)/Tminus

#log(price(t)/price(t-1)) gaussian

#plt.plot(df)
#plt.xlabel("Days")
#plt.ylabel("Percent Change")
#plt.legend()
#plt.show()


#Calculate Centred Moving Average
for i in range(len(df)):
    if i == 0 or i==1 or i == len(df)-1:
        pass
    else:
        df[i] = ((0.8*(df[i-1]))+(1.4*(df[i]))+(0.8*(df[i+1])))/3


#Split the data
n = len(df)
train_df = df[1:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]



def create_data(input_data, input_width):
    data_list = []
    label_list = []
    input_size = len(input_data)

    for i in range(input_size - input_width):
        data_slice = input_data[i:i+input_width]
        data_list.append(data_slice)
        label_list.append(input_data[i+input_width])

    #returning output data and labels as np arrays
    return np.asarray(data_list,dtype=np.float32), np.asarray(label_list,dtype=np.float32)


#TESTDATA = [1,2,3,4,5,6,7,8,9,10]


window_size = 5

train_data, train_labels = create_data(train_df, window_size)
val_data, val_labels = create_data(val_df, window_size)
test_data, test_labels = create_data(test_df, window_size)

print(train_data)
print(train_labels)


# build
network = models.Sequential()
network.add(layers.Dense(5, activation='relu',input_shape=(window_size, )))
network.add(layers.Dense(5, activation='tanh'))
network.add(layers.Dense(1))
network.compile(optimizer='Adagrad',loss='mae',metrics=['mae'])
print(network.summary())
#adagrad

network.fit(train_data, train_labels, validation_data = (val_data, val_labels) , epochs=100, batch_size=None, verbose=0)

# Train network on 200 values total, first 100 for ~100 epochs 
#  slide window +1 to next 100 values and train for ~10/20 epochs



#TEST MODEL ----------------------------------------------------------

#test_loss, test_acc = network.evaluate(test_df, train_labels)

network_predictions = network.predict(test_data)

plt.figure(figsize=(10,5))
plt.plot(test_labels, label="Labels")
plt.plot(network_predictions, label="Prediction")
plt.title("Single layer MLP, 5 neurons, relu activation, adagrad optimisation, MAE loss")
plt.xlabel("Days")
plt.ylabel("TSLA Returns")
plt.legend()
plt.show()
