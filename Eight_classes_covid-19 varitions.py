
from  Utils import *
from MyModels import *
#from MyModels_TCN import *
from sklearn.model_selection import train_test_split
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense,LSTM, Flatten
from tcn import TCN
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,LSTM, Flatten
#from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers import Input,Dense, Activation, Dropout, Bidirectional,LSTM, GRU, Conv1D,
from tensorflow.keras.layers import GlobalMaxPooling1D,Input,Dense,BatchNormalization, Activation, Dropout, Bidirectional,LSTM, GRU, Conv1D

import tensorflow as tf
n_classes = 8

classNames = ['CoV-2 (B)', 'CoV-2 (B.1.1.7)', 'CoV-2 (B.1.351)', 'CoV-2 (B.1.617.2)', 'CoV-2 (C.37)', 'CoV-2 (P.1)','B.1.617.2', 'Noraml']
all_data_class1 = read_seq_new(r'D:/Covid_Project/Dataset/SARS-CoV-2 (B)/ncbi_dataset/data/genomic.fna',0)
all_data_class2 = read_seq_new(r'D:/Covid_Project/Dataset/SARS-CoV-2 (B.1.1.7)/ncbi_dataset/data/genomic.fna',1)
all_data_class3 = read_seq_new(r'D:/Covid_Project/Dataset/SARS-CoV-2 (B.1.351)/ncbi_dataset/data/genomic.fna',2)
all_data_class4 = read_seq_new(r'D:/Covid_Project/Dataset/SARS-CoV-2 (B.1.617.2)/ncbi_dataset/data/genomic.fna',3)
all_data_class5 = read_seq_new(r'D:/Covid_Project/Dataset/SARS-CoV-2 (C.37)/ncbi_dataset/data/genomic.fna',4)
all_data_class6 = read_seq_new(r'D:/Covid_Project/Dataset/SARS-CoV-2 (P.1)/ncbi_dataset/data/genomic.fna',5)
all_data_class7 = read_seq_new(r'D:/Covid_Project/Dataset/SARS-CoV-2 (B.1.617.2)/ncbi_dataset/data/genomic.fna',6)
all_data_class8 = read_seq_new(r'D:\Covid_Project\Dataset\GRCh38_latest_genomic\GRCh38_latest_genomic.fna',7)


all_data=[]
for itm in all_data_class1:
    all_data.append(itm)
for itm in all_data_class2:
    all_data.append(itm)
for itm in all_data_class3:
    all_data.append(itm)
for itm in all_data_class4:
    all_data.append(itm)
for itm in all_data_class5:
    all_data.append(itm)
for itm in all_data_class6:
    all_data.append(itm)
for itm in all_data_class7:
    all_data.append(itm)
for itm in all_data_class8:
    all_data.append(itm)
shuffle(all_data)


x=[]
y=[]
for itm in all_data:
    x.append(itm[0])
    y.append(np.array(itm[1])) 

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2)
x_train,x_Val,y_train,y_Val= train_test_split(x_train,y_train, test_size=0.1)
x_train=np.asarray(x_train,dtype=np.float)
x_test=np.asarray(x_test,dtype=np.float)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)
encoded = to_categorical([y_train])
y_train = np.squeeze(encoded)
encoded = to_categorical([y_test])
y_test = np.squeeze(encoded)



####################################################################################
####################################################################################
model = LSTMModel()
model.build(x_train.shape)
print (model.summary())
opt = Adam(lr=0.0003,amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=1,epochs=10,shuffle=False)
y_scores= model.predict(x_test)
model.save("Eight_Class_LSTM_model")

plot_Acc_Loss(history)
plot_ROC(y_test,y_scores, classNames)
plot_confusion_matrix(y_test,y_scores, n_classes)

#################################################################################################################
#################################################################################################################


input_layer = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))

model = Sequential()
# model.add(Conv1D(128,4,padding='same',activation='relu', padding='same', input_shape=(x_train.shape[1],x_train.shape[2]))
# model.add(Conv1D(64,4, padding='same',activation='relu', padding='same'))
# model.add(MaxPooling1D(4))
# model.add(Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
# model.add(Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'))
# #model.add(MaxPooling1D(pool_size=2))
model.add(GRU(64, return_sequences=False))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.4))
model.add(Dense(n_classes))
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#TRtime=datetime.now()
hist = model.fit(x_train, y_train, epochs=5,batch_size=32, verbose=1, validation_split=0.2) 
model.save('model/'+DSName+'_'+name+'.h5')
pred = model.predict(valx, verbose=0)


####################################################################################
####################################################################################

import keras
model = Sequential()
model.add(Bidirectional(GRU(units=32, return_sequences=False)))
#model.add(Bidirectional(GRU(units=16, return_sequences=True)))
# model.add(Bidirectional(GRU(units=64, return_sequences=False)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(8))
model.add(Activation('softmax'))
opt =tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print('Training is started')
print('Please wait this will take a while: ...')
history1=model.fit(x_train,y_train,batch_size =32,epochs=10, validation_data=(x_Val,y_Val), verbose=1, shuffle=False)
print(y_train.shape)


####################################################################################
####################################################################################


TCN_1=TCN(nb_filters =128, nb_stacks = 2, return_sequences = True,
          use_skip_connections = True, activation = "relu", dropout_rate = 0.2)(input_layer)

fl=Flatten()(TCN_1)
fc1=Dense(32, activation='relu')(fl)
fc1_d1=Dropout(0.3)(fc1)

fc2=Dense(8, activation='softmax')(fc1)
model = tf.keras.models.Model(inputs=input_layer, outputs=fc2)
opt = Adam(lr=0.001,amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=1,epochs=1, shuffle=False, verbose=1, validation_split=0.3)


####################################################################################
####################################################################################

Model=GRUModel()
model.build(x_train.shape)
print (model.summary())
opt = Adam(lr=0.0003,amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
history = model.fit(x_train,y_train, batch_size=16, epochs=1, shuffle=False, validation_split=0.3)
y_scores= model.predict(x_test)





####################################################################################
####################################################################################

model = LSTMModel()
model.build(x_train.shape)
print (model.summary())
opt = Adam(lr=0.0003,amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
history = model.fit(x_train,y_train,batch_size=32,epochs=1, shuffle=False, validation_split=0.3)
#history=model.fit(x_train,y_train,batch_size =32,epochs=50, validation_data=(x_Val,y_Val), verbose=1)
y_scores= model.predict(x_test)
model.save("Eight_Class_LSTM_model")

plot_Acc_Loss(history)
plot_ROC(y_test,y_train, classNames)
plot_confusion_matrix(y_test,y_scores, n_classes)



confusion = confusion_matrix(np.argmax(y_train,axis=1), np.argmax(y_test,axis=1))

target_names = ['CoV-2 (B)', 'CoV-2 (B.1.1.7)', 'CoV-2 (B.1.351)', 'CoV-2 (B.1.617.2)', 'CoV-2 (C.37)', 'CoV-2 (P.1)','B.1.617.2', 'Noraml']

print('Classification Report')
print(classification_report(np.argmax(y_test,axis=1), np.argmax(y_scores,axis=1), target_names=target_names))

####################################################################################
####################################################################################

print(history.history)
history = history
plt.plot()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(' model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.savefig('training_waseem.jpg')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(' model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.jpg')
plt.show()


model.save('8_classTCNN.h5')

####################################################################################
####################################################################################

import seaborn as sns

# df = sns.heatmap(confusion, annot=True)
# df.figure.savefig("")

con = np.zeros((n_classes,n_classes))
for x in range(n_classes):
    for y in range(n_classes):
        con[x,y] = confusion[x,y]/np.sum(confusion[x,:])
        
print(con)
# sns_plot = sns.pairplot(df, hue='species', size=2.5)

df = sns.heatmap(con, annot=True,fmt='.2%', cmap='Blues',xticklabels= target_names , yticklabels= target_names)
df.figure.savefig("CM_image2_LSTM.png")