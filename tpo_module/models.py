#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
# import 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Input, concatenate, Flatten, 
                                     Activation, Dropout, Bidirectional)
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_inputV2
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


def create_mlp(dim, optimizer, loss, metrics):
    # define our MLP network
    model = Sequential()
    model.add(Dense(128, input_dim=dim, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # return our model
    return model


# In[3]:


def create_lstm(dim1, dim2, optimizer, loss, metrics):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, input_shape=(dim1,dim2), recurrent_activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# In[4]:


# create resnet50 model
def create_resnet50(height, width, depth, optimizer, loss, metrics):
    # Input layer
    input_layer=Input(shape=(height,width,depth))
    # set trainable layers in ResNet50
    resnet=ResNet50(include_top=False, input_shape=(height,width,depth), weights='imagenet', pooling='avg')
    output=resnet.layers[-1].output
    output=Flatten()(output)
    resnet=Model(resnet.input, output)
    resnet.trainable=True
    for layer in resnet.layers[:144]:
        layer.trainable=False
    layers = [(layer, layer.name, layer.trainable) for layer in resnet.layers]
   
    model = resnet(input_layer)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(input_layer, model)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# In[5]:


# create resnet50v2 model
def create_resnet50v2(height, width, depth, optimizer, loss, metrics):
    # Input layer
    input_layer=Input(shape=(height,width,depth))
    # set trainbale layers in ResNet50v2
    resnet = ResNet50V2(include_top=False, input_shape=(height,width,depth), weights='imagenet', pooling='avg')
    output = resnet.layers[-1].output
    output = Flatten()(output)
    resnet = Model(resnet.input, output)
    resnet.trainable = True
    for layer in resnet.layers[:144]:
        layer.trainable=False
    layers = [(layer, layer.name, layer.trainable) for layer in resnet.layers]
    
    model = resnet(input_layer)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(input_layer, model)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# In[6]:


# create mlp&resnet50 model
def create_mlpresnet50(dim, height, width, depth, optimizer, loss, metrics):
    #A model
    modelA = Sequential()
    modelA.add(Dense(128, input_dim=dim, activation="relu"))
    modelA.add(Dropout(0.2))
    modelA.add(Dense(1, activation="sigmoid"))
    
    # B model
    # Input layer
    input_layer=Input(shape=(height,width,depth))
    # set trainable layers in ResNet50
    resnet=ResNet50(include_top=False, input_shape=(height,width,depth), weights='imagenet', pooling='avg')
    output=resnet.layers[-1].output
    output=Flatten()(output)
    resnet=Model(resnet.input, output)
    resnet.trainable=True
    for layer in resnet.layers[:144]:
        layer.trainable=False
    layers = [(layer, layer.name, layer.trainable) for layer in resnet.layers]
   
    modelB = resnet(input_layer)
    modelB = Dense(1, activation='sigmoid')(modelB)
    modelB = Model(input_layer, modelB)
    
    #concatenate two models
    combined = concatenate([modelA.output, modelB.output])
    model_output = Dense(2, activation='relu')(combined)
    model_output = Dense(1, activation='sigmoid')(model_output)
    model = Model([modelA.input,modelB.input], model_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# In[7]:


# create lstm&resnet50 model
def create_lstmresnet50(batch, dim, height, width, depth, optimizer, loss, metrics):
    #A model
    modelA = Sequential()
    modelA.add(Input(shape=(batch, dim)))
    modelA.add(Bidirectional(LSTM(128, recurrent_activation='relu')))
    modelA.add(Dropout(0.2))
    modelA.add(Dense(1, activation="sigmoid"))
    # B model
    # ResNet Input layer
    input_layer=Input(shape=(height,width,depth))
    # set trainable layers in ResNet50
    resnet=ResNet50(include_top=False, input_shape=(height,width,depth), weights='imagenet', pooling='avg')
    output=resnet.layers[-1].output
    output=Flatten()(output)
    resnet=Model(resnet.input, output)
    resnet.trainable=True
    for layer in resnet.layers[:144]:
        layer.trainable=False
    layers = [(layer, layer.name, layer.trainable) for layer in resnet.layers]
   
    modelB = resnet(input_layer)
    modelB = Dense(1, activation='sigmoid')(modelB)
    modelB = Model(input_layer, modelB)
    
    #concatenate two models
    combined = concatenate([modelA.output, modelB.output])
    model_output = Dense(2, activation='relu')(combined)
    model_output = Dense(1, activation='sigmoid')(model_output)
    model = Model([modelA.input, modelB.input], model_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# In[8]:


# create mlp&resnet50v2 model
def create_mlpresnet50v2(dim, height, width, depth, optimizer, loss, metrics):
    #A model
    modelA = Sequential()
    modelA.add(Dense(128, input_dim=dim, activation="relu"))
    modelA.add(Dropout(0.2))
    modelA.add(Dense(1, activation="sigmoid"))
    
    #B model
    # Input layer
    input_layer=Input(shape=(height,width,depth))
    # set trainable layers in ResNet50v2
    resnet=ResNet50V2(include_top=False, input_shape=(height,width,depth), weights='imagenet', pooling='avg')
    output=resnet.layers[-1].output
    output=Flatten()(output)
    resnet=Model(resnet.input, output)
    resnet.trainable=True
    for layer in resnet.layers[:144]:
        layer.trainable=False
    layers = [(layer, layer.name, layer.trainable) for layer in resnet.layers]
   
    modelB = resnet(input_layer)
    modelB = Dense(1, activation='sigmoid')(modelB)
    modelB = Model(input_layer, modelB)
    
    #concatenate two models
    combined = concatenate([modelA.output, modelB.output])
    model_output = Dense(2, activation='relu')(combined)
    model_output = Dense(1, activation='sigmoid')(model_output)
    model = Model([modelA.input, modelB.input], model_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# In[9]:


# create lstm&resnet50v2 model
def create_lstmresnet50v2(batch, dim, height, width, depth, optimizer, loss, metrics):
    #A model
    modelA = Sequential()
    modelA.add(Input(shape=(batch, dim)))
    modelA.add(Bidirectional(LSTM(128, recurrent_activation='relu')))
    modelA.add(Dropout(0.2))
    modelA.add(Dense(1, activation="sigmoid"))
    #B model
    # ResNet Input layer
    input_layer=Input(shape=(height,width,depth))
    # set trainable layers in ResNet50v2
    resnet=ResNet50V2(include_top=False, input_shape=(height,width,depth), weights='imagenet', pooling='avg')
    output=resnet.layers[-1].output
    output=Flatten()(output)
    resnet=Model(resnet.input, output)
    resnet.trainable=True
    for layer in resnet.layers[:144]:
        layer.trainable=False
    layers = [(layer, layer.name, layer.trainable) for layer in resnet.layers]
   
    modelB = resnet(input_layer)
    modelB = Dense(1, activation='sigmoid')(modelB)
    modelB = Model(input_layer, modelB)
    
    #concatenate two models
    combined = concatenate([modelA.output, modelB.output])
    model_output = Dense(4, activation='relu')(combined)
    model_output = Dense(1, activation='sigmoid')(model_output)
    model = Model([modelA.input, modelB.input], model_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# In[10]:


# # train model with cross validation fold
# def train_model_with_cv(csv_data, image_data, model, cross_validation, epochs, batch_size):
#     print('Train Model')
#     print("==================================================")
#     # split data by 9:1
#     print("Divided data into a training set and a test set in a ratio of 9:1 randomly")
#     split = train_test_split(csv_data, image_data, test_size=0.1, random_state=42)
#     (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
#     testY = testAttrX["TPO"]
#     testAttrX=testAttrX.drop("TPO", axis=1).drop("ID", axis=1)
#     print("==================================================")
#     # cross validation data
#     print('Cross Validation')
#     kfold = StratifiedKFold(n_splits = cross_validation, shuffle = False)
#     cvscores = []
#     iteration = 1
#     t = trainAttrX.TPO
#     for train_index, valid_index in kfold.split(np.zeros(len(t)),t):
#         print('==================================================')
#         print("Iteration=", iteration)
#         iteration+=1
        
#         # prepare the data
#         trainY=trainAttrX["TPO"].reindex(index=train_index)
#         validY=trainAttrX["TPO"].reindex(index=valid_index)
#         trainAX=trainAttrX.reindex(index=train_index).drop("TPO", axis=1).drop("ID", axis=1)
#         trainBX=trainImagesX[train_index]
#         validAX=trainAttrX.reindex(index=valid_index).drop("TPO", axis=1).drop("ID", axis=1)
#         validBX=trainImagesX[valid_index]
        
#         print('==================================================')
        
#         # create model
#         print('Create Model')
#         model=model
        
#         # set an earlystop to avoid overfitting
#         earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
#         # fit the model
#         fit_stats = model.fit(x=[trainAX, trainBX], y=trainY, batch_size=batch_size, epochs=epochs, 
#                               verbose='auto', callbacks=earlystop, 
#                               validation_data=([validAX,validBX],validY), shuffle=True)
        
#         # print validation accuracy score 
#         acc=fit_stats.history['accuracy']
#         los=fit_stats.history['loss']
#         val_acc=fit_stats.history['val_accuracy']
#         val_loss=fit_stats.history['val_loss']
#         epoch=range(len(acc))
#         cvscores.append(np.mean(val_acc))
#         print("Accuracy: %.2f%%" % (np.mean(val_acc)*100))
        
#         # display learning curve
#         plt.plot(epoch, los, 'r', label='Training Loss')
#         plt.plot(epoch, val_loss, 'b', label='Validation Loss')
#         plt.title('Model Loss')
#         plt.legend()
#         plt.show()
#         plt.plot(epoch, acc, 'r', label='Training Accuracy')
#         plt.plot(epoch, val_acc, 'b', label='Validation Accuracy')
#         plt.title('Model Accuracy')
#         plt.legend()
#         plt.show()
        
#     # show the training results    
#     accuracy = np.mean(cvscores)
#     std=np.std(cvscores)
#     print('CV_Accuracy:%.2f%%(+/- %.2f%%)'%(accuracy*100,std*100))
#     model.save('best_cv_multi_model.h5')
    
#     # show evaluate score
#     test_score=model.evaluate(x=[testAttrX, testImagesX], y=testY, batch_size=batch_size, verbose=1)
#     print("%s%s: %.2f%%\n" % ("evaluate ",model.metrics_names[1], test_score[1]*100),
#           "%s%s: %.2f%%" % ("evaluate ",model.metrics_names[0], test_score[0]*100))
#     return accuracy, std


# In[11]:


# train single input model with cross validation fold
def train_model_with_cv(x, x_test, y, y_test, model, cross_validation, epochs, batch_size):
    print('Train Model')
    print("==================================================")
    # create model
    print('Create Model')
    model=model
    print("==================================================")    
    # set an earlystop to avoid overfitting
    earlystop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    print('Cross Validation')
    # fit the model
    cv=list(range(1, cross_validation+1)) #set number of cross validation
    iteration=1
    cvscores=[]
    cvloss=[]
    for i in cv:
        print('==================================================')
        print("Iteration=", iteration)
        iteration+=1
        fit_stats = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, 
                            verbose='auto', callbacks=earlystop, validation_split=0.2, shuffle=True)
        
        # print validation accuracy score 
        acc=fit_stats.history['accuracy']
        los=fit_stats.history['loss']
        val_acc=fit_stats.history['val_accuracy']
        val_loss=fit_stats.history['val_loss']
        epoch=range(len(acc))
        cvscores.append(np.mean(val_acc))
        cvloss.append(np.mean(val_loss))
        print("Val_Accuracy: %.2f%%\n" % (np.mean(val_acc)*100),
              "Val_Loss: %.2f%%\n" % (np.mean(val_loss)*100))
        # display learning curve
        plt.plot(epoch, los, 'r', label='Training Loss')
        plt.plot(epoch, val_loss, 'b', label='Validation Loss')
        plt.title('Model Loss(iteration={})'.format(i))
        plt.legend()
        plt.show()
        plt.plot(epoch, acc, 'r', label='Training Accuracy')
        plt.plot(epoch, val_acc, 'b', label='Validation Accuracy')
        plt.title('Model Accuracy(iteration={})'.format(i))
        plt.legend()
        plt.show()
        
    # show the training results    
    accuracy = np.mean(cvscores)
    loss = np.mean(cvloss)
    std=np.std(cvscores)
    loss_std = np.std(cvloss)
    print('CV_Accuracy:%.2f%%(+/- %.2f%%)\n'%(accuracy*100,std*100),
         'CV_Loss:%.2f%%(+/- %.2f%%)\n'%(loss*100,loss_std*100))
    model.save('trained_model/best_single_model.h5')
    
    # show evaluate score
    test_score=model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
    print("%s%s: %.2f%%\n" % ("evaluate ",model.metrics_names[1], test_score[1]*100),
          "%s%s: %.2f%%\n" % ("evaluate ",model.metrics_names[0], test_score[0]*100))
    return model


# In[12]:


# train multiple input model with cross validation fold
def train_multimodel_with_cv(trainAttrX, trainImages, testAttrX, testImages, trainY, testY, 
                            model, cross_validation, epochs, batch_size):
    print('Train Model')
    print("==================================================")
    # create model
    print('Create Model')
    model=model
    print("==================================================")    
    # set an earlystop to avoid overfitting
    earlystop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    print('Cross Validation')
    # fit the model
    cv=list(range(1, cross_validation+1)) #set number of cross validation
    iteration=1
    cvscores=[]
    cvloss = []
    for i in cv:
        print('==================================================')
        print("Iteration=", iteration)
        iteration+=1
        fit_stats = model.fit(x=[trainAttrX, trainImages], y=trainY, batch_size=batch_size, epochs=epochs, 
                            verbose='auto', callbacks=earlystop, validation_split=0.2, shuffle=True)
        
        # print validation accuracy score 
        acc=fit_stats.history['accuracy']
        los=fit_stats.history['loss']
        val_acc=fit_stats.history['val_accuracy']
        val_loss=fit_stats.history['val_loss']
        epoch=range(len(acc))
        cvscores.append(np.mean(val_acc))
        cvloss.append(np.mean(val_loss))
        print("Val_Accuracy: %.2f%%\n" % (np.mean(val_acc)*100),
              "Val_Loss: %.2f%%\n" % (np.mean(val_loss)*100))
        
        # display learning curve
        plt.plot(epoch, los, 'r', label='Training Loss')
        plt.plot(epoch, val_loss, 'b', label='Validation Loss')
        plt.title('Model Loss(iteration={})'.format(i))
        plt.legend()
        plt.show()
        plt.plot(epoch, acc, 'r', label='Training Accuracy')
        plt.plot(epoch, val_acc, 'b', label='Validation Accuracy')
        plt.title('Model Accuracy(iteration={})'.format(i))
        plt.legend()
        plt.show()
        
    # show the training results    
    accuracy = np.mean(cvscores)
    loss = np.mean(cvloss)
    std=np.std(cvscores)
    loss_std = np.std(cvloss)
    print('CV_Accuracy:%.2f%%(+/- %.2f%%)\n'%(accuracy*100,std*100),
         'CV_Loss:%.2f%%(+/- %.2f%%)\n'%(loss*100,loss_std*100))
    model.save('trained_model/best_multi_model.h5')
    
    # show evaluate score
    test_score=model.evaluate(x=[testAttrX, testImages], y=testY, batch_size=batch_size, verbose=1)
    print("%s%s: %.2f%%\n" % ("evaluate ",model.metrics_names[1], test_score[1]*100),
          "%s%s: %.2f%%\n" % ("evaluate ",model.metrics_names[0], test_score[0]*100))
    return model


# In[13]:


def auc_curve(y, prob):
    fpr,tpr,threshold = roc_curve(y, prob) # compute true positive and false positive
    roc_auc = auc(fpr,tpr) # compute auc score
    
    plt.figure()
    lw=2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area=%0.3f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:




