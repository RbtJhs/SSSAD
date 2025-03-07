# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:33:40 2022

@author: ole
"""

import os
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import layers
import numpy as np
from skimage.io import imread
from tensorflow.keras.applications.resnet import preprocess_input
from skimage.transform import resize
import argparse




def parse_args():
    parser = argparse.ArgumentParser(description="Cut images into overlapping tiles")


    parser.add_argument("--data_dir",type=str, default="/data/robert/SSSAD/MAD", help="path to load folder")
    parser.add_argument("--save_checkpoint_to",type=str, default="./checkpoints/TreeAttention.h5", help="path to model checkpoint")
    parser.add_argument("--batch_size",type=int, default=1, help="")


    args = parser.parse_args()
    return args



class DataGen(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=1, shuffle=True, train=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train=train

        
        #Daten in Training und Validation splitten; es wird nur eins von beiden in einem Generator benutzt
        if self.train:
            paths = input_img_paths #alle Pfade sammeln
            self.list_IDs = list(map(lambda path:path.split('/')[-1].split('.')[0], paths))#[::3]
        else:
            paths = val_img_paths #alle Pfade sammeln
            self.list_IDs = list(map(lambda path:path.split('/')[-1].split('.')[0], paths))#[::3]
        print(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #Liste der für das Batch genutzten IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        if(self.train):
            X = self.__data_generation(list_IDs_temp)
        else:
            X = self.__data_generation(list_IDs_temp)

        return X
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        #für jedes Sample im Batch
        if(self.train):
            img_path = input_dir
            mask_path = target_dir
        else:
            img_path = val_input_dir
            mask_path = val_target_dir
        for i, ID in enumerate(list_IDs_temp):
            #Bild laden
            X = imread(img_path + ID + ".jpg").astype(np.float64)
            h,w,c = X.shape
            if(h>w):
                hn=512
                wn= int((512/h)*w)
            else:
                wn=512
                hn= int((512/w)*h)
            X = resize(X, (hn, wn),anti_aliasing=True).astype(np.float64)
            #pre-processing für ResNet, immer ne gute Idee
            X = preprocess_input(X)
            X = np.expand_dims(X,0)

            
            #GT Laden, wird hier noch weiterverabreitet; am Ende ist y das entscheidende
            y = imread(mask_path + ID + ".jpg")
            #print(y)
            y = y>0
            #print(y)
            y  = resize(y, (hn, wn))
            yOri = y.astype(np.int64)
            y = np.expand_dims(yOri,0)
            #print(y)
            y = np.expand_dims(y,-1)
            #print(y.shape)
                                
            
            #ignore random background
            w = np.copy(yOri).astype(np.float64)
            
            posNumber = np.sum(y)*3
            coords = list(zip((np.where(yOri==0))[0],(np.where(yOri==0))[1]))
            np.random.shuffle(coords)
            cuttedCoords = tuple(zip(*coords[:posNumber]))
            
            w[cuttedCoords] = 1
            w = np.expand_dims(w,0)
            w = np.expand_dims(w,-1)

            
        return (X ,y, w)
    
    
    



def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = GroupNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    previous_block_activation32 = x

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = GroupNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = GroupNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        if filters == 64:
            previous_block_activation64 = x
        if filters == 128:
            previous_block_activation128 = x

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = GroupNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = GroupNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual

        # Concat skip Layers
        if filters == 256:
            x = layers.Concatenate()([x, previous_block_activation128])
        if filters == 128:
            x = layers.Concatenate()([x, previous_block_activation64])
        if filters == 64:
            x = layers.Concatenate()([x, previous_block_activation32])

        # Set aside next residual
        previous_block_activation = x

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model




args = parse_args()

img_size = (None, None)
num_classes = 1
batch_size = 1

input_dir = args.data_dir + "/train/"
input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
    ]
)
target_dir = args.data_dir + "/train_masks/"
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
    ]
)

val_input_dir = args.data_dir + "/val/"
val_img_paths = sorted(
    [
        os.path.join(val_input_dir, fname)
        for fname in os.listdir(val_input_dir)
    ]
)

val_target_dir = args.data_dir + "/val_masks/"
val_target_img_paths = sorted(
    [
        os.path.join(val_target_dir, fname)
        for fname in os.listdir(val_target_dir)
    ]
)

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()




# Instantiate data Sequences for each split
train_gen = DataGen(batch_size=args.batch_size,shuffle = True)
val_gen = DataGen(batch_size=args.batch_size,shuffle = True, train=False)


# Configure the model for training.
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"],run_eagerly=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(args.save_checkpoint_to, save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')
]

# Train the model, doing validation at the end of each epoch.
epochs = 50

model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

