# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:33:40 2022

@author: ole
"""

from tensorflow import keras
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import layers
import numpy as np
from skimage.io import imread, imsave
from tensorflow.keras.applications.resnet import preprocess_input
from skimage.transform import resize
import os
from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")


    parser.add_argument("--load_imgs_from",type=str, default="./data/MAD/test", help="path to load folder")
    parser.add_argument("--checkpoint",type=str, default="./checkpoints/TreeAttention.h5", help="path to model checkpoint")

    parser.add_argument("--result_dir",type=str, default="./results/TreeAttention/test/", help="path to result dir")


    args = parser.parse_args()
    return args

num_classes = 1


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


if __name__ == '__main__':
    args = parse_args()
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    # Build model
    model = get_model((None,None), 1)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights(args.checkpoint)
    
    if(not os.path.exists(args.result_dir)):
        os.makedirs(args.result_dir)
    paths = glob(args.load_imgs_from + "/*.jpg")
    
    print("number of imgs: ",len(paths))
    
    for path in paths:
        name = path.split("/")[-1].split(".")[0]
        X = imread(path).astype(np.float64)
        h,w,c = X.shape
        if(h>w):
            hn=512
            wn= int((512/h)*w)
        else:
            wn=512
            hn= int((512/w)*h)
        X = resize(X, (hn, wn),anti_aliasing=True).astype(np.float64)
        X = preprocess_input(X)
        X = np.expand_dims(X,0)
        pred = model.predict(X)
        imsave(args.result_dir + '/'+name+'.png',np.asarray(pred[0,:,:,:]*255,dtype=np.uint8))


