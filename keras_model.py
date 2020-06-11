"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (64*64*64*64*8*64*64*64*64)
    """
    inputLayer = Input(shape=(inputDim,))

    h = Dense(64)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)

    return Model(inputs=inputLayer, outputs=h)
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

    
