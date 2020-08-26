from RoadPavementSegmentation.models.layers import *
from RoadPavementSegmentation.models.customLayers import *
from RoadPavementSegmentation.models.losses import *
from RoadPavementSegmentation.models.utilities import *
import tensorflow.keras.backend as keras

import numpy as np
import os
from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras import backend as K

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'D:\drilled holes data for training\UNet4_res_assp_5x5_16k_320x320_coordConv_v2/'
weights_output_name = 'UNet4_res_assp_5x5_16k_320x320'

# https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8

# Encoder
# Returns flattened encoder data and tensor shape before flattening

def encoder(input_size = (320, 320, 1),
            number_of_kernels = 16,
            kernel_size = 3,
            stride = 1,
            number_of_output_neurons = 2000,
            max_pool = True,
            max_pool_size = 2,
            batch_norm = True):
    # Input
    encoder_input = Input(input_size)
    # encoding
    _, enc0 = EncodingLayer(encoder_input, number_of_kernels, 5, stride, max_pool, max_pool_size, batch_norm)
    _, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
    _, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
    _, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,batch_norm)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size)

    # reduce dimensions, because in this case when input is 320x320 with 16kernels, here will be tensor with dimension of 204800
    reduced_assp = Conv2D(number_of_kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(assp)
    if batch_norm == True:
        reduced_assp = BatchNormalization()(reduced_assp)
    reduced_assp = Activation('relu')(reduced_assp)

    # Required for reshaping latent vector while building Decoder
    shape_before_flattening = K.int_shape(reduced_assp)[1:]

    reduces_assp_flatten = Flatten()(reduced_assp) #in case of input 320x320 with 16 kernels, here should be 25600

    # Define model output, reduce output dimensions
    reduces_assp_flatten = Dense(number_of_output_neurons)(reduces_assp_flatten)

    mean_mu = Dense(number_of_output_neurons, name='mu')(reduces_assp_flatten)
    log_var = Dense(number_of_output_neurons, name='log_var')(reduces_assp_flatten)

    # Defining a function for sampling
    def sampling(args):
        mean_mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.)
        return mean_mu + K.exp(log_var / 2) * epsilon

        # Using a Keras Lambda Layer to include the sampling function as a layer
        # in the model

    encoder_output = Lambda(sampling, name='encoder_output')([mean_mu, log_var])

    return encoder_input, encoder_output, mean_mu, log_var, shape_before_flattening, Model(encoder_input, encoder_output)


def decoder(input = 2000, input_shape_before_flatten = (40, 40, 16), number_of_kernels = 16, batch_norm = True, kernel_size = 3):
    # Define model input
    decoder_input = Input(shape=(input,), name='decoder_input')

    x = Dense(np.prod(input_shape_before_flatten))(decoder_input)
    x = Reshape(input_shape_before_flatten)(x)

    #increase feature count with 1x1 convolution
    x = Conv2D(number_of_kernels * 8, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                          kernel_initializer='he_normal')(x)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    dec2 = DecodingLayerRes(x, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    decoder_output = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    return decoder_input, decoder_output, Model(decoder_input, decoder_output)

def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        self.overallIteration = 0
    def on_batch_end(self, iteration, logs={}):
        self.overallIteration += 1
        if self.overallIteration % 5000 == 0 and self.overallIteration != 0:  # or save after some epoch, each k-th epoch etc.
            print('Saving iteration ' + str(self.overallIteration))
            self.model.save(weights_output_dir + weights_output_name + "_{}.hdf5".format(self.overallIteration))

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    step = epoch // 4
    init_lr = 0.001
    lr = init_lr / 2**step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr

def train():
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = 26000
    # batch size. How many samples you want to feed in one iteration?
    batch_size = 4
    # number_of_epoch. How many epoch you want to train?
    number_of_epoch = 16

    vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder = encoder()
    vae_decoder_input, vae_decoder_output, vae_decoder = decoder()

    LOSS_FACTOR = 10000

    def kl_loss(y_true, y_pred):
        kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis=1)
        return kl_loss

    def total_loss(y_true, y_pred):
        return LOSS_FACTOR * r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

    # define full vae
    vae_input = vae_encoder_input
    vae_output = vae_decoder(vae_encoder_output)
    vae_model = Model(vae_input, vae_output)

    vae_model.compile(optimizer=Adam(lr = 0.001), loss=total_loss, metrics=[r_loss, kl_loss])
    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = r'D:\holesTrain_\Image_rois/'

    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    # Define data generator that will take images from directory
    data_flow = ImageDataGenerator(rescale=1. / 255).flow_from_directory(data_dir,
                                                                         target_size=(320,320),
                                                                         batch_size=batch_size,
                                                                         shuffle=True,
                                                                         class_mode='input',
                                                                         subset='training'
                                                                         )

    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
    # Custom saving
    saver = CustomSaver()
    # Learning rate scheduler
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Make checkpoint for saving each
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_name, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)
    vae_model.fit_generator(data_flow,steps_per_epoch=number_of_iteration,epochs=number_of_epoch,callbacks=[model_checkpoint, saver, learning_rate_scheduler], shuffle = True)

def main():
    train()

if __name__ == "__main__":
    main()
