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

from keras.losses import mse, binary_crossentropy

from keras.utils import plot_model

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'C:\Users\prorega\Downloads\holesTrain/'
weights_output_name = 'UNet4_res_assp_5x5_16k_320x320'

# https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/

# Official sample
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    mu_mean, z_log_var = args
    batch = K.shape(mu_mean)[0]
    dim = K.int_shape(mu_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return mu_mean + K.exp(0.5 * z_log_var) * epsilon


# Encoder
# Returns flattened encoder data and tensor shape before flattening
def encoder(input_size = (320, 320, 1),
            number_of_kernels = 8,
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
    reduces_assp_flatten = Dense(number_of_output_neurons, activation='relu')(reduces_assp_flatten)

    mean_mu = Dense(number_of_output_neurons, name='mu')(reduces_assp_flatten)
    log_var = Dense(number_of_output_neurons, name='log_var')(reduces_assp_flatten)

    encoder_output = Lambda(sampling)([mean_mu, log_var])

    return encoder_input, encoder_output, mean_mu, log_var, shape_before_flattening


def decoder(input = 2000, input_shape_before_flatten = (40, 40, 8), number_of_kernels = 8, batch_norm = True, kernel_size = 3):
    # Define model input
    decoder_input = Input(shape=(input,), name='decoder_input')

    x = Dense(np.prod(input_shape_before_flatten), activation='relu')(decoder_input)
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

    return decoder_input, decoder_output


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

LOSS_FACTOR = 2000

def train():
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = 19000
    # batch size. How many samples you want to feed in one iteration?
    batch_size = 1
    # number_of_epoch. How many epoch you want to train?
    number_of_epoch = 8

    latent_space_size = 1000

    encoder_input, encoder_output, mean_mu, log_var, vae_shape_before_flattening = encoder(number_of_output_neurons=latent_space_size)
    encoder_model = Model(encoder_input, [mean_mu, log_var, encoder_output], name = "encoder")
    #encoder_model.summary()
    decoder_input, decoder_output = decoder(input=latent_space_size, input_shape_before_flatten=vae_shape_before_flattening)
    decoder_model = Model(decoder_input, decoder_output)
    #decoder_model.summary()
    # define full vae
    vae_output = decoder_model(encoder_model(encoder_input)[2])
    vae_model = Model(encoder_input, vae_output)
    #vae_model.summary()


    #encoder_model.compile(optimizer=Adam(lr = 0.001), loss = 'binary_crossentropy')
    #tf.keras.utils.plot_model(encoder_model, to_file='encoder.png', show_shapes=True, show_layer_names=True)

    #decoder_model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
    #tf.keras.utils.plot_model(decoder_model, to_file='decoder.png', show_shapes=True, show_layer_names=True)

    def kl_reconstruction_loss(true, pred):
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * 320 * 320
        # KL divergence loss
        kl_loss = 1 + log_var - tf.square(mean_mu) - tf.exp(log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        mean = K.mean(reconstruction_loss + kl_loss)
        print(reconstruction_loss)
        print(kl_loss)
        #print(mean)
        return mean

    vae_model.compile(optimizer=Adam(), loss = kl_reconstruction_loss, metrics=[kl_reconstruction_loss])

    #tf.keras.utils.plot_model(vae_model, to_file='UNet4.png', show_shapes=True, show_layer_names=True)
    #https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/issues/3

    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = r'C:\Users\prorega\Downloads\holesTrain/'
    image_folder = 'Image_rois'

    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    # Define data generator that will take images from directory
    generator = trainGenerator(batch_size, data_dir, 'Image_rois', 'Label_rois', data_gen_args, save_to_dir=None,
                               target_size=(320, 320))

    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
    # Custom saving
    saver = CustomSaver()
    # Learning rate scheduler
    #learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Make checkpoint for saving each
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_name, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)
    vae_model.fit_generator(generator, steps_per_epoch=number_of_iteration,epochs=number_of_epoch,callbacks=[model_checkpoint, saver], shuffle = True)

def main():
    tf.config.experimental_run_functions_eagerly(True)
    train()

if __name__ == "__main__":
    main()
