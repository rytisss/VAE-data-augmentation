from RoadPavementSegmentation.models.layers import *
from RoadPavementSegmentation.models.customLayers import *
from RoadPavementSegmentation.models.losses import *
from RoadPavementSegmentation.models.utilities import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'C:\Users\prorega\Downloads\holesTrain/'
weights_output_name = 'UNet4_res_assp_5x5_16k_320x320'

# https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/

# Official sample
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Encoder
# Returns flattened encoder data and tensor shape before flattening
input_size = (320, 320, 1)
number_of_kernels = 8
kernel_size = 3
stride = 1
number_of_output_neurons = 2000
max_pool = True
max_pool_size = 2
batch_norm = True

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
shape_before_flattening = keras.backend.int_shape(reduced_assp)[1:]

reduces_assp_flatten = Flatten()(reduced_assp) #in case of input 320x320 with 16 kernels, here should be 25600

# Define model output, reduce output dimensions
reduces_assp_flatten = Dense(number_of_output_neurons, activation='relu')(reduces_assp_flatten)

z_mean = Dense(number_of_output_neurons, name='mu')(reduces_assp_flatten)
z_log_var = Dense(number_of_output_neurons, name='log_var')(reduces_assp_flatten)

z = Sampling()([z_mean, z_log_var])



input = 2000
input_shape_before_flatten = (40, 40, 8)
number_of_kernels = 8
batch_norm = True
kernel_size = 3
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

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs):
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 320 * 320
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

def train():
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = 19000
    # batch size. How many samples you want to feed in one iteration?
    batch_size = 1
    # number_of_epoch. How many epoch you want to train?
    number_of_epoch = 8

    latent_space_size = 100

    encoder_model = keras.Model(encoder_input, [z_mean, z_log_var, z], name = "encoder")
    #encoder_model.summary()
    decoder_model = keras.Model(decoder_input, decoder_output, name = "decoder")
    #decoder_model.summary()
    # define full vae
    vae = VAE(encoder_model, decoder_model)
    vae.compile(optimizer=keras.optimizers.Adam())
    #vae_model.summary()


    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    data_dir = r'C:\Users\prorega\Downloads\holesTrain/'

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
    vae.fit(generator, steps_per_epoch=number_of_iteration,epochs=number_of_epoch,callbacks=[model_checkpoint, saver], shuffle = True)

def main():
    #tf.config.experimental_run_functions_eagerly(True)
    train()

if __name__ == "__main__":
    main()
