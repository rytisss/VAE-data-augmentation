from RoadPavementSegmentation.models.layers import *
from RoadPavementSegmentation.models.customLayers import *
from RoadPavementSegmentation.models.losses import *
import tensorflow.keras.backend as keras

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras import backend as K

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'D:\drilled holes data for training\UNet4_res_assp_5x5_16k_320x320_coordConv_v2/'
weights_output_name = 'UNet4_res_assp_5x5_16k_320x320'

# part of the code from https://github.com/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

# VQ layer.
class VQVAELayer(Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                 shape=(self.embedding_dim, self.num_embeddings),
                                 initializer=self.initializer,
                                 trainable=True)

        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs ** 2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        # Metrics.
        # avg_probs = K.mean(encodings, axis=0)
        # perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))

        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

#encoder
def encoder(input_size = (320,320),
            number_of_kernels = 16,
            kernel_size = 3,
            stride = 1,
            max_pool = True,
            max_pool_size = 2,
            batch_norm = True):
    # Input
    inputs = Input(input_size)
    # encoding
    _, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                            batch_norm)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size)


# Calculate vq-vae loss.
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat) / data_variance

        e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
        q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
        loss = q_latent_loss + commitment_cost * e_latent_loss

        return recon_loss + loss  # * beta

    return vq_vae_loss

# Hyper Parameters.
epochs = 1000 # MAX
batch_size = 64
validation_split = 0.1

# VQ-VAE Hyper Parameters.
embedding_dim = 32 # Length of embedding vectors.
num_embeddings = 128 # Number of embedding vectors (high value = high bottleneck capacity).
commitment_cost = 0.25 # Controls the weighting of the loss terms.

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=True)



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

    # Define model
    model = UNet4_res_aspp_First5x5_CoordConv(number_of_kernels=16,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)

    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = r'D:\drilled holes data for training/'

    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    # Define data generator that will take images from directory
    generator = trainGenerator(batch_size, data_dir, 'images', 'labels', data_gen_args, save_to_dir = None, target_size = (320,320))

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
    model.fit_generator(generator,steps_per_epoch=number_of_iteration,epochs=number_of_epoch,callbacks=[model_checkpoint, saver, learning_rate_scheduler], shuffle = True)

def main():
    train()

if __name__ == "__main__":
    main()
