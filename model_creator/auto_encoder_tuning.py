import numpy as np
import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, LeakyReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

tf.compat.v1.disable_eager_execution()

class Autoencoder_Tuning(kt.HyperModel):

    """
    Deep Convolutionnal autoencoder tuning with mirrored encoder and decoder components
    """

    def __init__(self, input_shape, dico_param, save_path):
        """
        Initialisation of the class with the different parameters
        """
        #Shape of the input spectrogram
        self.input_shape = input_shape # [1024,256,1]

        #Dictionnaire
        self.dico_param = dico_param

        #The save path
        self.save_path = save_path

        #Encoder model part
        self.encoder = None
        #Decoder model part
        self.decoder = None
        #The entire model encoder + decoder
        self.model = None

        self._shape_before_bottleneck = None
        self._model_input = None


    def summary(self):
        '''
        Describe the network with the summary
        '''
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate, hp):
        """Compile the model"""

        optimizer = Adam(learning_rate  = learning_rate)
        #Weight given to the reconstruction loss
        self.reconstruction_loss_weight = 1000000

        self.model.compile(optimizer = optimizer,
                        loss = self._calculate_combined_loss, 
                        metrics = [self._calculate_reconstruction_loss, self._calculate_kl_loss])

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss

        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis = [1,2,3])
        return reconstruction_loss

    
    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = - 0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        return kl_loss

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

    def build(self, hp):

        self.__archi_choice(hp)
        #The size of the latent space
        self.latent_space_dim = hp.Int("Size_latent _space", min_value=128, max_value=1024, step=128)
        self._build_encoder(hp)
        self._build_decoder(hp)
        self._build_autoencoder()
        self.compile(hp.Float("learning_rate_tuning", min_value=0.00005, max_value=0.0002, step=0.00005),hp)

        return self.model

    def __archi_choice(self, hp):
        """
        Permet de choisir entre differents architectures de modèle de convolution
        """
        archi_name = hp.Choice("Choix_architecture", list(self.dico_param.keys()))

        #Shape of the differents convolutials layers
        self.conv_filters = self.dico_param[archi_name]["conv_filters"]
        #Shape of the kernels
        self.conv_kernels = self.dico_param[archi_name]["conv_kernels"]
        #The different strides
        self.conv_strides = self.dico_param[archi_name]["conv_strides"]
        #Know the 
        self._num_conv_layers = len(self.conv_filters)


    def _build_autoencoder(self):
        #Input of the model
        model_input = self._model_input
        #Encoder part
        model_encoder = self.encoder(model_input)
        #Decoder part
        model_output = self.decoder(model_encoder)
        #Entire model
        self.model = Model(model_input, model_output, name="autoencoder")

    
    def _build_decoder(self, hp):
        """
        Build the different layers for the decoder
        """

        #Decoder input
        decoder_input = self._add_decoder_input()
        #Dense layer before reshaping
        dropout = hp.Boolean("Decoder_dropout")
        dense_layer = self._add_dense_layer(decoder_input, dropout=dropout)
        #Reshape layer to pass from 1D to 2D array
        reshape_layer = self._add_reshape_layer(dense_layer)
        #Add convolutionnal transpose blocks
        conv_transpose_layer = self._add_conv_transpose_layers(reshape_layer)
        #Add decoder output
        decoder_output = self._add_decoder_output(conv_transpose_layer)
        #Model final composition
        self.decoder = Model(decoder_input, decoder_output, name = 'decoder')
    
    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input, dropout = True):
        """
        The different blocs to decompress the botleneck
        """

        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(self.latent_space_dim * 2, name="decoder_dense_1", activation= 'relu')(decoder_input)

        #Permet de tester avec ou sans dropout
        if dropout:
            dense_layer = Dropout(0.3)(dense_layer)
        dense_layer = Dense(self.latent_space_dim * 4, name="decoder_dense_2", activation= 'relu')(dense_layer)

        #Permet de tester avec ou sans dropout
        if dropout:
            dense_layer = Dropout(0.3)(dense_layer)
        dense_layer = Dense(num_neurons, name="decoder_dense_3", activation = 'relu')(dense_layer)

        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index], # [24, 24, 1]
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = LeakyReLU(name=f"decoder_leaky_relu_{layer_num}")(x)
        x = BatchNormalization(name =f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters = 1, # [24, 24, 1]
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = "same",
            name=f"decoder_conv_transpose_layer_output{self._num_conv_layers}"
        )

        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self, hp):
        """
        Build the different layers of the encoder
        """

        #Input
        encoder_input = self._add_encoder_input()
        #Convolution layer
        conv_layer = self._add_conv_layers(encoder_input)
        #Bottleneck with or without Dropout
        dropout = hp.Boolean("Encoder_dropout")
        bottleneck = self._add_bottleneck(conv_layer, dropout=dropout)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name = "encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name = "encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x
    
    def _add_conv_layer(self, layer_index, x):
        """
        Define the convolutionnal block
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"encoder_conv_layer_{layer_index + 1}"
        )

        x = conv_layer(x)
        x = LeakyReLU(name=f"encoder_leakyrelu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x, dropout = True):
        """Flatten data and add bottleneck with Gaussian sampling (Dense layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [7, 7, 32]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim * 4, name = "first_reduction", activation="relu")(x)
        if dropout:
            x = Dropout(0.3)(x)
        x = Dense(self.latent_space_dim * 2, name = "second_reduction", activation="relu")(x)
        if dropout:
            x = Dropout(0.3)(x)

        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon

            return sampled_point

        x = tf.keras.layers.Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])
        return x