import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, LeakyReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from model_creator.config_default import LOG_DIR

tf.compat.v1.disable_eager_execution()

class Autoencoder:
    """
    Deep Convolutionnal autoencoder with mirrored encoder and decoder components
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim, save_path = 'model'):
        """
        Initialisation of the class with the different parameters
        """

        #Shape of the input spectrogram
        self.input_shape = input_shape # [1024,256,1]
        #Shape of the differents convolutials layers
        self.conv_filters = conv_filters # [2,4,8]
        #Shape of the kernels
        self.conv_kernels = conv_kernels # [3,5,3]
        #The different strides
        self.conv_strides = conv_strides # [1,2,2]
        #The size of the latent space
        self.latent_space_dim = latent_space_dim # 256
        #Weight given to the reconstruction loss
        self.reconstruction_loss_weight = 1000000
        #The save path
        self.save_path = save_path

        #Encoder model part
        self.encoder = None
        #Decoder model part
        self.decoder = None
        #The entire model encoder + decoder
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        '''
        Describe the network with the summary
        '''
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate):
        """Compile the model"""
        optimizer = Adam(learning_rate  = learning_rate)

        self.model.compile(optimizer = optimizer,
                        loss = self._calculate_combined_loss, 
                        metrics = [self._calculate_reconstruction_loss, self._calculate_kl_loss])
    
    def train(self, x_train, batch_size, num_epochs, patience = 10):
        """Train the model"""
        early_stopping = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
        tensorboard_callback = TensorBoard(LOG_DIR, histogram_freq=1)

        history = self.model.fit(x_train,
                                x_train,
                                batch_size=batch_size,
                                epochs=num_epochs,
                                callbacks = [early_stopping, tensorboard_callback],
                                shuffle=True)
        return history
    
    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
    
    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)

        return autoencoder

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

    def _save_parameters(self, save_folder):
        """
        Save the architecture of the model
        """

        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
            ]

        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
    
    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        #Input of the model
        model_input = self._model_input
        #Encoder part
        model_encoder = self.encoder(model_input)
        #Decoder part
        model_output = self.decoder(model_encoder)
        #Entire model
        self.model = Model(model_input, model_output, name="autoencoder")

    
    def _build_decoder(self):
        """
        Build the different layers for the decoder
        """

        #Decoder input
        decoder_input = self._add_decoder_input()
        #Dense layer before reshaping
        dense_layer = self._add_dense_layer(decoder_input)
        #Reshape layer to pass from 1D to 2D array
        reshape_layer = self._add_reshape_layer(dense_layer)
        #Add convolutionnal transpose blocks
        conv_transpose_layer = self._add_conv_transpose_layers(reshape_layer)
        #Add decoder output
        decoder_output = self._add_decoder_output(conv_transpose_layer)
        #Model final composition
        self.decoder = Model(decoder_input, decoder_output, name = 'decoder')
    

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), dtype=tf.float32, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        """
        Te different blocs to decompress the botleneck
        """

        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(self.latent_space_dim * 2, name="decoder_dense_1", activation= 'relu')(decoder_input)
        dense_layer = Dropout(0.3)(dense_layer)
        dense_layer = Dense(self.latent_space_dim * 4, name="decoder_dense_2", activation= 'relu')(dense_layer)
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

    def _build_encoder(self):
        """
        Build the different layers of the encoder
        """

        #Input
        encoder_input = self._add_encoder_input()
        #Convolution layer
        conv_layer = self._add_conv_layers(encoder_input)
        #Bottleneck
        bottleneck = self._add_bottleneck(conv_layer)
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
    
    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Gaussian sampling (Dense layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [7, 7, 32]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim * 4, name = "first_reduction", activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(self.latent_space_dim * 2, name = "second_reduction", activation="relu")(x)
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