import keras_tuner as kt
import tensorflow as tf
from model_creator.auto_encoder import Autoencoder
from model_creator.auto_encoder_tuning import Autoencoder_Tuning
import model_creator.config_default as conf
import numpy as np
import os

class CreateData:
    def __init__(self, species_name : list) -> None:
        self.species_name = species_name

    def load_music(self):
        """
        First load the spectrograms construct the training song set        
        """

        x_train = []
        for specie in self.species_name:
            spectrograms_path = "./preprocessed_data/" + specie + "/spectrograms"
            
            for root, _, file_names in os.walk(spectrograms_path):
                for file_name in file_names:
                    file_path = os.path.join(root, file_name)
                    spectrogram = np.load(file_path) # (n_bins, n_frames)
                    x_train.append(spectrogram)

            x_train = np.array(x_train)
            x_train = x_train[..., np.newaxis]
            np.random.shuffle(x_train)
            x_train = x_train
        
        return x_train


class ClassiqueTrain:
    """
    Build the model with default parameters adapted to birds 
    """

    def __init__(self, taille_input : tuple) -> None:
        self.taille_input = taille_input

        self.autoencoder = Autoencoder(
            input_shape=(self.taille_input[0], self.taille_input[1], 1),
            conv_filters=(512,256, 128, 64, 32),
            conv_kernels=(3,3,3,3,2),
            conv_strides=(2,2,2,2, (2,1)),
            latent_space_dim=256,
            save_path=conf.MODEL_PATH_CLASSIQUE
        )

    def fit_classique(self, x_train : np.array):

        self.autoencoder.summary()
        self.autoencoder.compile(conf.DEFAULT_LEARNING_RATE)
        history = self.autoencoder.train(x_train, conf.DEFAULT_BATCH_SIZE,  conf.DEFAULT_EPOCHS)
        
        return self.autoencoder, history

class ParameterTuning:
    """
    Allow model hyper-tuning
    """
    def __init__(self, taille_input : tuple) -> None:
        self.taille_input = taille_input

        auto_tuner = Autoencoder_Tuning(input_shape=(self.taille_input[0], self.taille_input[1], 1),
            conv_filters=(512,256, 128, 64, 32),
            conv_kernels=(3,3,3,3,2),
            conv_strides=(2,2,2,2, (2,1)),
            save_path=conf.MODEL_PATH_CLASSIQUE)
        
        self.tuner = kt.RandomSearch(
            auto_tuner,
            objective="loss",
            max_trials=3,
            overwrite=True,
            directory="my_dir",
            project_name="tune_hypermodel",
        )
    
    def tune(self, x_train):
        self.tuner.search(x_train, x_train, epochs=30, callbacks=[tf.keras.callbacks.EarlyStopping('loss', patience=3)])