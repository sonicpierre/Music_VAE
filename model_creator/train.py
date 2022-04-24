from model_creator.auto_encoder import Autoencoder
import model_creator.config_default as conf
import numpy as np
import os

class ClassiqueTrain :
    """
    Build the model with default parameters adapted to birds 
    """

    def __init__(self, species_name : list, taille_input : tuple) -> None:
        self.species_name = species_name
        self.taille_input = taille_input

        self.autoencoder = Autoencoder(
            input_shape=(self.taille_input[0], self.taille_input[1], 1),
            conv_filters=(512,256, 128, 64, 32),
            conv_kernels=(3,3,3,3,2),
            conv_strides=(2,2,2,2, (2,1)),
            latent_space_dim=256
        )


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
        
        return x_train

    def train_classique(self, x_train : np.array):

        self.autoencoder.summary()
        self.autoencoder.compile(conf.DEFAULT_LEARNING_RATE)
        history = self.autoencoder.train(x_train, conf.DEFAULT_LEARNING_RATE, conf.DEFAULT_BATCH_SIZE,  conf.DEFAULT_EPOCHS)
        
        return self.autoencoder, history

    def fit_classique(self):
        """
        Start the training classique
        """
        x_train = self.load_music()
        self.train_classique(x_train)


class ParameterTuning:
    def __init__(self) -> None:
        pass