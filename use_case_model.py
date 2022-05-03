from model_creator.auto_encoder import Autoencoder
from model_creator.soundgenerator import SoundGenerator
from model_creator.config_default import HOP_LENGTH
import numpy as np
import os
import pickle


def recup_model(path_saved = './model'):
    """
    Allow to load the model
    """

    trained_auto_encoder = Autoencoder.load(path_saved)
    trained_encoder = trained_auto_encoder.encoder
    trained_decoder = trained_auto_encoder.decoder

    return trained_auto_encoder, trained_encoder, trained_decoder

def make_prediction_encoder(path_spectrogram) -> np.array:
    """
    Allow user to make prediction
    """
    global encoder

    #Load data
    example = np.load(path_spectrogram)
    #Change the dimentions
    example = np.expand_dims(example, 2)
    example = example[None,:,:,:]

    return encoder.predict(example)


def representation_oiseaux(espece : str) -> np.array:
    """
    Calcule la moyenne et la variance des différents sons d'oiseau
    """
    global encoder

    pred = []
    for tab_spec in os.listdir("preprocessed_data/" + espece + "/spectrograms"):
        pred.append(make_prediction_encoder("preprocessed_data/" + espece + "/spectrograms/" + tab_spec).reshape(256))
    pred = np.array(pred)

    return np.array([np.mean(pred, axis=1), np.var(pred, axis=1)])

def make_bird_song(encode_values : np.array, min_max_val : np.array, autoencoder : Autoencoder) -> np.array:
    """
    Permet de générer du son à partir des valeurs encodées
    """
    global decoder

    spectrogram_reconstruct = decoder.predict(encode_values)
    sound_generator = SoundGenerator(autoencoder, HOP_LENGTH)
    signal = sound_generator.convert_spectrograms_to_audio(spectrogram_reconstruct, min_max_val)

    return signal

def get_min_max_values(file_name : str, file_min_max : str) -> dict:
    """
    Get min and max value for a special file
    """

    with open(file_min_max, "rb") as f:
        min_max_values = pickle.load(f)

    return min_max_values[file_name]

if __name__ == "__main__":

    autoencoder, encoder, decoder = recup_model()
    #name = "Fringilla coelebs"
    #tab = representation_oiseaux(name)
    #np.save(arr=tab, file="model_result/" + name.replace(' ','_'))
