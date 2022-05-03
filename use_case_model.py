from model_creator.auto_encoder import Autoencoder
import numpy as np
import pandas as pd
import os


def recup_model(path_saved = './model'):
    """
    Allow to load the model
    """

    trained_auto_encoder = Autoencoder.load(path_saved)
    trained_encoder = trained_auto_encoder.encoder
    trained_decoder = trained_auto_encoder.decoder

    return trained_encoder, trained_decoder

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

    return np.array(np.mean(pred, axis=1), np.var(pred, axis=1))

if __name__ == "__main__":
    encoder, decoder = recup_model()
    name = "Fringilla coelebs"
    tab = representation_oiseaux(name)
    np.save(arr=tab, file="preprocessed_data/" + name.replace(' ','_'))

