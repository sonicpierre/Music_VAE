from model_creator.auto_encoder import Autoencoder
import numpy as np
import os

class Use_Case_Model:
    """
    Permet la construction des espaces latents moyens pour la génération de sons
    """
    
    def __init__(self, path_saved = './model'):
        self.path_saved = path_saved

    def recup_model(self):
        """
        Allow to load the model
        """

        trained_auto_encoder = Autoencoder.load(self.path_saved)
        trained_encoder = trained_auto_encoder.encoder
        trained_decoder = trained_auto_encoder.decoder

        return trained_auto_encoder, trained_encoder, trained_decoder

    def make_prediction_encoder(self, path_spectrogram, encoder) -> np.array:
        """
        Allow user to make prediction
        """

        #Load data
        example = np.load(path_spectrogram)
        #Change the dimentions
        example = np.expand_dims(example, 2)
        example = example[None,:,:,:]

        return encoder.predict(example)

    def representation_oiseaux(self, espece : str, encoder) -> np.array:
        """
        Calcule la moyenne et la variance des différents sons d'oiseau
        """

        pred = []
        for tab_spec in os.listdir("preprocessed_data/" + espece + "/spectrograms"):
            pred.append(self.make_prediction_encoder("preprocessed_data/" + espece + "/spectrograms/" + tab_spec, encoder).reshape(256))
        pred = np.array(pred)

        return np.array([np.mean(pred, axis=0), np.var(pred, axis=0)])


    def construction_utils(self, espece : list) -> None:
        """
        Permet de sauvegarder les moyennes des vecteurs
        """
        _, encoder,_ = self.recup_model()

        if not os.path.exists("model_result/"):
            os.makedirs("model_result")

        for esp in espece:
            tab = self.representation_oiseaux(esp, encoder)
            np.save(arr=tab, file="model_result/" + esp.replace(' ','_'))