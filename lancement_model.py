import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import config
from mutagen.wave import WAVE

from model_creator.decoupe import find_chunks, reconstruct_chunks
from model_creator.preprocess import Loader, Padder, LogSpectrogramExtractor, MinMaxNormaliser, Saver, PreprocessingPipeline
from model_creator.auto_encoder import Autoencoder
from model_creator import config_default

def chargement_espece(metadata : pd.DataFrame, nb_espece = None) -> list:
    """
    Charge les espece qu'on veut mettre dans le modèle
    """

    if(nb_espece == None):
        return pd.unique(metadata['Species'])[1:]
    return pd.unique(metadata['Species'])[1:nb_espece + 1]


def decoupe_son(espece : list, meta_df : pd.DataFrame) -> None:
    """
    Permet de découper les chants en se centrant sur les chants d'oiseau et les sauvegarder au bon endroit en créant l'architecture adaptée.    
    """

    for bird in espece:
        data = meta_df[meta_df['Species']==bird]

        data.index = range(data.shape[0])
        for song_num in tqdm(range(data.shape[0])):
            #Lien vers les fichiers MP3
            path_file = config.LIEN_DIR_MP3 + data.loc[song_num]['Path'].split('/')[-1]
            #Récupération de l'ID du son
            id_song = str(data['Recording_ID'].loc[song_num])
            #Decomposition des musiques en morceaux
            chunks = find_chunks(path_file, config.SILENCE_GAP, config.SILENCE_BAR)
            #Sauvegarde des musiques
            reconstruct_chunks(chunks,"preprocessed_data/" + bird + "/song/", id_song)


def sup_enregistrement_court(espece: str) -> int:
    """
    Permet la suppression des chants trop long ou trop courts.
    """
    for specie in espece:
        bird_dir = "preprocessed_data/" + specie +"/song/"
        compteur_sup = 0
        songs = os.listdir(bird_dir)
        for song in songs:
            audio = WAVE(bird_dir + song)
            audio_info = audio.info
            if audio_info.length < config.TOO_SHORT_LENGHT or audio_info.length > config.TOO_LONG_LENGHT:
                os.remove(bird_dir + song)
                compteur_sup+=1
    return compteur_sup

def creation_archi(bird_name : str) -> None:
    """
    Permet de créer les dossiers où l'on met les spectrograms
    """

    if not os.path.exists("./preprocessed_data/"+ bird_name +"/spectrograms/"):
        os.mkdir("./preprocessed_data/"+ bird_name +"/spectrograms/")
    if not os.path.exists("./model/"):
        os.mkdir("./model/")

def initialisation_pipeline(save_spec_dir : str, min_max_values : str) -> PreprocessingPipeline:
    """
    Initialisation de la pipeline de preprocessing
    """

    loader = Loader(config_default.SAMPLE_RATE, config_default.DURATION, config_default.MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(config_default.FRAME_SIZE, config_default.HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(save_spec_dir, min_max_values)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    return preprocessing_pipeline

def creation_spectrogram(espece : str) -> None:
    """
    Permet de créer les différents spectrograms pour l'entraînement de l'auto-encodeur
    """

    for bird_name in espece:
        file_dir = "./preprocessed_data/"+ bird_name +"/song/"
        spec_save_dir = "./preprocessed_data/"+ bird_name +"/spectrograms/"
        min_max_save_dir = "./preprocessed_data/"+ bird_name +"/"

        creation_archi(bird_name)
        preprocessing_pipeline = initialisation_pipeline(spec_save_dir, min_max_save_dir)
        preprocessing_pipeline.process(file_dir)

def load_music(species_name):
    x_train = []
    for specie in species_name:
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

def train(x_train : np.array, learning_rate : float, batch_size : int, epochs : int):
    """
    Start the training
    """

    autoencoder = Autoencoder(
        input_shape=(taille_input[0], taille_input[1], 1),
        conv_filters=(512,256, 128, 64, 32),
        conv_kernels=(3,3,3,3,2),
        conv_strides=(2,2,2,2, (2,1)),
        latent_space_dim=256
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    history = autoencoder.train(x_train, batch_size, epochs)
    
    return autoencoder, history


if __name__ == "__main__":

    meta_df = pd.read_csv(config.LIEN_METADATA)
    espece = chargement_espece(meta_df, config.NB_ESPECE)
    decoupe_son(espece, meta_df)
    num_supp = sup_enregistrement_court(espece)
    creation_spectrogram(espece)
    spec = os.listdir("./preprocessed_data/"+ espece[0] +"/spectrograms")
    taille_input = np.load("./preprocessed_data/"+ espece[0] +"/spectrograms/" + spec[0]).shape
    x_train = load_music(espece)
    autoencoder, history = train(x_train, 
                                config.MODEL_PARAM['learning_rate'], 
                                config.MODEL_PARAM['batch_size'], 
                                config.MODEL_PARAM['epochs'])
    autoencoder.save("model")