import numpy as np
import os
import soundfile as sf
import pickle
import model_creator.config_default as conf
from model_creator.soundgenerator import SoundGenerator
from model_creator.auto_encoder import Autoencoder

class Test:
    
    def __init__(self, spectrograms_path, min_max_path, original_save, generated_save, model_dir):
        self.spectrograms_path = spectrograms_path
        self.original_save = original_save
        self.generated_save = generated_save
        self.min_max_path = min_max_path
        self.autoencoder = Autoencoder.load(model_dir)
        self.sound_generator = SoundGenerator(self.autoencoder, conf.HOP_LENGTH)

    def load_audio(self):
        """
        Load audio from the path of the spectrograms
        """
        x_train = []
        file_paths = []
        for root, _, file_names in os.walk(self.spectrograms_path):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
                x_train.append(spectrogram)
                file_paths.append(file_path)
        x_train = np.array(x_train)
        x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
        return x_train, file_paths


    def select_spectrograms(self, spectrograms,file_paths,min_max_values,num_spectrograms=2):
        
        """
        Select randomly a number of sound to decompose and reconstruct

        spectrograms: 2D array wich represent the spectrogram
        file_paths: The path of the sound
        min_max_values: The differents minimum and maximum values for scaling
        num_spectrograms: The number of sounds you want to test reconstruction
        """
        
        sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
        sampled_spectrogrmas = spectrograms[sampled_indexes]
        file_paths = [file_paths[index] for index in sampled_indexes]
        sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

        return sampled_spectrogrmas, sampled_min_max_values


    def save_signals(self, signals, save_dir, sample_rate=22050):
        """
        Save the spectrograms created in a directory
        """

        for i, signal in enumerate(signals):
            save_path = os.path.join(save_dir, str(i) + ".wav")
            sf.write(save_path, signal, sample_rate)


    def test_reconstruction(self):

        with open(self.min_max_path, "rb") as f:
            min_max_values = pickle.load(f)

        specs, file_paths = self.load_audio()

        sampled_specs, sampled_min_max_values = self.select_spectrograms(specs, file_paths, min_max_values, 5)
        signals, _ = self.sound_generator.generate(sampled_specs, sampled_min_max_values)
        orignal_signals = self.sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)

        self.save_signals(signals, self.generated_save)
        self.save_signals(orignal_signals, self.original_save)