import librosa
from model_creator.preprocess import MinMaxNormaliser

class SoundGenerator:
    def __init__(self, autoencoder, hop_length):
        self.autoencoder=autoencoder
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.autoencoder.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            #reshape the log spectrograms
            log_spectrogram = spectrogram[:, :, 0]
            #apply denormalisation
            denorm_log_spec = self._min_max_normaliser.denormalise(log_spectrogram, min_max_value["min"], min_max_value["max"])
            #log spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            #apply Griffin-Lin
            signal = librosa.istft(spec, hop_length=self.hop_length)
            #append signal to "signals"
            signals.append(signal)
        return signals