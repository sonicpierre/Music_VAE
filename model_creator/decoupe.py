# Import the AudioSegment class for processing audio and the 
# split_on_silence function for separating out silent chunks.
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence


def find_chunks(file, minimum_silence_gap, silence_bar):
    """
    Permet de trouver les différents chunks où les oiseaux chantes
    """

    # Load your audio.
    song = AudioSegment.from_mp3(file)

    # Split track where the silence is 2 seconds or more and get chunks using 
    # the imported function.
    chunks = split_on_silence (
        # Use the loaded audio.
        song, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = minimum_silence_gap,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        # (You may want to adjust this parameter.)
        silence_thresh = silence_bar
    )
    return chunks

def reconstruct_chunks(chunks,exporting_dir,name,padding = 300):
    """
    Permet de reconstruire les différents chunks 
    """
    
    # Define a function to normalize a chunk to a target amplitude.
    def match_target_amplitude(aChunk, target_dBFS):
        ''' Normalize given audio chunk '''
        change_in_dBFS = target_dBFS - aChunk.dBFS
        return aChunk.apply_gain(change_in_dBFS)

    if not os.path.exists(exporting_dir):
        os.makedirs(exporting_dir)
        
    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 1000 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=padding)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
        # Export the audio chunk with new bitrate.
        #print(exporting_dir + "Exporting chunk{0}.mp3.".format(i))
        normalized_chunk.export(
            ".//"+ exporting_dir + name +"{0}.wav".format(i),
            bitrate = "192k",
            format = "wav"
        )