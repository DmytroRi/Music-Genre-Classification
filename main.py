import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

ALL_GENRES = ["blues", "classical", "country", "disco", "hip-hop", "jazz", "metal", "pop", "reggae", "rock"]
PATH_TO_WAV = "test_music/Pantera__Walk.wav"#"enter/your/path/to/audio"
SAMPLE_RATE = 25050
SEGMENT_DURATION = 6  # Duration of each segment in seconds (as used in training)
NUM_SEGMENTS = 1000  # Number of segments to extract

def get_mfccs (path_to_music, n_fft = 2048, n_mfcc = 13, hop_length = 512):
    # Load the audio file
    signal, _ = librosa.load(path_to_music, sr=SAMPLE_RATE)
    segment_samples  = int(SEGMENT_DURATION * SAMPLE_RATE)
    
    # Split the signal into consistent 6-second segments
    segments = []
    for start_sample in range(0, len(signal), segment_samples):
        end_sample = start_sample + segment_samples 
        if end_sample > len(signal):
            break
        segment = signal[start_sample : end_sample]
        segments.append(segment)

    # Shuffle the segments to get different parts of the song for analysis
    np.random.shuffle(segments)    

    num_segments_to_process = min(NUM_SEGMENTS, len(segments))
    # Initialize an array to store MFCCs of fixed shape (num_segments_to_process, 130, 13)
    mfccs_array = np.empty((num_segments_to_process, 130, 13))
    
    # Calculate MFCCs for each of the selected segments
    for i, segment in enumerate(segments[:num_segments_to_process]):
        mfcc = librosa.feature.mfcc(y = segment,
                                    sr = SAMPLE_RATE,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length)
        mfcc = mfcc.T
        if mfcc.shape[0] < 130:
            mfcc_padded = np.pad(mfcc, ((0, 130 - mfcc.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc_padded = mfcc[:130, :]
        mfccs_array[i] = mfcc_padded
    
    return mfccs_array

def decode_genre(index):
    genres = ["blues", "classical", "country", "disco", "hip-hop", "jazz", "metal", "pop", "reggae", "rock"]
    return genres[index] if 0 <= index < len(genres) else "unknown"

def plot_array(array):
    # Count the occurrences of each genre
    genre_counts = Counter(array)

    # Prepare data for the plot
    counts = [genre_counts.get(genre, 0) for genre in ALL_GENRES]
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(ALL_GENRES, counts, color='skyblue')
    plt.xlabel('Genres')
    plt.ylabel('Count')
    plt.title('Genre Distribution in Predictions ({})'.format(PATH_TO_WAV.split('/')[-1]))
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mfccs_inputs = get_mfccs(PATH_TO_WAV)

    model = tf.keras.models.load_model('saved_models/GenreClassification_CNN.keras')
    #model = tf.keras.models.load_model('saved_models/GenreClassification_MLP.keras')
    #model = tf.keras.models.load_model('saved_models/GenreClassification_RNN_LSTM.keras')
    predictions_list = []
    for mfcc_segment in mfccs_inputs:
        mfcc_segment = mfcc_segment[np.newaxis, ...]  # Adds batch dimension -> (1, 130, 13, 1)
        predictions = model.predict(mfcc_segment)
        genre_index = np.argmax(predictions, axis=1)
        predicted_genre = decode_genre(genre_index[0])
        predictions_list.append(predicted_genre)

    print(predictions_list)
    plot_array(predictions_list)