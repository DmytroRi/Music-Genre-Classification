import math
import os
import librosa
import json

DATASET_PATH = "archive/Data/genres_original"
JSON_PATH = "processed/music_genres.json"
SAMPLE_RATE = 25050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    print("Execution of save_mfcc funtcion has started.\n")
    # dictionary to store data
    data = {
        "mapping":  [],
        "mfcc":     [],
        "labels":   []
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    #look through all the genres 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            #process files for a specific genre
            print("Processing {}\n". format(semantic_label))
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # propcess sefmants extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    mfcc = librosa.feature.mfcc(y = signal[start_sample : finish_sample],
                                                sr = sample_rate,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)
                    
                    #store mfcc for segment if has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("The end of the execution.\n")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)