import os
import numpy as np
from tqdm import tqdm
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from constants import sampling_rate, random_state, test_size, directory, validation_size
from extract_feature import extract_features
from get_label_from_file_name import get_label_from_file_name

def remove_silence(audio, top_db=20):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    if len(non_silent_intervals) == 0:
        return np.array([])
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio

def process_audio_file(file_name):
    label = get_label_from_file_name(file_name)
    if label is None:
        print(f"Label not found for file: {file_name}")
        return None, None

    audio, sr = librosa.load(file_name, sr=sampling_rate)
    # audio = remove_silence(audio)

    if len(audio) == 0:
        print(f"Audio is silent after removing silence for file: {file_name}")
        return None, None

    features = extract_features(audio, sr)

    return features, label

def load_dataset(directory):
    features_list = []
    labels_list = []

    for root, _, files in os.walk(directory):
        for file_name in tqdm(files):
            if file_name.endswith((".wav",".flac")):
                file_path = os.path.join(root, file_name)
                features, label = process_audio_file(file_path)
                if features is not None and label is not None:
                    features_list.append(features)
                    labels_list.append(label)
                else:
                    print(f"Skipping file: {file_name}")

    if not features_list or not labels_list:
        raise ValueError("No features or labels were loaded.")

    return features_list, labels_list

def prepare_dataset():
    try:
        # Load dataset
        X_list, y = load_dataset(directory)

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Pad sequences to the same length
        max_length = max([x.shape[0] for x in X_list])
        print(f"Maximum sequence length: {max_length}")

        X_padded = []
        for i, x in enumerate(X_list):
            if x.shape[0] < max_length:
                pad_width = max_length - x.shape[0]
                x_padded = np.pad(x, ((0, pad_width), (0, 0)), mode='constant')
                print(f"Padded sample {i} with zeros: {pad_width} time steps")
            else:
                x_padded = x
            X_padded.append(x_padded)

        X = np.stack(X_padded)  # Shape: (n_samples, time_steps, features)

        # Stratified train/test/val split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=(test_size + validation_size),
            random_state=random_state, stratify=y_encoded
        )

        val_size_relative = validation_size / (test_size + validation_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size_relative,
            random_state=random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, le

    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        return None, None, None, None, None, None, None