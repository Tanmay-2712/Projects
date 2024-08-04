import os
import numpy as np
import librosa
import logging
from sklearn.model_selection import train_test_split

def extract_features(file_path, max_pad_len=300):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif pad_width < 0:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        logging.error(f"Error encountered while parsing file: {file_path}")
        logging.error(f"Error message: {str(e)}")
        return None

def load_ravdess_data(data_path):
    features = []
    labels = []
    
    for subdir, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                mfccs = extract_features(file_path)
                if mfccs is not None and mfccs.shape == (40, 300):
                    features.append(mfccs)
                    parts = file.split('-')
                    emotion = int(parts[2]) - 1
                    labels.append(emotion)
    
    return np.array(features), np.array(labels)

def load_tess_data(data_path):
    features = []
    labels = []
    
    emotion_map = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'neutral': 4, 'ps': 5, 'sad': 6
    }
    
    for emotion in emotion_map:
        emotion_dir = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_dir):
            logging.warning(f"Directory not found: {emotion_dir}")
            continue
        for file in os.listdir(emotion_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_dir, file)
                mfccs = extract_features(file_path)
                if mfccs is not None and mfccs.shape == (40, 300):
                    features.append(mfccs)
                    labels.append(emotion_map[emotion])
    
    return np.array(features), np.array(labels)

def load_emodb_data(data_path):
    features = []
    labels = []
    
    emotion_map = {
        'W': 0, 'L': 1, 'E': 2, 'A': 3, 'F': 4, 'T': 5, 'N': 6
    }
    
    for file in os.listdir(data_path):
        if file.endswith('.wav'):
            file_path = os.path.join(data_path, file)
            mfccs = extract_features(file_path)
            if mfccs is not None and mfccs.shape == (40, 300):
                features.append(mfccs)
                emotion = file[5]
                labels.append(emotion_map[emotion])
    
    return np.array(features), np.array(labels)

def load_dataset(dataset_name, data_path):
    if dataset_name == 'RAVDESS':
        return load_ravdess_data(data_path)
    elif dataset_name == 'TESS':
        return load_tess_data(data_path)
    elif dataset_name == 'EMODB':
        return load_emodb_data(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    return X_train, X_test, y_train, y_test