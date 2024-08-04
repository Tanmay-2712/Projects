```markdown
# Emotion Recognition for Car Safety: A Comparative Study of Machine Learning Models

## Table of Contents
1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Dataset Information](#dataset-information)
4. [Feature Extraction](#feature-extraction)
5. [Model Architectures](#model-architectures)
6. [Implementation Details](#implementation-details)
7. [Results and Visualizations](#results-and-visualizations)
8. [How to Run](#how-to-run)
9. [Performance Analysis](#performance-analysis)
10. [License](#license)

## Project Overview

This project implements and compares various machine learning models for emotion recognition in the context of car safety. The main goal is to analyze and compare the performance of different models, with a primary focus on a CNN-LSTM architecture, across multiple datasets.

### Key Features:

1. Implementation of multiple machine learning models:
   - CNN-LSTM (primary focus)
   - CNN
   - Support Vector Machine (SVM)
   - Random Forest (RF)
   - Logistic Regression (LR)

2. Evaluation across multiple datasets:
   - RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - TESS (Toronto Emotional Speech Set)
   - EMODB (Berlin Database of Emotional Speech)

3. Comprehensive performance analysis and visualization

## Motivation

Emotion recognition plays a crucial role in enhancing car safety systems. By detecting the emotional state of drivers or passengers, vehicles can potentially adjust their responses or alert systems accordingly. This project aims to explore and compare the effectiveness of various machine learning techniques in accurately recognizing emotions from audio data, which could be integrated into advanced driver assistance systems (ADAS) or autonomous vehicle technologies.

## Dataset Information

The project utilizes three different datasets for emotion recognition:

1. **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
   - Contains 1440 audio files
   - 24 professional actors (12 female, 12 male)
   - Emotions: calm, happy, sad, angry, fearful, surprise, and disgust
   - [RAVDESS Dataset Link](https://zenodo.org/record/1188976)

2. **TESS (Toronto Emotional Speech Set)**
   - Contains 2800 audio files
   - Two actresses (aged 26 and 64 years)
   - Emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral
   - [TESS Dataset Link](https://tspace.library.utoronto.ca/handle/1807/24487)

3. **EMODB (Berlin Database of Emotional Speech)**
   - Contains 535 audio files
   - 10 actors (5 female, 5 male)
   - Emotions: anger, boredom, disgust, fear, happiness, sadness, and neutral
   - [EMODB Dataset Link](http://emodb.bilderbar.info/download/)

## Feature Extraction

The project uses Mel-frequency cepstral coefficients (MFCCs) as the primary features for emotion recognition. The feature extraction process includes:

1. Loading audio files using librosa
2. Extracting 40 MFCCs from each audio sample
3. Padding or truncating MFCC features to ensure uniform length (300 frames)

The `extract_features` function in `emotion_datasets_loader.py` handles this process:

```python
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
```

## Model Architectures

### CNN-LSTM
A combination of Convolutional Neural Network (CNN) layers for feature extraction and Long Short-Term Memory (LSTM) layers for sequence modeling.

```python
def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

### CNN
A standard Convolutional Neural Network architecture.

```python
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

### SVM, RF, and LR
Traditional machine learning models implemented for comparison.

```python
def create_svm_model():
    return SVC(kernel='rbf', C=1.0, random_state=42)

def create_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def create_lr_model():
    return LogisticRegression(random_state=42, multi_class='ovr')
```

## Implementation Details

The project is structured into several Python scripts:

1. `main.py`: The main script that orchestrates the entire process.
   - Loads datasets
   - Trains and evaluates models
   - Generates visualizations

2. `emotion_datasets_loader.py`: Handles loading and preprocessing of the datasets.
   - Implements dataset-specific loading functions
   - Extracts MFCC features

3. `models.py`: Contains the implementations of different model architectures.
   - Defines CNN-LSTM, CNN, SVM, RF, and LR models

4. `utils.py`: Utility functions for plotting and saving results.
   - Implements confusion matrix plotting
   - Generates feature importance plots
   - Saves model summaries

## Results and Visualizations

The project generates various visualizations to compare model performance:

1. Confusion matrices for each model-dataset combination
2. Training history plots for neural network models (accuracy and loss)
3. Model comparison plots across different datasets and metrics (accuracy, precision, recall, F1-score)

Results are saved in the `results/` directory with the following naming convention:
- `{dataset_name}_{model_name}_confusion_matrix.png`
- `{dataset_name}_{model_name}_training_history.png`
- `model_comparison.png`

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/your_username/emotion-recognition-car-safety.git
   cd emotion-recognition-car-safety
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install numpy librosa scikit-learn tensorflow keras matplotlib seaborn
   ```

4. Organize your datasets in the following structure:
   ```
   datasets/
   ├── ravdess_dataset/
   ├── tess_dataset/
   └── emodb_dataset/
   ```

5. Run the main script:
   ```
   python main.py
   ```

6. Check the `results/` directory for output visualizations and the `logs/` directory for execution logs.

## Performance Analysis

The project provides a comprehensive analysis of model performance across different datasets and metrics. Key performance indicators include:

- Accuracy
- Precision
- Recall
- F1-score

The `plot_model_comparison` function in `main.py` generates a visualization comparing these metrics across all models and datasets.


## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

---

For any questions or issues, please open an issue on the GitHub repository.

