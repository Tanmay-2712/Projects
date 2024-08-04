import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from emotion_datasets_loader import load_dataset, prepare_data
from models import create_cnn_lstm_model, create_cnn_model, create_svm_model, create_rf_model, create_lr_model

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join('logs', 'emotion_recognition.log')),
                        logging.StreamHandler()
                    ])

def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    if model_name in ['CNN-LSTM', 'CNN']:
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        model.fit(X_train_flat, y_train)
        y_pred_classes = model.predict(X_test_flat)
        history = None

    return {
        'accuracy': accuracy_score(y_test, y_pred_classes),
        'precision': precision_score(y_test, y_pred_classes, average='weighted'),
        'recall': recall_score(y_test, y_pred_classes, average='weighted'),
        'f1': f1_score(y_test, y_pred_classes, average='weighted')
    }, history, y_pred_classes

def plot_confusion_matrix(y_true, y_pred, classes, dataset_name, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {dataset_name} - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'{dataset_name.lower()}_{model_name.lower()}_confusion_matrix.png'))
    plt.close()

def plot_training_history(history, dataset_name, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{dataset_name} - {model_name} - Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{dataset_name} - {model_name} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('results', f'{dataset_name.lower()}_{model_name.lower()}_training_history.png'))
    plt.close()

def plot_model_comparison(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Model Comparison Across Datasets', fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        for dataset in datasets:
            values = [results[dataset][model][metric] for model in models]
            ax.bar(np.arange(len(models)) + (datasets.index(dataset) * 0.2), values, width=0.2, label=dataset)
        
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(np.arange(len(models)) + 0.2)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('results', 'model_comparison.png'))
    plt.close()

def main():
    datasets = ['RAVDESS', 'TESS', 'EMODB']
    models = {
        'CNN-LSTM': create_cnn_lstm_model,
        'CNN': create_cnn_model,
        'SVM': create_svm_model,
        'RF': create_rf_model,
        'LR': create_lr_model
    }
    results = {}

    for dataset in datasets:
        results[dataset] = {}
        try:
            logging.info(f"Processing {dataset} dataset...")
            X, y = load_dataset(dataset, os.path.join('datasets', f'{dataset.lower()}_dataset'))
            X_train, X_test, y_train, y_test = prepare_data(X, y)

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            for model_name, model_func in models.items():
                logging.info(f"Training {model_name} on {dataset}...")
                if model_name in ['CNN-LSTM', 'CNN']:
                    model = model_func((X_train.shape[1], X_train.shape[2], 1), len(le.classes_))
                    results[dataset][model_name], history, y_pred = train_evaluate_model(
                        model, X_train, y_train_encoded, X_test, y_test_encoded, model_name
                    )
                    plot_training_history(history, dataset, model_name)
                else:
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    model = model_func()
                    results[dataset][model_name], _, y_pred = train_evaluate_model(
                        model, X_train_flat, y_train_encoded, X_test_flat, y_test_encoded, model_name
                    )

                plot_confusion_matrix(y_test_encoded, y_pred, le.classes_, dataset, model_name)

                logging.info(f"{dataset} - {model_name} Results:")
                for metric_name, value in results[dataset][model_name].items():
                    logging.info(f"  {metric_name}: {value:.4f}")

        except Exception as e:
            logging.error(f"Error processing {dataset} dataset: {str(e)}")

    plot_model_comparison(results)
    logging.info("Model comparison visualization saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()