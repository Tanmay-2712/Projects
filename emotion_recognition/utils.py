import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes, dataset_name):
    """
    Plot confusion matrix for a given dataset.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name.lower()}_confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names, dataset_name):
    """
    Plot feature importance for a given model (if applicable).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importances - {dataset_name}")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name.lower()}_feature_importance.png')
        plt.close()
    else:
        print(f"Feature importance not available for the model used with {dataset_name}")

def save_model_summary(model, dataset_name):
    """
    Save model summary to a text file.
    """
    with open(f'results/{dataset_name.lower()}_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))