import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from fastai.vision.all import *

def plot_training_metrics(learn):
    """Grafica las pérdidas de entrenamiento, precisión, recall y F1-score."""
    train_losses = [x[0] for x in learn.recorder.values]
    valid_losses = [x[1] for x in learn.recorder.values]
    accuracies = [x[2] for x in learn.recorder.values]
    precision_scores = [x[3] for x in learn.recorder.values]
    recall_scores = [x[4] for x in learn.recorder.values]
    f1_scores = [x[5] for x in learn.recorder.values]
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'r', label='Train Loss')
    plt.plot(epochs, valid_losses, 'b', label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracies, 'b', label='Accuracy')
    plt.plot(epochs, precision_scores, 'g', label='Precision')
    plt.plot(epochs, recall_scores, 'm', label='Recall')
    plt.plot(epochs, f1_scores, 'k', label='F1-score')
    plt.title('Accuracy, Precision, Recall and F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def evaluate_model(learn, test_dl):
    """Evalúa el modelo en el conjunto de prueba y calcula las métricas."""
    preds, targs = learn.get_preds(dl=test_dl)
    pred_labels = preds.argmax(dim=1)

    accuracy = accuracy_score(targs, pred_labels)
    precision = precision_score(targs, pred_labels, average='weighted')
    recall = recall_score(targs, pred_labels, average='weighted')
    f1 = f1_score(targs, pred_labels, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return preds, targs, pred_labels

def plot_roc_curve(targs, preds):
    """Genera la curva ROC."""
    probs = preds[:, 1]
    fpr, tpr, _ = roc_curve(targs, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(targs, pred_labels, dls):
    """Genera la matriz de confusión."""
    cm = confusion_matrix(targs, pred_labels)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dls.vocab, yticklabels=dls.vocab)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
