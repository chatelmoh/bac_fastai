from fastai.vision.all import *
from torchvision.models import vgg16_bn
import torch

def create_learner(dls, class_weights=None, metrics=[accuracy, Precision(), Recall(), F1Score()]):
    """Crea un objeto learner con un modelo preentrenado y métricas."""
    learn = cnn_learner(dls, vgg16_bn, metrics=metrics)
    
    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights)
        learn.loss_func = CrossEntropyLossFlat(weight=weight_tensor)
    
    return learn

def train_model(learn, epochs=15):
    """Entrena el modelo con los DataLoaders proporcionados y ajusta el modelo."""
    learn.fine_tune(epochs)
    return learn

def save_model(learn, model_name='weighted_vgg.pkl'):
    """Guarda el modelo entrenado en un archivo."""
    learn.export(model_name)
    print(f"Modelo guardado como {model_name}")
