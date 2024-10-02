from fastai.vision.all import *
from fastai.data.load import DataLoader

def get_dataloaders(train_df, valid_df, batch_size=8):
    """Crea los DataLoaders para entrenamiento y validación."""
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('image_path'),
        get_y=ColReader('label'),
        item_tfms=Resize(256),
        batch_tfms=aug_transforms(flip_vert=False, do_flip=False)
    )
    
    dls = dblock.dataloaders(train_df, valid_df=valid_df, bs=batch_size)
    return dls

def get_test_dataloader(test_df, batch_size=8):
    """Crea el DataLoader para el conjunto de prueba."""
    test_dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('image_path'),
        get_y=ColReader('label'),
        item_tfms=Resize(512)
    )
    
    test_dl = test_dblock.dataloaders(test_df, bs=batch_size)
    return test_dl

def get_class_weights(dls):
    """Calcula los pesos de las clases basados en el número de imágenes por clase."""
    classes = dls.vocab
    train_lbls = L(map(lambda x: classes[x[1]], dls.train_ds))
    label_counter = Counter(train_lbls)
    n_most_common_class = max(label_counter.values())
    print(f'Occurrences of the most common class {n_most_common_class}')
    weights = [n_most_common_class/v for k, v in label_counter.items() if v > 0]
    return weights
