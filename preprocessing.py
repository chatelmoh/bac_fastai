import pandas as pd
from fastai.vision.all import get_image_files
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np

def load_image_paths(bac_minus_folder, bac_plus_folder):
    """Carga los caminos de las imágenes y sus etiquetas."""
    bac_minus_paths = get_image_files(bac_minus_folder)
    bac_plus_paths = get_image_files(bac_plus_folder)
    
    bac_minus_labels = [0] * len(bac_minus_paths)
    bac_plus_labels = [1] * len(bac_plus_paths)
    
    all_paths = bac_minus_paths + bac_plus_paths
    all_labels = bac_minus_labels + bac_plus_labels
    
    df = pd.DataFrame({'image_path': all_paths, 'label': all_labels})
    return df

def undersample_classes(df):
    # Realizar undersampling de la clase mayoritaria
    bac_plus_paths = df[df['label'] == 1]['image_path'].tolist()
    bac_minus_paths = df[df['label'] == 0]['image_path'].tolist()

    undersampled_bac_minus_paths = np.random.choice(bac_minus_paths, size=len(bac_plus_paths), replace=False)
    undersampled_bac_minus_labels = [0] * len(undersampled_bac_minus_paths)

    undersampled_paths = list(undersampled_bac_minus_paths) + bac_plus_paths
    undersampled_labels = undersampled_bac_minus_labels + [1] * len(bac_plus_paths)

    undersampled_df = pd.DataFrame({'image_path': undersampled_paths, 'label': undersampled_labels})
    return undersampled_df

def balance_classes(df):
    """Aplica oversampling a las clases minoritarias."""
    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
    oversampled_paths, oversampled_labels = ros.fit_resample(df[['image_path']], df['label'])
    
    oversampled_df = pd.DataFrame({'image_path': oversampled_paths['image_path'], 'label': oversampled_labels})
    return oversampled_df

def split_data(oversampled_df, test_size=0.2, val_size=0.2, random_state=42):
    """Divide los datos en conjuntos de entrenamiento, validación y prueba."""
    train_val_df, test_df = train_test_split(
        oversampled_df, test_size=test_size, random_state=random_state, stratify=oversampled_df['label']
    )
    
    train_df, valid_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['label']
    )
    
    print(f"Total samples: {len(oversampled_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, valid_df, test_df
