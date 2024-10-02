from preprocessing import load_image_paths, undersample_classes, split_data
from data import get_dataloaders, get_test_dataloader
from train import create_learner, train_model, save_model
from evaluation import evaluate_model, plot_training_metrics, plot_roc_curve, plot_confusion_matrix

def main():
    # Definir las carpetas de las imágenes
    bac_minus_folder = 'NEGATIVOS'
    bac_plus_folder = 'POSITIVOS'

    # Cargar las imágenes y aplicar undersampling
    df = load_image_paths(bac_minus_folder, bac_plus_folder)
    undersampled_df = undersample_classes(df)

    # Dividir los datos en entrenamiento, validación y prueba
    train_df, valid_df, test_df = split_data(undersampled_df)

    # Obtener los DataLoaders para entrenamiento y validación
    dls = get_dataloaders(train_df, valid_df)

    # Crear el learner y entrenar el modelo
    learn = create_learner(dls)
    train_model(learn)
    
    # Graficar las métricas de entrenamiento
    plot_training_metrics(learn)
    
    # Guardar el modelo entrenado
    save_model(learn, model_name='undersampling_vgg.pkl')

    # Evaluar el modelo en el conjunto de prueba
    test_dl = get_test_dataloader(test_df)
    preds, targs, pred_labels = evaluate_model(learn, test_dl)
    
    # Graficar la curva ROC y la matriz de confusión
    plot_roc_curve(targs, preds)
    plot_confusion_matrix(targs, pred_labels, dls)

if __name__ == "__main__":
    main()
