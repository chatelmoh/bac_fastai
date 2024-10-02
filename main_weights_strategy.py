import os
import preprocessing
import data
import train
import evaluation
from fastai.vision.all import *
import torch

def main():
    # Preprocessing
    bac_minus_folder = 'NEGATIVOS'
    bac_plus_folder = 'POSITIVOS'
    all_paths, all_labels = preprocessing.load_and_process_images(bac_minus_folder, bac_plus_folder)

    # Data Preparation
    dls, train_df, valid_df, test_df = data.prepare_data(all_paths, all_labels)
    class_weights = data.get_class_weights(dls)

    # Train Model
    learn = train.train_model(dls, class_weights)

    # Evaluate Model
    evaluation.plot_metrics(learn)
    evaluation.evaluate_model(learn, dls, test_df)

    # Save Model
    evaluation.save_model(learn, 'weighted_vgg.pkl')

if __name__ == "__main__":
    main()
