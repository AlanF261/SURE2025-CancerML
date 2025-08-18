# import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

from config import Config
from bc_predictor import CancerClassificationModel
from data_processing import Tokenizer
from dataset import BCClassificationDataset

# The path to your pre-trained model directory
MODEL_PATH = "/home/alanf/scratch/breastCancerDataset/SURE2025-CancerML/dummy_model"
# Paths to your finetuning data files
TRAIN_CANCER_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/breast_cancer_validation_runs.txt"
TRAIN_HEALTHY_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/healthy_validation_runs.txt"
EVAL_CANCER_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/breast_cancer_validation_runs.txt"
EVAL_HEALTHY_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/healthy_validation_runs.txt"

def compute_metrics(eval_pred):
    """Computes accuracy and AUC for the evaluation set."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate AUC if there are at least two unique labels
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, logits[:, 1]) # ROC-AUC for class 1
        return {"accuracy": accuracy, "roc_auc": roc_auc}
    else:
        return {"accuracy": accuracy}

def plot_roc_curve(logits, labels):
    """Plots the ROC curve from predictions and labels."""
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, logits[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def main():
    # Load model configuration and tokenizer
    # We need to set num_labels for the classification head
    config = Config.from_pretrained(os.path.join(MODEL_PATH, "config.json"), num_labels=2)
    tokenizer = Tokenizer(os.path.join(MODEL_PATH, "tokenizer_config.json"))

    # Load the pre-trained model and initialize the classification head
    model = CancerClassificationModel.from_pretrained(MODEL_PATH, config=config)

    # Prepare datasets for fine-tuning
    # Use the new BCClassificationDataset
    train_dataset = BCClassificationDataset(
        tokenizer=tokenizer,
        filepath_class0=TRAIN_HEALTHY_FILE,
        filepath_class1=TRAIN_CANCER_FILE,
        block_size=12000 # Same block size as the pre-training script
    )
    eval_dataset = BCClassificationDataset(
        tokenizer=tokenizer,
        filepath_class0=EVAL_HEALTHY_FILE,
        filepath_class1=EVAL_CANCER_FILE,
        block_size=12000
    )

    # Use a DataCollatorWithPadding for classification
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="/home/alanf/scratch/breastCancerDataset/finetune_outputs/finetuned_model_checkpoints",  # Where checkpoints will be saved
        overwrite_output_dir=True,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model at the end of training
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir="/home/alanf/scratch/breastCancerDataset/finetune_outputs/finetune_logs",
        logging_steps=100,
        bf16=True,
        report_to="none",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    validation_files = []
    validation_filepaths_file = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/validation_cohort_filepaths.txt"

    try:
        with open(validation_filepaths_file, "r") as f:
            for line in f:
                filepath = line.strip()
                if filepath:
                    validation_files.append(filepath)
    except FileNotFoundError:
        print(f"Error: The file {validation_filepaths_file} was not found.")

    print("Starting per-file validation evaluation...")

    for file_path in validation_files:
        print(f"\nEvaluating file: {file_path}")

        # Create a new dataset containing only the current file
        single_file_dataset = BCClassificationDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=12000,
            model_name_or_path=model_name_or_path
        )

        metrics = trainer.evaluate(eval_dataset=single_file_dataset)

        accuracy = metrics.get('eval_accuracy')
        if accuracy is not None:
            print(f"Accuracy for {file_path}: {accuracy:.4f}")
        else:
            print("Accuracy metric not found. Check your `compute_metrics` function.")

    print("\nPer-file validation complete.")

    # --- Post-training step: Generate predictions for ROC curve ---
    print("Training complete. Generating predictions for ROC curve...")
    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    # Save the predictions and labels
    np.save("finetune_predictions.npy", logits)
    np.save("finetune_labels.npy", labels)

    print("Predictions and labels saved to finetune_predictions.npy and finetune_labels.npy.")

    # Plot the ROC curve
    plot_roc_curve(logits, labels)

if __name__ == "__main__":
    main()
