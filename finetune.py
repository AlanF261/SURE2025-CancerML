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

MODEL_PATH = "/home/alanf/scratch/breastCancerDataset/SURE2025-CancerML/dummy_model"
TRAIN_CANCER_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/breast_cancer_discovery_runs.txt"
TRAIN_HEALTHY_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/healthy_discovery_runs.txt"
EVAL_CANCER_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/breast_cancer_discovery_runs.txt"
EVAL_HEALTHY_FILE = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/healthy_discovery_runs.txt"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, logits[:, 1])
        return {"accuracy": accuracy, "roc_auc": roc_auc}
    else:
        return {"accuracy": accuracy}

def plot_roc_curve(logits, labels):
    fpr, tpr, thresholds = roc_curve(labels, logits[:, 1])
    roc_auc = auc(fpr, tpr)

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
    config = Config.from_pretrained(os.path.join(MODEL_PATH, "config.json"), num_labels=2)
    tokenizer = Tokenizer(os.path.join(MODEL_PATH, "tokenizer_config.json"))

    model = CancerClassificationModel.from_pretrained(MODEL_PATH, config=config)

    train_dataset = BCClassificationDataset(
        tokenizer=tokenizer,
        filepath_class0=TRAIN_HEALTHY_FILE,
        filepath_class1=TRAIN_CANCER_FILE,
        block_size=12000
    )
    eval_dataset = BCClassificationDataset(
        tokenizer=tokenizer,
        filepath_class0=EVAL_HEALTHY_FILE,
        filepath_class1=EVAL_CANCER_FILE,
        block_size=12000
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="/home/alanf/scratch/breastCancerDataset/finetune_outputs/finetuned_model_checkpoints",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir="/home/alanf/scratch/breastCancerDataset/finetune_outputs/finetune_logs",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    np.save("finetune_predictions.npy", logits)
    np.save("finetune_labels.npy", labels)


    plot_roc_curve(logits, labels)

if __name__ == "__main__":
    main()