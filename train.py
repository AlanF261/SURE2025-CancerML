import torch
# import os
from transformers import TrainingArguments, Trainer, default_data_collator

from config import Config
from bc_predictor import MaskedLM
from data_processing import Tokenizer
from dataset import LineByLineTextDataset

config_path = "./config.py"
tokenizer_config_path = "./tokenizer_config.json"

model_config = Config()

tokenizer = Tokenizer(tokenizer_config_path)

model = MaskedLM(model_config)

# if hasattr(model,'lm_head') and hasattr(model.lm_head,'decoder') and hasattr(model.ntv2.embeddings,'word_embeddings'):
#     model.lm_head.decoder.weight = model.ntv2.embeddings.word_embeddings.weight

input_filepath = ("/home/alanf/scratch/breastCancerDataset/aligned_sequences/"
                  "encode_aligned/wgEncodeHaibMethylRrbsPanisletsDukeRawDataRep1_bismark_bt2.bam")

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    filepath=input_filepath,
    block_size=12000
)


def create_scheduler(optimizer, num_warmup_steps, num_train_steps, lr_start, lr_end_decay=1e-5):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress_after_warmup = (current_step - num_warmup_steps) / (num_train_steps - num_warmup_steps)
        decay_factor = max(0.0, 1.0 - progress_after_warmup)**0.5
        return max(lr_end_decay / lr_start, decay_factor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    report_to="none",
    learning_rate=5e-5,
    lr_scheduler_type="lambda_lr",
    warmup_steps=0,
)

num_training_steps = (int(len(train_dataset) / training_args.per_device_train_batch_size) *
                      training_args.num_train_epochs)
warmup_steps = int(num_training_steps * 0.1)

training_args.warmup_steps = warmup_steps

def custom_scheduler(optimizer, num_training_steps):
    return create_scheduler(
        optimizer,
        num_warmup_steps=training_args.warmup_steps, # Use the updated warmup_steps
        num_train_steps=num_training_steps,
        lr_start=training_args.learning_rate,
        lr_end_decay=1e-8
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)

print("Starting training.")
trainer.train()
print("Finished training.")

trainer.save_model("./dummy_model")
# tokenizer.save_pretrained("./dummy_model")