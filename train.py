from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


from config import Config
from bc_predictor import MaskedLM
from data_processing import Tokenizer
from dataset import LineByLineTextDataset

config_path = "/home/alanf/scratch/breastCancerDataset/SURE2025-CancerML/config.py"
tokenizer_config_path = "/home/alanf/scratch/breastCancerDataset/SURE2025-CancerML/tokenizer_config.json"
input_filepath = "/home/alanf/scratch/breastCancerDataset/scripts/cohorts/discovery_filepaths.txt"

model_config = Config()

tokenizer = Tokenizer(tokenizer_config_path)

model = MaskedLM(model_config)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    filepath=input_filepath,
    block_size=12000
)

training_args = TrainingArguments(
    output_dir="/home/alanf/scratch/breastCancerDataset/scripts/pretrainresults",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="/home/alanf/scratch/breastCancerDataset/scripts/pretrainlogs",
    logging_steps=500,
    report_to="none",
    learning_rate=5e-5,
    lr_scheduler_type="polynomial",
    warmup_steps=0,
)

num_training_steps = (int(len(train_dataset) / training_args.per_device_train_batch_size) *
                      training_args.num_train_epochs)
warmup_steps = int(num_training_steps * 0.1)

training_args.warmup_steps = warmup_steps

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("/home/alanf/scratch/breastCancerDataset/scripts/dummy_model")
print("Success??")