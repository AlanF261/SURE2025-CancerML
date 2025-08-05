import torch
from torch.utils.data import Dataset
import os
import collections
from collections.abc import Mapping

from data_processing import Tokenizer


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, filepath: str, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        if os.path.isfile(filepath):
            if filepath.endswith(".txt"):
                with open(filepath, "r") as f:
                    bam_files = [line.strip() for line in f if line.strip()]
            elif filepath.endswith(".bam"):
                bam_files = [filepath]
            else:
                raise ValueError(f"{filepath} must be .bam or .txt")
        else:
            raise ValueError(f"{filepath} not found.")

        cache_signature = "_".join(sorted([os.path.basename(f) for f in bam_files]))
        cache_hash = str(hash(cache_signature))[:10]

        cache_dir = os.path.dirname(bam_files[0]) if bam_files else "."
        cached_features_file = os.path.join(
            cache_dir,
            f"cached_lm_{self.block_size}_{tokenizer.pad_token}_no_mlm_{cache_hash}.pt",
        )

        force_reprocessing = False

        if os.path.exists(cached_features_file) and not force_reprocessing:
            print(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = torch.load(handle)
        else:
            self.examples = []

            for bam_file_path in bam_files:
                if not os.path.isfile(bam_file_path):
                    print(f"File missing: {bam_file_path}.")
                    continue

                print(f"Processing {bam_file_path}...")
                tokenized_data = self.tokenizer(bam_file_path)

                input_ids_list = tokenized_data["input_ids"]
                methylation_ids_list = tokenized_data["methylation_ids"]
                age_ids_list = tokenized_data["age_ids"]

                for i in range(len(input_ids_list)):
                    input_ids = input_ids_list[i]
                    methylation_ids = methylation_ids_list[i]
                    age_ids = age_ids_list[i]

                    current_len = len(input_ids)
                    if current_len < self.block_size:
                        padding_len = self.block_size - current_len
                        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
                        methylation_ids = methylation_ids + [self.tokenizer.meth_unk_id] * padding_len
                        age_ids = age_ids + [self.tokenizer.age_unk_id] * padding_len
                    elif current_len > self.block_size:
                        input_ids = input_ids[:self.block_size]
                        methylation_ids = methylation_ids[:self.block_size]
                        age_ids = age_ids[:self.block_size]

                    attention_mask = [1] * min(current_len, self.block_size) + [0]*max(0, self.block_size - current_len)

                    self.examples.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "methylation_ids": torch.tensor(methylation_ids, dtype=torch.long),
                        "age_ids": torch.tensor(age_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    })

            with open(cached_features_file, "wb") as handle:
                torch.save(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> collections.abc.Mapping[str, torch.Tensor]:
        return self.examples[i]
