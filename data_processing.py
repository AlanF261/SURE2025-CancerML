# Code adapted from newer MethylBERT

import pysam
import json
from typing import Optional, List
from transformers import PreTrainedTokenizer
import os


class Tokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_config_path, **kwargs):
        try:
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                self.tokenizer_config = json.load(f)
        except Exception as e:
            print(f"Error with loading tokenizer config: {e}")

        special_tokens_map = {token["content"]: token["id"] for token in self.tokenizer_config.get("added_tokens", []) if token.get("special")}

        special_tokens = {
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
        }

        for token_content, token_id in special_tokens_map.items():
            if token_content == "[UNK]":
                special_tokens["unk_token"] = token_content
            elif token_content == "[CLS]":
                special_tokens["cls_token"] = token_content
            elif token_content == "[SEP]":
                special_tokens["sep_token"] = token_content
            elif token_content == "[PAD]":
                special_tokens["pad_token"] = token_content


        self.config = self.tokenizer_config
        self.vocab = self.tokenizer_config.get("model", {}).get("vocab", {})
        self.merges = [tuple(merge.split(" ")) for merge in self.tokenizer_config.get("model", {}).get("merges", [])]
        self.added_tokens = self.tokenizer_config.get("added_tokens", [])

        super().__init__(**special_tokens, **kwargs)

        self.meth_pad_id = 0  # hard coded for now, fix later
        self.max_age_embeddings = self.tokenizer_config.get("max_age_embeddings", 100)
        self.age_unk_id = self.tokenizer_config.get("age_unk_id", 0)

        self.unk_token_id = self.vocab.get(self.unk_token, self.unk_token_id)
        self.cls_token_id = self.vocab.get(self.cls_token, self.cls_token_id)
        self.sep_token_id = self.vocab.get(self.sep_token, self.sep_token_id)
        self.pad_token_id = self.vocab.get(self.pad_token, self.pad_token_id)

        self.methylation_vocab = self._generate_default_methylation_vocab()
        # try:
        #     with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        #         self.tokenizer_config = json.load(f)
        # except Exception as e:
        #     print(f"Error with loading tokenizer config: {e}")
        #
        # self.config = self.tokenizer_config
        # self.vocab = self.tokenizer_config.get("model", {}).get("vocab", {})
        # self.merges = [tuple(merge.split(" ")) for merge in self.tokenizer_config.get("model", {}).get("merges", {})]
        # self.added_tokens = self.tokenizer_config.get("added_tokens", [])
        # # self.special_tokens_map = {token["content"]: token["id"] for token in self.added_tokens if token.get("special")}
        #
        # self.unk_token = "[UNK]"
        # self.unk_token_id = self.special_tokens_map.get(self.unk_token, 0)
        # self.cls_token = "[CLS]"
        # self.cls_token_id = self.special_tokens_map.get(self.cls_token, 1)
        # self.sep_token = "[SEP]"
        # self.sep_token_id = self.special_tokens_map.get(self.sep_token, 2)
        # self.pad_token = "[PAD]"
        # self.pad_token_id = self.special_tokens_map.get(self.pad_token, 3)
        # self.meth_pad_id = 0  # hard coded for now, fix later
        #
        # # self.single_template = self.tokenizer_config.get("post_processor", {}).get("single", [])
        # # self.pair_template = self.tokenizer_config.get("post_processor", {}).get("pair", [])
        #
        # self.max_age_embeddings = self.tokenizer_config.get("max_age_embeddings", 100)
        # self.age_unk_id = self.tokenizer_config.get("age_unk_id", 0)
        #
        # self.methylation_vocab = self._generate_default_methylation_vocab()

    def _generate_default_methylation_vocab(self, max_len=16):
        self.methylation_vocab = {}
        current_id = 1

        self.methylation_vocab[current_id] = '0'
        current_id += 1
        self.methylation_vocab[current_id] = '1'
        current_id += 1

        current_combinations = ['0', '1']

        for length in range(2, max_len + 1):
            vocab_additions = []
            for previous_vocab in current_combinations:
                add_0 = previous_vocab + '0'
                add_1 = previous_vocab + '1'

                if add_0 not in self.methylation_vocab.values():
                    self.methylation_vocab[add_0] = current_id
                    # self.methylation_vocab[current_id] = add_0
                    current_id += 1
                    # current_id += 1
                vocab_additions.append(add_0)

                if add_1 not in self.methylation_vocab.values():
                    self.methylation_vocab[add_1] = current_id
                    # self.methylation_vocab[current_id] = add_1
                    current_id += 1
                    # current_id += 1
                vocab_additions.append(add_1)

            current_combinations = [combo for combo in vocab_additions if len(combo) <= max_len]
        return self.methylation_vocab

    def extract_methylated_sequence(self, bam_file_path):
        # Z, X, H, U for methylated; z, x, h, u for unmethylated
        dna_sequences = []
        methylation_tags = []

        try:
            with pysam.AlignmentFile(bam_file_path, "rb") as samfile:
                for read in samfile.fetch():
                    if read.has_tag('XM'):
                        dna_sequences.append(read.query_sequence)
                        methylation_tags.append(read.get_tag('XM'))
        except Exception as e:
            print(f"Error processing BAM file: {e}")

        print(f"Number of sequences extracted: {len(dna_sequences)}")
        print(f"Number of methylation tags extracted: {len(methylation_tags)}")
        return dna_sequences, methylation_tags
        # return "".join(dna_sequences), "".join(methylation_tags)

    def convert_meth(self, meth_seq: List):
        methyl_seq = list()

        for char in meth_seq:
            token = char
            if token in ['Z', 'X', 'H', 'U']:
                m = 1
            # elif token in ['z', 'x', 'h', 'u']: #fix for cpg representation
            #     m = 0
            else:  # Implementing full bit approach because it contains positional information, and
                # shouldn't have significant impacts on efficiency
                m = 0

            # six alphabets indicating cytosine methylation in bismark processed files
            # token = re.sub("[h|H|z|Z|x|X]", "C", token)
            # converted_seq.append(token)
            methyl_seq.append(str(m))

        return methyl_seq

    # def bpe_merge(self, tokens: list, methyl_seq: list):
    #
    #     tokens_copy = list(tokens)
    #     methyl_copy = list(methyl_seq)
    #
    #     while True:
    #         best_merge = None
    #
    #         for merge_idx, (p1, p2) in enumerate(self.merges):
    #             for i in range(len(tokens_copy) - 1):
    #                 if tokens_copy[i] == p1 and tokens_copy[i+1] == p2:
    #                     if best_merge is None or merge_idx < best_merge[0]:
    #                         best_merge = (merge_idx, i)
    #
    #         if best_merge is None:
    #             break
    #
    #         _, i = best_merge
    #
    #         merged_token = "".join(tokens_copy[i:i+2])
    #         tokens_copy[i:i+2] = [merged_token]
    #
    #         merged_methyl = "".join(methyl_copy[i:i+2])
    #         methyl_copy[i:i+2] = [merged_methyl]
    #
    #     final_tokens = []
    #     final_methyl = []
    #     for i, token in enumerate(tokens_copy):
    #         if token not in self.vocab:
    #             final_tokens.append(self.unk_token)
    #         else:
    #             final_tokens.append(token)
    #         final_methyl.append(methyl_copy[i])
    #
    #     return final_tokens, final_methyl

    def bpe_merge(self, tokens: list, methyl_seq: list):
        if not tokens:
            return [], []

        tokens_copy = list(tokens)
        methyl_copy = list(methyl_seq)

        for merge in self.merges:  # Process merges in order
            new_tokens = []
            new_methyl = []
            i = 0

            while i < len(tokens_copy):
                if (i < len(tokens_copy) - 1 and
                    tokens_copy[i] == merge[0] and
                    tokens_copy[i + 1] == merge[1]):

                    new_tokens.append(merge[0] + merge[1])
                    new_methyl.append(methyl_copy[i] + methyl_copy[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens_copy[i])
                    new_methyl.append(methyl_copy[i])
                    i += 1

            tokens_copy = new_tokens
            methyl_copy = new_methyl

        final_tokens = [token if token in self.vocab else self.unk_token for token in tokens_copy]
        return final_tokens, methyl_copy

    def methyl_encode(self, sequence_tokens: list, methylation_tokens: list):
        word_token_ids = []
        for token in sequence_tokens:
            word_token_ids.append(self.vocab.get(token, self.unk_token_id))

        methylation_ids = []
        for m_token in methylation_tokens:
            methylation_ids.append(self.methylation_vocab.get(m_token, self.meth_pad_id))

        return word_token_ids, methylation_ids

    def _get_age_id(self, age: Optional[int], sequence_length: int) -> List[int]:
        if age is None or not (0 <= age < self.max_age_embeddings):
            age_id_value = self.age_unk_id
        else:
            age_id_value = age

        return [age_id_value] * sequence_length

    def process_bam(self, bam_file_path, age: Optional[int] = None):

        sequence, methylation_seq = self.extract_methylated_sequence(bam_file_path)

        raw_methylation_tokens = self.convert_meth(methylation_seq)

        merged_sequence_tokens, merged_methylation_tokens = self.bpe_merge(sequence, raw_methylation_tokens)

        input_ids, methylation_ids = self.methyl_encode(
            merged_sequence_tokens, merged_methylation_tokens
        )

        age_ids = self._get_age_id(age, len(input_ids))

        print(f"Tokenized output for {bam_file_path}:")
        print(f"  input_ids length: {len(input_ids)}")
        print(f"  methylation_ids length: {len(methylation_ids)}")
        print(f"  age_ids length: {len(age_ids)}")

        return {
            "input_ids": input_ids,
            "methylation_ids": methylation_ids,
            "age_ids": age_ids,
        }

    def __call__(self, *args, **kwargs):
        if "bam_file_path" in kwargs:
            bam_file_path = kwargs.pop("bam_file_path")
        elif len(args) > 0:
            bam_file_path = args[0]
            args = args[1:]
        else:
            raise ValueError("A 'bam_file_path' must be provided.")

        age = kwargs.pop("age", None)
        return self.process_bam(bam_file_path, age)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        output_file_path = os.path.join(save_directory, f"{filename_prefix or ''}vocab.json")
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        return (output_file_path,)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


def get_pairs(tokens):
    pairs = set()
    prev_token = tokens[0]
    for token in tokens[1:]:
        pairs.add((prev_token, token))
        prev_token = token
    return pairs

