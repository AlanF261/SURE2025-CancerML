# Code adapted from newer MethylBERT

import pysam
import json
from typing import Optional, List

class Tokenizer:
    def __init__(self, tokenizer_config_path):

        try:
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                self.tokenizer_config = json.load(f)
        except Exception as e:
            print(f"Error with loading tokenizer config: {e}")

        self.config = self.tokenizer_config
        self.vocab = self.tokenizer_config.get("model", {}).get("vocab", {})
        self.merges = [tuple(merge.split(" ")) for merge in self.tokenizer_config.get("model", {}).get("merges", {})]
        self.added_tokens = self.tokenizer_config.get("added_tokens", [])
        self.special_tokens_map = {token["content"]: token["id"] for token in self.added_tokens if token.get("special")}

        self.unk_token = "[UNK]"
        self.unk_token_id = self.special_tokens_map.get(self.unk_token, 0)
        self.cls_token = "[CLS]"
        self.cls_token_id = self.special_tokens_map.get(self.cls_token, 1)
        self.sep_token = "[SEP]"
        self.sep_token_id = self.special_tokens_map.get(self.sep_token, 2)
        self.pad_token = "[PAD]"
        self.pad_token_id = self.special_tokens_map.get(self.pad_token, 3)
        self.meth_unk_id = 0 # hard coded for now, fix later

        # self.single_template = self.tokenizer_config.get("post_processor", {}).get("single", [])
        # self.pair_template = self.tokenizer_config.get("post_processor", {}).get("pair", [])

        self.max_age_embeddings = self.tokenizer_config.get("max_age_embeddings", 100)
        self.age_unk_id = self.tokenizer_config.get("age_unk_id", 0)


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
                    self.methylation_vocab[current_id] = add_0
                    current_id += 1
                vocab_additions.append(add_0)

                if add_1 not in self.methylation_vocab.values():
                    self.methylation_vocab[current_id] = add_1
                    current_id += 1
                vocab_additions.append(add_1)

            current_combinations = [combo for combo in vocab_additions if len(combo) <= max_len]

    def extract_methylated_sequence(self, bam_file_path):
        #Z, X, H, U for methylated; z, x, h, u for unmethylated
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
        return dna_sequences, methylation_tags

    def convert_meth(self, meth_seq: List):
        methyl_seq = list()

        for char in meth_seq:
            token = char
            if token in ['Z', 'X', 'H', 'U']:
                m = 1
            # elif token in ['z', 'x', 'h', 'u']: #fix for cpg representation
            #     m = 0
            else: #Implementing full bit approach because it contains positional information, and
                #shouldn't have significant impacts on efficiency
                m = 0

            # six alphabets indicating cytosine methylation in bismark processed files
            # token = re.sub("[h|H|z|Z|x|X]", "C", token)
            # converted_seq.append(token)
            methyl_seq.append(str(m))

        return methyl_seq

    def bpe_merge(self, tokens:list, methyl_seq: list):

        tokens_copy = list(tokens)
        methyl_copy = list(methyl_seq)

        while True:
            best_merge = None

            for merge_idx, (p1, p2) in enumerate(self.merges):
                for i in range(len(tokens_copy) - 1):
                    if tokens_copy[i] == p1 and tokens_copy[i+1] == p2:
                        if best_merge is None or merge_idx < best_merge[0]:
                            best_merge = (merge_idx, i)

            if best_merge is None:
                break

            _, i = best_merge

            merged_token = "".join(tokens_copy[i:i+2])
            tokens_copy[i:i+2] = [merged_token]

            merged_methyl = "".join(methyl_copy[i:i+2])
            methyl_copy[i:i+2] = [merged_methyl]

        final_tokens = []
        final_methyl = []
        for i, token in enumerate(tokens_copy):
            if token not in self.vocab:
                final_tokens.append(self.unk_token)
            else:
                final_tokens.append(token)
            final_methyl.append(methyl_copy[i])

        return final_tokens, final_methyl

    def encode(self, sequence_tokens: list, methylation_tokens: list):
        word_token_ids = []
        for token in sequence_tokens:
            word_token_ids.append(self.vocab.get(token, self.unk_token_id))

        methylation_ids = []
        for m_token in methylation_tokens:
            methylation_ids.append(self.methylation_vocab.get(m_token, self.meth_unk_id))

        return word_token_ids, methylation_ids

    def _get_age_id(self, age: Optional[int], sequence_length: int) -> List[int]:
        if age is None or not (0 <= age < self.max_age_embeddings):
            age_id_value = self.age_unk_id
        else:
            age_id_value = age

        return [age_id_value] * sequence_length

    def __call__(self, bam_file_path, age: Optional[int] = None):

        sequence, methylation_seq = self.extract_methylated_sequence(bam_file_path)

        raw_methylation_tokens = self.convert_meth(methylation_seq)

        merged_sequence_tokens, merged_methylation_tokens = self.bpe_merge(sequence, raw_methylation_tokens)

        input_ids, methylation_ids = self.encode(
            merged_sequence_tokens, merged_methylation_tokens #
        )

        age_ids = self._get_age_id(age, len(input_ids))

        # Step 6: Return the final tokenized outputs
        return {
            "input_ids": input_ids,
            "methylation_ids": methylation_ids,
            "age_ids": age_ids,
        }