# Code adapted from newer MethylBERT

import pysam
import json

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
        added_tokens = self.tokenizer_config.get("added_tokens", [])
        self.special_tokens_map = {token["content"]: token["id"] for token in added_tokens if token.get("special")}

        self.unk_token = "[UNK]"
        self.unk_token_id = added_tokens.get(self.unk_token, 0)
        self.cls_token = "[CLS]"
        self.cls_token_id = self.special_tokens_map.get(self.cls_token, 1)
        self.sep_token = "[SEP]"
        self.sep_token_id = self.special_tokens_map.get(self.sep_token, 2)

        self.single_template = self.tokenizer_config.get("post_processor", {}).get("single", [])
        self.pair_template = self.tokenizer_config.get("post_processor", {}).get("pair", [])

    def _generate_default_methylation_vocab(self, max_len=16):
        self.methylation_vocab = {}
        current_id = 0

        self.methylation_vocab[current_id] = '0'
        current_id += 1
        self.methylation_vocab[current_id] = '1'
        current_id += 1

        vocab = ['0', '1']

        for length in range(2, max_len + 1):
            vocab_additions = []
            for previous_vocab in vocab:
                add_0 = previous_vocab + '0'
                add_1 = previous_vocab + '1'

                if add_0 not in self.methylation_vocab.values():
                    self.methylation_vocab[add_0] = current_id
                    current_id += 1
                vocab_additions.append(add_0)

                if add_1 not in self.methylation_vocab.values():
                    self.methylation_vocab[add_1] = current_id
                    current_id += 1
                vocab_additions.append(add_1)

            vocab = [combo for combo in vocab_additions if len(combo) <= max_len]

    def tokenize(self, seq: str):
        converted_seq = list()
        methyl_seq = list()

        for char in seq:
            token = char
            if token in ['Z', 'X', 'H', 'U']: # Use 'in' for cleaner checking
                m = 1
            else:
                m = 0

            # six alphabets indicating cytosine methylation in bismark processed files
            # token = re.sub("[h|H|z|Z|x|X]", "C", token)
            converted_seq.append(token)
            methyl_seq.append(str(m))

        return converted_seq, methyl_seq

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
            methylation_ids.append(self.methylation_vocab.get(m_token, self.unk_token_id))

        return word_token_ids, methylation_ids

    def extract_methylated_sequence(bam_file_path):
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
