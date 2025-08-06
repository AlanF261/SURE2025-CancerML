# Taken from Nucleotide Transformer v2 with modifications

from transformers import PretrainedConfig


class Config(PretrainedConfig):

    model_type = "ntv2"

    def __init__(
            self,
            vocab_size=4095,
            mask_token_id=4,
            pad_token_id=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            # attention_probs_secondary_dropout=0, #0 because nn.func already applies dropout
            # max_position_embeddings=1026,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            position_embedding_type="absolute",
            use_cache=True,
            emb_layer_norm_before=None,
            token_dropout=True,
            is_folding_model=False,
            vocab_list=None,
            add_bias_fnn=True,
            segment_length=12000,
            segment_stride=11500,
            # chunk_size = 11500,
            # num_first_blocks = 600000,
            # num_second_blocks = 50,
            # num_third_blocks = 1,
            max_methylation_embeddings=131070,
            meth_pad_id=0,
            age_pad_id=0,
            max_age_embeddings=120,
            mask_ratio_train=0.12,
            **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.is_folding_model = is_folding_model
        self.vocab_list = vocab_list
        self.add_bias_fnn = add_bias_fnn
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        self.max_methylation_embeddings = max_methylation_embeddings
        self.meth_pad_id = meth_pad_id
        self.max_age_embeddings = max_age_embeddings
        self.age_pad_id = age_pad_id
        self.mask_ratio_train = mask_ratio_train
