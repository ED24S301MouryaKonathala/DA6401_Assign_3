class Config:
    def __init__(self):
        # Model params
        self.embedding_dim = 128
        self.hidden_dim = 512
        self.num_encoder_layers = 3
        self.num_decoder_layers = 5
        self.dropout = 0.2
        self.cell_type = 'gru'
        
        # Training params - optimized for transliteration
        self.batch_size = 512  # Increased for better throughput
        self.epochs = 30
        self.learning_rate = 0.0005
        self.beam_eval_epochs = 5  # Evaluate beam search every N epochs
        
        # Inference params
        self.beam_size = 3
        self.max_length = 30  # Set fixed max length based on data analysis
        
        # Attention flag
        self.attention = False
        
        # Will be set during data loading
        self.source_vocab_size = None
        self.target_vocab_size = None
