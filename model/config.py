class Config:
    def __init__(self):
        # Model params
        self.embedding_dim = 256
        self.hidden_dim = 512
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dropout = 0.3
        self.cell_type = 'lstm'
        
        # Training params - optimized for transliteration
        self.batch_size = 128  # Increased for better throughput
        self.epochs = 25
        self.learning_rate = 0.001
        self.beam_eval_epochs = 5  # Evaluate beam search every N epochs
        
        # Inference params
        self.beam_size = 3
        self.max_length = 30  # Set fixed max length based on data analysis
        
        # Attention flag
        self.attention = False
        
        # Will be set during data loading
        self.source_vocab_size = None
        self.target_vocab_size = None
