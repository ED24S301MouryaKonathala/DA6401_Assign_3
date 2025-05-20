import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Add embedding dropout
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(config.source_vocab_size, config.embedding_dim)
        
        # Add input dropout
        self.input_dropout = nn.Dropout(config.dropout)
        
        if config.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                config.embedding_dim,
                config.hidden_dim,
                config.num_encoder_layers,
                dropout=config.dropout if config.num_encoder_layers > 1 else 0,
                batch_first=True
            )
        elif config.cell_type == 'gru':
            self.rnn = nn.GRU(
                config.embedding_dim, 
                config.hidden_dim,
                config.num_encoder_layers,
                dropout=config.dropout if config.num_encoder_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                config.embedding_dim,
                config.hidden_dim,
                config.num_encoder_layers, 
                dropout=config.dropout if config.num_encoder_layers > 1 else 0,
                batch_first=True
            )
            
        # Add output dropout
        self.output_dropout = nn.Dropout(config.dropout)
            
    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.embedding_dropout(embedded)
        
        # Apply input dropout
        embedded = self.input_dropout(embedded)
        
        outputs, hidden = self.rnn(embedded)
        
        # Apply output dropout
        outputs = self.output_dropout(outputs)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = config.attention
        self.hidden_dim = config.hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(config.target_vocab_size, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Attention layers
        if self.attention:
            self.attention_hidden = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.attention_encoder = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.attention_combine = nn.Linear(config.hidden_dim + config.embedding_dim, 
                                            config.embedding_dim)
        
        # RNN layer
        rnn_input_size = config.embedding_dim
        if config.cell_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_size, config.hidden_dim,
                             config.num_decoder_layers,
                             batch_first=True,
                             dropout=config.dropout if config.num_decoder_layers > 1 else 0)
        elif config.cell_type == 'gru':
            self.rnn = nn.GRU(rnn_input_size, config.hidden_dim,
                             config.num_decoder_layers,
                             batch_first=True,
                             dropout=config.dropout if config.num_decoder_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(rnn_input_size, config.hidden_dim,
                             config.num_decoder_layers,
                             batch_first=True,
                             dropout=config.dropout if config.num_decoder_layers > 1 else 0)
        
        self.output = nn.Linear(config.hidden_dim, config.target_vocab_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def compute_attention(self, encoder_outputs, hidden):
        """Compute comprehensive attention weights for all positions"""
        batch_size, src_len, hidden_size = encoder_outputs.size()
        
        if isinstance(hidden, tuple):  # LSTM
            query = hidden[0][-1]
        else:  # GRU/RNN
            query = hidden[-1]
        
        # Transform both query and encoder outputs
        encoder_features = self.attention_encoder(encoder_outputs)  # [B, src_len, H]
        query = self.attention_hidden(query).unsqueeze(1)          # [B, 1, H]
        
        # Compute attention scores
        scores = torch.bmm(encoder_features, query.transpose(1, 2))  # [B, src_len, 1]
        
        # Ensure we have attention weights for all positions
        attn_weights = F.softmax(scores, dim=1)  # [B, src_len, 1]
        
        # Compute context vector
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)  # [B, 1, H]
        
        return context, attn_weights.squeeze(2)  # Return [B, src_len]
    
    def forward(self, trg, encoder_outputs, hidden):
        batch_size = trg.size(0)
        target_len = trg.size(1)
        src_len = encoder_outputs.size(1)
        
        # Embed input tokens
        embedded = self.embedding(trg)  # [B, T, E]
        embedded = self.embedding_dropout(embedded)
        
        outputs = []
        attention_weights = []
        
        # Process each target token
        for t in range(target_len):
            input_t = embedded[:, t:t+1]  # [B, 1, E]
            
            if self.attention:
                # Compute attention
                context, attn = self.compute_attention(encoder_outputs, hidden)
                attention_weights.append(attn)  # Store weights for each step
                
                # Combine context with input
                rnn_input = self.attention_combine(
                    torch.cat((input_t, context), dim=2))
            else:
                rnn_input = input_t
            
            # RNN forward pass
            output, hidden = self.rnn(rnn_input, hidden)
            output = self.dropout(output)
            pred = self.output(output)
            outputs.append(pred)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # [B, T, V]
        if self.attention:
            attention_matrix = torch.stack(attention_weights, dim=1)  # [B, T, S]
            return outputs, attention_matrix
        return outputs, None

class Seq2SeqModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.num_decoder_layers = config.num_decoder_layers
        self.num_encoder_layers = config.num_encoder_layers
        self.hidden_dim = config.hidden_dim
        self.cell_type = config.cell_type
        self.max_length = getattr(config, 'max_length', 30)  # Default to 30

    def _adjust_hidden_state(self, hidden, batch_size=None):
        """Properly adjust hidden state dimensions between encoder and decoder"""
        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            batch_size = h.size(1) if batch_size is None else batch_size
            
            # Always use num_decoder_layers for target hidden size
            target_layers = self.num_decoder_layers
            current_layers = h.size(0)
            
            if current_layers != target_layers:
                if current_layers < target_layers:
                    # Add extra layers if needed
                    h_extra = torch.zeros(
                        target_layers - current_layers,
                        batch_size, self.hidden_dim, device=h.device
                    ).contiguous()
                    c_extra = torch.zeros(
                        target_layers - current_layers,
                        batch_size, self.hidden_dim, device=c.device
                    ).contiguous()
                    h = torch.cat([h, h_extra], dim=0)
                    c = torch.cat([c, c_extra], dim=0)
                else:
                    # Take required layers
                    h = h[:target_layers]
                    c = c[:target_layers]
                    
            return (h.contiguous(), c.contiguous())
        else:  # GRU/RNN
            # Similar logic for single hidden state
            hidden = hidden.contiguous()
            target_layers = self.num_decoder_layers
            current_layers = hidden.size(0)
            
            if current_layers != target_layers:
                if current_layers < target_layers:
                    extra = torch.zeros(
                        target_layers - current_layers,
                        batch_size if batch_size is not None else hidden.size(1),
                        self.hidden_dim, device=hidden.device
                    ).contiguous()
                    hidden = torch.cat([hidden, extra], dim=0)
                else:
                    hidden = hidden[:target_layers]
                    
            return hidden.contiguous()

    def forward(self, src, tgt=None):
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encoder(src)
        hidden = self._adjust_hidden_state(hidden, batch_size)

        if tgt is None:  # Inference mode
            predictions = []
            current_input = torch.zeros((batch_size, 1), dtype=torch.long, device=src.device)
            
            for _ in range(self.max_length):
                output, _ = self.decoder(current_input, encoder_outputs, hidden)
                predictions.append(output[:, -1:, :])
                current_input = torch.argmax(output[:, -1:, :], dim=-1)
            
            return torch.cat(predictions, dim=1), None
        
        # Training mode
        return self.decoder(tgt, encoder_outputs, hidden)
