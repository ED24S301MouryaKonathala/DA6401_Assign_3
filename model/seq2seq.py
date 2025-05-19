import torch
import torch.nn as nn

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
        self.embedding = nn.Embedding(config.target_vocab_size, config.embedding_dim)
        
        # Add dropouts at each stage
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.input_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        if config.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                config.embedding_dim,
                config.hidden_dim,
                config.num_decoder_layers,
                dropout=config.dropout if config.num_decoder_layers > 1 else 0,
                batch_first=True
            )
        elif config.cell_type == 'gru':
            self.rnn = nn.GRU(
                config.embedding_dim,
                config.hidden_dim, 
                config.num_decoder_layers,
                dropout=config.dropout if config.num_decoder_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                config.embedding_dim,
                config.hidden_dim,
                config.num_decoder_layers,
                dropout=config.dropout if config.num_decoder_layers > 1 else 0,
                batch_first=True
            )
            
        if self.attention:
            self.attention_layer = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            self.combine = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            
        self.fc_out = nn.Linear(config.hidden_dim, config.target_vocab_size)
            
    def forward(self, trg, encoder_outputs, hidden):
        embedded = self.embedding(trg)
        embedded = self.embedding_dropout(embedded)
        embedded = self.input_dropout(embedded)
        
        if self.attention and encoder_outputs is not None:
            # Calculate attention weights
            attn_weights = torch.bmm(encoder_outputs, hidden[-1].unsqueeze(2))
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.bmm(encoder_outputs.transpose(1,2), attn_weights)
            rnn_input = torch.cat((embedded, context.squeeze(2)), dim=2)
            output, hidden = self.rnn(rnn_input, hidden)
            output = self.combine(torch.cat((output, context.transpose(1,2)), dim=2))
            output = self.output_dropout(output)
            prediction = self.fc_out(output)
            return prediction, attn_weights
        else:
            output, hidden = self.rnn(embedded, hidden)
            output = self.output_dropout(output)
            prediction = self.fc_out(output)
            return prediction, None

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
        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            batch_size = h.size(1) if batch_size is None else batch_size
            if self.num_encoder_layers < self.num_decoder_layers:
                h_extra = torch.zeros(
                    self.num_decoder_layers - self.num_encoder_layers,
                    batch_size, self.hidden_dim, device=h.device
                ).contiguous()
                c_extra = torch.zeros(
                    self.num_decoder_layers - self.num_encoder_layers,
                    batch_size, self.hidden_dim, device=c.device
                ).contiguous()
                h = torch.cat([h, h_extra], dim=0)
                c = torch.cat([c, c_extra], dim=0)
            else:
                h = h[-self.num_decoder_layers:]
                c = c[-self.num_decoder_layers:]
            return (h.contiguous(), c.contiguous())
        else:  # GRU/RNN
            batch_size = hidden.size(1) if batch_size is None else batch_size
            if self.num_encoder_layers < self.num_decoder_layers:
                extra = torch.zeros(
                    self.num_decoder_layers - self.num_encoder_layers,
                    batch_size, self.hidden_dim, device=hidden.device
                ).contiguous()
                return torch.cat([hidden, extra], dim=0).contiguous()
            return hidden[-self.num_decoder_layers:].contiguous()

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
