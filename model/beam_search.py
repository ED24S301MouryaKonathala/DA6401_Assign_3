import torch
import torch.nn.functional as F

class BeamSearchDecoder:
    def __init__(self, model, beam_size, max_length):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length

    def decode(self, src):
        batch_size = src.size(0)
        device = src.device
        
        # Get encoder outputs
        encoder_outputs, hidden = self.model.encoder(src)
        
        all_predictions = []
        for batch_idx in range(batch_size):
            # Get encoder outputs for this batch item
            batch_encoder_outputs = encoder_outputs[batch_idx:batch_idx+1]  # [1, src_len, H]
            
            # Initialize beams
            beams = [(
                torch.zeros(1, 1, dtype=torch.long, device=device),  # sequence
                0.0,  # score
                self._get_single_hidden(hidden, batch_idx)  # hidden state
            )]
            
            # Generate sequence
            for _ in range(self.max_length - 1):
                candidates = []
                
                for sequence, score, prev_hidden in beams:
                    # Forward pass through decoder with attention
                    output, attn = self.model.decoder(
                        sequence,
                        batch_encoder_outputs,
                        prev_hidden
                    )
                    
                    # Get probabilities for next token
                    log_probs = F.log_softmax(output[:, -1], dim=-1)
                    values, indices = log_probs.topk(self.beam_size)
                    
                    # Create candidates
                    for i in range(self.beam_size):
                        candidates.append((
                            torch.cat([sequence, indices[:, i:i+1]], dim=1),
                            score + values[:, i].item(),
                            prev_hidden  # Keep hidden state for continuing sequence
                        ))
                
                # Select top beams
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_size]
            
            # Add best prediction for this batch item
            all_predictions.append(beams[0][0])
        
        return torch.cat(all_predictions, dim=0)

    def _get_single_hidden(self, hidden, batch_idx):
        """Extract hidden state for single batch item and properly handle layer counts"""
        if isinstance(hidden, tuple):  # LSTM
            h = hidden[0][:, batch_idx:batch_idx+1].contiguous()
            c = hidden[1][:, batch_idx:batch_idx+1].contiguous()
            
            # Match decoder layer count
            if h.size(0) != self.model.num_decoder_layers:
                if h.size(0) < self.model.num_decoder_layers:
                    # Add extra layers
                    h_extra = torch.zeros(
                        self.model.num_decoder_layers - h.size(0),
                        1, self.model.hidden_dim, device=h.device
                    ).contiguous()
                    c_extra = torch.zeros(
                        self.model.num_decoder_layers - c.size(0),
                        1, self.model.hidden_dim, device=c.device
                    ).contiguous()
                    h = torch.cat([h, h_extra], dim=0)
                    c = torch.cat([c, c_extra], dim=0)
                else:
                    # Take last layers
                    h = h[-self.model.num_decoder_layers:]
                    c = c[-self.model.num_decoder_layers:]
            return (h, c)
        else:  # GRU/RNN
            hidden = hidden[:, batch_idx:batch_idx+1].contiguous()
            # Match decoder layer count
            if hidden.size(0) != self.model.num_decoder_layers:
                if hidden.size(0) < self.model.num_decoder_layers:
                    # Add extra layers
                    extra = torch.zeros(
                        self.model.num_decoder_layers - hidden.size(0),
                        1, self.model.hidden_dim, device=hidden.device
                    ).contiguous()
                    hidden = torch.cat([hidden, extra], dim=0)
                else:
                    # Take last layers
                    hidden = hidden[-self.model.num_decoder_layers:]
            return hidden
