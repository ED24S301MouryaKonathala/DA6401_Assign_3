import torch
import torch.nn.functional as F

class BeamSearchDecoder:
    def __init__(self, model, beam_size, max_length):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length

    def _adjust_hidden_state(self, hidden, batch_size, device):
        """Adjust hidden state dimensions and ensure contiguous memory"""
        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            if self.model.num_encoder_layers < self.model.num_decoder_layers:
                extra_h = torch.zeros(
                    self.model.num_decoder_layers - self.model.num_encoder_layers,
                    batch_size, self.model.hidden_dim, device=device
                ).contiguous()
                extra_c = torch.zeros(
                    self.model.num_decoder_layers - self.model.num_encoder_layers,
                    batch_size, self.model.hidden_dim, device=device
                ).contiguous()
                h = torch.cat([h, extra_h], dim=0).contiguous()
                c = torch.cat([c, extra_c], dim=0).contiguous()
            else:
                h = h[-self.model.num_decoder_layers:].contiguous()
                c = c[-self.model.num_decoder_layers:].contiguous()
            return (h, c)
        else:  # GRU or RNN
            if self.model.num_encoder_layers < self.model.num_decoder_layers:
                extra = torch.zeros(
                    self.model.num_decoder_layers - self.model.num_encoder_layers,
                    batch_size, self.model.hidden_dim, device=device
                ).contiguous()
                return torch.cat([hidden, extra], dim=0).contiguous()
            else:
                return hidden[-self.model.num_decoder_layers:].contiguous()

    def decode(self, src):
        batch_size = src.size(0)
        device = src.device
        
        encoder_outputs, hidden = self.model.encoder(src)
        
        all_predictions = []
        for batch_idx in range(batch_size):
            beams = [(
                torch.zeros(1, 1, dtype=torch.long, device=device),
                0.0,
                self._adjust_hidden_state(self._get_single_hidden(hidden, batch_idx), 1, device)
            )]
            
            for _ in range(self.max_length - 1):
                candidates = []
                for sequence, score, prev_hidden in beams:
                    decoder_output = self.model.decoder(
                        sequence,
                        encoder_outputs[batch_idx:batch_idx+1],
                        prev_hidden
                    )
                    
                    # Handle decoder output properly
                    if isinstance(decoder_output, tuple):
                        output = decoder_output[0]
                        new_hidden = decoder_output[1]
                    else:
                        output = decoder_output
                        new_hidden = prev_hidden
                    
                    log_probs = F.log_softmax(output[:, -1], dim=-1)
                    values, indices = log_probs.topk(self.beam_size)
                    
                    for i in range(self.beam_size):
                        candidates.append((
                            torch.cat([sequence, indices[:, i:i+1]], dim=1),
                            score + values[:, i].item(),
                            new_hidden
                        ))
                
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_size]
            
            # Add best sequence for this batch
            all_predictions.append(beams[0][0])
        
        return torch.cat(all_predictions, dim=0)

    def _get_single_hidden(self, hidden, batch_idx):
        """Extract hidden state for single batch item and make contiguous"""
        if isinstance(hidden, tuple):  # LSTM
            h = hidden[0][:, batch_idx:batch_idx+1].contiguous()
            c = hidden[1][:, batch_idx:batch_idx+1].contiguous()
            return (h, c)
        # GRU or RNN
        return hidden[:, batch_idx:batch_idx+1].contiguous()
