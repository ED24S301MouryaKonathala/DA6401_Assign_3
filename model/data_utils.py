import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TransliterationDataset:
    def __init__(self, filepath, char2idx_src=None, char2idx_tgt=None, max_length=30):
        self.data = []
        self.max_length = max_length
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    native = self._clean_text(parts[0].strip())   # Telugu
                    latin = self._clean_text(parts[1].strip())    # Latin
                    
                    # Store as (latin, native) - latin is source, telugu is target
                    if latin and native and len(latin) <= max_length and len(native) <= max_length:
                        self.data.append((latin, native))

        # Build vocabularies - now src is Latin, tgt is Telugu
        if char2idx_src is None:
            self.char2idx_src = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
            self._build_vocab([src for src, _ in self.data], self.char2idx_src)
        else:
            self.char2idx_src = char2idx_src
            
        if char2idx_tgt is None:
            self.char2idx_tgt = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
            self._build_vocab([tgt for _, tgt in self.data], self.char2idx_tgt)
        else:
            self.char2idx_tgt = char2idx_tgt
            
        self.idx2char_src = {v: k for k, v in self.char2idx_src.items()}
        self.idx2char_tgt = {v: k for k, v in self.char2idx_tgt.items()}

    def _clean_text(self, text):
        """Clean text by removing repeating characters"""
        cleaned = []
        prev_char = None
        for char in text:
            # Skip repeating characters and only keep printable ones
            if char != prev_char and char.isprintable():
                cleaned.append(char)
                prev_char = char
        return ''.join(cleaned)

    def _build_vocab(self, texts, char2idx):
        """Build vocabulary with special tokens"""
        chars = set()
        for text in texts:
            chars.update(list(text))
        
        # Add characters maintaining order
        for char in sorted(chars):
            if char not in char2idx:
                char2idx[char] = len(char2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        # Enforce max length
        src = src[:self.max_length]
        tgt = tgt[:self.max_length]
        
        # Convert to indices with special tokens
        src_indices = [self.char2idx_src[c] for c in src]
        tgt_indices = [self.char2idx_tgt[c] for c in tgt]
        
        # Add SOS/EOS tokens for target sequence
        tgt_indices = [self.char2idx_tgt['<sos>']] + tgt_indices + [self.char2idx_tgt['<eos>']]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    """Collate with proper padding"""
    src_seqs, tgt_seqs = zip(*batch)
    
    # Pad sequences
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded
