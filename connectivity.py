import torch
import wandb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from model.config import Config
from model.seq2seq import Seq2SeqModel
from model.data_utils import TransliterationDataset, collate_fn
from torch.utils.data import DataLoader

def plot_connectivity_heatmap(attention_weights, src_text, pred_text, save_path):
    """Plot attention connectivity with English chars in the heatmap cells"""
    plt.figure(figsize=(max(12, len(src_text)), len(pred_text)*0.8))
    
    try:
        telugu_font = fm.FontProperties(fname="NotoSansTelugu-Regular.ttf")
    except:
        print("Warning: Telugu font not found")
        telugu_font = None

    # Create a matrix of English characters for annotations
    char_matrix = [[src_char for src_char in src_text] for _ in range(len(pred_text))]
    
    # Create heatmap
    ax = sns.heatmap(
        attention_weights,
        cmap='YlOrRd',
        yticklabels=list(pred_text),
        xticklabels=['']*len(src_text),  # Empty x-axis labels
        cbar=True,
        square=True,
        vmin=0.0,
        vmax=1.0,
        annot=char_matrix,  # Use English characters as annotations
        fmt='',  # No formatting for annotations
        annot_kws={'size': 10}
    )
    
    # Set Telugu font for y-axis labels
    if telugu_font:
        ax.set_yticklabels(list(pred_text), fontproperties=telugu_font, fontsize=12)
    
    plt.ylabel('Predicted Telugu Characters', fontsize=12)
    plt.xticks([])  # Remove x-axis ticks completely
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    # Initialize wandb
    wandb.init(project="telugu-transliteration-attention", name="att_connectivity")
    
    # Create output directory
    os.makedirs('att_connectivity', exist_ok=True)
    
    # Load model
    checkpoint = torch.load('best_model_with_attention.pt')
    config = Config()
    for key, value in checkpoint['config'].items():
        setattr(config, key, value)
    
    model = Seq2SeqModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load test dataset
    test_dataset = TransliterationDataset(
        'dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.test.tsv',
        checkpoint['char2idx_src'],
        checkpoint['char2idx_tgt']
    )
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    
    # Generate connectivity plots
    model.eval()
    with torch.no_grad():
        # Get 3 different samples from test set
        test_samples = []
        test_iter = iter(test_loader)
        seen_sources = set()
        
        while len(test_samples) < 3:
            try:
                src, tgt = next(test_iter)
            except StopIteration:
                break
            
            # Only take samples with unique source text
            src_text = ''.join([test_dataset.idx2char_src[idx.item()] 
                              for idx in src[0] if test_dataset.idx2char_src[idx.item()] not in {'<pad>', '<sos>', '<eos>'}])
            
            if src_text not in seen_sources:
                test_samples.append((src, tgt))
                seen_sources.add(src_text)
        
        # Process each unique sample
        for idx, (src, tgt) in enumerate(test_samples):
            src = src.to(device)
            tgt = tgt.to(device)
            
            outputs, attention_weights = model(src, tgt)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Clean special tokens
            src_text = ''.join([checkpoint['idx2char_src'][idx.item()] 
                              for idx in src[0] if checkpoint['idx2char_src'][idx.item()] not in {'<pad>', '<sos>', '<eos>'}])
            pred_text = ''.join([checkpoint['idx2char_tgt'][idx.item()] 
                               for idx in predictions[0] if checkpoint['idx2char_tgt'][idx.item()] not in {'<pad>', '<sos>', '<eos>'}])
            
            # Get attention weights and normalize
            attn = attention_weights[0].cpu().numpy()
            attn = attn[:len(pred_text), :len(src_text)]
            attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-12)
            
            # Generate and save connectivity plot
            save_path = f'att_connectivity/connectivity_{idx+1}.png'
            plot_connectivity_heatmap(attn, src_text, pred_text, save_path)
            
            # Log to wandb
            wandb.log({
                f'connectivity_{idx+1}': wandb.Image(save_path),
                f'example_{idx+1}_source': src_text,
                f'example_{idx+1}_prediction': pred_text
            })

if __name__ == "__main__":
    main()
