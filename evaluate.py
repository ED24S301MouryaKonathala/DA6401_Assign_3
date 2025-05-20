import torch
import wandb
import argparse
from model.config import Config
from model.seq2seq import Seq2SeqModel
from model.data_utils import TransliterationDataset, collate_fn
from torch.utils.data import DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

def load_best_model(attention=False):
    """Load model from local checkpoint file"""
    model_path = 'best_model_with_attention.pt' if attention else 'best_model_without_attention.pt'
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path)
    config = Config()
    for key, value in checkpoint['config'].items():
        setattr(config, key, value)
    
    model = Seq2SeqModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config, checkpoint

def clean_special_tokens(text_indices, idx2char):
    """Remove special tokens and return clean text"""
    special_tokens = {'<pad>', '<sos>', '<eos>'}
    return ''.join(idx2char[idx.item()] for idx in text_indices 
                  if idx2char[idx.item()] not in special_tokens)

def plot_attention_heatmap(attention_weights, src_tokens, tgt_tokens, output_path):
    """Create attention heatmap visualization with clean tokens"""
    # Remove special tokens from source and target
    special_tokens = {'<pad>', '<sos>', '<eos>'}
    src_tokens = [t for t in src_tokens if t not in special_tokens]
    tgt_tokens = [t for t in tgt_tokens if t not in special_tokens]
    
    # Trim attention weights to match clean tokens - fixed bracket syntax
    clean_attention = attention_weights[:len(tgt_tokens), :len(src_tokens)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(clean_attention, xticklabels=src_tokens, yticklabels=tgt_tokens)
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.savefig(output_path)
    plt.close()

def plot_attention_heatmaps(inputs, attentions, predictions, targets, save_path):
    """Plot attention heatmaps showing complete character alignments"""
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    try:
        telugu_font = fm.FontProperties(fname="NotoSansTelugu-Regular.ttf")
    except:
        print("Warning: Telugu font not found")
        telugu_font = None

    for idx, (src_text, attn_matrix, pred_text, _) in enumerate(zip(inputs, attentions, predictions, targets)):
        if idx >= 9: break
        i, j = divmod(idx, 3)
        
        # Get full character sequences
        src_chars = list(src_text)
        pred_chars = list(pred_text)
        
        # Normalize attention matrix properly
        attn = np.array(attn_matrix)
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-12)
        
        # Create heatmap with improved settings
        sns.heatmap(
            attn,
            ax=axes[i, j],
            cmap='YlOrRd',
            xticklabels=src_chars,
            yticklabels=pred_chars,
            cbar=True,
            square=True,
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Attention Weight'}
        )
        
        # Configure labels
        axes[i,j].set_xticklabels(src_chars, rotation=45, ha='right', fontsize=10)
        if telugu_font:
            axes[i,j].set_yticklabels(pred_chars, fontproperties=telugu_font, fontsize=10)
        
        axes[i,j].set_xlabel('English', fontsize=12)
        axes[i,j].set_ylabel('Telugu', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def evaluate_model(model, test_loader, checkpoint, device, attention=False):
    """Evaluate model and collect attention matrices"""
    correct = 0
    total = 0
    predictions = []
    attention_maps = []
    
    model.eval()
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            if attention:
                outputs, attention_weights = model(src, tgt)
                # Store raw attention weights without any filtering
                attention_maps.extend([
                    weights.cpu().numpy() for weights in attention_weights
                ])
            
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            pred_tokens = torch.argmax(logits, dim=-1)
            
            for i in range(len(src)):
                pred_clean = clean_special_tokens(pred_tokens[i], checkpoint['idx2char_tgt'])
                target_clean = clean_special_tokens(tgt[i], checkpoint['idx2char_tgt'])
                src_clean = clean_special_tokens(src[i], checkpoint['idx2char_src'])
                
                correct += (pred_clean == target_clean)
                total += 1
                
                predictions.append({
                    'source': src_clean,
                    'prediction': pred_clean,
                    'target': target_clean,
                    'correct': pred_clean == target_clean
                })
    
    return correct/total, predictions, attention_maps

def evaluate_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', action='store_true', help='Evaluate attention model')
    args = parser.parse_args()
    
    project_name = "telugu-transliteration-attention" if args.attention else "telugu-transliteration"
    output_dir = 'predictions_attention' if args.attention else 'predictions_vanilla'
    os.makedirs(output_dir, exist_ok=True)
    
    wandb.init(project=project_name, name="test_evaluation")
    
    model, config, checkpoint = load_best_model(args.attention)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    test_dataset = TransliterationDataset(
        'dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.test.tsv',
        checkpoint['char2idx_src'],
        checkpoint['char2idx_tgt']
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    accuracy, predictions, attention_maps = evaluate_model(model, test_loader, checkpoint, device, args.attention)
    
    pd.DataFrame(predictions).to_csv(f'{output_dir}/test_predictions.csv', index=False)
    
    if args.attention:
        os.makedirs('attention_plots', exist_ok=True)
        for i in range(min(9, len(attention_maps))):
            src_text = predictions[i]['source']
            tgt_text = predictions[i]['prediction']
            plot_attention_heatmap(
                attention_maps[i],
                list(src_text),
                list(tgt_text),
                f'attention_plots/sample_{i}.png'
            )
        
        inputs = [pred['source'] for pred in predictions[:9]]
        targets = [pred['target'] for pred in predictions[:9]]
        preds = [pred['prediction'] for pred in predictions[:9]]
        plot_attention_heatmaps(inputs, attention_maps[:9], preds, targets, "attention_heatmaps.png")
        wandb.log({"attention_heatmaps": wandb.Image("attention_heatmaps.png")})
    
    print(f"\nTest Set Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    wandb.log({
        "test_accuracy": accuracy,
        "predictions": wandb.Table(dataframe=pd.DataFrame(predictions).sample(n=min(10, len(predictions))))
    })
    
    return accuracy

if __name__ == "__main__":
    evaluate_test()
