import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb
import os
from matplotlib.gridspec import GridSpec

def create_sample_grid(samples, save_path, ncols=3, nrows=3):
    """Create grid visualization of sample predictions"""
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(nrows, ncols, figure=fig)
    
    for idx, (src, tgt, pred) in enumerate(samples):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.text(0.1, 0.7, f"Source: {src}", wrap=True)
        ax.text(0.1, 0.5, f"Target: {tgt}", wrap=True)
        ax.text(0.1, 0.3, f"Pred: {pred}", wrap=True)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return wandb.Image(save_path)

def plot_attention_heatmap(attention_weights, src_text, tgt_text, pred_text, save_path):
    """Plot single attention heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.cpu().numpy(),
        xticklabels=list(src_text),
        yticklabels=list(pred_text),
        cmap='viridis'
    )
    plt.title(f'Source: {src_text}\nTarget: {tgt_text}\nPred: {pred_text}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return wandb.Image(save_path)

def plot_attention_grid(attention_heatmaps, save_path):
    """Create 3x3 grid of individual attention heatmaps"""
    fig = plt.figure(figsize=(30, 30))
    gs = GridSpec(3, 3, figure=fig)
    
    for idx, (attn, src, tgt, pred) in enumerate(attention_heatmaps[:9]):  # Ensure only 9 heatmaps
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sns.heatmap(attn.cpu().numpy(), ax=ax, cmap='viridis', xticklabels=list(src), yticklabels=list(pred))
        ax.set_title(f'Source: {src}\nTarget: {tgt}\nPred: {pred}', fontsize=8)
        ax.set_xlabel('Source Sequence')
        ax.set_ylabel('Output Sequence')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return wandb.Image(save_path)

def generate_attention_visualizations(model, test_loader, src_vocab, tgt_vocab, device, save_dir):
    """Generate 9 individual attention heatmaps"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    heatmaps = []
    
    with torch.no_grad():
        test_iter = iter(test_loader)
        for grid_idx in range(9):  # Generate 9 individual heatmaps
            try:
                src, tgt = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                src, tgt = next(test_iter)
                
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Get model outputs with attention
            output, attn_weights = model(src, tgt)
            pred = torch.argmax(output, dim=-1)
            
            # Convert first example in batch
            src_text = ''.join([src_vocab[idx.item()] for idx in src[0]])
            tgt_text = ''.join([tgt_vocab[idx.item()] for idx in tgt[0]])
            pred_text = ''.join([tgt_vocab[idx.item()] for idx in pred[0]])
            
            # Save individual heatmap
            heatmap_path = f'{save_dir}/heatmap_{grid_idx+1}.png'
            plot_attention_heatmap(
                attn_weights[0],  # Take first example's attention weights
                src_text,
                tgt_text,
                pred_text,
                heatmap_path
            )
            
            heatmaps.append((
                attn_weights[0],
                src_text,
                tgt_text,
                pred_text
            ))
            
            # Log individual heatmap to wandb
            wandb.log({
                f'attention_heatmap_{grid_idx+1}': wandb.Image(heatmap_path),
                f'example_{grid_idx+1}': {
                    'source': src_text,
                    'target': tgt_text,
                    'prediction': pred_text
                }
            })
    
    # Create and save complete grid
    grid_path = f'{save_dir}/attention_grid.png'
    grid_img = plot_attention_grid(heatmaps, grid_path)
    wandb.log({'attention_heatmap_grid': grid_img})
