import wandb
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import gc
import argparse
from model.config import Config
from model.data_utils import TransliterationDataset, collate_fn
from model.seq2seq import Seq2SeqModel
from model.beam_search import BeamSearchDecoder
from sweep_config import sweep_configuration
import time

# Constants for performance optimization
VALIDATION_INTERVAL = 5  # Validate every N epochs
BEAM_EVAL_INTERVAL = 5   # Run beam search every N epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracy(predictions, targets):
    """Calculate sequence-level accuracy (exact matches only)"""
    pred_chars = torch.argmax(predictions, dim=-1)
    correct = (pred_chars == targets).all(dim=1).sum().item()
    total = targets.size(0)
    return correct / total

def evaluate_beam_search(model, dataloader, beam_decoder, device, max_samples=1000):
    """Efficient batched beam search evaluation with progress bar"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(), tqdm(total=max_samples, desc="Beam Search Eval") as pbar:
        for src, tgt in dataloader:
            if total >= max_samples:
                break
                
            batch_size = min(src.size(0), max_samples - total)
            src = src[:batch_size].to(device)
            tgt = tgt[:batch_size].to(device)
            
            predicted = beam_decoder.decode(src)
            min_len = min(predicted.size(1), tgt.size(1))
            correct += (predicted[:, :min_len] == tgt[:, :min_len]).all(dim=1).sum().item()
            total += batch_size
            pbar.update(batch_size)
            
    return correct / total

def print_examples(src_texts, tgt_texts, pred_texts, n=5):
    """Print n example predictions with their sources and targets"""
    print("\nSample Predictions:")
    print("-" * 80)
    for i in range(min(n, len(src_texts))):
        print(f"Source:     {src_texts[i]}")
        print(f"Target:     {tgt_texts[i]}")
        print(f"Prediction: {pred_texts[i]}")
        print("-" * 80)

def train():
    # Initialize wandb with sweep
    run = wandb.init()
    
    # Get hyperparameters from sweep
    config = Config()
    for key, value in wandb.config.items():
        setattr(config, key, value)
    
    # Load datasets with dynamic max length
    train_dataset = TransliterationDataset('dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.train.tsv')
    val_dataset = TransliterationDataset('dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.dev.tsv',
                                       train_dataset.char2idx_src,
                                       train_dataset.char2idx_tgt)
    
    # Set vocabulary sizes in config
    config.source_vocab_size = len(train_dataset.char2idx_src)
    config.target_vocab_size = len(train_dataset.char2idx_tgt)
    
    # Optimized data loading
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset,
                           batch_size=config.batch_size * 2,  # Larger batch for validation
                           shuffle=False,
                           collate_fn=collate_fn,
                           pin_memory=True)
    
    # Initialize model and optimizer
    model = Seq2SeqModel(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Track gradients
    wandb.watch(model, log="all")
    
    best_val_accuracy = 0
    best_model_path = 'best_model.pt'  # Path for best model
    
    # Training loop with progress bars
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        train_accuracy = 0
        
        # Training progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{config.epochs}') as pbar:
            for batch_idx, (src, tgt) in enumerate(train_loader):
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                
                optimizer.zero_grad()
                output, _ = model(src, tgt[:, :-1])
                
                loss = criterion(output.view(-1, config.target_vocab_size),
                               tgt[:, 1:].contiguous().view(-1))
                
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
                optimizer.step()
                
                total_loss += loss.item()
                accuracy = calculate_accuracy(output, tgt[:, 1:])
                train_accuracy += accuracy
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
                
                # Free up memory
                del loss, output
                gc.collect()
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)
        
        # Full validation every epoch
        model.eval()
        val_loss = 0
        val_accuracy = 0
        example_srcs = []
        example_tgts = []
        example_preds = []
        
        with torch.no_grad(), tqdm(total=len(val_loader), desc='Validating') as pbar:
            for src, tgt in val_loader:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                
                output, _ = model(src, tgt[:, :-1])
                loss = criterion(output.view(-1, config.target_vocab_size),
                               tgt[:, 1:].contiguous().view(-1))
                
                # Save example predictions
                if len(example_srcs) < 5:
                    pred = torch.argmax(output, dim=-1)
                    for i in range(min(5 - len(example_srcs), src.size(0))):
                        example_srcs.append(''.join([train_dataset.idx2char_src[idx.item()] for idx in src[i]]))
                        example_tgts.append(''.join([train_dataset.idx2char_tgt[idx.item()] for idx in tgt[i]]))
                        example_preds.append(''.join([train_dataset.idx2char_tgt[idx.item()] for idx in pred[i]]))
                
                val_loss += loss.item()
                accuracy = calculate_accuracy(output, tgt[:, 1:])
                val_accuracy += accuracy
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
                pbar.update(1)
                
                # Free memory
                del loss, output
                gc.collect()
        
        val_accuracy = val_accuracy / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print examples after validation
        print_examples(example_srcs, example_tgts, example_preds)
        
        # Run beam search every BEAM_EVAL_INTERVAL epochs (for monitoring only)
        beam_accuracy = 0
        if (epoch + 1) % BEAM_EVAL_INTERVAL == 0:
            beam_decoder = BeamSearchDecoder(model, config.beam_size, config.max_length)
            beam_accuracy = evaluate_beam_search(model, val_loader, beam_decoder, DEVICE)
            print(f"\nBeam Search Accuracy: {beam_accuracy:.4f}")

        # Log metrics
        metrics = {
            'epoch': epoch,
            'train/loss': avg_train_loss,
            'train/accuracy': train_accuracy,
            'val/loss': avg_val_loss,
            'val/accuracy': val_accuracy,
            'val/beam_accuracy': beam_accuracy,
            # Learning metrics
            'learning_rate': optimizer.param_groups[0]['lr'],
            # Model parameters for correlation plots
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'num_encoder_layers': config.num_encoder_layers,
            'num_decoder_layers': config.num_decoder_layers,
            'cell_type': config.cell_type,
            'dropout': config.dropout,
            'batch_size': config.batch_size,
            'beam_size': config.beam_size
        }
        wandb.log(metrics)

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                **metrics,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'char2idx_src': train_dataset.char2idx_src,
                'char2idx_tgt': train_dataset.char2idx_tgt,
                'idx2char_src': train_dataset.idx2char_src,
                'idx2char_tgt': train_dataset.idx2char_tgt,
            }
            
            # Save locally
            torch.save(checkpoint, best_model_path)
            print(f"\nSaved best model with validation accuracy {val_accuracy:.4f} to {best_model_path}")
            
            # Save to wandb
            new_artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}-epoch-{epoch}",
                type="model",
                description=f"Model checkpoint from epoch {epoch}",
                metadata=dict(wandb.config)
            )
            new_artifact.add_file(best_model_path)
            wandb.log_artifact(new_artifact)
            
            # Update summary metrics
            wandb.run.summary.update({
                'best_val_accuracy': val_accuracy,
                'best_model_path': best_model_path,
                'best_beam_accuracy': beam_accuracy,  # Still track beam accuracy
                'best_epoch': epoch,
                'source_vocab_size': config.source_vocab_size,
                'target_vocab_size': config.target_vocab_size,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'convergence_epoch': epoch
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', action='store_true')
    args = parser.parse_args()
    
    # Update sweep config with attention
    sweep_configuration['parameters']['attention'] = {'values': [args.attention]}
    
    # Initialize sweep with correct project
    project_name = "telugu-transliteration-attention" if args.attention else "telugu-transliteration"
    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    
    # Run sweep
    wandb.agent(sweep_id, train, count=100)
