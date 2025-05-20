import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import gc
import argparse
from model.config import Config
from model.data_utils import TransliterationDataset, collate_fn
from model.seq2seq import Seq2SeqModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracy(predictions, targets):
    pred_chars = torch.argmax(predictions, dim=-1)
    correct = (pred_chars == targets).all(dim=1).sum().item()
    total = targets.size(0)
    return correct / total

def print_examples(src_texts, tgt_texts, pred_texts, n=5):
    print("\nSample Predictions:")
    print("-" * 80)
    for i in range(min(n, len(src_texts))):
        print(f"Source:     {src_texts[i]}")
        print(f"Target:     {tgt_texts[i]}")
        print(f"Prediction: {pred_texts[i]}")
        print("-" * 80)

def train(config, train_loader, val_loader, model, optimizer, criterion, train_dataset):
    best_val_accuracy = 0
    best_model_path = 'best_model_with_attention.pt' if config.attention else 'best_model_without_attention.pt'
    
    print(f"\nStarting training {'with' if config.attention else 'without'} attention")
    print(f"Model will be saved at: {best_model_path}")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        total_loss = 0
        train_accuracy = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{config.epochs}') as pbar:
            for src, tgt in train_loader:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                
                optimizer.zero_grad()
                output, _ = model(src, tgt[:, :-1])
                
                loss = criterion(output.view(-1, config.target_vocab_size),
                               tgt[:, 1:].contiguous().view(-1))
                
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                accuracy = calculate_accuracy(output, tgt[:, 1:])
                train_accuracy += accuracy
                total_loss += loss.item()
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
                
                del loss, output
                gc.collect()
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)
        
        # Validation
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
                
                if len(example_srcs) < 5:
                    pred = torch.argmax(output, dim=-1)
                    for i in range(min(5 - len(example_srcs), src.size(0))):
                        example_srcs.append(''.join([train_dataset.idx2char_src[idx.item()] for idx in src[i]]))
                        example_tgts.append(''.join([train_dataset.idx2char_tgt[idx.item()] for idx in tgt[i]]))
                        example_preds.append(''.join([train_dataset.idx2char_tgt[idx.item()] for idx in pred[i]]))
                
                val_loss += loss.item()
                accuracy = calculate_accuracy(output, tgt[:, 1:])
                val_accuracy += accuracy
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
                
                del loss, output
                gc.collect()
        
        val_accuracy = val_accuracy / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print_examples(example_srcs, example_tgts, example_preds)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'char2idx_src': train_dataset.char2idx_src,
                'char2idx_tgt': train_dataset.char2idx_tgt,
                'idx2char_src': train_dataset.idx2char_src,
                'idx2char_tgt': train_dataset.idx2char_tgt,
                'val_accuracy': val_accuracy
            }
            torch.save(checkpoint, best_model_path)
            print(f"\nSaved best model with validation accuracy {val_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', action='store_true')
    args = parser.parse_args()
    
    # Load config and set attention
    config = Config()
    config.attention = args.attention
    
    # Load datasets
    train_dataset = TransliterationDataset('dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.train.tsv')
    val_dataset = TransliterationDataset('dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.dev.tsv',
                                       train_dataset.char2idx_src,
                                       train_dataset.char2idx_tgt)
    
    # Update config with vocab sizes
    config.source_vocab_size = len(train_dataset.char2idx_src)
    config.target_vocab_size = len(train_dataset.char2idx_tgt)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset,
                           batch_size=config.batch_size * 2,
                           shuffle=False,
                           collate_fn=collate_fn,
                           pin_memory=True)
    
    # Initialize model and training components
    model = Seq2SeqModel(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train model
    train(config, train_loader, val_loader, model, optimizer, criterion, train_dataset)

if __name__ == "__main__":
    main()
