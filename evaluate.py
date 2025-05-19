import torch
import wandb
import argparse
from model.config import Config
from model.seq2seq import Seq2SeqModel
from model.data_utils import TransliterationDataset, collate_fn
from model.beam_search import BeamSearchDecoder
from torch.utils.data import DataLoader
import pandas as pd
import os
from utils.visualize import generate_attention_visualizations

def load_best_model(project_name="telugu-transliteration"):
    api = wandb.Api()
    runs = api.runs(f"mourya001-indian-institute-of-technology-madras/{project_name}")
    best_run = sorted(runs, key=lambda run: run.summary.get('best_val_accuracy', 0), reverse=True)[0]
    
    print(f"Loading best model from run: {best_run.name}")
    print(f"Best validation accuracy: {best_run.summary.get('best_val_accuracy', 0):.4f}")
    
    artifact = api.artifact(f'mourya001-indian-institute-of-technology-madras/{project_name}/model-{best_run.id}:latest')
    artifact_dir = artifact.download()
    
    checkpoint = torch.load(f'{artifact_dir}/checkpoint.pt')
    
    config = Config()
    for key, value in checkpoint['config'].items():
        setattr(config, key, value)
    
    model = Seq2SeqModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, checkpoint

def evaluate_greedy(model, test_loader, checkpoint, device):
    """Evaluate using greedy decoding (exact matches only)"""
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            outputs = model(src, None)
            pred_tokens = torch.argmax(outputs, dim=-1)
            
            # Only count exact matches
            correct += (pred_tokens == tgt).all(dim=1).sum().item()
            total += src.size(0)
            
            # Save predictions
            for i in range(len(src)):
                pred_text = ''.join([checkpoint['idx2char_tgt'][idx.item()] for idx in pred_tokens[i]])
                true_text = ''.join([checkpoint['idx2char_tgt'][idx.item()] for idx in tgt[i]])
                src_text = ''.join([checkpoint['idx2char_src'][idx.item()] for idx in src[i]])
                predictions.append({
                    'source': src_text,
                    'prediction': pred_text,
                    'target': true_text,
                    'correct': pred_text == true_text
                })
    
    return correct / total, predictions

def evaluate_beam_search(model, test_loader, config, checkpoint, device):
    """Evaluate using beam search decoding"""
    correct = 0
    total = 0
    predictions = []
    
    beam_decoder = BeamSearchDecoder(model, config.beam_size, config.max_length)
    
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Beam search decoding
            pred_tokens = beam_decoder.decode(src)
            
            correct += (pred_tokens == tgt).all(dim=1).sum().item()
            total += src.size(0)
            
            # Save predictions
            for i in range(len(src)):
                pred_text = ''.join([checkpoint['idx2char_tgt'][idx.item()] for idx in pred_tokens[i]])
                true_text = ''.join([checkpoint['idx2char_tgt'][idx.item()] for idx in tgt[i]])
                src_text = ''.join([checkpoint['idx2char_src'][idx.item()] for idx in src[i]])
                
                predictions.append({
                    'source': src_text,
                    'prediction': pred_text,
                    'target': true_text,
                    'correct': pred_text == true_text
                })
    
    return correct / total, predictions

def evaluate_test_set(model, config, test_loader, test_dataset, device, checkpoint):
    """Evaluate model and generate visualizations"""
    # Create prediction directories
    os.makedirs('predictions_vanilla', exist_ok=True)
    os.makedirs('predictions_beam', exist_ok=True)
    if config.attention:
        os.makedirs('predictions_attention', exist_ok=True)
        os.makedirs('attention_plots', exist_ok=True)
    
    # Evaluate with both methods
    greedy_accuracy, greedy_predictions = evaluate_greedy(model, test_loader, checkpoint, device)
    beam_accuracy, beam_predictions = evaluate_beam_search(model, test_loader, config, checkpoint, device)
    
    # Save predictions
    pred_folder = 'predictions_attention' if config.attention else 'predictions_vanilla'
    pd.DataFrame(greedy_predictions).to_csv(f'{pred_folder}/test_predictions.csv', index=False)
    pd.DataFrame(beam_predictions).to_csv('predictions_beam/test_predictions.csv', index=False)
    
    # Generate visualizations
    samples = [(p['source'], p['target'], p['prediction']) 
              for p in greedy_predictions[:9]]
    create_sample_grid(samples, f'{pred_folder}/sample_grid.png')
    
    # Generate attention visualizations
    if config.attention:
        generate_attention_visualizations(
            model,
            test_loader,
            test_dataset.idx2char_src,
            test_dataset.idx2char_tgt,
            device,
            'attention_plots'
        )
    
    return greedy_accuracy, beam_accuracy, greedy_predictions, beam_predictions

def evaluate_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', action='store_true', help='Evaluate attention model')
    args = parser.parse_args()
    
    # Select correct project
    project_name = "telugu-transliteration-attention" if args.attention else "telugu-transliteration"
    
    model, config, checkpoint = load_best_model(project_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = TransliterationDataset(
        'dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.test.tsv',
        checkpoint['char2idx_src'],
        checkpoint['char2idx_tgt']
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Call evaluate_test_set with all required parameters
    greedy_accuracy, beam_accuracy, greedy_predictions, beam_predictions = evaluate_test_set(
        model, config, test_loader, test_dataset, device, checkpoint
    )
    
    print(f"\nTest Set Results:")
    print(f"Greedy Decoding Accuracy: {greedy_accuracy:.4f}")
    print(f"Beam Search Accuracy: {beam_accuracy:.4f}")
    
    # Save predictions in correct folder
    predictions_folder = 'predictions_attention' if config.attention else 'predictions_vanilla'
    os.makedirs(predictions_folder, exist_ok=True)
    pd.DataFrame(greedy_predictions).to_csv(f'{predictions_folder}/test_predictions.csv', index=False)
    
    os.makedirs('predictions_beam', exist_ok=True)
    pd.DataFrame(beam_predictions).to_csv('predictions_beam/test_predictions.csv', index=False)
    
    # Initialize wandb with correct project
    wandb.init(project=project_name, name="test_evaluation")
    wandb.log({
        "test_greedy_accuracy": greedy_accuracy,
        "test_beam_accuracy": beam_accuracy,
        "greedy_predictions": wandb.Table(dataframe=pd.DataFrame(greedy_predictions).sample(n=min(10, len(greedy_predictions)))),
        "beam_predictions": wandb.Table(dataframe=pd.DataFrame(beam_predictions).sample(n=min(10, len(beam_predictions))))
    })
    
    return greedy_accuracy, beam_accuracy

if __name__ == "__main__":
    evaluate_test()
