# Telugu-English Transliteration with Attention

This project implements a sequence-to-sequence model with attention mechanism for transliterating English text to Telugu script using PyTorch.

## System Requirements
- Python 3.6+
- CUDA-capable GPU (recommended)
- 8GB RAM minimum

## Setup

1. Install dependencies:
```bash
pip install torch torchvision tqdm numpy pandas matplotlib seaborn wandb
```

2. Install Telugu font for visualization:
```bash
wget https://github.com/jenskutilek/free-fonts/blob/master/Noto/Noto%20Sans%20Telugu/TTF/NotoSansTelugu-Regular.ttf
```

3. Download and extract the Dakshina dataset:
```bash
wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
tar -xzf dakshina_dataset_v1.0.tar
```

4. Set up Weights & Biases:
```bash
wandb login
```

## Project Structure
```
.
├── model/
│   ├── config.py          # Model configuration (hidden size, embedding size)
│   ├── seq2seq.py        # Seq2SeqModel with Encoder, Decoder, and Attention
│   └── data_utils.py     # TransliterationDataset and data processing
├── train.py              # Training loop with validation
├── evaluate.py           # Evaluation and attention visualization
├── attention_plots/      # Individual attention heatmaps
├── predictions_attention/  # Predictions and results with attention
├── predictions_vanilla/   # Predictions without attention
└── README.md
```

## Model Architecture

### Encoder
- Multiple layer RNN (LSTM/GRU/RNN) based on configuration
- Character-level embeddings with dropout
- Three dropout layers:
  - Embedding dropout
  - Input dropout
  - Output dropout
- Configurable number of layers and hidden dimensions
- Batch-first processing

### Decoder
- Multiple layer RNN (LSTM/GRU/RNN) matching encoder type
- Embedding layer with dropout
- Hidden state adjustment between encoder-decoder
- Maximum sequence length: 30 (default)
- Batch-first processing

### Attention Mechanism
- Bahdanau (additive) attention when enabled
- Components:
  - Hidden state transformation: Linear(hidden_dim → hidden_dim)
  - Encoder output transformation: Linear(hidden_dim → hidden_dim)
  - Context vector combination: Linear(hidden_dim + embedding_dim → embedding_dim)
- Attention computation:
  1. Transform encoder outputs and decoder hidden state
  2. Combine with tanh activation
  3. Calculate attention scores with softmax
  4. Create context vector through weighted sum

### Configuration Parameters
```python
class Config:
    def __init__(self):
        self.embedding_dim = 256
        self.hidden_dim = 256
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dropout = 0.5
        self.cell_type = 'lstm'  # Options: 'lstm', 'gru', 'rnn'
        self.attention = True
        self.max_length = 30
```

## Training

### Without Attention
```bash
python train.py
```

### With Attention
```bash
python train.py  --attention
```

Training arguments:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--attention`: Enable attention mechanism
- `--hidden_size`: LSTM hidden size (default: 256)
- `--embedding_size`: Character embedding size (default: 256)
- `--dropout`: Dropout rate (default: 0.1)

## Training Tips
1. Start with a small learning rate (0.001)
2. Use batch size of 32 for optimal performance
3. Train for at least 20 epochs
4. Monitor validation accuracy for early stopping
5. Save best model based on validation performance

## Model Checkpoints
The model saves two types of checkpoints:
- `best_model_with_attention.pt`: Best model with attention
- `best_model_without_attention.pt`: Best model without attention

## Evaluation

### Without Attention
```bash
python evaluate.py
```

### With Attention
```bash
python evaluate.py --attention
```

The evaluation script will:
1. Load the best model checkpoint
2. Generate predictions on the test set
3. Calculate accuracy
4. Save predictions to CSV
5. Generate attention visualizations (if using attention model)
6. Log results to Weights & Biases

## Attention Visualization Details
The evaluation script generates attention heatmaps showing:
- X-axis: Source English characters (including special tokens)
- Y-axis: Generated Telugu characters
- Color intensity: Normalized attention weights (0-1)
- 3x3 grid showing different translation examples
- Special token handling (<sos>, <eos>, <pad>)

### Visualization Files
- Individual heatmaps: `attention_plots/sample_{i}.png`
- Combined grid view: `attention_heatmaps.png`
- Test predictions: `predictions_attention/test_predictions.csv`
- Visualization parameters:
  - Figure size: 20x20 inches
  - DPI: 300
  - Colormap: 'Reds'
  - Telugu font: NotoSansTelugu-Regular.ttf

## Hyperparameter Sweep
We used Weights & Biases sweeps to optimize model performance.

### Sweep Configuration
```yaml
method: random
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  hidden_size:
    values: [128, 256, 512]
  embedding_size:
    values: [128, 256]
  dropout:
    min: 0.1
    max: 0.5
  num_layers:
    values: [1, 2]
```

### Experiments Run
1. Baseline Model (No Attention)
   - Training Accuracy: 66.64%
   - Validation Accuracy: 45.13%
   - Test Accuracy: 44.18%
   - Learning rate: 0.001
   - Hidden size: 256
   - Embedding size: 256
   - Dropout: 0.2

2. Attention Model
   - Training Accuracy: 70.08%
   - Validation Accuracy: 45.59%
   - Test Accuracy: 45.26%
   - Learning rate: 0.001
   - Hidden size: 256
   - Embedding size: 256
   - Dropout: 0.2
   - Notable improvement in handling longer words

3. Key Findings
   - Attention mechanism improved overall performance:
     - Training accuracy improved by ~3.44%
     - Validation accuracy improved by ~0.46%
     - Test accuracy improved by ~1.08%
   - Model shows signs of overfitting with higher training accuracy
   - Attention helps slightly with generalization
   - Both models achieve similar validation performance

## Results
Results are logged to Weights & Biases projects:
- Without attention: "telugu-transliteration"
- With attention: "telugu-transliteration-attention"

Metrics tracked:
1. Training Metrics:
   - Training loss per epoch
   - Training accuracy per epoch
   - Learning rate
   - Batch size

2. Validation Metrics:
   - Validation loss per epoch
   - Validation accuracy per epoch
   - Best validation accuracy
   - Best epoch number

3. Model Parameters:
   - Embedding dimension
   - Hidden dimension
   - Number of encoder/decoder layers
   - Cell type (LSTM/GRU/RNN)
   - Dropout rate
   - Beam size (for inference)

4. Evaluation Outputs:
   - Test accuracy
   - Sample predictions table
   - Attention visualizations (for attention model)

Evaluation outputs are saved in:
- `predictions_vanilla/`: Model predictions without attention
  - `test_predictions.csv`: Test set predictions and accuracy
- `predictions_attention/`: Model predictions with attention
  - `test_predictions.csv`: Test set predictions and accuracy
  - `attention_plots/`: Individual attention heatmaps for samples
  - `attention_heatmaps.png`: 3x3 grid visualization of attention patterns


## Attention Visualization 
The evaluation script generates Character Connectivity Visualization:
   - Shows character-level connections using `connectivity.py`
   - Y-axis: Predicted Telugu characters
   - Each cell: Contains source English character with color intensity showing attention weight
   - Three different test samples
   - Located in: `att_connectivity/`
   - Helps understand character-to-character relationships in transliteration

### Running Connectivity Visualization
```bash
python connectivity.py