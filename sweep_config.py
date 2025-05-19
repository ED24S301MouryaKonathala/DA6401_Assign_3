sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'embedding_dim': {'values': [32, 64, 128]},
        'hidden_dim': {'values': [128, 256, 512]},
        'num_encoder_layers': {'values': [3, 4, 5]},
        'num_decoder_layers': {'values': [3, 4, 5]},
        'cell_type': {'values': ['rnn','lstm', 'gru']},  
        'dropout': {'values': [0.2, 0.3]},
        'beam_size': {'values': [3, 5]},
        'batch_size': {'values': [64, 128, 256]},
        'learning_rate': {'values': [0.001, 0.0005, 0.0001]}
    }
}
