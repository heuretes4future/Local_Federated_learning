server_config = {
    'comm_rounds': 100,
    'FL_algorithm': 'FedAvg',
    'Update_signal': 'Cosine_similarity',
    'N_clients': 10,
    'strategies': ['Baseline', 'LA', 'SGA'],
    'skew_levels': [10, 5, 2],
    'lr_step_rounds': 5,
    'base_lr': 1e-03,   # FIX: unified to match model_config — was 1e-02 causing mismatch
    'seeds': [42, 123, 999],
    'print_every': 5,
}

model_config = {
    'base_lr': 1e-03,
    'batch_size': 256,
    'kernel_size': 3,
    'padding': 1,
    'stride': 1,
    'dropout_rate': 0.1,
    'epochs': 1,
    'optimizer': 'SGD',
    'activation': 'ReLU',
    'momentum': 0.9,
    'weight_decay': 1e-04,
    'output_activation': 'Softmax',
    'in_channels': 3,
    'out_channels': 16,
    'num_classes': 10
}
