# Dictionary storing network parameters.
params = {
    'batch_size': 8,# Batch size.
    'num_epochs': 100,# Number of epochs to train for.
    'learning_rate': 2e-4,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 10,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'PatternNet'}# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!