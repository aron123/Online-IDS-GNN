import argparse
from pathlib import Path
import platform
import time
import numpy as np
import random
import pandas as pd
import torch

import warnings

import torch.version

from nids import NIDS
from processor import process_data
warnings.filterwarnings('ignore')

from logger import init_logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def check_cuda():
    cuda_available = torch.cuda.is_available()
    logger.debug(f"CUDA available: {cuda_available}")
    
    device = 'cpu'

    if cuda_available:
        device = torch.device("cuda")

    return device

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to data folder")
    parser.add_argument("--experiment_label", type=str, default="", help="Label of the experiment")
    
    parser.add_argument("--reconstruct", action="store_true", help="Perform graph reconstruction")
    parser.add_argument("--train", action="store_true", help="Run training phase")
    parser.add_argument("--test", action="store_true", help="Run testing phase")
    parser.add_argument("--binary", action="store_true", help="Perform binary classification")

    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden graph dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Attention dropout rate of GAT layers")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay (L2 regularization)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()
    return args

args = get_args()
logger = init_logger(args.experiment_label)

def get_data(logger, data_folder, reconstruct, binary):
    test_data, train_data = None, None

    if reconstruct:
        start_time = time.time()
        train_data = process_data(logger, Path(data_folder) / 'X_train.csv', Path(data_folder) / 'labels.csv', Path(data_folder) / 'idx_label.csv', binary)
        test_data = process_data(logger, Path(data_folder) / 'X_test.csv', Path(data_folder) / 'labels.csv', Path(data_folder) / 'idx_label.csv', binary)
        torch.save(train_data, './tmp/train.pt')
        torch.save(test_data, './tmp/test.pt')

        end_time = time.time()
        logger.info(f"Data preprocessing completed in {end_time - start_time:.2f} seconds.")
    else:
        train_data = torch.load('./tmp/train.pt')
        test_data = torch.load('./tmp/test.pt')
        logger.info("Preprocessed data is loaded successfully.")

    return test_data, train_data

if __name__ == "__main__":
    set_seed(args.random_seed)

    device = check_cuda()

    logger.info(f'Using Python version: {platform.python_version()}')
    logger.info(f'Using torch version: {torch.__version__}')
    logger.info(f'Using CUDA: {torch.version.cuda} (cuDNN {torch.backends.cudnn.version()})')
    logger.info(f'Using device: {device}')
    logger.info(f'Using dataset: {args.data_folder}')
    logger.info(f'Using model parameters: num_heads={args.num_heads}, num_layers={args.num_layers}, hidden_dim={args.hidden_dim}, dropout={args.dropout}')

    test_data, train_data = get_data(logger, args.data_folder, args.reconstruct, args.binary)
    labels = pd.read_csv(Path(args.data_folder) / 'idx_label.csv')

    edge_feat_dim = test_data[0][0].edata['feat'].shape[1]
    label_dim = 2 if args.binary else len(labels)

    model = NIDS(
        logger=logger,
        device=device,
        edge_feat_dim=edge_feat_dim,
        label_dim=label_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )

    if args.train:
        logger.info(f'Using training parameters: batch_size={args.batch_size}, max_epochs={args.max_epochs}, patience={args.patience}, lr={args.learning_rate}, weight_decay={args.weight_decay}')
        model.train(
            train_data,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    if args.test:
        targets = labels['label'].tolist() if args.binary is False else ['attack', 'normal']
        model.test(test_data, target_names=targets)
