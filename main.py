from pathlib import Path
import time
import numpy as np
import pandas as pd
import random
import torch.backends.cudnn as cudnn
import torch

import warnings

from model import Model
from processor import process_data
warnings.filterwarnings('ignore')

from logger import init_logger

logger = init_logger()

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
        logger.debug(f"Using device: {device}")

    return device

def get_data(logger, data_folder, reconstruct):
    test_data, train_data = None, None

    if reconstruct:
        start_time = time.time()
        train_data = process_data(logger, Path(data_folder) / 'X_train.csv', Path(data_folder) / 'combined-labels.csv', Path(data_folder) / 'idx_label.csv')
        test_data = process_data(logger, Path(data_folder) / 'X_test.csv', Path(data_folder) / 'combined-labels.csv', Path(data_folder) / 'idx_label.csv')
        torch.save(train_data, './tmp/train.pt')
        torch.save(test_data, './tmp/test.pt')

        end_time = time.time()
        logger.info(f"Data preprocessing completed in {end_time - start_time:.2f} seconds.")
    else:
        train_data = torch.load('./tmp/train.pt')
        test_data = torch.load('./tmp/test.pt')
        logger.info("Preprocessed data is loaded successfully.")

    return test_data, train_data

# TODO: handle command line arguments
random_seed = 42
data_folder = '/mnt/d/TON_IoT_Dataset/small'
reconstruct = False
train = True
test = True
batch_size = 128
max_epochs = 1

if __name__ == "__main__":
    set_seed(random_seed)
    device = check_cuda()

    test_data, train_data = get_data(logger, data_folder, reconstruct)

    model = Model(logger, device)

    if train:
        model.train(train_data, batch_size=batch_size, max_epochs=max_epochs)
        # TODO: save model state
    
    if test:
        # TODO: load model state if needed
        model.test(None)
