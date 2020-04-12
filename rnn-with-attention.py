import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import argparse
from functions import train_epoch, test

# Parameters manager
parser = argparse.ArgumentParser(description='RNN with Attention')
parser.add_argument('--train', action='store_true',
    help='Train the network')
parser.add_argument('--visualize', action='store_true',
    help='Visualize the attention vector')
parser.add_argument('--no_attention', action='store_true',
    help='Train without attention')
parser.add_argument('--no_save', action='store_true',
    help='Not save the model')
parser.add_argument('--save_path', default='/home/haodong/Data/attention_models', type=str,
    help='Path to save the model')
parser.add_argument('--checkpoint', default='checkpoint.pth', type=str,
    help='Path to checkpoint')
parser.add_argument('--epochs', default=300, type=int,
    help='Epochs for training')
parser.add_argument('--batch_size', default=32, type=int,
    help='Batch size for training or testing')
parser.add_argument('--lr', default=1e-4, type=float,
    help='Learning rate for training')
parser.add_argument('--weight_decay', default=1e-4, type=float,
    help='Weight decay for training')
parser.add_argument('--device', default='0', type=str,
    help='Cuda device to use')
parser.add_argument('--log_interval', default=100, type=int,
    help='Interval to print messages')
args = parser.parse_args()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Train the model
    if args.train:
        pass

    # Visualize attention map
    if args.visualize:
        pass
