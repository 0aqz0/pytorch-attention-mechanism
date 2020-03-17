import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import argparse
from datetime import datetime
from models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from functions import train_epoch, test

# Parameters manager
parser = argparse.ArgumentParser(description='CNN with Attention')
parser.add_argument('--train', action='store_true',
    help='Train the network')
parser.add_argument('--attention', default=True, type=bool,
    help='Train with attention')
parser.add_argument('--save', default=True, type=bool,
    help='Save the model')
parser.add_argument('--save_path', default='./', type=str,
    help='Path to save the model')
parser.add_argument('--visualize', action='store_true',
    help='Visualize the attention vector')
parser.add_argument('--epochs', default=100, type=int,
    help='Epochs for training')
parser.add_argument('--batch_size', default=32, type=int,
    help='Batch size for training or testing')
parser.add_argument('--lr', default=1e-1, type=float,
    help='Learning rate for training')
parser.add_argument('--device', default='0', type=str,
    help='Cuda device to use')
parser.add_argument('--log_interval', default=20, type=int,
    help='Interval to print messages')
args = parser.parse_args()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
batch_size = args.batch_size
learning_rate = args.lr

if __name__ == '__main__':
    # Load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # Create model
    model = ResNet18(attention=args.attention, num_classes=10).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Train
    if args.train:
        # Create loss criterion & optimizer & writer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        writer = SummaryWriter("runs/cnn_attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))

        for epoch in range(args.epochs):
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, args.log_interval, writer)
            test(model, criterion, test_loader, device, epoch, writer)
            if args.save:
                torch.save(model.state_dict(), os.path.join(args.save_path, "epoch{:03d}.pth".format(epoch+1)))

    # Visualize
    if args.visualize:
        pass