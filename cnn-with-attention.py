import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from tensorboardX import SummaryWriter
import os
import argparse
import numpy as np
from datetime import datetime
from models import AttnVGG
from functions import train_epoch, val_epoch, visualize_attn


# Parameters manager
parser = argparse.ArgumentParser(description='CNN with Attention')
parser.add_argument('--train', action='store_true',
    help='Train the network')
parser.add_argument('--visualize', action='store_true',
    help='Visualize the attention vector')
parser.add_argument('--no_save', action='store_true',
    help='Not save the model')
parser.add_argument('--save_path', default='/home/haodong/Data/attention_models', type=str,
    help='Path to save the model')
parser.add_argument('--checkpoint', default='cnn_checkpoint.pth', type=str,
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
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    # Create model
    model = AttnVGG(sample_size=32, num_classes=100).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Summary writer
    writer = SummaryWriter("runs/cnn_attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
    # Train
    if args.train:
        # Create loss criterion & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # lr_lambda = lambda epoch : np.power(0.5, int(epoch/25))
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        for epoch in range(args.epochs):
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, args.log_interval, writer)
            val_epoch(model, criterion, test_loader, device, epoch, writer)
            # adjust learning rate
            # scheduler.step()
            if not args.no_save:
                torch.save(model.state_dict(), os.path.join(args.save_path, "cnn_epoch{:03d}.pth".format(epoch+1)))
                print("Saving Model of Epoch {}".format(epoch+1))

    # Visualize
    if args.visualize:
        # Load model
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # get images
                inputs = inputs.to(device)
                if batch_idx == 0:
                    images = inputs[0:16,:,:,:]
                    I = utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
                    writer.add_image('origin', I)
                    _, c1, c2, c3 = model(images)
                    # print(I.shape, c1.shape, c2.shape, c3.shape, c4.shape)
                    attn1 = visualize_attn(I, c1)
                    writer.add_image('attn1', attn1)
                    attn2 = visualize_attn(I, c2)
                    writer.add_image('attn2', attn2)
                    attn3 = visualize_attn(I, c3)
                    writer.add_image('attn3', attn3)
                    break
