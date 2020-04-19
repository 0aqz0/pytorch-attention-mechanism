import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import math
import argparse
import numpy as np
from datetime import datetime
from functions import train_epoch, val_epoch, visualize_attn


# Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

"""
Implementation of Resnet+LSTM
"""
class ResCRNN(nn.Module):
    def __init__(self, sample_size=128, sample_duration=16, num_classes=100,
                lstm_hidden_size=512, lstm_num_layers=1, arch="resnet18",
                attention=False):
        super(ResCRNN, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.attention = attention

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        if self.attention:
            modules = list(resnet.children())[:-2]
        else:
            modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # calclatue kernal size
        self.last_size = int(math.ceil(self.sample_size / 32))
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        if self.attention:
            self.attn = nn.Linear(self.lstm_hidden_size, self.last_size**2, bias=False)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            if not self.attention:
                out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        if self.attention:
            N, T, C, H, W = cnn_embed_seq.size()
            # print(N, T, C, H, W)
            # (batch_size, channel, h, w)
            new_x = F.adaptive_avg_pool2d(cnn_embed_seq[:, 0, :, :, :], (1,1)).view(N, 1, C)
            out, (h_n, c_n) = self.lstm(new_x, None)
            for t in range(1, cnn_embed_seq.size(1)):
                # h_n: (num_layers * num_directions, batch, hidden_size)
                # take the first layer's hidden states
                score = self.attn(h_n[0])
                score = F.dropout(score, p=0.5, training=self.training)
                # print(score.shape)
                weights = F.softmax(score, dim=1)
                weights = weights.view(N, 1, self.last_size, self.last_size)
                weights = weights.expand_as(cnn_embed_seq[:, t, :, :, :])
                # print(weights.shape, cnn_embed_seq[:, t, :, :, :].shape)
                new_x = torch.mul(weights, cnn_embed_seq[:, t, :, :, :])
                # print(new_x.shape)
                new_x = new_x.view(N,C,-1).sum(dim=2)
                # print(cnn_embed_seq[:, t, :].shape, new_x.shape)
                out, (h_n, c_n) = self.lstm(new_x.unsqueeze(1), (h_n, c_n))
        else:
            out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = self.fc1(out[:, -1, :])

        return out

# Dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
Implementation of Chinese Sign Language Dataset(50 signers with 5 times)
"""
class CSL_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=500, train=True, transform=None):
        super(CSL_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.signers = 50
        self.repetition = 5
        if self.train:
            self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        else:
            self.videos_per_folder = int(0.2 * self.signers * self.repetition)
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        except Exception as e:
            print("Something wrong with your data path!!!")
            raise
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
        except Exception as e:
            raise

    def read_images(self, folder_path):
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        start = 1
        step = int(len(os.listdir(folder_path))/self.frames)
        for i in range(self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg').format(start+i*step))  #.convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __len__(self):
        return self.num_classes * self.videos_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx/self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = sorted([item for item in selected_folders if os.path.isdir(item)])
        if self.train:
            selected_folder = selected_folders[idx%self.videos_per_folder]
        else:
            selected_folder = selected_folders[idx%self.videos_per_folder + int(0.8*self.signers*self.repetition)]
        images = self.read_images(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.labels['{:06d}'.format(int(idx/self.videos_per_folder))])
        # label = self.labels['{:06d}'.format(int(idx/self.videos_per_folder))]
        label = torch.LongTensor([int(idx/self.videos_per_folder)])

        return images, label

    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]


# Parameters manager
parser = argparse.ArgumentParser(description='CNN-LSTM with Attention')
parser.add_argument('--train', action='store_true',
    help='Train the network')
parser.add_argument('--visualize', action='store_true',
    help='Visualize the attention vector')
parser.add_argument('--attention', action='store_true',
    help='Train with attention')
parser.add_argument('--save', action='store_true',
    help='Save the model')
parser.add_argument('--data_path', default='/home/haodong/Data/CSL_Isolated/color_video_125000', type=str,
    help='Path to data')
parser.add_argument('--label_path', default='/home/haodong/Data/CSL_Isolated/dictionary.txt', type=str,
    help='Path to labels')
parser.add_argument('--save_path', default='/home/haodong/Data/attention_models', type=str,
    help='Path to save the model')
parser.add_argument('--checkpoint', default='checkpoint.pth', type=str,
    help='Path to checkpoint')
parser.add_argument('--epochs', default=200, type=int,
    help='Epochs for training')
parser.add_argument('--batch_size', default=16, type=int,
    help='Batch size for training or testing')
parser.add_argument('--lr', default=1e-4, type=float,
    help='Learning rate for training')
parser.add_argument('--weight_decay', default=1e-5, type=float,
    help='Weight decay for training')
parser.add_argument('--device', default='0', type=str,
    help='Cuda device to use')
parser.add_argument('--log_interval', default=100, type=int,
    help='Interval to print messages')
parser.add_argument('--sample_size', default=128, type=int,
    help='Sample size for data')
parser.add_argument('--sample_duration', default=16, type=int,
    help='Sample duration for data')
parser.add_argument('--num_classes', default=100, type=int,
    help='Number of classes')
parser.add_argument('--lstm_hidden_size', default=512, type=int,
    help='Size of LSTM hidden states')
parser.add_argument('--lstm_num_layers', default=1, type=int,
    help='Number of LSTM layers')
args = parser.parse_args()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([args.sample_size, args.sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Isolated(data_path=args.data_path, label_path=args.label_path, frames=args.sample_duration,
        num_classes=args.num_classes, train=True, transform=transform)
    test_set = CSL_Isolated(data_path=args.data_path, label_path=args.label_path, frames=args.sample_duration,
        num_classes=args.num_classes, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    # Create model
    model = ResCRNN(sample_size=args.sample_size, sample_duration=args.sample_duration, num_classes=args.num_classes,
                lstm_hidden_size=args.lstm_hidden_size, lstm_num_layers=args.lstm_num_layers, attention=args.attention).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Summary writer
    writer = SummaryWriter("runs/cnnlstm_attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
    # Train
    if args.train:
        # Create loss criterion & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Start training
        print("Training Started".center(60, '#'))
        for epoch in range(args.epochs):
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, args.log_interval, writer)
            val_epoch(model, criterion, test_loader, device, epoch, writer)
            if args.save:
                torch.save(model.state_dict(), os.path.join(args.save_path, "cnnlstm_epoch{:03d}.pth".format(epoch+1)))

    # Visualize
    if args.visualize:
        pass
