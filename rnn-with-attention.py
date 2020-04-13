import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from models import AttnLSTM

# Parameters manager
parser = argparse.ArgumentParser(description='RNN with Attention')
parser.add_argument('--train', action='store_true',
    help='Train the network')
parser.add_argument('--visualize', action='store_true',
    help='Visualize the attention vector')
parser.add_argument('--no_save', action='store_true',
    help='Not save the model')
parser.add_argument('--save_path', default='/home/haodong/Data/attention_models', type=str,
    help='Path to save the model')
parser.add_argument('--checkpoint', default='rnn_checkpoint.pth', type=str,
    help='Path to checkpoint')
parser.add_argument('--epochs', default=30, type=int,
    help='Epochs for training')
parser.add_argument('--lr', default=1e-4, type=float,
    help='Learning rate for training')
parser.add_argument('--weight_decay', default=1e-4, type=float,
    help='Weight decay for training')
parser.add_argument('--device', default='0', type=str,
    help='Cuda device to use')
parser.add_argument('--log_interval', default=1000, type=int,
    help='Interval to print messages')
args = parser.parse_args()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data(n, seq_length, delimiter=0.0, index_1=None, index_2=None):
    x = np.random.uniform(0, 10, (n, seq_length))
    y = np.zeros(shape=(n, 1))
    for i in range(n):
        if index_1 is None and index_2 is None:
            a, b = np.random.choice(range(1, seq_length), size=2, replace=False)
        else:
            a, b = index_1, index_2
        y[i] = 0.5 * x[i, a] + 0.5 * x[i, b]
        x[i, a-1] = delimiter
        x[i, b-1] = delimiter
    x = np.expand_dims(x, axis=-1)
    return x, y


if __name__ == '__main__':
    # Generate data
    seq_length, train_length, val_length, test_length = 20, 20000, 4000, 10
    x_train, y_train = generate_data(train_length, seq_length)
    x_val, y_val = generate_data(val_length, seq_length)
    x_test, y_test = generate_data(test_length, seq_length, index_1=5, index_2=13)
    # Create the model
    model = AttnLSTM(input_size=1, hidden_size=128, num_layers=1).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Summary writer
    writer = SummaryWriter("runs/rnn_attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))

    if args.train:
        # Create loss criterion & optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            # train model
            model.train()
            losses = []
            for i in range(train_length):
                x = torch.Tensor(x_train[i, :]).unsqueeze(0).to(device)
                y = torch.Tensor(y_train[i, :]).unsqueeze(0).to(device)
                # print(x.shape, y.shape)
                optimizer.zero_grad()
                # forward
                pred, _ = model(x)
                # compute the loss
                loss = criterion(pred, y)
                losses.append(loss.item())
                # backward & optimize
                loss.backward()
                optimizer.step()

                if (i + 1) % args.log_interval == 0:
                    print("epoch {:3d} | iteration {:5d} | Loss {:.6f}".format(epoch+1, i+1, loss.item()))

            # calculate average loss
            training_loss = sum(losses)/len(losses)
            writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
            print("Average Training Loss of Epoch {}: {:.6f}".format(epoch+1, training_loss))

            # save model
            if not args.no_save:
                torch.save(model.state_dict(), os.path.join(args.save_path, "rnn_epoch{:03d}.pth".format(epoch+1)))
                print("Saving Model of Epoch {}".format(epoch+1))

            # validate model
            model.eval()
            losses = []
            for i in range(val_length):
                x = torch.Tensor(x_val[i, :]).unsqueeze(0).to(device)
                y = torch.Tensor(y_val[i, :]).unsqueeze(0).to(device)
                # forward
                pred, _ = model(x)
                # compute the loss
                loss = criterion(pred, y)
                losses.append(loss.item())

            # calculate average loss
            val_loss = sum(losses)/len(losses)
            writer.add_scalars('Loss', {'val': val_loss}, epoch+1)
            print("Average Validation Loss of Epoch {}: {:.6f}".format(epoch+1, val_loss))

    # Visualize attention map
    if args.visualize:
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        for i in range(test_length):
            with torch.no_grad():
                x = torch.Tensor(x_test[i, :]).unsqueeze(0).to(device)
                y = torch.Tensor(y_test[i, :]).unsqueeze(0).to(device)
                # forward
                pred, weights = model(x)
                # print(y, pred, weights)
                plt.title('Attention Weights')
                plt.xticks(np.arange(0, seq_length))
                plt.yticks(np.arange(0, 1, step=0.1))
                plt.bar(range(seq_length), weights.squeeze().cpu().numpy(), color='royalblue')
                plt.savefig('output_{}.png'.format(i))
