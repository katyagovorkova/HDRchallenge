import os
import argparse
from datetime import datetime
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


from model import ADClassifier
from dataset import TorchADDataset


def main(args):

    # load data
    background = np.load(os.path.join(args.data_path, 'background.npz'))['data']
    sglf = np.load(os.path.join(args.data_path, 'sglf_for_challenge.npy'))
    bbh = np.load(os.path.join(args.data_path, 'bbh_for_challenge.npy'))

    """
    Create signal and background classes and mix them together
    We have four classes available in total: glitch, background,
    BBH and sine-Gaussian. We will use as background not only background
    dataset but also glitch dataset. And we will identify
    as signal both BBH and SG.
    """

    signal = np.concatenate((sglf, bbh), axis=0)

    anomaly_class = {
        'background': 0,
        'signal': 1
    }

    background_ids = np.full(background.shape[0], anomaly_class['background'], dtype=int)
    signal_ids = np.full(signal.shape[0], anomaly_class['signal'], dtype=int)

    x = np.concatenate((background, signal), axis=0).reshape((-1,200,2))
    y = np.concatenate((background_ids, signal_ids), axis=0)

    """
    Now we need to reshape the data to match the expected input from the
    Transformer architecute.
    Mix different event types together before the split.

    """

    # mix events
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    y = y.reshape((-1,1))

    x_train, x_val, y_train, y_val = train_test_split(
         x, y, test_size=0.33, random_state=42)

    print(f'x train/test shapes: {x_train.shape} {x_val.shape}')
    print(f'y train/test shapes: {y_train.shape} {y_val.shape}')


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(
        TorchADDataset(x_train, y_train, device),
        batch_size=1024,
        shuffle=False
        )
    validation_loader = torch.utils.data.DataLoader(
        TorchADDataset(x_val, y_val, device),
        batch_size=1024,
        shuffle=False
        )

    model = ADClassifier().to(device)
    print(model)
    print(f'The number of trainable parameters of the model is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    loss_fn = torch.nn.BCELoss()
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        # Log the running loss averaged per batch
        # for both training and validation
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        train_losses.append(avg_loss)
        val_losses.append(avg_vloss.item())

        epoch_number += 1

    torch.save(model.state_dict(), args.model_path)

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', type=str,
                        help='Path to the input dataset')
    parser.add_argument('model_path', type=str,
                        help='Where to save the model')

    args = parser.parse_args()
    main(args)