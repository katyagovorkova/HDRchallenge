import os
import argparse
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F


from model import ADClassifier


def main(args):

    # Model class must be defined somewhere

    pretrained_model = ADClassifier()
    pretrained_model.load_state_dict(torch.load(args.modeldir))
    pretrained_model.eval()

    # Check its architecture
    print(pretrained_model)

    # load challenge test data
    blackbox = np.load(os.path.join(args.data_path, 'ligo_blackbox.npz'))['data'].reshape((-1,200,2))
    print('Blackbox shape:', blackbox.shape)

    # transform to float64
    x = torch.from_numpy(blackbox)
    x = x.to(torch.float32)

    blackbox_prediction = pretrained_model(x)
    np.save(args.submission_path, blackbox_prediction.detach().numpy())

    # scores = pretrained_model(x_val)
    # fpr, tpr, thresholds = metrics.roc_curve(y_val, scores)
    # auc = metrics.roc_auc_score(y_val, scores)
    # print(f'The total AUC is {auc*100:.1f} %')
    # plt.plot(fpr, tpr)
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.savefig('output/ROC.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', type=str,
                        help='Path to the input dataset')
    parser.add_argument('modeldir', type=str,
                        help='Where to save the model')
    parser.add_argument('submission_path', type=str,
                        help='Where to save the model')

    args = parser.parse_args()
    main(args)