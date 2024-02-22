import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

from model import build_model

"""# Download the data
Before running the followiing cell, go to https://drive.google.com/drive/folders/1_r03ZE9SQBuH8ksR6JnY_hzKB6YVJyo-?usp=sharing and create shortcut on your drive by doing right-click on the file --> "Add a shortcut to my Drive". In this case you will have no need to download the data.
"""

# Load my drive to access data and models
from google.colab import drive
drive.mount('/content/drive')


def main(args):

    data_path = '/content/drive/MyDrive/ligo_transformer_data/challenge/'

    """#Load files"""

    # load data
    background = np.load(os.path.join(data_path, 'background.npz'))['data']
    sglf = np.load(os.path.join(data_path, 'sglf_for_challenge.npy'))
    bbh = np.load(os.path.join(data_path, 'bbh_for_challenge.npy'))

    """# Create signal and background classes and mix them together
    We have four classes available in total: glitch, background, BBH and sine-Gaussian. We will use as background not only background dataset but also glitch dataset. And we will identify as signal both BBH and SG.
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

    """Now we need to reshape the data to match the expected input from the Transformer architecute. Mix different event types together before the split.

    """

    # mix events
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    x_train, x_test, y_train, y_test = train_test_split(
         x, y, test_size=0.33, random_state=42)

    print(f'x train/test shapes: {x_train.shape} {x_test.shape}')
    print(f'y train/test shapes: {y_train.shape} {y_test.shape}')


    """## Train and evaluate

    """

    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=64,
        num_heads=2,
        ff_dim=4,
        num_transformer_blocks=2,
        mlp_units=[8],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["binary_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=2,
        batch_size=1024,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)

    metric = "binary_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

    from sklearn import metrics
    scores = model.predict(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores)
    auc = metrics.roc_auc_score(y_test, scores)
    print(f'The total AUC is {auc*100:.1f} %')
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    # Save the entire model as a SavedModel.
    !mkdir -p args.savedir
    model.save(os.path.join(args.savedir, 'my_model'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data', type=str,
                        help='Input dataset')
    parser.add_argument('savedir', type=str,
                        help='Where to save the model')

    args = parser.parse_args()
    main(args)