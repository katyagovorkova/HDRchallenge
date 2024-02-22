import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

"""# Download the data
Before running the followiing cell, go to https://drive.google.com/drive/folders/1_r03ZE9SQBuH8ksR6JnY_hzKB6YVJyo-?usp=sharing and create shortcut on your drive by doing right-click on the file --> "Add a shortcut to my Drive". In this case you will have no need to download the data.
"""

# Load my drive to access data and models
from google.colab import drive
drive.mount('/content/drive')

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

"""## Build the model

Our model processes a tensor of shape `(batch size, sequence length, features)`,
where `sequence length` is the number of time steps and `features` is each input
timeseries.

We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`.

"""

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

"""The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer. For
this example, a `GlobalAveragePooling1D` layer is sufficient.
"""

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

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
!mkdir -p saved_model
model.save(os.path.join(data_path, 'saved_model/my_model'))

pretrained_model = tf.keras.models.load_model(os.path.join(data_path, 'saved_model/my_model'))

# Check its architecture
pretrained_model.summary()

# load challenge test data
blackbox = np.load(os.path.join(data_path, 'ligo_blackbox.npz'))['data'].reshape((-1,200,2))
print('Blackbox shape:', blackbox.shape)

blackbox_prediction = model.predict(blackbox)
np.save('submission.npy', blackbox_prediction)