import sys
import os
import numpy as np
import pandas as pd
import time


# Input directory to read test input from
input_dir = sys.argv[1]

# Output data directory to which to write predictions
output_dir = sys.argv[2]

program_dir = sys.argv[3]
submission_dir = sys.argv[4]

sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_prediction_data():

    # set test data and solution file
    test_data_file = os.path.join(input_dir, 'ligo_blackbox.npz')

    # Read Test data
    with np.load(test_data_file) as file:
        X_test = file[file.files[0]]

    return X_test

def save_prediction(prediction_prob):

    prediction_file = os.path.join(output_dir, 'test.predictions')

    predictions = prediction_prob[:,1]

    with open(prediction_file, 'w') as f:
        for ind, lbl in enumerate(predictions):
            str_label = str(round(lbl))
            if ind < len(predictions)-1:
                f.write(str_label + "\n")
            else:
                f.write(str_label)


def print_pretty(text):
    print("-------------------")
    print("#---",text)
    print("-------------------")


def main():
    """
     Run the pipeline
     > Load
     > Predict
     > Save
    """

    start = time.time()

    from model import Model

    print_pretty('Reading Data')
    X_test = get_prediction_data()

    print_pretty('Starting Learning')
    m = Model()
    m.load()

    print_pretty('Making Prediction')
    prediction_prob = m.predict(X_test)

    print_pretty('Saving Prediction')
    save_prediction(prediction_prob)

    duration = time.time() - start
    print_pretty(f'Total duration: {duration}')


if __name__ == '__main__':
    main()
