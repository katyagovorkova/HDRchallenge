import sys
import os
import numpy as np
import pandas as pd
import time

# Input data directory to read training data from
input_dir = 'input_data/'

# Output data directory to write predictions to
output_dir = 'output/'

sys.path.append(output_dir)


def get_prediction_data():

    # set test data and solution file
    test_data_file = os.path.join(input_dir, 'test.csv')

    # Read Test data
    X_test = pd.read_csv(test_data_file)

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
     > Train
     > Predict
     > Save
    """

    start = time.time()

    from model import Model

    print_pretty('Reading Data')
    X_test = get_prediction_data()

    print_pretty('Starting Learning')
    m = Model()
    # the model should be loaded here !!

    print_pretty('Making Prediction')
    prediction_prob = m.predict_score(X_test)

    print_pretty('Saving Prediction')
    save_prediction(prediction_prob)

    duration = time.time() - start
    print_pretty(f'Total duration: {duration}')


if __name__ == '__main__':
    main()
