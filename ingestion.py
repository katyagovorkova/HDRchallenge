#------------------------------------------
# Imports
#------------------------------------------
import sys
import os
import numpy as np
import pandas as pd 
import time



#------------------------------------------
# Directories
#------------------------------------------
# Input data directory to read training data from
input_dir = 'input_data/'

# Output data directory to write predictions to
output_dir = 'output/'

sys.path.append(output_dir)

#------------------------------------------
# Read Train Data
#------------------------------------------
def get_training_data():

    # set train data and solution file
    train_data_file = os.path.join(input_dir, 'train.csv')
    train_solution_file = os.path.join(input_dir, 'train.labels')
    
    # Read Train data
    X_train = pd.read_csv(train_data_file)

    # Read Train solution
    f = open(train_solution_file, "r")
    y_train = f.read().splitlines()
    y_train = np.array(y_train,dtype=float)


    return X_train, y_train

#------------------------------------------
# Read Test Data
#------------------------------------------
def get_prediction_data():

    # set test data and solution file
    test_data_file = os.path.join(input_dir, 'test.csv')

    # Read Test data
    X_test = pd.read_csv(test_data_file)

    return X_test

#------------------------------------------
# Save Predictions
#------------------------------------------
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

#------------------------------------------
# Run the pipeline 
# > Load 
# > Train 
# > Predict 
# > Save
#------------------------------------------
def main():

    #------------------------------------------
    # Start Timer
    #------------------------------------------
    start = time.time()


    #------------------------------------------
    # Import Model
    #------------------------------------------
    from model import Model
   
    #------------------------------------------
    # Read Data
    #------------------------------------------
    print_pretty('Reading Data')
    X_train, y_train = get_training_data()
    X_test = get_prediction_data()


    #------------------------------------------
    # Load Model
    #------------------------------------------
    print_pretty('Starting Learning')
    m = Model()

    #------------------------------------------
    # Train Model
    #------------------------------------------
    print_pretty('Training Model')
    m.fit(X_train, y_train)

    #------------------------------------------
    # Make Predictions
    #------------------------------------------
    print_pretty('Making Prediction')
    prediction_prob = m.predict_score(X_test)

    #------------------------------------------
    # Save  Predictions
    #------------------------------------------
    print_pretty('Saving Prediction')
    save_prediction(prediction_prob)


    #------------------------------------------
    # Show Ingestion Time
    #------------------------------------------
    duration = time.time() - start
    print_pretty(f'Total duration: {duration}')

if __name__ == '__main__':
    main()
