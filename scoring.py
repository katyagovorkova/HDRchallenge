#------------------------------------------
# Imports
#------------------------------------------
import sys
import os
import json
import numpy as np



#------------------------------------------
# Import Metric
#------------------------------------------
# from metric import auc_metric


#------------------------------------------
# constants
#------------------------------------------
data_name = "fair_universe"

#------------------------------------------
# directories
#------------------------------------------
# Directory read predictions and solutions from
input_dir = 'input_data/' 

# Directory to output computed score into
output_dir = 'output/'

#------------------------------------------
# Read Predictions
#------------------------------------------
def read_prediction():
    prediction_file = os.path.join(output_dir,'test.predictions')


    # Check if file exists
    if not os.path.isfile(prediction_file):
        print('[-] Test prediction file not found!')
        return


    f = open(prediction_file, "r")
    
    predicted_scores = f.read().splitlines()
    predicted_scores = np.array(predicted_scores,dtype=float)
    
    return predicted_scores

#------------------------------------------
# Read Solutions
#------------------------------------------
def read_solution():

    solution_file = os.path.join(input_dir, 'test.labels')

    # Check if file exists
    if not os.path.isfile(solution_file):
        print('[-] Test solution file not found!')
        return

    f = open(solution_file, "r")
    
    test_labels = f.read().splitlines()
    test_labels = np.array(test_labels,dtype=float)

    return test_labels

def save_score(score):
    score_file = os.path.join(output_dir, 'scores.json')

    scores = {
        'accuracy': score,
    }
    with open(score_file, 'w') as f_score:
        f_score.write(json.dumps(scores))
        f_score.close()

def print_pretty(text):
    print("-------------------")
    print("#---",text)
    print("-------------------")


    
def main():


    #------------------------------------------
    # Read prediction and solution
    #------------------------------------------
    print_pretty('Reading prediction')
    prediction = read_prediction()
    solution = read_solution()


    #------------------------------------------
    # Compute Score
    #------------------------------------------
    print_pretty('Computing score')
    score = (prediction == solution).sum() / len(prediction)
    print("Accuracy: ", score)

    #------------------------------------------
    # Write Score
    #------------------------------------------
    print_pretty('Saving prediction')
    save_score(score)



if __name__ == '__main__':
    main()
