import sys
import os
import json
import numpy as np

# Directory to read labels from
input_dir = sys.argv[1]

# Directory to output computed score into
output_dir = sys.argv[2]


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


def read_solution():

    solution_file = os.path.join(input_dir, 'ligo_blackbox.npz')

    # Check if file exists
    if not os.path.isfile(solution_file):
        print('[-] Test solution file not found!')
        return

    with np.load(solution_file) as file:
        test_labels = file[file.files[1]]
    
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

    # Read prediction and solution
    print_pretty('Reading prediction')
    prediction = read_prediction()
    solution = read_solution()

    # Compute Score
    print_pretty('Computing score')
    score = (prediction == solution).sum() / len(prediction)
    print("Accuracy: ", score)

    # Write Score
    print_pretty('Saving prediction')
    save_score(score)


if __name__ == '__main__':
    main()
