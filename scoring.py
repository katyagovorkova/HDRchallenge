import json
import os
import numpy as np

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

print('Reading prediction')
prediction = np.load(os.path.join(prediction_dir, 'submission.npy')).flatten()
truth = np.load(os.path.join(reference_dir, 'answer.npy')).flatten()

print('Checking Accuracy')
accuracy = np.abs(np.sum(truth - prediction))/50500*100
print('Scores:')
scores = {
    'accuracy': accuracy,
    'duration': 0
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
