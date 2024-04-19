from sklearn.ensemble import RandomForestClassifier
import pickle

class Model:
    def __init__(self):
        # Include a constructor sufficient to implement the base classifier, could also load (see method below)
        self.clf = RandomForestClassifier(max_depth=4, random_state=0)

    def fit(self, X, y):
        # Must have a method to fit the model to given input data and labels
        self.clf.fit(X=X, y=y)

    def predict(self, X):
        # Must have a method to predict classes given input data
        return self.clf.predict(X)

    def load(self, filename):
        # Must be able to load a pretrained model
        self.clf = pickle.load(open(filename, 'rb'))
