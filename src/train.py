import sys
import os
import yaml
from sklearn.naive_bayes import MultinomialNB
import pickle

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython3 train.py features-dir-path model-filename\n')
    sys.exit(1)

features_path = sys.argv[1]
model_filename = sys.argv[2]

params = yaml.safe_load(open('params.yaml'))['train']
alpha = params['alpha']

features_train_pkl = os.path.join(features_path, 'train.pkl')
with open(features_train_pkl, 'rb') as f:
    train_data = pickle.load(f)

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

clf = MultinomialNB(alpha=alpha)
clf.fit(X, y)

with open(model_filename, 'wb') as f:
    pickle.dump(clf, f)
