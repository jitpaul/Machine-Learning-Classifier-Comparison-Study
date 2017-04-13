import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# parameters
k=10

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataFrame = pandas.read_csv(url, names=names)
array = dataFrame.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kFold = model_selection.KFold(n_splits=k, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kFold)
print(results.mean())