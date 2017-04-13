import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


classifier=[tree.DecisionTreeClassifier(splitter='best', max_depth=5),  # 1
            Perceptron(n_iter=15),   # 2
            MLPClassifier(hidden_layer_sizes=(4, 2), learning_rate_init=0.01),  # 3
            MLPClassifier(hidden_layer_sizes=(15, 6), learning_rate_init=0.01),  # 4
            svm.SVC(kernel='linear'),  # 5
            MultinomialNB(fit_prior=True),  # 6
            LogisticRegression(tol=.00001),  # 7
            KNeighborsClassifier(n_neighbors=5, weights='distance'),  # 8
            BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100),  # 9
            RandomForestClassifier(n_estimators=100, max_features=2),  # 10
            AdaBoostClassifier(n_estimators=60),  # 11
            GradientBoostingClassifier(n_estimators=50) ]  # 12

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
columns = ['Pregnant', 'PlasmaGlucose', 'DiastolicBP', 'SkinFoldThickness', 'Insulin', 'BMI', 'DiabetesPedigreeF', 'Age', 'Class']
dataFrame = pandas.read_csv(url, names=columns)
array = dataFrame.values
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(array[:,0:8])
Y = array[:,8]

print ("Number of Instances in dataset:", dataFrame.shape[0]);
print ("Number of attributes in dataset:", dataFrame.shape[1]-1);
print("Number of  fold cross-validation performed:",10)
i=1
print("\nSNo", " ", "Average Accuracy", " ", "Average Precision")
for model in classifier:
    accuracy = model_selection.cross_val_score(model, X, Y, cv=10)
    precision = model_selection.cross_val_score(model, X, Y, cv=10, scoring='average_precision')
    print(i,"  ",accuracy.mean() , "  " ,precision.mean())
    i=i+1