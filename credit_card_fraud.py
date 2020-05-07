import pickle
import sys
import numpy
import matplotlib
import pandas
import scipy
import seaborn
import sklearn

print('Python:{}'.format(sys.version))
print('Numpy.{}'.format(numpy.__version__))
print('Pandas.{}'.format(pandas.__version__))
print('Matplotlib.{}'.format(matplotlib.__version__))
print('Scipy.{}'.format(scipy.__version__))
print('Seaborn.{}'.format(seaborn.__version__))
print('Sklearn.{}'.format(sklearn.__version__))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('creditcard.csv')
print(data.columns)

data = data.sample(frac=0.1, random_state=1)
print(data.shape)
print(data.describe())
data.hist(figsize= (20, 20))
plt.show()

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]
outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)
print('Fraud cases:{}'.format(len(Fraud)))
print('valid cases:{}'.format(len(Valid)))

#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, vmax= .8, square= True)
plt.show()


#get all the coloumns from the dataframe
columns = data.columns.tolist()

#filter the coloumns to remove the data we do not need
columns = [c for c in columns if c not in ["Class"]]

#store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

#print the shapes of x and y
print(X.shape)
print(Y.shape)


#applying the algorithms to the project
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#define random state
state = 1


#define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)
}

model_file = "model.pkl"
#fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

#reshape the prediction values to 0 for valid and 1 for fraudulent
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1

        n_errors = (y_pred != Y).sum()

#run classification metrics
        print('{} : {}'.format(clf_name, n_errors))
        print(accuracy_score(Y, y_pred))
        print(classification_report(Y, y_pred))
        #print(np.array(y_pred))


#using pickle
pickle.dump(clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))








