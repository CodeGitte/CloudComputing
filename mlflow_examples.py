#MLflow
#four different components: MLflow Tracking, Projects, Models, Repostitory

#import required modules
from sklearn import datasets as ds 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import mlflow
import pickle 


#load the sample data
data = ds.load_wine() #inbuilt dataset with sklearn

#split the data
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size = 0.2)

#################

#FIRST EXPERIMENT

#MLflow has experiments and runs, an experiment can hold any amount of runs with varying hyperparameters
#create a MLflow experiment
mlflow.create_experiment('RF_classification_03')
mlflow.set_experiment('RF_classification_03')

#log model parameters to MLflow
mlflow.log_param('maximum_depth', 5)
mlflow.log_param('criterion', 'gini')

#train a classifier
clf = RandomForestClassifier(max_depth=5, criterion='gini')
clf.fit(X_train, y_train)

#calculate the accuracy
y_pred = clf.predict(X_test) #predicts the classification for the testing data 
acc = metrics.accuracy_score(y_test, y_pred) #calculates accuracy 

#log accuracy to MLflow
mlflow.log_metric('Accuracy', acc)

#save the model as a pickle file and log it as an artifact to MLflow
with open('model_1.pkl', 'wb') as file:
    pickle.dump(clf, file)
mlflow.log_artifact('model_1.pkl')

#end the run
mlflow.end_run()

#################

#SECOND EXPERIMENT
#train a second model and use MLflow autlogging
mlflow.autolog()

#train a classifier
clf = RandomForestClassifier(max_depth=10, criterion="entropy")
clf.fit(X_train, y_train)

#calucate accuracy
y_pred = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)

#in cli: mlflow ui > localhost:5000 to compare the various models 

#################################################################################

#little recap on random forest (ensembling of decision trees)
#categorical > classification tree 
#root node: top of the tree
#decision node: node with statement 
#leaf node: arrows pointing to them, not pointing away from them /end node 
#method 1: bagging: first bootstrapping from sample to create smaller random subsets with replacement, from those random subsets decision trees are calculated and the outcomes are aggregrated 
#method 2: feature randomness encourages diverse trees
#> model is less sensitive to orginial data (preventes overfitting), very important: uncorrelated trees

#gini impurity: 1 - (probablilty of no)2 - (probablilty of yes)2 
#total gini impurity = weigthed average of gini impurities for the leaves 
#A Gini Impurity of 0 is the lowest and best possible impurity. It can only be achieved when everything is the same class (e.g. only blues or only greens).


#recap on accuracy 
#(TP+TN)/(TP+TN+FP+FN)

#recap on pickle 
#binary serialization format