#Feature selection using various Nature inspired algorithms for clickbait identification

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import CamelAlgorithm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import resultpresentation as result
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
import pandas as pd
import numpy as np
from numpy import load
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE

#Task Definition

class CamelFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.9):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(GaussianNB(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

#Data Preperation
print("Reading and preparing Data")
data= pd.read_csv("FeaturesAndLabel.csv")
label=data["targetClass Label"]
features=data.drop(['targetClass Label', 'id'], axis=1)

sm = BorderlineSMOTE(kind='borderline-1',random_state=42)
X_res, y_res = sm.fit_resample(features, label)

feat=np.asarray(X_res)
lab=np.asarray(y_res)

feature_names = np.asarray(list(features))
classes_list = ['Non-clickbait', 'Clickbait']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feat,label, test_size=0.2, random_state=89)


#Feature Optimization
print("Camel Algorithm running")
problem = CamelFeatureSelection(X_train, y_train)
task = Task(problem, max_iters=500)
algorithm = CamelAlgorithm(population_size=25,burden_factor=0.5, death_rate=0.5, visibility=0.5, supply_init=10, endurance_init=10, min_temperature=-10, max_temperature=10,seed=211)
best_features, best_fitness = algorithm.run(task)

#Optimized Features
selected_features = best_features > 0.5
print('Number of selected features:', selected_features.sum())
print('Selected features:', ', '.join(feature_names[selected_features].tolist()))

#Data Splitting for Training and Testing
X_train = load('X_train.npy')
y_train= load('y_train.npy')
X_test = load('X_test.npy')
y_test= load('y_test.npy')

#Moel Selection, Training and Testing
model_selected = SVC()
model_all = SVC()
model_selected.fit(X_train[:, selected_features], y_train)
y_pred_SVM = model_selected.predict(X_test[:, selected_features])
model_all.fit(X_train, y_train)
print('All Features Accuracy:', model_all.score(X_test, y_test))
print("SVM")
print(metrics.classification_report(y_test, y_pred_SVM))
print("Accuracy score =", accuracy_score(y_test, y_pred_SVM))
plt.clf()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_SVM)
result.plot_confusion_matrix(cnf_matrix, classes = classes_list, normalize = True)
plt.savefig("click_confusion_matrix_svm_camel.pdf", format = 'pdf', dpi =1000)
plt.clf()
result.plot_AUC_ROC(y_test, y_pred_SVM)
plt.savefig("click_ROC_svm_camel.pdf", format = 'pdf', dpi =1000)

from sklearn.ensemble import RandomForestClassifier
model_selected = RandomForestClassifier(n_estimators=50,max_depth=None,min_samples_split=20, random_state=0)
model_selected.fit(X_train[:, selected_features], y_train)
y_pred_rf = model_selected.predict(X_test[:, selected_features])
print("random")
print(metrics.classification_report(y_test, y_pred_rf))
print("Accuracy score =", accuracy_score(y_test, y_pred_rf))
plt.clf()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_rf)
result.plot_confusion_matrix(cnf_matrix, classes = classes_list, normalize = True)
plt.savefig("click_confusion_matrix_rf_camel.pdf", format = 'pdf', dpi =1000)
plt.clf()
result.plot_AUC_ROC(y_test, y_pred_rf)
plt.savefig("click_ROC_rf_camel.pdf", format = 'pdf', dpi =1000)

    
    
from sklearn.linear_model import LogisticRegression
model_selected = LogisticRegression()
model_selected.fit(X_train[:, selected_features], y_train)
y_pred_LR = model_selected.predict(X_test[:, selected_features])
print("Logistic Regression")
print(metrics.classification_report(y_test, y_pred_LR ))
print("Accuracy score =", accuracy_score(y_test, y_pred_LR))
plt.clf()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_LR)
result.plot_confusion_matrix(cnf_matrix, classes = classes_list, normalize = True)
plt.savefig("click_confusion_matrix_LR_camel.pdf", format = 'pdf', dpi =1000)
plt.clf()
result.plot_AUC_ROC(y_test, y_pred_LR)
plt.savefig("click_ROC_LR_camel.pdf", format = 'pdf', dpi =1000)

from sklearn.naive_bayes import GaussianNB
model_selected = GaussianNB()
model_selected.fit(X_train[:, selected_features], y_train)
y_pred_naive = model_selected.predict(X_test[:, selected_features])
print("Naive Bayes")
print(metrics.classification_report(y_test, y_pred_naive ))
print("Accuracy score =", accuracy_score(y_test, y_pred_naive))
plt.clf()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_naive)
result.plot_confusion_matrix(cnf_matrix, classes = classes_list, normalize = True)
plt.savefig("click_confusion_matrix_NB_camel.pdf", format = 'pdf', dpi =1000)
plt.clf()
result.plot_AUC_ROC(y_test, y_pred_naive)
plt.savefig("click_ROC_NB_camel.pdf", format = 'pdf', dpi =1000)


from sklearn.ensemble import GradientBoostingClassifier
model_selected = GradientBoostingClassifier(n_estimators=50,max_depth=None,min_samples_split=20, random_state=0)
model_selected.fit(X_train[:, selected_features], y_train)
y_pred_gradient = model_selected.predict(X_test[:, selected_features])
print("Gradient Boosting")
print(metrics.classification_report(y_test, y_pred_gradient ))
print("Accuracy score =", accuracy_score(y_test, y_pred_gradient))
plt.clf()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_gradient)
result.plot_confusion_matrix(cnf_matrix, classes = classes_list, normalize = True)
plt.savefig("click_confusion_matrix_GB_camel.pdf", format = 'pdf', dpi =1000)
plt.clf()
result.plot_AUC_ROC(y_test, y_pred_gradient)
plt.savefig("click_ROC_GB_camel.pdf", format = 'pdf', dpi =1000)


from sklearn.tree import DecisionTreeClassifier
model_selected = DecisionTreeClassifier()
model_selected.fit(X_train[:, selected_features], y_train)
y_pred_decision = model_selected.predict(X_test[:, selected_features])
print("Decision Tree")
print(metrics.classification_report(y_test, y_pred_decision ))
print("Accuracy score =", accuracy_score(y_test, y_pred_decision))
plt.clf()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_decision)
result.plot_confusion_matrix(cnf_matrix, classes = classes_list, normalize = True)
plt.savefig("click_confusion_matrix_DT_camel.pdf", format = 'pdf', dpi =1000)
plt.clf()
result.plot_AUC_ROC(y_test, y_pred_decision)
plt.savefig("click_ROC_DT_camel.pdf", format = 'pdf', dpi =1000)

