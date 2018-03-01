from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import itertools


class Danny_ML_CLF:
    def __init__(self):
        self.X = ''
        self.y = ''

        self.svm = ''
        self.tree = ''
        self.bayes = ''
        self.knn = ''
        self.xgb = ''
        self.stacking = ''
        self.voting = ''
        self.bagging = ''
        self.rf = ''       # random forest
        self.adaboost = ''

        self.svm_pred = ''
        self.tree_pred = ''
        self.bayes_pred = ''
        self.knn_pred = ''
        self.xgb_pred = ''
        self.stacking_pred = ''
        self.voting_pred = ''
        self.bagging_pred = ''
        self.rf_pred = ''
        self.adaboost_pred = ''

        self.svm_report = ''
        self.tree_report = ''
        self.bayes_report = ''
        self.knn_report = ''
        self.xgb_report = ''
        self.stacking_report = ''
        self.voting_report = ''
        self.bagging_report = ''
        self.rf_report = ''
        self.adaboost_report = ''

        self.svm_cm = ''
        self.tree_cm = ''
        self.bayes_cm = ''
        self.knn_cm = ''
        self.xgb_cm = ''
        self.stacking_cm = ''
        self.voting_cm = ''
        self.bagging_cm = ''
        self.rf_cm = ''
        self.adaboost_cm = ''

        self.svm_score = ''
        self.tree_score = ''
        self.bayes_score = ''
        self.knn_score = ''
        self.xgb_score = ''
        self.stacking_score = ''
        self.voting_score = ''
        self.bagging_score = ''
        self.rf_score = ''
        self.adaboost_score = ''

    def Fit_value(self, x, y):
        self.X = x
        self.y = y

    def Split_data(self,raw_X, raw_y, test_size, Standard=True):
        train_X, test_X, train_y, test_y = train_test_split(raw_X, raw_y, test_size=test_size, shuffle=True)
        if Standard:
            sc = StandardScaler()
            sc.fit(train_X)
            train_X = sc.transform(train_X)
            test_X = sc.transform(test_X)
        self.X = train_X
        self.y = train_y
        return train_X, test_X, train_y, test_y

    def SVM(self,C=1,kernel='rbf'):
        self.svm = SVC(C=C,kernel=kernel, degree=5, probability=True)
        self.svm.fit(self.X, self.y)
    def SVM_predict(self,pred_x):
        self.svm_pred = self.svm.predict(pred_x)
        return  self.svm_pred

    def Tree(self,criterion='gini', max_depth=5):
        self.tree = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
        self.tree.fit(self.X, self.y)
    def Tree_predict(self, pred_x):
        self.tree_pred = self.tree.predict(pred_x)
        return  self.tree_pred

    def Bayes(self):
        self.bayes = GaussianNB()
        self.bayes.fit(self.X, self.y)
    def Bayes_predict(self, pred_x):
        self.bayes_pred = self.bayes.predict(pred_x)
        return self.bayes_pred

    def KNN(self, n_neighbors=3, weights='distance'):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self.knn.fit(self.X, self.y)
    def KNN_predict(self, pred_x):
        self.knn_pred = self.knn.predict(pred_x)
        return self.knn_pred

    def XGB(self):
        self.xgb = xgb.XGBClassifier()
        self.xgb.fit(self.X, self.y)
    def XGB_prediction(self, pred_x):
        self.xgb_pred = self.xgb.predict(pred_x)
        return self.xgb_pred

    def Stacking(self):
        meta_clf = LogisticRegression()
        self.stacking = StackingClassifier(classifiers=[self.svm,
                                                        self.tree,
                                                        self.bayes,
                                                        self.knn,
                                                        self.xgb], meta_classifier=meta_clf)
        self.stacking.fit(self.X, self.y)
    def Stacking_prediction(self, pred_x):
        self.stacking_pred = self.stacking.predict(pred_x)
        return self.stacking_pred

    def Voting(self):
        self.voting = VotingClassifier(estimators=[('svm',self.svm),
                                      ('tree',self.tree), ('bayes',self.bayes),
                                      ('knn',self.knn), ('xgb',self.xgb)],
                                      voting='soft', weights=[1,1,1,1,1])
        self.voting.fit(self.X, self.y)
    def Voting_prediction(self, pred_x):
        self.voting_pred = self.voting.predict(pred_x)
        return self.voting_pred

    def Bagging(self,n_estimators=100, oob_score=False):
        self.bagging = BaggingClassifier(n_estimators=n_estimators,oob_score=oob_score)
        self.bagging.fit(self.X, self.y)
    def Bagging_prediction(self, pred_x):
        self.bagging_pred = self.bagging.predict(pred_x)
        return self.bagging_pred

    def RF(self,n_estimators=200, criterion='gini', max_features='auto', oob_score=False):
        self.rf = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,
                                         max_features=max_features, oob_score=oob_score)
        self.rf.fit(self.X, self.y)

    def RF_prediction(self, pred_x):
        self.rf_pred = self.rf.predict(pred_x)
        return self.rf_pred

    def Adaboost(self, n_estimators=100):
        self.adaboost = AdaBoostClassifier(n_estimators=n_estimators)
        self.adaboost.fit(self.X, self.y)
    def Adaboost_prediction(self, pred_x):
        self.adaboost_pred = self.adaboost.predict(pred_x)
        return self.adaboost_pred

    def Train(self):
        self.SVM()
        self.Tree()
        self.Bayes()
        self.KNN()
        self.XGB()
        self.Stacking()
        self.Voting()
        self.Bagging()
        self.RF()
        self.Adaboost()

    def Report(self, test_X, test_y, labels, show_cm=True):
        self.SVM_predict(test_X)
        self.Tree_predict(test_X)
        self.Bayes_predict(test_X)
        self.KNN_predict(test_X)
        self.XGB_prediction(test_X)
        self.Stacking_prediction(test_X)
        self.Voting_prediction(test_X)
        self.Bagging_prediction(test_X)
        self.RF_prediction(test_X)
        self.Adaboost_prediction(test_X)

        self.svm_score = self.svm.score(test_X, test_y)
        self.tree_score = self.tree.score(test_X, test_y)
        self.bayes_score = self.bayes.score(test_X, test_y)
        self.knn_score = self.knn.score(test_X, test_y)
        self.xgb_score = self.xgb.score(test_X, test_y)
        self.stacking_score = self.stacking.score(test_X, test_y)
        self.voting_score = self.voting.score(test_X, test_y)
        self.bagging_score = self.bagging.score(test_X, test_y)
        self.rf_score = self.rf.score(test_X, test_y)
        self.adaboost_score = self.adaboost.score(test_X, test_y)


        self.svm_report = metrics.classification_report(test_y, self.svm_pred)
        self.tree_report = metrics.classification_report(test_y, self.tree_pred)
        self.bayes_report = metrics.classification_report(test_y, self.bayes_pred)
        self.knn_report = metrics.classification_report(test_y, self.knn_pred)
        self.xgb_report = metrics.classification_report(test_y, self.xgb_pred)
        self.voting_report = metrics.classification_report(test_y, self.voting_pred)
        self.stacking_report = metrics.classification_report(test_y, self.stacking_pred)
        self.bagging_report = metrics.classification_report(test_y, self.bagging_pred)
        self.rf_report = metrics.classification_report(test_y, self.rf_pred)
        self.adaboost_report = metrics.classification_report(test_y, self.adaboost_pred)

        self.svm_cm = metrics.confusion_matrix(test_y, self.svm_pred,labels=labels)
        self.tree_cm = metrics.confusion_matrix(test_y, self.tree_pred,labels=labels)
        self.bayes_cm = metrics.confusion_matrix(test_y, self.bayes_pred,labels=labels)
        self.knn_cm = metrics.confusion_matrix(test_y, self.knn_pred,labels=labels)
        self.xgb_cm = metrics.confusion_matrix(test_y, self.xgb_pred, labels=labels)
        self.stacking_cm = metrics.confusion_matrix(test_y, self.stacking_pred, labels=labels)
        self.voting_cm = metrics.confusion_matrix(test_y, self.voting_pred, labels=labels)
        self.bagging_cm = metrics.confusion_matrix(test_y, self.bagging_pred, labels=labels)
        self.rf_cm = metrics.confusion_matrix(test_y, self.rf_pred, labels=labels)
        self.adaboost_cm = metrics.confusion_matrix(test_y, self.adaboost_pred, labels=labels)

        if show_cm:
            self.plot_confusion_matrix(self.svm_cm, classes=labels, title='SVM')
            self.plot_confusion_matrix(self.tree_cm, classes=labels, title='Tree')
            self.plot_confusion_matrix(self.bayes_cm, classes=labels, title='Bayes')
            self.plot_confusion_matrix(self.knn_cm, classes=labels, title='KNN')
            self.plot_confusion_matrix(self.xgb_cm, classes=labels, title='XGB')
            self.plot_confusion_matrix(self.stacking_cm, classes=labels, title='Stacking')
            self.plot_confusion_matrix(self.voting_cm, classes=labels, title='Voting')
            self.plot_confusion_matrix(self.bagging_cm, classes=labels, title='Bagging')
            self.plot_confusion_matrix(self.rf_cm, classes=labels, title='RF')
            self.plot_confusion_matrix(self.adaboost_cm, classes=labels, title='Adaboost')

    def History(self):
        print('******************\nSVM : ',self.svm_report)
        print('******************\nTree : ',self.tree_report)
        print('******************\nBayes : ',self.bayes_report)
        print('******************\nKNN : ',self.knn_report)
        print('******************\nXGB : ', self.xgb_report)
        print('******************\nStacking : ', self.stacking_report)
        print('******************\nVoting : ', self.voting_report)
        print('******************\nBagging : ', self.bagging_report)
        print('******************\nRF : ', self.rf_report)
        print('******************\nAdaboost : ', self.adaboost_report)

    def Score(self):
        print('SVM Score : ', self.svm_score)
        print('Tree Score : ', self.tree_score)
        print('Bayes Score : ', self.bayes_score)
        print('KNN Score : ', self.knn_score)
        print('XGB Score : ', self.xgb_score)
        print('Stacking Score : ', self.stacking_score)
        print('Voting Score : ', self.voting_score)
        print('Bagging Score : ', self.bagging_score)
        print('RF Score : ', self.rf_score)
        print('Adaboost Score : ', self.adaboost_score)

    def Report2txt(self, filename):
        f = open(filename, 'w')
        f.write('SVM Score : '+ str(self.svm_score) + '\n')
        f.write('Tree Score : '+ str(self.tree_score) +'\n')
        f.write('Bayes Score : '+ str(self.bayes_score) + '\n')
        f.write('KNN Score : '+ str(self.knn_score) + '\n')
        f.write('XGB Score : '+ str(self.xgb_score) + '\n')
        f.write('Stacking Score : '+ str(self.stacking_score) + '\n')
        f.write('Voting Score : '+ str(self.voting_score) + '\n')
        f.write('Bagging Score : '+ str(self.bagging_score) + '\n')
        f.write('RF Score : '+ str(self.rf_score) + '\n')
        f.write('Adaboost Score : '+ str(self.adaboost_score) + '\n')
        f.write('Adaboost Score : '+ str(self.adaboost_score) + '\n')
        f.write('XXXX\n')
        f.write('******************\nSVM : '+ str(self.svm_report) + '\n')
        f.write('******************\nTree : '+ str(self.tree_report) + '\n')
        f.write('******************\nBayes : '+ str(self.bayes_report) + '\n')
        f.write('******************\nKNN : '+ str(self.knn_report) + '\n')
        f.write('******************\nXGB : '+ str(self.xgb_report) + '\n')
        f.write('******************\nStacking : '+ str(self.stacking_report) + '\n')
        f.write('******************\nVoting : '+ str(self.voting_report) + '\n')
        f.write('******************\nBagging : '+ str(self.bagging_report) + '\n')
        f.write('******************\nRF : '+ str(self.rf_report) + '\n')
        f.write('******************\nAdaboost : '+ str(self.adaboost_report) + '\n')
        f.close()

    def plot_confusion_matrix(self,cm, classes,normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
        """
           This function prints and plots the confusion matrix.
           Normalization can be applied by setting `normalize=True`.
           """
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print(title, ' Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        # Source code from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py


