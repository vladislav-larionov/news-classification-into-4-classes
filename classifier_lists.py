from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def full_classifier_list(vectorizors):
    return [
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',)), f'SVM kernel=rbf'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='linear')), f'SVM linear'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly')), f'SVM kernel=poly'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', coef0=0.75)),
         f'SVM kernel=poly coef0=0.75'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',gamma=1)),
         f'SVM kernel=rbf gamma=1'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',gamma=1, C=10)),
         f'SVM kernel=rbf gamma=1 C=10'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',gamma=0.75, C=10)),
         f'SVM kernel=rbf gamma=0.75 C=10'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=4, coef0=0.75)),
         f'SVM kernel=poly degree=4 coef0=0.75'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=4, coef0=0.7)),
         f'SVM kernel=poly degree=4 coef0=0.7'),
        (make_pipeline(*vectorizors,
                       SVC(class_weight='balanced',kernel='poly', degree=4, coef0=0.7, gamma=1, C=0.1)),
         f'SVM kernel=poly degree=4 coef0=0.7 gamma=1 C=0.1'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',degree=5, coef0=0.75, C=10)),
         f'SVM kernel=rbf degree=5 coef0=0.75, C=10'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',degree=5, coef0=0.2)),
         f'SVM kernel=rbf degree=5 coef0=0.2'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=5)),
         f'SVM kernel=poly degree=5'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=5, coef0=0.65)),
         f'SVM kernel=poly degree=5 coef0=0.65'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=5, coef0=0.75)),
         f'SVM kernel=poly degree=5 coef0=0.75'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=5, coef0=0.7)),
         f'SVM kernel=poly degree=5 coef0=0.7'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=6)),
         f'SVM kernel=poly degree=6'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=6, coef0=0.75)),
         f'SVM kernel=poly degree=6 coef0=0.75'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced',kernel='poly', degree=6, coef0=0.7)),
         f'SVM kernel=poly degree=6 coef0=0.7'),
        (make_pipeline(*vectorizors, LogisticRegression(class_weight='balanced',max_iter=1000)),
         'LogisticRegression'),
        (make_pipeline(*vectorizors, LogisticRegression(class_weight='balanced',penalty="none", max_iter=1000)),
         'LogisticRegression penalty=none'),
        (make_pipeline(*vectorizors, KNeighborsClassifier()), 'KNeighbors'),
        (make_pipeline(*vectorizors, KNeighborsClassifier(weights='distance')),
         'KNeighbors weights=distance'),
        (make_pipeline(*vectorizors, ExtraTreesClassifier(class_weight='balanced', n_estimators=500)),
         f'ExtraTreesClassifier class_weight=balanced n_estimators=500'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(class_weight='balanced',)), f'RandomForest'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(bootstrap=False)), f'RandomForest bootstrap=False'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(max_features=None)), f'RandomForest max_features=None'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(criterion='entropy')), f'RandomForest entropy'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(criterion='entropy', max_features=None)),
         f'RandomForest entropy max_features=None'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(criterion='entropy', max_features='log2')),
         f'RandomForest entropy max_features=log2'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(criterion='entropy', bootstrap=False)),
         f'RandomForest entropy bootstrap=False'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(criterion='entropy', max_features=None, bootstrap=False)),
         f'RandomForest entropy max_features=None bootstrap=False'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(criterion='entropy', max_features='log2', bootstrap=False)),
         f'RandomForest entropy max_features=log2 bootstrap=False'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(n_estimators=150, criterion='entropy', bootstrap=False)),
         f'RandomForest n_estimators=150 entropy bootstrap=False'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(n_estimators=200, criterion='entropy', bootstrap=False)),
         f'RandomForest n_estimators=200 entropy bootstrap=False'),
        (make_pipeline(*vectorizors,
                       RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=False,
                                              max_features='log2')),
         f'RandomForest n_estimators=500 criterion=entropy bootstrap=False max_features=log2')
    ]


def short_classifier_list(vectorizors):
    return [
        (make_pipeline(*vectorizors, SVC(class_weight='balanced', gamma=1, C=10)),
         f'SVM class_weight=balanced kernel=rbf gamma=1 C=10'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced', kernel='poly', degree=4, coef0=0.7, gamma=1, C=0.1)),
         f'SVM class_weight=balanced kernel=poly degree=4 coef0=0.7 gamma=1 C=0.1'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced', kernel='poly', degree=5, coef0=0.75)),
         f'SVM class_weight=balanced kernel=poly degree=5 coef0=0.75'),
        (make_pipeline(*vectorizors, SVC(class_weight='balanced', kernel='poly', degree=5, coef0=0.75, C=10)),
         f'SVM class_weight=balanced kernel=poly degree=5 coef0=0.75, C=10'),
        (make_pipeline(*vectorizors, SVC(gamma=1, C=10)),
         f'SVM kernel=rbf gamma=1 C=10'),
        (make_pipeline(*vectorizors, SVC(kernel='poly', degree=4, coef0=0.7, gamma=1, C=0.1)),
         f'SVM kernel=poly degree=4 coef0=0.7 gamma=1 C=0.1'),
        (make_pipeline(*vectorizors, SVC(kernel='poly', degree=5, coef0=0.75)),
         f'SVM kernel=poly degree=5 coef0=0.75'),
        (make_pipeline(*vectorizors, SVC(kernel='poly', degree=5, coef0=0.75, C=10)),
         f'SVM kernel=poly degree=5 coef0=0.75, C=10'),
        (make_pipeline(*vectorizors, SVC(kernel='poly', degree=4, coef0=0.75)),
         f'SVM kernel=poly degree=4 coef0=0.75'),
        (make_pipeline(*vectorizors, LogisticRegression(penalty="none", max_iter=1000)),
         'LogisticRegression penalty=none'),
        (make_pipeline(*vectorizors, KNeighborsClassifier(weights='distance')),
         'KNeighbors weights=distance')
    ]