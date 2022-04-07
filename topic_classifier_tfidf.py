import collections

import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, StratifiedKFold, \
    GroupKFold, ShuffleSplit, StratifiedShuffleSplit, LeaveOneOut, LeavePOut, GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from clsassifier_iterator import classify_all
from io_utils import initialize_argument_parser


def vectorizors(use_std_sclr: bool = False):
    vectrs = []
    if use_std_sclr:
        vectrs.append(StandardScaler())
    return vectrs


def main(test_data_source: str, train_data_source: str, use_cross_validation: bool, use_std_sclr: bool):
    # http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    # data = pd.read_json('articles.json')
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    print(collections.Counter(data["user_categories"]))
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    print(label_map)
    test_size = 0.2
    coef0 = 0.7
    tfidf_params = dict(min_df=10, max_df=0.8, ngram_range=(1, 2))
    tfidfconverter = TfidfVectorizer(**tfidf_params)
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    x_train = tfidfconverter.fit_transform([' '.join(t) for t in x_train[train_data_source]]).toarray()
    x_test = tfidfconverter.transform([' '.join(t) for t in x_test[test_data_source]]).toarray()
    print(f'model_test_str = {test_data_source}')
    print(f'test_size = {test_size}')
    print(f'tfidf_params = {tfidf_params}')
    print(f'use_std_sclr = {use_std_sclr}')
    classifiers = []
    classifiers.extend([
        (make_pipeline(*vectorizors(use_std_sclr), SVC()), f'SVM kernel=rbf'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='linear')), f'SVM linear'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly')), f'SVM kernel=poly'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', coef0=coef0)), f'SVM kernel=poly coef0={coef0}'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=5)), f'SVM kernel=poly degree=5'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(degree=5, coef0=0.75, C=10)),
         f'SVM kernel=rbf, degree=5, coef0=0.75, C=10'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(degree=5, coef0=0.2)), f'SVM kernel=rbf, degree=5, coef0=0.2'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(gamma=1)), f'SVM kernel=rbf, gamma=1'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(gamma=1, C=10)), f'SVM kernel=rbf, gamma=1, C=10'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=4, coef0=0.75)),
         f'SVM kernel=poly, degree=4, coef0=0.75'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=4, coef0=0.7, gamma=1, C=0.1)),
         f'SVM kernel=poly, degree=4, coef0=0.7, gamma=1, C=0.1'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=5, coef0=0.65)),
         f'SVM kernel=poly, degree=5, coef0=0.65'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=4, coef0=0.75)),
         f'SVM kernel=poly degree=4, coef0=0.75'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=5, coef0=coef0)),
         f'SVM kernel=poly degree=5 coef0={coef0}'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=6)), f'SVM kernel=poly degree=6'),
        (make_pipeline(*vectorizors(use_std_sclr), SVC(kernel='poly', degree=6, coef0=coef0)),
         f'SVM kernel=poly degree=6 coef0={coef0}'),
        (make_pipeline(*vectorizors(use_std_sclr), DecisionTreeClassifier()), 'DecisionTree'),
        (make_pipeline(*vectorizors(use_std_sclr), LogisticRegression(max_iter=1000)), 'LogisticRegression'),
        (make_pipeline(*vectorizors(use_std_sclr), LogisticRegression(penalty="none", max_iter=1000)),
         'LogisticRegression penalty=none'),
        (make_pipeline(*vectorizors(use_std_sclr), KNeighborsClassifier()), 'KNeighbors'),
        (make_pipeline(*vectorizors(use_std_sclr), KNeighborsClassifier(weights='distance')),
         'KNeighbors weights=distance'),
        (make_pipeline(*vectorizors(use_std_sclr), NearestCentroid()), 'NearestCentroid'),
        (make_pipeline(*vectorizors(use_std_sclr), NearestCentroid(metric='cosine')), 'NearestCentroid metric=cosine'),
        (make_pipeline(*vectorizors(use_std_sclr), AdaBoostClassifier(n_estimators=70)), f'AdaBoost n_estimators={70}'),
        (make_pipeline(*vectorizors(use_std_sclr), MinMaxScaler(), MultinomialNB()), f'MultinomialNB MinMaxScaler'),
        (make_pipeline(*vectorizors(use_std_sclr), MultinomialNB()), f'MultinomialNB'),
        (make_pipeline(*vectorizors(use_std_sclr), GaussianNB()), f'GaussianNB'),
        (make_pipeline(*vectorizors(use_std_sclr), RandomForestClassifier()), f'RandomForest'),
        (make_pipeline(*vectorizors(use_std_sclr), RandomForestClassifier(bootstrap=False)),
         f'RandomForest bootstrap=False'),
        (make_pipeline(*vectorizors(use_std_sclr), RandomForestClassifier(max_features=None)),
         f'RandomForest max_features=None'),
        (make_pipeline(*vectorizors(use_std_sclr), RandomForestClassifier(criterion='entropy')),
         f'RandomForest entropy'),
        (make_pipeline(*vectorizors(use_std_sclr), RandomForestClassifier(criterion='entropy', max_features=None)),
         f'RandomForest entropy max_features=None'),
        (make_pipeline(*vectorizors(use_std_sclr), RandomForestClassifier(criterion='entropy', max_features='log2')),
         f'RandomForest entropy max_features=log2'),
        (make_pipeline(*vectorizors(use_std_sclr), RandomForestClassifier(criterion='entropy', bootstrap=False)),
         f'RandomForest entropy bootstrap=False'),
        (make_pipeline(*vectorizors(use_std_sclr),
                       RandomForestClassifier(criterion='entropy', max_features=None, bootstrap=False)),
         f'RandomForest entropy max_features=None bootstrap=False'),
        (make_pipeline(*vectorizors(use_std_sclr),
                       RandomForestClassifier(criterion='entropy', max_features='log2', bootstrap=False)),
         f'RandomForest entropy max_features=log2 bootstrap=False'),
        (make_pipeline(*vectorizors(use_std_sclr),
                       RandomForestClassifier(n_estimators=150, criterion='entropy', bootstrap=False)),
         f'RandomForest n_estimators=150 entropy bootstrap=False'),
        (make_pipeline(*vectorizors(use_std_sclr),
                       RandomForestClassifier(n_estimators=200, criterion='entropy', bootstrap=False)),
         f'RandomForest n_estimators=200 entropy bootstrap=False'),
        (make_pipeline(*vectorizors(use_std_sclr),
                       RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=False,
                                              max_features='log2')),
         f'RandomForest n_estimators=500, criterion=entropy, bootstrap=False, max_features=log2')
    ])
    classify_all(x_train,
                 x_test,
                 y_train,
                 y_test,
                 file_postfix='tfidf',
                 use_cross_validation=use_cross_validation,
                 target_names=list(label_map.keys()),
                 paint_err_matr=False,
                 print_table=True,
                 classifiers=classifiers)


def test(test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    tfidfconverter = TfidfVectorizer(min_df=5, max_df=0.7)
    X_train = tfidfconverter.fit_transform([' '.join(t) for t in x_train[train_data_source]]).toarray()
    # clsassifier = RandomForestClassifier(bootstrap=False, criterion='entropy', n_estimators=100)
    clsassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1, base_estimator=SVC(kernel='linear', probability=True))
    clsassifier.fit(X_train, y_train)
    X_test = tfidfconverter.transform([' '.join(t) for t in x_test[test_data_source]]).toarray()
    y_res = clsassifier.predict(X_test)
    # print(classification_report(y_test, y_res, target_names=list(label_map.keys()), digits=4))
    print(f'{"SVC TfidfVectorizer;":{" "}{"<"}{57}} '
          f'P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
          f' P_macro: {precision_score(y_test, y_res, average="macro"):1.4f};'
          f' R_micro: {recall_score(y_test, y_res, average="micro"):1.4f};'
          f' R_macro: {recall_score(y_test, y_res, average="macro"):1.4f};'
          f' F1_micro: {f1_score(y_test, y_res, average="micro"):1.4f};'
          f' F1_macro: {f1_score(y_test, y_res, average="macro"):1.4f};'
          )


def grid_search(use_whole_text: bool, test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    tfidfconverter = TfidfVectorizer(min_df=5, max_df=0.7, ngram_range=(1, 2))
    X = tfidfconverter.fit_transform([' '.join(t) for t in data[train_data_source]]).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    grid_param = {
        'classify__n_estimators': [100, 500],
        'classify__learning_rate': [1, 0.5, 0.3]
    }
    clsassifier = Pipeline([
        ('classify', AdaBoostClassifier(base_estimator=SVC(kernel='linear', probability=True)))
    ])

    gd_sr = GridSearchCV(estimator=clsassifier,
                         param_grid=grid_param,
                         # scoring='accuracy',
                         scoring='f1_macro',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(x_train, y_train)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_
    print(best_result)
    data = pd.DataFrame(gd_sr.cv_results_)
    print(data[['mean_test_score', 'std_test_score', 'params']])
    print()


if __name__ == '__main__':
    args = initialize_argument_parser().parse_args()
    # main(args.test_data_source, args.train_data_source, args.use_cross_validation, args.use_std_sclr)
    test(args.test_data_source, args.train_data_source)
    # grid_search(args.use_whole_text, args.test_data_source, args.train_data_source)