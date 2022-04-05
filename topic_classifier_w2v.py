import collections

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, StratifiedKFold, \
    GroupKFold, ShuffleSplit, StratifiedShuffleSplit, LeaveOneOut, LeavePOut, GridSearchCV
from gensim.models import Word2Vec
import pandas as pd
import csv
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

from clsassifier_iterator import classify_all
from io_utils import initialize_argument_parser
from mean_embedding_vectorizer import MeanEmbeddingVectorizer


def print_statistics(trues, res, label=None):
    if label:
        print(f'Statistics of {label}:')
    print(f'A:\t\t\t{accuracy_score(trues, res):1.4f}')
    print(f'P_micro:\t{precision_score(trues, res, average="micro"):1.4f}')
    print(f'P_macro:\t{precision_score(trues, res, average="macro"):1.4f}')
    print(f'R_micro:\t{recall_score(trues, res, average="micro"):1.4f}')
    print(f'R_macro:\t{recall_score(trues, res, average="macro"):1.4f}')
    print(f'F1_micro:\t{f1_score(trues, res, average="micro"):1.4f}')
    print(f'F1_macro:\t{f1_score(trues, res, average="macro"):1.4f}')


def create_w2v_model(x_train: list, use_whole_text: bool, train_data_source: str):
    vector_size = 70
    window = 8
    sg = 1
    epochs = 15
    print(f'vector_size = {vector_size}')
    print(f'window = {window}')
    print(f'use_whole_text = {use_whole_text}')
    print(f'model = {train_data_source}')
    print(f'epochs = {epochs}')
    print(f'sg = {sg}')
    if use_whole_text:
        model = Word2Vec(x_train[train_data_source], vector_size=vector_size, window=window, sg=sg, epochs=epochs,
                         workers=4)
    else:
        model_sent_str = train_data_source.rstrip('_w')
        sents = []
        for art in x_train[model_sent_str].values:
            if art:
                sents.extend(art)
        model = Word2Vec(sents, vector_size=vector_size, window=window, sg=sg, epochs=epochs, workers=4)
    return model


def create_vectorizors(w2v_model, use_standard_scaler: bool = False):
    vectrs = [MeanEmbeddingVectorizer(w2v_model)]
    if use_standard_scaler:
        vectrs.append(StandardScaler())
    return vectrs


def main(use_whole_text: bool, test_data_source: str, train_data_source: str, use_cross_validation: bool,
         use_std_sclr: bool):
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
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, train_data_source)
    x_train = x_train[train_data_source]
    x_test = x_test[test_data_source]
    print(f'model_test_str = {test_data_source}')
    print(f'test_size = {test_size}')
    print(f'coef0 = {coef0}')
    print(f'use_std_sclr = {use_std_sclr}')
    classifiers = []
    classifiers.extend([
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC()), f'SVM kernel=rbf '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='linear')), f'SVM linear '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly')), f'SVM kernel=poly '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', coef0=coef0)),
         f'SVM kernel=poly coef0={coef0} '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', degree=5)),
         f'SVM kernel=poly degree=5 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(degree=5, coef0=0.75, C=10)),
         f'SVM kernel=rbf, degree=5, coef0=0.75, C=10 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(degree=5, coef0=0.2)),
         f'SVM kernel=rbf, degree=5, coef0=0.2'),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(gamma=1)),
         f'SVM kernel=rbf, gamma=1 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(gamma=1, C=10)),
         f'SVM kernel=rbf, gamma=1, C=10 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', degree=4, coef0=0.75)),
         f'SVM kernel=poly, degree=4, coef0=0.75 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       SVC(kernel='poly', degree=4, coef0=0.7, gamma=1, C=0.1)),
         f'SVM kernel=poly, degree=4, coef0=0.7, gamma=1, C=0.1 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', degree=5, coef0=0.65)),
         f'SVM kernel=poly, degree=5, coef0=0.65 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', degree=4, coef0=0.75)),
         f'SVM kernel=poly degree=4, coef0=0.75 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', degree=5, coef0=coef0)),
         f'SVM kernel=poly degree=5 coef0={coef0} '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', degree=6)),
         f'SVM kernel=poly degree=6 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), SVC(kernel='poly', degree=6, coef0=coef0)),
         f'SVM kernel=poly degree=6 coef0={coef0} '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), DecisionTreeClassifier()), 'DecisionTree '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), LogisticRegression(max_iter=1000)),
         'LogisticRegression '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), LogisticRegression(penalty="none", max_iter=1000)),
         'LogisticRegression penalty=none '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), KNeighborsClassifier()), 'KNeighbors '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), KNeighborsClassifier(weights='distance')),
         'KNeighbors weights=distance '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), AdaBoostClassifier(n_estimators=70)),
         f'AdaBoost n_estimators={70} '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr), MinMaxScaler(), MultinomialNB()), f'MultinomialNB '),
        # (make_pipeline(*create_vectorizors(model, use_std_sclr), GaussianNB()), f'GaussianNB '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier()), f'RandomForest '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(bootstrap=False)), f'RandomForest bootstrap=False '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(max_features=None)), f'RandomForest max_features=None '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(criterion='entropy')), f'RandomForest entropy '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(criterion='entropy', max_features=None)),
         f'RandomForest entropy max_features=None '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(criterion='entropy', max_features='log2')),
         f'RandomForest entropy max_features=log2 '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(criterion='entropy', bootstrap=False)),
         f'RandomForest entropy bootstrap=False '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(criterion='entropy', max_features=None, bootstrap=False)),
         f'RandomForest entropy max_features=None bootstrap=False '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(criterion='entropy', max_features='log2', bootstrap=False)),
         f'RandomForest entropy max_features=log2 bootstrap=False '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(n_estimators=150, criterion='entropy', bootstrap=False)),
         f'RandomForest n_estimators=150 entropy bootstrap=False '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(n_estimators=200, criterion='entropy', bootstrap=False)),
         f'RandomForest n_estimators=200 entropy bootstrap=False '),
        (make_pipeline(*create_vectorizors(model, use_std_sclr),
                       RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=False,
                                              max_features='log2')),
         f'RandomForest n_estimators=500, criterion=entropy, bootstrap=False, max_features=log2 ')
    ])
    classify_all(x_train,
                 x_test,
                 y_train,
                 y_test,
                 file_postfix='w2v',
                 use_cross_validation=use_cross_validation,
                 target_names=list(label_map.keys()),
                 paint_err_matr=False,
                 print_table=True,
                 classifiers=classifiers)


def test(use_whole_text: bool, test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, train_data_source)
    clsassifier = make_pipeline(MeanEmbeddingVectorizer(model), SGDClassifier())
    # clsassifier = cross_validate_(clsassifier, x_train[train_data_source], y_train)
    clsassifier.fit(x_train[train_data_source], y_train)
    y_res = clsassifier.predict(x_test[test_data_source])
    # # print(classification_report(y_test, y_res, target_names=list(label_map.keys()), digits=4))
    print(f'{"GradientBoostingClassifier;":{" "}{"<"}{57}} '
          f'P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
          f' P_macro: {precision_score(y_test, y_res, average="macro"):1.4f};'
          f' R_micro: {recall_score(y_test, y_res, average="micro"):1.4f};'
          f' R_macro: {recall_score(y_test, y_res, average="macro"):1.4f};'
          f' F1_micro: {f1_score(y_test, y_res, average="micro"):1.4f};'
          f' F1_macro: {f1_score(y_test, y_res, average="macro"):1.4f};'
          )


def test2(use_whole_text: bool, test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, train_data_source)
    k_scores = []
    k_range = []
    for k in list(range(0, 40, 2)):
        k = 0.1 * k / 2.0
        k_range.append(k)
        classifier = make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='poly', degree=5, coef0=k))
        scores = cross_val_score(classifier, x_train[train_data_source], y_train, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
        m = max(k_scores)
        if scores.mean() == max(k_scores):
            print(m, k)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of coef0 for SVC')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def grid_search(use_whole_text: bool, test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, train_data_source)
    # grid_param = {
    #     'classify__n_estimators': [100, 300, 500, 800, 1000],
    #     'classify__max_features': [None, 'log2', 'auto'],
    #     'classify__criterion': ['gini', 'entropy'],
    #     'classify__bootstrap': [True, False]
    # }
    # clsassifier = Pipeline([
    #     ('vectorizer', MeanEmbeddingVectorizer(model)),
    #     ('classify', RandomForestClassifier())
    #     ])

    grid_param = {
        # 'classify__kernel': ['poly', 'rbf'],
        'classify__degree': [5, 6, 4],
        # 'classify__coef0': [0.75, 0.7, 0.8, 0.2],
        # 'classify__C': [0.1, 1, 10],
        # 'classify__gamma': [0.1, 1, 10],
        # 'classify__shrinking': [True, False]
    }
    clsassifier = Pipeline([
        ('vectorizer', MeanEmbeddingVectorizer(model)),
        # ('vectorizer', TfidfVectorizer().fit_transform(x_train[train_data_source])),
        ('classify', SVC(kernel='poly'))
    ])

    # grid_param = {
    #     'classify__weights': ['distance', 'uniform'],
    #     'classify__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #     'classify__leaf_size': [30, 15],
    # }
    # clsassifier = Pipeline([
    #     ('vectorizer', MeanEmbeddingVectorizer(model)),
    #     ('classify', KNeighborsClassifier(n_jobs=-1))
    #     ])
    gd_sr = GridSearchCV(estimator=clsassifier,
                         param_grid=grid_param,
                         scoring='accuracy',
                         # scoring=['f1_micro', 'f1_macro'],
                         # scoring='f1_macro',
                         cv=10,
                         n_jobs=-1)
    gd_sr.fit(x_train[train_data_source], y_train)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_
    print(best_result)
    data = pd.DataFrame(gd_sr.cv_results_)
    print(data[['mean_test_score', 'std_test_score', 'params']])
    print()
    # plt.plot(data['mean_test_score'], ['mean_test_score'])
    # plt.xlabel('Value of coef0 for SVC')
    # plt.ylabel('Cross-Validated Accuracy')
    # plt.show()


def test_ir():
    iris = load_iris()
    X = iris.data
    y = iris.target
    k_range = list(range(1, 31))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
        # print(k_scores)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()



if __name__ == '__main__':
    args = initialize_argument_parser().parse_args()
    main(args.use_whole_text, args.test_data_source, args.train_data_source, args.use_cross_validation, args.use_std_sclr)
    # test(args.use_whole_text, args.test_data_source, args.train_data_source)
    # grid_search(args.use_whole_text, args.test_data_source, args.train_data_source)
    # test_ir()
    # test2(args.use_whole_text, args.test_data_source, args.train_data_source)
