import collections

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.base import TransformerMixin
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from gensim.models import Word2Vec
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from classifier_lists import full_classifier_list, short_classifier_list
from clsassifier_iterator import classify_all
from io_utils import initialize_argument_parser
from mean_embedding_vectorizer import MeanEmbeddingVectorizer
from utils import form_y_prep, form_label_map, create_res_dir, parse_arguments, print_info, form_res_path
from w2v_vectorizer import Word2VecModel, Doc2VecModel


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
    print(f'epochs = {epochs}')
    print(f'sg = {sg}')
    if use_whole_text:
        model = Word2Vec(x_train[train_data_source], vector_size=vector_size, window=window, sg=sg, epochs=epochs,
                         workers=4)
    else:
        sents = []
        for art in x_train[train_data_source.rstrip('_w')].values:
            if art:
                sents.extend(art)
        model = Word2Vec(sents, vector_size=vector_size, window=window, sg=sg, epochs=epochs, workers=4)
    return model


def classify_with_w2v(**params):
    # http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    train_data_source = params['train_data_source']
    test_data_source = params.get('test_data_source', train_data_source)
    use_whole_text = params.get('use_whole_text', False)
    use_short_classifiers_list = params.get('short', False)
    use_std_sclr = params.get('use_std_sclr', False)
    res_dir = params.get('res_dir')
    use_pca = params.get('use_pca', False)
    save_err_matr = params.get('save_err_matr', True)
    test_size = 0.2
    n_components = 30
    print_info(**params, file=None, test_size=test_size, n_components=n_components,
               model_info=dict(vector_size=70, window=8, sg=1, epochs=15))
    data = pd.read_json('articles_w_m_t.json')
    y_prep = form_y_prep(data["user_categories"])
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, train_data_source)
    x_train = x_train[train_data_source]
    x_test = x_test[test_data_source]
    vectorizors = [MeanEmbeddingVectorizer(model)]
    if use_std_sclr:
        vectorizors.append(StandardScaler())
    if use_pca:
        vectorizors.append(PCA(n_components=n_components))
    if use_short_classifiers_list:
        classifiers = short_classifier_list(vectorizors)
    else:
        classifiers = full_classifier_list(vectorizors)
    if use_whole_text:
        file_postfix = 'whole_text'
    else:
        file_postfix = 'sentences'
    res_dir_path = form_res_path(res_dir, train_data_source, test_data_source)
    res_dir = create_res_dir(f'{res_dir_path}/w2v_{file_postfix}')
    print_info(**params, file=f'{res_dir_path}/w2v_{file_postfix}/info.txt', test_size=test_size,
               n_components=n_components,
               model_info=dict(vector_size=70, window=8, sg=1, epochs=15))
    classify_all(x_train,
                 x_test,
                 y_train,
                 y_test,
                 classifiers=classifiers,
                 res_dir=res_dir,
                 target_names=list(form_label_map(data["user_categories"]).keys()),
                 save_err_matr=save_err_matr,
                 paint_err_matr=False,
                 print_table=False)


def test(use_whole_text: bool, test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, train_data_source)
    # clsassifier = make_pipeline(MeanEmbeddingVectorizer(model), PCA(n_components=50), LinearSVC())
    clsassifier = make_pipeline(MeanEmbeddingVectorizer(model),
                                ExtraTreesClassifier(class_weight='balanced', n_estimators=500))
    clsassifier.fit(x_train[train_data_source], y_train)
    y_res = clsassifier.predict(x_test[test_data_source])
    print(f'{"ExtraTreesClassifier;":{" "}{"<"}{57}} '
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
    # model = create_w2v_model(x_train, use_whole_text, train_data_source)
    classifier = make_pipeline(Doc2VecModel(), SVC(kernel='poly', degree=5, coef0=0.75))
    classifier.fit(x_train[train_data_source], y_train)
    y_res = classifier.predict(x_test[test_data_source])
    print(f'{"SVC(kernel=poly, degree=5, coef0=0.75);":{" "}{"<"}{57}} '
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
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, train_data_source)
    grid_param = {
        # 'classify__metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
        # 'v__vector_size': [70],
        'c__C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'c__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        'c__kernel': ['rbf', 'poly', 'linear'],
    }
    clsassifier = Pipeline([
        ('v', MeanEmbeddingVectorizer(model)),
        ('c', SVC())
    ])
    gd_sr = GridSearchCV(estimator=clsassifier,
                         param_grid=grid_param,
                         # scoring='accuracy',
                         # scoring='f1_micro',
                         scoring='f1_macro',
                         refit=True, verbose=1, cv=5,
                         n_jobs=-1)
    gd_sr.fit(x_train[train_data_source], y_train)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_
    print(best_result)
    data = pd.DataFrame(gd_sr.cv_results_)
    print(data[['mean_test_score', 'std_test_score', 'params']])


if __name__ == '__main__':
    # args = initialize_argument_parser().parse_args()
    args = parse_arguments()
    # classify_with_w2v(args.use_whole_text, args.test_data_source, args.train_data_source, args.use_std_sclr)
    classify_with_w2v(vars(args))
    # grid_search(args.use_whole_text, args.test_data_source, args.train_data_source)
    # test(args.use_whole_text, args.test_data_source, args.train_data_source)
    # test2(args.use_whole_text, args.test_data_source, args.train_data_source)
