import collections

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
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
from gensim.models import Word2Vec, FastText
import pandas as pd
import csv
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

from classifier_lists import full_classifier_list, short_classifier_list
from clsassifier_iterator import classify_all
from io_utils import initialize_argument_parser
from mean_embedding_vectorizer import MeanEmbeddingVectorizer
from utils import form_y_prep, form_label_map, create_res_dir, parse_arguments, form_res_path, print_info


def create_model(x_train: list, use_whole_text: bool, train_data_source: str):
    vector_size = 70
    window = 8
    sg = 1
    epochs = 20
    print(f'vector_size = {vector_size}')
    print(f'window = {window}')
    print(f'use_whole_text = {use_whole_text}')
    print(f'model = {train_data_source}')
    print(f'epochs = {epochs}')
    print(f'sg = {sg}')
    model = FastText(vector_size=vector_size, window=window, sg=sg, epochs=epochs)
    model.build_vocab(x_train[train_data_source])
    if use_whole_text:
        model.train(corpus_iterable=x_train[train_data_source], total_examples=len(x_train[train_data_source]),
                    epochs=model.epochs)
    else:
        sents = []
        for art in x_train[train_data_source.rstrip('_w')].values:
            if art:
                sents.extend(art)
        model.train(corpus_iterable=sents, total_examples=len(sents), epochs=model.epochs)
    return model


def classify_with_ft(**params):
    # http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    # data = pd.read_json('articles.json')
    train_data_source = params['train_data_source']
    test_data_source = params.get('test_data_source', train_data_source)
    use_whole_text = params.get('use_whole_text', False)
    use_short_classifiers_list = params.get('short', False)
    use_std_sclr = params.get('use_std_sclr', False)
    res_dir = params.get('res_dir')
    use_pca = params.get('use_pca', False)
    save_err_matr = params.get('save_err_matr', True)
    test_size = 0.2
    n_components = 50
    print_info(**params, test_size=test_size, n_components=n_components)

    data = pd.read_json('articles_w_m_t.json')
    y_prep = form_y_prep(data["user_categories"])
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    model = create_model(x_train, use_whole_text, train_data_source)
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
    res_dir = create_res_dir(f'{res_dir_path}/ft_{file_postfix}')
    print_info(**params, file=f'{res_dir_path}/ft_{file_postfix}/info.txt', test_size=test_size, n_components=n_components)

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
    model = create_model(x_train, use_whole_text, train_data_source)
    clsassifier = make_pipeline(MeanEmbeddingVectorizer(model),
                                PCA(n_components=45),
                                SVC(kernel='poly', degree=4, coef0=0.75))
    clsassifier.fit(x_train[train_data_source], y_train)
    y_res = clsassifier.predict(x_test[test_data_source])
    print(f'SVC kernel=poly degree=4 coef0=0.75;'
          f' P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
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
    model = create_model(x_train, use_whole_text, train_data_source)
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

    # grid_param = {
    #     # 'classify__kernel': ['poly', 'rbf'],
    #     'classify__degree': [5, 6, 4],
    #     # 'classify__coef0': [0.75, 0.7, 0.8, 0.2],
    #     # 'classify__C': [0.1, 1, 10],
    #     # 'classify__gamma': [0.1, 1, 10],
    #     # 'classify__shrinking': [True, False]
    # }
    # clsassifier = Pipeline([
    #     ('vectorizer', MeanEmbeddingVectorizer(model)),
    #     # ('vectorizer', TfidfVectorizer().fit_transform(x_train[train_data_source])),
    #     ('classify', SVC(kernel='poly'))
    # ])

    # grid_param = {
    #     'classify__weights': ['distance', 'uniform'],
    #     'classify__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #     'classify__leaf_size': [30, 15],
    # }
    # clsassifier = Pipeline([
    #     ('vectorizer', MeanEmbeddingVectorizer(model)),
    #     ('classify', KNeighborsClassifier(n_jobs=-1))
    #     ])

    grid_param = {
        # 'classify__kernel': ['poly', 'rbf'],
        'classify__n_estimators': [500, 70, 100],
        # 'classify__coef0': [0.75, 0.7, 0.8, 0.2],
        # 'classify__C': [0.1, 1, 10],
        # 'classify__gamma': [0.1, 1, 10],
        # 'classify__shrinking': [True, False]
    }
    clsassifier = Pipeline([
        ('vectorizer', MeanEmbeddingVectorizer(model)),
        ('classify', AdaBoostClassifier(
            base_estimator=RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=False,
                                                  max_features='log2')))
    ])
    gd_sr = GridSearchCV(estimator=clsassifier,
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=1)
    gd_sr.fit(x_train[train_data_source], y_train)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_
    print(best_result)
    data = pd.DataFrame(gd_sr.cv_results_)
    print(data[['mean_test_score', 'std_test_score', 'params']])
    print()


if __name__ == '__main__':
    # args = initialize_argument_parser().parse_args()
    # classify_with_ft(args.use_whole_text, args.test_data_source, args.train_data_source,
    #      args.use_std_sclr)
    args = parse_arguments()
    classify_with_ft(vars(args))
    # grid_search(args.use_whole_text, args.test_data_source, args.train_data_source)
    # test(args.use_whole_text, args.test_data_source, args.train_data_source)
