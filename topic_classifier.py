import collections

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import pandas as pd
import csv
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

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


def classify_all(x_train, x_test, y_train, y_test, classifiers, target_names=None, paint_err_matr=False,
                 print_table=False):
    csvfile_all = open('res_all.csv', 'w')
    csv_writer_all = csv.writer(csvfile_all)
    with open('res_avg.csv', 'w', newline='') as csvfile_avg:
        # headers_avg = ['name', 'Score', 'P_micro', 'P_macro', 'R_micro', 'R_macro', 'F1_micro', 'F1_macro']
        headers_avg = ['name', 'P_micro', 'P_macro', 'R_micro', 'R_macro', 'F1_micro', 'F1_macro']
        csv_writer_avg = csv.writer(csvfile_avg)
        csv_writer_avg.writerow(headers_avg)
        with open('res_classes.csv', 'w', newline='') as csvfile_classes:
            headers_classes = ['name', 'P_не_релевантные', 'R_не_релевантные', 'F1_не_релевантные',
                               'P_общество', 'R_общество', 'F1_общество',
                               'P_образование', 'R_образование', 'F1_образование',
                               'P_наука_и_технологии', 'R_наука_и_технологии', 'F1_наука_и_технологии',
                               ]
            csv_writer_classes = csv.writer(csvfile_classes)
            csv_writer_classes.writerow(headers_classes)
            csv_writer_all.writerow(headers_avg + headers_classes[1:])
            for classifier_info in classifiers:
                classifier_info[0].fit(x_train, y_train)
                y_res = classifier_info[0].predict(x_test)
                # print_statistics(y_test, x_res, classifier_info[1])
                # print(f"Score:\t\t{classifier_info[0].score(x_train, y_train):1.4f}")
                # print(f'Report {classifier_info[1]}:')
                if target_names:
                    dict_res = classification_report(y_test, y_res, target_names=target_names, digits=4,
                                                     output_dict=True)
                    if print_table:
                        print(f'{classifier_info[1] + ";":{" "}{"<"}{57}} '
                              # f"Train_acc: {classifier_info[0].score(x_train, y_train):1.4f}"
                              f'P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
                              f' P_macro: {precision_score(y_test, y_res, average="macro"):1.4f};'
                              f' R_micro: {recall_score(y_test, y_res, average="micro"):1.4f};'
                              f' R_macro: {recall_score(y_test, y_res, average="macro"):1.4f};'
                              f' F1_micro: {f1_score(y_test, y_res, average="micro"):1.4f};'
                              f' F1_macro: {f1_score(y_test, y_res, average="macro"):1.4f};'
                              )
                        avg = [classifier_info[1],
                             # f"{classifier_info[0].score(x_train, y_train):1.4f}",
                             f'{precision_score(y_test, y_res, average="micro"):1.4f}',
                             f'{precision_score(y_test, y_res, average="macro"):1.4f}',
                             f'{recall_score(y_test, y_res, average="micro"):1.4f}',
                             f'{recall_score(y_test, y_res, average="macro"):1.4f}',
                             f'{f1_score(y_test, y_res, average="micro"):1.4f}',
                             f'{f1_score(y_test, y_res, average="macro"):1.4f}']
                        csv_writer_avg.writerow(avg)
                        classes = [classifier_info[1],
                                 f"{dict_res['not_relevant']['precision']:1.4f}",
                                 f"{dict_res['not_relevant']['recall']:1.4f}",
                                 f"{dict_res['not_relevant']['f1-score']:1.4f}",
                                 f"{dict_res['Общество']['precision']:1.4f}",
                                 f"{dict_res['Общество']['recall']:1.4f}",
                                 f"{dict_res['Общество']['f1-score']:1.4f}",
                                 f"{dict_res['Образование']['precision']:1.4f}",
                                 f"{dict_res['Образование']['recall']:1.4f}",
                                 f"{dict_res['Образование']['f1-score']:1.4f}",
                                 f"{dict_res['Наука и технологии']['precision']:1.4f}",
                                 f"{dict_res['Наука и технологии']['recall']:1.4f}",
                                 f"{dict_res['Наука и технологии']['f1-score']:1.4f}",
                                 ]
                        csv_writer_classes.writerow(classes)
                        csv_writer_all.writerow(avg + classes[1:])
                    else:
                        print(classification_report(y_test, y_res, target_names=target_names, digits=4))
                if paint_err_matr:
                    cm = metrics.confusion_matrix(y_test, y_res)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
                    ax.set(xlabel="Pred", ylabel="True", xticklabels=target_names,
                           yticklabels=target_names, title=classifier_info[1], )
                    plt.yticks(rotation=0)
                    plt.show()


def create_w2v_model(x_train: list, use_whole_text: bool, learning_data_source: str):
    vector_size = 70
    window = 8
    sg = 1
    epochs = 15
    print(f'vector_size = {vector_size}')
    print(f'window = {window}')
    print(f'use_whole_text = {use_whole_text}')
    print(f'model = {learning_data_source}')
    print(f'epochs = {epochs}')
    print(f'sg = {sg}')
    if use_whole_text:
        model = Word2Vec(x_train[learning_data_source], vector_size=vector_size, window=window, sg=sg, epochs=epochs,
                         workers=4)
    else:
        model_sent_str = learning_data_source.rstrip('_w')
        sents = []
        for art in x_train[model_sent_str].values:
            if art:
                sents.extend(art)
        model = Word2Vec(sents, vector_size=vector_size, window=window, sg=sg, epochs=epochs, workers=4)
    return model


def main(use_whole_text: bool, test_data_source: str, learning_data_source: str):
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
    model_str = learning_data_source
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42, stratify=y_prep)
    model = create_w2v_model(x_train, use_whole_text, learning_data_source)
    print(f'model_test_str = {test_data_source}')
    print(f'test_size = {test_size}')
    print(f'coef0 = {coef0}')
    classify_all(x_train[model_str],
                 x_test[test_data_source],
                 y_train,
                 y_test,
                 target_names=list(label_map.keys()),
                 paint_err_matr=False,
                 print_table=True,
                 classifiers=[
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC()), 'SVM rbf (default)'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='linear')), 'SVM linear'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='poly')), 'SVM kernel=poly'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='poly', coef0=coef0)), f'SVM kernel=poly coef0={coef0}'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='poly', degree=5)), 'SVM kernel=poly degree=5'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='poly', degree=5, coef0=coef0)), f'SVM kernel=poly degree=5 coef0={coef0}'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='poly', degree=6)), 'SVM kernel=poly degree=6'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), SVC(kernel='poly', degree=6, coef0=coef0)), f'SVM kernel=poly degree=6 coef0={coef0}'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), DecisionTreeClassifier()), 'DecisionTree'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), LogisticRegression(max_iter=300)), 'LogisticRegression'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), LogisticRegression(penalty="none", max_iter=300)), 'LogisticRegression penalty="none"'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), KNeighborsClassifier()), 'KNeighbors'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), KNeighborsClassifier(weights='distance')), 'KNeighbors weights=distance'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), AdaBoostClassifier(n_estimators=70)), f'AdaBoost n_estimators={70}'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), MinMaxScaler(), MultinomialNB()), 'MultinomialNB'),
                     (make_pipeline(MeanEmbeddingVectorizer(model), GaussianNB()), 'GaussianNB'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier()), 'RandomForest'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(max_features=None)), 'RandomForest max_features=None'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(criterion='entropy')), 'RandomForest entropy'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(criterion='entropy', max_features=None)),
                      'RandomForest entropy max_features=None'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(criterion='entropy', max_features='log2')),
                      'RandomForest entropy max_features=log2'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(criterion='entropy', bootstrap=False)),
                      'RandomForest entropy bootstrap=False'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(criterion='entropy', max_features=None, bootstrap=False)),
                      'RandomForest entropy max_features=None bootstrap=False'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(criterion='entropy', max_features='log2', bootstrap=False)),
                      'RandomForest entropy max_features=log2 bootstrap=False'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(n_estimators=150, criterion='entropy', bootstrap=False)),
                      'RandomForest n_estimators=150 entropy bootstrap=False'),
                     (make_pipeline(MeanEmbeddingVectorizer(model),
                                    RandomForestClassifier(n_estimators=200, criterion='entropy', bootstrap=False)),
                      'RandomForest n_estimators=200 entropy bootstrap=False')
                 ])


if __name__ == '__main__':
    args = initialize_argument_parser().parse_args()
    main(args.use_whole_text, args.test_data_source, args.learning_data_source)
