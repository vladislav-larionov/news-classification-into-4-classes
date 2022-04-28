import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bert_transformer_embedding import BertTransformerEmbedding, BertTransformerSentenceEmbedding
from classifier_lists import full_classifier_list, short_classifier_list
from io_utils import initialize_argument_parser
from clsassifier_iterator import classify_all
from utils import form_y_prep, form_label_map, create_res_dir, print_info, form_res_path


def test(use_whole_text: bool, test_data_source: str, train_data_source: str):
    # tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    # model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    # print(embed_bert_cls('привет мир', model, tokenizer).shape)
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    # 'cointegrated/LaBSE-en-ru'
    if use_whole_text:
        x_train = x_train[train_data_source]
        x_test = x_test[test_data_source]
        embeder = BertTransformerEmbedding()
    else:
        x_train = x_train[train_data_source.rstrip('_w')]
        x_test = x_test[test_data_source.rstrip('_w')]
        embeder = BertTransformerSentenceEmbedding()
    classifier = make_pipeline(embeder,
                               # PCA(n_components=30),
                               ExtraTreesClassifier(class_weight='balanced')
                               # LogisticRegression(penalty='none')
                               )
    classifier.fit(x_train, y_train)
    y_res = classifier.predict(x_test)
    print(f'{"SVC(kernel=poly, degree=5, coef0=7);":{" "}{"<"}{57}} '
          f'P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
          f' P_macro: {precision_score(y_test, y_res, average="macro"):1.4f};'
          f' R_micro: {recall_score(y_test, y_res, average="micro"):1.4f};'
          f' R_macro: {recall_score(y_test, y_res, average="macro"):1.4f};'
          f' F1_micro: {f1_score(y_test, y_res, average="micro"):1.4f};'
          f' F1_macro: {f1_score(y_test, y_res, average="macro"):1.4f};'
          )


def classify_with_bert(**params):
    train_data_source = params['train_data_source']
    test_data_source = params.get('test_data_source', train_data_source)
    use_whole_text = params.get('use_whole_text', False)
    use_short_classifiers_list = params.get('short', False)
    use_std_sclr = params.get('use_std_sclr', False)
    model = params.get('model', '')
    res_dir = params.get('res_dir')
    save_err_matr = params.get('save_err_matr', True)
    test_size = 0.2
    print_info(**params, file=None, test_size=test_size)

    data = pd.read_json('articles_w_m_t.json')
    y_prep = form_y_prep(data["user_categories"])
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42, stratify=y_prep)
    if use_whole_text:
        model = BertTransformerEmbedding(model_name=model)
        x_train = x_train[train_data_source]
        x_test = x_test[test_data_source]
        file_postfix = 'whole_text'
    else:
        x_train = x_train[train_data_source.rstrip('_w')]
        x_test = x_test[test_data_source.rstrip('_w')]
        model = BertTransformerSentenceEmbedding(model_name=model)
        file_postfix = 'sentences'

    res_dir_path = form_res_path(res_dir, train_data_source, test_data_source)
    res_dir = create_res_dir(f'{res_dir_path}/bert_{file_postfix}')
    print_info(**params, file=f'{res_dir_path}/bert_{file_postfix}/info.txt', test_size=test_size)
    vectorizors = [model]
    if use_std_sclr:
        vectorizors.append(StandardScaler())
    if use_short_classifiers_list:
        classifiers = short_classifier_list(vectorizors)
    else:
        classifiers = full_classifier_list(vectorizors)
    classify_all(x_train,
                 x_test,
                 y_train,
                 y_test,
                 classifiers=classifiers,
                 res_dir=res_dir,
                 target_names=list(form_label_map(data["user_categories"]).keys()),
                 save_err_matr=save_err_matr,
                 print_table=False)


if __name__ == '__main__':
    args = initialize_argument_parser().parse_args()
    # classify_with_bert(args.use_whole_text, args.test_data_source, args.train_data_source,
    #      args.use_std_sclr, args.short)
    classify_with_bert(vars(args))
    # test(args.use_whole_text, args.test_data_source, args.train_data_source)
