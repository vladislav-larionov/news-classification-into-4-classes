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
from classifier_lists import full_classifier_list, short_classifier_list, bert_short_classifier_list
from io_utils import initialize_argument_parser
from clsassifier_iterator import classify_all
from utils import form_y_prep, form_label_map, create_res_dir, print_info, form_res_path, parse_arguments


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
    classifiers = bert_short_classifier_list(vectorizors)
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
    args = parse_arguments()
    # --test_data_source lemmed_title_text_w --train_data_source lemmed_title_text_w --short
    # classify_with_bert(args.use_whole_text, args.test_data_source, args.train_data_source,
    #      args.use_std_sclr, args.short)
    classify_with_bert(**vars(args))
    # test(args.use_whole_text, args.test_data_source, args.train_data_source)
