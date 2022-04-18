import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
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



def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


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
    # 'DeepPavlov/rubert-base-cased-sentence'
    # 'sberbank-ai/sbert_large_nlu_ru'
    # use_whole_text = False
    if use_whole_text:
        x_train = x_train[train_data_source]
        x_test = x_test[test_data_source]
        embeder = BertTransformerEmbedding()
    else:
        x_train = x_train[train_data_source.rstrip('_w')]
        x_test = x_test[test_data_source.rstrip('_w')]
        embeder = BertTransformerSentenceEmbedding()
    classifier = make_pipeline(embeder,
                               SVC(kernel='poly', degree=5, coef0=0.7)
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


def vectorizors(w2v_model, use_standard_scaler: bool = False):
    vectrs = [w2v_model]
    if use_standard_scaler:
        vectrs.append(StandardScaler())
    return vectrs


def main(use_whole_text: bool, test_data_source: str, train_data_source: str, use_cross_validation: bool,
         use_std_sclr: bool, use_short_classifiers_list: bool):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    print(label_map)
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    if use_whole_text:
        model = BertTransformerEmbedding()
        x_train = x_train[train_data_source]
        x_test = x_test[test_data_source]
        file_postfix = 'bert_whole_text'
    else:
        x_train = x_train[train_data_source.rstrip('_w')]
        x_test = x_test[test_data_source.rstrip('_w')]
        model = BertTransformerSentenceEmbedding()
        file_postfix = 'bert_sentens'
    print(f'use_short_classifiers_list = {use_short_classifiers_list}')
    print(f'use_whole_text = {use_whole_text}')
    print(f'model_test_str = {test_data_source}')
    print(f'test_size = {test_size}')
    print(f'use_std_sclr = {use_std_sclr}')
    if use_short_classifiers_list:
        classifiers = short_classifier_list(vectorizors, model, use_std_sclr)
    else:
        classifiers = full_classifier_list(vectorizors, model, use_std_sclr)
    classify_all(x_train,
                 x_test,
                 y_train,
                 y_test,
                 file_postfix=file_postfix,
                 use_cross_validation=use_cross_validation,
                 target_names=list(label_map.keys()),
                 paint_err_matr=False,
                 print_table=True,
                 classifiers=classifiers)


if __name__ == '__main__':
    args = initialize_argument_parser().parse_args()
    main(args.use_whole_text, args.test_data_source, args.train_data_source, args.use_cross_validation,
         args.use_std_sclr, args.short)
    # test(args.use_whole_text, args.test_data_source, args.train_data_source)
