import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import form_y_prep, create_res_dir, form_label_map, parse_arguments, print_info, form_res_path

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class GPTWholeTextWordEmbedding(TransformerMixin, BaseEstimator):
    # https://stackoverflow.com/questions/60574112/can-we-use-gpt-2-sentence-embedding-for-classification-tasks
    # https://github.com/huggingface/transformers/issues/1458
    # https://github.com/ai-forever/ru-gpts#Pretraining-ruGPT3Small
    def __init__(self, model_name="sberbank-ai/rugpt3small_based_on_gpt2", batch_size=1, layer=-1):
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = []
        for text in X:
            res_txt = []
            for word in text:
                with torch.no_grad():
                    text_index = self.tokenizer.encode(word, add_prefix_space=True, return_tensors='pt')
                    vector = self.model.transformer.wte.weight[text_index, :]
                    res_txt.append(vector.numpy()[0])
            res.append(np.mean(np.concatenate(np.array(res_txt)), axis=0))
        return np.array(res)


class GPTSentenceEmbedding(GPTWholeTextWordEmbedding):
    def transform(self, X):
        res = []
        for text in X:
            res_txt = []
            for sent in text:
                with torch.no_grad():
                    text_index = self.tokenizer.encode(' '.join(sent), truncation=True,
                                                       add_prefix_space=True, max_length=512, return_tensors='pt')
                    vector = self.model.transformer.wte.weight[text_index, :]
                    res_txt.append(vector.numpy()[0])
            res.append(np.mean(np.concatenate(np.array(res_txt)), axis=0))
        return np.array(res)


class GPTWholeTextEmbedding(GPTWholeTextWordEmbedding):
    def transform(self, X):
        res = []
        for text in X:
            with torch.no_grad():
                text_index = self.tokenizer.encode(' '.join(text),
                                                   add_prefix_space=True, return_tensors='pt')
                vector = self.model.transformer.wte.weight[text_index, :]
            res.append(np.mean(vector.numpy()[0], axis=0))
        return np.array(res)


def test(use_whole_text: bool, test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    if use_whole_text:
        x_train = x_train[train_data_source]
        x_test = x_test[test_data_source]
        embeder = GPTWholeTextWordEmbedding('sberbank-ai/rugpt3large_based_on_gpt2')
    else:
        x_train = x_train[train_data_source.rstrip('_w')]
        x_test = x_test[test_data_source.rstrip('_w')]
        embeder = GPTWholeTextWordEmbedding('sberbank-ai/rugpt3large_based_on_gpt2')
    classifier = make_pipeline(embeder,
                               # PCA(n_components=30),
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


def classify_with_gpt(**params):
    train_data_source = params['train_data_source']
    test_data_source = params.get('test_data_source', train_data_source)
    use_whole_text = params.get('use_whole_text', False)
    use_short_classifiers_list = params.get('short', False)
    use_std_sclr = params.get('use_std_sclr', False)
    res_dir = params.get('res_dir')
    save_err_matr = params.get('save_err_matr', True)
    model = params.get('model', '')
    test_size = 0.2
    print_info(**params, test_size=test_size)

    data = pd.read_json('articles_w_m_t.json')
    y_prep = form_y_prep(data["user_categories"])
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    if use_whole_text:
        model = GPTWholeTextWordEmbedding(model_name=model)
        x_train = x_train[train_data_source]
        x_test = x_test[test_data_source]
        file_postfix = 'whole_text'
    else:
        x_train = x_train[train_data_source.rstrip('_w')]
        x_test = x_test[test_data_source.rstrip('_w')]
        model = GPTSentenceEmbedding(model_name=model)
        file_postfix = 'sentences'
    vectorizors = [model]
    if use_std_sclr:
        vectorizors.append(StandardScaler())
    if use_short_classifiers_list:
        classifiers = short_classifier_list(vectorizors)
    else:
        classifiers = full_classifier_list(vectorizors)
    res_dir_path = form_res_path(res_dir, train_data_source, test_data_source)
    res_dir = create_res_dir(f'{res_dir_path}/gpt_{file_postfix}')
    print_info(**params, file=f'{res_dir_path}/gpt_{file_postfix}/info.txt', test_size=test_size)

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


if __name__ == '__main__':
    # args = initialize_argument_parser().parse_args()
    # classify_with_gpt(args.use_whole_text, args.test_data_source, args.train_data_source,
    #      args.use_std_sclr, args.short)
    args = parse_arguments()
    classify_with_gpt(vars(args))
    # test(args.use_whole_text, args.test_data_source, args.train_data_source)
