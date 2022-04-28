from more_itertools import chunked
from sklearn.base import TransformerMixin, BaseEstimator
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, BertTokenizerFast


class BertTransformerSentenceEmbedding(TransformerMixin, BaseEstimator):
    # https://stackoverflow.com/questions/67105996/how-to-use-bert-and-elmo-embedding-with-sklearn
    # https://habr.com/ru/post/562064/
    # 'cointegrated/LaBSE-en-ru'
    # https://huggingface.co/cointegrated/LaBSE-en-ru
    # 'DeepPavlov/rubert-base-cased-sentence'
    # https://huggingface.co/DeepPavlov/rubert-base-cased-sentence
    # 'sberbank-ai/sbert_large_nlu_ru'
    # https://huggingface.co/sberbank-ai/sbert_large_nlu_ru
    def __init__(self, model_name='cointegrated/rubert-tiny', batch_size=1, layer=-1):
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = []
        for text in X:
            res_txt = []
            for sent in text:
                t = self.tokenizer(' '.join(sent), padding=True, truncation=True, max_length=512,
                                   add_special_tokens=True, return_tensors='pt')
                with torch.no_grad():
                    model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
                embeddings = model_output.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings)
                res_txt.append(embeddings)
            res.append(np.mean(np.concatenate(res_txt), axis=0))
        return np.array(res)


class BertTransformerEmbedding(BertTransformerSentenceEmbedding):
    def transform(self, X):
        res = []
        for text in X:
            t = self.tokenizer(' '.join(text), padding=True, truncation=True, max_length=510, add_special_tokens=True,
                               return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
            embeddings = model_output.last_hidden_state[:, 0, :]
            # embeddings = torch.nn.functional.normalize(embeddings)
            res.append(np.mean(np.array(embeddings), axis=0))
        return np.array(res)
        #     res.append(embeddings)
        # return np.concatenate(res)

