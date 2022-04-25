
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.base import BaseEstimator


# class MeanEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         # if a text is empty we should return a vector of zeros
#         # with the same dimensionality as all the other vectors
#         self.dim = word2vec.vector_size
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X):
#         return np.array([
#             np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
#                     or [np.zeros(self.dim)], axis=0)
#             for words in X
#         ])


class Word2VecModel(BaseEstimator):

    def __init__(self, vector_size=70, window=8, sg=1, workers=4, epochs=15):
        self.d2v_model = None
        self.vector_size = vector_size
        self.window = window
        self.sg = sg
        self.workers = workers
        self.epochs = epochs

    def fit(self, raw_documents, y=None):
        self.model = Word2Vec(raw_documents,
                              vector_size=self.vector_size,
                              window=self.window, sg=self.sg,
                              epochs=self.epochs, workers=self.workers)
        self.dim = self.model.vector_size
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.model.wv[w] for w in words if w in self.model.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)


class Doc2VecModel(BaseEstimator):
    # https://stackoverflow.com/questions/50278744/pipeline-and-gridsearch-for-doc2vec
    def __init__(self, vector_size=70, window=8, workers=4, epochs=15, dm=1, dbow_words=0, dm_mean=1):
        self.d2v_model = None
        self.vector_size = vector_size
        self.window = window
        self.workers = workers
        self.epochs = epochs
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_mean = dm_mean

    def fit(self, raw_documents, y=None):
        # Initialize model
        self.d2v_model = Doc2Vec(vector_size=self.vector_size, window=self.window, dm=self.dm,
                                 epochs=self.epochs, workers=4, dbow_words=self.dbow_words, dm_mean=self.dm_mean)
        # Tag docs
        tagged_documents = []
        for index, row in raw_documents.iteritems():
            tag = '{}_{}'.format("type", index)
            tokens = row
            tagged_documents.append(TaggedDocument(words=tokens, tags=[tag]))
        # Build vocabulary
        self.d2v_model.build_vocab(tagged_documents)
        # Train model
        self.d2v_model.train(tagged_documents, total_examples=len(tagged_documents), epochs=self.d2v_model.epochs)
        return self

    def transform(self, raw_documents):
        X = []
        for row in raw_documents:
            X.append(self.d2v_model.infer_vector(row))
        X = np.array(X)
        return X

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)