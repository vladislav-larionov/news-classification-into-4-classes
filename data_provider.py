import gc
import re

from nltk.corpus import stopwords
from pymongo import MongoClient
from stanza import Pipeline

from io_utils import write_articles, read_texts

YAR_STATE_UNI_REG_QUERIES = [
    re.compile(r"(^|\s|\'|\"|«|\()(ЯрГУ|ЯГУ)(?![а-я])((\s+(им\.?|имени))?\s*(П(\.|авла)\s*Г(\.|ригорьевича)\s*)?"
               r"демидов[а-я]*)?",
               re.IGNORECASE | re.MULTILINE | re.UNICODE),
    re.compile(r"(^|\s|\'|\"|«|\()"
               r"(ярославск[а-я]*\s+)?(государствен[а-я]*\s+)?демидовск[а-я]*\s+университет[а-я]*",
               re.IGNORECASE | re.MULTILINE | re.UNICODE),
    re.compile(r"(^|\s|\'|\"|«|\()ярославск[а-я]*\s+госуниверситет[а-я]*"
               r"((\s+(им\.?|имени))?\s*(П(\.|авла)\s*Г(\.|ригорьевича)\s*)?демидов[а-я]*)?",
               re.IGNORECASE | re.MULTILINE | re.UNICODE),
    re.compile(r"((^|\s|\'|\"|«|\()ярославск[а-я]*\s+государствен[а-я]*\s+университет[а-я]*"
               r"((\s+(им\.?|имени))?\s*(П(\.|авла)\s*Г(\.|ригорьевича)\s*)?демидов[а-я]*)?)"
               r"|(^|\s|\'|\")((ярославск[а-я]*\s+)?(государствен[а-я]*\s+)?университет[а-я]*"
               r"(\s+(им\.?|имени))?\s*(П(\.|авла)\s*Г(\.|ригорьевича)\s*)?демидов[а-я]*)",
               re.IGNORECASE | re.MULTILINE | re.UNICODE)
]


def build_join_custom_article_metadata_collection_params() -> list:
    """ Builds mongo query params to join the custom_article_metadata collection to a result. """
    return [
        {
            '$lookup': {
                'from': 'custom_article_metadata',
                'localField': '_id',
                'foreignField': '_id',
                'as': 'user_data'
            }
        },
        {'$addFields': {'user_categories': {'$arrayElemAt': ['$user_data.categories', 0]}}},
        {'$addFields': {'relevant': {'$arrayElemAt': ['$user_data.relevant', 0]}}}
    ]


def get_articles():
    client = MongoClient()
    query = []
    query.extend(build_join_custom_article_metadata_collection_params())
    query.append({'$match': {'$or': [
        {'$and': [
            {'user_categories': {'$size': 1}},
            {'user_categories': {'$ne': 'Бизнес'}},
        ]
        },
        {'relevant': False}
    ]}})
    arts = list(client.media_monitoring.articles.aggregate(query))
    print(len(arts))
    for art in arts:
        if not art['relevant']:
            art['user_categories'] = 'not_relevant'
        elif art['user_categories'][0] == 'Наука' or art['user_categories'][0] == 'Технологии':
            art['user_categories'] = 'Наука и технологии'
        else:
            art['user_categories'] = art['user_categories'][0]
    return arts


def tokenize(articles: list):
    pipeline = Pipeline(processors='tokenize,pos,lemma,depparse',
                        lang="ru", use_gpu=False, logging_level='ERROR')
    for art in articles:
        text = pipeline(art['full_text'])
        art['sentences'] = [[word.text.lower() for word in sent.words if word.upos != 'PUNCT']
                            for sent in text.sentences]
        art['lemmed_sentences'] = [[word.lemma.lower() for word in sent.words if word.upos != 'PUNCT']
                                   for sent in text.sentences]
        art['lemmed_sentences_no_stopwords'] = [[word.lemma.lower() for word in sent.words
                                                 if word.upos != 'PUNCT' and
                                                 word.lemma.lower() not in stopwords.words("russian")]
                                                for sent in text.sentences]
        title = pipeline(art['title'])
        art['tokenized_title'] = [[word.text.lower() for word in sent.words if word.upos != 'PUNCT']
                                  for sent in title.sentences][0]
        art['lemmed_tokenized_title'] = [[word.lemma.lower() for word in sent.words if word.upos != 'PUNCT']
                                         for sent in title.sentences][0]
        art['lemmed_tokenized_title_no_stopwords'] = [[word.lemma.lower() for word in sent.words
                                                       if word.upos != 'PUNCT' and
                                                       word.lemma.lower() not in stopwords.words("russian")]
                                                      for sent in title.sentences][0]
        del text
        del title
        gc.collect()


def text_to_word_list(text):
    words = []
    for sent in text:
        words.extend(sent)
    return words


def articles_to_word_lists(articles):
    for art in articles:
        art['sentences_w'] = text_to_word_list(art['sentences'])
        art['lemmed_sentences_w'] = text_to_word_list(art['lemmed_sentences'])
        art['lemmed_sentences_no_stopwords_w'] = text_to_word_list(art['lemmed_sentences_no_stopwords'])


def get_paragraph_with_mention(text):
    paragraphs = []
    for paragraph in text.split('\n'):
        for query in YAR_STATE_UNI_REG_QUERIES:
            if re.search(query, paragraph):
                paragraphs.append(paragraph)
                break
    return paragraphs


def tokenize_paragraphs_with_mention(articles):
    pipeline = Pipeline(processors='tokenize,pos,lemma,depparse',
                        lang="ru", use_gpu=False, logging_level='ERROR')
    for art in articles:
        pars = get_paragraph_with_mention(art['full_text'])
        text = pipeline('\n'.join(pars))
        art['mention_sentences'] = [[word.text.lower() for word in sent.words if word.upos != 'PUNCT']
                                    for sent in text.sentences]
        art['mention_lemmed_sentences'] = [[word.lemma.lower() for word in sent.words if word.upos != 'PUNCT']
                                           for sent in text.sentences]
        art['mention_lemmed_sentences_no_stopwords'] = [[word.lemma.lower() for word in sent.words
                                                         if word.upos != 'PUNCT' and
                                                         word.lemma.lower() not in stopwords.words("russian")]
                                                        for sent in text.sentences]
        art['mention_sentences_w'] = text_to_word_list(art['mention_sentences'])
        art['mention_lemmed_sentences_w'] = text_to_word_list(art['mention_lemmed_sentences'])
        art['mention_lemmed_sentences_no_stopwords_w'] = text_to_word_list(art['mention_lemmed_sentences_no_stopwords'])
        del text
        gc.collect()


def merge_title_and_text(articles):
    for art in articles:
        art['lemmed_title_text'] = [art['lemmed_tokenized_title']] + art['lemmed_sentences']
        art['lemmed_title_text_no_stopwords'] = [art['lemmed_tokenized_title_no_stopwords']] + \
                                                art['lemmed_sentences_no_stopwords']
        art['mention_lemmed_title_text'] = [art['lemmed_tokenized_title']] + art['mention_lemmed_sentences']
        art['mention_lemmed_title_text_no_stopwords'] = [art['lemmed_tokenized_title_no_stopwords']] + \
                                                        art['mention_lemmed_sentences_no_stopwords']

        art['lemmed_title_text_w'] = art['lemmed_tokenized_title'] + art['lemmed_sentences_w']
        art['lemmed_title_text_no_stopwords_w'] = art['lemmed_tokenized_title_no_stopwords'] + \
                                                  art['lemmed_sentences_no_stopwords_w']
        art['mention_lemmed_title_text_w'] = art['lemmed_tokenized_title'] + \
                                             art['mention_lemmed_sentences_w']
        art['mention_lemmed_title_text_no_stopwords_w'] = art['lemmed_tokenized_title_no_stopwords'] + \
                                                          art['mention_lemmed_sentences_no_stopwords_w']


if __name__ == "__main__":
    # articles = get_articles()
    # tokenize(articles)
    # articles_to_word_lists(articles)
    # write_articles(articles, 'articles.json')

    articles = read_texts('articles_w_m.json')
    merge_title_and_text(articles)
    write_articles(articles, 'articles_w_m_t.json')
