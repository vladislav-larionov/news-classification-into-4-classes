import string
from os import mkdir, rmdir

import gc
import numpy as np
import pandas as pd
import re
import csv

from nltk.corpus import stopwords
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
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


def split_and_save(articles):
    for source in ['lemmed_title_text_w', 'mention_lemmed_title_text_no_stopwords_w', 'lemmed_title_text_no_stopwords_w']:
        with open(f'data/{source}.tsv', 'w', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['phrase', 'label'])
            for art in articles:
                writer.writerow([' '.join(art[source]), art['user_categories']])
        data = pd.read_csv(f'data/{source}.tsv', sep='\t')
        y = np.asarray(data['label'])
        label_map = {cat: index for index, cat in enumerate(np.unique(y))}
        # print(label_map)
        y_prep = np.asarray([label_map[l] for l in y])
        x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=0.2, random_state=42,
                                                            stratify=y_prep)
        mkdir(f'data/{source}')
        with open(f'data/{source}/train.tsv', 'w', encoding='utf-8') as file:
            train = pd.DataFrame(data={"phrase": x_train['phrase'], "label": x_train['label']})
            train.to_csv(file, index=False, sep='\t')
        with open(f'data/{source}/test.tsv', 'w', encoding='utf-8') as file:
            test = pd.DataFrame(data={"phrase": x_test['phrase'], "label": x_test['label']})
            test.to_csv(file, index=False, sep='\t')
    print()


def save_full(articles):
    # mkdir(f'data/full')
    # with open(f'data/full/full.tsv', 'w', encoding='utf-8') as file:
    #     writer = csv.writer(file, delimiter='\t')
    #     writer.writerow(['phrase', 'label'])
    #     for art in articles:
    #         writer.writerow([art['full_text'].replace("\n", " ").replace("\t", " "), art['user_categories']])
    # data = pd.read_csv(f'data/full/full.tsv', sep='\t')
    # y = np.asarray(data['label'])
    # label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    # y_prep = np.asarray([label_map[l] for l in y])
    # x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=0.2, random_state=42,
    #                                                     stratify=y_prep)
    # with open(f'data/full/train.tsv', 'w', encoding='utf-8') as file:
    #     train = pd.DataFrame(data={"phrase": x_train['phrase'], "label": x_train['label']})
    #     train.to_csv(file, index=False, sep='\t')
    # with open(f'data/full/test.tsv', 'w', encoding='utf-8') as file:
    #     test = pd.DataFrame(data={"phrase": x_test['phrase'], "label": x_test['label']})
    #     test.to_csv(file, index=False, sep='\t')
    # print()
    #
    # mkdir(f'data/title_full')
    # with open(f'data/title_full/title_full.tsv', 'w', encoding='utf-8') as file:
    #     writer = csv.writer(file, delimiter='\t')
    #     writer.writerow(['phrase', 'label'])
    #     for art in articles:
    #         writer.writerow([art['title'] + '. ' + art['full_text'].replace("\n", " ").replace("\t", " "), art['user_categories']])
    # data = pd.read_csv(f'data/title_full/title_full.tsv', sep='\t')
    # y = np.asarray(data['label'])
    # label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    # y_prep = np.asarray([label_map[l] for l in y])
    # x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=0.2, random_state=42,
    #                                                     stratify=y_prep)
    #
    # with open(f'data/title_full/train.tsv', 'w', encoding='utf-8') as file:
    #     train = pd.DataFrame(data={"phrase": x_train['phrase'], "label": x_train['label']})
    #     train.to_csv(file, index=False, sep='\t')
    # with open(f'data/title_full/test.tsv', 'w', encoding='utf-8') as file:
    #     test = pd.DataFrame(data={"phrase": x_test['phrase'], "label": x_test['label']})
    #     test.to_csv(file, index=False, sep='\t')
    # print()


    mkdir(f'data/mentions')
    parts_with_title = []
    with open(f'data/mentions/mentions.tsv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['phrase', 'label'])
        for art in articles:
            pars = get_paragraph_with_mention(art['full_text'])
            pars = [par.replace("\n", " ").replace("\t", " ") for par in pars]
            parts_with_title.append([' '.join([art['title'] + '.'] + pars), art['user_categories']])
            writer.writerow([' '.join(pars), art['user_categories']])
    data = pd.read_csv(f'data/mentions/mentions.tsv', sep='\t')
    y = np.asarray(data['label'])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=0.2, random_state=42,
                                                        stratify=y_prep)
    with open(f'data/mentions/train.tsv', 'w', encoding='utf-8') as file:
        train = pd.DataFrame(data={"phrase": x_train['phrase'], "label": x_train['label']})
        train.to_csv(file, index=False, sep='\t')
    with open(f'data/mentions/test.tsv', 'w', encoding='utf-8') as file:
        test = pd.DataFrame(data={"phrase": x_test['phrase'], "label": x_test['label']})
        test.to_csv(file, index=False, sep='\t')
    print()

    mkdir(f'data/title_mentions')
    with open(f'data/title_mentions/title_mentions.tsv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['phrase', 'label'])
        writer.writerows(parts_with_title)
    data = pd.read_csv(f'data/title_mentions/title_mentions.tsv', sep='\t')
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=0.2, random_state=42,
                                                        stratify=y_prep)
    with open(f'data/title_mentions/train.tsv', 'w', encoding='utf-8') as file:
        train = pd.DataFrame(data={"phrase": x_train['phrase'], "label": x_train['label']})
        train.to_csv(file, index=False, sep='\t')
    with open(f'data/title_mentions/test.tsv', 'w', encoding='utf-8') as file:
        test = pd.DataFrame(data={"phrase": x_test['phrase'], "label": x_test['label']})
        test.to_csv(file, index=False, sep='\t')
    print()


def split_text_by_head_and_tail(path, head_count):
    res_path = f'{path}/head_and_tail_{head_count}_{512-head_count}'
    mkdir(res_path)
    for source in ['test.tsv', 'train.tsv']:
        data = pd.read_csv(f'{path}/{source}', sep='\t', header=0)
        phrases = []
        labels = []
        for phrase, label in zip(data['phrase'], data['label']):
            splitted = phrase.split(' ')
            total = len(splitted)
            if total < 512:
                phrases.append(phrase)
                labels.append(label)
                continue
            begin = []
            begin_size = 0
            while begin_size < head_count and begin_size < total:
                begin.append(splitted[begin_size])
                begin_size += 1
            while begin_size < total and not splitted[begin_size].endswith(('!', '.', '?')):
                begin.append(splitted[begin_size])
                begin_size += 1
            if begin_size < total:
                begin.append(splitted[begin_size])
                begin_size += 1
            rest_count = 512 - begin_size
            if rest_count > 0:
                end = splitted[-rest_count:]
            else:
                end = []
            end_shift = 0
            end_size = len(end)
            while end_shift < end_size and not end[end_shift].endswith(('!', '.', '?')):
                end_shift += 1
            end_shift += 1
            end = end[end_shift:]
            phrases.append(' '.join(begin + end))
            labels.append(label)
        with open(f'{res_path}/{source}', 'w', encoding='utf-8') as file:
            res = pd.DataFrame(data={"phrase": phrases, "label": labels})
            res.to_csv(file, index=False, sep='\t')


if __name__ == "__main__":
    # articles = get_articles()
    # tokenize(articles)
    # articles_to_word_lists(articles)
    # write_articles(articles, 'articles.json')

    # articles = read_texts('articles_w_m.json')
    # merge_title_and_text(articles)
    # write_articles(articles, 'articles_w_m_t.json')

    # articles = read_texts('articles_w_m_t.json')
    # save_full(articles)
    # print()
    for source in ['data/title_mentions', 'data/full', 'data/mentions', 'data/title_full']:
        split_text_by_head_and_tail(source, 128)
        split_text_by_head_and_tail(source, 256)