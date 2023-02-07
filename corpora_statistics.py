import re

import pandas as pd

from data_provider import get_paragraph_with_mention


def count_avg_text_len(texts_file_path):
    data = pd.read_csv(texts_file_path, sep='\t', header=0)
    texts = 0
    words = 0
    for text in data['phrase']:
        texts += 1
        words += len(re.split(r'\s', text))
    return words / texts


def count_avg_par_len(fulls):
    pars = 0
    words = 0
    for text in fulls:
        paragraphs = get_paragraph_with_mention(text)
        for paragraph in paragraphs:
            pars += 1
            words += len(re.split(r'\s', paragraph))
    return words / pars


if __name__ == "__main__":
    for source in ['data/full/full.tsv', 'data/mentions/mentions.tsv']:
        print(source)
        print(f'avg_text_len: {count_avg_text_len(source)}')
    fulls = pd.read_json('articles_w_m_t.json')['full_text']
    print(f'avg_par_len: {count_avg_par_len(fulls)}')
