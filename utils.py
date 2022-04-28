import sys

import argparse

import numpy as np
from pathlib import Path


def rename_not_relevant_category(user_categories):
    res = []
    for cat in np.asarray(user_categories):
        if cat == 'not_relevant':
            res.append('Нерелевантная')
        else:
            res.append(cat)
    return np.asarray(res)


def form_y_prep(user_categories):
    y = rename_not_relevant_category(user_categories)
    label_map = form_label_map(user_categories)
    return np.asarray([label_map[l] for l in y])


def form_label_map(user_categories):
    y = rename_not_relevant_category(user_categories)
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    if 'not_relevant' in label_map:
        label_map['Нерелевантная'] = label_map.pop('not_relevant')
    return label_map


def create_res_dir(path) -> Path:
    while Path.exists(Path(path)):
        path += '_1'
    Path(path).mkdir(parents=True)
    return Path(path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v", nargs='?', const=True, default=False)
    parser.add_argument("--ft", nargs='?', const=True, default=False)
    parser.add_argument("--tf", nargs='?', const=True, default=False)
    parser.add_argument("--gpt", nargs='?', const=True, default=False)
    parser.add_argument("--bert_tiny", nargs='?', const=True, default=False)
    parser.add_argument("--bert_sber", nargs='?', const=True, default=False)
    parser.add_argument("--use_whole_text", nargs='?', const=True, default=False)
    parser.add_argument("--use_pca", nargs='?', const=True, default=False)
    parser.add_argument("--short", nargs='?', const=True, default=False)
    parser.add_argument("--save_err_matr", nargs='?', const=True, default=False)
    parser.add_argument("--res_dir", default="result")
    return parser.parse_args()


def print_info(**params):
    train_data_source = params['train_data_source']
    test_data_source = params.get('test_data_source', train_data_source)
    use_whole_text = params.get('use_whole_text', False)
    use_short_classifiers_list = params.get('short', False)
    use_std_sclr = params.get('use_std_sclr', False)
    description = params.get('description', '')
    res_dir = params.get('res_dir')
    use_pca = params.get('use_pca', False)
    save_err_matr = params.get('save_err_matr', True)
    file = params.get('file')
    test_size = params.get('test_size')
    model_info = params.get('model_info')
    if file:
        stream = open(file, 'w')
    else:
        stream = sys.stdout
    print(f'description: {description}', file=stream)
    print(f'train: {train_data_source}', file=stream)
    print(f'test: {test_data_source}', file=stream)
    print(f'res_dir: {res_dir}', file=stream)
    print(f'use_short_classifiers_list: {use_short_classifiers_list}', file=stream)
    print(f'use_whole_text: {use_whole_text}', file=stream)
    print(f'test_size: {test_size}', file=stream)
    print(f'use_std_sclr: {use_std_sclr}', file=stream)
    print(f'save_err_matr: {save_err_matr}', file=stream)
    print(f'use_pca: {use_pca}', file=stream)
    if use_pca:
        n_components = params.get('n_components')
        print(f'n_components = {n_components}', file=stream)
    if model_info:
        print('model_info:', file=stream)
        for k, v in model_info.items():
            print(f'{k}: {v}', file=stream)
    if file:
        stream.close()


def form_res_path(res_dir, train_data_source, test_data_source):
    if train_data_source == test_data_source:
        return f'{res_dir}/{train_data_source}'
    return f'{res_dir}/{train_data_source}__{test_data_source}'