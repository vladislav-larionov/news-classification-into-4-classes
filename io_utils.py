import argparse

from bson.json_util import dumps, loads


def read_texts(filename='articles.json'):
    with open(filename, 'r+', encoding='utf-8') as f:
        return loads(f.read())


def write_articles(articles: list, filename):
    with open(filename, 'w+', encoding='utf-8') as file:
        file.write(dumps(articles))


def initialize_argument_parser() -> argparse.ArgumentParser:
    """
    Initializes parser of cmd arguments.

    :return: configured argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_whole_text', '-w',
        nargs='?',
        const=True,
        default=False,
        help='Если указан, то тексты при обработке будут объединяться в одно предложение.'
    )
    parser.add_argument(
        '--test_data_source', '-t',
        # nargs='1',
        help='Источник тестовых данных'
    )
    parser.add_argument(
        '--learning_data_source', '-l',
        # nargs='1',
        help='Источник обучающих данных'
    )
    return parser