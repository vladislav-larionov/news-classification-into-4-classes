from concurrent.futures import ProcessPoolExecutor

from topic_classifier_bert import classify_with_bert
from topic_classifier_ft import classify_with_ft
from topic_classifier_gpt import classify_with_gpt
from topic_classifier_tfidf import classify_with_tf
from topic_classifier_w2v import classify_with_w2v
from utils import create_res_dir, parse_arguments

sources = [
    {"train_data_source": "lemmed_title_text_w", "description": "Статьи с заголовками"},
    {"train_data_source": "mention_lemmed_title_text_no_stopwords_w", "description": "Упоминания ЯрГУ с заголовками без стоп-слов"},
    {"train_data_source": "lemmed_title_text_no_stopwords_w", "description": "Статьи с заголовками без стоп-слов"},
    {"train_data_source": "lemmed_sentences_w", "description": "Полные тексты статей"},
    {"train_data_source": "mention_lemmed_title_text_w", "description": "Упоминания ЯрГУ с заголовками"},
    {"train_data_source": "mention_lemmed_sentences_w", "description": "Упоминания ЯрГУ"},
    {"train_data_source": "mention_lemmed_sentences_no_stopwords_w", "description": "Упоминания ЯрГУ без стоп-слов"},
    {"train_data_source": "lemmed_sentences_no_stopwords_w", "description": "Статьи без стоп-слов"},
]


def get_classifier_function():
    args = parse_arguments()
    params = vars(args)
    if args.w2v:
        params.update(dict(embedding='w2v'))
        return classify_with_w2v, params
    if args.ft:
        params.update(dict(embedding='ft'))
        return classify_with_ft, params
    if args.tf:
        params.update(dict(embedding='tf'))
        return classify_with_tf, params
    if args.gpt:
        params.update(dict(embedding='gpt', model='sberbank-ai/rugpt3small_based_on_gpt2'))
        return classify_with_gpt, params
    if args.bert_tiny:
        params.update(dict(embedding='rubert_tiny', model='cointegrated/rubert-tiny'))
        return classify_with_bert, params
    if args.bert_sber:
        params.update(dict(embedding='sbert_large', model='sberbank-ai/sbert_large_nlu_ru'))
        return classify_with_bert, params


def main():
    classify_function, params = get_classifier_function()
    params['res_dir'] = create_res_dir(f"{params['res_dir']}/{params['embedding']}")
    with ProcessPoolExecutor(max_workers=4) as executor:
        for source in sources:
            params1 = dict(params)
            params1.update(source)
            executor.submit(classify_function, **params1)
    # for source in sources:
    #     params.update(source)
    #     classify_function(**params)
    #     print()
    #     print('####################')
    #     print()


if __name__ == "__main__":
    main()
