import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from classifier_lists import full_classifier_list, short_classifier_list
from clsassifier_iterator import classify_all
from utils import form_y_prep, form_label_map, create_res_dir, parse_arguments, print_info, form_res_path


def vectorizors(use_std_sclr: bool = False):
    vectrs = []
    if use_std_sclr:
        vectrs.append(StandardScaler())
    return vectrs


def classify_with_tf(**params):
    # http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    train_data_source = params['train_data_source']
    test_data_source = params.get('test_data_source', train_data_source)
    use_std_sclr = params.get('use_std_sclr', False)
    res_dir = params.get('res_dir')
    use_pca = params.get('use_pca', False)
    save_err_matr = params.get('save_err_matr', True)
    use_short_classifiers_list = params.get('short', False)
    test_size = 0.2
    n_components = 500
    tfidf_params = dict(min_df=10, max_df=0.8, ngram_range=(1, 2))
    print_info(**params, test_size=test_size, n_components=n_components,
               model_info=tfidf_params)

    data = pd.read_json('articles_w_m_t.json')
    y_prep = form_y_prep(data["user_categories"])
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    # x_train = tfidfconverter.fit_transform([' '.join(t) for t in x_train[train_data_source]]).toarray()
    # x_test = tfidfconverter.transform([' '.join(t) for t in x_test[test_data_source]]).toarray()
    x_train = [' '.join(t) for t in x_train[train_data_source]]
    x_test = [' '.join(t) for t in x_test[test_data_source]]

    vectorizer = TfidfVectorizer(**tfidf_params)
    vectorizer.fit(x_train)
    vectorizors = [vectorizer]
    if use_std_sclr:
        vectorizors.append(StandardScaler(with_mean=False))
    if use_pca:
        vectorizors.append(TruncatedSVD(n_components=n_components))
    if use_short_classifiers_list:
        classifiers = short_classifier_list(vectorizors)
    else:
        classifiers = full_classifier_list(vectorizors)

    res_dir_path = form_res_path(res_dir, train_data_source, test_data_source)
    res_dir = create_res_dir(res_dir_path)
    print_info(**params, file=f'{res_dir_path}/info.txt', test_size=test_size, n_components=n_components,
               model_info=tfidf_params)

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


def test(test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(data, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    tfidfconverter = TfidfVectorizer(min_df=5, max_df=0.7)
    X_train = tfidfconverter.fit_transform([' '.join(t) for t in x_train[train_data_source]]).toarray()
    # clsassifier = RandomForestClassifier(bootstrap=False, criterion='entropy', n_estimators=100)
    clsassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1, base_estimator=SVC(kernel='linear', probability=True))
    clsassifier.fit(X_train, y_train)
    X_test = tfidfconverter.transform([' '.join(t) for t in x_test[test_data_source]]).toarray()
    y_res = clsassifier.predict(X_test)
    # print(classification_report(y_test, y_res, target_names=list(label_map.keys()), digits=4))
    print(f'{"SVC TfidfVectorizer;":{" "}{"<"}{57}} '
          f'P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
          f' P_macro: {precision_score(y_test, y_res, average="macro"):1.4f};'
          f' R_micro: {recall_score(y_test, y_res, average="micro"):1.4f};'
          f' R_macro: {recall_score(y_test, y_res, average="macro"):1.4f};'
          f' F1_micro: {f1_score(y_test, y_res, average="micro"):1.4f};'
          f' F1_macro: {f1_score(y_test, y_res, average="macro"):1.4f};'
          )


def grid_search(use_whole_text: bool, test_data_source: str, train_data_source: str):
    data = pd.read_json('articles_w_m_t.json')
    y = np.asarray(data["user_categories"])
    label_map = {cat: index for index, cat in enumerate(np.unique(y))}
    y_prep = np.asarray([label_map[l] for l in y])
    test_size = 0.2
    tfidfconverter = TfidfVectorizer(min_df=5, max_df=0.7, ngram_range=(1, 2))
    X = tfidfconverter.fit_transform([' '.join(t) for t in data[train_data_source]]).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, y_prep, test_size=test_size, random_state=42,
                                                        stratify=y_prep)
    grid_param = {
        'classify__n_estimators': [100, 500],
        'classify__learning_rate': [1, 0.5, 0.3]
    }
    clsassifier = Pipeline([
        ('classify', AdaBoostClassifier(base_estimator=SVC(kernel='linear', probability=True)))
    ])

    gd_sr = GridSearchCV(estimator=clsassifier,
                         param_grid=grid_param,
                         # scoring='accuracy',
                         scoring='f1_macro',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(x_train, y_train)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_
    print(best_result)
    data = pd.DataFrame(gd_sr.cv_results_)
    print(data[['mean_test_score', 'std_test_score', 'params']])
    print()


if __name__ == '__main__':
    # args = initialize_argument_parser().parse_args()
    # classify_with_tf(args.test_data_source, args.train_data_source, args.use_cross_validation, args.use_std_sclr)
    args = parse_arguments()
    classify_with_tf(**vars(args))
    # test(args.test_data_source, args.train_data_source)
    # grid_search(args.use_whole_text, args.test_data_source, args.train_data_source)