import csv

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score


def classify_all(x_train, x_test, y_train, y_test, classifiers, file_postfix: str = '', target_names=None,
                 paint_err_matr=False,
                 print_table=False, use_cross_validation=False):
    postfix = "_" + file_postfix if file_postfix else ''
    csvfile_all = open(f'res_all{postfix}.csv', 'w')
    csv_writer_all = csv.writer(csvfile_all)
    with open(f'res_avg{postfix}.csv', 'w', newline='') as csvfile_avg:
        # headers_avg = ['name', 'Score', 'P_micro', 'P_macro', 'R_micro', 'R_macro', 'F1_micro', 'F1_macro']
        headers_avg = ['name', 'P_micro', 'P_macro', 'R_micro', 'R_macro', 'F1_micro', 'F1_macro']
        if use_cross_validation:
            headers_avg += ['Mean_CV', 'Std_SC']
        csv_writer_avg = csv.writer(csvfile_avg)
        csv_writer_avg.writerow(headers_avg)
        with open(f'res_classes{postfix}.csv', 'w', newline='') as csvfile_classes:
            headers_classes = ['name', 'P_не_релевантные', 'R_не_релевантные', 'F1_не_релевантные',
                               'P_общество', 'R_общество', 'F1_общество',
                               'P_образование', 'R_образование', 'F1_образование',
                               'P_наука_и_технологии', 'R_наука_и_технологии', 'F1_наука_и_технологии',
                               ]
            csv_writer_classes = csv.writer(csvfile_classes)
            csv_writer_classes.writerow(headers_classes)
            csv_writer_all.writerow(headers_avg + headers_classes[1:])
            for classifier_info in classifiers:
                if use_cross_validation:
                    classifier_info[0].fit(x_train, y_train)
                    y_res = classifier_info[0].predict(x_test)
                    cr_sc = cross_val_score(classifier_info[0], x_train, y_train, cv=5)
                    # print(f'{classifier_info[1]:{" "}{"<"}{57}}: mean = {cr_sc.mean():1.4f} std = {cr_sc.std():1.4f}')
                    cv_res = {'Mean_CV': cr_sc.mean(), 'Std_SC': cr_sc.std()}
                    # classifier = cross_validate_(classifier_info[0], x_train, y_train)
                    # y_res = classifier.predict(x_test)
                else:
                    classifier_info[0].fit(x_train, y_train)
                    y_res = classifier_info[0].predict(x_test)
                # print_statistics(y_test, x_res, classifier_info[1])
                # print(f"Score:\t\t{classifier_info[0].score(x_train, y_train):1.4f}")
                # print(f'Report {classifier_info[1]}:')
                if target_names:
                    dict_res = classification_report(y_test, y_res, target_names=target_names, digits=4,
                                                     output_dict=True)
                    if print_table:
                        print(f'{classifier_info[1] + ";":{" "}{"<"}{57}} '
                              f'P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
                              f' P_macro: {precision_score(y_test, y_res, average="macro"):1.4f};'
                              f' R_micro: {recall_score(y_test, y_res, average="micro"):1.4f};'
                              f' R_macro: {recall_score(y_test, y_res, average="macro"):1.4f};'
                              f' F1_micro: {f1_score(y_test, y_res, average="micro"):1.4f};'
                              f' F1_macro: {f1_score(y_test, y_res, average="macro"):1.4f};',
                              end='')
                        if use_cross_validation:
                            print(f' Mean_CV: {cv_res["Mean_CV"]:1.4f};'
                                  f' Std_SC: {cv_res["Std_SC"]:1.4f};')
                        else:
                            print()
                        avg = [classifier_info[1],
                               f'{precision_score(y_test, y_res, average="micro"):1.4f}',
                               f'{precision_score(y_test, y_res, average="macro"):1.4f}',
                               f'{recall_score(y_test, y_res, average="micro"):1.4f}',
                               f'{recall_score(y_test, y_res, average="macro"):1.4f}',
                               f'{f1_score(y_test, y_res, average="micro"):1.4f}',
                               f'{f1_score(y_test, y_res, average="macro"):1.4f}']
                        if use_cross_validation:
                            avg += [f'{cv_res["Mean_CV"]:1.4f}', f'{cv_res["Std_SC"]:1.4f}']
                        csv_writer_avg.writerow(avg)
                        classes = [classifier_info[1],
                                   f"{dict_res['not_relevant']['precision']:1.4f}",
                                   f"{dict_res['not_relevant']['recall']:1.4f}",
                                   f"{dict_res['not_relevant']['f1-score']:1.4f}",
                                   f"{dict_res['Общество']['precision']:1.4f}",
                                   f"{dict_res['Общество']['recall']:1.4f}",
                                   f"{dict_res['Общество']['f1-score']:1.4f}",
                                   f"{dict_res['Образование']['precision']:1.4f}",
                                   f"{dict_res['Образование']['recall']:1.4f}",
                                   f"{dict_res['Образование']['f1-score']:1.4f}",
                                   f"{dict_res['Наука и технологии']['precision']:1.4f}",
                                   f"{dict_res['Наука и технологии']['recall']:1.4f}",
                                   f"{dict_res['Наука и технологии']['f1-score']:1.4f}",
                                   ]
                        csv_writer_classes.writerow(classes)
                        csv_writer_all.writerow(avg + classes[1:])
                    else:
                        print(classification_report(y_test, y_res, target_names=target_names, digits=4))
                if paint_err_matr:
                    cm = metrics.confusion_matrix(y_test, y_res)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
                    ax.set(xlabel="Pred", ylabel="True", xticklabels=target_names,
                           yticklabels=target_names, title=classifier_info[1], )
                    plt.yticks(rotation=0)
                    plt.show()


def cross_validate_(clsassifier, x_train, y_train):
    k_fold = StratifiedKFold(n_splits=10)
    cross_validates = cross_validate(clsassifier, x_train, y_train, cv=k_fold, n_jobs=-1, return_estimator=True,
                                     # scoring='f1_macro',
                                     verbose=4
                                     )
    # print(cross_validates['test_score'])
    print(cross_validates['test_score'])
    m = 0
    for i, score in enumerate(cross_validates['test_score']):
        if score > m:
            m = score
            clsassifier = cross_validates['estimator'][i]
    print(m)
    return clsassifier