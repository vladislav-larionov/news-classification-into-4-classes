from pathlib import Path

import csv

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


def classify_all(x_train, x_test, y_train, y_test, classifiers, res_dir: Path, target_names=None,
                 paint_err_matr=False,
                 print_table=False,
                 save_err_matr=False
                 ):
    csvfile_all = open(f'{res_dir}/res_all.csv', 'w')
    csv_writer_all = csv.writer(csvfile_all)
    with open(f'{res_dir}/res_avg.csv', 'w', newline='') as csvfile_avg:
        headers_avg = ['name', 'P_micro', 'P_macro', 'R_micro', 'R_macro', 'F1_micro', 'F1_macro']
        csv_writer_avg = csv.writer(csvfile_avg)
        csv_writer_avg.writerow(headers_avg)
        with open(f'{res_dir}/res_classes.csv', 'w', newline='') as csvfile_classes:
            headers_classes = ['name', 'P_не_релевантные', 'R_не_релевантные', 'F1_не_релевантные',
                               'P_общество', 'R_общество', 'F1_общество',
                               'P_образование', 'R_образование', 'F1_образование',
                               'P_наука_и_технологии', 'R_наука_и_технологии', 'F1_наука_и_технологии',
                               ]
            csv_writer_classes = csv.writer(csvfile_classes)
            csv_writer_classes.writerow(headers_classes)
            csv_writer_all.writerow(headers_avg + headers_classes[1:])
            for classifier_info in classifiers:
                classifier_info[0].fit(x_train, y_train)
                y_res = classifier_info[0].predict(x_test)
                if target_names:
                    dict_res = classification_report(y_test, y_res, target_names=target_names, digits=4,
                                                     output_dict=True)
                    if not print_table:
                        print(f'{classifier_info[1] + ";":{" "}{"<"}{83}} '
                              f'P_micro: {precision_score(y_test, y_res, average="micro"):1.4f};'
                              f' P_macro: {precision_score(y_test, y_res, average="macro"):1.4f};'
                              f' R_micro: {recall_score(y_test, y_res, average="micro"):1.4f};'
                              f' R_macro: {recall_score(y_test, y_res, average="macro"):1.4f};'
                              f' F1_micro: {f1_score(y_test, y_res, average="micro"):1.4f};'
                              f' F1_macro: {f1_score(y_test, y_res, average="macro"):1.4f};',
                              end='')
                        print()
                        avg = [classifier_info[1],
                               f'{precision_score(y_test, y_res, average="micro"):1.4f}',
                               f'{precision_score(y_test, y_res, average="macro"):1.4f}',
                               f'{recall_score(y_test, y_res, average="micro"):1.4f}',
                               f'{recall_score(y_test, y_res, average="macro"):1.4f}',
                               f'{f1_score(y_test, y_res, average="micro"):1.4f}',
                               f'{f1_score(y_test, y_res, average="macro"):1.4f}']
                        csv_writer_avg.writerow(avg)
                        classes = [classifier_info[1],
                                   # f"{dict_res['not_relevant']['precision']:1.4f}",
                                   # f"{dict_res['not_relevant']['recall']:1.4f}",
                                   # f"{dict_res['not_relevant']['f1-score']:1.4f}",
                                   f"{dict_res['Нерелевантная']['precision']:1.4f}",
                                   f"{dict_res['Нерелевантная']['recall']:1.4f}",
                                   f"{dict_res['Нерелевантная']['f1-score']:1.4f}",
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
                    cm.show()
                if save_err_matr:
                    cm = create_confusion_matrix_figure(y_test, y_res, target_names, classifier_info)
                    cm.savefig(f'{res_dir}/cm {classifier_info[1]}.png')


def create_confusion_matrix_figure(y_test, y_res, target_names, classifier_info):
    cm = metrics.confusion_matrix(y_test, y_res)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=True)
    # ax.set(xlabel="Pred", ylabel="True", xticklabels=target_names,
    ax.set(xlabel="Назначенные", ylabel="Истинные", xticklabels=target_names,
           yticklabels=target_names, title=classifier_info[1], )
    plt.yticks(rotation=0)
    return fig