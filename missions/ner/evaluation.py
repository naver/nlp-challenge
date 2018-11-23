# -*- coding: utf-8 -*-
import numpy as np
import argparse


def diff_model_label(dataset, precision, recall, models, labels, seq_len):
    # precision = np.array([ 0. , 0. ]) # 전체 정답 중에 실제 정답 갯수
    # recall = np.array([ 0. , 0. ]) # 실제 태깅한 NER 태그 중에 정답 갯수
    reverse_tag = {v: k for k, v in dataset.necessary_data["ner_tag"].items()}
    for index, model, label in zip(range(0, len(models)), models, labels):
        modelAnswer = get_ner_tag_list_by_numeric(reverse_tag, model, seq_len[index])
        labelAnswer = get_ner_tag_list_by_numeric(reverse_tag, label, seq_len[index])

        recall += calculation_correct(modelAnswer, labelAnswer)
        precision += calculation_correct(labelAnswer, modelAnswer)

    return precision, recall


def calculation_measure(precision, recall):
    if precision[1] == 0:
        precisionRate = 0.0
    else:
        precisionRate = precision[0] / precision[1]

    if recall[1] == 0:
        recallRate = 0.0
    else:
        recallRate = recall[0] / recall[1]

    if precisionRate + recallRate == 0.0:
        f1Measure = 0.0
    else:
        f1Measure = (2 * precisionRate * recallRate) / (precisionRate + recallRate)

    return f1Measure, precisionRate, recallRate

def get_ner_bi_tag_list_in_sentence(reverse_tag, result, max_len):
    nerAnswer = []
    for m in result[:max_len]:
        nerTag = reverse_tag[m]
        if nerTag == "O" or nerTag == "PAD":
            nerAnswer.append("-")
        else:
            nerAnswer.append(nerTag)
    return nerAnswer

def get_ner_tag_list_by_numeric(reverse_tag, result, max_len):
    nerAnswer = []
    nerRange = -1
    nerPrev = ""
    for i, m in enumerate(result[:max_len], start=1):
        if m == 1 or m == 0:
            if nerRange > -1:
                nerAnswer.append(str(nerRange) + ":" + str(i - 1) + "_" + nerPrev)
            nerRange = -1
        else:
            nerTag, nerBI = reverse_tag[m].split("_")

            if nerBI == "B" or nerPrev != nerTag:
                if nerRange > -1:
                    nerAnswer.append(str(nerRange) + ":" + str(i - 1) + "_" + nerPrev)
                nerRange = i
            nerPrev = nerTag
    return nerAnswer


def get_ner_tag_list_by_string(results):
    nerAnswers = []
    nerRange = -1
    nerPrev = ""
    for result in results:
        nerAnswer = []
        for i, tag in enumerate(result, start=1):
            if tag == "-":
                if nerRange > -1:
                    nerAnswer.append(str(nerRange) + ":" + str(i - 1) + "_" + nerPrev)
                nerRange = -1
            else:
                nerTag, nerBI = tag.split("_")

                if nerBI == "B" or nerPrev != nerTag:
                    if nerRange > -1:
                        nerAnswer.append(str(nerRange) + ":" + str(i - 1) + "_" + nerPrev)
                    nerRange = i
                nerPrev = nerTag
        nerAnswers.append(nerAnswer)
    return nerAnswers

def read_prediction(prediction_file):
    pred_array = []
    for line in open(prediction_file, "r", encoding="utf-8"):
        line = line.strip()
        line = eval(line)
        pred_array.append(line)

    return pred_array


def read_ground_truth(ground_truth_file):
    gt_array = []
    for line in open(ground_truth_file, "r", encoding="utf-8"):
        line = line.strip()
        if line == "":
            gt_array.append([])
        else:
            gt_array.append(line.split(" "))
    return gt_array


def evaluation_metrics(prediction_file: str, ground_truth_file: str):
    # read prediction and ground truth from file
    prediction = read_prediction(prediction_file)
    prediction = get_ner_tag_list_by_string(prediction)
    ground_truth = read_ground_truth(ground_truth_file)

    return evaluate(prediction, ground_truth)


def evaluate(prediction, ground_truth):
    precision = np.array([0., 0.])
    recall = np.array([0., 0.])
    for pred, gt in zip(prediction, ground_truth):
        evaluate_by_tag_loc(precision, recall, pred, gt)

    f1, _, _ = calculation_measure(precision, recall)
    return f1


def evaluate_by_tag_loc(precision, recall, models, labels):
    recall += calculation_correct(models, labels)
    precision += calculation_correct(labels, models)

    return precision, recall


def calculation_correct(target, diff):
    value = [0., 0.]  # 0번 정답, 1번 전체갯수
    if isinstance(target, dict):
        for key in target:
            for nerRange in target[key]:
                value[1] += 1
                if key in diff and nerRange in diff[key]:
                    value[0] += 1
    elif isinstance(target, list):
        for ner in target:
            value[1] += 1
            if ner in diff:
                value[0] += 1

    return np.array(value)


if __name__ == "__main__":

    #output type
    #[ ['4:4_CVL', "20:21_EVT", "21:22_FLD"], [] ]

    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()

    test_label_path = '/data/NER/test/test_label'
    try:
        print(evaluation_metrics(config.prediction, test_label_path))
    except:
        # 에러로인한 0점
        print("0")

    # print(evaluation_metrics(config.prediction, test_label_path))
