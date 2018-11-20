#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import argparse
import os
import nsml
from nsml import DATASET_PATH
from model import Model
from dataset_batch import Dataset
from data_loader import data_loader
from evaluation import get_ner_bi_tag_list_in_sentence
from evaluation import diff_model_label
from evaluation import calculation_measure


def iteration_model(model, dataset, parameter, train=True):
    precision_count = np.array([ 0. , 0. ])
    recall_count = np.array([ 0. , 0. ])
 
    # 학습
    avg_cost = 0.0
    avg_correct = 0.0
    total_labels = 0.0
    for morph, ne_dict, character, seq_len, char_len, label, step in dataset.get_data_batch_size(parameter["batch_size"], train):
        feed_dict = { model.morph : morph,
                      model.ne_dict : ne_dict,
                      model.character : character,
                      model.sequence : seq_len,
                      model.character_len : char_len,
                      model.label : label,
                      model.dropout_rate : parameter["keep_prob"]
                    }

        if train:
            cost, tf_viterbi_sequence, _ = sess.run([model.cost, model.viterbi_sequence, model.train_op], feed_dict=feed_dict)
        else:
            cost, tf_viterbi_sequence = sess.run([model.cost, model.viterbi_sequence], feed_dict=feed_dict)
        avg_cost += cost

        mask = (np.expand_dims(np.arange(parameter["sentence_length"]), axis=0) <
                            np.expand_dims(seq_len, axis=1))
        total_labels += np.sum(seq_len)

        correct_labels = np.sum((label == tf_viterbi_sequence) * mask)
        avg_correct += correct_labels
        precision_count, recall_count = diff_model_label(dataset, precision_count, recall_count, tf_viterbi_sequence, label, seq_len)
        if train and step % 100 == 0:
            print('[Train step: {:>4}] cost = {:>.9} Accuracy = {:>.6}'.format(step + 1, avg_cost / (step+1), 100.0 * avg_correct / float(total_labels)))
        else:
            if step % 100 == 0:
                print('[Test step: {:>4}] cost = {:>.9} Accuracy = {:>.6}'.format(step + 1, avg_cost / (step + 1), 100.0 * avg_correct / float(total_labels)))

    return avg_cost / (step+1), 100.0 * avg_correct / float(total_labels), precision_count, recall_count

# nsml에 저장하고, 로드할 객체들을 bind 합니다.
# submit, fork 명령어를 사용할때 bind된 모델을 불러와서 진행합니다.
def bind_model(sess):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'), global_step=model.global_step)

    def load(dir_name):
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('model loaded!')

    def infer(input, **kwargs):
        pred = []

        # 학습용 데이터셋 구성
        dataset.parameter["train_lines"] = len(input)
        dataset.make_input_data(input)
        reverse_tag = {v: k for k, v in dataset.necessary_data["ner_tag"].items()}

        # 테스트 셋을 측정한다.
        for morph, ne_dict, character, seq_len, char_len, _, step in dataset.get_data_batch_size(len(input), False):
            feed_dict = { model.morph : morph,
                          model.ne_dict : ne_dict,
                          model.character : character,
                          model.sequence : seq_len,
                          model.character_len : char_len,
                          model.dropout_rate : 1.0
                        }

            viters = sess.run(model.viterbi_sequence, feed_dict=feed_dict)
            for index, viter in zip(range(0, len(viters)), viters):
                pred.append(get_ner_bi_tag_list_in_sentence(reverse_tag, viter, seq_len[index]))

        # 최종 output 포맷 예시
        #  [(0.0, ['NUM_B', '-', '-', '-']),
        #   (0.0, ['PER_B', 'PER_I', 'CVL_B', 'NUM_B', '-', '-', '-', '-', '-', '-']),
        #   ( ), ( )
        #  ]
        padded_array = np.zeros(len(pred))

        return list(zip(padded_array, pred))

    # DO NOT CHANGE
    nsml.bind(save=save, load=load, infer=infer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0] + " description")
    parser.add_argument('--verbose', default=False, required=False, action='store_true', help='verbose')

    parser.add_argument('--mode', type=str, default="train", required=False, help='Choice operation mode')
    parser.add_argument('--iteration', type=int, default=0, help='fork 명령어를 사용할때 iteration 값에 매칭되는 모델이 로드됩니다.')
    parser.add_argument('--pause', type=int, default=0, help='모델이 load 될때 1로 설정됩니다.')

    parser.add_argument('--input_dir', type=str, default="data_in", required=False, help='Input data directory')
    parser.add_argument('--output_dir', type=str, default="data_out", required=False, help='Output data directory')
    parser.add_argument('--necessary_file', type=str, default="necessary.pkl")
    parser.add_argument('--train_lines', type=int, default=50, required=False, help='Maximum train lines')

    parser.add_argument('--epochs', type=int, default=20, required=False, help='Epoch value')
    parser.add_argument('--batch_size', type=int, default=10, required=False, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.02, required=False, help='Set learning rate')
    parser.add_argument('--keep_prob', type=float, default=0.65, required=False, help='Dropout_rate')

    parser.add_argument("--word_embedding_size", type=int, default=16, required=False, help='Word, WordPos Embedding Size') 
    parser.add_argument("--char_embedding_size", type=int, default=16, required=False, help='Char Embedding Size') 
    parser.add_argument("--tag_embedding_size", type=int, default=16, required=False, help='Tag Embedding Size') 

    parser.add_argument('--lstm_units', type=int, default=16, required=False, help='Hidden unit size')
    parser.add_argument('--char_lstm_units', type=int, default=32, required=False, help='Hidden unit size for Char rnn')
    parser.add_argument('--sentence_length', type=int, default=180, required=False, help='Maximum words in sentence')
    parser.add_argument('--word_length', type=int, default=8, required=False, help='Maximum chars in word')

    try:
        parameter = vars(parser.parse_args())
    except:
        parser.print_help()
        sys.exit(0)

    # data_loader를 이용해서 전체 데이터셋 가져옴
    if nsml.HAS_DATASET:
        DATASET_PATH = nsml.DATASET_PATH
    else:
        DATASET_PATH = 'data'

    extern_data = []

    # 가져온 문장별 데이터셋을 이용해서 각종 정보 및 학습셋 구성
    dataset = Dataset(parameter, extern_data)


    # Model 불러오기
    model = Model(dataset.parameter)
    model.build_model()

    # tensorflow session 생성 및 초기화
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # DO NOT CHANGE
    bind_model(sess)
    if parameter["pause"] == 1:
        nsml.paused(scope=locals())

    # 학습
    if parameter["mode"] == "train":
        extern_data = data_loader(DATASET_PATH)
        dataset.make_input_data(extern_data)
        for epoch in range(parameter["epochs"]):
            avg_cost, avg_correct, precision_count, recall_count = iteration_model(model, dataset, parameter)
            print('[Epoch: {:>4}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, avg_cost, avg_correct))
            f1Measure, precision, recall = calculation_measure(precision_count, recall_count)
            print('[Train] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(f1Measure, precision, recall))
            nsml.report(summary=True, scope=locals(), train__loss=avg_cost, step=epoch)
            nsml.save(epoch)
