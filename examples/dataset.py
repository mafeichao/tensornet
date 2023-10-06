# -*- coding: utf-8 -*-
import tensorflow as tf
import tensornet as tn

from common.util import read_dataset, trained_delta_days, dump_predict
from common.config import Config as C

def parse_line_batch(example_proto):
    fea_desc = {
        "uniq_id": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    for slot in set(C.LINEAR_SLOTS + C.DEEP_SLOTS):
        fea_desc[slot]  = tf.io.VarLenFeature(tf.int64)

    feature_dict = tf.io.parse_example(example_proto, fea_desc)
    label = feature_dict.pop('label')
    return feature_dict, label

strategy = tn.distribute.PsStrategy()
with strategy.scope():
    days = C.TRAIN_DAYS[:1]
    train_dataset = read_dataset(C.DATA_DIR, days, C.FILE_MATCH_PATTERN, C.BATCH_SIZE,
                                         parse_line_batch)
    #print(train_dataset)
    for i, (feas, label) in enumerate(train_dataset):
        #print(i, feas, label)
        #print(i, feas['4'])
        print(i, feas['uniq_id'])
