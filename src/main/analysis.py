
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('../')
from interfaces.trainer import PairwiseTrainer
from interfaces.evaluator import Evaluator
from utils.dataset import TripletDataset, Vocab, batch_iter
from libs.hub_analysis.hubness import cal_skewness


def hub_analysis(args):

    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)

    # preparing data
    if args.task == 'kbc':
        test_dat = TripletDataset.load(args.data, ent_vocab, rel_vocab)
    elif args.task == 'pq':
        raise NotImplementedError
    else:
        raise ValueError('Invalid task: {}'.format(args.task))

    with tf.Graph().as_default():
        tf.set_random_seed(46)
        sess = tf.Session()

        print('building model...')
        if args.method == 'se':
            from models.structurede import StructuredE as Model
        elif args.method == 'transe':
            from models.transe import TransE as Model
        elif args.method == 'bi':
            from models.bilinear import Bilinear as Model
        elif args.method == 'bid':
            from models.bilinear_diag import BilinearDiag as Model
        elif args.method == 'bitri':
            from models.bilinear_tridiag import BilinearTridiag as Model
        elif args.method == 'bhole':
            from models.basic_hole import HolE as Model
        elif args.method == 'fhole':
            from models.fast_hole import HolE as Model
        elif args.method == 'compe':
            from models.compe import ComplexE as Model
        elif args.method == 'qe':
            from models.quaternione import QuaternionE as Model
        else:
            raise NotImplementedError

        model = Model.instantiate_model(args.model)
        saver = tf.train.Saver()
        saver.restore(sess, args.model)

        print('calculating scores...')
        score_mat = np.empty((len(test_dat), len(ent_vocab)), dtype=np.float32)
        for i, sample in enumerate(batch_iter(test_dat, 1, rand_flg=False)):
            sub, rel, obj = sample[0]
            feed_dict = {model.sub: [sub], model.rel: [rel]}
            scores = sess.run(model.scores, feed_dict)
            score_mat[i] = scores[:]

        print('calculating skewness...')
        skewness = cal_skewness(score_mat, args.k, 'similarity')
        print('skewness@{}: {}'.format(args.k, skewness))


if __name__ == '__main__':
    p = argparse.ArgumentParser('Hubness analysis for Link prediction models')
    p.add_argument('--task', default='kbc', type=str, help='link prediction task ["kbc", "pq"]')

    # dataset
    p.add_argument('--ent', type=str, help='entity list')
    p.add_argument('--rel', type=str, help='relation list')
    p.add_argument('--data', type=str, help='test data')

    # model
    p.add_argument('--method', default='se', type=str,
                   help='method ["se", "transe", "bi", "bid", "bhole", "fhole", "bitri", "compe", "qe"]')
    p.add_argument('--model', type=str, help='trained model path')

    # evaluation
    p.add_argument('--k', default=10, type=int, help='number of nearest neighbour')

    args = p.parse_args()

    hub_analysis(args)
