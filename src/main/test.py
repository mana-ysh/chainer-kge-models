
import argparse
from chainer import serializers
from datetime import datetime
import logging
import os
import sys

sys.path.append('../')
from interfaces.trainer import PairwiseTrainer
from interfaces.evaluator import Evaluator
from utils.dataset import TripletDataset, Vocab, PathQueryDataset


def test(args):

    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)

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
    serializers.load_hdf5(args.model, model)

    # preparing data
    print('preparing data...')
    if args.task == 'kbc':
        test_dat = TripletDataset.load(args.data, ent_vocab, rel_vocab)
    elif args.task == 'pq':
        test_dat = PathQueryDataset.load(args.data, ent_vocab, rel_vocab)
        if args.init_reverse:
            print('initialize reverse relations...')
            model.init_reverse()
    else:
        raise ValueError('Invalid task: {}'.format(args.task))

    evaluator = Evaluator(args.metric, args.nbest)
    print('evaluating...')
    res = evaluator.run(model, test_dat)

    if args.metric == 'mrr':
        print('MRR: {}'.format(res))
    elif args.metric == 'mr':
        print('MR: {}'.format(res))
    elif args.metric == 'hits':
        print('HITS@{}: {}'.format(args.nbest, res))
    else:
        raise ValueError


if __name__ == '__main__':
    p = argparse.ArgumentParser('Testing for Link prediction models')
    p.add_argument('--task', default='kbc', type=str, help='link prediction task ["kbc", "pq"]')
    p.add_argument('--gpu', default=-1, type=int, help='GPU ID. if using CPU, please set -1')

    # dataset
    p.add_argument('--ent', type=str, help='entity list')
    p.add_argument('--rel', type=str, help='relation list')
    p.add_argument('--data', type=str, help='test data')

    # model
    p.add_argument('--method', default='se', type=str,
                   help='method ["se", "transe", "bi", "bid", "bhole", "fhole", "bitri", "compe", "qe"]')
    p.add_argument('--model', type=str, help='trained model path')
    p.add_argument('--init_reverse', action='store_true',
                   help='initialize latent representations of reverse relations')

    # evaluation
    p.add_argument('--metric', default='mrr', type=str, help='evaluation metrics ["mrr", "hits", "mr"]')
    p.add_argument('--nbest', default=-1, type=int, help='n-best for hits metric')

    args = p.parse_args()

    test(args)
