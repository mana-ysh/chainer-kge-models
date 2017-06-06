
import argparse
from datetime import datetime
import logging
import os
import sys
from chainer import optimizers

sys.path.append('../')
from interfaces.trainer import *
from interfaces.evaluator import Evaluator
from utils.dataset import TripletDataset, Vocab, build_path_and_single, PathQueryDataset


MODEL_CONFIG_FILE = 'model.config'
DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H:%M')))


def train(args):
    # setting for logging
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.log, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('Arguments...')
    for arg, val in vars(args).items():
        logger.info('{:>10} -----> {}'.format(arg, val))

    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)
    n_entity, n_relation = len(ent_vocab), len(rel_vocab)

    if args.meanneg:
        logger.info('using mean negative. number of negative is forced to 1 !!!')
        args.negative = 1

    opt = optimizers.SGD(args.lr)  # TODO: make other optimizers available

    logger.info('building model...')
    if args.method == 'se':
        raise NotImplementedError
        from models.structurede import StructuredE
        model = StructuredE(n_entity=n_entity,
                            n_relation=n_relation,
                            margin=args.margin,
                            dim=args.dim,
                            mean_neg=args.meanneg)
    elif args.method == 'transe':
        from models.transe import TransE
        model = TransE(n_entity=n_entity,
                       n_relation=n_relation,
                       margin=args.margin,
                       dim=args.dim,
                       unit=args.unit)
    elif args.method == 'bi':
        raise NotImplementedError
        from models.bilinear import Bilinear
        model = Bilinear(n_entity=n_entity,
                         n_relation=n_relation,
                         margin=args.margin,
                         dim=args.dim)
    elif args.method == 'bid':
        raise NotImplementedError
        from models.bilinear_diag import BilinearDiag
        model = BilinearDiag(n_entity=n_entity,
                             n_relation=n_relation,
                             margin=args.margin,
                             dim=args.dim)
    elif args.method == 'bitri':
        from models.bilinear_tridiag import BilinearTridiag
        model = BilinearTridiag(n_entity=n_entity,
                                n_relation=n_relation,
                                margin=args.margin,
                                dim=args.dim)
    elif args.method == 'bhole':
        raise NotImplementedError
        from models.basic_hole import HolE
        model = HolE(n_entity=n_entity,
                     n_relation=n_relation,
                     margin=args.margin,
                     dim=args.dim,
                     comp=args.comp)
    elif args.method == 'fhole':
        raise NotImplementedError('including bug')
        from models.fast_hole import HolE
        model = HolE(n_entity=n_entity,
                     n_relation=n_relation,
                     margin=args.margin,
                     dim=args.dim,
                     comp=args.comp)
    elif args.method == 'compe':
        raise NotImplementedError
        from models.compe import ComplexE
        model = ComplexE(n_entity=n_entity,
                         n_relation=n_relation,
                         margin=args.margin,
                         dim=args.dim)
    elif args.method == 'qe':
        raise NotImplementedError
        from models.quaternione import QuaternionE
        model = QuaternionE(n_entity=n_entity,
                            n_relation=n_relation,
                            margin=args.margin,
                            dim=args.dim)
    else:
        raise NotImplementedError

    model.save_config(os.path.join(args.log, MODEL_CONFIG_FILE))

    evaluator = Evaluator(args.metric, args.nbest, args.gpu_id) if args.valid else None

    # preparing data
    if args.task == 'kbc':
        train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab)
        valid_dat = TripletDataset.load(args.valid, ent_vocab, rel_vocab) if args.valid else None
        trainer = PairwiseTrainer(model=model, opt=opt,
                                  batchsize=args.batch, logger=logger,
                                  evaluator=evaluator, valid_dat=valid_dat,
                                  n_negative=args.negative, epoch=args.epoch,
                                  model_dir=args.log, restart=args.restart,
                                  gpu_id=args.gpu_id)
        trainer.fit(train_dat)
    elif args.task == 'pq':
        train_pq_dat, train_single_dat = build_path_and_single(args.train, ent_vocab, rel_vocab)
        valid_dat = PathQueryDataset.load(args.valid, ent_vocab, rel_vocab) if args.valid else None
        trainer = PathPairwiseTrainer(model=model, opt=opt, single=args.single,
                                      batchsize=args.batch, logger=logger,
                                      evaluator=evaluator, valid_dat=valid_dat,
                                      n_negative=args.negative, epoch=args.epoch,
                                      model_dir=args.log, restart=args.restart,
                                      gpu_id=args.gpu_id)

        trainer.fit(train_single_dat, train_pq_dat)
    else:
        raise ValueError('Invalid task: {}'.format(args.task))

    # trainer = PairwiseTrainer(model=model, opt=opt, task=args.task,
    #                           batchsize=args.batch, logger=logger,
    #                           evaluator=evaluator, valid_dat=valid_dat,
    #                           n_negative=args.negative, epoch=args.epoch,
    #                           model_dir=args.log, restart=args.restart)
    #
    # trainer.fit(train_dat)


if __name__ == '__main__':
    p = argparse.ArgumentParser('Link prediction models')
    p.add_argument('--mode', default='pairwise', type=str, help='training mode ["pairwise", "single"]')
    p.add_argument('--task', default='kbc', type=str, help='link prediction task ["kbc", "pq"]')
    p.add_argument('--gpu_id', default=-1, type=int, help='GPU ID. if using CPU, please set -1')

    # dataset
    p.add_argument('--ent', type=str, help='entity list')
    p.add_argument('--rel', type=str, help='relation list')
    p.add_argument('--train', type=str, help='training data')
    p.add_argument('--valid', type=str, help='validation data')

    # model
    p.add_argument('--method', default='se', type=str,
                   help='method ["se", "transe", "bi", "bid", "bhole", "fhole", "bitri", "compe", "qe"]')
    p.add_argument('--restart', default=None, type=str, help='retraining model path')
    p.add_argument('--epoch', default=100, type=int, help='number of epochs')
    p.add_argument('--batch', default=128, type=int, help='batch size')
    p.add_argument('--lr', default=0.001, type=float, help='learning rate')
    p.add_argument('--dim', default=100, type=int, help='dimension of embeddings')
    p.add_argument('--margin', default=1., type=float, help='margin in max-margin loss for pairwise training')
    p.add_argument('--negative', default=10, type=int, help='number of negative samples for pairwise training')
    p.add_argument('--single', default=-1, type=int, help='number of epochs for single training')

    # model-specific config
    p.add_argument('--meanneg', action='store_true',
                   help='mean of representations as negative sample. and num negative is forced to 1')
    p.add_argument('--comp', default='conv', type=str, help='compositional function in HolE ["conv", "corr"]')
    p.add_argument('--unit', action='store_true', help='unitize entity vectors in TransE')

    # evaluation
    p.add_argument('--metric', default='mrr', type=str, help='evaluation metrics ["mr", "mrr", "hits"]')
    p.add_argument('--nbest', default=None, type=int, help='n-best for hits metric')

    # others
    p.add_argument('--log', default=DEFAULT_LOG_DIR, type=str, help='output log dir')

    args = p.parse_args()

    train(args)
