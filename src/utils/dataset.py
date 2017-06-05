"""
TODO
- writing Item class for supporting fancy indexing in PathQueryDataset
"""

import numpy as np
import random


np.random.seed(46)
MAX_NUM_REL = 20


class Dataset(object):
    def __init__(self, samples):
        assert type(samples) == list or type(samples) == np.ndarray
        self.samples = samples if type(samples) == np.ndarray else np.array(samples)

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        pass


class TripletDataset(Dataset):
    def __init__(self, samples):
        super(TripletDataset, self).__init__(samples)

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                samples.append((ent_vocab[sub], rel_vocab[rel], ent_vocab[obj]))
        return TripletDataset(samples)

    @classmethod
    def load_from_pq(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                if len(rel.split(',')) == 1:
                    samples.append((ent_vocab[sub], rel_vocab[rel], ent_vocab[obj]))
        return TripletDataset(samples)


class PathQueryDataset(Dataset):
    def __init__(self, samples):
        super(PathQueryDataset, self).__init__(samples)

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rels, obj = line.strip().split('\t')
                rels = [rel_vocab[r] for r in rels.split(',')]
                # samples.append((ent_vocab[sub], rels, ent_vocab[obj]))
                samples.append(Item(ent_vocab[sub], rels, ent_vocab[obj]))
        return PathQueryDataset(samples)


class Item(object):
    def __init__(self, s, rs, o):
        self.s = s
        self.rs = rs
        self.o = o


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    @classmethod
    def load(cls, vocab_path):
        v = Vocab()
        with open(vocab_path) as f:
            for word in f:
                v.add(word.strip())
        return v


def batch_iter(dataset, batchsize, rand_flg=True):
    n_sample = len(dataset)
    idxs = np.random.permutation(n_sample) if rand_flg else np.arange(n_sample)
    for start_idx in range(0, n_sample, batchsize):
        _data =  dataset[idxs[start_idx:start_idx+batchsize]]
        yield _data[:, 0], _data[:, 1], _data[:, 2]  # subject, relation, object


# for PathQueryDataset
def bucket_batch_iter(dataset, batchsize, rand_flg=True):
    n_r_dist = [0 for _ in range(MAX_NUM_REL)]
    n_r2idx = {i: [] for i in range(MAX_NUM_REL)}
    for i, pq_item in enumerate(dataset.samples):  # TODO: inefficient
        n_r_dist[len(pq_item.rs)-1] += 1
        n_r2idx[len(pq_item.rs)-1].append(i)

    # shufful
    if rand_flg:
        for idxs in n_r2idx.values():
            random.shuffle(idxs)

    while sum(n_r_dist) > 0:
        # select number of relation
        while True:
            i = random.randint(1, MAX_NUM_REL)
            if n_r_dist[i-1] > 0:
                break
        idxs = n_r2idx[i-1][:batchsize]
        _data = dataset[idxs]
        subs = [item.s for item in _data]  # TODO: maybe inefficient
        rels = [item.rs for item in _data]
        objs = [item.o for item in _data]
        yield subs, rels, objs

        n_r_dist[i-1] -= len(idxs)
        n_r2idx[i-1] = n_r2idx[i-1][batchsize:]



def single_batch_iter(dataset, batchsize, rand_flg=True):
    raise NotImplementedError


def build_path_and_single(data_path, ent_vocab, rel_vocab):
    pq_dat = PathQueryDataset.load(data_path, ent_vocab, rel_vocab)
    single_dat = TripletDataset.load_from_pq(data_path, ent_vocab, rel_vocab)
    return pq_dat, single_dat


if __name__ == '__main__':
    ent_file = '../../data/wordnet/train.head200.entlist'
    rel_file = '../../data/wordnet/train.head200.rellist'
    dat_file = '../../data/wordnet/train.head200'

    ent_v = Vocab.load(ent_file)
    rel_v = Vocab.load(rel_file)
    dataset = PathQueryDataset.load(dat_file, ent_v, rel_v)

    for subs, rels, objs in bucket_batch_iter(dataset, 3):
        print(subs)
        print(rels)
        print(objs)
        print('========')
