
import sys
import tensorflow as tf

sys.path.append('../')
from models.base_model import BaseModel


class BilinearDiag(BaseModel):
    def __init__(self, **kwargs):
        self.n_entity = kwargs['n_entity']
        self.n_relation = kwargs['n_relation']
        self.dim = kwargs['dim']
        if 'margin' in kwargs:
            self.margin = kwargs['margin']
            self._setup_pairwise()
        else:
            raise NotImplementedError
            self._setup_single()

        self._setup_scorer()
        self.model_config = kwargs

    def _setup_pairwise(self):
        self.pos_sub = tf.placeholder(tf.int32, [None])
        self.pos_rel = tf.placeholder(tf.int32, [None])
        self.pos_obj = tf.placeholder(tf.int32, [None])
        self.neg_sub = tf.placeholder(tf.int32, [None])
        self.neg_rel = tf.placeholder(tf.int32, [None])
        self.neg_obj = tf.placeholder(tf.int32, [None])

        with tf.name_scope('params'):
            initializer = tf.contrib.layers.xavier_initializer()
            self.ent_embeds = tf.get_variable(name='ent_embeds', shape=(self.n_entity, self.dim), initializer=initializer)
            self.rel_embeds = tf.get_variable(name='rel_embeds', shape=(self.n_relation, self.dim), initializer=initializer)

            pos_sub_embed = tf.nn.embedding_lookup(self.ent_embeds, self.pos_sub)
            pos_rel_embed = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rel)
            pos_obj_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.pos_obj), [-1, self.dim, 1])
            neg_sub_embed = tf.nn.embedding_lookup(self.ent_embeds, self.neg_sub)
            neg_rel_embed = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rel)
            neg_obj_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.neg_obj), [-1, self.dim, 1])

            pos_query = tf.reshape(tf.multiply(pos_sub_embed, pos_rel_embed), [-1, 1, self.dim])
            neg_query = tf.reshape(tf.multiply(neg_sub_embed, neg_rel_embed), [-1, 1, self.dim])

            pos_score = tf.matmul(pos_query, pos_obj_embed)
            neg_score = tf.matmul(neg_query, neg_obj_embed)

        with tf.name_scope('output'):
            self.loss = tf.reduce_sum(tf.maximum(0., self.margin - (pos_score - neg_score)))

    def _setup_scorer(self):
        self.sub = tf.placeholder(tf.int32, 1)
        self.rel = tf.placeholder(tf.int32, 1)
        with tf.name_scope('predict'):
            sub_embed = tf.nn.embedding_lookup(self.ent_embeds, self.sub)
            rel_embed = tf.nn.embedding_lookup(self.rel_embeds, self.rel)
            query = tf.multiply(sub_embed, rel_embed)
            self.scores = tf.reshape(tf.matmul(query, tf.transpose(self.ent_embeds)), [self.n_entity])

    def _setup_single(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

