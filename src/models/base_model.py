
import chainer
import os
import yaml


MODEL_CONFIG_FILE = 'model.config'


class BaseModel(object):
    def __call__(self, pos_samples, neg_samples):
        if pos_samples[1].ndim == 1:  # for KBC
            loss = self._single_forward(pos_samples, neg_samples)
        elif pos_samples[1].ndim == 2:  # for path query
            loss = self._path_forward(pos_samples, neg_samples)
        else:
            raise ValueError('Invalid')
        return loss

    def cal_scores(self, subs, rels):
        if rels.ndim == 1:
            scores = self._single_scores(subs, rels)
        elif rels.ndim == 2:
            scores = self._path_scores(subs, rels)
        else:
            raise ValueError('Invalid')
        return scores

    def _single_forward(self, pos_samples, neg_samples):
        raise NotImplementedError

    def _path_forward(self, pos_samples, neg_samples):
        raise NotImplementedError

    def _single_scores(self, subs, rels):
        raise NotImplementedError

    def _path_scores(self, subs, rels):
        raise NotImplementedError

    def _composite(self):
        raise NotImplementedError

    def _cal_similarity(self):
        raise NotImplementedError

    def init_reverse(self):
        raise NotImplementedError

    def save_config(self, config_path):
        with open(config_path, 'w') as fw:
            fw.write(yaml.dump(self.model_config))

    @classmethod
    def instantiate_model(cls, model_path):
        """
        instantiation of models from config file for testing
        """
        model_dir = os.path.dirname(model_path)
        with open(os.path.join(model_dir, MODEL_CONFIG_FILE)) as f:
            model_config = yaml.load(f)
        return cls(**model_config)
