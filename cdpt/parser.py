# -*- coding: utf-8 -*-

import json
import fire
import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .modules import XPOSTagger, UPOSTagger
from .modules import NERBIOTagger
from .modules import HSelParser, RelLabeler
from .modules import MWEJointDecoder
from .features import WordSequence
from .io import read_conll, write_conll
from .utils import buildVocab
from .data import DataProcessor, DataCollate, InfiniteDataLoader
from .adamw import AdamW
from .eval import parse_f1, bio_f1, parse_f1_ud


class CDParser:

    def __init__(self, **kwargs):
        pass

    def create_parser(self, **kwargs):
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
            sys.stdout.flush()

        self._args = kwargs

        self._gpu = kwargs.get("gpu", True)

        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)
        self._epsilon = kwargs.get("epsilon", 1e-8)
        self._weight_decay = kwargs.get("weight_decay", 0.)
        self._warmup = kwargs.get("warmup", 800)

        self._clip = kwargs.get("clip", 5.)

        self._batch_size = kwargs.get("batch_size", 16)

        self._word_smooth = kwargs.get("word_smooth", 0.25)
        self._char_smooth = kwargs.get("char_smooth", 0.25)

        self._wdims = kwargs.get("wdims", 128)
        self._edims = kwargs.get("edims", 0)
        self._cdims = kwargs.get("cdims", 32)
        self._pdims = kwargs.get("pdims", 0)

        self._word_dropout = kwargs.get("word_dropout", 0.0)

        self._char_hidden = kwargs.get("char_hidden", 128)
        self._char_dropout = kwargs.get("char_dropout", 0.0)
        self._bilstm_dims = kwargs.get("bilstm_dims", 256)
        self._bilstm_layers = kwargs.get("bilstm_layers", 2)
        self._bilstm_dropout = kwargs.get("bilstm_dropout", 0.0)

        self._utagger_dims = kwargs.get("utagger_dims", 256)
        self._utagger_layers = kwargs.get("utagger_layers", 1)
        self._utagger_dropout = kwargs.get("utagger_dropout", 0.0)
        self._utagger_weight = kwargs.get("upos_weight", 0.1)

        self._xtagger_weight = kwargs.get("xpos_weight", 0.1)

        self._hsel_dims = kwargs.get("hsel_dims", 200)
        self._hsel_dropout = kwargs.get("hsel_dropout", 0.0)

        self._rel_dims = kwargs.get("rel_dims", 50)
        self._rel_dropout = kwargs.get("rel_dropout", 0.0)

        self._parsing_weight = kwargs.get("parsing_weight", 0.5)

        self._biocrf = kwargs.get("biocrf", False)

        self._bert = kwargs.get("bert", True)
        self._transformer = kwargs.get("transformer", False)
        self._trans_pos_dim = kwargs.get("trans_pos_dim", 128)
        self._trans_ffn_dim = kwargs.get("trans_ffn_dim", 1024)
        self._trans_emb_dropout = kwargs.get("trans_emb_dropout", 0.)
        self._trans_num_layers = kwargs.get("trans_num_layers", 8)
        self._trans_num_heads = kwargs.get("trans_num_heads", 8)
        self._trans_attn_dropout = kwargs.get("trans_attn_dropout", 0.2)
        self._trans_actn_dropout = kwargs.get("trans_actn_dropout", 0.1)
        self._trans_res_dropout = kwargs.get("trans_res_dropout", 0.2)

        if self._bert:
            self._learning_rate = 1e-5
        elif self._transformer:
            self._weight_decay = 1e-5
            self._bilstm_dims = 512
            self._learning_rate = 1e-4
        else:
            self._learning_rate = 1e-3

        self._mode = kwargs.get("mode", "jointdecoding")


        self.init_model()
        return self

    def _load_vocab(self, vocab):
        self._fullvocab = vocab
        self._upos = {p: i + 1 for i, p in enumerate(vocab["upos"])}
        self._xpos = {p: i + 1 for i, p in enumerate(vocab["xpos"])}
        self._xupos = {p: i + 1 for i, p in enumerate(vocab["xupos"])}
        self._rels = {p: i + 1 for i, p in enumerate(vocab["rels"])}
        self._irels = ["**"] + vocab["rels"]
        self._vocab = {w: i + 2 for i, w in enumerate(vocab["vocab"])}
        self._charset = {c: i + 2 for i, c in enumerate(vocab["charset"])}
        self._wordfreq = vocab["wordfreq"]
        self._charfreq = vocab["charfreq"]

    def load_vocab(self, filename):
        with open(filename, "r") as f:
            vocab = json.load(f)
        self._load_vocab(vocab)
        return self

    def save_vocab(self, filename):
        with open(filename, "w") as f:
            json.dump(self._fullvocab, f)
        return self

    def build_vocab(self, filename, cutoff=1):
        graphs = read_conll(filename)

        self._fullvocab= buildVocab(graphs, cutoff)
        self._load_vocab(self._fullvocab)

        return self

    def load_embeddings(self, filename):
        if not os.path.isfile(filename + ".vocab"):
            return self

        with open(filename + ".vocab", "r") as f:
            _external_mappings = json.load(f)
        _external_embeddings = np.load(filename + ".npy")
        _external_embeddings /= np.std(_external_embeddings)

        count = 0
        for w in self._vocab:
            if w in _external_mappings:
                # self._bilstm.wordrep.word_embedding.weight.data[self._vocab[w]].copy_(torch.from_numpy(_external_embeddings[_external_mappings[w]]))
                count += 1
        print("Loaded embeddings from", filename, count, "hits out of", len(self._vocab))
        self._external_mappings = _external_mappings
        self._external_embeddings = _external_embeddings

        return self

    def save_model(self, filename):
        print("Saving model to", filename)
        self.save_vocab(filename + ".vocab")
        with open(filename + ".params", "w") as f:
            json.dump(self._args, f)
        torch.save(self._model.state_dict(), filename + '.model')

    def load_model(self, filename, **kwargs):
        print("Loading model from", filename)
        self.load_vocab(filename + ".vocab")
        with open(filename + ".params", "r") as f:
            args = json.load(f)
            args.update(kwargs)
            self.create_parser(**args)
        self._model.load_state_dict(torch.load(filename + ".model"))
        return self

    def init_model(self):
        self._seqrep = WordSequence(self)

        self._xpos_tagger = XPOSTagger(self, self._utagger_layers, self._utagger_dims, len(self._xpos) + 1, self._utagger_dropout)
        self._xpos_tagger.l_weight = self._xtagger_weight
        self._upos_tagger = UPOSTagger(self, self._utagger_layers, self._utagger_dims, len(self._upos) + 1, self._utagger_dropout)
        self._upos_tagger.l_weight = self._utagger_weight
        self._ner_tagger = NERBIOTagger(self, self._utagger_layers, self._utagger_dims, 4, self._utagger_dropout, crf=self._biocrf)
        if self._mode.startswith("joint"):
            self._ner_tagger.l_weight = 1. - self._parsing_weight
        else:
            self._ner_tagger.l_weight = 1.

        self._hsel_parser = HSelParser(self, self._hsel_dims, self._hsel_dropout)
        if self._mode.startswith("joint"):
            self._hsel_parser.l_weight = 0.5 * self._parsing_weight
        else:
            self._hsel_parser.l_weight = 0.5

        self._rel_labeler = RelLabeler(self, self._rel_dims, self._rel_dropout)
        if self._mode.startswith("joint"):
            self._rel_labeler.l_weight = 0.5 * self._parsing_weight
        else:
            self._rel_labeler.l_weight = 0.5

        self._joint_decoder = MWEJointDecoder(self._hsel_parser, self._rel_labeler, self._ner_tagger)
        self._joint_decoder.l_weight = 0.0

        # baseline parsing
        if self._mode == "parsing":
            self._modules = [self._upos_tagger, self._xpos_tagger, self._hsel_parser, self._rel_labeler]

        # baseline tagging
        elif self._mode == "tagging":
            self._modules = [self._upos_tagger, self._xpos_tagger, self._ner_tagger]

        # joint training
        elif self._mode == "jointparsing" or self._mode == "jointtagging":
            self._modules = [self._upos_tagger, self._xpos_tagger, self._ner_tagger, self._hsel_parser, self._rel_labeler]

        elif self._mode == "jointdecoding":
            self._modules = [self._upos_tagger, self._xpos_tagger, self._ner_tagger, self._hsel_parser, self._rel_labeler, self._joint_decoder]

        for m in self._modules:
            print(m.name, m.l_weight)

        self._model = nn.ModuleList([self._seqrep] + self._modules)

        if self._gpu:
            self._model.cuda()

        return self

    def train(self, filename, eval_steps=100, decay_evals=5, decay_times=0, decay_ratio=0.5, dev=None, save_prefix=None, **kwargs):

        train_graphs = DataProcessor(filename, self, self._model)
        train_loader = InfiniteDataLoader(train_graphs, batch_size=self._batch_size, shuffle=True, num_workers=1, collate_fn=DataCollate(self, train=True))
        gold_dev_graphs = DataProcessor(dev, self, self._model)
        dev_graphs = DataProcessor(dev, self, self._model)

        optimizer = AdamW(self._model.parameters(), lr=self._learning_rate, betas=(self._beta1, self._beta2), eps=self._epsilon, weight_decay=self._weight_decay, amsgrad=False, warmup=self._warmup)

        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=decay_ratio, patience=decay_evals, verbose=True, cooldown=1)
        cooldown_counter = 3

        print("Model")
        for param_tensor in self._model.state_dict():
            print(param_tensor, "\t", self._model.state_dict()[param_tensor].size())
        print("Opt")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        t0 = time.time()
        results, eloss = defaultdict(float), 0.
        max_dev = 0.

        for batch_i, batch in enumerate(train_loader):
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw":
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            self._model.train()
            self._model.zero_grad()

            loss = []
            seq_features = self._seqrep(batch)

            for module in self._modules:
                batch_label = module.batch_label(batch)
                l, pred = module.calculate_loss(seq_features, batch)
                if l is not None:
                    loss.append(l * module.l_weight)
                    module.evaluate(results, pred, batch_label, mask)

            loss = sum(loss)
            eloss += loss.item()
            loss.backward()

            skip = False

            if not skip:
                nn.utils.clip_grad_norm_(self._model.parameters(), self._clip)
                optimizer.step()

            if batch_i and batch_i % 100 == 0:
                for module in self._modules:
                    module.metrics(results)
                results["loss/loss"] = eloss
                print(batch_i // 100, "{:.2f}s".format(time.time() - t0), end=" ")
                sys.stdout.flush()
                results, eloss = defaultdict(float), 0.
                t0 = time.time()

            if batch_i and (batch_i % eval_steps == 0):
                results = self.evaluate(dev_graphs)

                if self._mode in {"jointdecoding", "jointparsing", "parsing"}:
                    if "mwe_NNP" in self._rels:
                        performance = parse_f1(gold_dev_graphs.graphs, dev_graphs.graphs)
                    else:
                        performance = parse_f1_ud(gold_dev_graphs.graphs, dev_graphs.graphs)
                else:
                    performance = bio_f1(gold_dev_graphs.graphs, dev_graphs.graphs)

                results = defaultdict(float)
                scheduler.step(performance)
                if scheduler.in_cooldown:
                    optimizer.state = defaultdict(dict)
                    print("Learning rate drops")
                    cooldown_counter -= 1
                    if cooldown_counter < 0:
                        break

                print()
                print(performance)
                if performance >= max_dev:
                    max_dev = performance
                    if save_prefix:
                        self.save_model("{}model".format(save_prefix))
                print(max_dev)
                print()

        return self

    def evaluate(self, data, output_file=None):
        results = defaultdict(float)
        pred_scores = []
        pred_results = []
        gold_results = []
        self._model.eval()
        batch_size = self._batch_size
        start_time = time.time()
        train_num = len(data)
        total_batch = train_num // batch_size + 1

        dev_loader = DataLoader(data, batch_size=self._batch_size, shuffle=False, num_workers=1, collate_fn=DataCollate(self, train=False))

        for batch in dev_loader:
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw":
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            seq_features = self._seqrep(batch)

            for module in self._modules:
                batch_label = module.batch_label(batch)
                pred = module(self, seq_features, batch)
                module.evaluate(results, pred, batch_label, mask)

            if "pred_head" in batch and "pred_rel" in batch:
                for idx, h, r in zip(batch["graphidx"], batch["pred_head"].cpu().data.numpy(), batch["pred_rel"].cpu().data.numpy()):
                    g = data.graphs[idx]
                    for i in range(1, len(g.nodes)):
                        g.heads[i] = h[i] - 1
                        g.rels[i] = self._irels[r[i]]
            if "pred_NERBIO" in batch:
                for idx, t in zip(batch["graphidx"], batch["pred_NERBIO"].cpu().data.numpy()):
                    g = data.graphs[idx]
                    for i in range(1, len(g.nodes)):
                        if t[i] == 1:
                            g.nodes[i].feats = "B"
                        elif t[i] == 2:
                            g.nodes[i].feats = "I"
                        else:
                            g.nodes[i].feats = "_"


        decode_time = time.time() - start_time
        results["speed/speed"] = len(data)/decode_time

        for module in self._modules:
            module.metrics(results)

        if output_file:
            write_conll(output_file, data.graphs)

        print(results)
        return results

    def finish(self, **kwargs):
        print()
        sys.stdout.flush()


if __name__ == '__main__':
    fire.Fire(CDParser)
