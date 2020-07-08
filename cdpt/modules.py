# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from .crf import CRF
from .attention import BilinearMatrixAttention

import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from .calgorithm import parse_proj, parse_mwe, parse_mwe_ud
from .crf_algorithm import crf_marginal


class ParserModule(ABC):

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @staticmethod
    @abstractmethod
    def load_data(parser, graph):
        pass

    @staticmethod
    @abstractmethod
    def batch_label(batch):
        pass

    @abstractmethod
    def evaluate(self, results, pred, gold, mask):
        pass

    @abstractmethod
    def metrics(self, results):
        pass


class SequenceLabeler(nn.Module, ParserModule):

    def __init__(self, parser, layer_size, hidden_size, label_size, dropout=0., crf=False):
        super(SequenceLabeler, self).__init__()
        print("build sequence labeling network...", self.__class__.name, "crf:", crf)

        self.use_crf = crf
        ## add two more label for downlayer lstm, use original label size for CRF
        self.label_size = label_size + 2

        lst = []
        for i in range(layer_size):
            if i == 0:
                lst.append(nn.Linear(parser._bilstm_dims, hidden_size))
            else:
                lst.append(nn.Linear(hidden_size, hidden_size))

            lst.append(nn.PReLU())
            lst.append(nn.Dropout(dropout))

        if layer_size > 0:
            lst.append(nn.Linear(hidden_size, self.label_size))
        else:
            lst.append(nn.Linear(parser._bilstm_dims, self.label_size))

        self.transform = nn.Sequential(*lst)

        if self.use_crf:
            self.crf = CRF(label_size)
        else:
            self.loss = nn.NLLLoss(ignore_index=0, reduction='sum')

    def calculate_loss(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        outs = self.transform(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask_h, batch_label)
            total_loss = total_loss / float(batch_size)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask_h)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = self.loss(score, batch_label.view(batch_size * seq_len))
            total_loss = total_loss / float(batch_size)
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)

        return total_loss, tag_seq


    def forward(self, parser, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        outs = self.transform(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask_h)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq  = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq

        batch["pred_" + self.name] = tag_seq

        return tag_seq

    def scores(self, lstm_features, batch):
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        batch_size = mask.size(0)
        seq_len = mask.size(1)
        if self.use_crf:
            outs = self.transform(lstm_features).view(batch_size, seq_len, -1)
            scores = outs.cpu().data.numpy().astype('float64')
            transitions = self.crf.transitions.cpu().data.numpy().astype('float64')
            lengths = torch.sum(mask_h.long(), dim=1).view(batch_size,).long().cpu().data.numpy()
            ret = torch.zeros_like(outs)

            for i in range(batch_size):
                length = lengths[i]
                ret[i, :length, :] = torch.Tensor(crf_marginal(scores[i, :length, :], transitions))
            return ret
        else:
            outs = self.transform(lstm_features).view(batch_size * seq_len, -1)
            return F.log_softmax(outs, dim=1).view(batch_size, seq_len, -1)

    def evaluate(self, results, pred, gold, mask):
        overlaped = (pred == gold).byte()
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = correct / (total + 1e-10) * 100.
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]


class XPOSTagger(SequenceLabeler):

    name = "XPOS"

    @staticmethod
    def load_data(parser, graph):
        labels = [0] + [parser._xpos.get(n.xpos, 1) for n in graph.nodes[1:]]
        return {"xpos": labels}

    @staticmethod
    def batch_label(batch):
        return batch["xpos"]


class UPOSTagger(SequenceLabeler):

    name = "UPOS"

    @staticmethod
    def load_data(parser, graph):
        labels = [0] + [parser._upos.get(n.upos, 1) for n in graph.nodes[1:]]
        return {"upos": labels}

    @staticmethod
    def batch_label(batch):
        return batch["upos"]


class NERBIOTagger(SequenceLabeler):

    name = "NERBIO"

    @staticmethod
    def load_data(parser, graph):
        ner_dic = {}
        labels = [3 for i in range(len(graph.nodes))]
        for i in range(1, len(graph.nodes)):
            if graph.nodes[i].feats == "B":
                labels[i] = 1
            elif graph.nodes[i].feats == "I":
                labels[i] = 2
        return {"nerbio": labels}

    @staticmethod
    def batch_label(batch):
        return batch["nerbio"]

    def evaluate(self, results, pred, gold, mask):
        batch_size, length = mask.shape
        pred = pred.cpu().data.numpy()
        gold = gold.cpu().data.numpy()
        mask = mask.cpu().data.numpy()
        pred_set = set()
        gold_set = set()
        for i in range(batch_size):
            curstart = None
            for j in range(1, length):
                if mask[i, j] == 0:
                    if curstart is not None:
                        pred_set.add((i, curstart, j))
                    break
                if pred[i, j] == 1:
                    if curstart is not None:
                        pred_set.add((i, curstart, j))
                    curstart = j
                elif pred[i, j] == 2:
                    if curstart is None:
                        curstart = j
                elif pred[i, j] == 3:
                    if curstart is not None:
                        pred_set.add((i, curstart, j))
                    curstart = None

        for i in range(batch_size):
            curstart = None
            for j in range(1, length):
                if mask[i, j] == 0:
                    if curstart is not None:
                        gold_set.add((i, curstart, j))
                    break
                if gold[i, j] == 1:
                    if curstart is not None:
                        gold_set.add((i, curstart, j))
                    curstart = j
                elif gold[i, j] == 2:
                    if curstart is None:
                        curstart = j
                elif gold[i, j] == 3:
                    if curstart is not None:
                        gold_set.add((i, curstart, j))
                    curstart = None

        results["{}-p".format(self.__class__.name)] += len(pred_set)
        results["{}-r".format(self.__class__.name)] += len(gold_set)
        results["{}-c".format(self.__class__.name)] += len(pred_set.intersection(gold_set))

    def metrics(self, results):
        p = results["{}-p".format(self.__class__.name)]
        r = results["{}-r".format(self.__class__.name)]
        c = results["{}-c".format(self.__class__.name)]
        precision = c / (p + 1e-6)
        recall = c / (r + 1e-6)
        results["metrics/{}-p".format(self.__class__.name)] = precision * 100.
        results["metrics/{}-r".format(self.__class__.name)] = recall * 100.
        results["metrics/{}-f1".format(self.__class__.name)] = 2. / (1./(precision + 1e-6) + 1./(recall+1e-6)) * 100.
        del results["{}-p".format(self.__class__.name)]
        del results["{}-r".format(self.__class__.name)]
        del results["{}-c".format(self.__class__.name)]


class PointerSelector(nn.Module, ParserModule):

    def __init__(self, parser, hidden_size, dropout=0.):
        super(PointerSelector, self).__init__()
        print("build pointer selector ...", self.__class__.name)
        self.head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.attention = BilinearMatrixAttention(hidden_size, hidden_size, True)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction='sum')

    def calculate_loss(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = self.batch_cand_mask(batch)

        heads = self.head_mlp(lstm_features)
        deps = self.dep_mlp(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = self.attention(deps, heads).masked_fill((1-mask_att).bool(), float("-inf")).view(batch_size * seq_len, -1)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1

        total_loss = self.loss(scores, (batch_label - 1).view(batch_size * seq_len))
        total_loss = total_loss / float(batch_size)

        return total_loss, tag_seq

    def forward(self, parser, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]

        mask_h = self.batch_cand_mask(batch)
        heads = self.head_mlp(lstm_features)
        deps = self.dep_mlp(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = self.attention(deps, heads).masked_fill((1-mask_att).bool(), float("-inf")).view(batch_size * seq_len, -1)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1

        scores = scores.view(batch_size, seq_len, -1).cpu().data.numpy().astype('float64')
        word_length = batch["word_length"].cpu().data.numpy()
        for i in range(batch_size):
            l = int(word_length[i])
            s = scores[i, :l, :l].T
            heads = parse_proj(s)
            tag_seq[i, :l] = torch.Tensor(heads + 1)

        ## filter padded position with zero
        tag_seq = mask.long() * tag_seq

        batch["pred_head"] = tag_seq

        return tag_seq

    def evaluate(self, results, pred, gold, mask):
        overlaped = (pred == gold).byte()
        # correct = np.sum(overlaped * mask)
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = correct / (total + 1e-10) * 100.
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]

    @staticmethod
    @abstractmethod
    def batch_cand_mask(batch):
        pass


class HSelParser(PointerSelector):

    name = "HSel"

    @staticmethod
    def load_data(parser, graph):
        return {"head": graph.heads + 1}

    @staticmethod
    def batch_label(batch):
        return batch["head"]

    @staticmethod
    def batch_cand_mask(batch):
        return batch["mask_h"]


class RelLabeler(nn.Module, ParserModule):

    name = "Rel"

    def __init__(self, parser, hidden_size, dropout=0.):
        super(RelLabeler, self).__init__()
        print("build rel labeler...", self.__class__.name)
        self.head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.attention = nn.Bilinear(hidden_size, hidden_size, len(parser._rels) + 1, True)
        self.bias_x = nn.Linear(hidden_size, len(parser._rels) + 1, False)
        self.bias_y = nn.Linear(hidden_size, len(parser._rels) + 1, False)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction='sum')

    def calculate_loss(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask = batch["mask"]
        head = torch.abs(batch["head"] - 1)

        ran = torch.arange(batch_size, device=head.get_device()).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        heads = self.head_mlp(lstm_features).view(batch_size * seq_len, -1)[idx]
        deps = self.dep_mlp(lstm_features).view(batch_size * seq_len, -1)

        scores = self.attention(deps, heads) + self.bias_x(heads) + self.bias_y(deps)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)

        total_loss = self.loss(scores, batch_label.view(batch_size * seq_len))
        total_loss = total_loss / float(batch_size)

        return total_loss, tag_seq


    def forward(self, parser, lstm_features, batch):
        batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask = batch["mask"]
        if "pred_head" in batch:
            head = torch.abs(batch["pred_head"] - 1)
        else:
            head = torch.abs(batch["head"] - 1)

        ran = torch.arange(batch_size, device=head.get_device()).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        heads = self.head_mlp(lstm_features).view(batch_size * seq_len, -1)[idx]
        deps = self.dep_mlp(lstm_features).view(batch_size * seq_len, -1)

        scores = self.attention(deps, heads) + self.bias_x(heads) + self.bias_y(deps)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        ## filter padded position with zero
        tag_seq = mask.long() * tag_seq

        batch["pred_rel"] = tag_seq

        return tag_seq

    def scores(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        heads = self.head_mlp(lstm_features)
        deps = self.dep_mlp(lstm_features)

        ret = torch.matmul(deps.unsqueeze(1), self.attention.weight) @ torch.transpose(heads.unsqueeze(1), -1, -2)
        ret = ret + self.bias_x(heads).transpose(-1, -2).unsqueeze(-2)
        ret = ret + self.bias_y(deps).transpose(-1, -2).unsqueeze(-1)
        ret = ret + self.attention.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return F.log_softmax(ret, dim=1).transpose(-1, -2)

    def evaluate(self, results, pred, gold, mask):
        overlaped = (pred == gold).byte()
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = correct / (total + 1e-10) * 100.
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]

    @staticmethod
    def load_data(parser, graph):
        labels = [0] + [parser._rels.get(r, 1) for r in graph.rels[1:]]
        return {"rel": labels}

    @staticmethod
    def batch_label(batch):
        return batch["rel"]


class MWEJointDecoder(nn.Module, ParserModule):

    name = "JointMWE"

    def __init__(self, hselparser, rellabeler, nerbiotagger):
        super(MWEJointDecoder, self).__init__()
        print("build MWE joint decoder ...")

        class Object(object):
            pass

        self.c = Object()

        self.c.hselparser = hselparser
        self.c.rellabeler = rellabeler
        self.c.nerbiotagger = nerbiotagger

    @staticmethod
    def load_data(parser, graph):
        return {}

    @staticmethod
    def batch_label(batch):
        return batch["head"]

    def evaluate(self, results, pred, gold, mask):
        overlaped = (pred == gold).byte()
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = correct / (total + 1e-10) * 100.
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]

    def calculate_loss(self, lstm_features, batch):
        return None, None

    def forward(self, parser, lstm_features, batch):
        mask = batch["mask"]
        batch_size = mask.size(0)
        seq_len = mask.size(1)

        mask_h = self.c.hselparser.batch_cand_mask(batch)
        heads = self.c.hselparser.head_mlp(lstm_features)
        deps = self.c.hselparser.dep_mlp(lstm_features)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = self.c.hselparser.attention(deps, heads).masked_fill((1-mask_att).bool(), float("-inf")).view(batch_size * seq_len, -1)
        scores = F.log_softmax(scores, dim=1)

        bio_scores = self.c.nerbiotagger.scores(lstm_features, batch).cpu().data.numpy().astype('float64')

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1
        _, rel_seq  = torch.max(scores, 1)
        rel_seq = rel_seq.view(batch_size, seq_len) + 1

        rel_scores = self.c.rellabeler.scores(lstm_features, batch).cpu().data.numpy().astype('float64')

        scores = scores.view(batch_size, seq_len, -1).cpu().data.numpy().astype('float64')
        word_length = batch["word_length"].cpu().data.numpy()
        for i in range(batch_size):
            l = int(word_length[i])
            s = scores[i, :l, :l].T
            t = bio_scores[i, :l]

            if "mwe_NNP" in parser._rels:
                heads, rels = parse_mwe(s, rel_scores[i, :, :l, :l], t * 1., parser._rels["mwe_NNP"])
            else:
                heads, rels = parse_mwe_ud(s, rel_scores[i, :, :l, :l], t * 1., parser._rels["flat"], parser._rels["punct"])
            tag_seq[i, :l] = torch.Tensor(heads + 1)
            rel_seq[i, :l] = torch.Tensor(rels)

        ## filter padded position with zero
        tag_seq = mask.long() * tag_seq
        rel_seq = mask.long() * rel_seq

        batch["pred_head"] = tag_seq
        batch["pred_rel"] = rel_seq

        return tag_seq
