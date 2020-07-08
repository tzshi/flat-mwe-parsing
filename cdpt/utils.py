# -*- coding: utf-8 -*-

import re
import random
from collections import Counter
import torch


BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
}


def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def buildVocab(graphs, cutoff=1):
    wordsCount = Counter()
    charsCount = Counter()
    relsCount = Counter()
    uposCount = Counter()
    xposCount = Counter()
    xuposCount = Counter()

    for graph in graphs:
        wordsCount.update([node.norm for node in graph.nodes])
        for node in graph.nodes[1:]:
            charsCount.update(list(node.word))
        uposCount.update([node.upos for node in graph.nodes])
        xuposCount.update([node.xpos + "|" + node.upos for node in graph.nodes])
        relsCount.update([r for r in graph.rels[1:]])

    print("Number of tokens in training corpora: {}".format(sum(wordsCount.values())))
    print("Vocab containing {} types before cutting off".format(len(wordsCount)))
    wordsCount = Counter({w: i for w, i in wordsCount.items() if i >= cutoff})

    print("Vocab containing {} types, covering {} words".format(len(wordsCount), sum(wordsCount.values())))
    print("Charset containing {} chars".format(len(charsCount)))
    print("UPOS containing {} tags".format(len(uposCount)), uposCount)
    print("XPOS containing {} tags".format(len(xposCount)), xposCount)
    print("Rel set containing {} tags".format(len(relsCount)), relsCount)

    ret = {
        "vocab": list(wordsCount.keys()),
        "wordfreq": wordsCount,
        "charset": list(charsCount.keys()),
        "charfreq": charsCount,
        "upos": list(uposCount.keys()),
        "xpos": list(xposCount.keys()),
        "xupos": list(xuposCount.keys()),
        "rels": list(relsCount.keys()),
    }

    return ret


def shuffled_stream(data, batch_size):
    len_data = len(data)
    ret = []
    while True:
        for d in random.sample(data, len_data):
            ret.append(d)
            if len(ret) >= batch_size:
                yield ret
                ret = []

def add_log(writer, prefix, results, batch_i):
    res = {}
    for r in results:
        k = prefix + "/" + r
        writer.add_scalar(k, results[r], batch_i)


if torch.cuda.is_available():
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda()
else:
    from torch import from_numpy
