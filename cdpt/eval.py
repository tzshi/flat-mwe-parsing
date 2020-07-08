# -*- coding: utf-8 -*-

def extract_spans_bio(g, only_multi=False):
    spans = set()

    start = None
    for i in range(1, len(g.nodes)):
        if g.nodes[i].feats == "B":
            if start is not None:
                if not only_multi or (i - 1 > start):
                    spans.add((start, i - 1))
            start = i
        elif g.nodes[i].feats == "I":
            if start is None:
                start = i
        else:
            if start is not None:
                if not only_multi or (i - 1 > start):
                    spans.add((start, i - 1))
            start = None
    if start is not None:
        if not only_multi or (len(g.nodes) - 1 > start):
            spans.add((start, len(g.nodes) - 1))

    return spans

def extract_spans_parsetree(g):
    spans = {}

    for i in range(1, len(g.nodes)):
        if g.rels[i] == "mwe_NNP":
            spans[g.heads[i]] = i

    spans = {(k, spans[k]) for k in spans}
    return spans

def extract_spans_parsetree_ud(g):
    spans = {}

    for i in range(1, len(g.nodes)):
        if g.rels[i] == "flat":
            spans[g.heads[i]] = i

    spans = {(k, spans[k]) for k in spans}
    return spans

def parse_f1(gold, pred):
    recall = 0
    precision = 0
    correct = 0

    mismatch = 0
    for d, g in zip(gold, pred):
        gold_spans = extract_spans_bio(d, only_multi=True)
        pred_spans = extract_spans_parsetree(g)
        precision += len(pred_spans)
        recall += len(gold_spans)
        correct += len(pred_spans.intersection(gold_spans))

    if correct == 0 or precision == 0 or recall == 0:
        return 0.

    precision = correct / precision
    recall = correct / recall
    f1 = 2. / (1./precision + 1./recall)

    return f1 * 100.

def parse_f1_ud(gold, pred):
    recall = 0
    precision = 0
    correct = 0

    mismatch = 0
    for d, g in zip(gold, pred):
        gold_spans = extract_spans_bio(d, only_multi=True)
        pred_spans = extract_spans_parsetree_ud(g)
        precision += len(pred_spans)
        recall += len(gold_spans)
        correct += len(pred_spans.intersection(gold_spans))

    if correct == 0 or precision == 0 or recall == 0:
        return 0.

    precision = correct / precision
    recall = correct / recall
    f1 = 2. / (1./precision + 1./recall)

    return f1 * 100.

def bio_f1(gold, pred):
    recall = 0
    precision = 0
    correct = 0

    mismatch = 0
    for d, g in zip(gold, pred):
        gold_spans = extract_spans_bio(d, only_multi=True)
        pred_spans = extract_spans_bio(g, only_multi=True)
        precision += len(pred_spans)
        recall += len(gold_spans)
        correct += len(pred_spans.intersection(gold_spans))

    if correct == 0 or precision == 0 or recall == 0:
        return 0.

    precision = correct / precision
    recall = correct / recall
    f1 = 2. / (1./precision + 1./recall)

    return f1 * 100.
