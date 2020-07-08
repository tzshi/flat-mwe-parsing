# -*- coding: utf-8 -*-

import fire
import sys
from .parser import CDParser
from .data import DataProcessor, InfiniteDataLoader, DataCollate
from .modules import MWEJointDecoder, MWEJointParentDecoder
from .eval import parse_f1, bio_f1, parse_f1_ud


class CDParserTest:

    def __init__(self, **kwargs):
        pass

    def test(self, **kwargs):
        model_dir = kwargs.get("model_dir", "")
        test_file = kwargs.get("test_file", "")
        output_file = kwargs.get("output_file", "")

        parser = CDParser()
        parser.load_model(model_dir)
        parser._model.eval()

        test_graphs = DataProcessor(test_file, parser, parser._model)
        gold_graphs = DataProcessor(test_file, parser, parser._model)

        result = parser.evaluate(test_graphs, output_file)

        if parser._mode in {"jointdecoding", "jointparsing", "parsing"}:
            if "mwe_NNP" in parser._rels:
                performance = parse_f1(gold_graphs.graphs, test_graphs.graphs)
            else:
                performance = parse_f1_ud(gold_graphs.graphs, test_graphs.graphs)
        else:
            performance = bio_f1(gold_graphs.graphs, test_graphs.graphs)
        print(performance)
        return self

    def finish(self, **kwargs):
        print()
        sys.stdout.flush()

if __name__ == '__main__':
    fire.Fire(CDParserTest)
