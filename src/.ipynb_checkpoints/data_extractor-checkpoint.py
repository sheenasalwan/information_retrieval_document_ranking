import copy
import os
import re

import pandas as pd
import torch

# https://stackoverflow.com/questions/1271320/resetting-generator-object-in-python


class GeneratorRestartHandler(object):
    def __init__(self, gen_func, argv, kwargv):
        self.gen_func = gen_func
        self.argv = copy.copy(argv)
        self.kwargv = copy.copy(kwargv)
        self.local_copy = iter(self)

    def __iter__(self):
        return self.gen_func(*self.argv, **self.kwargv)

    def __next__(self):
        return next(self.local_copy)


class DataExtractor:
    def __init__(self, queries_path, passages_path):
        self.passages_path = passages_path
        self.queries_path = queries_path

        # Passages key value pair
        self.passages = pd.read_csv(self.passages_path, sep="\t", header=None)
        self.passages_dict = {k: v for k, v in zip(self.passages[0], self.passages[1])}

        # Queries Key value pair
        self.queries = pd.read_csv(self.queries_path, sep="\t", header=None)
        self.queries_dict = {k: v for k, v in zip(self.queries[0], self.queries[1])}

    def restartable(g_func: callable) -> callable:
        def tmp(*argv, **kwargv):
            return GeneratorRestartHandler(g_func, argv, kwargv)

        return tmp

    @restartable
    def tsvGenerator(self, file_name, return_text=False):
        "Read the queries and passages tsv, return a generator with dict - {query: (pos doc, neg doc)}"
        # Read the file
        df = pd.read_csv(file_name, sep="\t", header=None)

        # Generate tuples with query, positive document, negative document
        for query, pos_doc, neg_doc in zip(df[0], df[1], df[2]):
            if not return_text:
                yield ({"query": query, "pos_doc": pos_doc, "neg_doc": neg_doc})
            else:
                yield (
                    {
                        "query": self.queries_dict[query],
                        "pos_doc": self.passages_dict[pos_doc],
                        "neg_doc": self.passages_dict[neg_doc],
                    }
                )


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, queries_path, passages_path):
        "Initilization"
        data_extractor = DataExtractor(
            queries_path=queries_path, passages_path=passages_path
        )
        queries, positive_doc, negative_doc = [], [], []

        pattern = re.compile(r"\.\.\.\.\.*")
        for sample in data_extractor.tsvGenerator(file_name, return_text=True):
            queries.append(pattern.sub(" ", sample["query"]))
            positive_doc.append(pattern.sub(" ", sample["pos_doc"]))
            negative_doc.append(pattern.sub(" ", sample["neg_doc"]))
        self.queries = queries
        self.positive_doc = positive_doc
        self.negative_doc = negative_doc

        # Test for documents check
        assert (
            len(self.queries) == len(self.positive_doc) == len(self.negative_doc)
        ), "Number of samples do not match, lengths are different"

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.queries)

    def __getitem__(self, index):
        "Returns a tuple with queries, positive doc and negative doc"
        return self.queries[index], self.positive_doc[index], self.negative_doc[index]
