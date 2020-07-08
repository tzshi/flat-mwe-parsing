# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        info = "p={}".format(self.p)
        if self.batch_first:
            info += ", batch_first={}".format(self.batch_first)

        return info

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask


class WordDropout(nn.Module):

    def __init__(self, p=0.):
        super(WordDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        info = "p={}".format(self.p)
        return info

    def forward(self, x):
        if self.training:
            mask = self.get_mask(x, self.p)
            x *= mask.unsqueeze(-1)
        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape[:-1], 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask
