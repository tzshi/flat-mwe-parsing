# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
from .dropout import SharedDropout, WordDropout
from .utils import BERT_TOKEN_MAPPING, from_numpy
from .transformer import TransformerEncoder


class CharCNN(nn.Module):

    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout):
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2,1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2,1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2,1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class CharBiLSTM(nn.Module):

    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout):
        super(CharBiLSTM, self).__init__()
        print("build char sequence feature extractor: LSTM ...")
        self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        return char_hidden[0].transpose(1,0).contiguous().view(batch_size,-1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_embeddings(input)
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1,0)

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class WordRep(nn.Module):

    def __init__(self, parser):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.use_char = (parser._cdims > 0)
        self.char_hidden_dim = 0
        if self.use_char:
            self.char_hidden_dim = parser._char_hidden
            self.char_embedding_dim = parser._cdims
            self.char_feature = CharBiLSTM(len(parser._charset) + 2, self.char_embedding_dim, self.char_hidden_dim, parser._char_dropout)

        self.wdims = parser._wdims

        if self.wdims > 0:
            self.word_embedding = nn.Embedding(len(parser._vocab) + 2, self.wdims)
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(len(parser._vocab) + 2, self.wdims)))


        self.pdims = parser._pdims

        if self.pdims > 0:
            self.pos_embedding = nn.Embedding(len(parser._xupos) + 2, self.pdims)
            self.pos_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(len(parser._xupos) + 2, self.pdims)))

        self.drop = WordDropout(parser._word_dropout)


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, xupos, emb=None):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)

        word_list = []

        if self.wdims > 0:
            word_list.append(self.word_embedding(word_inputs))

        if self.pdims > 0:
            word_list.append(self.pos_embedding(xupos))

        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            word_list.append(char_features)

        if emb is not None:
            word_list.append(emb)
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent


class WordSequence(nn.Module):

    def __init__(self, parser):
        super(WordSequence, self).__init__()
        print("build feature extractor...")
        print("use_char: ", parser._cdims > 0)
        print("use_pos: ", parser._pdims > 0)
        print("use_bert: ", parser._bert)

        self.use_bert = parser._bert
        self.use_transformer = parser._transformer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)

        if self.use_bert:
            self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
            self.bert_project = nn.Linear(self.bert_model.pooler.dense.in_features, parser._bilstm_dims, bias=False)
        else:
            self.use_char = (parser._cdims > 0)
            self.droplstm = SharedDropout(parser._bilstm_dropout)
            self.lstm_layer = parser._bilstm_layers
            self.wordrep = WordRep(parser)
            self.input_size = parser._wdims + parser._edims + parser._pdims
            if self.use_char:
                self.input_size += parser._char_hidden
            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            lstm_hidden = parser._bilstm_dims // 2

            if self.use_transformer:
                self.lstm = TransformerEncoder(self.input_size, parser._trans_pos_dim, parser._trans_emb_dropout,
                                               parser._trans_num_layers, parser._bilstm_dims, parser._trans_num_heads,
                                               parser._trans_attn_dropout, parser._trans_actn_dropout,
                                               parser._trans_ffn_dim, parser._trans_res_dropout)
            else:
                self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=True, dropout=parser._bilstm_dropout)

    def forward(self, batch):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_inputs = batch["word"]
        word_seq_lengths = batch["word_length"]
        char_inputs = batch["char"]
        char_seq_lengths = batch["char_length"]
        char_seq_recover = batch["char_recover"]
        xupos_inputs = batch["xupos"]
        mask = batch["mask"].transpose(1,0)
        emb = batch.get("emb", None)

        if self.use_bert:
            raw = batch["raw"]

            seq_max_len = len(raw[0])

            all_input_ids = np.zeros((len(raw), 2048), dtype=int)
            all_input_mask = np.zeros((len(raw), 2048), dtype=int)
            all_word_end_mask = np.zeros((len(raw), 2048), dtype=int)

            subword_max_len = 0

            for snum, sentence in enumerate(raw):
                tokens = []
                word_end_mask = []

                tokens.append("[CLS]")
                word_end_mask.append(1)

                cleaned_words = []
                for word in sentence[1:]:
                    word = BERT_TOKEN_MAPPING.get(word, word)
                    if word == "n't" and cleaned_words:
                        cleaned_words[-1] = cleaned_words[-1] + "n"
                        word = "'t"
                    cleaned_words.append(word)

                for word in cleaned_words:
                    word_tokens = self.bert_tokenizer.tokenize(word)
                    if len(word_tokens) == 0:
                        word_tokens = ['.']
                    for _ in range(len(word_tokens)):
                        word_end_mask.append(0)
                    word_end_mask[-1] = 1
                    tokens.extend(word_tokens)

                tokens.append("[SEP]")

                # pad to sequence length for every sentence
                for i in range(seq_max_len - len(sentence)):
                    word_end_mask.append(1)

                input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                subword_max_len = max(subword_max_len, len(word_end_mask) + 1)

                all_input_ids[snum, :len(input_ids)] = input_ids
                all_input_mask[snum, :len(input_mask)] = input_mask
                all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

            all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
            all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
            all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))
            all_encoder_layers, _ = self.bert_model(all_input_ids, attention_mask=all_input_mask)
            del _

            features = all_encoder_layers

            features_packed = features.masked_select(all_word_end_mask.to(torch.bool).unsqueeze(-1)).reshape(len(raw), seq_max_len, features.shape[-1])

            outputs = self.bert_project(features_packed)
        elif self.use_transformer:
            word_represent = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, xupos_inputs, emb=emb)
            outputs = self.lstm(word_represent, (1 - batch["mask_h"]))
        else:
            word_represent = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, xupos_inputs, emb=emb)
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            feature_out = self.droplstm(lstm_out)
            outputs = feature_out

        return outputs

    @staticmethod
    def load_data(parser, graph):
        raw = [n.word for n in graph.nodes[:]]
        norm = [n.norm for n in graph.nodes[:]]
        word = [parser._vocab.get(n.norm, 1) for n in graph.nodes[:]]
        xupos = [parser._xupos.get(n.xpos + "|" + n.upos, 1) for n in graph.nodes[:]]
        char = [[parser._charset.get(ch, 1) for ch in n.word] for n in graph.nodes[:]]

        return {"word": word, "char": char, "norm": norm, "xupos": xupos, 'raw': raw}
