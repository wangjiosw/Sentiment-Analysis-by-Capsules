import torchtext
from nlp_pre.seq2seq_data import seq2seqData
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext import data, datasets
from torch.nn import init
from nlp_pre import config
import dill
import os
import torch

# self.INPUT_TEXT = data.Field(batch_first=True, sequential=True, tokenize=in_tokenize, lower=True)
# self.OUTPUT_TEXT = data.Field(batch_first=True, sequential=False, use_vocab=False)


class seq2classesData(seq2seqData):
    def __init__(self, input_field, output_field, batch_size=config.BATCH_SIZE, device=config.DEVICE, data_path='.', vectors=None):
        super(seq2classesData, self).__init__(input_field, output_field, batch_size,
                                              device, data_path, in_vectors=vectors, out_vectors=None)

    def buildVocabulary(self, train, val):
        # build vocab
        if os.path.exists(self.input_vocab_path):
            print('using exist vocabulary')
            with open(self.input_vocab_path, 'rb')as f:
                self.INPUT_FIELD.vocab = dill.load(f)
        else:
            # test dataset may exist word out of vocabulary if build_vocab operation no include test
            # but more true
            print('create vocabulary')
            self.INPUT_FIELD.build_vocab(train, val, vectors=self.in_vectors)
            if not (self.in_vectors is None):
                self.INPUT_FIELD.vocab.vectors.unk_init = init.xavier_uniform

            with open(self.input_vocab_path, 'wb')as f:
                dill.dump(self.INPUT_FIELD.vocab, f)

        # return self.INPUT_TEXT.vocab, self.OUTPUT_TEXT.vocab
        return

    def getVocab(self):
        return self.INPUT_FIELD.vocab

    def getEmneddingMatrix(self):
        return self.INPUT_FIELD.vocab.vectors
    

    def index2word(self, index):
        return self.INPUT_FIELD.vocab.itos[index]

    def word2index(self, word):
        return self.INPUT_FIELD.vocab.stoi[word]
    
    def word2embeding(self, word):
        return self.INPUT_FIELD.vocab.vectors[self.INPUT_FIELD.vocab.stoi[word]]

