import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext import data, datasets
from torch.nn import init
import dill
import os
from nlp_pre import config

# self.INPUT_TEXT = Field(batch_first=True, tokenize=in_tokenize, lower=True)
# self.OUTPUT_TEXT = Field(batch_first=True, tokenize=out_tokenize,
#                          init_token="<sos>", eos_token="<eos>", lower=True)


class seq2seqData(object):
    """
    :param
        input_field:    input text's field
        output_field:   output text's field
        cols:           ["input_col_name","output_col_name"]
        batch_size:     iterator's batch size
        device:         iterator's device torch.device('cpu') or torch.device('cuda')
        data_path:      tran.csv, val.csv and test.csv（format: input_text, output_text)'s dir path, default current dir
        in_vectors:     pre-trained word embeddings
        out_vectors:    pre-trained word embeddings

        vectors: one of or a list containing instantiations of the
            GloVe, CharNGram, or Vectors classes. Alternatively, one
            of or a list of available pretrained vectors:
            charngram.100d
            fasttext.en.300d
            fasttext.simple.300d
            glove.42B.300d
            glove.840B.300d
            glove.twitter.27B.25d
            glove.twitter.27B.50d
            glove.twitter.27B.100d
            glove.twitter.27B.200d
            glove.6B.50d
            glove.6B.100d
            glove.6B.200d
            glove.6B.300d
        Remaining keyword arguments: Passed to the constructor of Vectors classes.
    """
    def __init__(self, input_field, output_field, batch_size=config.BATCH_SIZE, device=config.DEVICE, data_path='.',
                 in_vectors=None, out_vectors=None):
        self.DEVICE = device
        self.BATCH_SIZE = batch_size
        self.date_path = data_path
        self.in_vectors = in_vectors
        self.out_vectors = out_vectors
        self.cols = ['INPUT','OUTPUT']
        # define Field
        self.INPUT_FIELD = input_field
        self.OUTPUT_FIELD = output_field

        self.train_example_path = 'data/train_examples'
        self.val_example_path = 'data/val_examples'
        self.test_example_path = 'data/test_examples'

        self.input_vocab_path = 'data/INPUT_TEXT_VOCAB'
        self.output_vocab_path = 'data/OUT_TEXT_VOCAB'

    def createDataset(self):
        """
        associate the text in the col[0] column with the INPUT_TEXT field,
        and col[1] with OUTPUT_TEXT
        :return: train, val
        """
        if not os.path.exists('data'):
            os.mkdir('data')
        
        data_fields = [(self.cols[0], self.INPUT_FIELD), (self.cols[1], self.OUTPUT_FIELD)]

        if os.path.exists(self.train_example_path) and os.path.exists(self.val_example_path) \
                and os.path.exists(self.test_example_path):
            print('using exist examples')

            with open(self.train_example_path, 'rb')as f:
                train_examples = dill.load(f)
            train = data.Dataset(examples=train_examples, fields=data_fields)

            with open(self.val_example_path, 'rb')as f:
                val_examples = dill.load(f)
            val = data.Dataset(examples=val_examples, fields=data_fields)

            with open(self.test_example_path, 'rb')as f:
                test_examples = dill.load(f)
            test = data.Dataset(examples=test_examples, fields=data_fields)
        else:
            print('create datasets')
            train, val, test = data.TabularDataset.splits(path=self.date_path, train='train.csv', validation='val.csv',
                                                          test='test.csv', skip_header=True, format='csv',
                                                          fields=data_fields)

            with open(self.train_example_path, 'wb')as f:
                dill.dump(train.examples, f)

            with open(self.val_example_path, 'wb')as f:
                dill.dump(val.examples, f)

            with open(self.test_example_path, 'wb')as f:
                dill.dump(test.examples, f)
        return train, val, test

    def buildVocabulary(self, train, val):
        # build vocab
        if os.path.exists(self.input_vocab_path) and os.path.exists(self.output_vocab_path):
            print('using exist vocabulary')
            with open(self.input_vocab_path, 'rb')as f:
                self.INPUT_FIELD.vocab = dill.load(f)
            with open(self.output_vocab_path, 'rb')as f:
                self.OUTPUT_FIELD.vocab = dill.load(f)
        else:
            # test dataset may exist word out of vocabulary if build_vocab operation no include test
            # but more true
            print('create vocabulary')
            self.INPUT_FIELD.build_vocab(train, val, vectors=self.in_vectors)
            self.OUTPUT_FIELD.build_vocab(train, val, vectors=self.out_vectors)
            if not (self.in_vectors is None):
                self.INPUT_FIELD.vocab.vectors.unk_init = init.xavier_uniform
            if not (self.out_vectors is None):
                self.OUTPUT_FIELD.vocab.vectors.unk_init = init.xavier_uniform

            with open(self.input_vocab_path, 'wb')as f:
                dill.dump(self.INPUT_FIELD.vocab, f)

            with open(self.output_vocab_path, 'wb')as f:
                dill.dump(self.OUTPUT_FIELD.vocab, f)

        # return self.INPUT_TEXT.vocab, self.OUTPUT_TEXT.vocab
        return
    
    def getVocab(self):
        return self.INPUT_FIELD.vocab, self.OUTPUT_FIELD.vocab

    def generateIterator(self):
        # generate iterator
        train, val, test = self.createDataset()
        self.buildVocabulary(train, val)

        train_iter = data.BucketIterator(train, batch_size=self.BATCH_SIZE,
                                         sort_key=lambda x: len(x.INPUT),
                                         shuffle=True, device=self.DEVICE)

        val_iter = data.BucketIterator(val, batch_size=self.BATCH_SIZE,
                                       sort_key=lambda x: len(x.INPUT),
                                       shuffle=True, device=self.DEVICE)

        test_iter = data.BucketIterator(test, batch_size=self.BATCH_SIZE,
                                        sort_key=lambda x: len(x.INPUT),
                                        shuffle=True, device=self.DEVICE)

        return train_iter, val_iter, test_iter

    def out_index2word(self, index):
        return self.OUTPUT_FIELD.vocab.itos[index]

    def out_word2index(self, word):
        return self.OUTPUT_FIELD.vocab.stoi[word]

    def getEmneddingMatrix(self):
        return self.INPUT_FIELD.vocab.vectors, self.OUTPUT_FIELD.vocab.vectors