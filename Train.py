from torchtext import data
from nlp_pre.seq2classes_data import seq2classesData
from nlp_pre.config import DEVICE
import torch.nn.functional as F

import torch
from Model import Model
import spacy
from nltk.corpus import stopwords
from nlp_pre.TrainModel import train

en = spacy.load('en')


def tokenize_en(sent):
    return [tok.text for tok in en.tokenizer(sent)]


def runModel(model, inp, target):
    """
    :param model: seq2class model
    :param inp: batch.INPUT
    :param target: batch.OUTPUT
    :return: loss, y_pre, acc
    """
    label_num = 2
    label = (torch.eye(label_num) * -2 + 1).index_select(dim=0, index=target.cpu().data)
    label = label.to(DEVICE)

    v_s, p, v_c, r_s = model(inp)
    _, y_pre = torch.max(p, -1)
    lossJ = F.relu((label * p).sum(-1) + 1).sum()
    v_s = v_s.unsqueeze(-1)
    lossU = F.relu((label * (torch.matmul(r_s, v_s).squeeze(-1))).sum(-1) + 1).sum()
    loss = lossJ + lossU
    acc = torch.mean((torch.tensor(y_pre == target, dtype=torch.float)))
    return loss, y_pre, acc


if __name__ == "__main__":

    stopwords = stopwords.words('english')
    # print("stopwords", stopwords)

    input_field = data.Field(batch_first=True, sequential=True, tokenize=tokenize_en,
                             lower=True, stop_words=stopwords)
    output_field = data.Field(batch_first=True, sequential=False, use_vocab=False)

    dataset = seq2classesData(input_field, output_field, data_path='data', vectors='glove.6B.300d')
    train_iter, val_iter, test_iter = dataset.generateIterator()

    vocab = dataset.getVocab()

    model = Model(len(vocab))

    EmneddingMatrix = dataset.getEmneddingMatrix()
    model.embedding.weight.data.copy_(EmneddingMatrix)

    print(model)
    model.to(DEVICE)

    train(train_iter, val_iter, test_iter, model, runModel, n_epoch=30, lr=0.00001)


