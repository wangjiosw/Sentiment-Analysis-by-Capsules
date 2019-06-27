import os
from torchtext import data
from nlp_pre.seq2classes_data import seq2classesData
from nlp_pre.config import DEVICE
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
from Model import Model
import spacy
import config
from nltk.corpus import stopwords

en = spacy.load('en')


def tokenize_en(sent):
    return [tok.text for tok in en.tokenizer(sent)]


if __name__ == "__main__":

    stopwords = stopwords.words('english')
    print("stopwords", stopwords)

    input_field = data.Field(batch_first=True, sequential=True, tokenize=tokenize_en,
                             lower=True, stop_words=stopwords)
    output_field = data.Field(batch_first=True, sequential=False, use_vocab=False)

    dataset = seq2classesData(input_field, output_field, data_path='data', vectors='glove.6B.300d')
    train, val, test = dataset.createDataset()

    for index in range(10):
        print(train[index].INPUT, train[index].OUTPUT)

    train_iter, val_iter, test_iter = dataset.generateIterator()

    vocab = dataset.getVocab()

    model = Model(len(vocab))
    EmneddingMatrix = dataset.getEmneddingMatrix()
    print(EmneddingMatrix.shape, EmneddingMatrix)
    model.embedding.weight.data.copy_(EmneddingMatrix)
    print(model)

    if os.path.exists('params.pkl'):
        model.load_state_dict(torch.load('params.pkl'))

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epoch = 128

    best_val_acc = 0

    for epoch in range(n_epoch):

        for batch_idx, batch in enumerate(train_iter):
            train_x = batch.INPUT
            target = batch.OUTPUT
            target = (torch.sparse.torch.eye(config.classes)*-2+1).index_select(dim=0, index=target.cpu().data)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            v_s, p, v_c, r_s = model(train_x)

            lossJ = F.relu((target*p).sum(-1)+1).sum()

            v_s = v_s.unsqueeze(-1)
            lossU = F.relu((target*(torch.matmul(r_s, v_s).squeeze(-1))).sum(-1)+1).sum()
            loss = lossJ + lossU

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 20 == 0:
                _, y_pre = torch.max(p, -1)
                acc = torch.mean((torch.tensor(y_pre == batch.OUTPUT, dtype=torch.float)))
                print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f'
                      % (epoch, batch_idx, loss, acc))

        val_accs = []
        for batch_idx, batch in enumerate(val_iter):
            val_x = batch.INPUT
            target = batch.OUTPUT
            target = (torch.sparse.torch.eye(config.classes)*-2+1).index_select(dim=0, index=target.cpu().data)
            target = target.to(DEVICE)
            # data = data.permute(1, 0)
            v_s, p, v_c, r_s = model(val_x)

            _, y_pre = torch.max(p, -1)
            acc = torch.mean((torch.tensor(y_pre == batch.OUTPUT, dtype=torch.float)))
            val_accs.append(acc)

        acc = np.array(val_accs).mean()
        if acc > best_val_acc:
            print('val acc : %.4f > %.4f saving model' % (acc, best_val_acc))
            torch.save(model.state_dict(), 'params.pkl')
            best_val_acc = acc
        print('val acc: %.4f'%(acc))