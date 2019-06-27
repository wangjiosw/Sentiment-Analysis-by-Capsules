import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class RNNCapsule(nn.Module):

    def __init__(self, input_dim=512, capsule_num=config.classes):
        super(RNNCapsule, self).__init__()
        self.W_alpha = nn.Parameter(torch.randn(input_dim, capsule_num))
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param   x:   (batch_size, n, input_dim)
        :return: p:   (batch_size, capsule_num)
                 v_c: (batch_size, capsule_num , input_dim)
                 r_s: (batch_size, capsule_num , input_dim)
        """
        e = torch.matmul(x, self.W_alpha)
        alpha = F.softmax(e, 1)
        alpha = alpha.unsqueeze(3)
        x = x.unsqueeze(2)
        v_c = (alpha*x).sum(1)
        p = F.tanh(self.linear(v_c))
        r_s = p*v_c

        return p.squeeze(-1), v_c, r_s


class Model(nn.Module):

    def __init__(self, len_vocab, capsule=RNNCapsule):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 300)
        self.lstm = nn.LSTM(300, 256, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.capsule = capsule(512, config.classes)

    def forward(self, x):
        """
        :param x: (batch_size, config.fix_length)
        :return: v_s: (batch_size, input_dim)
                 p:   (batch_size, capsule_num)
                 v_c: (batch_size, capsule_num , input_dim)
                 r_s: (batch_size, capsule_num , input_dim)
        """
        vec = self.embedding(x)
        vec = F.dropout(vec, 0.3)
        # print('vec:', vec.shape)
        out, (hn, cn) = self.lstm(vec)
        # print('out:', out.shape)

        v_s = out.mean(1)
        p, v_c, r_s = self.capsule(out)
        p = F.dropout(p, 0.5)
        v_c = F.dropout(v_c, 0.5)
        r_s = F.dropout(r_s, 0.5)

        return v_s, p, v_c, r_s

