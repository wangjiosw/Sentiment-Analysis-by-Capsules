import torch
import torch.nn as nn
import torch.nn.functional as F
import config

qkv_dim = 16


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.WQ = nn.Parameter((torch.randn(config.cap_dim, qkv_dim)))
        self.WK = nn.Parameter((torch.randn(config.cap_dim, qkv_dim)))
        self.WV = nn.Parameter((torch.randn(config.cap_dim, qkv_dim)))

    def forward(self, x):
        """
        :param x: (batch_size, n , 512)
        :return: (batch_size, n , 512)
        """
        # (batch_size, n, 64)
        Q = torch.matmul(x, self.WQ)
        K = torch.matmul(x, self.WK)
        V = torch.matmul(x, self.WV)
        # (batch_size, n, 64)
        Z = torch.matmul(F.softmax(torch.matmul(Q, K.permute(0, 2, 1))/8.0, -1), V)
        return Z


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self):
        super(MultiHeadedSelfAttention, self).__init__()
        self.WO = nn.Parameter((torch.randn(qkv_dim*8, config.cap_dim)))
        self.attention1 = SelfAttention()
        self.attention2 = SelfAttention()
        self.attention3 = SelfAttention()
        self.attention4 = SelfAttention()
        self.attention5 = SelfAttention()
        self.attention6 = SelfAttention()
        self.attention7 = SelfAttention()
        self.attention8 = SelfAttention()

    def forward(self, x):
        out1 = self.attention1(x)
        out2 = self.attention2(x)
        out3 = self.attention3(x)
        out4 = self.attention4(x)
        out5 = self.attention5(x)
        out6 = self.attention6(x)
        out7 = self.attention7(x)
        out8 = self.attention8(x)

        # (batch_size, n, 64*8)
        out = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8], -1)
        out = torch.matmul(out, self.WO)

        return out

