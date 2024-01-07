import torch
import torch.nn as nn
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    SegRNN is an innovative RNN-based model designed for Long-term Time Series Forecasting (LTSF). It incorporates two fundamental
strategies:
        (i) the replacement of point-wise iterations with segment-wise iterations
        (2)  the substitution of Recurrent Multi-step Forecasting (RMF) with Parallel Multi-step Forecasting (PMF)
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.rnn_type = configs.rnn_type
        self.dec_way = configs.dec_way
        self.seg_len = configs.seg_len
        self.channel_id = configs.channel_id
        self.revin = configs.revin

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way in ['rmf', 'pmf']

        self.seg_num_x = self.seq_len//self.seg_len

        # build model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        if self.dec_way == "rmf":
            self.seg_num_y = self.pred_len // self.seg_len
            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
        elif self.dec_way == "pmf":
            self.seg_num_y = self.pred_len // self.seg_len

            if self.channel_id:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
                self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
            else:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x_enc.size(0)

        # normalization and permute     b,s,c -> b,c,s
        if self.revin:
            x = self.revinLayer(x_enc, 'norm').permute(0, 2, 1)
        else:
            seq_last = x_enc[:, -1:, :].detach()
            x = (x_enc - seq_last).permute(0, 2, 1) # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = x.unfold(dimension=-1, size=self.seg_len, step=self.seg_len).reshape(-1, self.seg_num_x, self.seg_len)
        x = self.valueEmbedding(x)

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x) # bc,n,d  1,bc,d

        # decoding
        if self.dec_way == "rmf":
            y = []
            for i in range(self.seg_num_y):
                yy = self.predict(hn)    # 1,bc,l
                yy = yy.permute(1,0,2)   # bc,1,l
                y.append(yy)
                yy = self.valueEmbedding(yy)
                if self.rnn_type == "lstm":
                    _, (hn, cn) = self.rnn(yy, (hn, cn))
                else:
                    _, hn = self.rnn(yy, hn)
            y = torch.stack(y, dim=1).squeeze(2).reshape(-1, self.enc_in, self.pred_len) # b,c,s
        elif self.dec_way == "pmf":
            if self.channel_id:
                # c,d//2 -> bc, md//2 -> bcm, 1, d//2
                # m,d//2 -> bcm,d//2 -> bcm, 1, d//2 -> bcm, 1, d
                # channel_emb = self.channel_emb.repeat(batch_size, self.seg_num_y).view(-1, 1, self.d_model//2)
                # pos_emb = torch.cat([self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1), channel_emb], dim=-1)
                pos_emb = torch.cat([
                    self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                    self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
                ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
            else:
                # m,d -> bc,m,d -> bcm, 1, d
                pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)
                # pos_emb = self.pos_emb.repeat(self.enc_in, 1).unsqueeze(1)

            # pos_emb: m,d -> bcm,d ->  bcm,1,d
            # hn, cn: 1,bc,d -> 1,bc,md -> 1,bcm,d
            if self.rnn_type == "lstm":
                _, (hy, cy) = self.rnn(pos_emb,
                                       (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                        cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
            else:
                _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
            # 1,bcm,d -> 1,bcm,w -> b,c,s
            y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y