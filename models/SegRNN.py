import torch
import torch.nn as nn

class Model(nn.Module):
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


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y

        # normalization and permute     b,s,c -> b,c,s
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
                # m,d//2 -> bc,m,d//2 -> bcm, 1, d//2 -> bcm, 1, d
                channel_emb = self.channel_emb.repeat(x_enc.size(0), self.seg_num_y).view(-1, 1, self.d_model//2)
                pos_emb = torch.cat([self.pos_emb.repeat(x_enc.size(0) * self.enc_in, 1).unsqueeze(1), channel_emb], dim=-1)
            else:
                # m,d -> bc,m,d -> bcm, 1, d
                pos_emb = self.pos_emb.repeat(x_enc.size(0) * self.enc_in, 1).unsqueeze(1)

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
        y = y.permute(0, 2, 1) + seq_last

        return y


# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model):
#         super(TemporalEmbedding, self).__init__()
#
#         hour_size = 24
#         weekday_size = 7
#
#
#         Embed = nn.Embedding
#         self.hour_embed = Embed(hour_size, d_model//2)
#         self.weekday_embed = Embed(weekday_size, d_model//2)
#
#         # self.embd = Embed(hour_size*weekday_size, d_model)
#
#
#     def forward(self, x):
#         x[:, :, 0] = (x[:, :, 0] + 0.5) * 23
#         x[:, :, 1] = (x[:, :, 1] + 0.5) * 6
#
#         x = x.int()
#
#         hour_x = self.hour_embed(x[:, :, 0])
#         weekday_x = self.weekday_embed(x[:, :, 1])
#
#         # time = self.embd(x[:, :, 1]+x[:, :, 2]*7)
#
#         return torch.cat([hour_x, weekday_x], dim=-1)
#         # return time
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#
#         # get parameters
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.d_model = configs.d_model
#         self.dropout = configs.dropout
#
#         self.rnn_type = configs.rnn_type
#         self.dec_way = configs.dec_way
#         self.revin = configs.revin
#         self.seg_len = configs.seg_len
#         self.timeId = True
#
#         assert self.rnn_type in ['rnn', 'gru', 'lstm']
#         assert self.dec_way in ['rmf', 'dmf', 'pmf']
#
#         self.seg_num_x = self.seq_len//self.seg_len
#
#         # build model
#         if self.revin:
#             self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=True)
#
#         self.valueEmbedding = nn.Sequential(
#             nn.Linear(self.seg_len, self.d_model),
#             nn.ReLU()
#         )
#
#         self.temporalEmbedding = TemporalEmbedding(d_model=self.d_model//2)
#         self.channelId = nn.Parameter(torch.randn(self.enc_in, self.d_model//4))
#
#         if self.rnn_type == "rnn":
#             self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
#                               batch_first=True, bidirectional=False)
#         elif self.rnn_type == "gru":
#             self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
#                               batch_first=True, bidirectional=False)
#         elif self.rnn_type == "lstm":
#             self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
#                               batch_first=True, bidirectional=False)
#
#         if self.dec_way == "rmf":
#             self.seg_num_y = self.pred_len // self.seg_len
#             self.predict = nn.Sequential(
#                 nn.Dropout(self.dropout),
#                 nn.Linear(self.d_model, self.seg_len)
#             )
#         elif self.dec_way == "dmf":
#             self.predict = nn.Sequential(
#                 nn.Dropout(self.dropout),
#                 nn.Linear(self.d_model*2, self.pred_len)
#             )
#         elif self.dec_way == "pmf":
#             self.seg_num_y = self.pred_len // self.seg_len
#
#             self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model//4))
#             self.predict = nn.Sequential(
#                 nn.Dropout(self.dropout),
#                 nn.Linear(self.d_model, self.seg_len)
#             )
#
#
#
#     def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
#
#         # if self.revin:
#         #     x = self.revinLayer(x_enc, 'norm').permute(0, 2, 1)
#         # else:
#         #     x = x_enc.permute(0, 2, 1) # b,c,s
#         seq_last = x_enc[:, -1:, :].detach()
#         x = x_enc - seq_last
#         x = x.permute(0, 2, 1) # b,c,s
#
#
#         # segment
#         x = x.unfold(dimension=-1, size=self.seg_len, step=self.seg_len).reshape(-1, self.seg_num_x, self.seg_len)
#
#         # x = torch.cat([torch.fft.fft(x, norm="ortho").real, torch.fft.fft(x, norm="ortho").imag], dim=-1)
#
#         x = self.valueEmbedding(x)
#
#         # encoding
#         if self.rnn_type == "lstm":
#             _, (hn, cn) = self.rnn(x)
#         else:
#             # _, hn = self.rnn(x) # bc,n,d  1,bc,d
#             _, hn = self.rnn(x) # bc,n,d  1,bc,d
#
#
#         # decoding
#         if self.dec_way == "rmf":
#             y = []
#             for i in range(self.seg_num_y):
#                 yy = self.predict(hn)    # 1,bc,l
#                 yy = yy.permute(1,0,2)   # bc,1,l
#                 y.append(yy)
#                 yy = self.valueEmbedding(yy)
#                 if self.rnn_type == "lstm":
#                     _, (hn, cn) = self.rnn(yy, (hn, cn))
#                 else:
#                     _, hn = self.rnn(yy, hn)
#             y = torch.stack(y, dim=1).squeeze(2).reshape(-1, self.enc_in, self.pred_len) # b,c,s
#         elif self.dec_way == "dmf":
#             y = self.predict(hn).reshape(-1, self.enc_in, self.pred_len)
#             # 1,bc,d
#             # hn = torch.cat([hn.view(-1, self.enc_in, self.d_model), self.channelId.repeat(x_enc.size(0), 1, 1)], dim=-1)
#             # y = self.predict(hn)
#
#         elif self.dec_way == "pmf":
#             if self.timeId:
#                 time = x_mark_dec[:, ::self.seg_len, :]  # b,m,4
#                 # b,m,4  -  b,m,d  -  b,c,m,d  - bcm,1,d
#                 time = self.temporalEmbedding(time).unsqueeze(1).repeat(1, self.enc_in, 1, 1).view(-1, 1, self.d_model//2)
#                 # c,d - bc, md
#                 channel = self.channelId.repeat(x_enc.size(0), self.seg_num_y).view(-1, 1, self.d_model//4)
#                 pe = torch.cat([self.pos_emb.repeat(x_enc.size(0) * self.enc_in, 1).unsqueeze(1), time, channel], dim=-1)
#             else:
#                 pe = self.pos_emb.repeat(x_enc.size(0) * self.enc_in, 1).unsqueeze(1)
#
#             # pos_emb: m,d -> bcm,d ->  bcm,1,d
#             # hn, cn: 1,bc,d -> 1,bc,md -> 1,bcm,d
#             if self.rnn_type == "lstm":
#                 _, (hy, cy) = self.rnn(self.pos_emb.repeat(x_enc.size(0) * self.enc_in, 1).unsqueeze(1),
#                                        (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
#                                         cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
#             else:
#                 _, hy = self.rnn(pe, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
#             # 1,bcm,d -> 1,bcm,w -> b,c,s
#             y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
#             # if self.timeId:
#             #     y = self.predict(torch.cat([hy, time],dim=-1)).view(-1, self.enc_in, self.pred_len)
#             # else:
#             #     y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
#
#
#         # if self.revin:
#         #     y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
#         # else:
#         #     y = y.permute(0, 2, 1)
#         y = y.permute(0, 2, 1)
#         y = y + seq_last
#
#         return y