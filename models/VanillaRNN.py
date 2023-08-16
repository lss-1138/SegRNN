import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """
     VanillaRNN is the most direct and traditional method for time series prediction using RNN-class methods.
     It completes multi-variable long time series prediction through multi-variable point-wise input and cyclic prediction.
     """
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.rnn_type = configs.rnn_type

        # build model
        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x = x_enc # b,s,c

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x) # b,s,d  1,b,d

        # decoding
        y = []
        if self.rnn_type == "lstm":
            for i in range(self.pred_len):
                yy = self.predict(hn)  # 1,b,c
                yy = yy.permute(1, 0, 2)  # b,1,c
                y.append(yy)
                _, (hn, cn) = self.rnn(yy, (hn, cn))
        else:
            for i in range(self.pred_len):
                yy = self.predict(hn)    # 1,b,c
                yy = yy.permute(1,0,2) # b,1,c
                y.append(yy)
                _, hn = self.rnn(yy, hn)
        y = torch.stack(y, dim=1).squeeze(2) # bc,s,1

        return y

