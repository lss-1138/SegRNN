import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.hidR = 100;
        self.hidC = 100;
        self.hidS = 5;
        self.Ck = 6;
        self.skip = 24;
        self.pt = (self.seq_len - self.Ck) // self.skip
        self.hw = 24
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.enc_in));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=0.2);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.enc_in);
        else:
            self.linear1 = nn.Linear(self.hidR, self.enc_in);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        batch_size = x_enc.size(0);

        # x b,s,c
        # CNN
        c = x_enc.view(-1, 1, self.seq_len, self.enc_in);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);

        # RNN
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r, 0));

        # skip-rnn
        s = c[:, :, int(-self.pt * self.skip):].contiguous();
        s = s.view(batch_size, self.hidC, self.pt, self.skip);
        s = s.permute(2, 0, 3, 1).contiguous();
        s = s.view(self.pt, batch_size * self.skip, self.hidC);
        _, s = self.GRUskip(s);
        s = s.view(batch_size, self.skip * self.hidS);
        s = self.dropout(s);
        r = torch.cat((r, s), 1);

        res = self.linear1(r);

        # highway
        z = x_enc[:, -self.hw:, :];
        z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
        z = self.highway(z);
        z = z.view(-1, self.enc_in);
        res = res + z;

        return res;
