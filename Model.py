import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, norm_type):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc_in = nn.Linear(self.in_features, self.hidden_features, bias = True)
        # self.ln_in = nn.LayerNorm(self.hidden_features)
        self.ac_in = nn.ReLU()
        self.fc_h0 = nn.Linear(self.hidden_features, self.hidden_features, bias = True)
        # self.ln_h0 = nn.LayerNorm(self.hidden_features)
        self.ac_h0 = nn.ReLU()
        self.fc_h1 = nn.Linear(self.hidden_features, self.hidden_features, bias = True)
        # self.ln_h1 = nn.LayerNorm(self.hidden_features)
        self.ac_h1 = nn.ReLU()
        self.fc_out = nn.Linear(self.hidden_features, self.out_features, bias = True)

        if norm_type == 'bn':
            self.norm_out = nn.BatchNorm1d(self.out_features)
        else:
            self.norm_out = nn.LayerNorm(self.out_features)

    def forward(self, x_in):
        x = self.fc_in(x_in)
        # x = self.ln_in(x)
        x = self.ac_in(x)

        x = self.fc_h0(x)
        # x = self.ln_h0(x)
        x = self.ac_h0(x)
        
        x = self.fc_h1(x)
        # x = self.ln_h1(x)
        x = self.ac_h1(x)

        x = self.fc_out(x)

        x = x.view(-1, x.size(-1))
        x = self.norm_out(x)
        x = x.view(x_in.size(0), x_in.size(1), x.size(-1))
        return x

class Decoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(Decoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc_in = nn.Linear(self.in_features, self.hidden_features, bias = True)
        self.ac_in = nn.ReLU()
        self.fc_h0 = nn.Linear(self.hidden_features, self.hidden_features, bias = True)
        self.ac_h0 = nn.ReLU()
        self.fc_h1 = nn.Linear(self.hidden_features, self.hidden_features, bias = True)
        self.ac_h1 = nn.ReLU()
        self.fc_out = nn.Linear(self.hidden_features, self.out_features, bias = True)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.ac_in(x)
        x = self.fc_h0(x)
        x = self.ac_h0(x)
        x = self.fc_h1(x)
        x = self.ac_h1(x)
        x = self.fc_out(x)
        return x


##NOTE: MLPs in Processor HAVE residual connection!!!    
class Processor(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, norm_type):
        super(Processor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc_in = nn.Linear(self.in_features, self.hidden_features, bias = True)
        # self.ln_in = nn.LayerNorm(self.hidden_features)
        self.ac_in = nn.ReLU()
        self.fc_h0 = nn.Linear(self.hidden_features, self.hidden_features, bias = True)
        # self.ln_h0 = nn.LayerNorm(self.hidden_features)
        self.ac_h0 = nn.ReLU()
        self.fc_h1 = nn.Linear(self.hidden_features, self.hidden_features, bias = True)
        # self.ln_h1 = nn.LayerNorm(self.hidden_features)
        self.ac_h1 = nn.ReLU()
        self.fc_out = nn.Linear(self.hidden_features, self.out_features, bias = True)
        if norm_type == 'bn':
            self.norm_out = nn.BatchNorm1d(self.out_features)
        else:
            self.norm_out = nn.LayerNorm(self.out_features)
    
    def forward(self, x_in):
        x = self.fc_in(x_in)
        # x = self.ln_in(x)
        x = self.ac_in(x)

        x = self.fc_h0(x)
        # x = self.ln_h0(x)
        x = self.ac_h0(x)

        x = self.fc_h1(x)
        # x = self.ln_h1(x)
        x = self.ac_h1(x)

        x = self.fc_out(x)
        x = x.view(-1, x.size(-1))
        x = self.norm_out(x)
        x = x.view(x_in.size(0), x_in.size(1), x.size(-1))
        return x
