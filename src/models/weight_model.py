from torch import nn

class Weight(nn.Module):
    def __init__(self, in_channel, hidden, out_channel, act_type='sigmoid'):
        super(Weight, self).__init__()

        self.linear1 = nn.Linear(in_channel, hidden)
        self.linear2 = nn.Linear(hidden, out_channel)
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky':
            self.act = nn.LeakyReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'softplus':
            self.act = nn.Softplus()
        else:
            raise ValueError('unknown activation type!' + act_type)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.act(x)
        return x