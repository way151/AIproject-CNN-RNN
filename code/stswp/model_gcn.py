import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution

#from    config import args

class GCN(nn.Module):


    def __init__(self):
        super(GCN, self).__init__()

        # self.input_dim = input_dim # 1433
        # self.output_dim = output_dim

        # self.weight = torch.nn.Linear(249, 249).cuda()
        # #self.adj = torch.nn.Linear(249,249).cuda()
        # self.dropout = 0
        # self.activation = F.relu
        # self.adj = nn.Parameter(torch.randn(6, 6)).cuda()



        self.layers = nn.Sequential(GraphConvolution(),
                                    #GraphConvolution(),
                                    GraphConvolution(),

                                    )
    def forward(self, x):

        x = x.permute(0,2,1)
        x = self.layers(x)
        x = x.mean(1)

        # AXW = AXW.mean(1)

        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss
