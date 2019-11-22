import  torch
from    torch import nn
from    torch.nn import functional as F
#from    utils import sparse_dropout, dot
import torchsnooper
import math
class GraphConvolution(nn.Module):

    def __init__(self):
        super(GraphConvolution, self).__init__()
        self.weight = torch.nn.Linear(48, 48).cuda()
#        self.adj = torch.nn.Linear(6, 6).cuda()
        mat = torch.zeros(55, 55)
        for i in range(55):
            for j in range(55):
                if i+1 == j or i-1 == j or i == j:
                    mat[i,j] = 1
        print(mat)

        self.adj = nn.Parameter(mat, requires_grad=False)
        # self.adj = nn.Parameter(torch.empty(55, 55), requires_grad=True)

        # nn.init.xavier_normal_(self.adj)
        self.dropout = 0
        self.activation = F.relu

        # self.featureless = featureless
        # self.num_features_nonzero = num_features_nonzero

        #self.weight = nn.Parameter(torch.randn(input_dim, output_dim)).cuda()
        # self.bias = None
        # if bias:
        #     self.bias = nn.Parameter(torch.zeros(output_dim))
#        self.myparameters = nn.ParameterList(self.weight)

    def forward(self, x):
        # print('inputs:', inputs)

        #if self.training and self.is_sparse_inputs:
        #    x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        #elif self.training:
        # x = F.dropout(x,0.9).cuda()
        #with torchsnooper.snoop():
        A = self.adj.cuda()
        # A = torch.mm(self.adj,self.adj.permute(1,0)).cuda()
        DA = torch.mm(torch.inverse(torch.diag(torch.sum(A,1))), A).cuda()
    #        dt = torch.pow(torch.inverse(D),0.5)
    #        A = torch.mm(dt,A)
    #        A = torch.mm(A,dt)
    #        A = torch.mm(torch.inverse(D), A)

        A = DA.repeat(x.shape[0],1,1).cuda()
        AX = torch.bmm(A,x)
    #        AX = self.adj(x.permute(0,2,1)).permute(0,2,1)
        AXW = self.activation(self.weight(AX))
        # x.mul(torch.eye(249,249))
                #xw = torch.bmm(x, self.weight.repeat(x.shape[0],1,1))

        return AXW#self.activation(x)

