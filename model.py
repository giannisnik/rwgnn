import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

class RW_NN(nn.Module):
    def __init__(self, input_dim, max_step, hidden_graphs, size_hidden_graphs, hidden_dim, penultimate_dim, normalize, n_classes, dropout, device):
        super(RW_NN, self).__init__()
        self.max_step = max_step
        self.hidden_graphs = hidden_graphs
        self.size_hidden_graphs = size_hidden_graphs
        self.normalize = normalize
        self.device = device
        self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, (size_hidden_graphs*(size_hidden_graphs-1))//2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, size_hidden_graphs, hidden_dim))
        self.fc = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(hidden_graphs*max_step)
        self.fc1 = torch.nn.Linear(hidden_graphs*max_step, penultimate_dim)
        self.fc2 = torch.nn.Linear(penultimate_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
        
    def forward(self, features, adj, n_graphs):
        adj_hidden_norm = torch.zeros(self.hidden_graphs, self.size_hidden_graphs, self.size_hidden_graphs).to(self.device)
        idx = torch.triu_indices(self.size_hidden_graphs, self.size_hidden_graphs, 1)
        adj_hidden_norm[:,idx[0],idx[1]] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2) 
        x = self.sigmoid(self.fc(features))
        z = self.features_hidden
        t = x.view(n_graphs[0], -1, x.size(1))
        zx = torch.einsum("abc,dec->abde", (z, t))

        norm = torch.spmm(adj, torch.ones(adj.size(1), 1).to(self.device)).squeeze()
        norm = norm.view(n_graphs[0], -1)
        adj_0 = torch.zeros(norm.size()).to(self.device)
        adj_0[norm>0] = 1
        adj_0 = adj_0.unsqueeze(2)
        adj_0 = adj_0.repeat(1, 1, t.size(2))

        if self.normalize:
            norm = norm.size(1) - (norm == 0).sum(dim=1).unsqueeze(1)
            norm = torch.repeat_interleave(norm, self.hidden_graphs, dim=1)

        out = list()
        for i in range(self.max_step):
            if i == 0:
                t = torch.mul(adj_0, t)
                eye = torch.eye(self.size_hidden_graphs).to(self.device)
                eye = eye.repeat(self.hidden_graphs, 1, 1)
                o = torch.einsum("abc,acd->abd", (eye, z))
                t = torch.einsum("abc,dec->abde", (o, t))
            else:
                x = torch.spmm(adj, x)
                o = x.view(n_graphs[0], -1, x.size(1))
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                t = torch.einsum("abc,dec->abde", (z, o))
            t = self.dropout(t)
            t = torch.mul(zx, t)
            t = torch.sum(t, dim=(1,3))
            t = torch.transpose(t, 0, 1)
            if self.normalize:
                t /= norm
            out.append(t)
            
        out = torch.cat(out, dim=1)
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)