import torch.nn as nn
import torch.nn.functional as F


def get_model(params):

    if params.model == 'mlp':
        return MLP(params)
        
    elif params.model == 'res-mlp':
        return ResMLP(params)
        

class MLP(nn.Module):

    def __init__(self, params):
        super(MLP, self).__init__()
    
        stage_fn = self._bn_stage if params.batch_norm else self._linear_stage
        self.stages = nn.ModuleList([stage_fn(params.dim) for _ in range(params.depth)])
        self.dropout = nn.Dropout(params.dropout)
        self.last = nn.Linear(params.dim, params.dim)
    
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
            x = F.relu(x)
            
        x = self.dropout(x)
        x = self.last(x)
        return x
    
    @staticmethod
    def _linear_stage(dim):
        return nn.Linear(dim, dim)
    
    @staticmethod
    def _bn_stage(dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        

class ResMLP(nn.Module):

    def __init__(self, params):
        super(ResMLP, self).__init__()
    
        self.blocks = nn.ModuleList([self._res_block(params.dim) for _ in range(params.depth)])    
        self.dropout = nn.Dropout(params.dropout)
        self.last = nn.Linear(params.dim, params.dim)    
    
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        x = self.dropout(x)
        x = self.last(x)
        return x

    @staticmethod
    def _res_block(dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

