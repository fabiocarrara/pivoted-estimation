import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(params):
    return get_model_from_dict(params.to_dict())

def get_model_from_dict(params):
    model_type = params.get('model', 'res-mlp')

    if model_type == 'mlp':
        return MLP(params)

    elif model_type == 'res-mlp':
        return ResMLP(params)


class MLP(nn.Module):

    def __init__(self, params):
        super(MLP, self).__init__()

        stage_fn = self._bn_stage if params.get('batch_norm', True) else self._linear_stage

        fusion_strategy = params.get('fusion', 'early')
        depth = params.get('depth', 1)
        if fusion_strategy == 'early':
            pre, post = 0, depth
        elif fusion_strategy == 'mid':
            pre = depth // 2
            post = depth - pre
        elif fusion_strategy == 'late':
            pre, post = depth, 0

        op_dim = params.get('dim', 64)
        pp_dim = op_dim * (op_dim - 1) // 2
        post_dim = op_dim + pp_dim

        self.pre_stages_op = nn.ModuleList([stage_fn(op_dim) for _ in range(pre)])
        self.pre_stages_pp = nn.ModuleList([stage_fn(pp_dim) for _ in range(pre)])

        self.post_stages = nn.ModuleList([stage_fn(post_dim) for _ in range(post)])
        self.dropout = nn.Dropout(params.get('dropout', 0))
        self.last = nn.Linear(post_dim, op_dim)

    def forward(self, op, pp):
        # object-pivot branch
        for stage in self.pre_stages_op:
            op = stage(op)

        # pivot-pivot branch
        if pp.dim() < 2:
            pp = pp.unsqueeze(0)

        for stage in self.pre_stages_pp:
            pp = stage(pp)

        # combined branch
        pp = pp.expand(op.shape[0], -1)  # expand to batch_size
        x = torch.cat((op, pp), dim=1)  # fusion (concatenate)

        for stage in self.post_stages:
            x = stage(x)

        x = self.dropout(x)
        x = self.last(x)
        return x

    @staticmethod
    def _linear_stage(dim):
        return nn.Sequential( nn.Linear(dim, dim), nn.ReLU() )

    @staticmethod
    def _bn_stage(dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )


class ResMLP(nn.Module):

    def __init__(self, params):
        super(ResMLP, self).__init__()

        block_fn = self._bn_res_block if params.get('batch_norm', True) else self._res_block

        fusion_strategy = params.get('fusion', 'early')
        depth = params.get('depth', 1)
        if fusion_strategy == 'early':
            pre, post = 0, depth
        elif fusion_strategy == 'mid':
            pre = depth // 2
            post = depth - pre
        elif fusion_strategy == 'late':
            pre, post = depth, 0

        op_dim = params.get('dim', 64)
        pp_dim = op_dim * (op_dim - 1) // 2
        post_dim = op_dim + pp_dim

        self.pre_stages_op = nn.ModuleList([block_fn(op_dim) for _ in range(pre)])
        self.pre_stages_pp = nn.ModuleList([block_fn(pp_dim) for _ in range(pre)])

        self.post_stages = nn.ModuleList([block_fn(post_dim) for _ in range(post)])
        self.dropout = nn.Dropout(params.get('dropout', 0))
        self.last = nn.Linear(post_dim, op_dim)

    def forward(self, op, pp):
        # object-pivot branch
        for stage in self.pre_stages_op:
            op = op + stage(op)

        # pivot-pivot branch
        if pp.dim() < 2:
            pp = pp.unsqueeze(0)
            pp = pp.expand(op.shape[0], -1)  # expand to batch_size

        for stage in self.pre_stages_pp:
            pp = pp + stage(pp)

        # combined branch
        x = torch.cat((op, pp), dim=1)  # fusion (concatenate)

        for stage in self.post_stages:
            x = x + stage(x)

        x = self.dropout(x)
        x = self.last(x)
        return x

    @staticmethod
    def _bn_res_block(dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    @staticmethod
    def _res_block(dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

