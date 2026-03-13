from helper import *
from model.message_passing import MessagePassing

class CompGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
        super(self.__class__, self).__init__()

        self.p 			= params
        self.in_channels	= in_channels
        self.out_channels	= out_channels
        self.num_rels 		= num_rels
        self.act 		= act
        self.device		= None

        self.w_loop		= get_param((in_channels, out_channels))
        self.w_in		= get_param((in_channels, out_channels))
        self.w_out		= get_param((in_channels, out_channels))
        self.w_rel 		= get_param((in_channels, out_channels))
        self.loop_rel 		= get_param((1, in_channels))

        self.drop		= torch.nn.Dropout(self.p.dropout)
        self.bn			= torch.nn.BatchNorm1d(out_channels)

        if self.p.bias: 
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

    # MODIFIED: 增加 edge_weight 参数
    def forward(self, x, edge_index, edge_type, rel_embed, edge_weight=None):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent   = x.size(0)

        # 拆分索引和类型
        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

        # MODIFIED: 拆分权重，如果提供了的话
        if edge_weight is not None:
            self.in_weight = edge_weight[:num_edges]
            self.out_weight = edge_weight[num_edges:]
        else:
            self.in_weight = None
            self.out_weight = None

        # 自环相关
        self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)
        # 自环没有权重，设为 None

        # 归一化系数（基于度，与边权重无关）
        self.in_norm     = self.compute_norm(self.in_index,  num_ent)
        self.out_norm    = self.compute_norm(self.out_index, num_ent)

        # MODIFIED: propagate 调用时传入 edge_weight 参数
        in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   
                                      rel_embed=rel_embed, edge_norm=self.in_norm, 
                                      edge_weight=self.in_weight, mode='in')
        loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, 
                                      rel_embed=rel_embed, edge_norm=None, 
                                      edge_weight=None, mode='loop')
        out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  
                                      rel_embed=rel_embed, edge_norm=self.out_norm,
                                      edge_weight=self.out_weight, mode='out')

        out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

        if self.p.bias: 
            out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # 忽略自环关系

    def rel_transform(self, ent_embed, rel_embed):
        if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
        elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
        else: raise NotImplementedError
        return trans_embed

    # MODIFIED: message 增加 edge_weight 参数，并在返回前乘以权重
    def message(self, x_j, edge_type, rel_embed, edge_norm, edge_weight, mode):
        weight 	= getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel  = self.rel_transform(x_j, rel_emb)
        out	    = torch.mm(xj_rel, weight)

        # 先乘边权重（如果有）
        if edge_weight is not None:
            out = out * edge_weight.view(-1, 1)
        # 再乘归一化系数（如果有）
        if edge_norm is not None:
            out = out * edge_norm.view(-1, 1)

        return out

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col	= edge_index
        edge_weight 	= torch.ones_like(row).float()
        deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)   # 度数
        deg_inv		= deg.pow(-0.5)						      # D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm		= deg_inv[row] * edge_weight * deg_inv[col]		      # D^{-0.5} * A * D^{-0.5}
        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)