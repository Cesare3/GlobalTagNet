import math
from typing import Dict, List, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from dgl.ops import edge_softmax
from torch import nn

# 2020WWW - Heterogeneous Graph Transformer

class HGTLayer(nn.Module):
    """Heterogeneous Graph Transformer Layer"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ntype2idx: Dict[str, int],
        etype2idx: Dict[str, int],
        n_heads: int,
        dropout: float = 0.2,
        use_norm: bool = True,
        attn_dropout: float = 0.2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ntype2idx = ntype2idx
        self.etype2idx = etype2idx
        self.num_types = len(ntype2idx)
        self.num_relations = len(etype2idx)
        # (head, edge, tail)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        assert out_dim % self.n_heads == 0
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        # multi-head attention mechanism
        # K, Q, V are three learnable parameters (matrix)
        # H is the hidden state (tensor)
        # H_{i} / H_{j} is the i-/j-th vector
        # QH_{i} * KH_{j}
        # softmax(·)
        # VH_{j}
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        # [num_relations, n_heads]
        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        # [num_relations, n_heads, d_k, d_k]
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        # [num_relations, n_heads, d_h, d_k]
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(
        self, graph: dgl.DGLHeteroGraph,
        h: Dict[str, torch.Tensor],
        etypes: List[Tuple[str, str, str]] = None,
    ) -> Dict[str, torch.Tensor]:
        etypes = etypes or graph.canonical_etypes

        with graph.local_scope():
            node_dict, edge_dict = self.ntype2idx, self.etype2idx

            for ntype in graph.ntypes:
                # str/tuple -> int
                k_linear = self.k_linears[node_dict[ntype]]
                v_linear = self.v_linears[node_dict[ntype]]
                q_linear = self.q_linears[node_dict[ntype]]
                # [bsz, n_heads, d_k]
                # in_dim, out_dim
                # [bsz, in_dim] -> [bzs, out_dim] -> (bsz, n_heads, d_k) n_heads * d_k = out_dim
                graph.nodes[ntype].data[f'{ntype}_k'] = k_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                graph.nodes[ntype].data[f'{ntype}_v'] = v_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                graph.nodes[ntype].data[f'{ntype}_q'] = q_linear(h[ntype]).view(-1, self.n_heads, self.d_k)

            for srctype, etype, dsttype in etypes:
                sub_graph = graph[srctype, etype, dsttype]

                # [bsz, n_heads, d_k]
                k = graph.nodes[srctype].data[f'{srctype}_k']
                v = graph.nodes[srctype].data[f'{srctype}_v']
                q = graph.nodes[dsttype].data[f'{dsttype}_q']

                e_id = self.etype2idx[etype]
                # [n_heads, d_k, d_k]
                relation_pri = self.relation_pri[e_id]  # [n_heads]
                relation_att = self.relation_att[e_id]
                relation_msg = self.relation_msg[e_id]

                # [bsz, n_heads, d_k]
                # [bsz, n_heads, d_k] [n_heads, d_k, d_k] => [bsz, n_heads, d_k]
                k = torch.einsum('bij,ijk->bik', k, relation_att)
                v = torch.einsum('bij,ijk->bik', v, relation_msg)

                # (head, edge, tail)
                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{srctype}_{etype}_{dsttype}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = self.attn_drop(edge_softmax(sub_graph, attn_score, norm_by='dst'))

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            graph.multi_update_all({
                (srctype, etype, dsttype): (fn.u_mul_e(f'v_{srctype}_{etype}_{dsttype}', 't', 'm'), fn.sum('m', 't'))
                for srctype, etype, dsttype in etypes
            }, cross_reducer='mean')

            new_h = {}
            for ntype in graph.ntypes:
                n_id = node_dict[ntype]
                t = graph.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out + h[ntype]
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out

            return new_h


class HGT(nn.Module):
    def __init__(
        self,
        ntype2idx: Dict[str, int],
        etype2idx: Dict[str, int],
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        n_heads: int,
        use_norm: bool = True,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.ntype2idx = ntype2idx
        self.etype2id = etype2idx
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        if self.in_dim != self.hidden_dim:
            for t in range(len(ntype2idx)):
                self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_layers):
            self.gcs.append(
                HGTLayer(hidden_dim, hidden_dim,
                         ntype2idx, etype2idx, n_heads,
                         use_norm=use_norm, dropout=dropout, attn_dropout=attn_drop)
            )

        if hidden_dim != out_dim:
            self.out = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, graph: dgl.DGLHeteroGraph,
        features: Dict[str, torch.Tensor] = None,
        etypes: List[Tuple[str, str, str]] = None,
    ):
        h = {}
        for ntype in graph.ntypes:
            input_feat = None if features is None else features.get(ntype)
            if input_feat is None:
                input_feat = graph.nodes[ntype].data['feature']
            if self.in_dim != self.hidden_dim:
                n_id = self.ntype2idx[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](input_feat))
            else:
                h[ntype] = input_feat

        for i in range(self.n_layers):
            h = self.gcs[i](graph, h, etypes=etypes)

        return {
            k: self.out(h[k]) for k in h
        } if self.hidden_dim != self.out_dim else h


if __name__ == "__main__":
    from dgl.data import CSVDataset

    csv = CSVDataset("/Users/saner/Desktop/第四节课/pl_mltc/data/aapd/full_connected_label_graph")
    graph = csv[0]

    print(graph.ntypes[0])
    print(graph.etypes[0])
    print(graph.nodes["item"].data["feature"])
    print(graph.nodes["item"].data["feature"].shape)
    model = HGT(
        ntype2idx={"item": 0},
        etype2idx={graph.etypes[0]: 0},
        in_dim=128,
        hidden_dim=64,
        out_dim=64,
        n_layers=2,
        n_heads=4,
        use_norm=True,
        dropout=0.1,
        attn_drop=0.1,
    )

    h = model(graph)
    print(h['item'].shape)
