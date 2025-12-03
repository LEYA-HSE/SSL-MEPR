
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, JumpingKnowledge, global_mean_pool
from torch_geometric.nn.dense.linear import Linear
from typing import Optional

import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.inits import glorot, zeros
from typing import List, Optional, Literal
import numpy as np
import torch

def get_edge_index_and_theta(adj):
    if isinstance(adj, torch.Tensor):
        A = adj.cpu().numpy()
    elif isinstance(adj, np.ndarray):
        A = adj
    triu_edges = np.array(np.triu_indices(len(A))).T[A[np.triu_indices(len(A))] != 0]
    tril_edges = np.array(np.tril_indices(len(A))).T[A[np.tril_indices(len(A))] != 0]
    tril_edges_flip = np.copy(tril_edges)
    tril_edges_flip[:, [0, 1]] = tril_edges_flip[:, [1, 0]]

    triu_edges_set = set(tuple(el) for el in triu_edges)
    tril_edges_set = set(tuple(el) for el in tril_edges)
    tril_edges_flip_set = set(tuple(el) for el in tril_edges_flip)

    triu_symm_edges_set = triu_edges_set & tril_edges_flip_set
    tril_symm_edges_set = set(el[::-1] for el in triu_symm_edges_set)

    triu_dir_edges_set = triu_edges_set - triu_symm_edges_set
    tril_dir_edges_set = tril_edges_set - tril_symm_edges_set

    triu_dir_edges = sorted(list(triu_dir_edges_set))
    tril_dir_edges = sorted(list(tril_dir_edges_set))
    triu_symm_edges = sorted(list(triu_symm_edges_set))

    if len(triu_symm_edges) > 0:
        if len(triu_dir_edges) == 0 and len(tril_dir_edges) == 0:
            processed_edges = np.array(triu_symm_edges)
            theta = [np.pi / 4] * len(triu_symm_edges)
        else:
            processed_edges = np.vstack((triu_dir_edges, tril_dir_edges, triu_symm_edges))
            theta = [0] * (len(triu_dir_edges) + len(tril_dir_edges)) + [np.pi / 4] * len(triu_symm_edges)
    else:
        processed_edges = np.vstack((triu_dir_edges, tril_dir_edges))
        theta = [0] * (len(triu_dir_edges) + len(tril_dir_edges))

    theta = np.array(theta)
    theta = torch.tensor(theta)
    true_theta = theta

    edge_index_fuzzy = torch.from_numpy(processed_edges.T)
    edge_index_fuzzy_reverse = torch.stack(tuple(edge_index_fuzzy)[::-1])

    if isinstance(adj, torch.Tensor):
        return edge_index_fuzzy.to(adj.device), edge_index_fuzzy_reverse.to(adj.device), theta.to(adj.device)
    elif isinstance(adj, np.ndarray):
        return edge_index_fuzzy, edge_index_fuzzy_reverse, theta


def get_fuzzy_laplacian(
        edge_index: torch.Tensor,
        theta: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        edge_weight: Optional[torch.Tensor] = None,
        add_self_loop: Optional[bool] = False,
):
    assert num_edges == theta.size(0)
    if edge_weight is not None:
        assert num_edges == edge_weight.size(0)

    if edge_weight is not None:
        edge_weight = edge_weight
    else:
        edge_weight = torch.ones(edge_index.size(1), dtype=theta.dtype)
        edge_weight.to(theta.device)

    senders, receivers = edge_index[:, :num_edges]
    conv_senders = torch.cat((senders, receivers))
    conv_receivers = torch.cat((receivers, senders))

    edge_director_src_to_tgt = torch.exp(1j * theta)
    edge_director_tgt_to_src = torch.exp(1j * (torch.pi / 2 - theta))
    edge_director = torch.cat((edge_director_src_to_tgt, edge_director_tgt_to_src))
    edge_weight = torch.cat((edge_weight, edge_weight))

    if add_self_loop:
        self_loops = torch.arange(num_nodes).to(conv_senders.device)
        conv_senders = torch.cat((conv_senders, self_loops))
        conv_receivers = torch.cat((conv_receivers, self_loops))
        edge_weight = torch.cat((edge_weight, torch.ones(num_nodes).to(conv_senders.device)))
        edge_director = torch.cat((edge_director, torch.full((num_nodes,), 1 + 1j).to(conv_senders.device)))

    out_weight = edge_director.real ** 2 * edge_weight
    in_weight = edge_director.imag ** 2 * edge_weight

    deg_senders = torch.zeros(num_nodes, dtype=out_weight.dtype, device=theta.device) + 1e-12
    deg_senders.scatter_add_(0, conv_senders, out_weight)
    deg_inv_sqrt_senders = torch.where(deg_senders < 1e-11, 0.0, torch.rsqrt(deg_senders))

    deg_receivers = torch.zeros(num_nodes, dtype=out_weight.dtype, device=theta.device) + 1e-12
    deg_receivers.scatter_add_(0, conv_senders, in_weight)
    deg_inv_sqrt_receivers = torch.where(deg_receivers < 1e-11, 0.0, torch.rsqrt(deg_receivers))

    edge_weight_src_to_tgt = deg_inv_sqrt_senders[conv_senders] * out_weight * deg_inv_sqrt_receivers[conv_receivers]
    edge_weight_tgt_to_src = deg_inv_sqrt_receivers[conv_senders] * in_weight * deg_inv_sqrt_senders[conv_receivers]


    num_repeat = edge_index.shape[1] // num_edges
    conv_senders_batch, conv_receivers_batch = [conv_senders], [conv_receivers]
    for n in range(1, num_repeat):

        conv_senders_batch.append(num_nodes * n + conv_senders)
        conv_receivers_batch.append(num_nodes * n + conv_receivers)
    conv_senders_batch = torch.cat(conv_senders_batch)
    conv_receivers_batch = torch.cat(conv_receivers_batch)

    edge_weight_src_to_tgt_batch = edge_weight_src_to_tgt.repeat(num_repeat).unsqueeze(-1)
    edge_weight_tgt_to_src_batch = edge_weight_tgt_to_src.repeat(num_repeat).unsqueeze(-1)

    conv_edge_index = torch.stack((conv_senders_batch, conv_receivers_batch), dim=0)
    conv_edge_weight = (edge_weight_src_to_tgt_batch, edge_weight_tgt_to_src_batch)

    return conv_edge_index, conv_edge_weight


class FuzzyDirGCNConv(MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        aggr_method="add",
        self_feature_transform=False,
        dtype=torch.float,
    ):
        super(FuzzyDirGCNConv, self).__init__(aggr=aggr_method)
        self.aggr = aggr_method
        self.self_feature_transform = self_feature_transform

        self.lin_src_to_dst = Linear(
            in_channels, out_channels,
            bias=False, weight_initializer='glorot')

        self.lin_dst_to_src = Linear(
            in_channels, out_channels,
            bias=False, weight_initializer='glorot')

        if self.self_feature_transform:
            self.lin_self = Linear(
                in_channels, out_channels,
                bias=False, weight_initializer='glorot')
        else:
            self.lin_self = None

        if bias:
            self.bias_src_to_dst = Parameter(torch.empty(out_channels))
            self.bias_dst_to_src = Parameter(torch.empty(out_channels))
            if self.self_feature_transform:
                self.bias_self = Parameter(torch.empty(out_channels))
            else:
                self.bias_self = None
        else:
            self.bias_src_to_dst = None
            self.bias_dst_to_src = None
            self.bias_self = None

    def reset_parameters(self):
        glorot(self.lin_src_to_dst)
        glorot(self.lin_dst_to_src)
        zeros(self.bias_src_to_dst)
        zeros(self.bias_dst_to_src)
        if self.lin_self is not None:
            glorot(self.lin_dst_to_src)
        if self.bias_self is not None:
            zeros(self.bias_self)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        edge_weight_src_to_tgt, edge_weight_tgt_to_src = edge_weight

        x_src_to_dst = self.propagate(
            edge_index, x=x, edge_weight=edge_weight_src_to_tgt)
        x_dst_to_src = self.propagate(
            edge_index, x=x, edge_weight=edge_weight_tgt_to_src)

        x_src_to_dst = self.lin_src_to_dst(x_src_to_dst)
        x_dst_to_src = self.lin_dst_to_src(x_dst_to_src)

        if self.bias_src_to_dst is not None:
            x_src_to_dst = x_src_to_dst + self.bias_src_to_dst
            x_dst_to_src = x_dst_to_src + self.bias_dst_to_src

        if self.self_feature_transform:
            x_self = self.lin_self(x)
            if self.bias_self is not None:
                x_self = x_self + self.bias_self
            return x_src_to_dst, x_dst_to_src, x_self
        else:
            return x_src_to_dst, x_dst_to_src

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)

def _dense_adj_to_edge_index(adj_2d: torch.Tensor) -> torch.Tensor:

    idx = (adj_2d > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=adj_2d.device)
    return idx.t().contiguous()


class GraphAttentionLayer_v2(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        *,
        alpha: Optional[float] = 0.0,
        self_feature_transform: bool = False,
        self_loop: bool = False,
        dropout: float = 0.2,
        normalize: bool = False,
        aggr_method: str = "add",
    ):
        super().__init__()
        out_dim = out_dim or in_dim

        self.out_dim = out_dim
        self.alpha = alpha
        self.self_feature_transform = self_feature_transform
        self.self_loop = self_loop
        self.dropout = dropout
        self.normalize = normalize

        self.conv = FuzzyDirGCNConv(
            in_channels=in_dim,
            out_channels=out_dim,
            aggr_method=aggr_method,
            self_feature_transform=self_feature_transform,
        )
        if hasattr(self.conv, "reset_parameters"):
            self.conv.reset_parameters()

    @torch.no_grad()
    def _build_batched_edges(
            self,
            adj_b: torch.Tensor,
            theta_b: Optional[torch.Tensor | list],
            edge_w_b: Optional[torch.Tensor | list],
    ):

        device = adj_b.device
        B, N, _ = adj_b.shape

        all_ei, all_ws2d, all_wd2s = [], [], []
        offset = 0

        for b in range(B):
            adj = adj_b[b]

            ei_fwd, ei_rev, theta_auto = get_edge_index_and_theta(adj)
            ei = ei_fwd.to(device)
            E_b = ei.size(1)

            if theta_b is None:
                th = theta_auto.to(device).float()
            else:
                if isinstance(theta_b, list):
                    th = theta_b[b].to(device).float()
                else:
                    th = theta_b[b]
                    if th.dim() == 2 and th.shape[:2] == (N, N):
                        th = th.to(device).float()[ei[0], ei[1]]
                    elif th.dim() == 1:
                        th = th.to(device).float()
                        if th.numel() != E_b:
                            raise ValueError(f"theta length {th.numel()} != num_edges {E_b}")
                    else:
                        raise ValueError(f"Unsupported theta shape: {tuple(th.shape)}")

            if edge_w_b is None:
                w = torch.ones(E_b, dtype=th.dtype, device=device)
            else:
                if isinstance(edge_w_b, list):
                    w = edge_w_b[b].to(device)
                else:
                    wb = edge_w_b[b]
                    if wb.dim() == 2 and wb.shape[:2] == (N, N):
                        w = wb.to(device)[ei[0], ei[1]]
                    elif wb.dim() == 1 and wb.numel() == E_b:
                        w = wb.to(device)
                    else:
                        w = torch.ones(E_b, dtype=th.dtype, device=device)

            conv_ei, (w_s2d, w_d2s) = get_fuzzy_laplacian(
                ei, th, N, E_b, w, self.self_loop
            )

            all_ei.append(conv_ei.to(device) + offset)
            all_ws2d.append(w_s2d.to(device))
            all_wd2s.append(w_d2s.to(device))
            offset += N

        edge_index = torch.cat(all_ei, dim=1) if all_ei else torch.empty(2, 0, dtype=torch.long, device=device)
        w_src2dst = torch.cat(all_ws2d, dim=0) if all_ws2d else torch.empty(0, device=device)
        w_dst2src = torch.cat(all_wd2s, dim=0) if all_wd2s else torch.empty(0, device=device)
        return edge_index, (w_src2dst, w_dst2src)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, N, D_in = h.shape

        edge_index, (w_s2d, w_d2s) = self._build_batched_edges(adj, theta, edge_weight)

        x = h.reshape(B * N, D_in)

        xs = self.conv(x, edge_index, (w_s2d, w_d2s))

        if isinstance(xs, tuple) and len(xs) == 3:
            x_in, x_out, x_self = xs
            if self.alpha is not None:
                y = self.alpha * x_in + (1.0 - self.alpha) * x_out + x_self
            else:
                y = x_in + x_out + x_self
        elif isinstance(xs, tuple) and len(xs) == 2:
            x_in, x_out = xs
            if self.alpha is not None:
                y = self.alpha * x_in + (1.0 - self.alpha) * x_out
            else:
                y = x_in + x_out
        else:
            raise RuntimeError("FuzzyDirGCNConv вернул неожиданный формат выхода")

        y = y.view(B, N, -1)

        if self.dropout and self.training:
            y = F.dropout(y, p=self.dropout, training=True)
        if self.normalize:
            y = F.normalize(y, p=2, dim=-1)

        return y
