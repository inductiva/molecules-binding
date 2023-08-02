"""Redefining GATConv with sparsemax activation instead of softmax"""
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
# from torch_geometric.utils import softmax
from torch_geometric.typing import OptTensor
from torch_geometric.nn import GATConv



def sparsemax(a: Tensor) -> Tensor:
    zs = torch.sort(a, descending=True, dim=0).values
    size = zs.size(0)
    indices = torch.arange(start=1,
                         end=size + 1,
                         step=1,
                         dtype=int,
                         device=a.device).reshape(size, 1)
    bound = torch.as_tensor(1, device=a.device) + indices * zs
    cum_sum_zs = torch.cumsum(zs, dim=0)
    is_ge = torch.ge(bound, cum_sum_zs)
    k = torch.max(is_ge * indices)
    tau = (cum_sum_zs[k - 1] - torch.as_tensor(1, device=a.device)) / k
    return torch.relu(a - tau)


def sparsemax_pyg(src, index, ptr, size_i) -> Tensor:
    unique_indices = torch.unique(index)
    # print("index foi usado")
    result = torch.zeros_like(src)

    for i in unique_indices:
        mask = index == i
        result[mask] = sparsemax(src[mask])
    return result


# def sparsemax_pyg(src, index, ptr, size_i) -> Tensor:
#     count_per_index = torch.bincount(index)

#     indexes_sorted = torch.sort(index).indices

#     separated_tensors = torch.split(src[indexes_sorted],
#                                     count_per_index.tolist())
#     separated_outputs = list(map(sparsemax, separated_tensors))
#     output = torch.cat(separated_outputs, dim=0)[indexes_sorted.argsort()]
#     return output


class GATConvSparsemax(GATConv):
    """ GAT layer with sparsemax activation instead of softmax"""
    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # alpha = softmax(alpha, index, ptr, size_i)
        alpha = sparsemax_pyg(alpha, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
