from collections import defaultdict

import torch
from torch_scatter import scatter
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


def _eval_mrr(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor):
    y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
    argsort = torch.argsort(y_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    mrr_list = 1. / ranking_list.to(torch.float)
    return mrr_list


def eval_mrr(batch: Batch):
    mrrs = defaultdict(list)
    for data in batch.to_data_list():
        pred = data.x @ data.x.transpose(0, 1)

        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        num_pos_edges = pos_edge_index.shape[1]

        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]

        assert num_pos_edges > 0

        # raw MRR
        neg_mask = torch.ones([num_pos_edges, data.num_nodes], dtype=torch.bool)
        neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
        pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
        mrrs['raw'].append(_eval_mrr(pred_pos, pred_neg).mean().item())
        pred_masked = pred.clone()
        pred_masked[pos_edge_index[0], pos_edge_index[1]] -= float("inf")
        pred_neg = pred_masked[pos_edge_index[0]]
        mrrs['filter'].append(_eval_mrr(pred_pos, pred_neg).mean().item())

        pred_masked[torch.arange(data.num_nodes), torch.arange(data.num_nodes)] -= float("inf")
        pred_neg = pred_masked[pos_edge_index[0]]
        mrrs['ext_filter'].append(_eval_mrr(pred_pos, pred_neg).mean().item())

    return mrrs


def _eval_mrr_batch(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor, pos_edge_batch_index: torch.Tensor):
    concat_pos_neg_pred = torch.cat([y_pred_pos, y_pred_neg], dim=1)
    argsort = torch.argsort(concat_pos_neg_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    mrr_list = scatter(1. / ranking_list.to(torch.float), pos_edge_batch_index, dim=0, reduce='mean')
    return mrr_list


def eval_mrr_batch(batch: Batch):
    device = batch.x.device
    npreds = (batch._slice_dict['edge_label'][1:] - batch._slice_dict['edge_label'][:-1]).to(device)
    nnodes = (batch._slice_dict['x'][1:] - batch._slice_dict['x'][:-1]).to(device)

    reshaped_pred, _ = to_dense_batch(
        batch.x,
        batch=batch.batch,
        max_num_nodes=nnodes.max())
    y_pred = torch.einsum('bnf,bmf->bnm', reshaped_pred, reshaped_pred)

    y_true = batch.edge_label

    # un-batch the edge label index
    num_graphs = len(nnodes)

    offset = torch.cat([nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]])
    offset = torch.repeat_interleave(offset, npreds)
    edge_label_idx = batch.edge_label_index - offset[None]

    arange_num_graphs = torch.arange(num_graphs, device=device)  # a shared tensor
    edge_batch_index = torch.repeat_interleave(arange_num_graphs, npreds)

    # get positive edges
    pos_edge_index = edge_label_idx[:, y_true == 1]
    num_pos_edges_list = scatter(y_true.long(), edge_batch_index, dim=0, reduce='sum')
    assert num_pos_edges_list.min() > 0
    num_pos_edges = num_pos_edges_list.sum()
    pos_edge_batch_index = edge_batch_index[y_true == 1]
    pred_pos = y_pred[pos_edge_batch_index, pos_edge_index[0], pos_edge_index[1]].reshape(num_pos_edges, 1)

    # get negative edges
    # pad some out of range entries
    y_pred[arange_num_graphs.repeat_interleave(nnodes.max() - nnodes), :,
    torch.cat([torch.arange(n, nnodes.max(), device=device) for n in nnodes])] -= float('inf')

    neg_mask = torch.ones(num_pos_edges, nnodes.max(), dtype=torch.bool, device=device)
    neg_mask[torch.arange(num_pos_edges, device=device), pos_edge_index[1]] = False
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0], :][neg_mask].reshape(num_pos_edges, nnodes.max() - 1)
    mrr_list_raw = _eval_mrr_batch(pred_pos, pred_neg, pos_edge_batch_index)

    # filtered
    y_pred[pos_edge_batch_index, pos_edge_index[0], pos_edge_index[1]] -= float("inf")
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0], :]
    mrr_list_filtered = _eval_mrr_batch(pred_pos, pred_neg, pos_edge_batch_index)

    diag_arange = torch.arange(nnodes.max(), device=device)  # a shared tensor
    # self filtered
    y_pred[:, diag_arange, diag_arange] -= float("inf")
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0], :]
    mrr_list_self_filtered = _eval_mrr_batch(pred_pos, pred_neg, pos_edge_batch_index)

    return {'raw': mrr_list_raw, 'filter': mrr_list_filtered, 'ext_filter': mrr_list_self_filtered}
