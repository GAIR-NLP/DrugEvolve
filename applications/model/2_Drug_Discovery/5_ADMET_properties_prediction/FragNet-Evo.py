import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.norm import BatchNorm

def scatter_add(src, index, dim=0, dim_size=None):
    """Compile-friendly grouped sum. Callers must provide static dim_size."""
    if dim_size is None:
        raise ValueError("scatter_add requires explicit dim_size for compile-friendly execution")
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    return out.scatter_add_(0, idx.expand_as(src), src)


def scatter_mean(src, index, dim=0, dim_size=None):
    """Compile-friendly grouped mean. Callers provide static dim_size."""
    if dim_size is None:
        raise ValueError("scatter_mean requires a static dim_size")
    ones = src.new_ones((src.size(0),) + (1,) * (src.dim() - 1))
    denom = scatter_add(ones, index, dim=dim, dim_size=dim_size).clamp_min(1.0)
    return scatter_add(src, index, dim=dim, dim_size=dim_size) / denom


def scatter_max(src, index, dim=0, dim_size=None):
    """Compile-friendly grouped max. Empty groups are filled with zeros."""
    if dim_size is None:
        raise ValueError("scatter_max requires a static dim_size")
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    out = src.new_full((dim_size,) + tuple(src.shape[1:]), float("-inf"))
    out = out.scatter_reduce_(0, idx.expand_as(src), src, reduce="amax", include_self=False)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def scatter_softmax(src, index, dim=0, dim_size=None):
    """Numerically-stable softmax over groups defined by `index` along dim 0."""
    if dim_size is None:
        raise ValueError("scatter_softmax requires explicit dim_size for compile-friendly execution")
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    mx = src.new_full((dim_size,) + tuple(src.shape[1:]), float("-inf"))
    mx = mx.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
    e = (src - mx.gather(0, idx)).exp()
    s = scatter_add(e, index, dim_size=dim_size).gather(0, idx)
    return e / s.clamp_min(torch.finfo(e.dtype).tiny)


class FragNetLayerA(nn.Module):
    def __init__(
        self,
        atom_in=128,
        atom_out=128,
        frag_in=128,
        frag_out=128,
        edge_in=128,
        edge_out=128,
        fedge_in=128,
        num_heads=2,
        bond_edge_in=1,
        fbond_edge_in=8,
        return_attentions=False,
        add_frag_self_loops=False,
        bond_mask = None,
        frag_bond_mask=None,
        atom_mask_individual=None
    ):
        super().__init__()

        self.add_frag_self_loops = add_frag_self_loops
        self.return_attentions = return_attentions
        self.edge_out = edge_out
        self.atom_self_loop_attr = nn.Parameter(torch.zeros(1, edge_out))
        self.atom_embed = nn.Linear(atom_in, atom_out, bias=True)
        self.frag_embed = nn.Linear(frag_in, frag_out)
        self.edge_embed = nn.Linear(edge_in, edge_out)
        self.fedge_embed = nn.Linear(fedge_in, edge_out)
        self.bond_edge_embed = nn.Linear(edge_in, edge_out)

        self.frag_message_mlp = nn.Linear(atom_out * 2, atom_out)
        self.frag_fuse_gate = nn.Linear(atom_out * 2, atom_out)
        self.atom_mlp = torch.nn.Sequential(
            torch.nn.Linear(atom_out, 2 * atom_out),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * atom_out, atom_out),
        )

        self.frag_mlp = torch.nn.Sequential(
            torch.nn.Linear(atom_out, 2 * atom_out),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * atom_out, atom_out),
        )
        self.bias = Parameter(torch.Tensor(atom_out))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.num_heads = num_heads
        self.edge_attr_bond_embed2 = nn.Linear(edge_out, edge_out)

        edge_out = edge_out // self.num_heads
        self.projection_b = nn.Linear(edge_in, edge_out * self.num_heads, bias=True)
        self.projection_fb = nn.Linear(fedge_in, edge_out * self.num_heads, bias=True)

        self.edge_attr_bond_embed = nn.Linear(bond_edge_in, edge_out)
        self.edge_attr_fbond_embed = nn.Linear(fbond_edge_in, edge_out)

        atom_out = atom_out // self.num_heads
        self.projection_a = nn.Linear(
            atom_in, atom_out * self.num_heads
        )  # NOTE: check this
        self.a_b = nn.Parameter(
            torch.Tensor(self.num_heads, 2 * edge_out + edge_out)
        )  # One per head
        self.a = nn.Parameter(
            torch.Tensor(self.num_heads, 2 * atom_out + edge_out * self.num_heads)
        )
        self.f = nn.Parameter(
            torch.Tensor(self.num_heads, 2 * atom_out + edge_out * self.num_heads)
        )  # replace this with embed_dim
        self.f_a_b = nn.Parameter(
            torch.Tensor(self.num_heads, 2 * edge_out + edge_out)
        )  # replace this with embed_dim

        # Zero-start, bounded value/residual paths.  Edge descriptors can become
        # message payloads after pretraining, but the initial computation remains
        # close to the stable baseline and avoids regression scale drift.
        self.edge_value_scale_b = nn.Parameter(torch.zeros(1, self.num_heads, edge_out))
        self.edge_value_scale_a = nn.Parameter(torch.zeros(1, self.num_heads, edge_out))
        self.edge_value_scale_fb = nn.Parameter(torch.zeros(1, self.num_heads, edge_out))
        self.edge_value_scale_f = nn.Parameter(torch.zeros(1, self.num_heads, edge_out))
        self.edge_skip_scale = nn.Parameter(torch.zeros(1))
        self.fedge_skip_scale = nn.Parameter(torch.zeros(1))
        self.atom_skip_scale = nn.Parameter(torch.zeros(1))
        self.frag_skip_scale = nn.Parameter(torch.zeros(1))
        self.frag_graph_skip_scale = nn.Parameter(torch.zeros(1))
        self.atom_from_frag_scale = nn.Parameter(torch.zeros(1))

        # Narrow attention-temperature adaptation: enough to tune sharpness, but
        # capped so high-lr regression finetuning cannot collapse attention.
        self.attn_logit_scale_b = nn.Parameter(torch.zeros(1, self.num_heads))
        self.attn_logit_scale_a = nn.Parameter(torch.zeros(1, self.num_heads))
        self.attn_logit_scale_fb = nn.Parameter(torch.zeros(1, self.num_heads))
        self.attn_logit_scale_f = nn.Parameter(torch.zeros(1, self.num_heads))

        nn.init.xavier_uniform_(self.projection_b.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_b.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.f.data, gain=1.414)
        nn.init.xavier_uniform_(self.f_a_b.data, gain=1.414)
        nn.init.zeros_(self.frag_fuse_gate.weight)
        nn.init.constant_(self.frag_fuse_gate.bias, -2.0)

        self.bond_mask = bond_mask
        self.frag_bond_mask = frag_bond_mask
        self.atom_mask_individual = atom_mask_individual

    def forward(
        self,
        x_atoms,
        edge_index,
        edge_attr,
        frag_index,
        x_frags,
        atom_to_frag_ids,
        node_feautures_bond_graph,
        edge_index_bonds_graph,
        edge_attr_bond_graph,
        node_feautures_fbond_graph,
        edge_index_fbond_graph,
        edge_attr_fbond_graph,
        edge_index_sl=None,
    ):

        num_frags = x_frags.size(0)  # static dim_size for the atom->fragment scatter

        # new bond features from bond graph
        target, source = edge_index_bonds_graph
        edge_attr_bond_graph = self.edge_attr_bond_embed(edge_attr_bond_graph)
        ea_bonds = edge_attr_bond_graph.unsqueeze(1).expand(-1, self.num_heads, -1)
        num_nodes_b = node_feautures_bond_graph.size(0)
        node_feats_b = self.projection_b(node_feautures_bond_graph)
        node_feats_b = node_feats_b.view(num_nodes_b, self.num_heads, -1)

        # select source and target features
        source_features = torch.index_select(input=node_feats_b, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_b, index=target, dim=0)
        message = torch.cat([target_features, ea_bonds, source_features], dim=-1)

        attn_logits = torch.sum(message * self.a_b, dim=2)
        attn_logits = self.leakyrelu(attn_logits)
        attn_logits = attn_logits * (1.0 + 0.1 * torch.tanh(self.attn_logit_scale_b)).to(attn_logits.dtype)

        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_b).to(attn_logits.dtype)  # eij; fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_feats_b, index=source, dim=0)  # select hj
        value_b = hj + 0.5 * torch.tanh(self.edge_value_scale_b).to(hj.dtype) * ea_bonds

        node_feats_b = (
            attn_probs[..., None] * value_b
        )  # multipy the attention coefficients corresponding to (neighbors - target atom) with the attention
        # coefficients corresponding to the neighbors. [since we are considering edge features when finding the
        # attention coefficients, should we use something like attn_probs[..., None]*(hj+ea_bonds) or

        node_feats_sum_b = scatter_add(
            src=node_feats_b, index=target, dim_size=num_nodes_b
        )  # get the sum of neighboring node features
        # summed attn weights are only returned for viz (return_attentions); skip in training
        summed_attn_weights_bonds = (
            scatter_add(attn_probs, source, dim=0, dim_size=num_nodes_b) if self.return_attentions else None
        )

        new_bond_features = node_feats_sum_b.view(num_nodes_b, -1)
        new_bond_features = new_bond_features + torch.tanh(self.edge_skip_scale) * self.edge_embed(node_feautures_bond_graph)

        # Apply the bond mask here.
        if self.bond_mask is not None:
            print(f'bond mask value: {self.bond_mask}')
            print('applying bond mask')
            with torch.no_grad():
                new_bond_features[self.bond_mask:self.bond_mask+2, :] = 0.0
        
        # self-looped atom edge_index is identical across layers; reuse if given
        if edge_index_sl is None:
            edge_index_sl, _ = add_self_loops(edge_index=edge_index)
        edge_index = edge_index_sl

        self_loop_attr = self.atom_self_loop_attr.to(
            dtype=new_bond_features.dtype, device=new_bond_features.device
        ).expand(x_atoms.size(0), -1)
        # TODO: use a nn transform concatenated new_bond_features and edge_attr to new edge_attr
        edge_attr = torch.cat(
            (new_bond_features, self_loop_attr), dim=0
        )  # <- new

        source, target = edge_index

        node_features_atom_graph = self.projection_a(x_atoms)
        num_nodes_a = node_features_atom_graph.size(0)

        node_features_atom_graph = node_features_atom_graph.view(
            num_nodes_a, self.num_heads, -1
        )

        source_features = torch.index_select(
            input=node_features_atom_graph, index=source, dim=0
        )  # here, source and target are for the edge_index with self loops added above
        target_features = torch.index_select(
            input=node_features_atom_graph, index=target, dim=0
        )  # so, in the message below, we can use the new edge attr we found above

        edge_attr_repeat = edge_attr.unsqueeze(1).expand(-1, self.num_heads, -1)
        message = torch.cat(
            [target_features, edge_attr_repeat, source_features], dim=-1
        )

        attn_logits = torch.sum(message * self.a, dim=2)  # NOTE: replace a by self.a
        attn_logits = self.leakyrelu(attn_logits)
        attn_logits = attn_logits * (1.0 + 0.1 * torch.tanh(self.attn_logit_scale_a)).to(attn_logits.dtype)
        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_a).to(attn_logits.dtype)  # fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_features_atom_graph, index=source, dim=0)
        edge_value = edge_attr.view(edge_attr.size(0), self.num_heads, self.edge_out // self.num_heads)
        value_a = hj + 0.5 * torch.tanh(self.edge_value_scale_a).to(hj.dtype) * edge_value

        node_features_atom_graph = (
            attn_probs[..., None] * value_a
        )  # multiply the node features by the attention weights
        node_feats_sum_a = scatter_add(
            src=node_features_atom_graph, index=target, dim_size=num_nodes_a
        )  # nodes in the bond graph are the edges in the atom graph
        summed_attn_weights_atoms = (
            scatter_add(attn_probs, source, dim=0, dim_size=num_nodes_a) if self.return_attentions else None
        )

        # new atom features
        x_atoms_new = node_feats_sum_a.view(num_nodes_a, -1)
        frag_prior = self.frag_embed(x_frags)

        # Size-normalized fragment context gives each atom scaffold information
        # without repeated-count amplification when atoms are summed back to frags.
        frag_atom_count = scatter_add(
            src=x_atoms_new.new_ones((x_atoms_new.size(0), 1)),
            index=atom_to_frag_ids,
            dim_size=num_frags,
        ).clamp_min(1.0)
        frag_context = frag_prior / frag_atom_count
        frag_context_atoms = torch.index_select(input=frag_context, index=atom_to_frag_ids, dim=0)
        x_atoms_new = (
            x_atoms_new
            + torch.tanh(self.atom_skip_scale) * self.atom_embed(x_atoms)
            + 0.1 * torch.tanh(self.atom_from_frag_scale) * frag_context_atoms
        )

        # apply the mask for atom
        if self.atom_mask_individual is not None:
            # print(x_atoms_new, x_atoms_new.shape)
            with torch.no_grad():
                print('applying atom mask')
                x_atoms_new[self.atom_mask_individual, :] = 0.0
            print(x_atoms_new, x_atoms_new.shape)
        # Preserve the extensive atom-sum fragment signal and add only a capped
        # BRICS descriptor correction.  This keeps regression calibration while
        # retaining fragment-prior chemistry for motif classification.
        x_frags_from_atoms = scatter_add(src=x_atoms_new, index=atom_to_frag_ids, dim_size=num_frags)
        frag_desc_gate = 0.5 * torch.sigmoid(
            self.frag_fuse_gate(torch.cat((x_frags_from_atoms, frag_prior), dim=-1))
        )
        x_frags = (
            x_frags_from_atoms
            + frag_desc_gate * frag_prior
            + torch.tanh(self.frag_skip_scale) * frag_prior
        )



        # get frag bond features.
        target, source = edge_index_fbond_graph

        
        edge_attr_fbond_graph = self.edge_attr_fbond_embed(edge_attr_fbond_graph)
                
                
        ea_fbonds = edge_attr_fbond_graph.unsqueeze(1).expand(-1, self.num_heads, -1)
        num_nodes_fb = node_feautures_fbond_graph.size(0)
        node_feats_fb = self.projection_fb(node_feautures_fbond_graph)
        node_feats_fb = node_feats_fb.view(num_nodes_fb, self.num_heads, -1)

        source_features = torch.index_select(input=node_feats_fb, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_fb, index=target, dim=0)
        message = torch.cat([target_features, ea_fbonds, source_features], dim=-1)

        attn_logits = torch.sum(message * self.f_a_b, dim=2)
        attn_logits = self.leakyrelu(attn_logits)
        attn_logits = attn_logits * (1.0 + 0.1 * torch.tanh(self.attn_logit_scale_fb)).to(attn_logits.dtype)

        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_fb).to(attn_logits.dtype)  # eij; fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_feats_fb, index=source, dim=0)  # select hj
        value_fb = hj + 0.5 * torch.tanh(self.edge_value_scale_fb).to(hj.dtype) * ea_fbonds

        node_feats_fb = (
            attn_probs[..., None] * value_fb
        )  # multipy the attention coefficients corresponding to (neighbors - target atom) with the attention
        # coefficients corresponding to the neighbors. [since we are considering edge features when finding the
        # attention coefficients, should we use something like attn_probs[..., None]*(hj+ea_bonds) or
        node_feats_sum_fb = scatter_add(
            src=node_feats_fb, index=target, dim_size=num_nodes_fb
        )  # get the sum of neighboring node features
        summed_attn_weights_fbonds = (
            scatter_add(attn_probs, source, dim=0, dim_size=num_nodes_fb) if self.return_attentions else None
        )

        new_fbond_features = node_feats_sum_fb.view(num_nodes_fb, -1)
        new_fbond_features = new_fbond_features + torch.tanh(self.fedge_skip_scale) * self.fedge_embed(node_feautures_fbond_graph)

        # apply the mask for fragment bond: zero out both directed edges (rows 2k and 2k+1)
        if self.frag_bond_mask is not None:
            with torch.no_grad():
                new_fbond_features[2 * self.frag_bond_mask, :] = 0.0
                new_fbond_features[2 * self.frag_bond_mask + 1, :] = 0.0

        # get frag bond features
        edge_attr_fbond_new = new_fbond_features

        source, target = frag_index
        num_nodes_f = x_frags.size(0)
        node_features_frag_graph = x_frags.view(num_nodes_f, self.num_heads, -1)
        source_features = torch.index_select(
            input=node_features_frag_graph, index=source, dim=0
        )  # here, source and target are for the edge_index with self loops added above
        target_features = torch.index_select(
            input=node_features_frag_graph, index=target, dim=0
        )  # so, in the message below, we can use the new edge attr we found above

        edge_attr_fbond_repeat = edge_attr_fbond_new.unsqueeze(1).expand(-1, self.num_heads, -1)
        message = torch.cat(
            [target_features, edge_attr_fbond_repeat, source_features], dim=-1
        )

        attn_logits = torch.sum(message * self.f, dim=2)
        attn_logits = self.leakyrelu(attn_logits)
        attn_logits = attn_logits * (1.0 + 0.1 * torch.tanh(self.attn_logit_scale_f)).to(attn_logits.dtype)

        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_f).to(attn_logits.dtype)  # fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_features_frag_graph, index=source, dim=0)
        fedge_value = edge_attr_fbond_new.view(
            edge_attr_fbond_new.size(0), self.num_heads, self.edge_out // self.num_heads
        )
        value_f = hj + 0.5 * torch.tanh(self.edge_value_scale_f).to(hj.dtype) * fedge_value

        node_features_frag_graph = (
            attn_probs[..., None] * value_f
        )  # multiply the node features by the attention weights
        node_feats_sum_f = scatter_add(
            src=node_features_frag_graph, index=target, dim_size=num_nodes_f
        )  # nodes in the bond graph are the edges in the atom graph
        summed_attn_weights_frags = (
            scatter_add(attn_probs, source, dim=0, dim_size=num_nodes_f) if self.return_attentions else None
        )

        x_frags_new = node_feats_sum_f.view(num_nodes_f, -1)
        x_frags_new = x_frags_new + torch.tanh(self.frag_graph_skip_scale) * x_frags

        if self.return_attentions:
            return (
                x_atoms_new,
                x_frags_new,
                new_bond_features,
                new_fbond_features,
                summed_attn_weights_atoms,
                summed_attn_weights_frags,
                summed_attn_weights_bonds,
                summed_attn_weights_fbonds,
            )
        else:
            return x_atoms_new, x_frags_new, new_bond_features, new_fbond_features


class FragNet(nn.Module):

    def __init__(
        self,
        num_layer,
        drop_ratio=0.2,
        emb_dim=128,
        atom_features=167,
        frag_features=167,
        edge_features=17,
        fedge_in=6,
        fbond_edge_in=6,
        num_heads=4,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.dropout = nn.Dropout(p=drop_ratio)
        self.act = nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            FragNetLayerA(
                atom_in=atom_features,
                atom_out=emb_dim,
                frag_in=frag_features,
                frag_out=emb_dim,
                edge_in=edge_features,
                fedge_in=fedge_in,
                fbond_edge_in=fbond_edge_in,
                edge_out=emb_dim,
                num_heads=num_heads,
            )
        )

        for i in range(num_layer - 1):
            self.layers.append(
                FragNetLayerA(
                    atom_in=emb_dim,
                    atom_out=emb_dim,
                    frag_in=emb_dim,
                    frag_out=emb_dim,
                    edge_in=emb_dim,
                    edge_out=emb_dim,
                    fedge_in=emb_dim,
                    fbond_edge_in=fbond_edge_in,
                    num_heads=num_heads,
                )
            )

        self.atom_res_proj = nn.ModuleList(
            [nn.Linear(atom_features, emb_dim, bias=False)]
            + [nn.Identity() for _ in range(num_layer - 1)]
        )
        self.frag_res_proj = nn.ModuleList(
            [nn.Linear(frag_features, emb_dim, bias=False)]
            + [nn.Identity() for _ in range(num_layer - 1)]
        )
        self.edge_res_proj = nn.ModuleList(
            [nn.Linear(edge_features, emb_dim, bias=False)]
            + [nn.Identity() for _ in range(num_layer - 1)]
        )
        self.fedge_res_proj = nn.ModuleList(
            [nn.Linear(fedge_in, emb_dim, bias=False)]
            + [nn.Identity() for _ in range(num_layer - 1)]
        )
        self.atom_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(num_layer)])
        self.frag_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(num_layer)])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(num_layer)])
        self.fedge_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(num_layer)])
        self.res_gates = nn.Parameter(torch.full((num_layer, 4), 0.5))

        # Final-layer-biased output Jumping Knowledge: starts close to the existing
        # final representation, but can recover shallow/local geometry and motif
        # information after full pretraining without changing boundary shapes.
        jk_len = max(num_layer, 1)
        jk_init = torch.full((jk_len,), -2.0)
        jk_init[jk_len - 1] = 2.0
        self.jk_atom_logits = Parameter(jk_init.clone())
        self.jk_frag_logits = Parameter(jk_init.clone())
        self.jk_edge_logits = Parameter(jk_init.clone())
        self.jk_fedge_logits = Parameter(jk_init.clone())

    def forward(self, batch):

        x_atoms = batch["x_atoms"]
        edge_index = batch["edge_index"]
        frag_index = batch["frag_index"]
        x_frags = batch["x_frags"]
        edge_attr = batch["edge_attr"]
        atom_to_frag_ids = batch["atom_to_frag_ids"]
        node_feautures_bond_graph = batch["node_features_bonds"]
        edge_index_bonds_graph = batch["edge_index_bonds_graph"]
        edge_attr_bond_graph = batch["edge_attr_bonds"]
        node_feautures_fbondg = batch["node_features_fbonds"]
        edge_index_fbondg = batch["edge_index_fbonds"]
        edge_attr_fbondg = batch["edge_attr_fbonds"]

        x_atoms = self.dropout(x_atoms)
        x_frags = self.dropout(x_frags)

        # self-looped atom edge_index is identical for every layer -> compute once
        edge_index_sl, _ = add_self_loops(edge_index=edge_index)

        atom_states = []
        frag_states = []
        edge_states = []
        fedge_states = []

        old_atoms, old_frags = x_atoms, x_frags
        old_edges, old_fedges = node_feautures_bond_graph, node_feautures_fbondg

        for layer_idx, layer in enumerate(self.layers):
            edge_attr_in = edge_attr if layer_idx == 0 else old_edges

            x_atoms_update, x_frags_update, edge_update, fedge_update = layer(
                old_atoms,
                edge_index,
                edge_attr_in,
                frag_index,
                old_frags,
                atom_to_frag_ids,
                old_edges,
                edge_index_bonds_graph,
                edge_attr_bond_graph,
                old_fedges,
                edge_index_fbondg,
                edge_attr_fbondg,
                edge_index_sl=edge_index_sl,
            )

            gate = torch.sigmoid(self.res_gates[layer_idx]).to(dtype=x_atoms_update.dtype)
            x_atoms = self.act(
                self.atom_norms[layer_idx](
                    self.atom_res_proj[layer_idx](old_atoms) + gate[0] * self.dropout(x_atoms_update)
                )
            )
            x_frags = self.act(
                self.frag_norms[layer_idx](
                    self.frag_res_proj[layer_idx](old_frags) + gate[1] * self.dropout(x_frags_update)
                )
            )
            edge_features = self.act(
                self.edge_norms[layer_idx](
                    self.edge_res_proj[layer_idx](old_edges) + gate[2] * self.dropout(edge_update)
                )
            )
            fedge_features = self.act(
                self.fedge_norms[layer_idx](
                    self.fedge_res_proj[layer_idx](old_fedges) + gate[3] * self.dropout(fedge_update)
                )
            )

            atom_states.append(x_atoms)
            frag_states.append(x_frags)
            edge_states.append(edge_features)
            fedge_states.append(fedge_features)

            old_atoms, old_frags = x_atoms, x_frags
            old_edges, old_fedges = edge_features, fedge_features

        def _jk_mix(states, logits):
            stacked = torch.stack(states, dim=0)
            weights = F.softmax(logits[: len(states)].float(), dim=0).to(stacked.dtype)
            return torch.sum(stacked * weights.view(-1, 1, 1), dim=0)

        x_atoms = _jk_mix(atom_states, self.jk_atom_logits)
        x_frags = _jk_mix(frag_states, self.jk_frag_logits)
        edge_features = _jk_mix(edge_states, self.jk_edge_logits)
        fedge_features = _jk_mix(fedge_states, self.jk_fedge_logits)

        return x_atoms, x_frags, edge_features, fedge_features


class IdentityMultiAggPool(nn.Module):
    """Exact-sum readout with zero-start rich graph statistics.

    The dominant path is always additive sum pooling, matching the calibrated
    baseline at initialization.  A zero-initialized correction can learn smooth
    mean/count effects and non-additive max/attention saliency after finetuning,
    which is important for rare ADMET motifs without sacrificing step-zero scale.
    """
    def __init__(self, emb_dim=128, drop_ratio=0.0):
        super().__init__()
        hidden = max(emb_dim, 64)
        self.attn_gate = nn.Linear(emb_dim, 1)
        self.context_proj = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden),
            nn.CELU(),
            nn.Dropout(p=drop_ratio),
            nn.Linear(hidden, emb_dim),
        )
        self.count_proj = nn.Linear(1, emb_dim)

        # Uniform attention and exact sum-pooling behavior at initialization.
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.zeros_(self.attn_gate.bias)
        nn.init.zeros_(self.context_proj[-1].weight)
        nn.init.zeros_(self.context_proj[-1].bias)
        nn.init.zeros_(self.count_proj.weight)
        nn.init.zeros_(self.count_proj.bias)

    def forward(self, x, batch_index, dim_size):
        pooled_sum = scatter_add(src=x, index=batch_index, dim_size=dim_size)
        count = scatter_add(
            src=x.new_ones((x.size(0), 1)),
            index=batch_index,
            dim_size=dim_size,
        ).clamp_min(1.0)

        pooled_mean = pooled_sum / count
        pooled_max = scatter_max(src=x, index=batch_index, dim_size=dim_size)

        attn_logits = self.attn_gate(x).float()
        # Soft cap keeps learned graph attention finite under high-lr regression
        # while remaining linear near the zero-initialized uniform-attention state.
        attn_logits = 5.0 * torch.tanh(attn_logits / 5.0)
        attn = scatter_softmax(attn_logits, batch_index, dim_size=dim_size).to(x.dtype)
        pooled_attn = scatter_add(src=attn * x, index=batch_index, dim_size=dim_size)

        rich_context = self.context_proj(
            torch.cat((pooled_mean, pooled_max, pooled_attn), dim=-1)
        )
        count_corr = self.count_proj(torch.log1p(count))
        return pooled_sum + rich_context + count_corr


class FTHead3(nn.Sequential):
    def __init__(
        self,
        input_dim=128,
        h1=128,
        h2=1024,
        h3=1024,
        h4=512,
        drop_ratio=0.2,
        n_classes=1,
        act="relu",
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=drop_ratio)
        if act == "relu":
            self.activation = nn.ReLU()
        elif act == "silu":
            self.activation = nn.SiLU()
        elif act == "gelu":
            self.activation = nn.GELU()
        elif act == "celu":
            self.activation = nn.CELU()
        elif act == "selu":
            self.activation = nn.SELU()
        elif act == "rrelu":
            self.activation = nn.RReLU()
        elif act == "relu6":
            self.activation = nn.ReLU6()
        elif act == "prelu":
            self.activation = nn.PReLU()
        elif act == "leakyrelu":
            self.activation = nn.LeakyReLU()

        self.hidden_dims = [h1, h2, h3, h4]
        layer_size = len(self.hidden_dims) + 1
        dims = [input_dim * 2] + self.hidden_dims + [n_classes]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]
        )

    def forward(self, enc):

        for i in range(0, len(self.predictor) - 1):
            enc = self.activation(self.dropout(self.predictor[i](enc)))
        out = self.predictor[-1](enc)

        return out


class FragNetFineTune(nn.Module):
    """
    Trainer class for finetuning.
    """

    def __init__(
        self,
        n_classes=1,
        atom_features=167,
        frag_features=167,
        edge_features=17,
        num_layer=4,
        num_heads=4,
        drop_ratio=0.15,
        h1=256,
        h2=256,
        h3=256,
        h4=256,
        act="celu",
        emb_dim=128,
        fthead="FTHead3",
    ):
        super().__init__()

        self.pretrain = FragNet(
            num_layer=num_layer,
            drop_ratio=drop_ratio,
            num_heads=num_heads,
            emb_dim=emb_dim,
            atom_features=atom_features,
            frag_features=frag_features,
            edge_features=edge_features,
        )

        # Restore the stronger sum-preserving rich readout.  Mean/max/attention
        # and count corrections are zero-start, so finetuning begins as exact
        # atom-sum + fragment-sum pooling but can recover salient motif statistics.
        self.atom_pool = IdentityMultiAggPool(emb_dim=emb_dim, drop_ratio=drop_ratio)
        self.frag_pool = IdentityMultiAggPool(emb_dim=emb_dim, drop_ratio=drop_ratio)

        if fthead == "FTHead1":
            self.fthead = FTHead1(n_classes=n_classes)
        elif fthead == "FTHead2":
            print("using FTHead2")
            self.fthead = FTHead2(n_classes=n_classes)
        elif fthead == "FTHead3":
            print("using FTHead3")
            self.fthead = FTHead3(
                n_classes=n_classes,
                input_dim=emb_dim,
                h1=h1,
                h2=h2,
                h3=h3,
                h4=h4,
                drop_ratio=drop_ratio,
                act=act,
            )

        elif fthead == "FTHead4":
            print("using FTHead4")
            self.fthead = FTHead4(
                n_classes=n_classes, h1=h1, drop_ratio=drop_ratio, act=act
            )

    def forward(self, batch):

        x_atoms, x_frags, x_edge, x_fedge = self.pretrain(batch)

        n_graphs = batch["y"].shape[0]  # static dim_size for pooling (compile)
        x_atoms_pooled = self.atom_pool(x_atoms, batch["batch"], n_graphs)
        x_frags_pooled = self.frag_pool(x_frags, batch["frag_batch"], n_graphs)

        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.fthead(cat)

        return x

