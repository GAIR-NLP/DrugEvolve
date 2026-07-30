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
    if dim_size is None:
        dim_size = int(index.max()) + 1   # eager fallback (not used on the compile path)
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    return out.scatter_add_(0, idx.expand_as(src), src)


def scatter_softmax(src, index, dim=0, dim_size=None):
    """Numerically-stable softmax over groups defined by `index` along dim 0."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    mx = src.new_full((dim_size,) + tuple(src.shape[1:]), float("-inf"))
    mx = mx.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
    e = (src - mx.gather(0, idx)).exp()
    s = scatter_add(e, index, dim_size=dim_size).gather(0, idx)
    return e / s


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
        self.atom_embed = nn.Linear(atom_in, atom_out, bias=True)
        self.frag_embed = nn.Linear(frag_in, frag_out)
        self.edge_embed = nn.Linear(edge_in, edge_out)
        self.bond_edge_embed = nn.Linear(edge_in, edge_out)

        self.frag_message_mlp = nn.Linear(atom_out * 2, atom_out)
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

        nn.init.xavier_uniform_(self.projection_b.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_b.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.f.data, gain=1.414)
        nn.init.xavier_uniform_(self.f_a_b.data, gain=1.414)

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
        ea_bonds = edge_attr_bond_graph.repeat(self.num_heads, 1, 1).permute(1, 0, 2)
        num_nodes_b = node_feautures_bond_graph.size(0)
        node_feats_b = self.projection_b(node_feautures_bond_graph)
        node_feats_b = node_feats_b.view(num_nodes_b, self.num_heads, -1)

        # select source and target features
        source_features = torch.index_select(input=node_feats_b, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_b, index=target, dim=0)
        message = torch.cat([target_features, ea_bonds, source_features], dim=-1)

        attn_logits = torch.sum(message * self.a_b, dim=2)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_b).to(attn_logits.dtype)  # eij; fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_feats_b, index=source, dim=0)  # select hj

        node_feats_b = (
            attn_probs[..., None] * hj
        )  # multipy the attention coefficients corresponding to (neighbors - target atom) with the attention
        # coefficients corresponding to the neighbors. [since we are considering edge features when finding the
        # attention coefficients, should we use something like attn_probs[..., None]*(hj+ea_bonds) or

        node_feats_sum_b = scatter_add(
            src=node_feats_b, index=target, dim_size=num_nodes_b
        )  # get the sum of neighboring node features
        # summed attn weights are only returned for viz (return_attentions); skip in training
        summed_attn_weights_bonds = (
            scatter_add(attn_probs, source, dim=0) if self.return_attentions else None
        )

        new_bond_features = node_feats_sum_b.view(num_nodes_b, -1)

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

        self_loop_attr = new_bond_features.new_zeros(x_atoms.size(0), self.edge_out)
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

        edge_attr_repeat = edge_attr.repeat(self.num_heads, 1, 1).permute(1, 0, 2)
        message = torch.cat(
            [target_features, edge_attr_repeat, source_features], dim=-1
        )

        attn_logits = torch.sum(message * self.a, dim=2)  # NOTE: replace a by self.a
        attn_logits = self.leakyrelu(attn_logits)
        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_a).to(attn_logits.dtype)  # fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_features_atom_graph, index=source, dim=0)

        node_features_atom_graph = (
            attn_probs[..., None] * hj
        )  # multiply the node features by the attention weights
        node_feats_sum_a = scatter_add(
            src=node_features_atom_graph, index=target, dim_size=num_nodes_a
        )  # nodes in the bond graph are the edges in the atom graph
        summed_attn_weights_atoms = (
            scatter_add(attn_probs, source, dim=0) if self.return_attentions else None
        )

        # new atom features
        x_atoms_new = node_feats_sum_a.view(num_nodes_a, -1)

        # apply the mask for atom
        if self.atom_mask_individual is not None:
            # print(x_atoms_new, x_atoms_new.shape)
            with torch.no_grad():
                print('applying atom mask')
                x_atoms_new[self.atom_mask_individual, :] = 0.0
            print(x_atoms_new, x_atoms_new.shape)
        # new fragment features
        x_frags = scatter_add(src=x_atoms_new, index=atom_to_frag_ids, dim_size=num_frags)



        # get frag bond features.
        target, source = edge_index_fbond_graph

        
        edge_attr_fbond_graph = self.edge_attr_fbond_embed(edge_attr_fbond_graph)
                
                
        ea_fbonds = edge_attr_fbond_graph.repeat(self.num_heads, 1, 1).permute(1, 0, 2)
        num_nodes_fb = node_feautures_fbond_graph.size(0)
        node_feats_fb = self.projection_fb(node_feautures_fbond_graph)
        node_feats_fb = node_feats_fb.view(num_nodes_fb, self.num_heads, -1)

        source_features = torch.index_select(input=node_feats_fb, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_fb, index=target, dim=0)
        message = torch.cat([target_features, ea_fbonds, source_features], dim=-1)

        attn_logits = torch.sum(message * self.f_a_b, dim=2)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_fb).to(attn_logits.dtype)  # eij; fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_feats_fb, index=source, dim=0)  # select hj

        node_feats_fb = (
            attn_probs[..., None] * hj
        )  # multipy the attention coefficients corresponding to (neighbors - target atom) with the attention
        # coefficients corresponding to the neighbors. [since we are considering edge features when finding the
        # attention coefficients, should we use something like attn_probs[..., None]*(hj+ea_bonds) or
        node_feats_sum_fb = scatter_add(
            src=node_feats_fb, index=target, dim_size=num_nodes_fb
        )  # get the sum of neighboring node features
        summed_attn_weights_fbonds = (
            scatter_add(attn_probs, source, dim=0) if self.return_attentions else None
        )

        new_fbond_features = node_feats_sum_fb.view(num_nodes_fb, -1)

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

        edge_attr_fbond_repeat = edge_attr_fbond_new.repeat(
            self.num_heads, 1, 1
        ).permute(1, 0, 2)
        message = torch.cat(
            [target_features, edge_attr_fbond_repeat, source_features], dim=-1
        )

        attn_logits = torch.sum(message * self.f, dim=2)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = scatter_softmax(attn_logits.float(), target, dim_size=num_nodes_f).to(attn_logits.dtype)  # fp32 softmax for AMP numerical stability
        hj = torch.index_select(input=node_features_frag_graph, index=source, dim=0)

        node_features_frag_graph = (
            attn_probs[..., None] * hj
        )  # multiply the node features by the attention weights
        node_feats_sum_f = scatter_add(
            src=node_features_frag_graph, index=target, dim_size=num_nodes_f
        )  # nodes in the bond graph are the edges in the atom graph
        summed_attn_weights_frags = (
            scatter_add(attn_probs, source, dim=0) if self.return_attentions else None
        )

        x_frags_new = node_feats_sum_f.view(num_nodes_f, -1)

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

        x_atoms, x_frags, edge_features, fedge_features = self.layers[0](
            x_atoms,
            edge_index,
            edge_attr,
            frag_index,
            x_frags,
            atom_to_frag_ids,
            node_feautures_bond_graph,
            edge_index_bonds_graph,
            edge_attr_bond_graph,
            node_feautures_fbondg,
            edge_index_fbondg,
            edge_attr_fbondg,
            edge_index_sl=edge_index_sl,
        )

        x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(
            self.dropout(x_frags)
        )
        edge_features = self.act(self.dropout(edge_features))
        fedge_features = self.act(self.dropout(fedge_features))

        for layer in self.layers[1:]:
            x_atoms, x_frags, edge_features, fedge_features = layer(
                x_atoms,
                edge_index,
                edge_features,
                frag_index,
                x_frags,
                atom_to_frag_ids,
                edge_features,
                edge_index_bonds_graph,
                edge_attr_bond_graph,
                fedge_features,
                edge_index_fbondg,
                edge_attr_fbondg,
                edge_index_sl=edge_index_sl,
            )

            x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(
                self.dropout(x_frags)
            )
            edge_features = self.act(self.dropout(edge_features))
            fedge_features = self.act(self.dropout(fedge_features))

        return x_atoms, x_frags, edge_features, fedge_features


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
        x_frags_pooled = scatter_add(src=x_frags, index=batch["frag_batch"], dim_size=n_graphs)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch["batch"], dim_size=n_graphs)

        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.fthead(cat)

        return x

