import math

import dgl
import numpy as np
import ot
import torch
from torch import nn
from dgl import function as fn


# Feature dimensions are part of the graph-cache contract.  Keeping them here
# makes this model independent of commons.process_mols while staying
# byte-for-byte compatible with the cached graph feature columns.
lig_feature_dims = ([119, 4, 12, 12, 8, 10, 6, 6, 2, 8, 2, 2, 2, 2, 2, 2], 1)
rec_atom_feature_dims = ([38, 119, 23, 38], 2)
rec_residue_feature_dims = ([38], 2)

# Evolvable architecture configuration. Parameters tied to the copied graph
# cache (notably use_rec_atoms=False) must remain compatible with that cache.
MODEL_CONFIG = {
    "geometry_reg_step_size": 0.001,
    "geometry_regularization": True,
    "use_evolved_lig": True,
    "standard_norm_order": True,
    "unnormalized_kpt_weights": False,
    "lig_evolve": True,
    "rec_evolve": True,
    "rec_no_softmax": False,
    "lig_no_softmax": False,
    "centroid_keypts_construction_rec": False,
    "centroid_keypts_construction_lig": False,
    "centroid_keypts_construction": False,
    "move_keypts_back": True,
    "normalize_Z_rec_directions": False,
    "normalize_Z_lig_directions": False,
    "n_lays": 8,
    "debug": False,
    "use_rec_atoms": False,
    "shared_layers": False,
    "noise_decay_rate": 0.5,
    "noise_initial": 1,
    "use_edge_features_in_gmn": True,
    "use_mean_node_features": True,
    "residue_emb_dim": 64,
    "iegmn_lay_hid_dim": 64,
    "num_att_heads": 30,
    "dropout": 0.1,
    "nonlin": "lkyrelu",
    "leakyrelu_neg_slope": 1.0e-2,
    "cross_msgs": True,
    "layer_norm": "BN",
    "layer_norm_coords": "0",
    "final_h_layer_norm": "0",
    "pre_crossmsg_norm_type": "0",
    "post_crossmsg_norm_type": "0",
    "use_dist_in_layers": True,
    "skip_weight_h": 0.5,
    "x_connection_init": 0.25,
    "random_vec_dim": 0,
    "random_vec_std": 1,
    "use_scalar_features": False,
    "num_lig_feats": None,
    "normalize_coordinate_update": True,
    "rec_square_distance_scale": 10,
}


class AtomEncoder(nn.Module):
    """Encode categorical graph features and optional scalar features."""

    def __init__(self, emb_dim, feature_dims, use_scalar_feat=True, n_feats_to_use=None):
        super().__init__()
        self.use_scalar_feat = use_scalar_feat
        self.n_feats_to_use = n_feats_to_use
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        self.atom_embedding_list = nn.ModuleList()

        for index, dimension in enumerate(feature_dims[0]):
            embedding = nn.Embedding(dimension, emb_dim)
            nn.init.xavier_uniform_(embedding.weight.data)
            self.atom_embedding_list.append(embedding)
            if index + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0:
            self.linear = nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, features):
        expected = self.num_categorical_features + self.num_scalar_features
        if features.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} atom features, received {features.shape[1]}"
            )

        encoded = 0
        for index in range(self.num_categorical_features):
            encoded = encoded + self.atom_embedding_list[index](features[:, index].long())
            if index + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0 and self.use_scalar_feat:
            encoded = encoded + self.linear(features[:, self.num_categorical_features:])
        return encoded


class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h, node_type):
        graph_size = g.batch_num_nodes(node_type) if self.is_node else g.batch_num_edges(node_type)
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    elif type == 'relu':
        return nn.ReLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def get_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return GraphNorm(dim)
    else:
        assert layer_norm_type == '0' or layer_norm_type == 0
        return nn.Identity()


def apply_norm(g, h, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h)
    return norm_layer(h)


class CoordsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coords):
        norm = coords.norm(dim=-1, keepdim=True)
        normed_coords = coords / norm.clamp(min=self.eps)
        return normed_coords * self.scale


def cross_attention(queries, keys, values, mask, cross_msgs):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    Args:
      queries: NxD float tensor --> queries
      keys: MxD float tensor --> keys
      values: Mxd
      mask: NxM
    Returns:
      attention_x: Nxd float tensor.
    """
    if not cross_msgs:
        return queries * 0.
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0)) - 1000. * (1. - mask)
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    attention_x = torch.mm(a_x, values)  # (N,d)
    return attention_x


def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = ligand_batch_num_nodes.sum()
    cols = receptor_batch_num_nodes.sum()
    mask = torch.zeros(rows, cols, device=device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask


class IEGMN_Layer(nn.Module):
    def __init__(
            self,
            orig_h_feats_dim,
            h_feats_dim,  # in dim of h
            out_feats_dim,  # out dim of h
            lig_input_edge_feats_dim,
            rec_input_edge_feats_dim,
            nonlin,
            cross_msgs,
            layer_norm,
            layer_norm_coords,
            final_h_layer_norm,
            use_dist_in_layers,
            skip_weight_h,
            x_connection_init,
            leakyrelu_neg_slope,
            debug,
            device,
            dropout,
            save_trajectories=False,
            rec_square_distance_scale=1,
            standard_norm_order=False,
            normalize_coordinate_update=False,
            lig_evolve=True,
            rec_evolve=True,
            fine_tune=False,
            geometry_regularization=False,
            pre_crossmsg_norm_type=0,
            post_crossmsg_norm_type=0,
            norm_cross_coords_update= False,
            loss_geometry_regularization = False,
            geom_reg_steps= 1,
        geometry_reg_step_size=0.1
    ):

        super(IEGMN_Layer, self).__init__()

        self.fine_tune = fine_tune
        self.cross_msgs = cross_msgs
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.rec_square_distance_scale = rec_square_distance_scale
        self.geometry_reg_step_size = geometry_reg_step_size
        self.norm_cross_coords_update =norm_cross_coords_update
        self.loss_geometry_regularization = loss_geometry_regularization

        self.debug = debug
        self.device = device
        self.lig_evolve = lig_evolve
        self.rec_evolve = rec_evolve
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories

        # EDGES
        lig_edge_mlp_input_dim = (h_feats_dim * 2) + lig_input_edge_feats_dim
        if self.use_dist_in_layers and self.lig_evolve:
            lig_edge_mlp_input_dim += len(self.all_sigmas_dist)
        self.lig_edge_mlp = nn.Sequential(
            nn.Linear(lig_edge_mlp_input_dim, self.out_feats_dim),
            get_layer_norm(layer_norm, self.out_feats_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            get_layer_norm(layer_norm, self.out_feats_dim),
        )
        rec_edge_mlp_input_dim = (h_feats_dim * 2) + rec_input_edge_feats_dim
        if self.use_dist_in_layers and self.rec_evolve:
            rec_edge_mlp_input_dim += len(self.all_sigmas_dist)
        self.rec_edge_mlp = nn.Sequential(
            nn.Linear(rec_edge_mlp_input_dim, self.out_feats_dim),
            get_layer_norm(layer_norm, self.out_feats_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            get_layer_norm(layer_norm, self.out_feats_dim),
        )

        # NODES
        self.node_norm = nn.Identity()  # nn.LayerNorm(h_feats_dim)

        # normalization of x_i - x_j is not currently used
        if self.normalize_coordinate_update:
            self.lig_coords_norm = CoordsNorm(scale_init=1e-2)
            self.rec_coords_norm = CoordsNorm(scale_init=1e-2)

        self.att_mlp_Q_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        self.node_mlp_lig = nn.Sequential(
            nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
            get_layer_norm(layer_norm, h_feats_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(h_feats_dim, out_feats_dim),
            get_layer_norm(layer_norm, out_feats_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
            get_layer_norm(layer_norm, h_feats_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(h_feats_dim, out_feats_dim),
            get_layer_norm(layer_norm, out_feats_dim),
        )

        self.final_h_layernorm_layer_lig = get_norm(self.final_h_layer_norm, out_feats_dim)
        self.final_h_layernorm_layer = get_norm(self.final_h_layer_norm, out_feats_dim)

        self.pre_crossmsg_norm_lig = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)
        self.pre_crossmsg_norm_rec = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)

        self.post_crossmsg_norm_lig = get_norm(self.post_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_rec = get_norm(self.post_crossmsg_norm_type, h_feats_dim)

        self.coords_mlp_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            get_layer_norm(layer_norm_coords, self.out_feats_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(self.out_feats_dim, 1)
        )
        self.coords_mlp_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            get_layer_norm(layer_norm_coords, self.out_feats_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(self.out_feats_dim, 1)
        )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges_lig(self, edges):
        if self.use_dist_in_layers and self.lig_evolve:
            x_rel_mag = edges.data['x_rel'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.lig_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.lig_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def apply_edges_rec(self, edges):
        if self.use_dist_in_layers and self.rec_evolve:
            squared_distance = torch.sum(edges.data['x_rel'] ** 2, dim=1, keepdim=True)
            # divide square distance by 10 to have a nicer separation instead of many 0.00000
            x_rel_mag = torch.cat([torch.exp(-(squared_distance / self.rec_square_distance_scale) / sigma) for sigma in
                                   self.all_sigmas_dist], dim=-1)
            return {'msg': self.rec_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.rec_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def update_x_moment_lig(self, edges):
        edge_coef_ligand = self.coords_mlp_lig(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.lig_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_ligand}  # (x_i - x_j) * \phi^x(m_{i->j})

    def update_x_moment_rec(self, edges):
        edge_coef_rec = self.coords_mlp_rec(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.rec_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_rec}  # (x_i - x_j) * \phi^x(m_{i->j})

    def forward(self, lig_graph, rec_graph, coords_lig, h_feats_lig, original_ligand_node_features, orig_coords_lig,
                coords_rec, h_feats_rec, original_receptor_node_features, orig_coords_rec, mask, geometry_graph):
        with lig_graph.local_scope() and rec_graph.local_scope():
            lig_graph.ndata['x_now'] = coords_lig
            rec_graph.ndata['x_now'] = coords_rec
            lig_graph.ndata['feat'] = h_feats_lig  # first time set here
            rec_graph.ndata['feat'] = h_feats_rec

            if self.lig_evolve:
                lig_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))  # x_i - x_j
            if self.rec_evolve:
                rec_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))

            lig_graph.apply_edges(self.apply_edges_lig)  ## i->j edge:  [h_i h_j]
            rec_graph.apply_edges(self.apply_edges_rec)

            h_feats_lig_norm = apply_norm(lig_graph, h_feats_lig, self.final_h_layer_norm, self.final_h_layernorm_layer)
            h_feats_rec_norm = apply_norm(rec_graph, h_feats_rec, self.final_h_layer_norm, self.final_h_layernorm_layer)
            cross_attention_lig_feat = cross_attention(self.att_mlp_Q_lig(h_feats_lig_norm),
                                                       self.att_mlp_K(h_feats_rec_norm),
                                                       self.att_mlp_V(h_feats_rec_norm), mask, self.cross_msgs)
            cross_attention_rec_feat = cross_attention(self.att_mlp_Q(h_feats_rec_norm),
                                                       self.att_mlp_K_lig(h_feats_lig_norm),
                                                       self.att_mlp_V_lig(h_feats_lig_norm), mask.transpose(0, 1),
                                                       self.cross_msgs)
            cross_attention_lig_feat = apply_norm(lig_graph, cross_attention_lig_feat, self.final_h_layer_norm,
                                                  self.final_h_layernorm_layer)
            cross_attention_rec_feat = apply_norm(rec_graph, cross_attention_rec_feat, self.final_h_layer_norm,
                                                  self.final_h_layernorm_layer)

            if self.lig_evolve:
                lig_graph.update_all(self.update_x_moment_lig, fn.mean('m', 'x_update'))
                # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
                x_evolved_lig = self.x_connection_init * orig_coords_lig + (1. - self.x_connection_init) * \
                                lig_graph.ndata['x_now'] + lig_graph.ndata['x_update']
            else:
                x_evolved_lig = coords_lig

            if self.rec_evolve:
                rec_graph.update_all(self.update_x_moment_rec, fn.mean('m', 'x_update'))
                x_evolved_rec = self.x_connection_init * orig_coords_rec + (1. - self.x_connection_init) * \
                                rec_graph.ndata['x_now'] + rec_graph.ndata['x_update']
            else:
                x_evolved_rec = coords_rec

            lig_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))
            rec_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))

            geom_loss = 0
            if self.geometry_regularization:
                src, dst = geometry_graph.edges()
                src = src.long()
                dst = dst.long()
                for step in range(self.geom_reg_steps):
                    d_squared = torch.sum((x_evolved_lig[src] - x_evolved_lig[dst]) ** 2, dim=1)
                    Loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
                    grad_d_squared = 2 * (x_evolved_lig[src] - x_evolved_lig[dst])
                    geometry_graph.edata['partial_grads'] = 2 * (d_squared - geometry_graph.edata['feat'] ** 2)[:,None] * grad_d_squared
                    geometry_graph.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),
                                              fn.sum('partial_grads_msg', 'grad_x_evolved'))
                    grad_x_evolved = geometry_graph.ndata['grad_x_evolved']
                    x_evolved_lig = x_evolved_lig + self.geometry_reg_step_size * grad_x_evolved

            input_node_upd_ligand = torch.cat((self.node_norm(lig_graph.ndata['feat']),
                                               lig_graph.ndata['aggr_msg'],
                                               cross_attention_lig_feat,
                                               original_ligand_node_features), dim=-1)

            input_node_upd_receptor = torch.cat((self.node_norm(rec_graph.ndata['feat']),
                                                 rec_graph.ndata['aggr_msg'],
                                                 cross_attention_rec_feat,
                                                 original_receptor_node_features), dim=-1)

            # Skip connections
            if self.h_feats_dim == self.out_feats_dim:
                node_upd_ligand = self.skip_weight_h * self.node_mlp_lig(input_node_upd_ligand) + (
                        1. - self.skip_weight_h) * h_feats_lig
                node_upd_receptor = self.skip_weight_h * self.node_mlp(input_node_upd_receptor) + (
                        1. - self.skip_weight_h) * h_feats_rec
            else:
                node_upd_ligand = self.node_mlp_lig(input_node_upd_ligand)
                node_upd_receptor = self.node_mlp(input_node_upd_receptor)

            node_upd_ligand = apply_norm(lig_graph, node_upd_ligand, self.final_h_layer_norm,
                                         self.final_h_layernorm_layer_lig)
            node_upd_receptor = apply_norm(rec_graph, node_upd_receptor,
                                           self.final_h_layer_norm, self.final_h_layernorm_layer)
            return x_evolved_lig, node_upd_ligand, x_evolved_rec, node_upd_receptor, geom_loss

    def __repr__(self):
        return "IEGMN Layer " + str(self.__dict__)


# =================================================================================================================
class IEGMN(nn.Module):

    def __init__(self, n_lays, debug, device, use_rec_atoms, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, residue_emb_dim, iegmn_lay_hid_dim, num_att_heads,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_lig_feats=None, move_keypts_back=False, normalize_Z_lig_directions=False,
                 unnormalized_kpt_weights=False, centroid_keypts_construction_rec=False,
                 centroid_keypts_construction_lig=False, rec_no_softmax=False, lig_no_softmax=False,
                 normalize_Z_rec_directions=False,
                 centroid_keypts_construction=False, evolve_only=False, separate_lig=False, save_trajectories=False, **kwargs):
        super(IEGMN, self).__init__()
        self.debug = debug
        self.cross_msgs = cross_msgs
        self.device = device
        self.save_trajectories = save_trajectories
        self.unnormalized_kpt_weights = unnormalized_kpt_weights
        self.separate_lig =separate_lig
        self.use_rec_atoms = use_rec_atoms
        self.noise_decay_rate = noise_decay_rate
        self.noise_initial = noise_initial
        self.use_edge_features_in_gmn = use_edge_features_in_gmn
        self.use_mean_node_features = use_mean_node_features
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        self.move_keypts_back = move_keypts_back
        self.normalize_Z_lig_directions = normalize_Z_lig_directions
        self.centroid_keypts_construction = centroid_keypts_construction
        self.centroid_keypts_construction_rec = centroid_keypts_construction_rec
        self.centroid_keypts_construction_lig = centroid_keypts_construction_lig
        self.normalize_Z_rec_directions = normalize_Z_rec_directions
        self.rec_no_softmax = rec_no_softmax
        self.lig_no_softmax = lig_no_softmax
        self.evolve_only = evolve_only

        self.lig_atom_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                             feature_dims=lig_feature_dims, use_scalar_feat=use_scalar_features,
                                             n_feats_to_use=num_lig_feats)
        self.rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                        feature_dims=rec_residue_feature_dims, use_scalar_feat=use_scalar_features)

        input_node_feats_dim = residue_emb_dim
        if self.use_mean_node_features:
            input_node_feats_dim += 5  ### Additional features from mu_r_norm
        self.iegmn_layers = nn.ModuleList()
        self.iegmn_layers.append(
            IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                        h_feats_dim=input_node_feats_dim,
                        out_feats_dim=iegmn_lay_hid_dim,
                        nonlin=nonlin,
                        cross_msgs=self.cross_msgs,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        dropout=dropout,
                        save_trajectories=save_trajectories,**kwargs))

        for layer_idx in range(1, n_lays):
            self.iegmn_layers.append(
                IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                            h_feats_dim=iegmn_lay_hid_dim,
                            out_feats_dim=iegmn_lay_hid_dim,
                            cross_msgs=self.cross_msgs,
                            nonlin=nonlin,
                            leakyrelu_neg_slope=leakyrelu_neg_slope,
                            debug=False,
                            device=device,
                            dropout=dropout,
                            save_trajectories=save_trajectories,**kwargs))
        # Attention layers
        self.num_att_heads = num_att_heads
        self.out_feats_dim = iegmn_lay_hid_dim
        self.keypts_attention_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_queries_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_attention_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_queries_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))

        self.h_mean_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.h_mean_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )

        # self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, lig_graph, rec_graph, geometry_graph, complex_names, epoch):
        orig_coords_lig = lig_graph.ndata['new_x']
        orig_coords_rec = rec_graph.ndata['x']

        coords_lig = lig_graph.ndata['new_x']
        coords_rec = rec_graph.ndata['x']

        h_feats_lig = self.lig_atom_embedder(lig_graph.ndata['feat'])
        h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])  # (N_res, emb_dim)

        # random_vec_dim == 0: the original random-vector concat is a no-op, omitted.

        # random noise:
        if self.noise_initial > 0:
            noise_level = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
            h_feats_lig = h_feats_lig + noise_level * torch.randn_like(h_feats_lig)
            h_feats_rec = h_feats_rec + noise_level * torch.randn_like(h_feats_rec)
            coords_lig = coords_lig + noise_level * torch.randn_like(coords_lig)
            coords_rec = coords_rec + noise_level * torch.randn_like(coords_rec)

        if self.use_mean_node_features:
            h_feats_lig = torch.cat([h_feats_lig, torch.log(lig_graph.ndata['mu_r_norm'])],
                                    dim=1)
            h_feats_rec = torch.cat(
                [h_feats_rec, torch.log(rec_graph.ndata['mu_r_norm'])], dim=1)

        original_ligand_node_features = h_feats_lig
        original_receptor_node_features = h_feats_rec
        lig_graph.edata['feat'] *= self.use_edge_features_in_gmn
        rec_graph.edata['feat'] *= self.use_edge_features_in_gmn

        mask = None
        if self.cross_msgs:
            mask = get_mask(lig_graph.batch_num_nodes(), rec_graph.batch_num_nodes(), self.device)
        geom_losses = 0
        for i, layer in enumerate(self.iegmn_layers):
            coords_lig, \
            h_feats_lig, \
            coords_rec, \
            h_feats_rec, geom_loss = layer(lig_graph=lig_graph,
                                rec_graph=rec_graph,
                                coords_lig=coords_lig,
                                h_feats_lig=h_feats_lig,
                                original_ligand_node_features=original_ligand_node_features,
                                orig_coords_lig=orig_coords_lig,
                                coords_rec=coords_rec,
                                h_feats_rec=h_feats_rec,
                                original_receptor_node_features=original_receptor_node_features,
                                orig_coords_rec=orig_coords_rec,
                                mask=mask,
                                geometry_graph=geometry_graph
                                )
            geom_losses = geom_losses + geom_loss

        rotations = []
        translations = []
        recs_keypts = []
        ligs_keypts = []
        ligs_evolved = []
        rec_keypts_means = []
        lig_keypts_means = []
        kabsch_covariances = []
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        recs_node_idx = torch.cumsum(rec_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx.insert(0, 0)

        for idx in range(len(ligs_node_idx) - 1):
            lig_start = ligs_node_idx[idx]
            lig_end = ligs_node_idx[idx + 1]
            rec_start = recs_node_idx[idx]
            rec_end = recs_node_idx[idx + 1]
            # Get H vectors

            rec_feats = h_feats_rec[rec_start:rec_end]  # (m, d)
            rec_feats_mean = torch.mean(self.h_mean_rec(rec_feats), dim=0, keepdim=True)  # (1, d)
            lig_feats = h_feats_lig[lig_start:lig_end]  # (n, d)
            lig_feats_mean = torch.mean(self.h_mean_lig(lig_feats), dim=0, keepdim=True)  # (1, d)

            d = lig_feats.shape[1]
            assert d == self.out_feats_dim
            # Z coordinates
            Z_rec_coords = coords_rec[rec_start:rec_end]
            Z_lig_coords = coords_lig[lig_start:lig_end]

            # Att weights to compute the receptor centroid. They query is the average_h_ligand. Keys are each h_receptor_j
            att_weights_rec = (self.keypts_attention_rec(rec_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @
                               self.keypts_queries_rec(lig_feats_mean).view(1, self.num_att_heads, d).transpose(0,
                                                                                                                1).transpose(
                                   1, 2) /
                               math.sqrt(d)).view(self.num_att_heads, -1)
            att_weights_rec = torch.softmax(att_weights_rec, dim=1)
            att_weights_rec = att_weights_rec.view(self.num_att_heads, -1)

            # Att weights to compute the ligand centroid. They query is the average_h_receptor. Keys are each h_ligand_i
            att_weights_lig = (self.keypts_attention_lig(lig_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @
                               self.keypts_queries_lig(rec_feats_mean).view(1, self.num_att_heads, d).transpose(0,
                                                                                                                1).transpose(
                                   1, 2) /
                               math.sqrt(d))
            att_weights_lig = torch.softmax(att_weights_lig, dim=1)
            att_weights_lig = att_weights_lig.view(self.num_att_heads, -1)

            rec_keypts = att_weights_rec @ Z_rec_coords  # K_heads, 3
            lig_keypts = att_weights_lig @ Z_lig_coords  # K_heads, 3

            recs_keypts.append(rec_keypts)
            ligs_keypts.append(lig_keypts)
            rec_keypts_mean = rec_keypts.mean(dim=0, keepdim=True)  # (1,3)
            lig_keypts_mean = lig_keypts.mean(dim=0, keepdim=True)  # (1,3)
            A = (rec_keypts - rec_keypts_mean).transpose(0, 1) @ (lig_keypts - lig_keypts_mean) / float(
                self.num_att_heads)  # 3, 3
            rec_keypts_means.append(rec_keypts_mean)
            lig_keypts_means.append(lig_keypts_mean)
            kabsch_covariances.append(A)
            ligs_evolved.append(Z_lig_coords)

        lig_keypts_batch = torch.stack(ligs_keypts)
        covariances = torch.stack(kabsch_covariances)
        assert torch.isfinite(lig_keypts_batch).all()
        assert torch.isfinite(covariances).all()

        U_batch, S_batch, Vt_batch = torch.linalg.svd(covariances)
        squared_singular_values = S_batch ** 2
        singular_value_differences = torch.abs(
            squared_singular_values.unsqueeze(2) - squared_singular_values.unsqueeze(1) +
            torch.eye(3, device=self.device).unsqueeze(0))
        unstable = ((S_batch.min(dim=1).values < 1e-3) |
                    (singular_value_differences.amin(dim=(1, 2)) < 1e-2)).tolist()

        final_covariances = []
        final_u = []
        final_vt = []
        for idx, is_unstable in enumerate(unstable):
            A = covariances[idx]
            U = U_batch[idx]
            S = S_batch[idx]
            Vt = Vt_batch[idx]
            if is_unstable:
                num_it = 0
                while torch.min(S) < 1e-3 or torch.min(
                        torch.abs((S ** 2).view(1, 3) - (S ** 2).view(3, 1) +
                                  torch.eye(3).to(self.device))) < 1e-2:
                    A = A + torch.rand(3, 3).to(self.device) * torch.eye(3).to(self.device)
                    U, S, Vt = torch.linalg.svd(A)
                    num_it += 1
                    if num_it > 10: raise Exception('SVD was consitantly unstable')
            final_covariances.append(A)
            final_u.append(U)
            final_vt.append(Vt)

        final_covariances = torch.stack(final_covariances)
        U_batch = torch.stack(final_u)
        Vt_batch = torch.stack(final_vt)
        corr_mats = torch.eye(3, dtype=final_covariances.dtype, device=self.device) \
            .unsqueeze(0).repeat(len(kabsch_covariances), 1, 1)
        corr_mats[:, 2, 2] = torch.sign(torch.linalg.det(final_covariances)).detach()
        rotations_batch = (U_batch @ corr_mats) @ Vt_batch
        rec_means = torch.cat(rec_keypts_means, dim=0)
        lig_means = torch.cat(lig_keypts_means, dim=0)
        translations_batch = rec_means - torch.bmm(rotations_batch, lig_means.unsqueeze(2)).squeeze(2)

        rotations = list(rotations_batch.unbind(0))
        translations = [translation.unsqueeze(0) for translation in translations_batch.unbind(0)]

        return [rotations, translations, ligs_keypts, recs_keypts, ligs_evolved, geom_losses]

    def __repr__(self):
        return "IEGMN " + str(self.__dict__)


# =================================================================================================================


class EquiBind(nn.Module):

    def __init__(self, device='cuda:0', debug=False, use_evolved_lig=False, evolve_only=False, **kwargs):
        super(EquiBind, self).__init__()
        self.use_evolved_lig = use_evolved_lig
        self.device = device
        self.iegmn = IEGMN(device=self.device, debug=debug, **kwargs)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, lig_graph, rec_graph, geometry_graph=None, complex_names=None, epoch=0):
        predicted_ligs_coords_list = []
        outputs = self.iegmn(lig_graph, rec_graph, geometry_graph, complex_names, epoch)
        evolved_ligs = outputs[4]
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        for idx in range(len(ligs_node_idx) - 1):
            start = ligs_node_idx[idx]
            end = ligs_node_idx[idx + 1]
            orig_coords_lig = lig_graph.ndata['new_x'][start:end]
            rotation = outputs[0][idx]
            translation = outputs[1][idx]
            assert translation.shape[0] == 1 and translation.shape[1] == 3

            if self.use_evolved_lig:
                predicted_coords = (rotation @ evolved_ligs[idx].t()).t() + translation  # (n,3)
            else:
                predicted_coords = (rotation @ orig_coords_lig.t()).t() + translation  # (n,3)
            predicted_ligs_coords_list.append(predicted_coords)
        return predicted_ligs_coords_list, outputs[2], outputs[3], outputs[0], outputs[1], outputs[5]

    def __repr__(self):
        return "EquiBind " + str(self.__dict__)


def squared_distance_matrix(first, second):
    """Pairwise squared Euclidean distances used by the OT objective."""
    return torch.sum((first[:, None, :] - second[None, :, :]) ** 2, dim=2)


def body_intersection_loss(ligand, receptor, sigma, surface_ct):
    """Penalize predicted ligand/receptor body intersections."""

    def g_function(protein_coords, points):
        exponent = torch.exp(
            -torch.sum((protein_coords[None, :, :] - points[:, None, :]) ** 2, dim=2)
            / float(sigma)
        )
        return -sigma * torch.log(1e-3 + exponent.sum(dim=1))

    return torch.mean(torch.clamp(surface_ct - g_function(receptor, ligand), min=0)) + \
        torch.mean(torch.clamp(surface_ct - g_function(ligand, receptor), min=0))


def body_intersection_loss_batched(ligands, receptors, sigma, surface_ct):
    """Batched, numerically-equivalent body_intersection_loss over a list of
    per-complex (ligand, receptor) coordinate tensors. Pads to the batch max and
    masks padded points out of both the exp-sum and the per-complex mean, so it
    matches summing the per-complex scalar losses (float64-exact vs the loop).
    Returns a (B,) tensor of per-complex body losses."""
    from torch.nn.utils.rnn import pad_sequence
    device = ligands[0].device
    lig_lens = torch.tensor([t.shape[0] for t in ligands], device=device, dtype=ligands[0].dtype)
    rec_lens = torch.tensor([t.shape[0] for t in receptors], device=device, dtype=receptors[0].dtype)
    lig = pad_sequence(ligands, batch_first=True)     # (B, L, 3)
    rec = pad_sequence(receptors, batch_first=True)    # (B, M, 3)
    L, M = lig.shape[1], rec.shape[1]
    lig_mask = (torch.arange(L, device=device)[None, :] < lig_lens[:, None].long()).to(lig.dtype)
    rec_mask = (torch.arange(M, device=device)[None, :] < rec_lens[:, None].long()).to(rec.dtype)
    dist2 = torch.sum((lig[:, :, None, :] - rec[:, None, :, :]) ** 2, dim=3)
    expo = torch.exp(-dist2 / float(sigma))
    s_lig = (expo * rec_mask[:, None, :]).sum(dim=2)            # (B, L)
    g_rl = -sigma * torch.log(1e-3 + s_lig)
    term1 = (torch.clamp(surface_ct - g_rl, min=0) * lig_mask).sum(dim=1) / lig_lens
    s_rec = (expo * lig_mask[:, :, None]).sum(dim=1)            # (B, M)
    g_lr = -sigma * torch.log(1e-3 + s_rec)
    term2 = (torch.clamp(surface_ct - g_lr, min=0) * rec_mask).sum(dim=1) / rec_lens
    return term1 + term2


def solve_ot(cost_matrix):
    """Solve the uniform Earth Mover's Distance problem used by EquiBind."""
    matrix = cost_matrix.detach().cpu().numpy()
    first = np.ones(matrix.shape[0]) / matrix.shape[0]
    second = np.ones(matrix.shape[1]) / matrix.shape[1]
    transport = ot.emd(first, second, matrix, numItermax=10000, numThreads=1)
    return torch.as_tensor(
        transport, dtype=cost_matrix.dtype, device=cost_matrix.device
    )


class BindingLoss(nn.Module):
    """Evolvable EquiBind objective under the fixed training/evaluation harness."""

    def __init__(
        self,
        ot_loss_weight=1.0,
        intersection_loss_weight=1.0,
        kabsch_rmsd_weight=1.0,
        geom_reg_loss_weight=1.0,
        intersection_sigma=8.0,
        intersection_surface_ct=1.0,
    ):
        super().__init__()
        self.ot_loss_weight = ot_loss_weight
        self.intersection_loss_weight = intersection_loss_weight
        self.kabsch_rmsd_weight = kabsch_rmsd_weight
        self.geom_reg_loss_weight = geom_reg_loss_weight
        self.intersection_sigma = intersection_sigma
        self.intersection_surface_ct = intersection_surface_ct
        self.mse = nn.MSELoss()

    def forward(
        self,
        ligand_coords,
        receptor_coords,
        model_output,
        ligand_pockets,
        receptor_pockets,
        device,
    ):
        predictions, ligand_keypoints, receptor_keypoints, _, _, geom_loss = model_output
        coordinate_loss = torch.zeros((), device=device)
        ot_loss = torch.zeros((), device=device)
        kabsch_loss = torch.zeros((), device=device)
        covariances = []
        predicted_means = []
        target_means = []

        # Intersection penalty batched out of the per-complex loop (float64-exact
        # vs the loop); the loop below keeps only the inherently per-complex OT solve.
        intersection_loss = body_intersection_loss_batched(
            predictions, receptor_coords,
            sigma=self.intersection_sigma, surface_ct=self.intersection_surface_ct,
        ).sum()

        for index, prediction in enumerate(predictions):
            target = ligand_coords[index]
            coordinate_loss = coordinate_loss + self.mse(prediction, target)

            ligand_cost = squared_distance_matrix(
                ligand_pockets[index], ligand_keypoints[index]
            )
            receptor_cost = squared_distance_matrix(
                receptor_pockets[index], receptor_keypoints[index]
            )
            total_cost = ligand_cost + receptor_cost
            transport = solve_ot(total_cost)
            ot_loss = ot_loss + torch.sum(transport * total_cost)

            predicted_mean = prediction.mean(dim=0, keepdim=True)
            target_mean = target.mean(dim=0, keepdim=True)
            covariances.append(
                (prediction - predicted_mean).T @ (target - target_mean)
            )
            predicted_means.append(predicted_mean)
            target_means.append(target_mean)

        covariance_batch = torch.stack(covariances)
        u_matrix, _, vt_matrix = torch.linalg.svd(covariance_batch)
        correction = torch.eye(
            3, dtype=covariance_batch.dtype, device=device
        ).unsqueeze(0).repeat(len(predictions), 1, 1)
        correction[:, 2, 2] = torch.sign(
            torch.linalg.det(covariance_batch)
        ).detach()
        aligned_rotations = (u_matrix @ correction) @ vt_matrix
        predicted_means = torch.cat(predicted_means)
        target_means = torch.cat(target_means)
        aligned_translations = predicted_means - torch.bmm(
            aligned_rotations, target_means.unsqueeze(2)
        ).squeeze(2)
        for index, rotation in enumerate(aligned_rotations):
            aligned = (
                rotation @ ligand_coords[index].T
            ).T + aligned_translations[index]
            kabsch_loss = kabsch_loss + self.mse(aligned, predictions[index])

        count = float(len(predictions))
        components = {
            "ligs_coords_loss": coordinate_loss / count,
            "ot_loss": ot_loss / count,
            "intersection_loss": intersection_loss / count,
            "kabsch_rmsd_loss": kabsch_loss / count,
            "geom_reg_loss": geom_loss / count,
        }
        total = (
            components["ligs_coords_loss"]
            + self.ot_loss_weight * components["ot_loss"]
            + self.intersection_loss_weight * components["intersection_loss"]
            + self.kabsch_rmsd_weight * components["kabsch_rmsd_loss"]
            + self.geom_reg_loss_weight * components["geom_reg_loss"]
        )
        return total, components


def build_model(device, lig_edge_dim, rec_edge_dim):
    """Stable hook used by train.py to construct the evolvable architecture."""
    return EquiBind(
        device=device,
        lig_input_edge_feats_dim=lig_edge_dim,
        rec_input_edge_feats_dim=rec_edge_dim,
        **MODEL_CONFIG,
    )


def build_loss():
    """Stable hook used by train.py; Agent iterations may evolve its internals."""
    return BindingLoss()
