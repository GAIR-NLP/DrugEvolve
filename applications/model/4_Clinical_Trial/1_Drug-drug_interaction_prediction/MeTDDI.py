"""
MeTDDI shared-backbone baseline for the DrugEvolve auto-iteration framework.

ONE shared architecture (atom-level + motif-level local-global self-attention encoders
plus a co-attention module) is used for BOTH tasks; only the output head differs:
  - classification (S2 unseen-one-drug + S3 unseen-two-drugs): Dense(4, softmax)
  - regression (AUC FC on FDA val + 2023 external set): Dense(1)

This file is the EVOLVABLE target: the agent may modify the ARCHITECTURE
(build_model / get_config / layers) and OPTIONALLY the training strategy
(compile_and_fit). Data loading, the train / S2 / S3 / val / external SPLITS,
prediction and CSV writing are OWNED BY THE FIXED HARNESS
(metddi/scripts/run_model.py) and must NOT appear here — this is what prevents
test data from leaking into training.

The fixed harness imports `build_model` (required) and `compile_and_fit`
(optional) from this file and drives data + training + prediction itself:
  python run_model.py <this model.py> --task classification --output_dir <dir>
  python run_model.py <this model.py> --task regression     --output_dir <dir>

Required env vars (set by run.sh):
  METDDI_SUPPORT : dir containing ddi_data.py, utils.py, mol_graph.py, features.py
  METDDI_DATA    : path to MeTDDI-main/data
  METDDI_CODE    : path to MeTDDI-main/code  (for token_id.json + preprocessed npy)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

# --- bootstrap imports of the (non-evolvable) preprocessing modules ---
# METDDI_SUPPORT : dir with ddi_data.py (baseline scripts dir, stays fixed even when
#                  this model.py is copied into algorithm/<name>/ by the framework)
# METDDI_PREP    : MeTDDI source dir providing utils.py / mol_graph.py / features.py
SUPPORT_DIR = os.environ.get("METDDI_SUPPORT", os.path.dirname(os.path.abspath(__file__)))
if SUPPORT_DIR not in sys.path:
    sys.path.insert(0, SUPPORT_DIR)
PREP_DIR = os.environ.get("METDDI_PREP")
if PREP_DIR and PREP_DIR not in sys.path:
    sys.path.insert(0, PREP_DIR)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras

from ddi_data import DDIDataset
from utils import Mol_Tokenizer
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# ============================================================================
# Per-task architecture configs (the agent may tune / replace these).
# ONE shared backbone code is used for both tasks; only these hyper-parameters
# and the output head differ. Classification uses MeTDDI's original (larger)
# config; regression uses MeTDDI's original (smaller) config.
# ============================================================================
CLASSIFICATION_CONFIG = {
    'motif_d_model': 512,   # 256 * 2
    'atom_d_model': 256,
    'dff': 512,
    'num_layers_motif': 4,
    'num_layers_atom': 2,
    'num_heads': 8,
    'co_attn_k': 128,
    'dropout': 0.1,
}
REGRESSION_CONFIG = {
    'motif_d_model': 256,
    'atom_d_model': 128,
    'dff': 256,
    'num_layers_motif': 2,
    'num_layers_atom': 2,
    'num_heads': 4,
    'co_attn_k': 128,
    'dropout': 0.1,
}

def get_config(task):
    return CLASSIFICATION_CONFIG if task == 'classification' else REGRESSION_CONFIG

# ============================================================================
# Architecture (inlined from MeTDDI atom-level + motif-level encoders)
# ============================================================================
def rescale_distance_matrix(w):
    c = tf.constant(1.0, dtype=tf.float32)
    return (c + tf.math.exp(c)) / (c + tf.math.exp(c - w))

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))

def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

def create_padding_mask_atom(batch_data):
    padding_mask = tf.cast(tf.math.equal(tf.reduce_sum(batch_data, axis=-1), 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    if dist_matrix is not None:
        matmul_qk = tf.nn.relu(tf.matmul(q, k, transpose_b=True))
        dist_matrix = rescale_distance_matrix(dist_matrix)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = (tf.multiply(matmul_qk, dist_matrix)) / tf.math.sqrt(dk)
    else:
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        return self.dense(concat_attention), attention_weights

def feed_forward_network(d_model, dff):
    return keras.Sequential([keras.layers.Dense(dff, activation=gelu), keras.layers.Dense(d_model)])

class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, training, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = tf.split(x, 2, -1)
        x_l, attn_local = self.mha1(x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None)
        x_g, attn_global = self.mha2(x2, x2, x2, encoder_padding_mask, adjoin_matrix=None, dist_matrix=dist_matrix)
        attn_output = tf.concat([x_l, x_g], axis=-1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        return out2, attn_local, attn_global

class EncoderModel_atom(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderModel_atom, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Dense(self.d_model, activation='relu')
        self.global_embedding = keras.layers.Dense(dff, activation='relu')
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [EncoderLayer(int(d_model), num_heads, dff, rate) for _ in range(num_layers)]

    def call(self, x, training, adjoin_matrix=None, dist_matrix=None, atom_match_matrix=None, sum_atoms=None):
        encoder_padding_mask = create_padding_mask_atom(x)
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
        if dist_matrix is not None:
            dist_matrix = dist_matrix[:, tf.newaxis, :, :]
        x = self.embedding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, _, _ = self.encoder_layers[i](x, training, encoder_padding_mask, adjoin_matrix, dist_matrix=dist_matrix)
        x = tf.matmul(atom_match_matrix, x) / sum_atoms
        x = self.global_embedding(x)
        return x, None, None, encoder_padding_mask

class EncoderModel_motif(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderModel_motif, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size, self.d_model)
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [EncoderLayer(int(d_model), num_heads, dff, rate) for _ in range(num_layers)]

    def call(self, x, training, atom_level_features, adjoin_matrix=None, dist_matrix=None):
        encoder_padding_mask = create_padding_mask(x)
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
        if dist_matrix is not None:
            dist_matrix = dist_matrix[:, tf.newaxis, :, :]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        x_temp = x[:, 1:, :] + atom_level_features
        x = tf.concat([x[:, 0:1, :], x_temp], axis=1)
        for i in range(self.num_layers):
            x, _, _ = self.encoder_layers[i](x, training, encoder_padding_mask, adjoin_matrix, dist_matrix=dist_matrix)
        return x, None, None, encoder_padding_mask

class Co_Attention_Layer(keras.layers.Layer):
    def __init__(self, graph_feat_size, k, **kwargs):
        self.k = k
        self.graph_feat_size = graph_feat_size
        super(Co_Attention_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        init = tf.compat.v1.glorot_uniform_initializer()
        self.W_m = self.add_weight(shape=(self.k, self.graph_feat_size), initializer=init, name='W_m', trainable=True)
        self.W_v = self.add_weight(shape=(self.k, self.graph_feat_size), initializer=init, name='W_v', trainable=True)
        self.W_q = self.add_weight(shape=(self.k, self.graph_feat_size), initializer=init, name='W_q', trainable=True)
        self.W_h = self.add_weight(shape=(1, self.k), initializer=init, name='W_h', trainable=True)
        super(Co_Attention_Layer, self).build(input_shape)

    def call(self, inputs):
        V_n, Q_n = inputs[0], inputs[1]
        V_0 = V_n[:, 0, :][:, :, tf.newaxis]
        Q_0 = Q_n[:, 0, :][:, :, tf.newaxis]
        V_r = tf.transpose(V_n[:, 1:, :], [0, 2, 1])
        Q_r = tf.transpose(Q_n[:, 1:, :], [0, 2, 1])
        M_0 = tf.multiply(V_0, Q_0)
        H_v = tf.multiply(tf.tanh(tf.matmul(self.W_v, V_r)), tf.tanh(tf.matmul(self.W_m, M_0)))
        H_q = tf.multiply(tf.tanh(tf.matmul(self.W_q, Q_r)), tf.tanh(tf.matmul(self.W_m, M_0)))
        alpha_v = tf.nn.softmax(tf.matmul(self.W_h, H_v), axis=-1)
        alpha_q = tf.nn.softmax(tf.matmul(self.W_h, H_q), axis=-1)
        vector_v = tf.matmul(alpha_v, tf.transpose(V_r, [0, 2, 1]))
        vector_q = tf.matmul(alpha_q, tf.transpose(Q_r, [0, 2, 1]))
        return tf.squeeze(vector_v, 1), tf.squeeze(vector_q, 1), alpha_v, alpha_q


# ============================================================================
# Shared model builder (ONE shared backbone; task-specific config + head)
# ============================================================================
def make_inputs(suffix):
    """The 8 per-drug input tensors (order matches the shared backbone's inputs)."""
    return [
        Input(shape=(None, 61), name=f"atom_features{suffix}"),
        Input(shape=(None, None), name=f"adjoin_matrix{suffix}_atom"),
        Input(shape=(None, None), name=f"dist_matrix{suffix}_atom"),
        Input(shape=(None, None), name=f"atom_match_matrix{suffix}"),
        Input(shape=(None, None), name=f"sum_atoms{suffix}"),
        Input(shape=(None,), name=f"molecule_sequence{suffix}"),
        Input(shape=(None, None), name=f"adj_matrix{suffix}"),
        Input(shape=(None, None), name=f"dist_matrix{suffix}"),
    ]


def build_backbone(cfg, input_vocab_size, training=False):
    """SHARED single-drug graph encoder (atom-level + motif-level encoders).

    BOTH tasks (classification and regression) MUST build their model on top of
    THIS same backbone. Do NOT create separate / task-specific encoder
    architectures. You may change the internals here, but it must remain a single
    shared definition used by both tasks; only the `get_config` hyper-parameters
    and the final task head are allowed to differ between tasks.
    """
    d_model = cfg['motif_d_model']
    atom_d_model = cfg['atom_d_model']
    dff = cfg['dff']
    num_heads = cfg['num_heads']

    atom_inputs = Input(shape=(None, 61), name="atom_features")
    atom_adj_inputs = Input(shape=(None, None), name="atom_adj_matrix")
    atom_dist_inputs = Input(shape=(None, None), name="atom_dist_matrix")
    atom_match_matrix = Input(shape=(None, None), name="atom_match_matrix")
    sum_atoms = Input(shape=(None, None), name="sum_atoms")
    motif_inputs = Input(shape=(None,), name="molecule_sequence")
    motif_adj_inputs = Input(shape=(None, None), name="adj_matrix")
    motif_dist_inputs = Input(shape=(None, None), name="dist_matrix")

    Outseq_atom, *_ , _ = EncoderModel_atom(
        num_layers=cfg['num_layers_atom'], d_model=atom_d_model, dff=dff, num_heads=num_heads)(
        atom_inputs, adjoin_matrix=atom_adj_inputs, dist_matrix=atom_dist_inputs,
        atom_match_matrix=atom_match_matrix, sum_atoms=sum_atoms, training=training)
    Outseq_motif, *_ , _ = EncoderModel_motif(
        num_layers=cfg['num_layers_motif'], d_model=d_model, dff=dff, num_heads=num_heads,
        input_vocab_size=input_vocab_size)(
        motif_inputs, adjoin_matrix=motif_adj_inputs, dist_matrix=motif_dist_inputs,
        atom_level_features=Outseq_atom, training=training)
    return Model(
        inputs=[atom_inputs, atom_adj_inputs, atom_dist_inputs, atom_match_matrix, sum_atoms,
                motif_inputs, motif_adj_inputs, motif_dist_inputs],
        outputs=[Outseq_motif], name="shared_backbone")


def build_model(task, input_vocab_size, training=False):
    """Dual-drug DDI model: SHARED backbone (both drugs) + co-attention + task head."""
    cfg = get_config(task)
    d_model = cfg['motif_d_model']
    co_attn_k = cfg['co_attn_k']
    dropout = cfg['dropout']

    backbone = build_backbone(cfg, input_vocab_size, training)

    in_a = make_inputs("1")
    in_b = make_inputs("2")
    druga_trans = backbone(in_a)   # both drugs pass through the SAME backbone
    drugb_trans = backbone(in_b)

    Wa = keras.layers.Dense(d_model)
    Wb = keras.layers.Dense(d_model)
    co_attn = Co_Attention_Layer(d_model, k=co_attn_k, name='Co_attention_layer')
    druga_, drugb_, *_ = co_attn([Wa(druga_trans), Wb(drugb_trans)])
    x = keras.layers.Concatenate()([druga_, drugb_])
    x = keras.layers.Dense(d_model / 2, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x, training=training)
    x = keras.layers.Dense(d_model / 4, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x, training=training)
    if task == 'classification':
        out = keras.layers.Dense(4, activation='softmax')(x)
    else:
        out = keras.layers.Dense(1)(x)

    return Model(inputs=in_a + in_b, outputs=[out])


# ============================================================================
# OPTIONAL training strategy (the agent MAY override the body of this function).
#
# IMPORTANT — data is OWNED BY THE FIXED HARNESS (metddi/scripts/run_model.py):
#   Data file paths, the train / S2 / S3 / val / external SPLITS, prediction and
#   CSV writing all live in the fixed harness and are NOT part of this file.
#   This file must ONLY define the architecture (build_model / get_config /
#   layers) and may optionally define compile_and_fit() to customize training.
#
#   DO NOT read data files, glob the filesystem (os.walk/glob), re-split or
#   discover datasets, write prediction CSVs, or add an argparse/main entrypoint
#   here. The harness imports build_model (+ this optional compile_and_fit) and
#   drives everything else. This is what guarantees S2/S3/external stay held out
#   and that no test data can leak into training.
#
# Contract:
#   build_model(task, input_vocab_size, training=False) -> UNCOMPILED keras.Model
#       output: classification -> [B,4] softmax ; regression -> [B,1]
#   compile_and_fit(model, task, train_ds, val_ds, epochs) -> None   (optional)
#       Receives ONLY train_ds and val_ds (= S2 / regression-val for early
#       stopping). It NEVER sees S3 / external, so test data cannot leak in.
# ============================================================================
def compile_and_fit(model, task, train_ds, val_ds, epochs):
    """Canonical MeTDDI training. Override the body to change the training
    strategy (loss / optimizer / lr schedule / layer-level auxiliary losses via
    self.add_loss), but KEEP this signature — the harness calls it exactly so."""
    if task == 'classification':
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ['sparse_categorical_accuracy']
        for lr in (1e-4, 1e-5):
            model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=metrics)
            es = EarlyStopping(monitor='val_loss', patience=5, min_delta=5e-4,
                               restore_best_weights=True)
            model.fit(train_ds, epochs=epochs, callbacks=[es], validation_data=val_ds)
    else:
        model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4),
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        es = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-4,
                           restore_best_weights=True)
        model.fit(train_ds, epochs=epochs, callbacks=[es], validation_data=val_ds)
