import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvNCF(nn.Module):
    def __init__(self, drugs_dim, sides_dim, embed_dim, bathsize, dropout1=0.8, dropout2=0.8):
        super(ConvNCF, self).__init__()

        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.batchsize = bathsize
        self.drug_dim = self.drugs_dim // 10
        self.side_dim = self.sides_dim // 4
        self.embed_dim = embed_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2

        # Global encoders (full drug/side feature vectors) with mixed normalization
        self.drugs_encoder = nn.Sequential(
            nn.Linear(self.drugs_dim, embed_dim),
            nn.BatchNorm1d(embed_dim, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.sides_encoder = nn.Sequential(
            nn.Linear(self.sides_dim, embed_dim),
            nn.BatchNorm1d(embed_dim, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Shared block encoders (weight-tied across blocks) with residual
        self.drug_block_encoder = nn.Sequential(
            nn.Linear(self.drug_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)  # no ReLU here (applied after residual)
        )
        self.side_block_encoder = nn.Sequential(
            nn.Linear(self.side_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # Residual projections for block encoders (map raw block to embed_dim)
        self.drug_block_residual = nn.Linear(self.drug_dim, embed_dim)
        self.side_block_residual = nn.Linear(self.side_dim, embed_dim)

        # Learned block-specific embeddings to differentiate blocks
        self.drug_block_embeddings = nn.Parameter(torch.Tensor(10, embed_dim))
        self.side_block_embeddings = nn.Parameter(torch.Tensor(4, embed_dim))
        nn.init.normal_(self.drug_block_embeddings, std=0.1)
        nn.init.normal_(self.side_block_embeddings, std=0.1)

        # Interaction projections: project to a small dimension for efficient outer-product CNN
        self.inter_dim = 16  # compact interaction dimension to reduce overfitting
        self.drug_inter_proj = nn.Linear(embed_dim, self.inter_dim)
        self.side_inter_proj = nn.Linear(embed_dim, self.inter_dim)

        # Two-layer CNN on the interaction map for richer pattern extraction
        self.inter_cnn = nn.Sequential(
            nn.Conv2d(self.inter_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1))  # -> (batch, 32, 1, 1)
        )

        # Fusion for classification: concatenate global drug, global side, and interaction features
        fusion_dim = 2 * embed_dim + 32
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.BatchNorm1d(embed_dim, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2)
        )

        # Common projection with residual
        self.common_proj = nn.Linear(embed_dim, embed_dim)

        # Classification head
        self.classifier = nn.Linear(embed_dim, 1)

        # Regression head from global encodings only (separate path)
        self.reg_encoder = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(embed_dim, 1)
        )



    def forward(self, drug_features, side_features, device):
        # Move all input tensors to the correct device before any operations
        drug_features = drug_features.to(device)
        side_features = side_features.to(device)

        # Global encodings
        x_drugs = self.drugs_encoder(drug_features)   # (batch, embed_dim)
        x_sides = self.sides_encoder(side_features)   # (batch, embed_dim)

        # Block encodings with shared encoder + residual + learned embeddings
        drug_blocks = drug_features.chunk(10, 1)  # list of 10 tensors (batch, drug_dim)
        side_blocks = side_features.chunk(4, 1)   # list of 4 tensors (batch, side_dim)

        drug_reps = []
        for i, block in enumerate(drug_blocks):
            h = self.drug_block_encoder(block)
            residual = self.drug_block_residual(block)
            h = h + residual
            h = F.relu(h, inplace=True)
            h = h + self.drug_block_embeddings[i].unsqueeze(0)
            drug_reps.append(h.unsqueeze(1))
        drug_reps = torch.cat(drug_reps, dim=1)  # (batch, 10, embed_dim)

        side_reps = []
        for i, block in enumerate(side_blocks):
            h = self.side_block_encoder(block)
            residual = self.side_block_residual(block)
            h = h + residual
            h = F.relu(h, inplace=True)
            h = h + self.side_block_embeddings[i].unsqueeze(0)
            side_reps.append(h.unsqueeze(1))
        side_reps = torch.cat(side_reps, dim=1)  # (batch, 4, embed_dim)

        # Project block representations to interaction space
        drug_proj = self.drug_inter_proj(drug_reps)  # (batch, 10, inter_dim)
        side_proj = self.side_inter_proj(side_reps)  # (batch, 4, inter_dim)

        # Compute element-wise product: (batch, 10, 4, inter_dim)
        inter_map_raw = drug_proj.unsqueeze(2) * side_proj.unsqueeze(1)
        # Reshape to (batch, inter_dim, 10, 4) for 2D convolution
        inter_map = inter_map_raw.permute(0, 3, 1, 2).contiguous()  # (batch, inter_dim, 10, 4)

        # Apply deeper CNN
        inter_feat = self.inter_cnn(inter_map).view(inter_map.size(0), -1)  # (batch, 64)

        # Fusion for classification
        combined = torch.cat([x_drugs, x_sides, inter_feat], dim=1)  # (batch, fusion_dim)
        h = self.fusion_layer(combined)  # (batch, embed_dim)
        h = self.common_proj(h) + h  # residual connection
        h = F.relu(h, inplace=True)

        classification = self.classifier(h)

        # Regression from global encodings only (separate path, avoids overfitting from interaction)
        reg_input = torch.cat([x_drugs, x_sides], dim=1)  # (batch, 2*embed_dim)
        regression = F.softplus(self.reg_encoder(reg_input).squeeze(1))

        return classification.squeeze(), regression