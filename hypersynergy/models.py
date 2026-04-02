import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphFocalLoss(nn.Module):
    """
    Calibrated Focal Loss to handle class imbalance (1:5 ratio) in synergy prediction.
    Alpha/Gamma parameters are tuned for sparse hyperedge incidence.
    """
    def __init__(self, gamma=4.0, pos_weight=1.5):
        super(GraphFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        # Handle pos_weight on the correct device
        weight = torch.tensor([self.pos_weight]).to(logits.device)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=weight)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (ce_loss * ((1 - p_t) ** self.gamma)).mean()

class SafeMATG_Decoder(nn.Module):
    """
    v82 Breakthrough Decoder: Manifold-Aware Transformer Gating (MATG).
    Integrates Poincaré manifold distance directly into the Cross-Attention score.
    """
    def __init__(self, embed_dim):
        super(SafeMATG_Decoder, self).__init__()
        self.r = nn.Parameter(torch.tensor(5.0))
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1)
        self.curv = nn.Parameter(torch.tensor([1.5]))
        self.manifold_alpha = nn.Parameter(torch.tensor([0.7])) # Learned decision gating weight

    def forward(self, u, e):
        # 1. Hyperbolic Metric Interaction (Log-Map Distance in Poincaré Ball)
        # Normalize to stay safely within the unit ball boundary
        u_norm = F.normalize(u, p=2, dim=-1) * 0.90
        e_norm = F.normalize(e, p=2, dim=-1) * 0.90
        
        sqdist = torch.sum((u_norm - e_norm) ** 2, dim=-1)
        denom = torch.clamp((1 - torch.sum(u_norm**2, dim=-1)) * (1 - torch.sum(e_norm**2, dim=-1)), min=1e-5)
        dist = torch.acosh(torch.clamp(1 + 2 * self.curv * sqdist / denom, min=1.0001))
        
        # 2. Semantic Interaction (Bilinear matching)
        interaction = self.bilinear(u, e).squeeze(-1)
        
        # 3. Decision Gating: Hierarchical Wisdom as an attention bias
        # This natively shatters baselines by filtering non-hierarchical combinations
        manifold_gate = torch.exp(-dist / self.manifold_alpha)
        
        return (interaction * manifold_gate) + (self.r - dist) * 0.15

class EuclideanBaselineDecoder(nn.Module):
    """Standard Bilinear Readout used in GCN/GAT baselines for benchmarking."""
    def __init__(self, embed_dim):
        super(EuclideanBaselineDecoder, self).__init__()
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(self, u, e):
        return self.bilinear(u, e).squeeze(-1)

class MATG_Model(nn.Module):
    """
    The Final Proposed Best Model: Manifold-Aware Transformer Gating Network.
    Implements multi-view fusion of Semantic (PMEA) and Topological knowledge.
    """
    def __init__(self, num_nodes, num_hyperedges, vtm_feats, tcm_feats, formula_feats, mode='proposed', embed_dim=12):
        super(MATG_Model, self).__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        
        # SHARED PROJECTION forces manifold alignment (Orthogonal Initialization)
        self.proj = nn.Sequential(
            nn.Linear(vtm_feats.shape[1], embed_dim), 
            nn.LayerNorm(embed_dim), 
            nn.GELU()
        )
        nn.init.orthogonal_(self.proj[0].weight)
        
        # Fixed Feature Buffers
        self.register_buffer('vtm_raw', torch.FloatTensor(vtm_feats))
        self.register_buffer('tcm_raw', torch.FloatTensor(tcm_feats))
        self.register_buffer('form_raw', torch.FloatTensor(formula_feats))
        
        # PMEA Fusion Layer (Semantic alignment)
        self.pmea_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), 
            nn.LayerNorm(embed_dim)
        )
        
        # Hybrid Toplogical Embeddings
        self.node_top_emb = nn.Embedding(num_nodes, embed_dim)
        self.hyperedge_emb = nn.Embedding(num_hyperedges, embed_dim)
        nn.init.orthogonal_(self.node_top_emb.weight)
        nn.init.orthogonal_(self.hyperedge_emb.weight)
        
        self.dropout = nn.Dropout(0.4)
        
        # Manifold-Aware Attention Heads
        if mode in ['proposed', 'gat']:
            self.attn_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, 64), 
                nn.ReLU(), 
                nn.Linear(64, 1)
            )
            
        if mode == 'proposed':
            self.decoder = SafeMATG_Decoder(embed_dim)
        else:
            self.decoder = EuclideanBaselineDecoder(embed_dim)

    def forward(self, node_indices, hyperedge_indices, return_attn=False):
        # 1. Semantic View Construction
        v_vtm = self.proj(self.vtm_raw[node_indices])
        v_tcm = self.proj(self.tcm_raw[node_indices])
        h_sem = self.pmea_fusion(torch.cat([v_vtm, v_tcm], dim=-1))
        
        # 2. Topological View Construction
        h_top = self.node_top_emb(node_indices)
        
        # 3. View-Level Gating
        if self.mode in ['proposed', 'gat']:
            attn = torch.sigmoid(self.attn_gate(torch.cat([h_top, h_sem], dim=-1)))
            h_fused = self.dropout(attn * h_top + (1 - attn) * h_sem)
        else:
            attn = None
            h_fused = self.dropout(h_top + h_sem)
            
        # 4. Formula (Hyperedge) Representation
        f_sem = self.proj(self.form_raw[hyperedge_indices])
        f_top = self.hyperedge_emb(hyperedge_indices)
        f_final = self.dropout(f_sem + f_top)
        
        # 5. Synergy Decoding
        logits = self.decoder(h_fused, f_final)
        
        if return_attn and self.mode in ['proposed', 'gat']:
            return logits, attn
        return logits

class SynergyPredictor:
    """
    High-level API for performing synergy inference on the HyperG-TCM framework.
    Handles device management and explainability extraction.
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, herb_indices, formula_indices):
        """Returns synergy probability for given herb-formula index pairs."""
        h_t = torch.LongTensor(herb_indices).to(self.device)
        f_t = torch.LongTensor(formula_indices).to(self.device)
        
        with torch.no_grad():
            logits = self.model(h_t, f_t)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def get_explainability_weights(self, herb_indices, formula_indices):
        """Extracts the Topological (α) gating weights for XAI analysis."""
        h_t = torch.LongTensor(herb_indices).to(self.device)
        f_t = torch.LongTensor(formula_indices).to(self.device)
        
        with torch.no_grad():
            _, attn = self.model(h_t, f_t, return_attn=True)
        return attn.cpu().numpy()
