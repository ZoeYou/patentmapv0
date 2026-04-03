import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import gc

def cleanup_memory():
    """
    Systematic memory cleanup utility.
    Call this after explicitly deleting tensor references to force cleanup.
    
    Note: This function cannot delete tensor references from the calling scope.
    You must explicitly delete tensors first, then call this function.
    
    Example usage:
        del tensor1, tensor2, tensor3
        cleanup_memory()
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ContrastiveLearningOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

    loss_MLM: Optional[torch.FloatTensor] = None
    loss_reg: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None  # InfoNCE loss
    
    # Additional regularization loss details
    barlow_twins_loss: Optional[torch.FloatTensor] = None
    on_diagonal: Optional[torch.FloatTensor] = None
    off_diagonal: Optional[torch.FloatTensor] = None
    
    vicreg_loss: Optional[torch.FloatTensor] = None
    invariance_loss: Optional[torch.FloatTensor] = None
    variance_loss: Optional[torch.FloatTensor] = None
    covariance_loss: Optional[torch.FloatTensor] = None

    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, force_fp32=False):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.force_fp32 = force_fp32
        
    def forward(self, x):
        if self.force_fp32:
            # Force FP32 computation for high precision training
            original_dtype = x.dtype
            # Convert everything to FP32 for computation
            with torch.amp.autocast('cuda', enabled=False):
                x = x.float()
                
                # Use functional operations to avoid modifying model parameters
                # Get weights and bias in FP32
                weight = self.proj.weight.float()
                bias = self.proj.bias.float() if self.proj.bias is not None else None
                
                # Manual linear transformation in FP32
                proj_out = F.linear(x, weight, bias)
                x_lin, x_gate = proj_out.chunk(2, dim=-1)
                result = x_lin * F.silu(x_gate)  # Swish = silu
                
                return result.to(original_dtype) if original_dtype != torch.float32 else result
        else:
            x_lin, x_gate = self.proj(x).chunk(2, dim=-1)
            return x_lin * F.silu(x_gate)  # Swish = silu



class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim if output_dim is not None else input_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x



class VICRegProjector(nn.Module):
    """
    VICReg projector adapted for NLP/text:
    - 3-layer MLP with LayerNorm and SwiGLU activation
    - Final layer has no normalization or activation (following VICReg paper)
    - NLP-optimized dimensions: input_dim -> hidden_dim -> hidden_dim -> output_dim
    - For text: typically input_dim=768/1024, hidden_dim=2304/3072, output_dim=2304/3072
    
    Architecture:
      Layer 1: LayerNorm -> SwiGLU (input_dim -> hidden_dim)
      Layer 2: LayerNorm -> SwiGLU (hidden_dim -> hidden_dim) 
      Layer 3: Linear only (hidden_dim -> output_dim, no norm/activation)
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, ln_eps=1e-6, force_fp32=False):
        super().__init__()
        
        # NLP-optimized defaults based on input dimension
        if hidden_dim is None:
            # Use 3x input dimension for hidden layers, reasonable for text
            hidden_dim = int(input_dim * 3)
        
        if output_dim is None:
            output_dim = hidden_dim
        
        # Store dimensions for logging
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Store configuration
        self.force_fp32 = force_fp32
        
        # Build the network with SwiGLU activation
        # Layer 1: input -> hidden with SwiGLU
        self.layer1_norm = nn.LayerNorm(input_dim, eps=ln_eps)
        self.layer1_swiglu = SwiGLU(input_dim, hidden_dim, bias=True, force_fp32=force_fp32)
        
        # Layer 2: hidden -> hidden with SwiGLU
        self.layer2_norm = nn.LayerNorm(hidden_dim, eps=ln_eps)
        self.layer2_swiglu = SwiGLU(hidden_dim, hidden_dim, bias=True, force_fp32=force_fp32)

        # Final layer: hidden -> output (no normalization, no activation)
        # Following VICReg paper design
        self.final_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        if self.force_fp32:
            # Force FP32 computation for high precision training
            original_dtype = x.dtype
            # Use autocast disabled context to ensure FP32 computation
            with torch.amp.autocast('cuda', enabled=False):
                x = x.float()
                
                # Layer 1: Manual LayerNorm -> SwiGLU computation in FP32
                # Manual LayerNorm computation to avoid modifying model parameters
                layer1_weight = self.layer1_norm.weight.float()
                layer1_bias = self.layer1_norm.bias.float()
                layer1_eps = self.layer1_norm.eps
                
                # Manual layer normalization
                mean = x.mean(-1, keepdim=True)
                var = ((x - mean) ** 2).mean(-1, keepdim=True)
                x = (x - mean) / torch.sqrt(var + layer1_eps)
                x = x * layer1_weight + layer1_bias
                
                x = self.layer1_swiglu(x)
                
                # Layer 2: Manual LayerNorm -> SwiGLU computation in FP32
                layer2_weight = self.layer2_norm.weight.float()
                layer2_bias = self.layer2_norm.bias.float()
                layer2_eps = self.layer2_norm.eps
                
                # Manual layer normalization
                mean = x.mean(-1, keepdim=True)
                var = ((x - mean) ** 2).mean(-1, keepdim=True)
                x = (x - mean) / torch.sqrt(var + layer2_eps)
                x = x * layer2_weight + layer2_bias
                
                x = self.layer2_swiglu(x)

                # Final layer: Manual Linear computation in FP32
                final_weight = self.final_proj.weight.float()
                final_bias = self.final_proj.bias.float() if self.final_proj.bias is not None else None
                x = F.linear(x, final_weight, final_bias)
                
                # Convert back to original dtype if needed
                return x.to(original_dtype) if original_dtype != torch.float32 else x
        else:
            # Standard mixed precision compatible forward pass
            # Layer 1: LayerNorm -> SwiGLU
            x = self.layer1_norm(x)
            x = self.layer1_swiglu(x)
            
            # Layer 2: LayerNorm -> SwiGLU
            x = self.layer2_norm(x)
            x = self.layer2_swiglu(x)

            # Final layer: Linear only (no normalization, no activation)
            x = self.final_proj(x)
            
            return x


class BarlowTwinsProjector(nn.Module):
    """
    Barlow Twins projector adapted for NLP/text:
    - 3-layer MLP with LayerNorm and GELU activation
    - No bias on final layer
    - NLP-optimized dimensions: input_dim -> hidden_dim -> hidden_dim -> output_dim
    - For text: typically input_dim=768/1024, hidden_dim=1024/1536, output_dim=256/512
    """
    def __init__(self, input_dim, hidden_dims=None, output_dim=None):
        super().__init__()
        
        # NLP-optimized defaults
        if hidden_dims is None:
            # Use 2x input dimension for hidden layers, reasonable for text
            hidden_dim = int(input_dim * 2)
            hidden_dims = f"{hidden_dim}-{hidden_dim}"
        
        if output_dim is None:
            # If hidden_dims is a string, parse it to get the last dimension
            if isinstance(hidden_dims, str):
                hidden_sizes = list(map(int, hidden_dims.split('-')))
                output_dim = hidden_sizes[-1]
            else:
                output_dim = hidden_dims[-1] if isinstance(hidden_dims, list) else hidden_dim
        
        # Store for logging
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Parse hidden dimensions
        if isinstance(hidden_dims, str):
            hidden_sizes = list(map(int, hidden_dims.split('-')))
            sizes = [input_dim] + hidden_sizes + [output_dim]
        elif isinstance(hidden_dims, list):
            sizes = [input_dim] + hidden_dims + [output_dim]
        else:
            # Single hidden dimension
            sizes = [input_dim, hidden_dims, output_dim]
        
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
        
        # Final layer (no bias)
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        
        self.projector = nn.Sequential(*layers)
        
        # Normalization layer for the representations (affine=False like in original)
        self.ln_final = nn.LayerNorm(sizes[-1], elementwise_affine=False)

    def forward(self, x):
        # Project the features
        projected = self.projector(x)
        
        # Apply layer normalization to the final representations
        # This is used for computing the cross-correlation matrix
        normalized = self.ln_final(projected)
        
        return projected, normalized



class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).clamp(min=1e-6).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).clamp(min=1e-6).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).clamp(min=1e-6).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)

    if cls.model_args.pooler_type == "cls":
        if cls.model_args.regularization == "barlow_twins":
            cls.mlp = MLPLayer(config.hidden_size, config.hidden_size)
            
            # Get projector dimensions from model args or use NLP-optimized defaults
            bt_hidden_dim = getattr(cls.model_args, 'barlow_twins_hidden_dim', None)
            bt_output_dim = getattr(cls.model_args, 'barlow_twins_output_dim', None)
            
            cls.mlp_reg = BarlowTwinsProjector(
                input_dim=config.hidden_size,
                hidden_dims=bt_hidden_dim,
                output_dim=bt_output_dim
            )
            
            # Log projector configuration
            print(f"🔧 Barlow Twins Projector: {config.hidden_size} → {cls.mlp_reg.hidden_dims} → {cls.mlp_reg.output_dim}")
            total_params = sum(p.numel() for p in cls.mlp_reg.parameters())
            print(f"📊 Barlow Twins Projector Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        
        elif cls.model_args.regularization == "vicreg":
            cls.mlp = MLPLayer(config.hidden_size, config.hidden_size)
            
            # Get projector dimensions and FP32 setting from model args
            vic_hidden_dim = getattr(cls.model_args, 'vicreg_hidden_dim', None)
            vic_output_dim = getattr(cls.model_args, 'vicreg_output_dim', None)
            vic_force_fp32 = getattr(cls.model_args, 'vicreg_force_fp32', True)  # Default to True for SwiGLU stability
            
            cls.mlp_reg = VICRegProjector(
                input_dim=config.hidden_size,
                hidden_dim=vic_hidden_dim,
                output_dim=vic_output_dim,
                force_fp32=vic_force_fp32
            )
            
            # Log projector configuration
            fp32_status = "FP32" if vic_force_fp32 else "Mixed Precision"
            print(f"🔧 VICReg Projector (LayerNorm + SwiGLU, {fp32_status}): {config.hidden_size} → {cls.mlp_reg.hidden_dim} → {cls.mlp_reg.hidden_dim} → {cls.mlp_reg.output_dim}")
            total_params = sum(p.numel() for p in cls.mlp_reg.parameters())
            print(f"📊 VICReg Projector Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

        else:
            cls.mlp = MLPLayer(config.hidden_size, config.hidden_size)
            # Only create mlp_reg if regularization is actually used
            if cls.model_args.regularization is not None:
                cls.mlp_reg = MLPLayer(config.hidden_size, config.hidden_size)


def barlow_twins_loss(z1, z2, lambd=5e-3):
    """
    Barlow Twins loss function
    z1, z2: [batch_size, dim] - embeddings from two views
    lambd: weight for off-diagonal penalty
    """
    batch_size, dim = z1.size()
    
    # Normalize representations
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-8)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)
    
    # Compute cross-correlation matrix
    c = torch.mm(z1_norm.T, z2_norm) / batch_size  # [dim, dim]
    
    # Loss components
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = (c.flatten().pow_(2).sum() - torch.diagonal(c).pow_(2).sum())
    
    loss = on_diag + lambd * off_diag
    return loss, on_diag, off_diag


def vicreg_loss_fn(z1, z2, invariance_weight=25.0, variance_weight=25.0, covariance_weight=1.0, eps=1e-4, gamma=1.0, return_details=False):
    """
    VICReg loss function
    z1, z2: [N, D] - embeddings from two views
    """
    N, D = z1.size()

    # Variance loss: encourage each embedding dimension to have std > gamma
    def variance_loss(z):
        std = torch.sqrt(z.var(dim=0) + eps)  # [D]
        return torch.mean(F.relu(gamma - std) ** 2)  # penalize if std < gamma

    loss_var = variance_loss(z1) + variance_loss(z2)

    # Invariance loss: MSE between corresponding pairs
    loss_inv = F.mse_loss(z1, z2)

    # Covariance loss: penalize off-diagonal elements in covariance matrix
    def covariance_loss(z):
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (N - 1)
        off_diag = cov.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
        return (off_diag ** 2).sum() / D # penalize off-diagonal elements

    loss_cov = covariance_loss(z1) + covariance_loss(z2)

    # Final weighted loss
    loss = (
        invariance_weight * loss_inv +
        variance_weight * loss_var +
        covariance_weight * loss_cov
    )

    if return_details:
        return loss, loss_inv, loss_var, loss_cov
    return loss



def cl_forward(
    cls, encoder,
    input_ids=None, attention_mask=None, token_type_ids=None,
    return_dict=None, mlm_input_ids=None, mlm_labels=None,
    need_dropout=None, dropout_rate=None,
):
    """
    Contrastive learning forward pass with InfoNCE loss.
    Always expects 2 views, handles different data augmentation strategies.
    
    Args:
        input_ids: [B, 2, L] - Two views for each sample
        need_dropout: [B] - Boolean tensor indicating whether each sample needs dropout augmentation
                     If None, all samples use standard dropout-based augmentation
        dropout_rate: float - Custom dropout rate to use for dropout augmentation (0.0-1.0)
                     If None, uses model's default dropout settings
    """
    # Input validation
    if input_ids is None:
        raise ValueError("input_ids cannot be None")
    
    if input_ids.dim() != 3:
        raise ValueError(f"Expected input_ids to have 3 dimensions [B, 2, L], got {input_ids.dim()}")

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    B, V, L = input_ids.size()
    if V != 2:
        raise ValueError(f"Expected exactly 2 views, got {V}")

    device = input_ids.device
    dist_on = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_on else 0
    world_sz = dist.get_world_size() if dist_on else 1

    # Prepare inputs
    flat_ids = input_ids.view(-1, L)  # [2B, L]
    flat_attn = attention_mask.view(-1, L)  # [2B, L]
    flat_tok = token_type_ids.view(-1, L) if token_type_ids is not None else None
    need_hs = cls.pooler_type in ['avg_top2', 'avg_first_last']

    def _encode_with_dropout_control(ids, attn, tok, use_dropout=True, custom_dropout_rate=None):
        """Encode with explicit dropout control and optional custom dropout rate"""
        if use_dropout:
            encoder.train()  # Enable dropout
            
            # Set custom dropout rate if provided
            if custom_dropout_rate is not None:
                # Store original dropout rates for restoration
                original_rates = {}
                
                # Set dropout rate for attention dropout
                for layer in encoder.encoder.layer:
                    if hasattr(layer.attention.self, 'dropout'):
                        original_rates[f'attn_{id(layer)}'] = layer.attention.self.dropout.p
                        layer.attention.self.dropout.p = custom_dropout_rate
                    if hasattr(layer.attention.output, 'dropout'):
                        original_rates[f'attn_out_{id(layer)}'] = layer.attention.output.dropout.p
                        layer.attention.output.dropout.p = custom_dropout_rate
                        
                # Set dropout rate for feed-forward dropout
                for layer in encoder.encoder.layer:
                    if hasattr(layer.output, 'dropout'):
                        original_rates[f'ff_{id(layer)}'] = layer.output.dropout.p
                        layer.output.dropout.p = custom_dropout_rate
                
                # Set dropout rate for embedding dropout
                if hasattr(encoder.embeddings, 'dropout'):
                    original_rates['emb'] = encoder.embeddings.dropout.p
                    encoder.embeddings.dropout.p = custom_dropout_rate
                    
                try:
                    # Forward pass with custom dropout rates
                    outputs = encoder(
                        input_ids=ids,
                        attention_mask=attn,
                        token_type_ids=tok,
                        output_hidden_states=need_hs,
                        return_dict=True,
                    )
                finally:
                    # Restore original dropout rates
                    for layer in encoder.encoder.layer:
                        if hasattr(layer.attention.self, 'dropout') and f'attn_{id(layer)}' in original_rates:
                            layer.attention.self.dropout.p = original_rates[f'attn_{id(layer)}']
                        if hasattr(layer.attention.output, 'dropout') and f'attn_out_{id(layer)}' in original_rates:
                            layer.attention.output.dropout.p = original_rates[f'attn_out_{id(layer)}']
                        if hasattr(layer.output, 'dropout') and f'ff_{id(layer)}' in original_rates:
                            layer.output.dropout.p = original_rates[f'ff_{id(layer)}']
                    if hasattr(encoder.embeddings, 'dropout') and 'emb' in original_rates:
                        encoder.embeddings.dropout.p = original_rates['emb']
            else:
                # Use default dropout rates
                outputs = encoder(
                    input_ids=ids,
                    attention_mask=attn,
                    token_type_ids=tok,
                    output_hidden_states=need_hs,
                    return_dict=True,
                )
        else:
            encoder.eval()   # Disable dropout
            outputs = encoder(
                input_ids=ids,
                attention_mask=attn,
                token_type_ids=tok,
                output_hidden_states=need_hs,
                return_dict=True,
            )
        
        # Get pooled representation
        pooled = cls.pooler(attn, outputs)
        if cls.pooler_type == "cls":
            pooled = cls.mlp(pooled)
        
        return pooled

    # Strategy 1: Standard SimCSE-style with dropout for both views
    if need_dropout is None or need_dropout.all():
        # Both views use dropout (standard SimCSE)
        z_both_views = _encode_with_dropout_control(flat_ids, flat_attn, flat_tok, use_dropout=True, custom_dropout_rate=dropout_rate)
        z_both_views = z_both_views.view(B, 2, -1)  # [B, 2, D]
        z1, z2 = z_both_views[:, 0], z_both_views[:, 1]  # [B, D] each
    else:
        # Mixed strategy: some samples need dropout, others have pre-augmented data
        # Split samples based on need_dropout flag
        dropout_mask = need_dropout.bool()
        no_dropout_mask = ~dropout_mask
        
        if dropout_mask.any():
            # Process samples that need dropout augmentation
            dropout_ids = input_ids[dropout_mask]  # [B_drop, 2, L]
            dropout_attn = attention_mask[dropout_mask]  # [B_drop, 2, L]
            dropout_tok = token_type_ids[dropout_mask] if token_type_ids is not None else None
            
            # Encode both views with dropout
            dropout_flat_ids = dropout_ids.view(-1, L)
            dropout_flat_attn = dropout_attn.view(-1, L)
            dropout_flat_tok = dropout_tok.view(-1, L) if dropout_tok is not None else None
            
            z_dropout = _encode_with_dropout_control(
                dropout_flat_ids, dropout_flat_attn, dropout_flat_tok, use_dropout=True, custom_dropout_rate=dropout_rate
            )
            z_dropout = z_dropout.view(-1, 2, z_dropout.size(-1))  # [B_drop, 2, D]
            
            z1_dropout, z2_dropout = z_dropout[:, 0], z_dropout[:, 1]
        else:
            z1_dropout, z2_dropout = None, None
            
        if no_dropout_mask.any():
            # Process samples with pre-augmented data (no additional dropout needed)
            no_dropout_ids = input_ids[no_dropout_mask]  # [B_no_drop, 2, L]
            no_dropout_attn = attention_mask[no_dropout_mask]  # [B_no_drop, 2, L]
            no_dropout_tok = token_type_ids[no_dropout_mask] if token_type_ids is not None else None
            
            # Encode both views without dropout (data is already augmented)
            no_dropout_flat_ids = no_dropout_ids.view(-1, L)
            no_dropout_flat_attn = no_dropout_attn.view(-1, L)
            no_dropout_flat_tok = no_dropout_tok.view(-1, L) if no_dropout_tok is not None else None
            
            z_no_dropout = _encode_with_dropout_control(
                no_dropout_flat_ids, no_dropout_flat_attn, no_dropout_flat_tok, use_dropout=False, custom_dropout_rate=None
            )
            z_no_dropout = z_no_dropout.view(-1, 2, z_no_dropout.size(-1))  # [B_no_drop, 2, D]
            
            z1_no_dropout, z2_no_dropout = z_no_dropout[:, 0], z_no_dropout[:, 1]
        else:
            z1_no_dropout, z2_no_dropout = None, None
        
        # Combine results in original order
        embed_dim = (z1_dropout.size(-1) if z1_dropout is not None else 
                    z1_no_dropout.size(-1) if z1_no_dropout is not None else 
                    cls.config.hidden_size)
        z1 = torch.zeros(B, embed_dim, device=device, dtype=torch.float32)
        z2 = torch.zeros_like(z1)
        
        if z1_dropout is not None:
            z1[dropout_mask] = z1_dropout
            z2[dropout_mask] = z2_dropout
        if z1_no_dropout is not None:
            z1[no_dropout_mask] = z1_no_dropout
            z2[no_dropout_mask] = z2_no_dropout

    # Distributed gathering while preserving gradients
    def gather_with_grad(tensor):
        if not dist_on or world_sz == 1:
            return tensor
        
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_sz)]
        dist.all_gather(gathered_tensors, tensor.contiguous())
        # Replace current rank's tensor to preserve gradients
        gathered_tensors[rank] = tensor
        return torch.cat(gathered_tensors, dim=0)

    # Normalize and gather embeddings
    z1_norm = F.normalize(gather_with_grad(z1), dim=-1)  # [B*world_sz, D] 
    z2_norm = F.normalize(gather_with_grad(z2), dim=-1)  # [B*world_sz, D]

    # Compute InfoNCE loss
    # ============================================================================
    # Each z1[i] is positive with z2[i], negative with z1[j] for j≠i
    # All in-batch negatives are valid, only mask diagonal (self-similarities)
    # ============================================================================

    # Standard SimCSE InfoNCE loss with all in-batch negatives from view1
    # Each z1[i] is positive with z2[i], negative with z1[j] for all j != i
    batch_size_total = z1_norm.size(0)
    
    # Compute positive similarities: each z1[i] with its corresponding z2[i]
    pos_sim = torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True) / cls.model_args.temperature  # [B, 1]
    
    # Compute negative similarities: each z1[i] with all other z1[j] where j != i
    neg_sim = torch.matmul(z1_norm, z1_norm.T) / cls.model_args.temperature  # [B, B]
    
    # Mask out self-similarities (diagonal elements) in negatives
    # In standard SimCSE, all off-diagonal elements are valid negatives
    self_mask = torch.eye(batch_size_total, device=device, dtype=torch.bool)
    # Use -1e4 instead of -1e9 to avoid float16 overflow
    neg_sim.masked_fill_(self_mask, -1e4)  # Large negative value to exclude self-similarities
    
    # Combine positive and negative logits
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1 + B]
    
    # Labels: positive is always at index 0
    labels = torch.zeros(batch_size_total, dtype=torch.long, device=device)
    contrastive_loss = F.cross_entropy(logits, labels, reduction='mean')
    
    # Ensure loss is always a scalar
    if contrastive_loss.numel() > 1:
        contrastive_loss = contrastive_loss.mean()

    # Apply InfoNCE weight
    final_loss = cls.model_args.infonce_weight * contrastive_loss

    # ------------------------- Regularization Losses -------------------------
    reg_loss = None
    reg_loss_details = {}
    
    # Add Barlow Twins loss
    if cls.model_args.regularization == "barlow_twins":
        # Use embeddings from both views (before normalization and gathering)
        z1_bt = z1  # First view [B, D]
        z2_bt = z2  # Second view [B, D]
        
        # Apply Barlow Twins projector to get both projected and normalized embeddings
        z1_projected, z1_normalized = cls.mlp_reg(z1_bt)
        z2_projected, z2_normalized = cls.mlp_reg(z2_bt)
        
        # Distributed gathering for Barlow Twins computation
        if dist_on:
            z1_norm_all = F.normalize(gather_with_grad(z1_normalized), dim=-1)
            z2_norm_all = F.normalize(gather_with_grad(z2_normalized), dim=-1)
        else:
            z1_norm_all = F.normalize(z1_normalized, dim=-1)
            z2_norm_all = F.normalize(z2_normalized, dim=-1)
        
        # Compute Barlow Twins loss using normalized embeddings
        bt_loss, on_diag, off_diag = barlow_twins_loss(
            z1_norm_all, z2_norm_all, lambd=getattr(cls.model_args, 'barlow_twins_lambda', 5e-3)
        )
        reg_loss = bt_loss
        reg_loss_details = {
            "barlow_twins_loss": bt_loss,
            "on_diagonal": on_diag,
            "off_diagonal": off_diag
        }
        
        # Add to final loss
        final_loss += cls.model_args.regularization_weight * reg_loss

    # Add VICReg loss
    elif cls.model_args.regularization == "vicreg":
        # Use embeddings from both views (before normalization and gathering)
        z1_vic = z1  # First view [B, D]
        z2_vic = z2  # Second view [B, D]
        
        # Apply VICReg projector
        z1_projected = cls.mlp_reg(z1_vic)
        z2_projected = cls.mlp_reg(z2_vic)
        
        # Distributed gathering for VICReg computation
        if dist_on:
            z1_vic_all = gather_with_grad(z1_projected)
            z2_vic_all = gather_with_grad(z2_projected)
        else:
            z1_vic_all = z1_projected
            z2_vic_all = z2_projected
        
        # Compute VICReg loss using projected embeddings
        reg_loss, loss_inv, loss_var, loss_cov = vicreg_loss_fn(
            z1_vic_all, z2_vic_all, 
            invariance_weight=cls.model_args.vicreg_invariance_weight, 
            variance_weight=cls.model_args.vicreg_variance_weight, 
            covariance_weight=cls.model_args.vicreg_covariance_weight, 
            gamma=cls.model_args.vicreg_gamma,
            return_details=True
        )

        reg_loss_details = {
            "vicreg_loss": reg_loss,
            "invariance_loss": loss_inv,
            "variance_loss": loss_var,
            "covariance_loss": loss_cov
        }
        
        # Add to final loss
        final_loss += cls.model_args.regularization_weight * reg_loss

    # ------------------------- MLM Loss (Memory-Optimized Strategy) -------------------------
    masked_lm_loss = None
    if cls.model_args.do_mlm and mlm_input_ids is not None and mlm_labels is not None:
        # MLM strategy is controlled by data collator, but we handle the computation here
        # The collator ensures that only relevant views have valid labels (!= -100)
        
        # Count valid MLM positions across all views
        valid_mlm_positions = (mlm_labels != -100).sum()
        
        if valid_mlm_positions > 0:
            # Memory optimization: process MLM in smaller chunks to reduce peak memory usage
            chunk_size = min(B, 8)  # Process maximum 8 samples at a time
            total_mlm_loss = 0.0
            total_chunks = 0
            
            # Set encoder to training mode once before processing chunks
            encoder.train()
            
            # Process MLM inputs in chunks
            for chunk_start in range(0, B * 2, chunk_size):
                chunk_end = min(chunk_start + chunk_size, B * 2)
                
                # Extract chunk data
                chunk_mlm_ids = mlm_input_ids.view(-1, L)[chunk_start:chunk_end]  # [chunk_size, L]
                chunk_mlm_labels = mlm_labels.view(-1)[chunk_start * L:chunk_end * L]  # [chunk_size * L]
                chunk_mlm_attn = attention_mask.view(-1, L)[chunk_start:chunk_end]  # [chunk_size, L]
                chunk_mlm_tok = token_type_ids.view(-1, L)[chunk_start:chunk_end] if token_type_ids is not None else None
                
                # Only compute if this chunk has valid masks
                chunk_masked_indices = chunk_mlm_labels != -100
                if chunk_masked_indices.sum() == 0:
                    continue
                
                try:
                    # Use gradient checkpointing to save memory
                    if hasattr(encoder, 'gradient_checkpointing') and encoder.gradient_checkpointing:
                        # Already enabled, just do forward pass
                        chunk_mlm_outputs = encoder(
                            input_ids=chunk_mlm_ids,
                            attention_mask=chunk_mlm_attn,
                            token_type_ids=chunk_mlm_tok,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    else:
                        # Enable gradient checkpointing temporarily for this chunk
                        checkpoint_enabled = torch.utils.checkpoint.checkpoint
                        chunk_mlm_outputs = checkpoint_enabled(
                            encoder,
                            chunk_mlm_ids,
                            chunk_mlm_attn,
                            chunk_mlm_tok,
                            use_reentrant=False
                        )
                        if not hasattr(chunk_mlm_outputs, 'last_hidden_state'):
                            # Fallback to regular forward pass if checkpointing fails
                            chunk_mlm_outputs = encoder(
                                input_ids=chunk_mlm_ids,
                                attention_mask=chunk_mlm_attn,
                                token_type_ids=chunk_mlm_tok,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=True,
                            )
                    
                    # Get hidden states and compute MLM predictions for this chunk
                    chunk_hidden_states = chunk_mlm_outputs.last_hidden_state.view(-1, chunk_mlm_outputs.last_hidden_state.size(-1))
                    chunk_prediction_scores = cls.lm_head(chunk_hidden_states[chunk_masked_indices])
                    
                    # Compute MLM loss for this chunk
                    loss_fct = nn.CrossEntropyLoss()
                    chunk_mlm_loss = loss_fct(chunk_prediction_scores, chunk_mlm_labels[chunk_masked_indices])
                    
                    # Accumulate loss (weighted by number of masked tokens in this chunk)
                    chunk_weight = chunk_masked_indices.sum().float() / valid_mlm_positions.float()
                    total_mlm_loss += chunk_mlm_loss * chunk_weight
                    total_chunks += 1
                    
                finally:
                    # Aggressive memory cleanup after each chunk
                    if 'chunk_hidden_states' in locals():
                        del chunk_hidden_states
                    if 'chunk_prediction_scores' in locals():
                        del chunk_prediction_scores
                    if 'chunk_mlm_outputs' in locals():
                        del chunk_mlm_outputs
                    if 'chunk_mlm_ids' in locals():
                        del chunk_mlm_ids, chunk_mlm_labels, chunk_mlm_attn
                        if chunk_mlm_tok is not None:
                            del chunk_mlm_tok
                    cleanup_memory()
            
            if total_chunks > 0:
                masked_lm_loss = total_mlm_loss
                
                # Synchronize MLM loss across devices if distributed
                if dist_on:
                    dist.all_reduce(masked_lm_loss, op=dist.ReduceOp.SUM)
                    masked_lm_loss /= world_sz

                # Add to final loss
                final_loss += cls.model_args.mlm_weight * masked_lm_loss

    # Return results: ensure loss is always a scalar (e.g. for DDP/DeepSpeed)
    if final_loss.numel() > 1:
        final_loss = final_loss.mean()

    if not return_dict:
        return (final_loss,)

    return ContrastiveLearningOutput(
        loss=final_loss,
        loss_MLM=masked_lm_loss if (cls.model_args.do_mlm and masked_lm_loss is not None) else None,
        loss_reg=reg_loss,
        contrastive_loss=contrastive_loss,  # Add InfoNCE loss for logging
        # Barlow Twins specific losses
        barlow_twins_loss=reg_loss_details.get("barlow_twins_loss"),
        on_diagonal=reg_loss_details.get("on_diagonal"),
        off_diagonal=reg_loss_details.get("off_diagonal"),
        # VICReg specific losses
        vicreg_loss=reg_loss_details.get("vicreg_loss"),
        invariance_loss=reg_loss_details.get("invariance_loss"),
        variance_loss=reg_loss_details.get("variance_loss"),
        covariance_loss=reg_loss_details.get("covariance_loss"),
    )


def sentemb_forward( 
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    """
    This function is used to get sentence embeddings from BERT/RoBERTa's CLS representation, because the CLS representation is not used in the contrastive learning task.
    args:
        cls: model class
        encoder: BERT/RoBERTa encoder
        input_ids: input ids
        attention_mask: attention mask
        token_type_ids: token type ids
        position_ids: position ids
        head_mask: head mask
        inputs_embeds: input embeddings
        labels: labels
        output_attentions: output attentions
        output_hidden_states: output hidden states
        return_dict: return dict

    return:
        outputs[0]: last hidden states
        pooler_output: pooler output
        outputs[2:]: hidden states
    """

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )



class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward_from_embeddings(self, embeddings, attention_mask=None, token_type_ids=None, 
                                position_ids=None, head_mask=None, output_attentions=None, 
                                output_hidden_states=None, return_dict=None):
        """
        Runs the precomputed embeddings through the BERT encoder.
        """
        # Get the extended attention mask from the embeddings shape.
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, embeddings.size()[:2], embeddings.device
        )
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        # Wrap outputs into a BaseModelOutputWithPoolingAndCrossAttentions object.
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, "hidden_states") else None,
            attentions=encoder_outputs.attentions if hasattr(encoder_outputs, "attentions") else None,
            pooler_output=None,
        )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        need_dropout=None,
        dropout_rate=None,
    ):  
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # For inference: if input_ids is 2D [B, L], return raw encoder outputs
        elif input_ids is not None and input_ids.dim() == 2:
            return self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # For training: input_ids is 3D [B, 2, L] for contrastive learning
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                need_dropout=need_dropout,
                dropout_rate=dropout_rate,
            )
