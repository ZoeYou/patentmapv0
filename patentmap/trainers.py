import math
import sys
import os
import gc
import json
import random
import time


from transformers import Trainer
from transformers.utils import logging
from transformers.trainer_callback import TrainerState

import dataclasses

from transformers.file_utils import is_apex_available
import torch
import torch.distributed as dist
from typing import Dict, List, Optional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Sampler
from bisect import bisect_right


# Set path to patentmap_eval
PATH_TO_PATENTEVAL = './patentmap_eval'
PATH_TO_DATA = './patentmap_eval/data'
TRAINER_STATE_NAME = "trainer_state.json"

# Import patentmap_eval
sys.path.insert(0, PATH_TO_PATENTEVAL)
import patenteval
import numpy as np
from datetime import datetime
from filelock import FileLock


logger = logging.get_logger(__name__)

# ===========================
# Contrastive Learning Diagnostics
# ===========================

@torch.no_grad()
def encode_with_dropout(model, batch, device, use_dropout=True):
    """
    Encode a batch with optional dropout control.
    Assumes batch is already on CPU; we move tensors to device here.
    Returns L2-normalized embeddings (pooled outputs).
    
    Args:
        model: The model to use for encoding
        batch: Input batch dictionary
        device: Device to run on
        use_dropout: If True, enable dropout. If False, disable dropout.
    """
    if use_dropout:
        model.train()  # keep dropout active
    else:
        model.eval()   # disable dropout
    
    # Ensure batch data is in the correct format - fail fast if incorrect
    batch_on_device = {}
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # Handle both [B, L] and [B, 2, L] shapes
            if v.dim() == 3:
                # Take first view: [B, 2, L] -> [B, L]
                batch_on_device[k] = v[:, 0, :].to(device)
            else:
                # Direct tensor - move to device
                batch_on_device[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0:
            # Handle list - try to get first element if it's a tensor
            first_item = v[0]
            if isinstance(first_item, torch.Tensor):
                batch_on_device[k] = first_item.to(device)
            else:
                raise ValueError(f"Key {k}: list contains non-tensor {type(first_item)}. Expected tensor data.")
        else:
            raise ValueError(f"Key {k}: unsupported type {type(v)}. Expected tensor or list of tensors.")
    
    if not batch_on_device:
        raise ValueError("No valid tensor data found in batch. Check tokenization and data preprocessing.")
    
    # Forward pass - let any model errors propagate
    outputs = model(**batch_on_device, output_hidden_states=True, return_dict=True, sent_emb=True)
    
    # Validate model outputs - fail fast if unexpected structure
    if not hasattr(outputs, 'pooler_output'):
        if hasattr(outputs, 'last_hidden_state'):
            logger.warning("Model output missing pooler_output, using last_hidden_state[:, 0] as fallback")
            z = outputs.last_hidden_state[:, 0]  # [B, d]
        else:
            available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
            raise ValueError(
                f"Model output missing both pooler_output and last_hidden_state. "
                f"Available attributes: {available_attrs}. Check model configuration."
            )
    else:
        # Use pooler_output for embeddings
        z = outputs.pooler_output  # [B, d]
        
    z = torch.nn.functional.normalize(z, dim=-1)  # L2-normalize
    return z


@torch.no_grad()
def compute_positive_cosine(model, probe_loader, device, max_batches=None, use_dropout=True):
    """
    Compute mean cos(z1, z2) over the probe set.
    
    Args:
        model: The model to evaluate
        probe_loader: DataLoader with probe data
        device: Device to run on
        max_batches: Maximum number of batches to process
        use_dropout: If True, apply dropout augmentation (for dropout training method)
                    If False, use pre-prepared two views (for section_pair method)
    """
    original_training_state = model.training
    model.eval()  # we'll manually control dropout inside encode_with_dropout if needed
    cos_values = []

    for step, batch in enumerate(probe_loader):
        if max_batches is not None and step >= max_batches:
            break
        
        # Validate batch structure - fail fast if incorrect
        if not isinstance(batch, dict):
            raise ValueError(f"Expected dict batch, got {type(batch)}. DataLoader configuration is incorrect.")
            
        # Debug batch structure for first batch
        if step == 0:
            logger.debug(f"Batch keys: {list(batch.keys())}")
            for k, v in batch.items():
                logger.debug(f"Key {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")
        
        # Check if data is already paired (section_pair) or single text (dropout)
        input_ids = batch['input_ids']
        
        # Data collator always produces [B, 2, L] format regardless of augmentation method
        if input_ids.dim() != 3:
            raise ValueError(f"Expected input_ids shape [B, 2, L], got {input_ids.shape}. Data collator may be misconfigured.")
        
        # Extract the two views
        batch_view1 = {k: v[:, 0, :] if v.dim() == 3 else v for k, v in batch.items()}
        batch_view2 = {k: v[:, 1, :] if v.dim() == 3 else v for k, v in batch.items()}
        
        # Determine which samples need dropout augmentation
        # If views are identical, we need dropout to create positive pairs
        # If views are different, they're already augmented (section_pair)
        view1_ids = batch_view1['input_ids'].to(device)
        view2_ids = batch_view2['input_ids'].to(device)
        need_dropout_mask = torch.all(view1_ids == view2_ids, dim=1)  # [B], True if views are identical
        
        # Encode based on whether each sample needs dropout
        if use_dropout:
            # For dropout training: apply dropout when views are identical
            z1_list = []
            z2_list = []
            
            for i in range(len(need_dropout_mask)):
                sample_view1 = {k: v[i:i+1] for k, v in batch_view1.items()}
                sample_view2 = {k: v[i:i+1] for k, v in batch_view2.items()}
                
                if need_dropout_mask[i]:
                    # Views are identical, need dropout
                    z1_i = encode_with_dropout(model, sample_view1, device, use_dropout=True)
                    z2_i = encode_with_dropout(model, sample_view2, device, use_dropout=True)
                else:
                    # Views are different, use as-is
                    z1_i = encode_with_dropout(model, sample_view1, device, use_dropout=False)
                    z2_i = encode_with_dropout(model, sample_view2, device, use_dropout=False)
                
                z1_list.append(z1_i)
                z2_list.append(z2_i)
            
            z1 = torch.cat(z1_list, dim=0)
            z2 = torch.cat(z2_list, dim=0)
        else:
            # Section_pair training: never apply dropout, views are already different
            model.eval()
            z1 = encode_with_dropout(model, batch_view1, device, use_dropout=False)  # [B, d]
            z2 = encode_with_dropout(model, batch_view2, device, use_dropout=False)  # [B, d]

        cos = (z1 * z2).sum(dim=-1)  # cosine since already L2-normalized
        cos_values.append(cos.cpu())

    if not cos_values:
        raise RuntimeError("No cosine values computed - probe_loader is empty or max_batches=0")

    cos_values = torch.cat(cos_values, dim=0)
    result = cos_values.mean().item()
    
    # Restore training mode before returning
    model.train(original_training_state)
    return result


@torch.no_grad()
def compute_gradient_metrics(model, probe_loader, device, tau=0.05, max_batches=None, use_dropout=True):
    """
    Compute InfoNCE gradient-based metrics:
      - positive_prob (p_i+): mean probability assigned to positive pairs
      - max_neg_prob: mean of max probability among negatives per anchor
      - grad_alignment_ratio: ratio of positive vs negative gradient term magnitudes
      - effective_num_negatives: entropy-based measure of contributing negatives
      - hard_negative_fraction: percentage of negatives with p_ij > threshold
    
    Based on InfoNCE gradient:
      ∇_hi L_i = (1 - p_i+)(h_i - h_i+) - Σ_j≠i p_ij(h_i - h_j+)
    
    Args:
        model: The model to evaluate
        probe_loader: DataLoader with probe data
        device: Device to run on
        tau: Temperature parameter
        max_batches: Maximum number of batches to process
        use_dropout: If True, apply dropout augmentation (for dropout training method)
                    If False, use pre-prepared two views (for section_pair method)
    """
    original_training_state = model.training
    model.eval()
    
    pos_probs = []
    max_neg_probs = []
    grad_ratios = []
    eff_num_negs = []
    hard_neg_fractions = []
    
    hard_neg_threshold = 0.1  # Threshold for "hard" negatives
    
    for step, batch in enumerate(probe_loader):
        if max_batches is not None and step >= max_batches:
            break
            
        if not isinstance(batch, dict):
            raise ValueError(f"Expected dict batch, got {type(batch)}. DataLoader configuration is incorrect.")
        
        # Data collator always produces [B, 2, L] format regardless of augmentation method
        input_ids = batch['input_ids']
        
        if input_ids.dim() != 3:
            raise ValueError(f"Expected input_ids shape [B, 2, L], got {input_ids.shape}. Data collator may be misconfigured.")
        
        # Extract the two views
        batch_view1 = {k: v[:, 0, :] if v.dim() == 3 else v for k, v in batch.items()}
        batch_view2 = {k: v[:, 1, :] if v.dim() == 3 else v for k, v in batch.items()}
        
        # Determine which samples need dropout augmentation
        # If views are identical, we need dropout to create positive pairs
        view1_ids = batch_view1['input_ids'].to(device)
        view2_ids = batch_view2['input_ids'].to(device)
        need_dropout_mask = torch.all(view1_ids == view2_ids, dim=1)  # [B], True if views are identical
        
        # Encode based on augmentation method and whether views are identical
        if use_dropout:
            # For dropout training: apply dropout when views are identical
            z1_list = []
            z2_list = []
            
            for i in range(len(need_dropout_mask)):
                sample_view1 = {k: v[i:i+1] for k, v in batch_view1.items()}
                sample_view2 = {k: v[i:i+1] for k, v in batch_view2.items()}
                
                if need_dropout_mask[i]:
                    # Views are identical, need dropout
                    z1_i = encode_with_dropout(model, sample_view1, device, use_dropout=True)
                    z2_i = encode_with_dropout(model, sample_view2, device, use_dropout=True)
                else:
                    # Views are different, use as-is
                    z1_i = encode_with_dropout(model, sample_view1, device, use_dropout=False)
                    z2_i = encode_with_dropout(model, sample_view2, device, use_dropout=False)
                
                z1_list.append(z1_i)
                z2_list.append(z2_i)
            
            z1 = torch.cat(z1_list, dim=0)  # [B, d] - anchors
            z2 = torch.cat(z2_list, dim=0)  # [B, d] - positives
        else:
            # Section_pair training: never apply dropout, views are already different
            model.eval()
            z1 = encode_with_dropout(model, batch_view1, device, use_dropout=False)  # [B, d] - anchors
            z2 = encode_with_dropout(model, batch_view2, device, use_dropout=False)  # [B, d] - positives
        
        B = z1.size(0)
        
        if B < 2:
            raise ValueError(f"Batch size {B} too small. Need at least 2 samples.")
        
        # Compute similarity matrix: anchor z1[i] with all positives z2[j]
        # This matches InfoNCE where we compare h_i with all h_j+
        sim_matrix = (z1 @ z2.t()) / tau  # [B, B]
        
        # Compute probabilities via softmax over all samples
        logits = sim_matrix  # [B, B]
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [B, B]
        
        # Extract diagonal (positive pairs) - this is p_i+
        positive_prob = torch.diag(probs)  # [B]
        pos_probs.append(positive_prob.cpu())
        
        # For each anchor, get max probability among negatives (off-diagonal)
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        neg_probs = probs.masked_fill(~mask, 0.0)  # Zero out diagonal
        max_neg_prob = neg_probs.max(dim=-1)[0]  # [B]
        max_neg_probs.append(max_neg_prob.cpu())
        
        # Compute gradient term magnitudes
        # Positive term magnitude: ||(1 - p_i+)(h_i - h_i+)||
        # For normalized embeddings with cosine similarity, we can approximate:
        # The magnitude is proportional to (1 - p_i+) * distance
        # Since embeddings are normalized, ||h_i - h_i+||^2 ≈ 2(1 - cos(h_i, h_i+))
        pos_cos = torch.diag(z1 @ z2.t())  # cosine similarity to positive
        pos_distance = torch.sqrt(2 * (1 - pos_cos).clamp(min=1e-8))  # [B]
        pos_term_mag = (1 - positive_prob) * pos_distance  # [B]
        
        # Negative term magnitude: ||Σ_j≠i p_ij(h_i - h_j+)||
        # Sum of weighted distances to negatives
        neg_term_mag = torch.zeros(B, device=device)
        for i in range(B):
            neg_mask_i = torch.ones(B, dtype=torch.bool, device=device)
            neg_mask_i[i] = False
            # Weighted sum of distances to negatives
            for j in range(B):
                if i != j:
                    dist_ij = torch.norm(z1[i] - z2[j])
                    neg_term_mag[i] += probs[i, j] * dist_ij
        
        # Gradient alignment ratio: positive_term / negative_term
        # Higher ratio means gradient is more focused on positive alignment
        ratio = pos_term_mag / (neg_term_mag + 1e-8)
        grad_ratios.append(ratio.cpu())
        
        # Effective number of negatives: exp(entropy of negative distribution)
        # Compute entropy over negatives only (excluding diagonal)
        neg_probs_normalized = probs.clone()
        neg_probs_normalized = neg_probs_normalized.masked_fill(~mask, 0.0)
        # Renormalize to get proper distribution over negatives
        neg_probs_sum = neg_probs_normalized.sum(dim=-1, keepdim=True)
        neg_probs_normalized = neg_probs_normalized / (neg_probs_sum + 1e-12)
        
        # Compute entropy of negative distribution
        eps = 1e-12
        log_probs = torch.where(
            neg_probs_normalized > eps,
            torch.log(neg_probs_normalized + eps),
            torch.zeros_like(neg_probs_normalized)
        )
        neg_entropy = -torch.sum(neg_probs_normalized * log_probs, dim=-1)  # [B]
        effective_negs = torch.exp(neg_entropy)  # [B]
        eff_num_negs.append(effective_negs.cpu())
        
        # Hard negative fraction: percentage of negatives with prob > threshold
        hard_negs = (neg_probs > hard_neg_threshold).float()
        hard_neg_count = hard_negs.sum(dim=-1)  # [B]
        hard_neg_frac = hard_neg_count / (B - 1)  # Normalize by number of negatives
        hard_neg_fractions.append(hard_neg_frac.cpu())
    
    if not pos_probs:
        raise RuntimeError("No gradient metrics computed - probe_loader is empty or max_batches=0")
    
    # Aggregate results
    results = {
        'positive_prob': torch.cat(pos_probs).mean().item(),
        'max_neg_prob': torch.cat(max_neg_probs).mean().item(),
        'grad_alignment_ratio': torch.cat(grad_ratios).mean().item(),
        'effective_num_negatives': torch.cat(eff_num_negs).mean().item(),
        'hard_negative_fraction': torch.cat(hard_neg_fractions).mean().item(),
    }
    
    model.train(original_training_state)
    return results


@torch.no_grad()
def compute_negative_entropy(model, probe_loader, device, tau=0.05, max_batches=None, use_dropout=True):
    """
    Compute normalized negative entropy over in-batch negatives:
      - For each anchor i, logits l_ij = (z_i · z_j) / tau, j != i
      - p_ij = softmax_j(l_ij)
      - H_i = -Σ_j p_ij log p_ij
      - H_i_normalized = H_i / log(B-1)
    Returns mean normalized entropy over anchors.
    
    Args:
        model: The model to evaluate
        probe_loader: DataLoader with probe data
        device: Device to run on
        tau: Temperature parameter
        max_batches: Maximum number of batches to process
        use_dropout: If True, apply dropout augmentation (for dropout training method)
                    If False, use pre-prepared two views (for section_pair method)
    """
    original_training_state = model.training
    model.eval()
    entropies = []
    debug_info = {}

    for step, batch in enumerate(probe_loader):
        if max_batches is not None and step >= max_batches:
            break
            
        # Validate batch structure - fail fast if incorrect
        if not isinstance(batch, dict):
            raise ValueError(f"Expected dict batch, got {type(batch)}. DataLoader configuration is incorrect.")

        # Data collator always produces [B, 2, L] format regardless of augmentation method
        input_ids = batch['input_ids']
        
        if input_ids.dim() != 3:
            raise ValueError(f"Expected input_ids shape [B, 2, L], got {input_ids.shape}. Data collator may be misconfigured.")
        
        # Extract both views to check if they're identical
        batch_view1 = {k: v[:, 0, :] if v.dim() == 3 else v for k, v in batch.items()}
        batch_view2 = {k: v[:, 1, :] if v.dim() == 3 else v for k, v in batch.items()}
        
        # Determine which samples need dropout augmentation
        view1_ids = batch_view1['input_ids'].to(device)
        view2_ids = batch_view2['input_ids'].to(device)
        need_dropout_mask = torch.all(view1_ids == view2_ids, dim=1)  # [B], True if views are identical
        
        # Encode based on augmentation method and whether views are identical
        if use_dropout:
            # For dropout training: apply dropout when views are identical
            z_list = []
            
            for i in range(len(need_dropout_mask)):
                sample_view1 = {k: v[i:i+1] for k, v in batch_view1.items()}
                
                if need_dropout_mask[i]:
                    # Views are identical, need dropout
                    z_i = encode_with_dropout(model, sample_view1, device, use_dropout=True)
                else:
                    # Views are different, use as-is
                    z_i = encode_with_dropout(model, sample_view1, device, use_dropout=False)
                
                z_list.append(z_i)
            
            z = torch.cat(z_list, dim=0)  # [B, d]
        else:
            # Section_pair training: never apply dropout, views are already different
            model.eval()
            z = encode_with_dropout(model, batch_view1, device, use_dropout=False)  # [B, d]
        
        B = z.size(0)

        if B < 2:
            raise ValueError(f"Batch size {B} too small for entropy computation. Need at least 2 samples.")

        # Check embedding similarity - this is the likely cause of entropy=0
        pairwise_cosine = z @ z.t()  # [B, B] cosine similarities
        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        off_diag_cosines = pairwise_cosine[off_diag_mask]
        
        # Early numerical check for extreme similarities
        if torch.isnan(pairwise_cosine).any() or torch.isinf(pairwise_cosine).any():
            raise RuntimeError(f"NaN/Inf detected in pairwise cosine similarities. Model outputs may be corrupted.")
        
        # Cosine sim = dot product because z is normalized
        logits = (z @ z.t()) / tau  # [B, B]
        
        # Check for extreme logit values that could cause numerical instability
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(f"NaN/Inf detected in logits after temperature scaling (tau={tau}).")
        
        # Additional check for extreme values
        logit_max = logits.max().item()
        logit_min = logits.min().item()
        if logit_max > 1000 or logit_min < -1000:
            print(f"WARNING: Extreme logit values detected: min={logit_min:.2f}, max={logit_max:.2f}")
            print(f"  This may cause numerical instability in softmax computation.")

        # Mask self-similarity on diagonal
        mask = torch.eye(B, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(mask, float('-inf'))
        
        # Check logits after masking
        if torch.isnan(logits).any():
            raise RuntimeError("NaN detected in logits after diagonal masking.")

        # Softmax over negatives j != i with numerical stability
        # Subtract max for numerical stability (standard softmax trick)
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        logits_max = torch.where(torch.isinf(logits_max), torch.zeros_like(logits_max), logits_max)
        logits_stable = logits - logits_max
        
        probs = torch.nn.functional.softmax(logits_stable, dim=-1)  # [B, B]
        
        # Check probabilities for numerical issues
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"WARNING: NaN/Inf in probabilities after softmax")
            print(f"  logits_stable range: [{logits_stable.min():.2f}, {logits_stable.max():.2f}]")
            print(f"  probs range: [{probs.min():.6f}, {probs.max():.6f}]")
            # Replace nan/inf with uniform distribution as fallback
            uniform_prob = 1.0 / (B - 1)
            probs = torch.where(torch.isnan(probs) | torch.isinf(probs), 
                              uniform_prob, probs)
            # Re-normalize to ensure probabilities sum to 1
            prob_sums = probs.sum(dim=-1, keepdim=True)
            probs = probs / prob_sums
        
        # Check probability concentration (main cause of entropy=0)
        prob_max = probs.max(dim=-1)[0]
        concentrated_count = (prob_max > 0.95).sum().item()

        # Entropy per anchor with numerical stability
        # Mask out diagonal entries for entropy computation
        entropy_mask = ~torch.eye(B, dtype=torch.bool, device=probs.device)
        masked_probs = probs * entropy_mask.float()  # Zero out diagonal
        
        # Compute entropy with better numerical stability
        # Only compute log for non-zero probabilities
        eps = 1e-12
        log_probs = torch.where(
            masked_probs > eps,
            torch.log(masked_probs),
            torch.zeros_like(masked_probs)
        )
        entropy = -torch.sum(masked_probs * log_probs, dim=-1)  # [B]

        # Normalize by log(B-1) so that uniform distribution → 1
        norm_factor = torch.log(torch.tensor(B - 1., device=logits.device))
        entropy_norm = entropy / norm_factor
        
        # Check for numerical issues before proceeding
        if torch.isnan(entropy).any() or torch.isinf(entropy).any():
            print(f"WARNING: Raw entropy contains nan/inf values")
            print(f"  entropy stats: min={entropy.min():.6f}, max={entropy.max():.6f}, mean={entropy.mean():.6f}")
            print(f"  masked_probs stats: min={masked_probs.min():.6f}, max={masked_probs.max():.6f}")
            print(f"  log_probs stats: min={log_probs.min():.6f}, max={log_probs.max():.6f}")
            # Replace nan/inf with 0 for now
            entropy = torch.where(torch.isnan(entropy) | torch.isinf(entropy), 
                                torch.zeros_like(entropy), entropy)
            entropy_norm = entropy / norm_factor
            
        if torch.isnan(entropy_norm).any() or torch.isinf(entropy_norm).any():
            print(f"WARNING: Normalized entropy contains nan/inf values")
            print(f"  norm_factor: {norm_factor:.6f}")
            print(f"  entropy_norm stats: min={entropy_norm.min():.6f}, max={entropy_norm.max():.6f}")
            # Replace nan/inf with 0 for now
            entropy_norm = torch.where(torch.isnan(entropy_norm) | torch.isinf(entropy_norm), 
                                     torch.zeros_like(entropy_norm), entropy_norm)

        entropies.append(entropy_norm.cpu())
        
        # Collect debug info for first batch to diagnose issues
        if step == 0:
            debug_info = {
                'batch_size': B,
                'tau': tau,
                'cosine_sim_mean': off_diag_cosines.mean().item(),
                'cosine_sim_max': off_diag_cosines.max().item(),
                'cosine_sim_min': off_diag_cosines.min().item(),
                'prob_max_mean': prob_max.mean().item(),
                'concentrated_samples': concentrated_count,
                'raw_entropy_mean': entropy.mean().item(),
                'final_entropy_mean': entropy_norm.mean().item()
            }
            logger.info(f"Entropy debug info: {debug_info}")
            
            # Fail fast if embeddings are too similar (indicating potential model collapse)
            high_sim_ratio = (off_diag_cosines > 0.95).float().mean().item()
            if high_sim_ratio > 0.8:
                raise RuntimeError(
                    f"Model collapse detected: {high_sim_ratio:.2%} of embedding pairs have cosine > 0.95. "
                    f"This leads to entropy≈0. Model may need different training configuration."
                )

    if not entropies:
        raise RuntimeError("No entropy values computed - probe_loader is empty or max_batches=0")

    entropies = torch.cat(entropies, dim=0)
    result = entropies.mean().item()
    
    # Restore training mode before returning
    model.train(original_training_state)
    return result


class ContrastiveDiagnostics:
    """
    Lightweight hook that computes contrastive learning diagnostics:
      - Positive cosine similarity: consistency with training augmentation
      - Negative entropy: uniformity of negative distribution
      - InfoNCE gradient metrics:
        * Positive probability (p_i+): confidence in positive pairs
        * Max negative probability: hardest negative strength
        * Gradient alignment ratio: positive vs negative gradient balance
        * Effective number of negatives: diversity of contributing negatives
        * Hard negative fraction: proportion of challenging negatives
    
    Now uses the same augmentation method as training for accurate diagnostics.
    """
    def __init__(
        self,
        model,
        probe_dataset,
        tokenizer=None,  # if you need tokenization inside; else ignore
        device="cuda",
        batch_size=512,     # same batch_size as training
        tau=0.05,
        log_every=5000,
        max_batches=4,
        logger=None,  # e.g. lambda dict: trainer.log(dict)
        prefix="diag",
        collate_fn=None,  # Custom collate function for data loading
        use_dropout=True  # True for dropout augmentation, False for pre-paired data (section_pair)
    ):
        self.model = model
        self.device = device
        self.tau = tau
        self.log_every = log_every
        self.max_batches = max_batches
        self.logger = logger or (lambda metrics: print(metrics))
        self.prefix = prefix
        self.use_dropout = use_dropout  # Store augmentation method

        # If probe_dataset is already tokenized, just wrap in DataLoader
        # Use simple DataLoader configuration to avoid batch sampling issues
        
        # Use DataLoader with optional custom collate function
        self.probe_loader = DataLoader(
            probe_dataset,
            batch_size=batch_size,
            shuffle=False,       # No shuffling for deterministic diagnostics
            drop_last=False,     # Don't drop incomplete batches
            pin_memory=False,    # Disable pin_memory to avoid issues
            num_workers=0,       # Use main thread to avoid multiprocessing issues
            collate_fn=collate_fn  # Use custom collate function if provided
        )
        
    def maybe_log(self, global_step):
        if global_step % self.log_every != 0:
            return

        # Save current training mode so we can restore it later
        was_training = self.model.training
        
        # Validate probe_loader - fail fast if misconfigured
        if len(self.probe_loader) == 0:
            raise RuntimeError("Probe loader is empty - diagnostic dataset not properly configured")

        # Compute diagnostics on fixed probe set - let any errors propagate
        pos_cos = compute_positive_cosine(
            self.model, self.probe_loader, self.device,
            max_batches=self.max_batches, use_dropout=self.use_dropout
        )
        neg_ent = compute_negative_entropy(
            self.model, self.probe_loader, self.device,
            tau=self.tau, max_batches=self.max_batches, use_dropout=self.use_dropout
        )
        
        # Compute InfoNCE gradient-based metrics
        grad_metrics = compute_gradient_metrics(
            self.model, self.probe_loader, self.device,
            tau=self.tau, max_batches=self.max_batches, use_dropout=self.use_dropout
        )

        # Validate diagnostic values - fail if abnormal
        if not (0.0 <= pos_cos <= 1.0):
            raise ValueError(f"Invalid positive cosine value: {pos_cos}. Expected range [0, 1].")
        
        if not (0.0 <= neg_ent <= 1.0):
            raise ValueError(f"Invalid normalized entropy value: {neg_ent}. Expected range [0, 1].")
        
        # Validate gradient metrics
        if not (0.0 <= grad_metrics['positive_prob'] <= 1.0):
            raise ValueError(f"Invalid positive probability: {grad_metrics['positive_prob']}. Expected range [0, 1].")
        
        if not (0.0 <= grad_metrics['max_neg_prob'] <= 1.0):
            raise ValueError(f"Invalid max negative probability: {grad_metrics['max_neg_prob']}. Expected range [0, 1].")
        
        if not (0.0 <= grad_metrics['hard_negative_fraction'] <= 1.0):
            raise ValueError(f"Invalid hard negative fraction: {grad_metrics['hard_negative_fraction']}. Expected range [0, 1].")

        # Log the metrics
        metrics = {
            f"{self.prefix}_pos_cosine": pos_cos,
            f"{self.prefix}_neg_entropy_norm": neg_ent,
            f"{self.prefix}_positive_prob": grad_metrics['positive_prob'],
            f"{self.prefix}_max_neg_prob": grad_metrics['max_neg_prob'],
            f"{self.prefix}_grad_alignment_ratio": grad_metrics['grad_alignment_ratio'],
            f"{self.prefix}_effective_num_negatives": grad_metrics['effective_num_negatives'],
            f"{self.prefix}_hard_negative_fraction": grad_metrics['hard_negative_fraction'],
            f"{self.prefix}_global_step": global_step,
        }
        
        # Restore training mode before logging
        self.model.train(was_training)
        
        self.logger(metrics)

def clean_state(state):
    """Recursively convert non-serializable objects to serializable ones."""
    if isinstance(state, dict):
        return {k: clean_state(v) for k, v in state.items()}
    elif isinstance(state, list):
        return [clean_state(v) for v in state]
    elif isinstance(state, tuple):
        return tuple(clean_state(v) for v in state)
    # Convert numpy types to Python types
    elif isinstance(state, (np.float16, np.float32, np.float64)):
        return float(state)
    elif isinstance(state, (np.int8, np.int16, np.int32, np.int64)):
        return int(state)
    elif isinstance(state, (np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(state)
    elif isinstance(state, np.bool_):
        return bool(state)
    elif isinstance(state, np.ndarray):
        return state.tolist()
    # Handle torch tensors if present
    elif hasattr(state, 'item') and callable(getattr(state, 'item')):
        try:
            return state.item()
        except (ValueError, RuntimeError) as e:
            # Log the error but don't crash
            logger.warning(f"Could not convert tensor to scalar: {e}")
            return None
    # Handle datetime objects
    elif isinstance(state, datetime):
        return state.isoformat()
    # Handle None, basic Python types
    elif state is None or isinstance(state, (int, float, str, bool)):
        return state
    else:
        # Log unknown types for debugging
        logger.warning(f"Unknown type in state: {type(state)} - {state}")
        return str(state)  # Fallback to string representation

# Ensure all metrics are JSON-serializable
def sanitize_json_metrics(metrics_dict):
    """Legacy function - now just uses clean_state for consistency"""
    return clean_state(metrics_dict)





class ChunkBatchSampler(Sampler[list]):
    def __init__(self, ds, batch_size, shuffle=True):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Pre-store mapping of global idx → (chunk_id, local_idx)
        self.ptrs = [[] for _ in ds.chunk_paths]          # One list per chunk
        for gidx in range(len(ds)):
            cid = bisect_right(ds._prefix, gidx) - 1
            self.ptrs[cid].append(gidx)
    def __iter__(self):
        order = list(range(len(self.ptrs)))
        if self.shuffle: random.shuffle(order)
        for cid in order:
            chunk_indices = self.ptrs[cid]
            if self.shuffle: random.shuffle(chunk_indices)
            for i in range(0, len(chunk_indices), self.batch_size):
                yield chunk_indices[i : i + self.batch_size]
    def __len__(self):
        return math.ceil(len(self.ds) / self.batch_size)


class CLTrainer(Trainer):
    def __init__(self,  model_args=None, data_args=None, tokenizer=None, 
                 probe_dataset=None, enable_diagnostics=False, 
                 diagnostic_log_every=50, diagnostic_max_batches=4,
                 diagnostic_tau=None, *args, **kwargs):
        super(CLTrainer, self).__init__(*args, **kwargs)
        self.model_args = model_args
        self.data_args = data_args
        self.custom_tokenizer = tokenizer
        self.loss_buffer = []
        self.use_apex = is_apex_available() and self.args.fp16_opt_level is not None
        self.metrics = {}
        
        # Ensure diagnostic_tau matches training temperature
        if diagnostic_tau is None:
            # Use model's temperature if available, otherwise default to 0.05
            if model_args and hasattr(model_args, 'temperature'):
                diagnostic_tau = model_args.temperature
                logger.info(f"Using model temperature {diagnostic_tau} for diagnostic tau")
            else:
                diagnostic_tau = 0.05
                logger.warning("Model args temperature not available, using default diagnostic_tau=0.05")
        else:
            # Validate that provided diagnostic_tau matches model temperature
            if model_args and hasattr(model_args, 'temperature'):
                model_temp = model_args.temperature
                if abs(diagnostic_tau - model_temp) > 1e-6:
                    logger.warning(f"Diagnostic tau ({diagnostic_tau}) differs from model temperature ({model_temp}). "
                                  f"For accurate diagnostics, they should match. Using diagnostic_tau={diagnostic_tau} as specified.")
                else:
                    logger.info(f"Diagnostic tau ({diagnostic_tau}) matches model temperature ({model_temp})")
        
        # Initialize contrastive diagnostics if enabled and probe dataset provided
        self.diag_hook = None
        if enable_diagnostics and probe_dataset is not None:
            try:
                device = self.args.device if hasattr(self.args, 'device') else 'cuda'
                if not torch.cuda.is_available():
                    device = 'cpu'
                
                # Determine augmentation method from data_args
                use_dropout = True  # Default to dropout
                aug_methods_str = "dropout"  # Default description
                
                if data_args and hasattr(data_args, 'data_augmentation'):
                    aug_methods = data_args.data_augmentation
                    
                    if len(aug_methods) > 1:
                        # Multiple augmentation methods - diagnostics will mix them
                        use_dropout = 'mixed'  # Special flag for mixed mode
                        aug_methods_str = '+'.join(aug_methods)
                        logger.info(f"Diagnostic augmentation: MIXED ({aug_methods_str}) - randomly choosing like training")
                    elif 'section_pair' in aug_methods:
                        # Section_pair only
                        use_dropout = False
                        aug_methods_str = "section_pair"
                        logger.info(f"Diagnostic augmentation: section_pair only")
                    else:
                        # Dropout or other single method
                        use_dropout = True
                        aug_methods_str = aug_methods[0] if aug_methods else "dropout"
                        logger.info(f"Diagnostic augmentation: {aug_methods_str}")
                    
                self.diag_hook = ContrastiveDiagnostics(
                    model=self.model,
                    probe_dataset=probe_dataset,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=min(512, len(probe_dataset) if hasattr(probe_dataset, '__len__') else 512),
                    tau=diagnostic_tau,
                    log_every=diagnostic_log_every,
                    max_batches=diagnostic_max_batches,
                    logger=self._diagnostic_logger,
                    prefix="contrastive_diag",
                    collate_fn=self.data_collator,  # Use the same data collator as training
                    use_dropout=use_dropout  # Pass augmentation method
                )
                logger.info(f"Initialized contrastive diagnostics with probe dataset of size {len(probe_dataset) if hasattr(probe_dataset, '__len__') else 'unknown'}")
            except Exception as e:
                logger.warning(f"Failed to initialize contrastive diagnostics: {e}")
                self.diag_hook = None
        elif enable_diagnostics:
            logger.info("Contrastive diagnostics enabled but no probe dataset provided")
            
    def _diagnostic_logger(self, metrics):
        """Logger function for diagnostic metrics that integrates with trainer logging."""
        try:
            # Clean metrics for JSON serialization
            clean_metrics = clean_state(metrics)
            
            # Log to trainer
            if hasattr(self, 'log'):
                self.log(clean_metrics)
            
            if hasattr(self.state, 'global_step'):
                logger.info(f"Diagnostics: {clean_metrics} (step={self.state.global_step})")
                
        except Exception as e:
            logger.warning(f"Error logging diagnostic metrics: {e}")


    def get_train_dataloader(self):
        # For IterableDataset, don't use sampler
        # Check both torch.utils.data.IterableDataset and datasets.IterableDataset
        is_iterable_dataset = (
            isinstance(self.train_dataset, torch.utils.data.IterableDataset) or
            hasattr(self.train_dataset, '__iter__') and hasattr(self.train_dataset, '__len__') and
            not hasattr(self.train_dataset, '__getitem__')
        )
        
        if is_iterable_dataset:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                # Pas de sampler pour IterableDataset
                drop_last=True,
                collate_fn=self.data_collator,
                num_workers=0,
                pin_memory=False,
            )
        else:
            # Original logic for classic datasets
            # Check if the dataset has necessary attributes for ChunkBatchSampler
            if hasattr(self.train_dataset, 'chunk_paths') and hasattr(self.train_dataset, '_prefix'):
                # Use ChunkBatchSampler if attributes are available
                batch_sampler = ChunkBatchSampler(
                    self.train_dataset, 
                    self.args.train_batch_size, 
                    shuffle=True
                )
                return DataLoader(
                    self.train_dataset,
                    batch_sampler=batch_sampler,
                    drop_last=True,
                    collate_fn=self.data_collator,
                    num_workers=0,
                    pin_memory=False,
                )
            else:
                # Use standard sampler if ChunkBatchSampler is not applicable
                if dist.is_available() and dist.is_initialized():
                    sampler = DistributedSampler(self.train_dataset, shuffle=True)
                else:
                    sampler = RandomSampler(self.train_dataset)

                return DataLoader(
                    self.train_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=sampler,
                    drop_last=True,
                    collate_fn=self.data_collator,
                    num_workers=0,
                    pin_memory=False,
                )


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not hasattr(model, 'global_step'):
            model.global_step = 0
        else:
            model.global_step = self.state.global_step

        # Add dropout_rate parameter if data_args contains it and dropout is in augmentation
        extra_kwargs = {}
        if (hasattr(self, 'data_args') and self.data_args is not None and 
            hasattr(self.data_args, 'dropout_rate') and hasattr(self.data_args, 'data_augmentation') and
            'dropout' in self.data_args.data_augmentation):
            extra_kwargs['dropout_rate'] = self.data_args.dropout_rate

        outputs = model(**inputs, return_dict=True, **extra_kwargs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if hasattr(model, "swav_head"):
            model.swav_head.update_ema()
            
        if self.is_world_process_zero():
            logs = {'loss': loss.item()}

            # Track InfoNCE contrastive loss separately
            if hasattr(outputs, "contrastive_loss") and outputs.contrastive_loss is not None:
                logs['contrastive_loss'] = outputs.contrastive_loss.item()

            # if MLM training
            if self.model_args.regularization == "mlm" and hasattr(outputs, "loss_MLM") and outputs.loss_MLM is not None:
                logs['mlm_loss'] = outputs.loss_MLM.item()

            # Track losses for different regularization methods
            if hasattr(outputs, "barlow_twins_loss") and outputs.barlow_twins_loss is not None:
                logs['barlow_twins_loss'] = outputs.barlow_twins_loss.item()
                if hasattr(outputs, "on_diagonal") and outputs.on_diagonal is not None:
                    logs['bt_on_diagonal'] = outputs.on_diagonal.item()
                if hasattr(outputs, "off_diagonal") and outputs.off_diagonal is not None:
                    logs['bt_off_diagonal'] = outputs.off_diagonal.item()
            
            if hasattr(outputs, "vicreg_loss") and outputs.vicreg_loss is not None:
                logs['vicreg_loss'] = outputs.vicreg_loss.item()
                if hasattr(outputs, "invariance_loss") and outputs.invariance_loss is not None:
                    logs['vic_invariance_loss'] = outputs.invariance_loss.item()
                if hasattr(outputs, "variance_loss") and outputs.variance_loss is not None:
                    logs['vic_variance_loss'] = outputs.variance_loss.item()
                if hasattr(outputs, "covariance_loss") and outputs.covariance_loss is not None:
                    logs['vic_covariance_loss'] = outputs.covariance_loss.item()
            
            # Track regularization loss if present
            if hasattr(outputs, "loss_reg") and outputs.loss_reg is not None:
                logs['loss_reg'] = outputs.loss_reg.item()

            # Log learning rate if the optimizer has param groups
            if hasattr(self, "optimizer") and len(self.optimizer.param_groups) > 0:
                logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

            clean_logs = clean_state(logs)
            logger.info(f"Logs: {clean_logs} (step={self.state.global_step})")
            
            # Run contrastive diagnostics if enabled
            if self.diag_hook is not None:
                self.diag_hook.maybe_log(self.state.global_step)

        return (loss, outputs) if return_outputs else loss


    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to log gradient norm after backward pass.
        """
        # Call parent's training_step which does forward, backward, and returns loss
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Now gradients are available - log gradient norm
        if self.is_world_process_zero() and hasattr(self, "optimizer") and self.optimizer is not None:
            grad_norms = [torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]
            if grad_norms:
                total_norm = torch.norm(torch.stack(grad_norms), 2).item()
                # Log gradient norm
                logger.info({'grad_norm': total_norm, 'step': self.state.global_step})
        
        return loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        final_eval: bool = False,
    ) -> Dict[str, float]:
        """
        Customized evaluation logic that supports distributed training.
        """
        # Initialize metrics for all processes
        metrics = {}

        if self.args.local_rank != -1 and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.is_world_process_zero():

            def prepare(params, samples):
                return

            def batcher(params, batch, dropout_active=False):
                sentences = [' '.join(s) if isinstance(s, list) else s for s in batch]

                batch_inputs = self.custom_tokenizer.batch_encode_plus(
                    sentences, return_tensors='pt', padding=True, truncation=True,
                    max_length=self.data_args.max_seq_length)

                batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}

                was_training = self.model.training
                was_gc = getattr(self.model.config, 'gradient_checkpointing', False)
                self.model.train(dropout_active)
                if was_gc:
                    self.model.config.gradient_checkpointing = False

                with torch.no_grad():
                    outputs = self.model(**batch_inputs, output_hidden_states=True, return_dict=True, sent_emb=True)
                    pooler_output = outputs.pooler_output.to(torch.float32).cpu().numpy()

                self.model.train(was_training)
                if was_gc:
                    self.model.config.gradient_checkpointing = True

                return pooler_output

            params = {
                'task_path': PATH_TO_DATA,
                'usepytorch': True,
                'max_input_len': self.data_args.max_seq_length,
                'batcher_batch_size': self.args.per_device_eval_batch_size,
                'embedding_dim': 1024,
                'classifier': {
                    'nhid': 0, 'optim': 'rmsprop', 'save_path': self.args.output_dir,
                    'tenacity': 3, 'epoch_size': 2
                },
                'current_step': self.state.global_step,
                'model_output_path': self.args.output_dir,
                'final_eval': final_eval,
            }

            # Set the task path based on the final_eval flag
            if final_eval:
                tasks = ['PriorArt', 'Alignment', 'SingularSpectrum', 'Uniformity', 'IPC-Classification', 'IPC-KNN']
            else:
                tasks = ['PriorArt', 'IPC-Classification', 'IPC-KNN', 'SingularSpectrum', 'Uniformity', 'Alignment']
                params['eval_sample_train'] = 25000
                params['eval_sample_test'] = 2500

            se = patenteval.engine.PE(params, batcher, prepare)

            # check if model is on the correct device (e.g., GPU)
            # if not send it to the correct device
            if self.model.device.type != 'cuda' and torch.cuda.is_available():
                logger.warning(f"Model is on device {self.model.device.type}, but should be on 'cuda'. Moving model to 'cuda'.")
                self.model.to('cuda')

            self.model.eval()
            results = se.eval(tasks)

            for task_name, result in results.items():
                if task_name in ["SingularSpectrum", "Uniformity", "Alignment"]:
                    for section, value in result.items():
                        metric_key = f"eval_{task_name.lower()}_{section}"
                        
                        # Handle different value types using the same logic as combined metrics
                        if isinstance(value, dict):
                            # Extract numeric value from dict using generic function
                            numeric_value = self._extract_numeric_from_dict(value, metric_key)
                            if numeric_value is not None:
                                metrics[metric_key] = numeric_value
                            else:
                                logger.warning(f"Could not extract numeric value from {metric_key}: {value}")
                        elif isinstance(value, (int, float, np.integer, np.floating)):
                            # Direct numeric value (including numpy numeric types)
                            # Convert to Python float for consistency
                            numeric_value = float(value)
                            if not (np.isnan(numeric_value) or np.isinf(numeric_value)):
                                metrics[metric_key] = numeric_value
                            else:
                                logger.warning(f"Invalid numeric value for {metric_key}: {value}")
                        else:
                            logger.warning(f"Unexpected value type for {metric_key}: {type(value)} = {value}")

                elif task_name == "IPC-Classification":
                    for k, v in result.items():
                        prefix = f"eval_ipc_cls_{k}"
                        for metric_name, value in v.items():
                            metrics[f"{prefix}_{metric_name}"] = value

                elif task_name == "IPC-KNN":
                    for k, v in result.items():
                        prefix = f"eval_ipc_knn_{k}"
                        for metric_name, value in v.items():
                            metrics[f"{prefix}_{metric_name}"] = value

            if "PriorArt" in results:
                for pair_key, metrics_dict in results["PriorArt"].items():
                    if not isinstance(metrics_dict, dict):
                        continue
                    q_section, d_section = pair_key.split("->")
                    abbrev_map = {"abstract": "a", "claim": "c", "invention": "i", "summary": "s", "description": "d", "all": "all"}
                    q_abbr = abbrev_map.get(q_section, q_section[0])
                    d_abbr = abbrev_map.get(d_section, d_section[0])
                    combo_key = f"{q_abbr}2{d_abbr}"
                    for metric_name, value in metrics_dict.items():
                        if metric_name == "retrieved_sections":
                            continue
                        elif metric_name == "section_analysis":
                            # Handle section analysis results
                            if isinstance(value, dict):
                                for top_k, k_results in value.items():
                                    if isinstance(k_results, dict):
                                        for section, section_stats in k_results.items():
                                            if isinstance(section_stats, dict) and 'percentage' in section_stats:
                                                full_key = f"eval_sections_{combo_key}_{top_k}_{section}_pct"
                                                metrics[full_key] = section_stats['percentage']
                        elif metric_name.startswith("recall@") or metric_name.startswith("ndcg@"):
                            full_key = f"eval_priorArt_{metric_name}_{combo_key}"
                            metrics[full_key] = value

            # Grouped WandB logging
            grouped_metrics = {}
            for k, v in metrics.items():
                if k.startswith("eval_ipc_cls_"):
                    grouped_metrics[f"ipc_cls/{k}"] = v
                elif k.startswith("eval_ipc_knn_"):
                    grouped_metrics[f"ipc_knn/{k}"] = v
                elif k.startswith("eval_priorArt_"):
                    grouped_metrics[f"priorart/{k}"] = v
                elif k.startswith("eval_singularspectrum"):
                    grouped_metrics[f"stats/singularspectrum/{k}"] = v
                elif k.startswith("eval_uniformity"):
                    grouped_metrics[f"stats/uniformity/{k}"] = v
                elif k.startswith("eval_alignment"):
                    grouped_metrics[f"stats/alignment/{k}"] = v
                elif k.startswith("eval_sections_"):
                    grouped_metrics[f"section_analysis/{k}"] = v
                else:
                    grouped_metrics[k] = v

            # Clean metrics before logging and saving
            metrics = clean_state(metrics)
            grouped_metrics = clean_state(grouped_metrics)

            # Compute and log combined metric if configured
            combined_metric_value = None
            if (hasattr(self.args, 'metric_for_best_model_list') and 
                self.args.metric_for_best_model_list is not None and 
                len(self.args.metric_for_best_model_list) > 0):
                combined_metric_value = self._compute_combined_metric(metrics)
                if combined_metric_value is not None:
                    # Add to regular metrics for saving
                    metrics['eval_combined_metric'] = combined_metric_value
                    # Add to wandb grouped metrics for visualization
                    grouped_metrics['combined_metric/eval_combined_metric'] = combined_metric_value

            self.log(metrics)
            logger.info(f"Grouped metrics: {grouped_metrics} (step={self.state.global_step})")
            self.metrics = metrics

            # Save best model if this is the best metric so far
            self._save_best_model_if_needed(metrics)

        if self.args.local_rank != -1 and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.is_world_process_zero():
            # Ensure output directory exists
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            metrics_file_path = os.path.join(self.args.output_dir, f"eval_metrics_step{self.state.global_step}.json")
            try:
                with open(metrics_file_path, "w") as f:
                    json.dump(sanitize_json_metrics(metrics), f, indent=2)
                logger.info(f"Successfully saved metrics to {metrics_file_path}")
            except Exception as e:
                logger.error(f"Failed to save metrics file {metrics_file_path}: {e}")
        else:
            # Non-master processes: return empty metrics to avoid file reading issues
            # The main process handles all evaluation and metric computation
            metrics = {}

        return metrics


    def _compute_combined_metric(self, metrics):
        """
        Compute a combined metric value from multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            Combined metric value or None if computation fails
        """
        if not hasattr(self.args, 'metric_for_best_model_list') or not self.args.metric_for_best_model_list:
            logger.debug("metric_for_best_model_list not configured, skipping combined metric computation")
            return None
            
        metric_list = self.args.metric_for_best_model_list
        strategy = getattr(self.args, 'metric_combination_strategy', 'multiply')
        weights = getattr(self.args, 'metric_weights', None)
        
        logger.info(f"Computing combined metric from: {metric_list} using strategy: {strategy}")
        
        # Collect metric values
        metric_values = []
        missing_metrics = []
        invalid_metrics = []
        
        for metric_name in metric_list:
            # Add eval_ prefix if not present
            metric_to_check = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
            
            if metric_to_check in metrics:
                value = metrics[metric_to_check]
                
                # Handle different value types
                numeric_value = None
                
                if isinstance(value, (int, float, np.integer, np.floating)):
                    # Direct numeric value (including numpy numeric types)
                    numeric_value = float(value)
                    if not (np.isnan(numeric_value) or np.isinf(numeric_value)):
                        pass  # numeric_value is already set
                    else:
                        numeric_value = None
                elif isinstance(value, dict):
                    # Dictionary value - try to extract numeric value
                    numeric_value = self._extract_numeric_from_dict(value, metric_to_check)
                else:
                    logger.warning(f"Unexpected value type for {metric_to_check}: {type(value)} = {value}")
                
                if numeric_value is not None:
                    metric_values.append(numeric_value)
                    logger.debug(f"  {metric_to_check}: {numeric_value} (original: {value})")
                else:
                    logger.warning(f"Could not extract numeric value from {metric_to_check}: {value} (type: {type(value)})")
                    invalid_metrics.append((metric_to_check, value))
            else:
                missing_metrics.append(metric_to_check)
        
        if missing_metrics:
            logger.warning(f"Missing metrics for combined metric computation: {missing_metrics}")
            logger.warning(f"Available metrics: {list(metrics.keys())}")
            raise ValueError("Missing metrics for combined metric computation")
            
        if invalid_metrics:
            logger.warning(f"Invalid metrics found: {invalid_metrics}")
            raise ValueError("Invalid metrics for combined metric computation")
            
        if len(metric_values) != len(metric_list):
            logger.warning(f"Could not collect all required metrics. Expected {len(metric_list)}, got {len(metric_values)}")
            raise ValueError("Incomplete metrics for combined metric computation")

        # Compute combined metric based on strategy
        if strategy == "multiply":
            combined_value = 1.0
            for value in metric_values:
                combined_value *= value
                
        elif strategy == "weighted_sum":
            if weights is None:
                # Equal weights
                weights = [1.0 / len(metric_values)] * len(metric_values)
            elif len(weights) != len(metric_values):
                logger.error(f"Number of weights ({len(weights)}) does not match number of metrics ({len(metric_values)})")
                raise ValueError("Weights length mismatch")
            
            combined_value = sum(w * v for w, v in zip(weights, metric_values))
            
        elif strategy == "geometric_mean":
            # Geometric mean: (product of values) ^ (1/n)
            if any(v <= 0 for v in metric_values):
                logger.warning("Geometric mean requires all positive values. Found non-positive values.")
                raise ValueError("Invalid metrics for combined metric computation")
            product = 1.0
            for value in metric_values:
                product *= value
            combined_value = product ** (1.0 / len(metric_values))
            
        else:
            logger.error(f"Unknown metric combination strategy: {strategy}")
            raise ValueError("Unknown metric combination strategy")
            
        logger.info(f"Combined metric ({strategy}): {combined_value:.6f} from metrics {dict(zip(metric_list, metric_values))}")
        return combined_value


    def _save_best_model_if_needed(self, metrics):
        """
        Save the best model if current metrics are better than previous best.
        """
        if not self.is_world_process_zero() or not self.args.should_save:
            return

        if metrics is not None:
            # Check if we should use combined metrics or single metric
            use_combined_metrics = (
                hasattr(self.args, 'metric_for_best_model_list') and 
                self.args.metric_for_best_model_list is not None and 
                len(self.args.metric_for_best_model_list) > 0
            )
            
            if use_combined_metrics:
                # Use combined metrics
                metric_value = self._compute_combined_metric(metrics)
                metric_to_check = "combined_metric"
                logger.info(f"Using combined metric for best model selection: {metric_value}")
            else:
                # Fall back to single metric (original behavior)
                if self.args.metric_for_best_model is None:
                    return
                    
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                
                metric_value = metrics.get(metric_to_check, None)
            
            if metric_value is not None:
                # Use the same greater_is_better setting for both single and combined metrics
                greater_is_better = self.args.greater_is_better
                    
                operator = np.greater if greater_is_better else np.less
                is_best = (
                    self.state.best_metric is None
                    or operator(metric_value, self.state.best_metric)
                )

                if is_best:
                    # Convert metric_value to a JSON-serializable type
                    self.state.best_metric = clean_state(metric_value)
                    best_model_output_dir = os.path.join(self.args.output_dir, "best_model")
                    self.state.best_model_checkpoint = best_model_output_dir

                    if use_combined_metrics:
                        logger.info(
                            f"Saving new best model to {best_model_output_dir} with combined metric = {metric_value:.6f}"
                        )
                    else:
                        logger.info(
                            f"Saving new best model to {best_model_output_dir} with {metric_to_check} = {metric_value}"
                        )

                    # Create best model directory if it doesn't exist
                    os.makedirs(best_model_output_dir, exist_ok=True)
                    
                    # Save the model
                    self.save_model(best_model_output_dir, _internal_call=True)

                    # For best model, only save essential state to reduce save time
                    # Skip optimizer and scheduler for best model to avoid DeepSpeed hanging
                    # Only save trainer state
                    state_dict = {
                        'global_step': self.state.global_step,
                        'epoch': self.state.epoch,
                        'best_metric': self.state.best_metric,
                        'best_model_checkpoint': self.state.best_model_checkpoint,
                        'train_batch_size': self.args.train_batch_size,
                        'num_train_epochs': self.args.num_train_epochs,
                    }
                    cleaned_state = clean_state(state_dict)
                    with open(os.path.join(best_model_output_dir, TRAINER_STATE_NAME), "w") as f:
                        f.write(json.dumps(cleaned_state, indent=2, sort_keys=True) + "\n")
                    
                    logger.info(f"Best model saved successfully to {best_model_output_dir}")
                    
                    # Optional: Save full training state only if explicitly requested
                    if not self.args.save_only_model and getattr(self.args, 'save_best_model_full_state', False):
                        logger.info("Saving full training state for best model...")
                        self._save_optimizer_and_scheduler(best_model_output_dir)
                        self._save_rng_state(best_model_output_dir)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to handle DeepSpeed properly with timeout and error handling.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Force memory cleanup before saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Handle DeepSpeed model saving with timeout
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # DeepSpeed model saving
            try:
                import signal
                from contextlib import contextmanager
                
                logger.info(f"Starting DeepSpeed model save to {output_dir}")
                start_time = time.time()
                
                @contextmanager
                def timeout_context(timeout_seconds=600):  # 10 minutes timeout for DeepSpeed
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Model save operation timed out after {timeout_seconds} seconds")
                    
                    # Set up signal handler only on main process and if signal is available
                    if self.is_world_process_zero() and hasattr(signal, 'SIGALRM'):
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout_seconds)
                        try:
                            yield
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
                    else:
                        yield
                
                # Save with timeout protection
                with timeout_context():
                    logger.info(f"Initiating DeepSpeed checkpoint save to {output_dir}")
                    
                    # Use DeepSpeed's save_checkpoint method which handles ZeRO properly
                    checkpoint_dir = os.path.join(output_dir, f"global_step{self.state.global_step}")
                    
                    # For best_model saving, use a simpler name and skip some optimizations
                    if "best_model" in output_dir:
                        checkpoint_dir = output_dir
                        logger.info("Saving best model - using simplified checkpoint structure")
                        
                        # For best models, try a lightweight save first
                        try:
                            logger.info("Attempting lightweight model save for best model...")
                            
                            # Only save model weights, skip optimizer state
                            client_state = {"save_type": "best_model"}
                            
                            # Use save_checkpoint with minimal state
                            self.deepspeed.save_checkpoint(
                                checkpoint_dir, 
                                client_state=client_state,
                                save_latest=False,
                                exclude_frozen_parameters=True
                            )
                            
                            save_time = time.time() - start_time
                            logger.info(f"Best model DeepSpeed checkpoint saved in {save_time:.1f}s to {checkpoint_dir}")
                            
                        except Exception as save_error:
                            logger.warning(f"Lightweight save failed: {save_error}")
                            logger.info("Falling back to standard checkpoint save...")
                            
                            # Fallback to standard save
                            self.deepspeed.save_checkpoint(checkpoint_dir, exclude_frozen_parameters=True)
                            save_time = time.time() - start_time  
                            logger.info(f"Fallback DeepSpeed checkpoint saved in {save_time:.1f}s to {checkpoint_dir}")
                    else:
                        # Standard checkpoint save
                        logger.info("Saving standard training checkpoint...")
                        self.deepspeed.save_checkpoint(checkpoint_dir, exclude_frozen_parameters=True)
                        save_time = time.time() - start_time
                        logger.info(f"Standard DeepSpeed checkpoint saved in {save_time:.1f}s to {checkpoint_dir}")
                    
                    # Also save the model config and tokenizer (only on main process)
                    if self.is_world_process_zero():
                        logger.info("Saving model config and tokenizer...")
                        config_start = time.time()
                        
                        self.model.config.save_pretrained(output_dir)
                        if self.custom_tokenizer is not None:
                            self.custom_tokenizer.save_pretrained(output_dir)
                        elif self.tokenizer is not None:
                            self.tokenizer.save_pretrained(output_dir)
                        
                        # Also save the model in standard format for compatibility
                        logger.info("Converting DeepSpeed checkpoint to standard format...")
                        try:
                            # Extract state dict from DeepSpeed model
                            state_dict = self.deepspeed.module_state_dict()
                            
                            # Save as standard pytorch_model.bin
                            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                            logger.info("Standard pytorch_model.bin saved successfully!")
                        except Exception as e:
                            logger.warning(f"Failed to save standard format: {e}")
                            logger.info("Model can be converted later using zero_to_fp32.py")
                        
                        config_time = time.time() - config_start
                        logger.info(f"Model config and tokenizer saved in {config_time:.1f}s to {output_dir}")
                        
                    total_time = time.time() - start_time
                    logger.info(f"Total DeepSpeed save operation completed in {total_time:.1f}s")
                        
            except TimeoutError as e:
                save_time = time.time() - start_time
                logger.error(f"DeepSpeed model save timed out after {save_time:.1f}s: {e}")
                
                # Try a simpler save method as fallback
                logger.info("Attempting emergency fallback model save...")
                try:
                    fallback_start = time.time()
                    
                    # Force collect state dict from DeepSpeed
                    logger.info("Collecting model state dict from DeepSpeed...")
                    state_dict = self.deepspeed.module_state_dict()
                    
                    if self.is_world_process_zero():
                        logger.info("Saving emergency model backup...")
                        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                        self.model.config.save_pretrained(output_dir)
                        if self.custom_tokenizer is not None:
                            self.custom_tokenizer.save_pretrained(output_dir)
                        elif self.tokenizer is not None:
                            self.tokenizer.save_pretrained(output_dir)
                        
                        fallback_time = time.time() - fallback_start
                        logger.info(f"Emergency fallback save completed in {fallback_time:.1f}s to {output_dir}")
                        
                except Exception as fallback_error:
                    logger.error(f"Emergency fallback save also failed: {fallback_error}")
                    raise e
                    
            except Exception as e:
                save_time = time.time() - start_time
                logger.error(f"DeepSpeed model save failed after {save_time:.1f}s: {e}")
                raise
        else:
            # Standard model saving (non-DeepSpeed)
            logger.info(f"Saving standard model to {output_dir}")
            super().save_model(output_dir, _internal_call)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override the default _save_checkpoint to handle serialization errors robustly.
        """
        if not self.is_world_process_zero():
            # Only save checkpoints on the main process
            return super()._save_checkpoint(model, trial)
        
        # Save the original state
        original_state_dict = dataclasses.asdict(self.state)
        
        # Clean the state before calling parent method
        cleaned_state_dict = clean_state(original_state_dict)
        
        # Validate that we have all required fields for TrainerState
        required_fields = ['epoch', 'global_step', 'max_steps', 'num_train_epochs']
        for field in required_fields:
            if field not in cleaned_state_dict:
                logger.warning(f"Missing required field {field} in state, setting default")
                cleaned_state_dict[field] = getattr(self.state, field, 0)
        
        # Temporarily replace the state with cleaned version
        try:
            temp_state = TrainerState(**cleaned_state_dict)
        except Exception as e:
            logger.error(f"Failed to create TrainerState from cleaned data: {e}")
            logger.error(f"Cleaned state keys: {list(cleaned_state_dict.keys())}")
            # Fallback: try with minimal state
            minimal_state = {
                'epoch': cleaned_state_dict.get('epoch', 0),
                'global_step': cleaned_state_dict.get('global_step', 0),
                'max_steps': cleaned_state_dict.get('max_steps', 0),
                'num_train_epochs': cleaned_state_dict.get('num_train_epochs', 0),
                'log_history': cleaned_state_dict.get('log_history', []),
                'best_metric': cleaned_state_dict.get('best_metric', None),
                'best_model_checkpoint': cleaned_state_dict.get('best_model_checkpoint', None),
                'total_flos': cleaned_state_dict.get('total_flos', 0),
                'trial_name': cleaned_state_dict.get('trial_name', None),
                'trial_params': cleaned_state_dict.get('trial_params', None),
            }
            temp_state = TrainerState(**minimal_state)
        
        original_state = self.state
        self.state = temp_state
        
        max_retries = 3
        retry_count = 0
        checkpoint_folder = None
        
        while retry_count < max_retries:
            try:
                # Call the parent method with cleaned state
                checkpoint_folder = super()._save_checkpoint(model, trial)
                logger.info(f"Successfully saved checkpoint to {checkpoint_folder}")
                break
            except RuntimeError as e:
                retry_count += 1
                error_msg = str(e)
                
                if "PytorchStreamWriter failed writing file data" in error_msg or "unexpected pos" in error_msg:
                    logger.warning(f"Checkpoint save failed with serialization error (attempt {retry_count}/{max_retries}): {e}")
                    
                    if retry_count < max_retries:
                        # Wait a bit and try again
                        time.sleep(2 ** retry_count)  # Exponential backoff
                        
                        # Clean up any partially written checkpoint
                        if checkpoint_folder and os.path.exists(checkpoint_folder):
                            try:
                                import shutil
                                shutil.rmtree(checkpoint_folder)
                                logger.info(f"Cleaned up partially written checkpoint: {checkpoint_folder}")
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to clean up partial checkpoint: {cleanup_error}")
                        
                        # Force garbage collection to free memory
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        continue
                    else:
                        logger.error(f"Failed to save checkpoint after {max_retries} attempts: {e}")
                        logger.error("Continuing training without saving this checkpoint to avoid crashing...")
                        # Return a fake checkpoint folder to prevent trainer from crashing
                        checkpoint_folder = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
                        break
                else:
                    logger.error(f"Checkpoint save failed with unexpected error: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error during checkpoint save: {e}")
                if retry_count == max_retries - 1:
                    raise
                retry_count += 1
            finally:
                # Always restore the original state
                self.state = original_state
        
        return checkpoint_folder

    def _load_best_model_at_end(self):
        """
        Load the best model at the end of training if load_best_model_at_end is True.
        This should be called at the end of training, before final evaluation.
        """
        if (
            self.args.load_best_model_at_end 
            and self.state.best_model_checkpoint is not None 
            and self.is_world_process_zero()
        ):
            logger.info(f"Loading best model from {self.state.best_model_checkpoint} for final evaluation")
            
            # Load the best model
            if hasattr(self.model, "load_state_dict"):
                # For regular PyTorch models
                state_dict_path = os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin")
                if os.path.exists(state_dict_path):
                    state_dict = torch.load(state_dict_path, map_location="cpu")
                    # Remove "module." prefix if present (from DDP)
                    if any(key.startswith("module.") for key in state_dict.keys()):
                        state_dict = {key[7:]: value for key, value in state_dict.items()}
                    self.model.load_state_dict(state_dict)
                    logger.info("Successfully loaded best model state dict")
                else:
                    logger.warning(f"Could not find pytorch_model.bin in {self.state.best_model_checkpoint}")
            
            # For models with from_pretrained method
            elif hasattr(self.model.__class__, "from_pretrained"):
                try:
                    # This approach preserves the model architecture and config
                    best_model = self.model.__class__.from_pretrained(
                        self.state.best_model_checkpoint,
                        model_args=getattr(self, 'model_args', None)
                    )
                    # Copy the loaded model's state to current model
                    self.model.load_state_dict(best_model.state_dict())
                    logger.info("Successfully loaded best model using from_pretrained")
                except Exception as e:
                    logger.warning(f"Failed to load best model using from_pretrained: {e}")
            
            # Ensure model is on correct device
            if torch.cuda.is_available():
                self.model = self.model.to(self.model.device)
                
        elif self.args.load_best_model_at_end and self.state.best_model_checkpoint is None:
            logger.warning("load_best_model_at_end is True but no best model checkpoint found")

    def _save_rng_state(self, output_dir):
        """
        Override to save RNG state in a format compatible with weights_only=True loading.
        """
        # Get the RNG states
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            if self.args.parallel_mode == "distributed":
                # In distributed mode, save RNG state for all devices
                rng_states["cuda"] = torch.cuda.get_rng_state_all()
            else:
                # Save only current device state
                rng_states["cuda"] = torch.cuda.get_rng_state()

        # Convert numpy arrays to tensors to make it compatible with weights_only=True
        try:
            # Convert numpy random state to a format that's weights_only compatible
            numpy_state = rng_states["numpy"]
            if isinstance(numpy_state, tuple) and len(numpy_state) >= 2:
                # numpy_state is typically ('MT19937', array, pos, has_gauss, gauss)
                state_name = numpy_state[0]
                state_array = numpy_state[1] if len(numpy_state) > 1 else None
                state_pos = numpy_state[2] if len(numpy_state) > 2 else 0
                state_gauss_info = numpy_state[3:] if len(numpy_state) > 3 else ()
                
                # Convert to tensors where possible, handling dtype issues
                if state_array is not None:
                    try:
                        # Convert numpy array to tensor, ensuring compatible dtype
                        if state_array.dtype == np.uint32:
                            # Convert uint32 to int64 to avoid serialization issues
                            state_tensor = torch.from_numpy(state_array.astype(np.int64))
                        elif state_array.dtype in [np.uint8, np.uint16, np.uint64]:
                            # Convert other unsigned types to signed equivalents
                            state_tensor = torch.from_numpy(state_array.astype(np.int64))
                        else:
                            # For other dtypes, try direct conversion
                            state_tensor = torch.from_numpy(state_array)
                    except Exception as conv_error:
                        logger.warning(f"Failed to convert numpy array to tensor: {conv_error}")
                        # Fall back to storing as list
                        state_tensor = state_array.tolist()
                else:
                    state_tensor = None
                
                converted_numpy_state = {
                    "state_name": state_name,
                    "state_array": state_tensor,
                    "state_pos": state_pos,
                    "state_gauss_info": state_gauss_info,
                }
                rng_states["numpy"] = converted_numpy_state

            # Convert python random state tuple to dict 
            python_state = rng_states["python"]
            if isinstance(python_state, tuple) and len(python_state) >= 3:
                # python random state is (version, tuple_of_ints, gauss_next)
                version = python_state[0]
                state_tuple = python_state[1] if len(python_state) > 1 else ()
                gauss_next = python_state[2] if len(python_state) > 2 else None
                
                # Convert to a serializable format (avoid tensor conversion for large tuples)
                converted_python_state = {
                    "version": version,
                    "state_data": list(state_tuple) if state_tuple else [],
                    "gauss_next": gauss_next,
                }
                rng_states["python"] = converted_python_state

        except Exception as e:
            logger.warning(f"Could not convert RNG states to weights_only compatible format: {e}")
            logger.info("Saving RNG states in original format")

        # Save the RNG state
        rng_file = os.path.join(output_dir, "rng_state.pth")
        torch.save(rng_states, rng_file)
        logger.info(f"Saved RNG state to {rng_file}")

    def _load_rng_state(self, checkpoint):
        """
        Override to handle loading RNG state files that contain numpy objects.
        """
        if checkpoint is None:
            return

        # Find the RNG file in the checkpoint directory
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            logger.info(
                f"Didn't find an RNG file in {checkpoint}, if you are resuming a training where no random states were saved, you can ignore this warning."
            )
            return

        try:
            # First try with weights_only=True for security
            checkpoint_rng_state = torch.load(rng_file, weights_only=True)
        except Exception as e:
            logger.warning(f"Failed to load RNG state with weights_only=True: {e}")
            try:
                # Fallback to weights_only=False for compatibility with older checkpoints
                logger.info("Attempting to load RNG state with weights_only=False for compatibility")
                checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            except Exception as e2:
                logger.error(f"Failed to load RNG state even with weights_only=False: {e2}")
                logger.warning("Skipping RNG state loading. Training will continue with random initialization.")
                return

        # Apply the loaded RNG state
        try:
            # Python random state
            if "python" in checkpoint_rng_state:
                python_state = checkpoint_rng_state["python"]
                if isinstance(python_state, dict):
                    # New format: reconstruct tuple from dict
                    version = python_state.get("version", 3)
                    state_data = tuple(python_state.get("state_data", []))
                    gauss_next = python_state.get("gauss_next", None)
                    reconstructed_state = (version, state_data, gauss_next)
                    random.setstate(reconstructed_state)
                else:
                    # Old format: direct tuple
                    random.setstate(python_state)

            # Numpy state
            if "numpy" in checkpoint_rng_state:
                numpy_state = checkpoint_rng_state["numpy"]
                if isinstance(numpy_state, dict):
                    # New format: reconstruct from dict
                    state_name = numpy_state.get("state_name", "MT19937")
                    state_array = numpy_state.get("state_array")
                    state_pos = numpy_state.get("state_pos", 0)
                    state_gauss_info = numpy_state.get("state_gauss_info", ())
                    
                    if state_array is not None:
                        # Convert back to numpy array
                        if hasattr(state_array, 'numpy'):
                            # It's a tensor - convert to numpy
                            state_array_np = state_array.numpy()
                            # Convert back to uint32 if it was originally uint32
                            if state_array_np.dtype == np.int64:
                                # Assume it was originally uint32, convert back
                                state_array_np = state_array_np.astype(np.uint32)
                        elif isinstance(state_array, list):
                            # It's a list - convert to numpy array
                            state_array_np = np.array(state_array, dtype=np.uint32)
                        else:
                            # Direct numpy array
                            state_array_np = state_array
                        
                        reconstructed_state = (state_name, state_array_np, state_pos) + tuple(state_gauss_info)
                        np.random.set_state(reconstructed_state)
                else:
                    # Old format: direct tuple/list
                    np.random.set_state(numpy_state)

            # PyTorch CPU state
            if "cpu" in checkpoint_rng_state:
                torch.set_rng_state(checkpoint_rng_state["cpu"])

            # PyTorch CUDA state
            if "cuda" in checkpoint_rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint_rng_state["cuda"])
                
            logger.info("Successfully loaded RNG state from checkpoint")
            
        except Exception as e:
            logger.warning(f"Failed to apply RNG state: {e}")
            logger.warning("Continuing with current RNG state")

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        Main training method that handles loading the best model at the end.
        """
        # Call the parent train method
        train_result = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )
        
        # Load best model at the end if specified
        self._load_best_model_at_end()
        
        return train_result

    def _extract_numeric_from_dict(self, value_dict, metric_name):
        """
        Extract numeric value from dictionary-formatted metrics.
        
        Args:
            value_dict: Dictionary containing metric value
            metric_name: Name of the metric for logging purposes
            
        Returns:
            Numeric value if extraction successful, None otherwise
        """
        if not isinstance(value_dict, dict):
            return None
            
        # Common patterns for extracting values from dicts
        extraction_patterns = [
            # Alignment metrics
            'mean_alignment',
            # Section analysis metrics  
            'percentage',
            # Other potential patterns
            'value',
            'score',
            'metric',
            'result',
            'mean',
            'average'
        ]
        
        # Try each extraction pattern
        for pattern in extraction_patterns:
            if pattern in value_dict:
                extracted_value = value_dict[pattern]
                if isinstance(extracted_value, (int, float)) and not (isinstance(extracted_value, float) and (np.isnan(extracted_value) or np.isinf(extracted_value))):
                    logger.debug(f"Successfully extracted '{pattern}' from {metric_name}: {extracted_value}")
                    return float(extracted_value)
        
        # If no common pattern found, try to find any numeric value in the dict
        numeric_values = []
        for key, val in value_dict.items():
            if isinstance(val, (int, float)) and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                numeric_values.append((key, float(val)))
        
        if len(numeric_values) == 1:
            # If there's exactly one numeric value, use it
            key, val = numeric_values[0]
            logger.info(f"Found single numeric value '{key}' in {metric_name}: {val}")
            return val
        elif len(numeric_values) > 1:
            logger.warning(f"Multiple numeric values found in {metric_name}: {numeric_values}")
            logger.warning(f"Please specify which key to extract, or add it to extraction_patterns")
            # Return the first one as fallback, but warn the user
            key, val = numeric_values[0]
            logger.warning(f"Using first numeric value '{key}': {val}")
            return val
        else:
            logger.warning(f"No numeric values found in dictionary {metric_name}: {value_dict}")
            return None

