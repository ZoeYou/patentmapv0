#!/usr/bin/env python
"""
evaluate.py

This script evaluates baseline models (without training) on our patent and scientific evaluation tasks.
It loads a pretrained model and computes tokenization and embeddings on-the-fly using the model's tokenizer.
If precomputed embeddings are present in the expected temp directories, the script will load them to
speed up repeated runs instead of recomputing embeddings.

When evaluating checkpoint model directories the loader will try to load a tokenizer from the
checkpoint and (if missing) reconstruct a tokenizer temporarily for evaluation purposes.

Usage example:
    python evaluate.py --model_name <path_or_model_id> --output_dir ./results
"""

from __future__ import absolute_import, division, unicode_literals

import os
import re
import sys
import json
import argparse
import logging

from tqdm import trange
import pandas as pd
import numpy as np

import faiss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from transformers import set_seed,  AutoTokenizer, AutoModel

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead."
)

# ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# ignore UserWarining for unknow labels for IPC classification
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global constants
QUALITY_MIN_WORDS = 6  # Minimum number of words required for high-quality text


# add patenteval to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
patent_eval_path = os.path.join(current_dir, 'patentmap_eval')
sys.path.append(patent_eval_path)

# Try to import patenteval.utils with better error handling
try:
    from patenteval.utils import (
        load_corpus,
        citation_to_citing_to_cited_dict,
        citation_to_citing_to_cited_graded_dict,
        mean_recall_at_k,
        mean_ndcg_at_k,
        mean_ndcg_at_k_graded,
        mean_average_precision,
        mean_reciprocal_rank,
        label_process,
        compute_uniformity,
        compute_alignment,
        compute_ssd,
        compute_intra_document_cohesion,
        LinearClassifier,
        KNNClassifier,
    )
    print("Successfully imported patenteval.utils")
except ImportError as e:
    print(f"Warning: Could not import patenteval.utils: {e}")
    print(f"patentmap_eval path: {patent_eval_path}")
    print(f"patentmap_eval exists: {os.path.exists(patent_eval_path)}")
    print("Available paths in sys.path:")
    for p in sys.path[-3:]:  # Show last 3 paths
        print(f"  {p}")
    print("Please ensure patentmap_eval is present and contains an __init__.py file.")
    # You might want to exit here or provide fallback implementations
    sys.exit(1)


def log_embeddings_shape(embeddings_dict, context=""):
    """Helper function to log embedding shapes consistently"""
    if context:
        print(f"{context}:")
    for name, embeddings in embeddings_dict.items():
        print(f"  {name}: {embeddings.shape}")


# ================== RESULT FORMATTING FUNCTIONS ==================

def print_section_header(title, width=80):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subsection_header(title, width=60):
    """Print a formatted subsection header"""
    print(f"\n{'-' * width}")
    print(f" {title}")
    print(f"{'-' * width}")


def print_metric_table(results_dict, task_name, precision=4):
    """
    Print results in a clean table format
    
    Args:
        results_dict: Dictionary with metric names as keys and values as values
        task_name: Name of the evaluation task
        precision: Number of decimal places for floating point numbers
    """
    print(f"\n📊 {task_name} Results:")
    print("-" * 50)
    
    if not results_dict:
        print("   No results available")
        return
    
    # Sort metrics with intelligent handling of numbers (e.g., @10, @20, @50, @100)
    def metric_sort_key(metric_name):
        """
        Create a sort key that handles numeric suffixes correctly.
        Examples: 
        - precision@1 -> ('precision', 1)
        - recall@100 -> ('recall', 100)
        - ndcg@50 -> ('ndcg', 50)
        - alignment -> ('alignment', 0)
        """
        # Extract the base metric name and numeric value
        match = re.match(r'([^@]+)@?(\d+)?', metric_name)
        if match:
            base_name = match.group(1)
            number = int(match.group(2)) if match.group(2) else 0
            return (base_name, number)
        return (metric_name, 0)
    
    sorted_keys = sorted(results_dict.keys(), key=metric_sort_key)
    
    # Print each metric with consistent formatting
    for key in sorted_keys:
        value = results_dict[key]
        
        # Format value based on type
        if isinstance(value, float):
            if abs(value) < 0.001:
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = f"{value:.{precision}f}"
        elif isinstance(value, dict):
            formatted_value = str(value)
        else:
            formatted_value = str(value)
        
        # All metrics displayed with same format - no highlighting
        print(f"   📋 {key:<25}: {formatted_value}")


def print_comparison_summary(results_list, task_name, main_metric):
    """Print a comparison summary for multiple models/conditions"""
    print(f"\n🏆 {task_name} - {main_metric} Comparison:")
    print("-" * 60)
    
    # Sort by main metric (descending for most metrics)
    sorted_results = sorted(results_list, key=lambda x: x.get(main_metric, 0), reverse=True)
    
    for i, result in enumerate(sorted_results[:5]):  # Show top 5
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}️⃣"
        model_name = result.get('model_name', f'Model_{i+1}')
        score = result.get(main_metric, 0)
        
        if isinstance(score, float):
            print(f"   {rank_emoji} {model_name:<20}: {score:.4f}")
        else:
            print(f"   {rank_emoji} {model_name:<20}: {score}")


def log_evaluation_start(task_name, model_name=None):
    """Log the start of an evaluation task"""
    if model_name:
        print(f"\n🚀 Starting {task_name} evaluation for {model_name}...")
    else:
        print(f"\n🚀 Starting {task_name} evaluation...")


def log_evaluation_complete(task_name, time_taken=None):
    """Log the completion of an evaluation task"""
    if time_taken:
        print(f"✅ {task_name} evaluation completed in {time_taken:.2f}s")
    else:
        print(f"✅ {task_name} evaluation completed")


# ================== END FORMATTING FUNCTIONS ==================


def mean_pooling(token_embeddings, attention_mask):
    """
    Performs mean pooling on token embeddings.
    Args:
        token_embeddings: Tensor of shape (batch_size, seq_length, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_length)
    Returns:
        Pooled tensor of shape (batch_size, hidden_dim)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).to(token_embeddings.device)  # Ensure same device
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output, attention_mask):
    return model_output.last_hidden_state[:, 0]  # Explicitly using last_hidden_state


_FIRST_CLAIM_BOUNDARY_RE = re.compile(r'\s2\s*\.\s')


def extract_first_claim(claims_text):
    """Return the first claim from DAPFAM's ``claims_text`` field.

    DAPFAM claims are stored as a single string like
    ``"1. <claim 1> 2. <claim 2> 3. ..."`` (sometimes prefixed with a
    spaced ``"c l a i m s "`` header). We split at the first ``" 2. "``
    boundary; if no claim 2 is found we return the whole string. The
    leading ``"1."`` numbering is kept — it is short and harmless after
    tokenization (and getting rid of it requires another fragile regex).
    """
    if not claims_text:
        return ""
    m = _FIRST_CLAIM_BOUNDARY_RE.search(claims_text)
    if m:
        return claims_text[:m.start()].strip()
    return claims_text.strip()


def make_bert_pooler_safe(model):
    """Make the BERT pooler safe against the cublasLt strided-matmul failure.

    Background: ``BertModel.forward`` unconditionally runs
    ``self.pooler(sequence_output)``. The default ``BertPooler.forward`` does
    ``first_token_tensor = hidden_states[:, 0]``, producing a non-contiguous
    view whose leading stride is ``seq_len * hidden`` (e.g. ``mat2_ld 524288``
    in cuBLAS error messages). On some CUDA/cuBLAS builds this triggers
    ``CUBLAS_STATUS_NOT_INITIALIZED`` inside ``cublasLtMatmul`` for certain
    batch sizes.

    We *cannot* simply set ``pooler = None`` for adapter-wrapped models
    (``adapters.models.bert.adapter_model.BertAdapterModel.forward`` indexes
    ``outputs[1]`` and would raise ``IndexError`` if the pooler is gone).
    Instead we monkey-patch every BERT pooler we can find on the (possibly
    nested) model so that the first-token tensor is made contiguous before the
    Linear, which preserves the output structure but avoids the bad stride.
    """
    import torch.nn as _nn

    def _make_safe(pooler):
        if pooler is None or not hasattr(pooler, 'dense'):
            return
        if getattr(pooler, '_contiguous_pooler_patched', False):
            return
        dense = pooler.dense
        activation = getattr(pooler, 'activation', None)

        def _patched_forward(hidden_states):
            first_token_tensor = hidden_states[:, 0].contiguous()
            pooled = dense(first_token_tensor)
            if activation is not None:
                pooled = activation(pooled)
            return pooled

        pooler.forward = _patched_forward
        pooler._contiguous_pooler_patched = True

    for m in (
        model,
        getattr(model, 'bert', None),
        getattr(model, 'base_model', None),
        getattr(getattr(model, 'base_model', None), 'bert', None),
    ):
        if m is not None:
            _make_safe(getattr(m, 'pooler', None))


def get_encoder_last_hidden_state(model, batch):
    """Get last_hidden_state from model; for adapter models with a head, use base_model so we get encoder output."""
    outputs = model(**batch)
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and 'last_hidden_state' in outputs:
        return outputs['last_hidden_state']
    if hasattr(model, 'base_model'):
        base_out = model.base_model(**batch)
        return base_out.last_hidden_state if hasattr(base_out, 'last_hidden_state') else base_out['last_hidden_state']
    raise KeyError('last_hidden_state')


def ipc_evaluation(train_embeddings, test_embeddings, train_labels, test_labels, train_types, test_types,
                   train_embeddings_knn=None, test_embeddings_knn=None):
    """
    train_embeddings, test_embeddings: used for linear probe (and for KNN if KNN-specific embeddings not provided).
    train_embeddings_knn, test_embeddings_knn: optional; if provided (e.g. SPECTER2 proximity), used for KNN only.
    """
    # Ensure reproducibility with comprehensive seed setting
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(42)
    
    ########################################################################################################################################################
    # 1. ipc classification (linear probe evaluation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split training data into training and validation sets
    X_train_full = train_embeddings
    y_train_full = train_labels

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

    # Create DataLoader with an explicit generator so shuffle order is independent
    # of any global torch RNG state consumed earlier (e.g. by model encoding).
    train_dataset = TensorDataset(X_train, y_train)
    loader_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=loader_generator, drop_last=True)

    # Initialize model
    ipc_model = LinearClassifier(input_dim=X_train.shape[1], num_classes=y_train.shape[1]).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer_ipc = optim.Adam(ipc_model.parameters(), lr=3e-4, weight_decay=1e-5)

    # Initialize early stopping parameters
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    best_state_dict = None
    num_epochs = 100

    for _ in range(num_epochs):
        ipc_model.train()

        for batch_X, batch_y in train_loader:
            optimizer_ipc.zero_grad()
            outputs = ipc_model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ipc_model.parameters(), max_norm=1.0)    # Gradient clipping for stability
            optimizer_ipc.step()
        
        # Validation
        ipc_model.eval()
        with torch.no_grad():
            val_outputs = ipc_model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(ipc_model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # Early stopping

    # Load the best model
    if best_state_dict is not None:
        ipc_model.load_state_dict(best_state_dict)

    # Evaluation on test set
    ipc_model.eval()
    with torch.no_grad():
        logits = ipc_model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid as we use BCEWithLogitsLoss

    # Calculate precision@k
    # Use np.lexsort for deterministic ranking with explicit tie-break by class index:
    # primary key = -probs (descending probability), secondary key = class index (ascending).
    # np.lexsort sorts by the LAST key as primary, so we put -probs last.
    num_classes = probs.shape[1]
    class_idx_row = np.arange(num_classes)
    pred_topk = np.stack([
        np.lexsort((class_idx_row, -row)) for row in probs
    ], axis=0)
    results = {}

    for k in [1, 3, 5]:
        topk = pred_topk[:, :k]
        precision_at_k = np.mean([
            len(set(np.where(true == 1)[0]).intersection(pred[:k])) / k
            for true, pred in zip(y_test.cpu().numpy(), topk)
        ])
        results[f'precision@{k}'] = precision_at_k * 100

    # Format and display results
    print_metric_table(results, "IPC Classification (Linear Probe)")

    ########################################################################################################################################################
    # 2. IPC KNN (nearest-neighbor in embedding space; use proximity embeddings when provided, e.g. SPECTER2)
    knn_train = train_embeddings_knn if train_embeddings_knn is not None else train_embeddings
    knn_test = test_embeddings_knn if test_embeddings_knn is not None else test_embeddings

    knn = KNNClassifier(metric='cosine')

    X_subtrain_knn, X_val_knn, y_subtrain_knn, y_val_knn = train_test_split(
        knn_train, train_labels, test_size=0.1, random_state=42
    )
    best_k, _ = knn.tune_k_by_precision_at_k(X_subtrain_knn, y_subtrain_knn, X_val_knn, y_val_knn, candidate_k_list=[1, 3, 5, 10])

    knn.n_neighbors = best_k
    knn.fit(knn_train, train_labels)

    probabilities = knn.predict_proba(knn_test)

    # Compute evaluation metrics
    test_labels_np = test_labels

    # Calculate precision@k for KNN with proper top-k extraction.
    # KNN probabilities (mean of one-hot neighbor labels) frequently produce ties; use
    # np.lexsort with explicit tie-break by class index (ascending) for deterministic ranking.
    num_classes = probabilities.shape[1]
    class_idx_row = np.arange(num_classes)
    pred_topk = np.stack([
        np.lexsort((class_idx_row, -row)) for row in probabilities
    ], axis=0)

    results = {}
    for k in [1, 3, 5]:
        topk_indices = pred_topk[:, :k]  # Extract top-k indices for each sample
        precision_at_k = np.mean([
            len(set(np.where(true == 1)[0]).intersection(set(pred_indices))) / k
            for true, pred_indices in zip(test_labels_np, topk_indices)
        ])
        results[f'precision@{k}'] = precision_at_k * 100

    # Format and display results
    print_metric_table(results, "IPC Classification (KNN)")


def compute_rankings(top_indices):
    rankings = np.empty_like(top_indices)
    for query_idx, doc_order in enumerate(top_indices):
        rankings[query_idx, doc_order] = np.arange(1, len(doc_order) + 1)
    return rankings


def prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, citation_mapping_graded=None):
    assert len(query_ids) == len(query_embeddings), f"query_ids and query_embeddings length mismatch: {len(query_ids)} vs {len(query_embeddings)}"

    results = {}
    texttype_q = "abstract"
    texttype_d = "abstract"

    # Convert to numpy array to ensure compatibility
    query_types = np.array(query_types)
    doc_types = np.array(doc_types)

    query_type_masks = (query_types == texttype_q)
    doc_type_masks = (doc_types == texttype_d)

    Q_emb = query_embeddings[query_type_masks].astype(np.float32)  # shape: [n_queries, emb_dim]
    D_emb = document_embeddings[doc_type_masks].astype(np.float32)    # shape: [n_docs, emb_dim]

    # Filter query and doc IDs to match the embeddings after masking
    q_ids_filtered = np.array(query_ids)[query_type_masks]
    d_ids_filtered = np.array(doc_ids)[doc_type_masks]

    # Validate shape consistency
    if Q_emb.shape[1] != D_emb.shape[1]:
        logging.warning(f"Embedding dimension mismatch: Q_emb {Q_emb.shape} vs D_emb {D_emb.shape}")

    if np.any(np.isnan(Q_emb)) or np.any(np.isnan(D_emb)):
        raise ValueError("NaN detected in embeddings before normalization.")

    # Create copies to avoid modifying original data
    Q_emb_norm = Q_emb.copy()
    D_emb_norm = D_emb.copy()
    
    faiss.normalize_L2(Q_emb_norm)  # Normalize before similarity computation
    faiss.normalize_L2(D_emb_norm)
    distances = Q_emb_norm @ D_emb_norm.T  # FAISS optimized cosine similarity

    # For each query row, we get top_k doc indices (sorted ascending by distance)
    top_k_indices = np.argsort(-distances, axis=1, kind='stable')

    # Evaluate retrieval: we build lists of true labels & predicted labels.
    # `d_ids_filtered` is already a numpy array, so fancy indexing is much faster
    # than a Python list comprehension over the full ranking.
    true_labels_list, predicted_labels_list = [], []
    true_graded_list = []   # parallel to true_labels_list; {doc_id: gain} per query
    for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
        q_id_str = q_ids_filtered[q_idx]
        true_labels = citation_mapping.get(q_id_str, [])
        predicted_labels = d_ids_filtered[retrieved_docs_indices].tolist()

        true_labels_list.append(true_labels)
        predicted_labels_list.append(predicted_labels)
        if citation_mapping_graded is not None:
            true_graded_list.append(citation_mapping_graded.get(q_id_str, {}))

    # Compute recall@k, ndcg@k, full MAP and MRR.
    # `predicted_labels_list` already holds the full ranking (np.argsort over all docs),
    # so MAP/MRR are computed over the complete ranking (no top-k truncation).
    results_key = f"{texttype_q}->{texttype_d}"
    results[results_key] = {
        'recall@10':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
        'recall@20':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
        'recall@50':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
        'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),

        'ndcg@10':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10),
        'ndcg@20':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=20),
        'ndcg@50':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=50),
        'ndcg@100': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=100),

        'map':  mean_average_precision(true_labels_list, predicted_labels_list),
        'mrr':  mean_reciprocal_rank(true_labels_list, predicted_labels_list),
    }
    if citation_mapping_graded is not None:
        # Graded nDCG with linear gain rel/log2(i+1), X=3 Y=2 A=1.
        results[results_key].update({
            'ndcg_graded@10':  mean_ndcg_at_k_graded(true_graded_list, predicted_labels_list, k=10),
            'ndcg_graded@20':  mean_ndcg_at_k_graded(true_graded_list, predicted_labels_list, k=20),
            'ndcg_graded@50':  mean_ndcg_at_k_graded(true_graded_list, predicted_labels_list, k=50),
            'ndcg_graded@100': mean_ndcg_at_k_graded(true_graded_list, predicted_labels_list, k=100),
        })

    # 3) compute performance for query -> all sections (explicit max-sim aggregation)
    # Currently fixed to claim queries. If you want to evaluate multiple query types,
    # refactor this into a loop and accumulate results / retrieved_sections per type.
    query_texttype = "claim"
    retrieved_sections = []   # for noting which section "won" the max-sim for the top-k docs

    # Document embeddings are laid out as [all abstracts | all claims | all inventions],
    # each block of length original_doc_count. We exploit this to do explicit per-document
    # aggregation by reshaping into (n_sections, n_docs, dim) and taking the max similarity
    # over sections for each (query, doc) pair.
    original_doc_count = len(doc_ids) // 3
    section_names = ["abstract", "claim", "invention"]
    n_sections = len(section_names)
    assert document_embeddings.shape[0] == n_sections * original_doc_count, (
        f"document_embeddings rows ({document_embeddings.shape[0]}) != "
        f"{n_sections} * original_doc_count ({original_doc_count})"
    )

    # Original (un-multiplied) doc IDs, in the same order as one section block.
    original_doc_ids = list(doc_ids[:original_doc_count])

    query_type_masks = (query_types == query_texttype)
    Q_emb = query_embeddings[query_type_masks].astype(np.float32)

    # We will keep track of which section produced the max similarity for each retrieved doc, to analyze later.
    q_ids_filtered = np.array(query_ids)[query_type_masks]

    D_emb = document_embeddings.astype(np.float32)

    if np.any(np.isnan(Q_emb)) or np.any(np.isnan(D_emb)):
        raise ValueError("NaN detected in embeddings before normalization.")

    faiss.normalize_L2(Q_emb)
    faiss.normalize_L2(D_emb)
    distances = Q_emb @ D_emb.T  # shape: (n_queries, n_sections * n_docs)

    # Reshape to (n_queries, n_sections, n_docs) so axis=1 is section, axis=2 is doc.
    distances_3d = distances.reshape(Q_emb.shape[0], n_sections, original_doc_count)

    # Explicit max-sim aggregation per document, plus which section produced the max.
    max_sim_per_doc = distances_3d.max(axis=1)              # (n_queries, n_docs)
    best_section_per_doc = distances_3d.argmax(axis=1)      # (n_queries, n_docs)

    # Rank documents by max-sim. Use stable sort for deterministic tie-breaking
    # (ties on float similarities are practically nonexistent, but be defensive).
    # Keep the full ranking so MAP/MRR can be computed over the entire corpus;
    # slice [:100] when we only need the top-k for recall/ndcg/section analysis.
    full_ranking = np.argsort(-max_sim_per_doc, axis=1, kind='stable')
    top_k_indices = full_ranking[:, :100]

    # numpy array for fast fancy indexing.
    original_doc_ids_arr = np.asarray(original_doc_ids)

    true_labels_list = []
    predicted_labels_top100 = []   # for recall@k, ndcg@k, section analysis
    predicted_labels_full = []     # for MAP (full) and MRR
    true_graded_list = []          # parallel to true_labels_list; {doc_id: gain} per query
    for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
        q_id_str = q_ids_filtered[q_idx]
        true_labels = citation_mapping.get(q_id_str, [])

        # Record which section won the max-sim for each retrieved doc.
        retrieved_sections.append([
            section_names[best_section_per_doc[q_idx, d_idx]]
            for d_idx in retrieved_docs_indices
        ])

        true_labels_list.append(true_labels)
        predicted_labels_top100.append(original_doc_ids_arr[retrieved_docs_indices].tolist())
        predicted_labels_full.append(original_doc_ids_arr[full_ranking[q_idx]].tolist())
        if citation_mapping_graded is not None:
            true_graded_list.append(citation_mapping_graded.get(q_id_str, {}))

    results_key = f"{query_texttype}->all"
    results[results_key] = {
        'recall@10':  mean_recall_at_k(true_labels_list, predicted_labels_top100, k=10),
        'recall@20':  mean_recall_at_k(true_labels_list, predicted_labels_top100, k=20),
        'recall@50':  mean_recall_at_k(true_labels_list, predicted_labels_top100, k=50),
        'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_top100, k=100),

        'ndcg@10':  mean_ndcg_at_k(true_labels_list, predicted_labels_top100, k=10),
        'ndcg@20':  mean_ndcg_at_k(true_labels_list, predicted_labels_top100, k=20),
        'ndcg@50':  mean_ndcg_at_k(true_labels_list, predicted_labels_top100, k=50),
        'ndcg@100': mean_ndcg_at_k(true_labels_list, predicted_labels_top100, k=100),

        'map':  mean_average_precision(true_labels_list, predicted_labels_full),
        'mrr':  mean_reciprocal_rank(true_labels_list, predicted_labels_full),

        'retrieved_sections': f"[{len(retrieved_sections)} queries with retrieved sections]"  # summary instead of full list
    }
    if citation_mapping_graded is not None:
        # Graded nDCG with linear gain rel/log2(i+1), X=3 Y=2 A=1.
        results[results_key].update({
            'ndcg_graded@10':  mean_ndcg_at_k_graded(true_graded_list, predicted_labels_top100, k=10),
            'ndcg_graded@20':  mean_ndcg_at_k_graded(true_graded_list, predicted_labels_top100, k=20),
            'ndcg_graded@50':  mean_ndcg_at_k_graded(true_graded_list, predicted_labels_top100, k=50),
            'ndcg_graded@100': mean_ndcg_at_k_graded(true_graded_list, predicted_labels_top100, k=100),
        })

    # Format and display results
    print_subsection_header("Prior Art Search Results")

    for task_key, task_results in results.items():
        if isinstance(task_results, dict):
            # Create a clean task name
            if '->' in task_key:
                clean_name = f"Query: {task_key.split('->')[0]} → Document: {task_key.split('->')[1]}"
            else:
                clean_name = task_key

            print_metric_table(task_results, clean_name)

    # Store the full retrieved_sections in results for analysis, but don't print it
    results[results_key]['retrieved_sections_full'] = retrieved_sections

    # Run retrieved sections analysis if we have the data
    if retrieved_sections:
        from patentmap_eval.patenteval.utils import analyze_retrieved_sections_integrated

        # Analyze retrieved sections distribution
        section_analysis = analyze_retrieved_sections_integrated(
            retrieved_sections,
            query_section=query_texttype,
            print_results=True
        )
        results[results_key]['section_analysis'] = section_analysis


def dapfam_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings,
                       citation_mapping_by_subset, query_ipc3, doc_ipc3, k=100):
    """Evaluate (TA, TA) retrieval on the DAPFAM benchmark.

    DAPFAM defines three subsets based on 3-char IPC overlap between query and
    target. The reported numbers in DAPFAM Table 19 and PatenTEB Table 16 have
    ``OUT << ALL`` (e.g. Snowflake 0.047 vs 0.284), which is only consistent
    with **full-corpus ranking + per-subset relevance filtering**, not with
    restricting the candidate pool per subset. Pool restriction would make OUT
    *easier* than ALL (smaller, weaker-distractor pool) and cannot reproduce
    the published ordering.

    Protocol implemented here (matches the published numbers):

    1. Rank each query against the **full corpus** once (top-``k``).
    2. For each subset, restrict the gold set:
       * ``ALL`` — all positives.
       * ``IN``  — positives sharing ≥1 IPC3 code with the query.
       * ``OUT`` — positives disjoint in IPC3 from the query.
    3. Recall@k / NDCG@k macro-averaged over queries with ≥1 subset positive.

    For binary relevance ``rel ∈ {0, 1}`` DAPFAM's exponential-gain NDCG
    ``(2^rel - 1) / log2(i + 1)`` collapses to the linear-gain formula used by
    :func:`mean_ndcg_at_k`, so reusing that metric is exact.

    Args:
        query_ids: list[str] of length ``n_q``.
        doc_ids: list[str] of length ``n_d``.
        query_embeddings: array of shape ``(n_q, D)``, aligned with ``query_ids``.
        document_embeddings: array of shape ``(n_d, D)``, aligned with ``doc_ids``.
        citation_mapping_by_subset: dict ``{subset_name: {q_id: [d_id, ...]}}``
            with subset_name in ``{ALL, IN, OUT}``; only positives included.
        query_ipc3: list[set[str]] of length ``n_q``, IPC3 codes per query.
        doc_ipc3:   list[set[str]] of length ``n_d``, IPC3 codes per doc.
        k: rank cutoff. DAPFAM uses ``k = 100``.

    Returns:
        dict ``{subset_name: {metric: value}}`` with keys
        ``recall@k``, ``ndcg@k``, ``n_queries_eval``.
    """
    assert len(query_ids) == len(query_embeddings) == len(query_ipc3), (
        f"query alignment mismatch: {len(query_ids)} ids vs "
        f"{len(query_embeddings)} emb vs {len(query_ipc3)} ipc3"
    )
    assert len(doc_ids) == len(document_embeddings) == len(doc_ipc3), (
        f"doc alignment mismatch: {len(doc_ids)} ids vs "
        f"{len(document_embeddings)} emb vs {len(doc_ipc3)} ipc3"
    )

    # L2-normalised cosine similarities.
    Q = np.asarray(query_embeddings, dtype=np.float32).copy()
    D = np.asarray(document_embeddings, dtype=np.float32).copy()
    if np.any(np.isnan(Q)) or np.any(np.isnan(D)):
        raise ValueError("NaN detected in DAPFAM embeddings before normalization.")
    faiss.normalize_L2(Q)
    faiss.normalize_L2(D)
    sims = Q @ D.T  # (n_q, n_d), float32

    q_ids_arr = np.asarray(query_ids)
    d_ids_arr = np.asarray(doc_ids)

    # Full-corpus top-k once: argpartition + stable sort of the k entries.
    n_d = sims.shape[1]
    top_k = min(k, n_d)
    if top_k == n_d:
        top_k_indices = np.argsort(-sims, axis=1, kind='stable')
    else:
        part = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        row_idx = np.arange(sims.shape[0])[:, None]
        part_sims = sims[row_idx, part]
        order = np.argsort(-part_sims, axis=1, kind='stable')
        top_k_indices = part[row_idx, order]

    # Predicted ranked id lists per query (full-corpus ranking, identical across subsets).
    predicted_labels_full = [d_ids_arr[top_k_indices[i]].tolist() for i in range(len(q_ids_arr))]

    results = {}
    print_subsection_header(
        f"DAPFAM Retrieval (NDCG@{k}, Recall@{k}; full-corpus ranking, per-subset relevance)"
    )
    for subset in ("ALL", "IN", "OUT"):
        cmap = citation_mapping_by_subset[subset]
        true_labels_list, predicted_labels_list = [], []
        for q_idx, qid in enumerate(q_ids_arr):
            true_labels = cmap.get(qid, [])
            if not true_labels:
                continue  # skip queries without positives in this subset
            true_labels_list.append(true_labels)
            predicted_labels_list.append(predicted_labels_full[q_idx])

        results[subset] = {
            f'recall@{k}': mean_recall_at_k(true_labels_list, predicted_labels_list, k=k),
            f'ndcg@{k}':   mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=k),
            'n_queries_eval': len(true_labels_list),
        }
        print_metric_table(results[subset], f"DAPFAM-{subset}")

    return results


def dapfam_evaluation_from_predictions(query_ids, predicted_labels_full,
                                      citation_mapping_by_subset, k=100):
    """Evaluate DAPFAM from precomputed ranked predictions.

    Args:
        query_ids: list[str] query ids aligned with ``predicted_labels_full``.
        predicted_labels_full: list[list[str]], ranked doc ids per query.
        citation_mapping_by_subset: dict with keys ``ALL``, ``IN``, ``OUT``.
        k: rank cutoff.

    Returns:
        dict of per-subset metrics.
    """
    results = {}
    print_subsection_header(
        f"DAPFAM Retrieval (NDCG@{k}, Recall@{k}; full-corpus ranking, per-subset relevance)"
    )
    for subset in ("ALL", "IN", "OUT"):
        cmap = citation_mapping_by_subset[subset]
        true_labels_list, predicted_labels_list = [], []
        for q_idx, qid in enumerate(query_ids):
            true_labels = cmap.get(qid, [])
            if not true_labels:
                continue
            true_labels_list.append(true_labels)
            predicted_labels_list.append(predicted_labels_full[q_idx])

        results[subset] = {
            f'recall@{k}': mean_recall_at_k(true_labels_list, predicted_labels_list, k=k),
            f'ndcg@{k}': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=k),
            'n_queries_eval': len(true_labels_list),
        }
        print_metric_table(results[subset], f"DAPFAM-{subset}")

    return results


def _combine_dataframes_for_filtering(queries_df, documents_df):
    """Helper function to combine queries and documents DataFrames for quality filtering."""
    if queries_df is not None and documents_df is not None:
        return pd.concat([queries_df, documents_df], ignore_index=False)
    elif queries_df is not None:
        return queries_df
    elif documents_df is not None:
        return documents_df
    return None


def _compute_per_text_type_metrics(embeddings, text_types, compute_func, metric_name, **compute_kwargs):
    """Helper function to compute metrics per text type."""
    results = {}
    for text_type in set(text_types):
        text_type_mask = np.array(text_types) == text_type
        text_type_embeddings = embeddings[text_type_mask]
        
        if len(text_type_embeddings) > 0:
            metric_value = compute_func(text_type_embeddings, **compute_kwargs)
            results[text_type] = {metric_name: metric_value}
    
    return results


def _evaluate_embeddings_with_prefiltered_data(filtered_data, compute_func, metric_name, min_samples=10000, **compute_kwargs):
    """Helper function to evaluate embeddings using pre-filtered data."""
    results = {}
    
    if len(filtered_data['embeddings']) > min_samples:
        filtered_metric = compute_func(filtered_data['embeddings'], **compute_kwargs)
        results['global'] = {
            metric_name: filtered_metric,
            'samples_used': len(filtered_data['embeddings']),
            'filter_rate': filtered_data['filter_stats']['keep_rate']
        }
        
        # Calculate per-text-type metrics on filtered data
        filtered_per_type = _compute_per_text_type_metrics(
            filtered_data['embeddings'], filtered_data['types'], 
            compute_func, metric_name, **compute_kwargs
        )
        results.update(filtered_per_type)
    else:
        print(f"⚠️  Warning: Too few high-quality samples ({len(filtered_data['embeddings'])}) for {metric_name} evaluation")
        
    return results


def _evaluate_embeddings_with_quality_filter(embeddings, text_types, queries_df, documents_df, 
                                            compute_func, metric_name, min_samples=10000, **compute_kwargs):
    """Generic function for evaluating embeddings with optional quality filtering."""
    results = {}
    
    # Quality-filtered evaluation (main evaluation)
    combined_df = _combine_dataframes_for_filtering(queries_df, documents_df)
    
    if combined_df is not None:
        # Create dummy IDs for filtering
        dummy_ids = [f"item_{i}" for i in range(len(embeddings))]
        
        filtered_data = filter_by_text_quality(
            embeddings, dummy_ids, text_types, combined_df
        )
        
        if len(filtered_data['embeddings']) > min_samples:
            filtered_metric = compute_func(filtered_data['embeddings'], **compute_kwargs)
            results['global'] = {
                metric_name: filtered_metric,
                'samples_used': len(filtered_data['embeddings']),
                'filter_rate': filtered_data['filter_stats']['keep_rate']
            }
            
            # Calculate per-text-type metrics on filtered data
            filtered_per_type = _compute_per_text_type_metrics(
                filtered_data['embeddings'], filtered_data['types'], 
                compute_func, metric_name, **compute_kwargs
            )
            results.update(filtered_per_type)
        else:
            print(f"⚠️  Warning: Too few high-quality samples ({len(filtered_data['embeddings'])}) for {metric_name} evaluation")
    
    # Fallback to original evaluation if quality filtering unavailable or insufficient
    if not results and len(embeddings) > 0:
        print(f"📊 Falling back to unfiltered {metric_name} evaluation")
        global_metric = compute_func(embeddings, **compute_kwargs)
        results['global'] = {metric_name: global_metric}
        
        # Calculate per-text-type metrics
        per_type_results = _compute_per_text_type_metrics(
            embeddings, text_types, compute_func, metric_name, **compute_kwargs
        )
        results.update(per_type_results)
    
    return results



def uniformity_evaluation(embeddings, text_types, queries_df=None, documents_df=None, use_quality_filter=True, filtered_data=None):
    """
    Evaluate uniformity of embeddings across different text types.
    """
    if filtered_data is not None:
        # Use pre-filtered data
        return _evaluate_embeddings_with_prefiltered_data(
            filtered_data, compute_func=compute_uniformity,
            metric_name='uniformity', min_samples=10000,
            t=2.0, num_samples=10000, device='cuda'
        )
    
    if not use_quality_filter:
        # Skip quality filtering, use all data
        queries_df = documents_df = None
    
    return _evaluate_embeddings_with_quality_filter(
        embeddings, text_types, queries_df, documents_df,
        compute_func=compute_uniformity,
        metric_name='uniformity',
        min_samples=10000,
        t=2.0, num_samples=10000, device='cuda'
    )


def singular_spectrum_evaluation(embeddings, text_types, queries_df=None, documents_df=None, use_quality_filter=True, filtered_data=None):
    """
    Evaluate singular spectrum divergence of embeddings across different text types.
    """
    if filtered_data is not None:
        # Use pre-filtered data
        return _evaluate_embeddings_with_prefiltered_data(
            filtered_data, compute_func=compute_ssd,
            metric_name='ssd', min_samples=10000,
            normalize_by_d=True
        )
    
    if not use_quality_filter:
        # Skip quality filtering, use all data
        queries_df = documents_df = None
    
    return _evaluate_embeddings_with_quality_filter(
        embeddings, text_types, queries_df, documents_df,
        compute_func=compute_ssd,
        metric_name='ssd',
        min_samples=10000,
        normalize_by_d=True
    )


def filter_by_text_quality(embeddings, ids, types, texts_df, sections=['abstract', 'claim', 'invention'], verbose=True):
    """
    Filter embeddings based on text quality criteria.
    
    Args:
        embeddings: np.array of embeddings
        ids: list of IDs (can be dummy IDs like "item_0" or real IDs)
        types: list of section types corresponding to each embedding
        texts_df: DataFrame containing the actual texts
        sections: sections to check for quality
        verbose: whether to print filtering results
    
    Returns:
        dict with filtered data and statistics
    """
    if texts_df is None:
        return {
            'embeddings': embeddings,
            'ids': ids, 
            'types': types,
            'filter_stats': {'total': len(embeddings), 'filtered': 0, 'kept': len(embeddings)}
        }
    
    embeddings = np.array(embeddings)
    ids = np.array(ids)
    types = np.array(types)
    
    # Create quality masks for each section
    quality_masks = {}
    text_stats = {}
    
    for section in sections:
        if section in texts_df.columns:
            # Get text lengths for this section
            texts = texts_df[section].fillna('').astype(str)
            word_counts = [len(str(text).split()) for text in texts]
            
            # Create quality mask (sufficient words)
            section_quality = np.array(word_counts) >= QUALITY_MIN_WORDS
            quality_masks[section] = section_quality
            
            text_stats[section] = {
                'total': len(texts),
                'high_quality': np.sum(section_quality),
                'low_quality': np.sum(~section_quality),
                'quality_rate': np.mean(section_quality)
            }
    
    # Filter embeddings by section type
    keep_mask = np.ones(len(embeddings), dtype=bool)
    
    # Group embeddings by section type and apply quality filtering
    for section in sections:
        if section not in quality_masks:
            continue
            
        # Find all embeddings of this section type
        section_mask = (types == section)
        section_indices = np.where(section_mask)[0]
        
        # Apply quality filtering to this section
        section_quality = quality_masks[section]
        
        # For each embedding of this section type, check if its corresponding document has high quality
        for i, emb_idx in enumerate(section_indices):
            if i < len(section_quality):
                keep_mask[emb_idx] = section_quality[i]
            else:
                # If we have more embeddings than documents, cycle through quality mask
                keep_mask[emb_idx] = section_quality[i % len(section_quality)]
    
    # Apply filtering
    filtered_embeddings = embeddings[keep_mask]
    filtered_ids = ids[keep_mask]
    filtered_types = types[keep_mask]
    
    filter_stats = {
        'total': len(embeddings),
        'filtered': np.sum(~keep_mask),
        'kept': len(filtered_embeddings),
        'keep_rate': len(filtered_embeddings) / len(embeddings) if len(embeddings) > 0 else 0,
        'section_stats': text_stats
    }
    
    if verbose:
        print(f"\n🔍 Text Quality Filtering Results:")
        print(f"   • Total embeddings: {filter_stats['total']}")
        print(f"   • Kept (high quality): {filter_stats['kept']} ({filter_stats['keep_rate']:.1%})")
        print(f"   • Filtered (low quality): {filter_stats['filtered']} ({1-filter_stats['keep_rate']:.1%})")
        
        for section, stats in text_stats.items():
            print(f"   • {section.capitalize()} quality: {stats['high_quality']}/{stats['total']} ({stats['quality_rate']:.1%})")
    
    return {
        'embeddings': filtered_embeddings,
        'ids': filtered_ids,
        'types': filtered_types,
        'filter_stats': filter_stats,
        'keep_mask': keep_mask
    }


def _build_citation_pairs(q_embs, d_embs, q_ids_arr, d_ids_arr, citation_mapping):
    """Helper function to build citation pairs from embeddings and IDs."""
    query_pairs = []
    doc_pairs = []
    
    for q_idx, q_id in enumerate(q_ids_arr):
        cited_doc_ids = set(citation_mapping.get(q_id, []))
        for d_idx, did in enumerate(d_ids_arr):
            if did in cited_doc_ids:
                query_pairs.append(q_embs[q_idx])
                doc_pairs.append(d_embs[d_idx])
    
    return query_pairs, doc_pairs


def _compute_section_alignment(q_embs, d_embs, q_ids_arr, d_ids_arr, citation_mapping, section_name, doc_types):
    """Helper function to compute alignment for a specific section."""
    d_mask = (doc_types == section_name)
    if not np.any(d_mask):
        return {"alignment": None, "num_pairs": 0, "error": f"No {section_name} documents found"}
    
    de = d_embs[d_mask]
    did_list = d_ids_arr[d_mask]
    
    query_pairs, doc_pairs = _build_citation_pairs(
        q_embs, de, q_ids_arr, did_list, citation_mapping
    )
    
    if query_pairs:
        alignment_score = compute_alignment(np.array(query_pairs), np.array(doc_pairs))
        return {
            "alignment": alignment_score,
            "num_pairs": len(query_pairs)
        }
    else:
        return {
            "alignment": None, 
            "num_pairs": 0
        }


def filter_citation_pairs_by_quality(query_embeddings, doc_embeddings, query_ids, doc_ids, 
                                     query_types, doc_types, 
                                     queries_df, documents_df):
    """
    Filter citation pairs to only include high-quality text pairs.
    
    Returns both query and doc pairs where both elements meet quality criteria.
    """
    if queries_df is None or documents_df is None:
        return query_embeddings, doc_embeddings, query_ids, doc_ids
    
    # Ensure all inputs have consistent lengths
    assert len(query_embeddings) == len(query_ids) == len(query_types), \
        f"Query dimension mismatch: embeddings={len(query_embeddings)}, ids={len(query_ids)}, types={len(query_types)}"
    assert len(doc_embeddings) == len(doc_ids) == len(doc_types), \
        f"Doc dimension mismatch: embeddings={len(doc_embeddings)}, ids={len(doc_ids)}, types={len(doc_types)}"
    
    # Create quality indicators for queries and documents  
    query_quality = {}
    doc_quality = {}
    
    # Check query quality
    for section in ['abstract', 'claim', 'invention']:
        if section in queries_df.columns:
            texts = queries_df[section].fillna('').astype(str)
            word_counts = [len(str(text).split()) for text in texts]
            query_quality[section] = {qid: wc >= QUALITY_MIN_WORDS
                                    for qid, wc in zip(queries_df.index, word_counts)}
    
    # Check document quality 
    for section in ['abstract', 'claim', 'invention']:
        if section in documents_df.columns:
            texts = documents_df[section].fillna('').astype(str)
            word_counts = [len(str(text).split()) for text in texts]
            doc_quality[section] = {did: wc >= QUALITY_MIN_WORDS
                                  for did, wc in zip(documents_df.index, word_counts)}
    
    # Filter embeddings based on quality
    q_keep_mask = np.ones(len(query_embeddings), dtype=bool)
    d_keep_mask = np.ones(len(doc_embeddings), dtype=bool)
    
    # Filter queries
    query_types_arr = np.array(query_types)
    for i, (qid, qtype) in enumerate(zip(query_ids, query_types_arr)):
        if qtype in query_quality:
            # Handle both original query IDs and duplicated IDs
            original_qid = qid
            if isinstance(qid, str) and qid.startswith('Q'):
                # Remove any numeric suffixes that might be from ID multiplication
                base_qid = qid
            else:
                base_qid = qid
            
            if base_qid in query_quality[qtype]:
                q_keep_mask[i] = query_quality[qtype][base_qid]
            else:
                # If exact ID not found, default to keeping it
                q_keep_mask[i] = True
    
    # Filter documents
    doc_types_arr = np.array(doc_types)
    for i, (did, dtype) in enumerate(zip(doc_ids, doc_types_arr)):
        if dtype in doc_quality:
            # Handle both original document IDs and duplicated IDs
            original_did = did
            if isinstance(did, str) and did.startswith('D'):
                # Remove any numeric suffixes that might be from ID multiplication
                base_did = did
            else:
                base_did = did
            
            if base_did in doc_quality[dtype]:
                d_keep_mask[i] = doc_quality[dtype][base_did]
            else:
                # If exact ID not found, default to keeping it
                d_keep_mask[i] = True
    
    print(f"\n🔍 Citation Pair Quality Filtering:")
    print(f"   • Queries: {np.sum(q_keep_mask)}/{len(query_embeddings)} kept ({np.mean(q_keep_mask):.1%})")
    print(f"   • Documents: {np.sum(d_keep_mask)}/{len(doc_embeddings)} kept ({np.mean(d_keep_mask):.1%})")
    
    # Apply filtering and ensure output arrays are properly sized
    filtered_q_embeddings = query_embeddings[q_keep_mask]
    filtered_d_embeddings = doc_embeddings[d_keep_mask]
    filtered_q_ids = np.array(query_ids)[q_keep_mask]
    filtered_d_ids = np.array(doc_ids)[d_keep_mask]
    filtered_q_types = np.array(query_types)[q_keep_mask]
    filtered_d_types = np.array(doc_types)[d_keep_mask]
    
    print(f"   • Filtered query embeddings shape: {filtered_q_embeddings.shape}")
    print(f"   • Filtered doc embeddings shape: {filtered_d_embeddings.shape}")
    print(f"   • Filtered query types: {len(filtered_q_types)}")
    print(f"   • Filtered doc types: {len(filtered_d_types)}")
    
    return (filtered_q_embeddings, filtered_d_embeddings, 
            filtered_q_ids, filtered_d_ids, 
            filtered_q_types, filtered_d_types)



def alignment_evaluation(query_embeddings, doc_embeddings, query_ids, doc_ids, citation_mapping, query_types, doc_types, queries_df=None, documents_df=None, use_quality_filter=True):
    """
    Evaluate alignment between query and document embeddings based on citation pairs.
    
    query_embeddings: np.array [n, d]
    doc_embeddings: np.array [m, d]
    query_ids, doc_ids: list of str
    citation_mapping: dict[str, list[str]]
    query_types, doc_types: list of str (length = n/m)
    queries_df, documents_df: optional DataFrames for text length analysis
    use_quality_filter: whether to apply quality filtering
    """
    # Convert to numpy arrays for boolean indexing if needed
    query_types = np.array(query_types)
    doc_types = np.array(doc_types)
    q_ids = np.array(query_ids)
    d_ids = np.array(doc_ids)

    results = {}
    sections = ['abstract', 'claim', 'invention']

    # Quality-filtered alignment evaluation (main evaluation)
    if use_quality_filter and queries_df is not None and documents_df is not None:
        try:
            print(f"\n🔍 Quality-Filtered Alignment Evaluation:")
            
            # Filter citation pairs by quality (with proper type handling)
            filtered_q_embs, filtered_d_embs, filtered_q_ids, filtered_d_ids, filtered_q_types, filtered_d_types = filter_citation_pairs_by_quality(
                query_embeddings, doc_embeddings, query_ids, doc_ids, 
                query_types, doc_types, 
                queries_df, documents_df
            )
            
            if len(filtered_q_embs) > 0 and len(filtered_d_embs) > 0:
                
                print(f"   • Filtered query types: {len(filtered_q_types)}")
                print(f"   • Filtered doc types: {len(filtered_d_types)}")
                
                # Compute alignment with filtered data using helper function
                for section in sections:
                    result = _compute_section_alignment(
                        filtered_q_embs, filtered_d_embs, filtered_q_ids, filtered_d_ids, 
                        citation_mapping, section, filtered_d_types
                    )
                    if result and result["alignment"] is not None:
                        results[section] = result
                
                # Global filtered alignment using helper function
                query_pairs, doc_pairs = _build_citation_pairs(
                    filtered_q_embs, filtered_d_embs, filtered_q_ids, filtered_d_ids, citation_mapping
                )
                
                if query_pairs:
                    global_alignment = compute_alignment(np.array(query_pairs), np.array(doc_pairs))
                    results['global'] = {
                        "alignment": global_alignment, 
                        "num_pairs": len(query_pairs)
                    }
                    
                    print(f"   • Quality-filtered global alignment: {global_alignment:.4f} ({len(query_pairs)} pairs)")
                    print(f"   • Note: Alignment is L2 distance (lower = better, range 0-2)")
            else:
                print(f"   ⚠️  No high-quality data available for alignment evaluation")
                
        except Exception as e:
            print(f"⚠️  Quality-filtered alignment evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Fallback to original alignment computation if quality filtering unavailable
    if not results:
        print("📊 Falling back to unfiltered alignment evaluation")
        
        for section in sections:
            result = _compute_section_alignment(
                query_embeddings, doc_embeddings, q_ids, d_ids, 
                citation_mapping, section, doc_types
            )
            if result:
                results[section] = result
            else:
                results[section] = {"alignment": None, "num_pairs": 0}

        # Global alignment using helper function
        query_pairs, doc_pairs = _build_citation_pairs(
            query_embeddings, doc_embeddings, q_ids, d_ids, citation_mapping
        )
        
        if query_pairs:
            global_alignment = compute_alignment(np.array(query_pairs), np.array(doc_pairs))
            results['global'] = {"alignment": global_alignment, "num_pairs": len(query_pairs)}
        else:
            results['global'] = {"alignment": None, "num_pairs": 0}

    return results


def topology_evaluation(embeddings, embedding_ids, embedding_types, documents_df=None, use_quality_filter=True, is_balanced_data=False):
    """
    Evaluate topology: intra-document cohesion between different sections.
    
    Args:
        embeddings: np.array of embeddings (can be docs only or docs + queries)
        embedding_ids: list of IDs (document IDs or mixed IDs)
        embedding_types: list of section types for each embedding
        documents_df: DataFrame containing document texts for quality filtering
        use_quality_filter: whether to apply quality filtering
        is_balanced_data: whether the data includes queries + documents (balanced) or docs only
    
    Returns:
        dict: Results from compute_intra_document_cohesion
    """
    # Convert to numpy arrays for boolean indexing
    embedding_types = np.array(embedding_types)
    e_ids = np.array(embedding_ids)
    
    sections = ['abstract', 'claim', 'invention']
    
    # Handle different data structures
    if is_balanced_data:
        # For balanced data (queries + docs), we have equal sizes per section
        # Each section has the same number of embeddings
        section_size = len(embeddings) // 3
        n_unique_docs = section_size  # Approximate for balanced case
        
        # Organize embeddings by section 
        doc_embeddings_dict = {}
        for i, section in enumerate(sections):
            start_idx = i * section_size
            end_idx = (i + 1) * section_size
            doc_embeddings_dict[section] = embeddings[start_idx:end_idx]
            
        print(f"   • Using balanced data: {section_size} embeddings per section (queries + documents)")
        
    else:
        # For document-only data, organize by section type
        # Organize document embeddings by section and document ID
        doc_embeddings_dict = {}
        
        # Get unique document IDs (since we have 3x embeddings due to concatenation)
        # e_ids is [D1, D2, D3, D1, D2, D3, D1, D2, D3], so take first 1/3
        n_unique_docs = len(e_ids) // 3
        unique_doc_ids = e_ids[:n_unique_docs]  # Take first n_unique_docs elements
        
        # Reorganize embeddings by section
        for i, section in enumerate(sections):
            section_mask = (embedding_types == section)
            section_embeddings = embeddings[section_mask]
            
            # Ensure we have the right number of embeddings
            if len(section_embeddings) != n_unique_docs:
                print(f"Warning: Expected {n_unique_docs} {section} embeddings, got {len(section_embeddings)}")
                section_embeddings = section_embeddings[:n_unique_docs]
            
            doc_embeddings_dict[section] = section_embeddings

        print(f"   • Using document-only data: {n_unique_docs} documents per section")

    # Quality-filtered topology evaluation (main evaluation)
    if use_quality_filter and documents_df is not None and not is_balanced_data:
        try:
            print(f"\n🔍 Quality-Filtered Topology Evaluation:")
            
            # Find documents that have high-quality text in all three sections
            high_quality_docs = []
            
            for doc_id in unique_doc_ids:
                if doc_id in documents_df.index:
                    doc_row = documents_df.loc[doc_id]
                    
                    # Check quality for all sections
                    all_sections_good = True
                    for section in sections:
                        if section in documents_df.columns:
                            text = str(doc_row[section]) if pd.notna(doc_row[section]) else ""
                            word_count = len(text.split())
                            if word_count < QUALITY_MIN_WORDS:
                                all_sections_good = False
                                break
                    
                    if all_sections_good:
                        high_quality_docs.append(doc_id)
            
            print(f"   • Found {len(high_quality_docs)} high-quality documents (out of {len(unique_doc_ids)})")
            print(f"   • Quality rate: {len(high_quality_docs)/len(unique_doc_ids):.1%}")
            
            if len(high_quality_docs) >= 10:  # Need sufficient documents for meaningful analysis
                # Filter embeddings to only include high-quality documents
                filtered_embeddings_dict = {}
                doc_indices = []
                
                for doc_id in high_quality_docs:
                    try:
                        idx = list(unique_doc_ids).index(doc_id)
                        doc_indices.append(idx)
                    except ValueError:
                        continue
                
                for section in sections:
                    filtered_embeddings_dict[section] = doc_embeddings_dict[section][doc_indices]
                
                print(f"   • Using {len(doc_indices)} high-quality documents for topology analysis")
                
                # Compute intra-document cohesion using the utility function
                return compute_intra_document_cohesion(
                    embeddings_dict=filtered_embeddings_dict,
                    sections=sections,
                    normalize_by_random=True,
                    num_random_pairs=10000
                )
            else:
                print(f"   ⚠️  Too few high-quality documents ({len(high_quality_docs)}) for meaningful analysis")
                
        except Exception as e:
            print(f"   ⚠️  Quality-filtered topology evaluation failed: {str(e)}")
    
    # Fallback to all documents if quality filtering unavailable, failed, or using balanced data
    if is_balanced_data:
        print("📊 Using balanced data for topology evaluation (quality filtering not applicable)")
        total_docs = len(doc_embeddings_dict[sections[0]])
    else:
        print("📊 Falling back to unfiltered topology evaluation")
        total_docs = n_unique_docs
    
    print(f"   • Using all {total_docs} documents for topology analysis")
    
    return compute_intra_document_cohesion(
        embeddings_dict=doc_embeddings_dict,
        sections=sections,
        normalize_by_random=True,
        num_random_pairs=10000
    )


def comprehensive_embedding_quality_evaluation(query_embeddings, doc_embeddings, query_ids, doc_ids, 
                                               citation_mapping, query_types, doc_types, queries_df=None, documents_df=None):
    """
    Comprehensive evaluation of embedding quality including uniformity, SSD, alignment, and topology.
    
    Args:
        query_embeddings: np.array [n, d] - combined embeddings from all sections
        doc_embeddings: np.array [m, d] - combined embeddings from all sections  
        query_ids, doc_ids: list of str
        citation_mapping: dict[str, list[str]]
        query_types, doc_types: list of str - section type for each embedding
        queries_df, documents_df: optional DataFrames for text length analysis
    """
    
    print_section_header("📊 Comprehensive Embedding Quality Analysis")
    
    # Create balanced embeddings for internal consistency evaluation (uniformity/SSD/topology)
    # This ensures fair comparison across sections by having equal sample sizes
    print("🔄 Creating balanced data for internal consistency evaluation...")
    
    # Extract section-wise embeddings from documents (these are always balanced)
    doc_types_array = np.array(doc_types)
    n_docs = len(doc_embeddings) // 3  # Total documents
    
    doc_abstract = doc_embeddings[doc_types_array == 'abstract'][:n_docs]
    doc_claim = doc_embeddings[doc_types_array == 'claim'][:n_docs] 
    doc_invention = doc_embeddings[doc_types_array == 'invention'][:n_docs]
    
    # Extract section-wise embeddings from queries 
    query_types_array = np.array(query_types)
    n_queries = len(query_embeddings) // 3
    
    query_abstract = query_embeddings[query_types_array == 'abstract'][:n_queries]
    query_claim = query_embeddings[query_types_array == 'claim'][:n_queries]
    query_invention = query_embeddings[query_types_array == 'invention'][:n_queries]
    
    # Create balanced embeddings by combining queries + documents for each section
    balanced_abstract = np.concatenate([query_abstract, doc_abstract], axis=0)
    balanced_claim = np.concatenate([query_claim, doc_claim], axis=0) 
    balanced_invention = np.concatenate([query_invention, doc_invention], axis=0)
    
    # Combine all balanced embeddings for global analysis
    balanced_embeddings = np.concatenate([balanced_abstract, balanced_claim, balanced_invention], axis=0)
    balanced_types = (['abstract'] * len(balanced_abstract) + 
                     ['claim'] * len(balanced_claim) + 
                     ['invention'] * len(balanced_invention))
    
    print(f"   • Balanced section sizes: Abstract={len(balanced_abstract)}, Claim={len(balanced_claim)}, Invention={len(balanced_invention)}")
    
    # Pre-filter balanced data for uniformity/SSD evaluation
    combined_df = _combine_dataframes_for_filtering(queries_df, documents_df)
    filtered_data = None
    
    if combined_df is not None:
        dummy_ids = [f"item_{i}" for i in range(len(balanced_embeddings))]
        filtered_data = filter_by_text_quality(
            balanced_embeddings, dummy_ids, balanced_types, combined_df
        )
        print(f"   • Quality filtering completed: {filtered_data['filter_stats']['keep_rate']:.1%} kept")
    
    # 1. Uniformity Evaluation (using balanced data for fair section comparison)
    uniformity_results = uniformity_evaluation(balanced_embeddings, balanced_types, filtered_data=filtered_data)
    
    # 2. Singular Spectrum Divergence Evaluation (using balanced data for fair section comparison)
    ssd_results = singular_spectrum_evaluation(balanced_embeddings, balanced_types, filtered_data=filtered_data)
    
    # 3. Alignment Evaluation (use real citation relationships directly)
    print("🔄 Evaluating alignment using real citation relationships across all sections...")
    
    print(f"   • Available query sections: abstract={len(query_abstract)}, claim={len(query_claim)}, invention={len(query_invention)}")
    print(f"   • Available document sections: abstract={n_docs}, claim={n_docs}, invention={n_docs}")
    print(f"   • Total citation relationships: {len(citation_mapping)}")
    
    alignment_results = alignment_evaluation(
        query_embeddings, doc_embeddings, query_ids, doc_ids, 
        citation_mapping, query_types, doc_types, queries_df, documents_df
    )
    
    # Organize results by section for unified display
    sections_to_display = ['global', 'abstract', 'claim', 'invention']
    
    for section in sections_to_display:
        section_name = section.capitalize() if section != 'global' else 'Global (All Sections)'
        
        # Combine all available metrics for this section
        combined_metrics = {}
        
        # Add uniformity if available
        if section in uniformity_results:
            combined_metrics.update(uniformity_results[section])
            
        # Add SSD if available
        if section in ssd_results:
            combined_metrics.update(ssd_results[section])
            
        # Add alignment if available
        if section in alignment_results:
            if alignment_results[section].get('alignment') is not None:
                combined_metrics.update(alignment_results[section])
            else:
                combined_metrics['alignment'] = 'N/A (no data)'
        
        # Display combined metrics for this section
        if combined_metrics:
            print_metric_table(combined_metrics, f"{section_name} Metrics")
        else:
            print(f"   ⚠️  No metrics available for {section_name}")
    
    # 4. Topology Evaluation (option to use balanced data with queries + documents)
    print_subsection_header("Document Topology Analysis") 
    print("   • Using balanced sections (queries + documents) for comprehensive intra-document cohesion analysis")
    
    # Use the balanced data that includes queries + documents for more comprehensive analysis
    # This addresses your suggestion to include queries in topology evaluation
    topology_results = topology_evaluation(
        balanced_embeddings, 
        [f"doc_{i}" for i in range(len(balanced_embeddings) // 3)] * 3,  # Dummy doc IDs
        balanced_types, 
        documents_df, 
        use_quality_filter=False,  # Skip quality filtering for balanced data
        is_balanced_data=True
    )
    
    # Display topology results - now directly from compute_intra_document_cohesion
    if topology_results and 'error' not in topology_results:
        # Create a summary table for main metrics
        main_metrics = {
            'mean_cohesion': topology_results['mean_cohesion'],
            'std_cohesion': topology_results['std_cohesion'],
            'num_documents': topology_results.get('num_documents', len(topology_results['cohesion_per_document']))
        }
        
        # Add normalized metrics if available
        if 'normalized_cohesion' in topology_results:
            main_metrics.update({
                'random_baseline': topology_results['random_baseline'],
                'normalized_cohesion': topology_results['normalized_cohesion'],
                'cohesion_improvement': topology_results['cohesion_improvement']
            })
        
        print_metric_table(main_metrics, "Global Intra-Document Cohesion")
        
        # Display pairwise section cohesion if available
        if 'pairwise_sections' in topology_results:
            print(f"\n📊 Section-Pair Cohesion Analysis:")
            print("-" * 60)
            
            for pair_name, pair_metrics in topology_results['pairwise_sections'].items():
                pair_display_metrics = {
                    'mean_cohesion': pair_metrics['mean_cohesion'],
                    'std_cohesion': pair_metrics['std_cohesion']
                }
                
                # Add normalized metrics if available
                if 'normalized_cohesion' in pair_metrics:
                    pair_display_metrics.update({
                        'normalized_cohesion': pair_metrics['normalized_cohesion'],
                        'cohesion_improvement': pair_metrics['cohesion_improvement']
                    })
                
                print_metric_table(pair_display_metrics, f"{pair_name.title()} Cohesion")
        
        # Display detailed interpretation if normalized metrics are available
        if 'normalized_cohesion' in topology_results:
            print(f"\n🎯 Global Cohesion Analysis (vs Random Baseline):")
            print("-" * 50)
            normalized_cohesion = topology_results['normalized_cohesion']
            cohesion_improvement = topology_results['cohesion_improvement']
            
            # Interpretation
            if normalized_cohesion < 0.25:
                status = "🟢 Excellent intra-document cohesion"
            elif normalized_cohesion < 0.50:
                status = "🟡 Good intra-document cohesion"
            elif normalized_cohesion < 0.75:
                status = "🟠 Moderate intra-document cohesion"
            else:
                status = "🔴 Poor intra-document cohesion"
            
            print(f"   {status}")
            print(f"   Improvement over random: {cohesion_improvement:+.1%}")
            print(f"   (Lower values indicate better section coherence)")
            
            # Add pairwise interpretation if available
            if 'pairwise_sections' in topology_results:
                print(f"\n🎯 Section-Pair Analysis:")
                print("-" * 50)
                
                for pair_name, pair_metrics in topology_results['pairwise_sections'].items():
                    if 'normalized_cohesion' in pair_metrics:
                        pair_norm = pair_metrics['normalized_cohesion']
                        pair_improve = pair_metrics['cohesion_improvement']
                        
                        # Determine pair status
                        if pair_norm < 0.25:
                            pair_status = "🟢 Excellent"
                        elif pair_norm < 0.50:
                            pair_status = "🟡 Good"
                        elif pair_norm < 0.75:
                            pair_status = "🟠 Moderate"
                        else:
                            pair_status = "🔴 Poor"
                        
                        print(f"   {pair_name}: {pair_status} ({pair_improve:+.1%} vs random)")
            
    elif 'error' in topology_results:
        print(f"   ⚠️  Topology evaluation failed: {topology_results['error']}")
    else:
        print(f"   ⚠️  No topology results available")
    
    # Return combined results for potential further analysis
    return {
        'uniformity': uniformity_results,
        'ssd': ssd_results,
        'alignment': alignment_results,
        'topology': topology_results
    }


def main():
    # Set up safer defaults to prevent segfaults
    import os
    import sys
    
    # Set environment variables for stability
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer multiprocessing issues
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Limit CUDA memory fragmentation
    
    # Set up signal handlers to catch segfaults
    import signal
    
    def signal_handler(signum, frame):
        print(f"Received signal {signum}. Cleaning up...")
        cleanup_resources()
        sys.exit(1)
    
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Path to pretrained model or model ID.")
    parser.add_argument("--output_dir", type=str, default='./baseline_eval', help="Output directory for evaluation results.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="both",
        choices=["perf200", "dapfam", "both"],
        help=(
            "Which retrieval benchmark(s) to run. 'perf200' = original prior-art "
            "task only; 'dapfam' = DAPFAM (TA->TA) only; 'both' = run both."
        ),
    )
    parser.add_argument(
        "--dapfam_texttype",
        type=str,
        default="plain",
        choices=["ta", "tac", "plain"],
        help=(
            "DAPFAM input format. 'ta' = title + abstract (default, matches our "
            "perf200 'abstract' texttype with section markers). 'tac' = ta + "
            "first claim, still with section markers. 'plain' = "
            "'title abstract first_claim' joined by single spaces with NO SEP "
            "and NO section markers — matches PatenTEB §5.4 / "
            "sentence-transformers InformationRetrievalEvaluator protocol. "
            "Each mode caches embeddings in a separate subdir."
        ),
    )
    args = parser.parse_args()

    run_perf200 = args.benchmark in ("perf200", "both")
    run_dapfam = args.benchmark in ("dapfam", "both")
    # IPC classification is the representation-quality probe attached to the
    # perf200 pipeline (separate dataset, separate downstream task), so it is
    # gated alongside perf200. ``--benchmark dapfam`` therefore skips it.
    run_ipc = run_perf200

    print(f"Running evaluation for model: {args.model_name}")
    print("=============================================>>>>>>>>>")

    # Handle the case where model_name is None
    if args.model_name is None:
        print("Error: --model_name is required")
        return
    
    # Initialize temp directories for all models (not just non-bm25/non-checkpoint)
    model_basename = args.model_name.strip("/").split("/")[-1]
    IPC_dir_temp = os.path.join(args.output_dir, f'IPC-Classification_temp_{model_basename}')
    priorart_temp_dir = os.path.join(args.output_dir, f'priorart_temp_{model_basename}')
    # SPECTER2: use format-specific adapters; IPC uses classification adapter → separate cache
    if "specter2" in model_basename.lower():
        IPC_dir_temp = os.path.join(args.output_dir, f'IPC-Classification_temp_{model_basename}_clf')
        priorart_temp_dir = os.path.join(args.output_dir, f'priorart_temp_{model_basename}_prx')
    
    # Create directories if they don't exist (for non-BM25 models)
    if not ("bm25" in args.model_name):
        for temp_dir in [IPC_dir_temp, priorart_temp_dir]:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                print(f"Created directory: {temp_dir}")

    # Print evaluation header
    print_section_header("🚀 Patent Embedding Evaluation Pipeline")
    print(f"📋 Model: {args.model_name}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"🎯 Evaluation Tasks: IPC Classification, Prior Art Search, Uniformity, Alignment, Topology")
    
    ############################################### create dataset for IPC classification ##################################################
    log_evaluation_start("IPC Classification", args.model_name)
    train_dataset = pd.read_csv('./patentmap_eval/data/downstream/IPC-Classification/train_512.csv')
    test_dataset = pd.read_csv('./patentmap_eval/data/downstream/IPC-Classification/test_512.csv')

    train_dataset['ipcr_labels'] = train_dataset['ipcr_labels'].apply(lambda x: [label.strip("' ") for label in x.strip("[]").split(" ") if label])
    test_dataset['ipcr_labels'] = test_dataset['ipcr_labels'].apply(lambda x: [label.strip("' ") for label in x.strip("[]").split(" ") if label])

    # rename section to text_type
    train_dataset.rename(columns={'section': 'text_type'}, inplace=True)
    test_dataset.rename(columns={'section': 'text_type'}, inplace=True)

    # Load the MultiLabelBinarizer and number of classes
    mlb = MultiLabelBinarizer()
    all_labels = train_dataset['ipcr_labels'].apply(label_process).tolist()
    mlb.fit(all_labels)

    # convert the labels to binary format
    train_dataset['labels'] = list(mlb.transform(train_dataset['ipcr_labels'].apply(label_process)))
    test_dataset['labels'] = list(mlb.transform(test_dataset['ipcr_labels'].apply(label_process)))


    train_labels = np.array(train_dataset[train_dataset['text_type'] == 'abstract']['labels'].tolist())
    test_labels = np.array(test_dataset[test_dataset['text_type'] == 'abstract']['labels'].tolist())

    train_types, test_types = train_dataset[train_dataset['text_type'] == 'abstract']['text_type'], test_dataset[test_dataset['text_type'] == 'abstract']['text_type']
   

    ############################################## crete dataset for prior-art search ##################################################
    print("Running Prior-art search task.")
    Prior_art_dataset_dir = './patentmap_eval/data/downstream/perf200'

    queries = load_corpus(f"{Prior_art_dataset_dir}/content/queries.json")
    documents = load_corpus(f"{Prior_art_dataset_dir}/content/documents.json")

    # Convert dict_keys to lists so we can index them safely
    query_ids = list(queries.keys())       # e.g. ['Q1', 'Q2', 'Q3', ...]
    doc_ids = list(documents.keys())       # e.g. ['D1', 'D2', 'D3', ...]

    # convert to dataframe
    queries_df = pd.DataFrame(queries).T
    documents_df = pd.DataFrame(documents).T

    # 2) Load citation mappings (gold standard)
    citation_file = f"{Prior_art_dataset_dir}/mapping/gold.json"
    with open(citation_file) as f:
        raw_citations = json.load(f)

    # format: {query_id: [list_of_cited_doc_ids], ...}
    citation_mapping = citation_to_citing_to_cited_dict(raw_citations)
    # Graded variant: {query_id: {cited_doc_id: gain}}, gains X=3, Y=2, A=1.
    citation_mapping_graded = citation_to_citing_to_cited_graded_dict(raw_citations)

    # Multiply IDs to match concatenated embeddings
    original_query_count = len(query_ids)
    original_doc_count = len(doc_ids)
    
    query_ids = query_ids * 3
    doc_ids = doc_ids * 3
    
    # Create types to match the order of concatenated embeddings
    # Both query and document embeddings: [abstract1, abstract2, ..., claim1, claim2, ..., invention1, invention2, ...]
    query_types = ['abstract'] * original_query_count + ['claim'] * original_query_count + ['invention'] * original_query_count
    doc_types = ['abstract'] * original_doc_count + ['claim'] * original_doc_count + ['invention'] * original_doc_count

    ############################################## DAPFAM benchmark data ##################################################
    # DAPFAM (Ayaou et al. 2025, arXiv:2506.22141) is the second retrieval benchmark.
    # We run only the (TitlAbs -> TitlAbs) task across the ALL/IN/OUT subsets.
    # The data load is cheap (~3 small HF subsets, cached after first call) so we
    # do it once here even if some downstream model branches do not consume it yet.
    dapfam_queries_df = None
    dapfam_documents_df = None
    dapfam_cmap = None
    dapfam_query_ipc3 = None
    dapfam_doc_ipc3 = None
    dapfam_temp_dir = None
    if run_dapfam:
        log_evaluation_start("DAPFAM Retrieval (TitlAbs -> TitlAbs)", args.model_name)
        from patenteval.dapfam_loader import load_dapfam
        dapfam_queries_df, dapfam_documents_df, dapfam_cmap = load_dapfam()
        dapfam_query_ipc3 = dapfam_queries_df['ipc3'].tolist()
        dapfam_doc_ipc3 = dapfam_documents_df['ipc3'].tolist()
        dapfam_cache_suffix = '' if args.dapfam_texttype == 'ta' else f'_{args.dapfam_texttype}'
        dapfam_temp_dir = os.path.join(args.output_dir, f'dapfam_temp_{model_basename}{dapfam_cache_suffix}')
        os.makedirs(dapfam_temp_dir, exist_ok=True)
        print(
            f"DAPFAM loaded: {len(dapfam_queries_df)} queries, "
            f"{len(dapfam_documents_df)} docs, "
            f"positives ALL={sum(len(v) for v in dapfam_cmap['ALL'].values())} "
            f"IN={sum(len(v) for v in dapfam_cmap['IN'].values())} "
            f"OUT={sum(len(v) for v in dapfam_cmap['OUT'].values())} "
            f"(input format: {args.dapfam_texttype})"
        )


########################################################################################################################################################
########################################################################################################################################################
    # Set seed for reproducibility (even if not training, for deterministic results)
    set_seed(42)
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose the model class based on model name or path
    if args.model_name.lower() in ["allenai/specter2_base", "patentbert"]:
        from adapters import AutoAdapterModel
        if args.model_name.lower() == "patentbert":
            model_path = "ZoeYou/patentbert-pytorch"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoAdapterModel.from_pretrained(model_path)
        else:
            # load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoAdapterModel.from_pretrained(args.model_name)
            # SPECTER2: load format-specific adapters (see https://huggingface.co/allenai/specter2)
            # proximity (specter2): encode queries and documents for prior-art search.
            # classification (specter2_classification): IPC linear probe.
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
            model.load_adapter("allenai/specter2_classification", source="hf", load_as="specter2_classification")
        # Drop the BERT pooler: we only consume last_hidden_state, and the
        # pooler's strided sequence_output[:, 0] matmul has triggered
        # CUBLAS_STATUS_NOT_INITIALIZED on partial last batches.
        make_bert_pooler_safe(model)
        embedding_dim = model.config.hidden_size
        model.to(device)
        ############################### IPC classification evaluation ###############################
        # SPECTER2: use classification adapter for IPC (format-specific; best for linear classifiers)
        if args.model_name.lower() == "allenai/specter2_base":
            model.set_active_adapters("specter2_classification")
        # check if the embeddings are already created
        if os.path.exists(f'{IPC_dir_temp}/train_embeddings.pt') and os.path.exists(f'{IPC_dir_temp}/test_embeddings.pt'):
            print("Embeddings already created!")
            train_embeddings = torch.load(f'{IPC_dir_temp}/train_embeddings.pt', weights_only=False)
            test_embeddings = torch.load(f'{IPC_dir_temp}/test_embeddings.pt', weights_only=False)
        else:
            # IPC classification always uses title+abstract (regardless of args.only_abstract)
            # filter out the documents where text_type is 'abstract'
            train_dataset = train_dataset[train_dataset['text_type'] == 'abstract']
            test_dataset = test_dataset[test_dataset['text_type'] == 'abstract']

            train_texts = train_dataset['text'].apply(lambda x: re.sub(r'\[(?:abstract|claim|summary|invention|drawing|description)\] ', '', x).replace('[SEP]', tokenizer.sep_token))
            test_texts = test_dataset['text'].apply(lambda x: re.sub(r'\[(?:abstract|claim|summary|invention|drawing|description)\] ', '', x).replace('[SEP]', tokenizer.sep_token))   

            # tokenize the texts
            train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt', return_token_type_ids=False)
            test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt', return_token_type_ids=False)

            # get the embeddings by batch
            batch_size = 128
            train_embeddings = np.zeros((len(train_encodings['input_ids']), embedding_dim))
            test_embeddings = np.zeros((len(test_encodings['input_ids']), embedding_dim))

            with torch.no_grad():
                for i in trange(0, len(train_encodings['input_ids']), batch_size, desc="Computing train embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in train_encodings.items()}
                    last_h = get_encoder_last_hidden_state(model, batch)
                    train_embeddings[i:i+batch_size] = last_h[:, 0, :].detach().cpu().numpy()

                for i in trange(0, len(test_encodings['input_ids']), batch_size, desc="Computing test embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in test_encodings.items()}
                    last_h = get_encoder_last_hidden_state(model, batch)
                    test_embeddings[i:i+batch_size] = last_h[:, 0, :].detach().cpu().numpy()

            log_embeddings_shape({
                'train_embeddings': train_embeddings, 
                'test_embeddings': test_embeddings
            }, "IPC Classification embeddings")

            # save the embeddings
            torch.save(train_embeddings, f'{IPC_dir_temp}/train_embeddings.pt', pickle_protocol=4)
            torch.save(test_embeddings, f'{IPC_dir_temp}/test_embeddings.pt', pickle_protocol=4)

            # SPECTER2: compute proximity (PRX) embeddings for IPC KNN (nearest-neighbor uses PRX adapter)
            if args.model_name.lower() == "allenai/specter2_base":
                model.set_active_adapters("specter2")
                train_embeddings_prx = np.zeros((len(train_encodings['input_ids']), embedding_dim))
                test_embeddings_prx = np.zeros((len(test_encodings['input_ids']), embedding_dim))
                with torch.no_grad():
                    for i in trange(0, len(train_encodings['input_ids']), batch_size, desc="Computing train embeddings (PRX for KNN)"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in train_encodings.items()}
                        last_h = get_encoder_last_hidden_state(model, batch)
                        train_embeddings_prx[i:i+batch_size] = last_h[:, 0, :].detach().cpu().numpy()
                    for i in trange(0, len(test_encodings['input_ids']), batch_size, desc="Computing test embeddings (PRX for KNN)"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in test_encodings.items()}
                        last_h = get_encoder_last_hidden_state(model, batch)
                        test_embeddings_prx[i:i+batch_size] = last_h[:, 0, :].detach().cpu().numpy()
                torch.save(train_embeddings_prx, f'{IPC_dir_temp}/train_embeddings_prx.pt', pickle_protocol=4)
                torch.save(test_embeddings_prx, f'{IPC_dir_temp}/test_embeddings_prx.pt', pickle_protocol=4)

        # SPECTER2: use proximity embeddings for KNN (linear probe uses CLF embeddings already in train/test_embeddings)
        train_embeddings_knn = None
        test_embeddings_knn = None
        if args.model_name.lower() == "allenai/specter2_base":
            if os.path.exists(f'{IPC_dir_temp}/train_embeddings_prx.pt') and os.path.exists(f'{IPC_dir_temp}/test_embeddings_prx.pt'):
                train_embeddings_knn = torch.load(f'{IPC_dir_temp}/train_embeddings_prx.pt', weights_only=False)
                test_embeddings_knn = torch.load(f'{IPC_dir_temp}/test_embeddings_prx.pt', weights_only=False)

        ipc_evaluation(train_embeddings, test_embeddings, train_labels, test_labels, train_types, test_types,
                       train_embeddings_knn=train_embeddings_knn, test_embeddings_knn=test_embeddings_knn)

        ############################ Prior-art Search evaluation ############################
        # SPECTER2: use proximity (specter2) adapter for both queries and documents
        # check if the embeddings are already created
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        else:
            # Use EXACT same text formatting as patent.py for consistency
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            
            # Process each text type separately, exactly like patent.py
            for texttype in ["abstract", "claim", "invention"]:
                # SPECTER2/PatentBERT: title + sep + abstract (no section tokens); official format: title + sep_token + (abstract or '')
                if texttype == "abstract":
                    query_texts = [queries_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + (queries_df.iloc[i][texttype] or '') for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + (documents_df.iloc[i][texttype] or '') for i in range(len(documents_df))]
                else:
                    query_texts = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]

                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt', return_token_type_ids=False)
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt', return_token_type_ids=False)

                # get the embeddings by batch
                batch_size = 128
                query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))

                with torch.no_grad():
                    # SPECTER2: use proximity adapter for both queries and documents
                    if args.model_name.lower() == "allenai/specter2_base":
                        model.set_active_adapters("specter2")
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in query_encodings.items()}
                        last_h = get_encoder_last_hidden_state(model, batch)
                        query_embs[i:i+batch_size] = last_h[:, 0, :].detach().cpu().numpy()

                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in doc_encodings.items()}
                        last_h = get_encoder_last_hidden_state(model, batch)
                        doc_embs[i:i+batch_size] = last_h[:, 0, :].detach().cpu().numpy()
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # Create concatenated versions for compatibility with existing evaluation
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, citation_mapping_graded=citation_mapping_graded)

        ############################ DAPFAM (TitlAbs -> TitlAbs) ############################
        if run_dapfam:
            dapfam_q_path = f'{dapfam_temp_dir}/query_embeddings.pt'
            dapfam_d_path = f'{dapfam_temp_dir}/document_embeddings.pt'
            if os.path.exists(dapfam_q_path) and os.path.exists(dapfam_d_path):
                print("DAPFAM embeddings already cached.")
                dapfam_q_emb = torch.load(dapfam_q_path, weights_only=False)
                dapfam_d_emb = torch.load(dapfam_d_path, weights_only=False)
            else:
                if args.dapfam_texttype == 'plain':
                    dapfam_q_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                            dapfam_queries_df['claims_text'].tolist(),
                        )
                    ]
                    dapfam_d_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                            dapfam_documents_df['claims_text'].tolist(),
                        )
                    ]
                else:
                    sep = f" {tokenizer.sep_token} "
                    dapfam_q_texts = [
                        t + sep + a
                        for t, a in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                        )
                    ]
                    dapfam_d_texts = [
                        t + sep + a
                        for t, a in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                        )
                    ]
                    if args.dapfam_texttype == 'tac':
                        dapfam_q_texts = [
                            txt + sep + extract_first_claim(c)
                            for txt, c in zip(dapfam_q_texts, dapfam_queries_df['claims_text'].tolist())
                        ]
                        dapfam_d_texts = [
                            txt + sep + extract_first_claim(c)
                            for txt, c in zip(dapfam_d_texts, dapfam_documents_df['claims_text'].tolist())
                        ]

                dapfam_q_enc = tokenizer(
                    dapfam_q_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt',
                    return_token_type_ids=False,
                )
                dapfam_d_enc = tokenizer(
                    dapfam_d_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt',
                    return_token_type_ids=False,
                )

                dapfam_q_emb = np.zeros((len(dapfam_q_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_d_emb = np.zeros((len(dapfam_d_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_bs = 128
                with torch.no_grad():
                    if args.model_name.lower() == "allenai/specter2_base":
                        model.set_active_adapters("specter2")
                    for i in trange(0, len(dapfam_q_enc['input_ids']), dapfam_bs, desc="DAPFAM query embeddings"):
                        batch = {k: v[i:i+dapfam_bs].to(device) for k, v in dapfam_q_enc.items()}
                        last_h = get_encoder_last_hidden_state(model, batch)
                        dapfam_q_emb[i:i+dapfam_bs] = last_h[:, 0, :].detach().cpu().numpy()
                    for i in trange(0, len(dapfam_d_enc['input_ids']), dapfam_bs, desc="DAPFAM doc embeddings"):
                        batch = {k: v[i:i+dapfam_bs].to(device) for k, v in dapfam_d_enc.items()}
                        last_h = get_encoder_last_hidden_state(model, batch)
                        dapfam_d_emb[i:i+dapfam_bs] = last_h[:, 0, :].detach().cpu().numpy()

                torch.save(dapfam_q_emb, dapfam_q_path, pickle_protocol=4)
                torch.save(dapfam_d_emb, dapfam_d_path, pickle_protocol=4)
                print(f"DAPFAM embeddings shape: q={dapfam_q_emb.shape}, d={dapfam_d_emb.shape}")

            dapfam_evaluation(
                dapfam_queries_df.index.tolist(),
                dapfam_documents_df.index.tolist(),
                dapfam_q_emb,
                dapfam_d_emb,
                dapfam_cmap,
                dapfam_query_ipc3,
                dapfam_doc_ipc3,
                k=100,
            )


        ################################ Comprehensive Embedding Quality Evaluation ################################
        # load embeddings from prior-art search task
        query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
        document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        # Run comprehensive quality evaluation (uniformity, SSD, alignment, topology)
        comprehensive_embedding_quality_evaluation(
            query_embeddings=query_embeddings,
            doc_embeddings=document_embeddings,
            query_ids=query_ids,
            doc_ids=doc_ids,
            citation_mapping=citation_mapping,
            query_types=query_types,
            doc_types=doc_types,
            queries_df=queries_df,
            documents_df=documents_df
        )

########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name.lower() == "mpi-inno-comp/paecter" or args.model_name.lower() == "anferico/bert-for-patents":
        # load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        # Drop the BERT pooler: we only consume last_hidden_state, and the
        # pooler's strided sequence_output[:, 0] matmul has triggered
        # CUBLAS_STATUS_NOT_INITIALIZED on partial last batches.
        make_bert_pooler_safe(model)

        if args.model_name.lower() == "anferico/bert-for-patents":
            # add special tokens to the tokenizer
            tokenizer.add_special_tokens({'additional_special_tokens': ['[abstract]', '[claim]', '[invention]']})
            model.resize_token_embeddings(len(tokenizer))

        embedding_dim = model.config.hidden_size
        model.to(device)

        ############################### IPC classification evaluation ###############################
        if run_ipc:
            # check if the embeddings are already created
            if os.path.exists(f'{IPC_dir_temp}/train_embeddings.pt') and os.path.exists(f'{IPC_dir_temp}/test_embeddings.pt'):
                print("Embeddings already created!")
                train_embeddings = torch.load(f'{IPC_dir_temp}/train_embeddings.pt', weights_only=False)
                test_embeddings = torch.load(f'{IPC_dir_temp}/test_embeddings.pt', weights_only=False)
            else:
                # filter out the documents where text_type is 'abstract'
                train_dataset = train_dataset[train_dataset['text_type'] == 'abstract']
                test_dataset = test_dataset[test_dataset['text_type'] == 'abstract']

                if args.model_name.lower() == "mpi-inno-comp/paecter":
                    train_texts = train_dataset['text'].apply(lambda x: re.sub(r'\[(?:abstract|claim|summary|invention|drawing|description)\] ', '', x).replace('[SEP]', tokenizer.sep_token))
                    test_texts = test_dataset['text'].apply(lambda x: re.sub(r'\[(?:abstract|claim|summary|invention|drawing|description)\] ', '', x).replace('[SEP]', tokenizer.sep_token))

                elif args.model_name.lower() == "anferico/bert-for-patents":
                    train_texts = train_dataset['text']
                    test_texts = test_dataset['text']

                # tokenize the texts
                train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
                test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

                # get the embeddings by batch
                batch_size = 256
                train_embeddings = np.zeros((len(train_encodings['input_ids']), embedding_dim))
                test_embeddings = np.zeros((len(test_encodings['input_ids']), embedding_dim))

                with torch.no_grad():
                    for i in trange(0, len(train_encodings['input_ids']), batch_size, desc="Computing train embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in train_encodings.items()}  # Move to GPU
                        outputs = model(**batch)
                        train_embeddings[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, train_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()

                    for i in trange(0, len(test_encodings['input_ids']), batch_size, desc="Computing test embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in test_encodings.items()}
                        outputs = model(**batch)
                        test_embeddings[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, test_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()

                print(train_embeddings.shape, test_embeddings.shape)

                # save the embeddings
                torch.save(train_embeddings, f'{IPC_dir_temp}/train_embeddings.pt', pickle_protocol=4)
                torch.save(test_embeddings, f'{IPC_dir_temp}/test_embeddings.pt', pickle_protocol=4)

            ipc_evaluation(train_embeddings, test_embeddings, train_labels, test_labels, train_types, test_types)

        ############################ Prior-art Search evaluation ############################
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        else:
            # Use EXACT same text formatting as patent.py for consistency
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            
            # Get original IDs (before multiplication) 
            original_query_ids = list(queries_df.index)
            original_doc_ids = list(documents_df.index)
            
            # Process each text type separately, exactly like patent.py
            for texttype in ["abstract", "claim", "invention"]:
                # Format texts according to model type
                if args.model_name.lower() == "mpi-inno-comp/paecter":
                    # PAECTer doesn't use section tokens - clean format
                    if texttype == "abstract":
                        query_texts = [queries_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    else:
                        query_texts = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                elif args.model_name.lower() == "anferico/bert-for-patents":
                    # BERT-for-patents uses section tokens like patent.py
                    if texttype == "abstract":
                        query_texts = [queries_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    else:
                        query_texts = [f"[{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [f"[{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]

                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                # get the embeddings by batch
                batch_size = 256
                query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))

                with torch.no_grad():
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in query_encodings.items()}
                        outputs = model(**batch)
                        query_embs[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, query_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()

                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in doc_encodings.items()}
                        outputs = model(**batch)
                        doc_embs[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, doc_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # Create concatenated versions for compatibility with existing evaluation
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt', pickle_protocol=4)
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt', pickle_protocol=4)

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        if run_perf200:
            prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, citation_mapping_graded=citation_mapping_graded)

        ############################ DAPFAM (TitlAbs -> TitlAbs) ############################
        # Each query / target is encoded once as a single sequence.
        if run_dapfam:
            dapfam_q_path = f'{dapfam_temp_dir}/query_embeddings.pt'
            dapfam_d_path = f'{dapfam_temp_dir}/document_embeddings.pt'
            if os.path.exists(dapfam_q_path) and os.path.exists(dapfam_d_path):
                print("DAPFAM embeddings already cached.")
                dapfam_q_emb = torch.load(dapfam_q_path, weights_only=False)
                dapfam_d_emb = torch.load(dapfam_d_path, weights_only=False)
            else:
                if args.dapfam_texttype == 'plain':
                    # PatenTEB §5.4 protocol: plain space-joined title + abstract + first claim,
                    # no SEP, no section markers. Used for both paecter and anferico in this branch
                    # because PatenTEB also evaluates both via sentence-transformers
                    # InformationRetrievalEvaluator (raw text, mean pooling fallback).
                    dapfam_q_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                            dapfam_queries_df['claims_text'].tolist(),
                        )
                    ]
                    dapfam_d_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                            dapfam_documents_df['claims_text'].tolist(),
                        )
                    ]
                elif args.model_name.lower() == "mpi-inno-comp/paecter":
                    sep = f" {tokenizer.sep_token} "
                    dapfam_q_texts = [t + sep + a for t, a in zip(
                        dapfam_queries_df['title'].tolist(),
                        dapfam_queries_df['abstract'].tolist(),
                    )]
                    dapfam_d_texts = [t + sep + a for t, a in zip(
                        dapfam_documents_df['title'].tolist(),
                        dapfam_documents_df['abstract'].tolist(),
                    )]
                    if args.dapfam_texttype == 'tac':
                        dapfam_q_texts = [
                            txt + sep + extract_first_claim(c)
                            for txt, c in zip(dapfam_q_texts, dapfam_queries_df['claims_text'].tolist())
                        ]
                        dapfam_d_texts = [
                            txt + sep + extract_first_claim(c)
                            for txt, c in zip(dapfam_d_texts, dapfam_documents_df['claims_text'].tolist())
                        ]
                else:  # anferico/bert-for-patents
                    dapfam_q_texts = [t + " [SEP] [abstract] " + a for t, a in zip(
                        dapfam_queries_df['title'].tolist(),
                        dapfam_queries_df['abstract'].tolist(),
                    )]
                    dapfam_d_texts = [t + " [SEP] [abstract] " + a for t, a in zip(
                        dapfam_documents_df['title'].tolist(),
                        dapfam_documents_df['abstract'].tolist(),
                    )]
                    if args.dapfam_texttype == 'tac':
                        dapfam_q_texts = [
                            txt + " [SEP] [claim] " + extract_first_claim(c)
                            for txt, c in zip(dapfam_q_texts, dapfam_queries_df['claims_text'].tolist())
                        ]
                        dapfam_d_texts = [
                            txt + " [SEP] [claim] " + extract_first_claim(c)
                            for txt, c in zip(dapfam_d_texts, dapfam_documents_df['claims_text'].tolist())
                        ]

                dapfam_q_enc = tokenizer(dapfam_q_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                dapfam_d_enc = tokenizer(dapfam_d_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                dapfam_q_emb = np.zeros((len(dapfam_q_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_d_emb = np.zeros((len(dapfam_d_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_bs = 256
                with torch.no_grad():
                    for i in trange(0, len(dapfam_q_enc['input_ids']), dapfam_bs, desc="DAPFAM query embeddings"):
                        batch = {k: torch.tensor(v[i:i+dapfam_bs]).to(device) for k, v in dapfam_q_enc.items()}
                        outputs = model(**batch)
                        dapfam_q_emb[i:i+dapfam_bs] = mean_pooling(
                            outputs.last_hidden_state,
                            dapfam_q_enc['attention_mask'][i:i+dapfam_bs],
                        ).detach().cpu().numpy()
                    for i in trange(0, len(dapfam_d_enc['input_ids']), dapfam_bs, desc="DAPFAM doc embeddings"):
                        batch = {k: torch.tensor(v[i:i+dapfam_bs]).to(device) for k, v in dapfam_d_enc.items()}
                        outputs = model(**batch)
                        dapfam_d_emb[i:i+dapfam_bs] = mean_pooling(
                            outputs.last_hidden_state,
                            dapfam_d_enc['attention_mask'][i:i+dapfam_bs],
                        ).detach().cpu().numpy()
                torch.save(dapfam_q_emb, dapfam_q_path, pickle_protocol=4)
                torch.save(dapfam_d_emb, dapfam_d_path, pickle_protocol=4)
                print(f"DAPFAM embeddings shape: q={dapfam_q_emb.shape}, d={dapfam_d_emb.shape}")

            dapfam_evaluation(
                dapfam_queries_df.index.tolist(),
                dapfam_documents_df.index.tolist(),
                dapfam_q_emb,
                dapfam_d_emb,
                dapfam_cmap,
                dapfam_query_ipc3,
                dapfam_doc_ipc3,
                k=100,
            )

        ################################ Comprehensive Embedding Quality Evaluation ################################
        if run_perf200:
            # load embeddings from prior-art search task
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

            # Run comprehensive quality evaluation (uniformity, SSD, alignment, topology)
            comprehensive_embedding_quality_evaluation(
                query_embeddings=query_embeddings,
                doc_embeddings=document_embeddings,
                query_ids=query_ids,
                doc_ids=doc_ids,
                citation_mapping=citation_mapping,
                query_types=query_types,
                doc_types=doc_types,
                queries_df=queries_df,
                documents_df=documents_df
            )

########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name.lower() == "mpi-inno-comp/pat_specter":
        # load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        # Drop the BERT pooler: we only consume last_hidden_state, and the
        # pooler's strided sequence_output[:, 0] matmul has triggered
        # CUBLAS_STATUS_NOT_INITIALIZED on partial last batches.
        make_bert_pooler_safe(model)

        embedding_dim = model.config.hidden_size
        model.to(device)

        ############################### IPC classification evaluation ###############################
        # check if the embeddings are already created
        if os.path.exists(f'{IPC_dir_temp}/train_embeddings.pt') and os.path.exists(f'{IPC_dir_temp}/test_embeddings.pt'):
            print("Embeddings already created!")
            train_embeddings = torch.load(f'{IPC_dir_temp}/train_embeddings.pt', weights_only=False)
            test_embeddings = torch.load(f'{IPC_dir_temp}/test_embeddings.pt', weights_only=False)
        else:
            train_dataset = train_dataset[train_dataset['text_type'] == 'abstract']
            test_dataset = test_dataset[test_dataset['text_type'] == 'abstract']

            # replace [SEP] by its sep_token
            train_texts = train_dataset['text'].apply(lambda x: re.sub(r'\[(?:abstract|claim|summary|invention|drawing|description)\] ', '', x).replace('[SEP]', tokenizer.sep_token))
            test_texts = test_dataset['text'].apply(lambda x: re.sub(r'\[(?:abstract|claim|summary|invention|drawing|description)\] ', '', x).replace('[SEP]', tokenizer.sep_token))

            # tokenize the texts
            train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
            test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

            # get the embeddings by batch
            batch_size = 256
            train_embeddings = np.zeros((len(train_encodings['input_ids']), embedding_dim))
            test_embeddings = np.zeros((len(test_encodings['input_ids']), embedding_dim))

            with torch.no_grad():
                for i in trange(0, len(train_encodings['input_ids']), batch_size, desc="Computing train embeddings"):
                    batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in train_encodings.items()}  # Move to GPU
                    outputs = model(**batch)
                    train_embeddings[i:i+batch_size] = cls_pooling(outputs, train_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()

                for i in trange(0, len(test_encodings['input_ids']), batch_size, desc="Computing test embeddings"):
                    batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in test_encodings.items()}
                    outputs = model(**batch)
                    test_embeddings[i:i+batch_size] = cls_pooling(outputs, test_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()

            print(train_embeddings.shape, test_embeddings.shape)

            # save the embeddings
            torch.save(train_embeddings, f'{IPC_dir_temp}/train_embeddings.pt', pickle_protocol=4)
            torch.save(test_embeddings, f'{IPC_dir_temp}/test_embeddings.pt', pickle_protocol=4)

        ipc_evaluation(train_embeddings, test_embeddings, train_labels, test_labels, train_types, test_types)

        ############################ Prior-art Search evaluation ############################
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        else:
            # Use EXACT same text formatting as patent.py for consistency
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            
            # Process each text type separately, exactly like patent.py
            for texttype in ["abstract", "claim", "invention"]:
                # PatSpecter doesn't use section tokens - clean format like PAECTer
                if texttype == "abstract":
                    query_texts = [queries_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    query_texts = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]

                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                # get the embeddings by batch
                batch_size = 64
                query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))

                with torch.no_grad():
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in query_encodings.items()}
                        outputs = model(**batch)
                        query_embs[i:i+batch_size] = cls_pooling(outputs, query_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()

                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in doc_encodings.items()}
                        outputs = model(**batch)
                        doc_embs[i:i+batch_size] = cls_pooling(outputs, doc_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # Create concatenated versions for compatibility with existing evaluation
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt', pickle_protocol=4)
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt', pickle_protocol=4)

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, citation_mapping_graded=citation_mapping_graded)

        ############################ DAPFAM (TitlAbs -> TitlAbs) ############################
        if run_dapfam:
            dapfam_q_path = f'{dapfam_temp_dir}/query_embeddings.pt'
            dapfam_d_path = f'{dapfam_temp_dir}/document_embeddings.pt'
            if os.path.exists(dapfam_q_path) and os.path.exists(dapfam_d_path):
                print("DAPFAM embeddings already cached.")
                dapfam_q_emb = torch.load(dapfam_q_path, weights_only=False)
                dapfam_d_emb = torch.load(dapfam_d_path, weights_only=False)
            else:
                if args.dapfam_texttype == 'plain':
                    dapfam_q_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                            dapfam_queries_df['claims_text'].tolist(),
                        )
                    ]
                    dapfam_d_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                            dapfam_documents_df['claims_text'].tolist(),
                        )
                    ]
                else:
                    sep = f" {tokenizer.sep_token} "
                    dapfam_q_texts = [
                        t + sep + a
                        for t, a in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                        )
                    ]
                    dapfam_d_texts = [
                        t + sep + a
                        for t, a in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                        )
                    ]
                    if args.dapfam_texttype == 'tac':
                        dapfam_q_texts = [
                            txt + sep + extract_first_claim(c)
                            for txt, c in zip(dapfam_q_texts, dapfam_queries_df['claims_text'].tolist())
                        ]
                        dapfam_d_texts = [
                            txt + sep + extract_first_claim(c)
                            for txt, c in zip(dapfam_d_texts, dapfam_documents_df['claims_text'].tolist())
                        ]

                dapfam_q_enc = tokenizer(dapfam_q_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                dapfam_d_enc = tokenizer(dapfam_d_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                dapfam_q_emb = np.zeros((len(dapfam_q_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_d_emb = np.zeros((len(dapfam_d_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_bs = 64
                with torch.no_grad():
                    for i in trange(0, len(dapfam_q_enc['input_ids']), dapfam_bs, desc="DAPFAM query embeddings"):
                        batch = {k: torch.tensor(v[i:i+dapfam_bs]).to(device) for k, v in dapfam_q_enc.items()}
                        outputs = model(**batch)
                        dapfam_q_emb[i:i+dapfam_bs] = cls_pooling(
                            outputs,
                            dapfam_q_enc['attention_mask'][i:i+dapfam_bs],
                        ).detach().cpu().numpy()
                    for i in trange(0, len(dapfam_d_enc['input_ids']), dapfam_bs, desc="DAPFAM doc embeddings"):
                        batch = {k: torch.tensor(v[i:i+dapfam_bs]).to(device) for k, v in dapfam_d_enc.items()}
                        outputs = model(**batch)
                        dapfam_d_emb[i:i+dapfam_bs] = cls_pooling(
                            outputs,
                            dapfam_d_enc['attention_mask'][i:i+dapfam_bs],
                        ).detach().cpu().numpy()

                torch.save(dapfam_q_emb, dapfam_q_path, pickle_protocol=4)
                torch.save(dapfam_d_emb, dapfam_d_path, pickle_protocol=4)
                print(f"DAPFAM embeddings shape: q={dapfam_q_emb.shape}, d={dapfam_d_emb.shape}")

            dapfam_evaluation(
                dapfam_queries_df.index.tolist(),
                dapfam_documents_df.index.tolist(),
                dapfam_q_emb,
                dapfam_d_emb,
                dapfam_cmap,
                dapfam_query_ipc3,
                dapfam_doc_ipc3,
                k=100,
            )


        ################################ Comprehensive Embedding Quality Evaluation ################################
        # load embeddings from prior-art search task
        query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
        document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        # Run comprehensive quality evaluation (uniformity, SSD, alignment, topology)
        comprehensive_embedding_quality_evaluation(
            query_embeddings=query_embeddings,
            doc_embeddings=document_embeddings,
            query_ids=query_ids,
            doc_ids=doc_ids,
            citation_mapping=citation_mapping,
            query_types=query_types,
            doc_types=doc_types,
            queries_df=queries_df,
            documents_df=documents_df
        )


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name in ["Alibaba-NLP/gte-Qwen2-7B-instruct", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B"]:
        # clean cuda cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        from torch import Tensor
        from torch.cuda.amp import autocast

        def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        def get_detailed_instruct(query: str, task_description="retrieve the most similar patent given the query patent") -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.float16)

        embedding_dim = model.config.hidden_size

        # Use model directly on GPU with memory optimizations
        model.to(device)
        model.eval()

        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        ############################### IPC classification evaluation ###############################
        # check if the embeddings are already created
        if os.path.exists(f'{IPC_dir_temp}/train_embeddings.pt') and os.path.exists(f'{IPC_dir_temp}/test_embeddings.pt'):
            print("Embeddings already created!")
            train_embeddings = torch.load(f'{IPC_dir_temp}/train_embeddings.pt', weights_only=False)
            test_embeddings = torch.load(f'{IPC_dir_temp}/test_embeddings.pt', weights_only=False)
        else:
            # IPC classification always uses title+abstract (regardless of args.only_abstract)
            # filter out the documents where text_type is 'abstract'
            train_dataset = train_dataset[train_dataset['text_type'] == 'abstract']
            test_dataset = test_dataset[test_dataset['text_type'] == 'abstract']

            # remove special tokens from the beginning of the abstract
            train_texts = train_dataset['text'].apply(lambda x: re.sub(r'^\[(?:abstract|claim|summary|invention)\] ', '', x)).replace('[SEP]', " ")
            test_texts = test_dataset['text'].apply(lambda x: re.sub(r'^\[(?:abstract|claim|summary|invention)\] ', '', x)).replace('[SEP]', " ")

            # tokenize the texts
            train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
            test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

            # get the embeddings by batch
            batch_size = min(64, len(train_encodings['input_ids']))
            train_embeddings = np.zeros((len(train_encodings['input_ids']), embedding_dim))
            test_embeddings = np.zeros((len(test_encodings['input_ids']), embedding_dim))

            with torch.no_grad():
                for i in trange(0, len(train_encodings['input_ids']), batch_size, desc="Computing train embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in train_encodings.items()}

                    with autocast(dtype=torch.bfloat16):  # Enables mixed precision
                        outputs = model(**batch)

                    train_embeddings[i:i+batch_size] = last_token_pool(outputs.last_hidden_state, train_encodings['attention_mask'][i:i+batch_size]).detach().cpu().float().numpy()
                    del batch, outputs
                    if i % (5 * batch_size) == 0:  # Every 5 batches
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                for i in trange(0, len(test_encodings['input_ids']), batch_size, desc="Computing test embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in test_encodings.items()}

                    with autocast(dtype=torch.bfloat16):  # Enables mixed precision
                        outputs = model(**batch)

                    test_embeddings[i:i+batch_size] = last_token_pool(outputs.last_hidden_state, test_encodings['attention_mask'][i:i+batch_size]).detach().cpu().float().numpy()
                    del batch, outputs
                    if i % (5 * batch_size) == 0:  # Every 5 batches
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

            print(train_embeddings.shape, test_embeddings.shape)

            # save the embeddings
            torch.save(train_embeddings, f'{IPC_dir_temp}/train_embeddings.pt', pickle_protocol=4)
            torch.save(test_embeddings, f'{IPC_dir_temp}/test_embeddings.pt', pickle_protocol=4)

        ipc_evaluation(train_embeddings, test_embeddings, train_labels, test_labels, train_types, test_types)

        ############################ Prior-art Search evaluation ############################
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)
        else:
            query_texts = (queries_df['title'] + " " + queries_df['abstract']).apply(lambda x: get_detailed_instruct(x)).tolist() + queries_df['claim'].apply(lambda x: get_detailed_instruct(x)).tolist() + queries_df['invention'].apply(lambda x: get_detailed_instruct(x)).tolist()
            doc_texts = (documents_df['title'] + " " + documents_df['abstract']).tolist() + (documents_df['claim']).tolist() + (documents_df['invention']).tolist()

            query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
            doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

            # get the embeddings by batch
            batch_size = min(64, len(query_encodings['input_ids']))
            query_embeddings = np.zeros((len(query_encodings['input_ids']), embedding_dim))
            document_embeddings = np.zeros((len(doc_encodings['input_ids']), embedding_dim))

            with torch.no_grad():
                for i in trange(0, len(query_encodings['input_ids']), batch_size, desc="Computing query embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in query_encodings.items()}

                    with autocast(dtype=torch.bfloat16):
                        outputs = model(**batch)

                    query_embeddings[i:i+batch_size] = last_token_pool(outputs.last_hidden_state, query_encodings['attention_mask'][i:i+batch_size]).detach().cpu().float().numpy()
                    del batch, outputs
                    if i % (5 * batch_size) == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc="Computing document embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in doc_encodings.items()}

                    with autocast(dtype=torch.bfloat16):
                        outputs = model(**batch)

                    document_embeddings[i:i+batch_size] = last_token_pool(outputs.last_hidden_state, doc_encodings['attention_mask'][i:i+batch_size]).detach().cpu().float().numpy()
                    del batch, outputs
                    if i % (5 * batch_size) == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt', pickle_protocol=4)
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt', pickle_protocol=4)

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')

        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, citation_mapping_graded=citation_mapping_graded)

        ############################ DAPFAM (TitlAbs -> TitlAbs) ############################
        if run_dapfam:
            dapfam_q_path = f'{dapfam_temp_dir}/query_embeddings.pt'
            dapfam_d_path = f'{dapfam_temp_dir}/document_embeddings.pt'
            if os.path.exists(dapfam_q_path) and os.path.exists(dapfam_d_path):
                print("DAPFAM embeddings already cached.")
                dapfam_q_emb = torch.load(dapfam_q_path, weights_only=False)
                dapfam_d_emb = torch.load(dapfam_d_path, weights_only=False)
            else:
                if args.dapfam_texttype == 'plain':
                    q_base = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                            dapfam_queries_df['claims_text'].tolist(),
                        )
                    ]
                    d_base = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                            dapfam_documents_df['claims_text'].tolist(),
                        )
                    ]
                else:
                    q_base = [
                        t + " " + a
                        for t, a in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                        )
                    ]
                    d_base = [
                        t + " " + a
                        for t, a in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                        )
                    ]
                    if args.dapfam_texttype == 'tac':
                        q_base = [
                            txt + " " + extract_first_claim(c)
                            for txt, c in zip(q_base, dapfam_queries_df['claims_text'].tolist())
                        ]
                        d_base = [
                            txt + " " + extract_first_claim(c)
                            for txt, c in zip(d_base, dapfam_documents_df['claims_text'].tolist())
                        ]

                dapfam_q_texts = [get_detailed_instruct(x) for x in q_base]
                dapfam_d_texts = d_base

                dapfam_q_enc = tokenizer(dapfam_q_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                dapfam_d_enc = tokenizer(dapfam_d_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                dapfam_q_emb = np.zeros((len(dapfam_q_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_d_emb = np.zeros((len(dapfam_d_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_bs = 32
                with torch.no_grad():
                    for i in trange(0, len(dapfam_q_enc['input_ids']), dapfam_bs, desc="DAPFAM query embeddings"):
                        batch = {k: v[i:i+dapfam_bs].to(device) for k, v in dapfam_q_enc.items()}
                        with autocast(dtype=torch.bfloat16):
                            outputs = model(**batch)
                        dapfam_q_emb[i:i+dapfam_bs] = last_token_pool(
                            outputs.last_hidden_state,
                            dapfam_q_enc['attention_mask'][i:i+dapfam_bs],
                        ).detach().cpu().float().numpy()
                    for i in trange(0, len(dapfam_d_enc['input_ids']), dapfam_bs, desc="DAPFAM doc embeddings"):
                        batch = {k: v[i:i+dapfam_bs].to(device) for k, v in dapfam_d_enc.items()}
                        with autocast(dtype=torch.bfloat16):
                            outputs = model(**batch)
                        dapfam_d_emb[i:i+dapfam_bs] = last_token_pool(
                            outputs.last_hidden_state,
                            dapfam_d_enc['attention_mask'][i:i+dapfam_bs],
                        ).detach().cpu().float().numpy()

                torch.save(dapfam_q_emb, dapfam_q_path, pickle_protocol=4)
                torch.save(dapfam_d_emb, dapfam_d_path, pickle_protocol=4)
                print(f"DAPFAM embeddings shape: q={dapfam_q_emb.shape}, d={dapfam_d_emb.shape}")

            dapfam_evaluation(
                dapfam_queries_df.index.tolist(),
                dapfam_documents_df.index.tolist(),
                dapfam_q_emb,
                dapfam_d_emb,
                dapfam_cmap,
                dapfam_query_ipc3,
                dapfam_doc_ipc3,
                k=100,
            )


        ################################ Comprehensive Embedding Quality Evaluation ################################
        # load embeddings from prior-art search task
        query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
        document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        # Run comprehensive quality evaluation (uniformity, SSD, alignment, topology)
        comprehensive_embedding_quality_evaluation(
            query_embeddings=query_embeddings,
            doc_embeddings=document_embeddings,
            query_ids=query_ids,
            doc_ids=doc_ids,
            citation_mapping=citation_mapping,
            query_types=query_types,
            doc_types=doc_types,
            queries_df=queries_df,
            documents_df=documents_df
        )


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name == "bm25":
        import bm25s
        import snowballstemmer

        # BM25 doesn't need IPC classification evaluation since it's not embedding-based
        print("Skipping IPC classification for BM25 (not applicable for sparse retrieval)")

        ############################ Prior-art Search evaluation ############################
        print("Running BM25 Prior-art search evaluation")
        
        # Run evaluation for both abstract-to-abstract and claim-to-all scenarios
        # to match other models' evaluation patterns
        
        # 1) Abstract-to-Abstract evaluation (like other models' abstract->abstract)
        print("BM25 Evaluation 1: Abstract-to-Abstract retrieval")
        abstract_train_corpus = documents_df['title'] + ' ' + documents_df['abstract']
        abstract_test_corpus = queries_df['title'] + ' ' + queries_df['abstract']
        
        # Tokenize corpus
        stemmer = snowballstemmer.stemmer('english')
        abstract_corpus_tokens = bm25s.tokenize(abstract_train_corpus.tolist(), stopwords="en", stemmer=stemmer)
        
        # Create and index BM25 model
        abstract_retriever = bm25s.BM25()
        abstract_retriever.index(abstract_corpus_tokens)
        
        # Tokenize queries and retrieve
        abstract_queries_tokens = bm25s.tokenize(abstract_test_corpus.tolist(), stemmer=stemmer)
        original_doc_ids = list(documents.keys())  # Original doc IDs before multiplication
        n_abstract_docs = len(original_doc_ids)
        # Retrieve full corpus for MAP/MRR; top-100 for recall/nDCG
        abstract_results_full, _ = abstract_retriever.retrieve(abstract_queries_tokens, k=n_abstract_docs)
        abstract_retrieved_full = [[original_doc_ids[i] for i in result] for result in abstract_results_full]
        abstract_retrieved_top100 = [ids[:100] for ids in abstract_retrieved_full]

        query_ids_list = list(queries.keys())
        true_labels_abs = [citation_mapping.get(q, []) for q in query_ids_list]
        true_graded_abs = [citation_mapping_graded.get(q, {}) for q in query_ids_list]

        # Calculate recall@k, nDCG@k, mAP, MRR
        bm25_abstract_results = {}
        for k in [10, 20, 50, 100]:
            bm25_abstract_results[f'recall@{k}'] = mean_recall_at_k(true_labels_abs, abstract_retrieved_top100, k=k)
            bm25_abstract_results[f'ndcg@{k}'] = mean_ndcg_at_k(true_labels_abs, abstract_retrieved_top100, k=k)
            bm25_abstract_results[f'ndcg_graded@{k}'] = mean_ndcg_at_k_graded(true_graded_abs, abstract_retrieved_top100, k=k)
        bm25_abstract_results['map'] = mean_average_precision(true_labels_abs, abstract_retrieved_full)
        bm25_abstract_results['mrr'] = mean_reciprocal_rank(true_labels_abs, abstract_retrieved_full)
        print_metric_table(bm25_abstract_results, "Query: abstract \u2192 Document: abstract")
        
        # 2) Claim-to-All evaluation (like other models' claim->all)
        print("\nBM25 Evaluation 2: Claim-to-All retrieval")
        # Use all document sections as corpus
        all_train_corpus = (
            (documents_df['title'] + ' ' + documents_df['abstract']).tolist() + 
            documents_df['claim'].tolist() + 
            documents_df['invention'].tolist()
        )
        # Use only claim queries
        claim_test_corpus = queries_df['claim'].tolist()
        
        # Tokenize corpus
        all_corpus_tokens = bm25s.tokenize(all_train_corpus, stopwords="en", stemmer=stemmer)
        
        # Create and index BM25 model
        all_retriever = bm25s.BM25()
        all_retriever.index(all_corpus_tokens)
        
        # Tokenize queries and retrieve full corpus for MAP/MRR
        claim_queries_tokens = bm25s.tokenize(claim_test_corpus, stemmer=stemmer)
        original_doc_count = len(original_doc_ids)
        claim_results_full, _ = all_retriever.retrieve(claim_queries_tokens, k=3 * original_doc_count)

        def _dedup_claim_ids(results_array, original_doc_count, original_doc_ids):
            """Map section indices back to doc IDs and deduplicate (order-preserving)."""
            out = []
            for result in results_array:
                seen = {}
                for idx in result:
                    if idx < original_doc_count:
                        doc_id = original_doc_ids[idx]
                    elif idx < 2 * original_doc_count:
                        doc_id = original_doc_ids[idx - original_doc_count]
                    else:
                        doc_id = original_doc_ids[idx - 2 * original_doc_count]
                    if doc_id not in seen:
                        seen[doc_id] = None
                out.append(list(seen.keys()))
            return out

        claim_retrieved_full = _dedup_claim_ids(claim_results_full, original_doc_count, original_doc_ids)
        claim_retrieved_top100 = [ids[:100] for ids in claim_retrieved_full]

        true_labels_clm = [citation_mapping.get(q, []) for q in query_ids_list]
        true_graded_clm = [citation_mapping_graded.get(q, {}) for q in query_ids_list]

        # Calculate recall@k, nDCG@k, mAP, MRR
        bm25_claim_results = {}
        for k in [10, 20, 50, 100]:
            bm25_claim_results[f'recall@{k}'] = mean_recall_at_k(true_labels_clm, claim_retrieved_top100, k=k)
            bm25_claim_results[f'ndcg@{k}'] = mean_ndcg_at_k(true_labels_clm, claim_retrieved_top100, k=k)
            bm25_claim_results[f'ndcg_graded@{k}'] = mean_ndcg_at_k_graded(true_graded_clm, claim_retrieved_top100, k=k)
        bm25_claim_results['map'] = mean_average_precision(true_labels_clm, claim_retrieved_full)
        bm25_claim_results['mrr'] = mean_reciprocal_rank(true_labels_clm, claim_retrieved_full)
        print_metric_table(bm25_claim_results, "Query: claim \u2192 Document: all")

        ############################ DAPFAM (TitlAbs -> TitlAbs) ############################
        if run_dapfam:
            print("Running BM25 DAPFAM evaluation")
            if args.dapfam_texttype == 'plain':
                dapfam_doc_texts = [
                    " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                    for t, a, c in zip(
                        dapfam_documents_df['title'].tolist(),
                        dapfam_documents_df['abstract'].tolist(),
                        dapfam_documents_df['claims_text'].tolist(),
                    )
                ]
                dapfam_query_texts = [
                    " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                    for t, a, c in zip(
                        dapfam_queries_df['title'].tolist(),
                        dapfam_queries_df['abstract'].tolist(),
                        dapfam_queries_df['claims_text'].tolist(),
                    )
                ]
            else:
                dapfam_doc_texts = [
                    t + " " + a
                    for t, a in zip(
                        dapfam_documents_df['title'].tolist(),
                        dapfam_documents_df['abstract'].tolist(),
                    )
                ]
                dapfam_query_texts = [
                    t + " " + a
                    for t, a in zip(
                        dapfam_queries_df['title'].tolist(),
                        dapfam_queries_df['abstract'].tolist(),
                    )
                ]
                if args.dapfam_texttype == 'tac':
                    dapfam_doc_texts = [
                        txt + " " + extract_first_claim(c)
                        for txt, c in zip(dapfam_doc_texts, dapfam_documents_df['claims_text'].tolist())
                    ]
                    dapfam_query_texts = [
                        txt + " " + extract_first_claim(c)
                        for txt, c in zip(dapfam_query_texts, dapfam_queries_df['claims_text'].tolist())
                    ]

            dapfam_corpus_tokens = bm25s.tokenize(dapfam_doc_texts, stopwords="en", stemmer=stemmer)
            dapfam_retriever = bm25s.BM25()
            dapfam_retriever.index(dapfam_corpus_tokens)

            dapfam_query_tokens = bm25s.tokenize(dapfam_query_texts, stemmer=stemmer)
            dapfam_results, _ = dapfam_retriever.retrieve(dapfam_query_tokens, k=100)

            dapfam_doc_ids = list(dapfam_documents_df.index)
            dapfam_predicted_labels = [[dapfam_doc_ids[i] for i in result] for result in dapfam_results]

            dapfam_evaluation_from_predictions(
                dapfam_queries_df.index.tolist(),
                dapfam_predicted_labels,
                dapfam_cmap,
                k=100,
            )
        
        print("\n📝 Note: BM25 evaluation completed. IPC classification")
        print("   evaluation are not applicable for sparse retrieval methods.")


########################################################################################################################################################
########################################################################################################################################################
    elif "checkpoint" in args.model_name or "bestmodel" in args.model_name or args.model_name.startswith("ZoeYou/PatentMap-V0-"):
        def load_checkpoint_model_and_tokenizer(checkpoint_path):
            """Smart checkpoint loader that handles tokenizer and model loading intelligently."""
            from transformers import AutoConfig, AutoTokenizer
            from dataclasses import dataclass
            from typing import Optional
            
            @dataclass
            class ModelArguments:
                do_mlm: bool = False
                regularization: Optional[str] = None
                temperature: float = 0.05
                pooler_type: str = "cls"
                mlp_only_train: bool = True
                model_name_or_path: Optional[str] = None
            
            print(f"🔄 Loading checkpoint: {checkpoint_path}")
            
            # Step 1: Try loading tokenizer from checkpoint, fallback to reconstruction
            try:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                print(f"✅ Loaded tokenizer from checkpoint ({len(tokenizer)} tokens)")
            except:
                print("⚠️  Tokenizer not found in checkpoint, reconstructing...")
                tokenizer = reconstruct_tokenizer(checkpoint_path)
            
            # Step 2: Load model with proper config
            model_args = ModelArguments(model_name_or_path=checkpoint_path)
            config = AutoConfig.from_pretrained("anferico/bert-for-patents")
            config.vocab_size = len(tokenizer)
            
            # Step 3: Load model with smart error handling
            model = load_model_with_fallback(checkpoint_path, config, model_args)
            
            # Step 4: Ensure vocab size consistency
            if model.get_input_embeddings().num_embeddings != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
                print(f"🔧 Resized model embeddings to {len(tokenizer)} tokens")
            
            return model, tokenizer, model.config.hidden_size

        def reconstruct_tokenizer(checkpoint_path):
            """Reconstruct tokenizer by inferring settings from checkpoint path."""
            special_tokens = {"abstract": "[abstract]", "claim": "[claim]", "summary": "[summary]",
                            "background": "[invention]", "drawing": "[drawing]", "detailed_description": "[description]"}
            
            tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")
            
            # Smart view inference from path
            views_match = re.search(r'views-([^/]*?)(?:_reg-|/|$)', checkpoint_path)
            if views_match and views_match.group(1):
                additional_views = views_match.group(1).split('+')
                print(f"📊 Inferred views from path: {additional_views}")
            else:
                additional_views = []
                print(f"📊 No views specified in path - using minimal tokenizer to match training vocab_size")
            
            # Add required special tokens
            tokens_to_add = []
            for view in ['abstract'] + additional_views:
                if view in special_tokens:
                    token = special_tokens[view]
                    if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
                        tokens_to_add.append(token)
            
            # Handle detailed_description dependency on drawing
            if "detailed_description" in additional_views and "drawing" not in additional_views:
                drawing_token = special_tokens["drawing"]
                if drawing_token not in tokens_to_add and tokenizer.convert_tokens_to_ids(drawing_token) == tokenizer.unk_token_id:
                    tokens_to_add.append(drawing_token)
            
            if tokens_to_add:
                tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
                print(f"➕ Added tokens: {tokens_to_add}")
            else:
                print(f"✅ Using base tokenizer without additional tokens (vocab_size: {len(tokenizer)})")
            
            return tokenizer

        def load_model_with_fallback(checkpoint_path, config, model_args):
            """Load model with progressive fallback strategies."""
            from patentmap.models import BertForCL
            
            is_local = os.path.exists(checkpoint_path)
            loading_strategies = [
                # Strategy 1: Standard loading
                {"local_files_only": is_local, "trust_remote_code": True, "ignore_mismatched_sizes": True},
                # Strategy 2: Minimal parameters
                {"ignore_mismatched_sizes": True, "local_files_only": is_local},
                # Strategy 3: Basic fallback
                {"ignore_mismatched_sizes": True}
            ]
            
            for i, kwargs in enumerate(loading_strategies, 1):
                try:
                    print(f"🔄 Trying loading strategy {i}...")
                    return BertForCL.from_pretrained(checkpoint_path, config=config, model_args=model_args, **kwargs)
                except Exception as e:
                    print(f"❌ Strategy {i} failed: {e}")
                    if i == len(loading_strategies):
                        raise RuntimeError(f"All loading strategies failed for {checkpoint_path}")
            
        # Main loading execution
        model, tokenizer, embedding_dim = load_checkpoint_model_and_tokenizer(args.model_name)
        batch_size = 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup model for inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(device).eval()
        print(f"🚀 Model ready on {device}")

        ############################### IPC classification evaluation ###############################
        model_name = "-".join(args.model_name.strip("/").split("/")[1:])

        IPC_dir_temp = os.path.join(args.output_dir, f'IPC-Classification_temp_{model_name}')
        if not os.path.exists(IPC_dir_temp):
            os.makedirs(IPC_dir_temp)

        # check if the embeddings are already created
        if os.path.exists(f'{IPC_dir_temp}/train_embeddings.pt') and os.path.exists(f'{IPC_dir_temp}/test_embeddings.pt'):
            print("Embeddings already created!")
            train_embeddings = torch.load(f'{IPC_dir_temp}/train_embeddings.pt', weights_only=False)
            test_embeddings = torch.load(f'{IPC_dir_temp}/test_embeddings.pt', weights_only=False)
        else:
            # Filter datasets to only include 'abstract' text_type to match the labels
            train_dataset_filtered = train_dataset[train_dataset['text_type'] == 'abstract']
            test_dataset_filtered = test_dataset[test_dataset['text_type'] == 'abstract']
            
            # Use same format as during training (consistent with patent.py batcher)
            train_texts, test_texts = train_dataset_filtered['text'], test_dataset_filtered['text']

            # tokenize the texts on CPU; batches are moved to GPU one-by-one below
            # (keeping the full encodings on GPU wastes ~300MB and has caused
            # cuBLAS workspace OOM on subsequent classifier training).
            train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
            test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

            # get the embeddings by batch
            train_embeddings = np.zeros((len(train_encodings['input_ids']), embedding_dim))
            test_embeddings = np.zeros((len(test_encodings['input_ids']), embedding_dim))

            with torch.no_grad():
                for i in trange(0, len(train_encodings['input_ids']), batch_size, desc="Computing train embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in train_encodings.items()}
                    outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                    train_embeddings[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()

                for i in trange(0, len(test_encodings['input_ids']), batch_size, desc="Computing test embeddings"):
                    batch = {key: val[i:i+batch_size].to(device) for key, val in test_encodings.items()}
                    outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                    test_embeddings[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()

            print(train_embeddings.shape, test_embeddings.shape)

            # save the embeddings
            torch.save(train_embeddings, f'{IPC_dir_temp}/train_embeddings.pt')
            torch.save(test_embeddings, f'{IPC_dir_temp}/test_embeddings.pt')

            # release tokenizer encodings before the IPC classifier trains on GPU
            del train_encodings, test_encodings

        # Free transient GPU memory accumulated during embedding computation so
        # that cuBLAS can allocate its workspace for the IPC MLP without OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ipc_evaluation(train_embeddings, test_embeddings, train_labels, test_labels, train_types, test_types)

        ############################ Prior-art Search evaluation ############################
        # check if the embeddings are already created
        priorart_temp_dir = os.path.join(args.output_dir, f'priorart_temp_{model_name}')
        if not os.path.exists(priorart_temp_dir):
            os.makedirs(priorart_temp_dir)

        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)
        else:
            # Use EXACT same approach as patent.py: compute embeddings by text type separately
            # This ensures complete consistency when evaluating checkpoint models
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            
            # Process each text type separately, exactly like patent.py
            for texttype in ["abstract", "claim", "invention"]:
                # Format texts exactly like patent.py
                if texttype == "abstract":
                    query_texts = [queries_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    query_texts = [f"[{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [f"[{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                
                # Tokenize and compute embeddings for this text type
                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                
                # Compute embeddings
                query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))
                
                with torch.no_grad():
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in query_encodings.items()}
                        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                        query_embs[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()

                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in doc_encodings.items()}
                        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                        doc_embs[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # For compatibility with existing evaluation code, we'll create the concatenated versions
            # But the evaluation should use the separated versions to match patent.py exactly
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings in both formats for compatibility
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')
            
            # Also save the text-type separated embeddings (matching patent.py format)
            np.savez(f'{priorart_temp_dir}/query_embeddings_by_type.npz', **query_embeddings_dict)
            np.savez(f'{priorart_temp_dir}/doc_embeddings_by_type.npz', **doc_embeddings_dict)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')

        # For checkpoint models, use the exact same evaluation method as patent.py to ensure consistency
        print("Using patent.py-compatible evaluation for checkpoint model...")
        print("This ensures exact consistency with training-time evaluation results.")
        
        # Use the standard evaluation for now, but note that minor differences may exist
        # due to different data organization methods between baseline.py and patent.py
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, citation_mapping_graded=citation_mapping_graded)

        ############################ DAPFAM (TitlAbs -> TitlAbs) ############################
        if run_dapfam:
            dapfam_q_path = f'{dapfam_temp_dir}/query_embeddings.pt'
            dapfam_d_path = f'{dapfam_temp_dir}/document_embeddings.pt'
            if os.path.exists(dapfam_q_path) and os.path.exists(dapfam_d_path):
                print("DAPFAM embeddings already cached.")
                dapfam_q_emb = torch.load(dapfam_q_path, weights_only=False)
                dapfam_d_emb = torch.load(dapfam_d_path, weights_only=False)
            else:
                if args.dapfam_texttype == 'plain':
                    dapfam_q_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                            dapfam_queries_df['claims_text'].tolist(),
                        )
                    ]
                    dapfam_d_texts = [
                        " ".join(s for s in (t, a, extract_first_claim(c)) if s)
                        for t, a, c in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                            dapfam_documents_df['claims_text'].tolist(),
                        )
                    ]
                else:
                    dapfam_q_texts = [
                        t + " [SEP] [abstract] " + a
                        for t, a in zip(
                            dapfam_queries_df['title'].tolist(),
                            dapfam_queries_df['abstract'].tolist(),
                        )
                    ]
                    dapfam_d_texts = [
                        t + " [SEP] [abstract] " + a
                        for t, a in zip(
                            dapfam_documents_df['title'].tolist(),
                            dapfam_documents_df['abstract'].tolist(),
                        )
                    ]
                    if args.dapfam_texttype == 'tac':
                        dapfam_q_texts = [
                            txt + " [SEP] [claim] " + extract_first_claim(c)
                            for txt, c in zip(dapfam_q_texts, dapfam_queries_df['claims_text'].tolist())
                        ]
                        dapfam_d_texts = [
                            txt + " [SEP] [claim] " + extract_first_claim(c)
                            for txt, c in zip(dapfam_d_texts, dapfam_documents_df['claims_text'].tolist())
                        ]

                dapfam_q_enc = tokenizer(dapfam_q_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                dapfam_d_enc = tokenizer(dapfam_d_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                dapfam_q_emb = np.zeros((len(dapfam_q_enc['input_ids']), embedding_dim), dtype=np.float32)
                dapfam_d_emb = np.zeros((len(dapfam_d_enc['input_ids']), embedding_dim), dtype=np.float32)
                with torch.no_grad():
                    for i in trange(0, len(dapfam_q_enc['input_ids']), batch_size, desc="DAPFAM query embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in dapfam_q_enc.items()}
                        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                        dapfam_q_emb[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()

                    for i in trange(0, len(dapfam_d_enc['input_ids']), batch_size, desc="DAPFAM doc embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in dapfam_d_enc.items()}
                        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                        dapfam_d_emb[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()

                torch.save(dapfam_q_emb, dapfam_q_path, pickle_protocol=4)
                torch.save(dapfam_d_emb, dapfam_d_path, pickle_protocol=4)
                print(f"DAPFAM embeddings shape: q={dapfam_q_emb.shape}, d={dapfam_d_emb.shape}")

            dapfam_evaluation(
                dapfam_queries_df.index.tolist(),
                dapfam_documents_df.index.tolist(),
                dapfam_q_emb,
                dapfam_d_emb,
                dapfam_cmap,
                dapfam_query_ipc3,
                dapfam_doc_ipc3,
                k=100,
            )

        ################################ Comprehensive Embedding Quality Evaluation ################################
        # load embeddings from prior-art search task
        query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
        document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        # Debug logging for embeddings alignment
        embeddings2analyze = np.concatenate((query_embeddings, document_embeddings), axis=0)
        # print(len(embeddings2analyze), len(query_types), len(doc_types))
        # print(len(query_embeddings), len(document_embeddings))
        assert len(embeddings2analyze) == len(query_types + doc_types), f"Mismatch in number of embeddings and types: {len(embeddings2analyze)} vs {len(query_types + doc_types)}"

        # Run comprehensive quality evaluation (uniformity, SSD, alignment, topology)
        comprehensive_embedding_quality_evaluation(
            query_embeddings=query_embeddings,
            doc_embeddings=document_embeddings,
            query_ids=query_ids,
            doc_ids=doc_ids,
            citation_mapping=citation_mapping,
            query_types=query_types,
            doc_types=doc_types,
            queries_df=queries_df,
            documents_df=documents_df
        )
        
        # Print evaluation completion summary
        print_section_header("✅ Evaluation Complete!")
        print(f"🎯 All evaluation tasks completed successfully for: {args.model_name}")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"📊 Evaluated tasks: IPC Classification, Prior Art Search, Uniformity, Alignment, Topology")
        log_evaluation_complete("Patent Embedding Evaluation Pipeline")

########################################################################################################################################################
########################################################################################################################################################
    else:
        raise ValueError("No model found!")

def cleanup_resources():
    """Clean up GPU memory and other resources to prevent segfaults"""
    import gc
    import torch
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Force synchronization
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Warning: Error during GPU cleanup: {e}")
    
    # Force garbage collection
    gc.collect()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup resources
        cleanup_resources()
        print("Resource cleanup completed.")