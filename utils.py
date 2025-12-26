"""
Utility functions and classes for patent document processing and contrastive learning.

This module provides:
- Tokenizer setup and special token management
- Dataset classes for training and diagnostics
- Data augmentation strategies
- Text processing utilities (sentence segmentation, cropping, shuffling)
- Data streaming and shuffling strategies
- IPC-based grouping for contrastive learning
"""

import random
import re
import math
import os
import threading
import pickle
import logging
import json
import traceback
import numpy as np
from collections import defaultdict, Counter
from typing import Callable, Dict, List, Tuple, Optional, NamedTuple
from pathlib import Path

import pandas as pd
import torch
import spacy
from spacy.language import Language
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import IterableDataset, Dataset as TorchDataset, DataLoader


# Configure logger
logger = logging.getLogger(__name__)

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True


class Float32Encoder(json.JSONEncoder):
    """JSON encoder that handles numpy float32/float64 types"""
    def default(self, obj):
        # convert np.float32 -> Python float
        if isinstance(obj, np.float32):
            return float(obj)
        # convert np.float64 (etc.) if needed
        if isinstance(obj, np.float64):
            return float(obj)
        return super().default(obj)


class IterableWithLen(IterableDataset):
    """A wrapper class that adds length information to an iterable."""
    def __init__(self, iterable_func, length=None):
        super().__init__()
        self.iterable_func = iterable_func
        self._length = length
    
    def __iter__(self):
        return self.iterable_func()
    
    def __len__(self):
        if self._length is not None:
            return self._length
        # If no length provided, try to estimate or return a default
        return 0


class EvaluateAtLogarithmicStepsCallback(TrainerCallback):
    """Callback for evaluating at logarithmic steps during training"""
    def __init__(self, trainer=None, tolerance: float = 1e-3):
        self.tolerance = tolerance
        self.trainer = trainer
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        This function is called at the end of each training step for controling the evaluation steps.
        """
        step = state.global_step
        if step < 128:
            log_step = math.log(step, 4)
            if abs(round(log_step) - log_step) < self.tolerance:
                control.should_evaluate = True
        else:
            if step % 125 == 0:
                control.should_evaluate = True
        return control


def shorten_task_id(task_id, max_length=128):
    """
    Shorten task_id to fit within wandb's 128 character limit while preserving key information.
    """
    if len(task_id) <= max_length:
        return task_id
    
    # Extract timestamp (always at the end)
    timestamp_pos = task_id.rfind('_20')  # Look for timestamp pattern
    if timestamp_pos != -1:
        timestamp = task_id[timestamp_pos:]
        prefix = task_id[:timestamp_pos]
    else:
        timestamp = ""
        prefix = task_id
    
    # Calculate available space for prefix
    available_space = max_length - len(timestamp)
    
    if len(prefix) <= available_space:
        return task_id
    
    # Shorten the prefix while keeping key components
    # Priority: da_aug -> views -> mlm -> adaptive-aug -> reg/vicreg
    parts = prefix.split('_')
    shortened_parts = []
    
    for part in parts:
        # Abbreviate long component names
        if part.startswith('adaptive-aug-'):
            part = 'aa-' + part[13:]  # shorten adaptive-aug
        elif part.startswith('vicreg_lambda-'):
            part = 'vl-' + part[14:]  # shorten vicreg_lambda
        elif 'vicreg' in part and len(part) > 10:
            # Shorten vicreg parameter names
            part = part.replace('vicreg_', 'v').replace('invariance_weight', 'iw').replace('variance_weight', 'vw').replace('covariance_weight', 'cw').replace('hidden_dim', 'hd')
        
        shortened_parts.append(part)
    
    shortened_prefix = '_'.join(shortened_parts)
    
    # If still too long, truncate from the middle
    if len(shortened_prefix) + len(timestamp) > max_length:
        max_prefix_len = max_length - len(timestamp) - 3  # -3 for "..."
        if max_prefix_len > 20:  # Only truncate if we have reasonable space
            shortened_prefix = shortened_prefix[:max_prefix_len//2] + "..." + shortened_prefix[-(max_prefix_len//2):]
        else:
            shortened_prefix = shortened_prefix[:max_prefix_len]
    
    return shortened_prefix + timestamp


class ProbeDataset(TorchDataset):
    """Simple dataset wrapper for diagnostic probe data."""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            raise ValueError(f"ProbeDataset received list of indices - DataLoader misconfiguration!")
        
        if not isinstance(idx, int):
            raise ValueError(f"ProbeDataset expected int index, got {type(idx)}: {idx}")
        
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.samples)}")
        
        return self.samples[idx]
    
    def __getitems__(self, indices):
        """Handle batch access properly for DataLoader optimization"""
        if not isinstance(indices, list):
            raise ValueError(f"__getitems__ expected list of indices, got {type(indices)}")
        
        for idx in indices:
            if not isinstance(idx, int):
                raise ValueError(f"Invalid index type in batch: {type(idx)}")
            if idx < 0 or idx >= len(self.samples):
                raise IndexError(f"Index {idx} out of bounds")
        
        return [self.samples[idx] for idx in indices]


def setup_patent_special_tokens(tokenizer, additional_views):
    """
    Add patent section special tokens to tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        additional_views: List of section names to add (e.g., ['claim', 'summary'])
    
    Returns:
        tuple: (tokenizer, special_tokens_map) - Modified tokenizer and dict mapping section names to token strings
    """
    special_tokens_map = {
        "abstract": "[abstract]",
        "claim": "[claim]",
        "summary": "[summary]",
        "background": "[invention]",
        "drawing": "[drawing]",
        "detailed_description": "[description]"
    }
    
    assert all([section in special_tokens_map for section in additional_views]), \
        f"Invalid additional views: {additional_views}. Must be one of {list(special_tokens_map.keys())}."
    
    # Collect tokens to add
    seen = set()
    tokens_to_add = []
    for view in ['abstract'] + additional_views:
        if view not in seen:
            tokens_to_add.append(special_tokens_map[view])
            seen.add(view)
    
    # Special case: detailed_description needs drawing token
    if "detailed_description" in additional_views and "drawing" not in additional_views:
        if special_tokens_map["drawing"] not in tokens_to_add:
            tokens_to_add.append(special_tokens_map["drawing"])
    
    # Sort tokens to maintain consistent order
    tokens_to_add = sorted(tokens_to_add, key=lambda x: list(special_tokens_map.values()).index(x))
    
    # Filter out existing special tokens
    existing_specials = set(tokenizer.additional_special_tokens or [])
    tokens2add = [tok for tok in tokens_to_add if tok not in existing_specials]
    
    if tokens2add:
        tokenizer.add_special_tokens({'additional_special_tokens': tokens2add})
    
    return tokenizer, special_tokens_map


def create_probe_dataset(data_args, model_args, tokenizer, special_tokens_map, get_tokenized_dataset_path_fn, diagnostic_probe_size=2000):
    """
    Create a small diagnostic probe dataset by directly sampling from tokenized training data.
    This ensures the probe data has exactly the same distribution as training data.
    
    Args:
        data_args: Data training arguments
        model_args: Model arguments
        tokenizer: Tokenizer instance
        special_tokens_map: Special tokens mapping
        get_tokenized_dataset_path_fn: Function to get tokenized dataset path
        diagnostic_probe_size: Target size of probe dataset
    
    Returns:
        ProbeDataset instance or None if creation fails
    """
    
    logger.info(f"Creating diagnostic probe dataset with {diagnostic_probe_size} samples...")
    logger.info(f"Sampling directly from tokenized training data...")
    
    try:
        # Collect samples from tokenized data chunks
        probe_samples = []
        target_size = diagnostic_probe_size
        
        # Sample from the first year's data
        probe_year = data_args.start_year
        base_path = Path(get_tokenized_dataset_path_fn(probe_year, data_args, model_args))
        chunk_paths = sorted(base_path.parent.glob(f"{base_path.name}_chunk*.pt"))
        
        if not chunk_paths:
            logger.warning(f"No tokenized data found for year {probe_year}. Skipping diagnostic setup.")
            return None
        
        # Load chunks and sample randomly
        for chunk_path in chunk_paths:
            if len(probe_samples) >= target_size:
                break
            
            logger.info(f"Loading chunk: {chunk_path.name}")
            chunk_data = torch.load(chunk_path, weights_only=False)
            
            # Calculate how many samples to take from this chunk
            remaining = target_size - len(probe_samples)
            chunk_size = len(chunk_data)
            sample_size = min(remaining, chunk_size)
            
            # Randomly sample indices from this chunk
            rng = random.Random(42)  # Fixed seed for reproducibility
            sampled_indices = rng.sample(range(chunk_size), sample_size)
            
            # Extract samples
            for idx in sampled_indices:
                probe_samples.append(chunk_data[idx])
            
            logger.info(f"Collected {len(probe_samples)}/{target_size} samples")
        
        if len(probe_samples) < 100:
            logger.warning(f"Only {len(probe_samples)} samples collected. Diagnostic results may not be reliable.")
            return None
        
        logger.info(f"Successfully created probe dataset with {len(probe_samples)} samples")
        
        # Create probe dataset instance
        probe_dataset = ProbeDataset(probe_samples)
        logger.info(f"Diagnostic probe dataset ready with {len(probe_dataset)} samples")
        return probe_dataset
        
    except Exception as e:
        logger.error(f"Failed to create probe dataset: {e}")
        logger.error(traceback.format_exc())
        return None



# Thread-local storage and caching for spaCy models
_thread_local = threading.local()
_nlp_cache = {}


@Language.component("unified_patent_sentence_rules")
def unified_patent_sentence_rules(doc):
    """
    Unified patent document sentence segmentation rules, processing all cases in one pass:
    1. Merge patent numbers '1.' '2.' with main text
    2. Handle Fig./FIG. abbreviations
    3. Merge "no." with patent numbers
    4. Merge "et al." academic citations
    """
    tokens = list(doc)
    n = len(tokens)
    
    # Mark token indices that need to have sentence start cancelled
    cancel_sent_start = set()
    
    for i in range(n):
        tok = tokens[i]
        
        # Rule 1: Handle patent numbers "1." "2." etc.
        if (i + 2 < n and 
            tok.like_num and 
            tokens[i + 1].text == "."):
            cancel_sent_start.add(i + 2)
        
        # Rule 2: Handle Fig./FIG. abbreviations
        elif (i + 2 < n and
              tok.text.lower() in {"fig", "figs"} and
              tokens[i + 1].text == "."):
            cancel_sent_start.add(i + 2)
        
        # Rule 3: Handle "no." with patent numbers
        elif tok.text.lower() in {"no", "no."}:
            next_idx = i + 1
            if tok.text.lower() == "no" and next_idx < n and tokens[next_idx].text == ".":
                next_idx = i + 2  # Skip period
            
            if next_idx < n:
                # Cancel sentence start markers for all patent number related tokens
                num_idx = next_idx
                while (num_idx < n and 
                       (tokens[num_idx].like_num or 
                        tokens[num_idx].text in {",", "/", "-", ".", ":"} or
                        tokens[num_idx].text.isdigit())):
                    cancel_sent_start.add(num_idx)
                    num_idx += 1
                    
                    # If period is followed by a letter, stop merging
                    if (num_idx < n - 1 and 
                        tokens[num_idx - 1].text == "." and 
                        tokens[num_idx].text.isalpha()):
                        break
        
        # Rule 4: Handle "et al." academic citations
        elif (i + 3 < n and
              tok.text.lower() == "et" and 
              tokens[i + 1].text.lower() == "al" and
              tokens[i + 2].text == "."):
            
            # Start merging from after "et al."
            next_idx = i + 3
            while next_idx + 1 < n:
                current_token = tokens[next_idx]
                
                # Conditions for continuing merge
                if (current_token.text in {",", ";", "(", ")", "-"} or
                    current_token.like_num or
                    current_token.text.isdigit() or
                    current_token.text.lower() in {"p", "pp", "vol", "no"}):
                    
                    cancel_sent_start.add(next_idx)
                    next_idx += 1
                
                # True sentence end
                elif (current_token.text == "." and 
                      next_idx + 1 < n and 
                      tokens[next_idx + 1].text and 
                      tokens[next_idx + 1].text[0].isupper()):
                    break
                
                # Other punctuation continues merging one token
                elif current_token.text in {".", ":", "!"}:
                    cancel_sent_start.add(next_idx)
                    next_idx += 1
                    
                    # Check next token
                    if (next_idx < n and 
                        (tokens[next_idx].text.islower() or 
                         tokens[next_idx].like_num or
                         tokens[next_idx].text in {"-", "(", ")"})):
                        cancel_sent_start.add(next_idx)
                        next_idx += 1
                    else:
                        break
                else:
                    break
    
    # Apply all rules: cancel sentence start markers for specified tokens
    for idx in cancel_sent_start:
        if idx < len(doc):
            doc[idx].is_sent_start = False
    
    return doc


def make_sentencizer_pipeline():
    """Contains only sentencizer + patent rules, extremely lightweight."""
    nlp = spacy.blank("en")                         # Don't load tagger / parser
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("unified_patent_sentence_rules", last=True)
    return nlp


def get_nlp():
    """Safely get spaCy nlp instance in multi-process/multi-thread environment."""
    pid = os.getpid()
    if pid not in _nlp_cache:
        # === 1) Optional: Enable GPU acceleration for sentence segmentation (spaCy v3.5+ recommended) ===
        gpu_id = None
        try:
            # Check spaCy version and GPU availability
            from packaging import version
            
            if torch.cuda.is_available():
                # Use different GPU initialization methods based on spaCy version
                spacy_version = version.parse(spacy.__version__)
                
                if spacy_version >= version.parse("3.4.0"):
                    # New version spaCy uses require_gpu
                    try:
                        spacy.require_gpu(0)
                        gpu_id = 0
                        print(f"[spaCy] Process {pid} using GPU:0 for sentencizer")
                    except (RuntimeError, ImportError, AttributeError) as e:
                        print(f"[spaCy] GPU initialization failed: {e}, falling back to CPU")
                        gpu_id = None
                else:
                    # Older versions might have to_gpu method
                    gpu_id = 0
                    print(f"[spaCy] Process {pid} attempting GPU:0 for sentencizer")
            
        except (ImportError, RuntimeError, OSError) as e:
            print(f"[spaCy] GPU setup error: {e}, using CPU")
            gpu_id = None
        
        nlp = make_sentencizer_pipeline()
        nlp.initialize()    # v3 requires manual initialize
        
        # Use different GPU methods based on spaCy version
        if gpu_id is not None:
            try:
                if hasattr(nlp, 'to_gpu'):
                    nlp.to_gpu(gpu_id)
                elif hasattr(nlp, 'use_gpu'):
                    nlp.use_gpu(gpu_id)
                else:
                    print(f"[spaCy] GPU methods not available, using CPU")
            except Exception as e:
                print(f"[spaCy] GPU assignment failed: {e}, using CPU")
        
        _nlp_cache[pid] = nlp
    return _nlp_cache[pid]


def shuffle_sentences(text: str) -> str:
    """Randomly shuffle sentence order in text."""
    nlp = get_nlp()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    random.shuffle(sentences)
    return " ".join(sentences)


def rand_crop_continuous(text: str, ratio: float = 0.1) -> str:
    """Continuous segment cropping - randomly select a continuous subsequence"""
    tokens = text.split()
    if len(tokens) < 5:  # Too short, don't process
        return text
    
    # Calculate number of tokens to keep
    keep_length = max(3, int(len(tokens) * (1 - ratio)))
    keep_length = min(keep_length, len(tokens) - 1)  # Ensure at least one token is deleted
    
    # Randomly select start position
    max_start = len(tokens) - keep_length
    start_idx = random.randint(0, max_start)
    
    # Extract continuous segment
    cropped_tokens = tokens[start_idx:start_idx + keep_length]
    return " ".join(cropped_tokens)


def rand_crop_span(text: str, ratio: float = 0.1) -> str:
    """Randomly delete a continuous segment - span deletion"""
    tokens = text.split()
    if len(tokens) < 5:
        return text
    
    # Calculate length of continuous segment to delete
    delete_length = max(1, int(len(tokens) * ratio))
    delete_length = min(delete_length, len(tokens) - 3)  # Keep at least 3 tokens
    
    # Randomly select deletion position
    max_start = len(tokens) - delete_length
    start_idx = random.randint(0, max_start)
    
    # Delete continuous segment
    result_tokens = tokens[:start_idx] + tokens[start_idx + delete_length:]
    return " ".join(result_tokens)


def rand_crop_hybrid(text: str, ratio: float = 0.1) -> str:
    """Hybrid strategy random cropping"""
    tokens = text.split()
    if len(tokens) < 5:
        return text
    
    # Randomly select cropping strategy
    strategies = ['continuous', 'span_deletion', 'random_tokens', 'smart']
    strategy = random.choice(strategies)
    
    if strategy == 'continuous':
        return rand_crop_continuous(text, ratio)
    elif strategy == 'span_deletion':
        return rand_crop_span(text, ratio)
    elif strategy == 'smart':
        return rand_crop_smart(text, ratio)
    else:  # random_tokens - improved version of original method
        return rand_crop_tokens_improved(text, ratio)


def rand_crop_tokens_improved(text: str, ratio: float = 0.1) -> str:
    """Improved random token deletion"""
    tokens = text.split()
    if len(tokens) < 5:
        return text
    
    # Calculate deletion count, limit maximum deletion ratio
    max_delete_ratio = min(ratio, 0.3)  # Delete at most 30%
    delete_count = max(1, int(len(tokens) * max_delete_ratio))
    delete_count = min(delete_count, len(tokens) - 3)  # Keep at least 3 tokens
    
    # Avoid deleting important tokens at beginning and end of sentence
    protected_start = min(2, len(tokens) // 4)  # Protect first few tokens
    protected_end = min(2, len(tokens) // 4)    # Protect last few tokens
    
    # Range of deletable tokens
    deleteable_indices = list(range(protected_start, len(tokens) - protected_end))
    
    if len(deleteable_indices) < delete_count:
        # If deletable range is too small, reduce deletion count
        delete_count = max(1, len(deleteable_indices) // 2)
    
    # Randomly select deletion positions
    indices_to_delete = set(random.sample(deleteable_indices, delete_count))
    
    # Build result
    result_tokens = [token for i, token in enumerate(tokens) if i not in indices_to_delete]
    return " ".join(result_tokens)


def rand_crop_smart(text: str, ratio: float = 0.1) -> str:
    """Smart cropping - avoid deleting important vocabulary"""
    tokens = text.split()
    if len(tokens) < 5:
        return text
    
    # Define important words (patent domain specific)
    important_words = {
        'invention', 'method', 'apparatus', 'system', 'device', 'process',
        'comprising', 'including', 'wherein', 'characterized', 'embodiment',
        'claim', 'figure', 'example', 'preferred', 'according'
    }
    
    # Define unimportant words (can be deleted preferentially)
    unimportant_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'very', 'quite', 'rather', 'really', 'also', 'just', 'still', 'even'
    }
    
    # Calculate deletion count
    delete_count = max(1, int(len(tokens) * ratio))
    delete_count = min(delete_count, len(tokens) - 3)
    
    # Create candidate deletion list (priority: unimportant > normal > important)
    deletion_candidates = []
    
    for i, token in enumerate(tokens):
        token_lower = token.lower().strip('.,!?;:')
        if token_lower in unimportant_words:
            deletion_candidates.append((i, 0))  # Priority 0 (highest)
        elif token_lower not in important_words:
            deletion_candidates.append((i, 1))  # Priority 1 (medium)
        else:
            deletion_candidates.append((i, 2))  # Priority 2 (lowest)
    
    # Sort by priority and select deletion positions
    deletion_candidates.sort(key=lambda x: (x[1], random.random()))
    indices_to_delete = {idx for idx, _ in deletion_candidates[:delete_count]}
    
    # Build result
    result_tokens = [token for i, token in enumerate(tokens) if i not in indices_to_delete]
    return " ".join(result_tokens)


def rand_crop(text: str, ratio: float = 0.1) -> str:
    """Cropping strategy specifically designed for patent documents"""
    
    # First perform sentence segmentation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if len(sentences) <= 1:
        # Single sentence, use token-level cropping
        return rand_crop_hybrid(text, ratio)
    
    # Strategy selection for multi-sentence cases
    strategy = random.choice(['sentence_level', 'token_level', 'mixed'])
    
    if strategy == 'sentence_level':
        # Sentence-level cropping: delete entire sentences
        delete_count = max(1, int(len(sentences) * ratio))
        delete_count = min(delete_count, len(sentences) - 1)  # Keep at least one sentence
        
        indices_to_keep = random.sample(range(len(sentences)), len(sentences) - delete_count)
        indices_to_keep.sort()
        
        result_sentences = [sentences[i] for i in indices_to_keep]
        return " ".join(result_sentences)
    
    elif strategy == 'mixed':
        # Mixed strategy: delete some sentences + token-level cropping
        if len(sentences) > 2:
            # First delete one sentence
            sentence_to_remove = random.randint(1, len(sentences) - 1)  # Avoid deleting first sentence
            remaining_sentences = sentences[:sentence_to_remove] + sentences[sentence_to_remove + 1:]
            text = " ".join(remaining_sentences)
        
        # Then perform token-level cropping
        return rand_crop_tokens_improved(text, ratio * 0.5)  # Halve ratio
    
    else:  # token_level
        return rand_crop_hybrid(text, ratio)



aug_registry: Dict[str, Callable[[dict, "Ctx"], Tuple[str, str, bool]]] = {}

class Ctx(NamedTuple):
    rng: random.Random              # Thread/process independent RNG
    crop_ratio: float
    section_candidates: List[str]   # Section candidates for section_pair
    ipc_groups: Dict[Tuple[str], List[str]]
    paraphraser: Optional[Callable[[str], str]]
    tokenizer: Optional[object] = None  # tokenizer for special tokens
    special_tokens_map: Optional[Dict[str, str]] = None

def register(name):
    def wrap(fn): aug_registry[name] = fn; return fn
    return wrap

def create_adaptive_wrapper(base_aug_fn):
    """
    Create an adaptive wrapper for any augmentation function.
    
    With 50% probability, swaps v1 and v2 from the base augmentation function.
    This provides the same adaptive behavior as section_pair_adaptive but 
    for any augmentation method.
    
    Args:
        base_aug_fn: The base augmentation function to wrap
    
    Returns:
        Wrapped function that applies adaptive v1/v2 swapping
    """
    def adaptive_wrapper(ex, ctx):
        # Get the original result from base augmentation
        v1, v2, need_dropout = base_aug_fn(ex, ctx)
        
        # 🎲 Adaptive decision: randomly choose whether to swap
        swap_probability = 0.5  # 50% chance to swap
        should_swap = ctx.rng.random() < swap_probability
        
        if should_swap:
            # Swap v1 and v2
            return v2, v1, need_dropout
        else:
            # Keep original order
            return v1, v2, need_dropout
    
    return adaptive_wrapper

# ---------- 1) section_pair ----------
# Note: Use adaptive_augmentation=True for v1/v2 swapping instead of the old section_pair_adaptive
@register("section_pair")
def aug_section_pair(ex, ctx: Ctx):
    # v1 is always formatted abstract (Title [SEP] [abstract] content)
    v1 = ex.get("abstract_formatted", ex.get("abstract", ""))
    # v2 selected from other section candidates (excluding abstract)
    other_sections = [s for s in ctx.section_candidates if s != "abstract"]
    if not other_sections:
        return v1, v1, True
    
    # Randomly select another section
    s2 = ctx.rng.choice(other_sections)
    
    # Check if this section exists and has content
    section_content = ex.get(s2, "")

    # For non-abstract sections, only use section token + content (no title and [SEP])
    if section_content and ctx.special_tokens_map:
        section_token = ctx.special_tokens_map.get(s2, "")
        if section_token:
            v2 = f"{section_token} {section_content.strip()}"
        else:
            v2 = section_content.strip()
    else:
        v2 = section_content.strip() if section_content else ""
    
    # Check if v2 is valid
    if not v2 or len(v2.strip()) < 20:
        return v1, v1, True
    return v1, v2, False

# ---------- 2) dropout ----------
@register("dropout")
def aug_dropout(ex, ctx):
    # Use formatted abstract (including title and section token)
    txt = ex.get("abstract_formatted", ex.get("abstract", ""))
    return txt, txt, True               # True = needs dropout

# ---------- 3) rand_crop ----------
@register("rand_crop")
def aug_rand_crop(ex, ctx):
    # Separate title and content from formatted text
    formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
    title, section_content, section_token = extract_content_for_augmentation(formatted_text, ctx.special_tokens_map or {})
    
    # Only crop section content, keep title unchanged
    cropped_content = rand_crop(section_content, ctx.crop_ratio) if section_content else ""
    
    # Reconstruct formatted text
    view1 = formatted_text
    view2 = reconstruct_from_augmented_content(title, cropped_content, section_token)
    
    if view2 == view1:
        return view1, view1, True
    else:
        return view1, view2, False

# ---------- 4) sentence_shuffle ----------
@register("sentence_shuffle")
def aug_sent_shuffle(ex, ctx):
    # Separate title and content from formatted text
    formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
    title, section_content, section_token = extract_content_for_augmentation(formatted_text, ctx.special_tokens_map or {})
    
    # Only shuffle section content sentences, keep title unchanged
    shuffled_content = shuffle_sentences(section_content) if section_content else ""
    
    # Reconstruct formatted text
    view1 = formatted_text
    view2 = reconstruct_from_augmented_content(title, shuffled_content, section_token)
    
    if view2 == view1:
        return view1, view1, True
    else:
        return view1, view2, False

# ---------- 5) ipc8 ----------
@register("ipc8")
def aug_ipc8(ex, ctx):
    # Use formatted abstract
    base = ex.get("abstract_formatted", ex.get("abstract", ""))
    
    # Get current sample's IPC labels
    ipc_labels = ex.get("ipcr_labels", [])
    if not ipc_labels or len(ipc_labels) == 0:
        return base, base, True
    
    # Ensure IPC labels format is correct
    try:
        if hasattr(ipc_labels, '__iter__') and not isinstance(ipc_labels, str):
            ipc_list = list(ipc_labels)
            ipc_list = [ipc for ipc in ipc_list if ipc and str(ipc).strip()]
            if not ipc_list:
                return base, base, True
        else:
            return base, base, True
    except:
        return base, base, True
    
    ipc_key = tuple(sorted(ipc_list))
    
    pool = ctx.ipc_groups.get(ipc_key, [])
    if len(pool) <= 1:
        return base, base, True
    
    # Select different samples
    for _ in range(5):
        candidate = ctx.rng.choice(pool)
        # IPC groups store original text, need formatting
        candidate_text = candidate.get("abstract", "")
        candidate_title = candidate.get("title", "")
        
        # Construct formatted text for candidate sample
        if ctx.special_tokens_map and ctx.tokenizer:
            view2 = concatenate_title_and_section(
                candidate_title, candidate_text, "abstract", 
                ctx.tokenizer, ctx.special_tokens_map
            )
        else:
            view2 = candidate_text
            
        if view2 != base and len(view2.strip()) > 50:
            return base, view2, False
        
    return base, base, True

# ---------- 6) paraphrase ----------
@register("paraphrase")
def aug_para(ex, ctx):
    # For compatibility, we keep the individual logic
    # but the real performance gain will come from batch_paraphrase_augmentation
    formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
    title, section_content, section_token = extract_content_for_augmentation(formatted_text, ctx.special_tokens_map or {})
    
    # Seulement faire la paraphrase si le paraphraser est disponible et le contenu est suffisant
    if ctx.paraphraser and section_content and len(section_content.strip()) > 20:
        paraphrased_content = ctx.paraphraser.batch_paraphrase([section_content])[0]
    else:
        paraphrased_content = section_content
    
    # Reconstruct the formatted text
    view1 = formatted_text
    view2 = reconstruct_from_augmented_content(title, paraphrased_content, section_token)
    return view1, view2, False
######################### new data augmentation registry [END] #########################



def build_efficient_ipc_groups(start_year, end_year, feather_dir, cache_dir=None, max_group_size=300):
    """Build efficient IPC grouping index - optimized version"""
    
    # 1. Caching strategy
    if cache_dir:
        cache_key = f"ipc_groups_{start_year}_{end_year}_{max_group_size}"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            print(f"-> Loading cached IPC groups from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Use more efficient data structures
    ipc_counter = Counter()
    raw_samples = defaultdict(list)  # Temporarily store all samples
    
    # First pass: collect all data and statistics
    for year in range(start_year, end_year + 1):
        feather_path = os.path.join(feather_dir, f"patentmap_dataset_{year}.feather")
        if not os.path.exists(feather_path):
            continue
            
        print(f"   - Scanning year {year}...")
        try:
            df = pd.read_feather(feather_path, columns=["ipcr_labels", "abstract", "title"])
        except Exception as e:
            print(f"   - Warning: Could not read {feather_path}: {e}")
            continue
        
        # Batch processing
        for _, row in df.iterrows():
            ipc_labels = row["ipcr_labels"]
            abstract = row["abstract"]
            title = row.get("title", "")  # Get title, use empty string if not exists
            
            # Fix data type checking
            valid_ipc = False
            processed_labels = None
            
            if ipc_labels is not None:
                # Handle different data types
                if isinstance(ipc_labels, (list, tuple)):
                    if len(ipc_labels) > 0:
                        processed_labels = list(ipc_labels)
                        valid_ipc = True
                elif isinstance(ipc_labels, pd.Series):
                    if len(ipc_labels) > 0:
                        processed_labels = ipc_labels.tolist()
                        valid_ipc = True
                elif hasattr(ipc_labels, '__iter__') and hasattr(ipc_labels, '__len__'):
                    # Handle numpy arrays or other iterable objects
                    try:
                        if len(ipc_labels) > 0:
                            processed_labels = list(ipc_labels)
                            valid_ipc = True
                    except:
                        pass
            
            # Quality pre-filtering
            if (valid_ipc and processed_labels and 
                isinstance(abstract, str) and len(abstract.strip()) > 50):
                ipc_key = tuple(sorted(processed_labels))
                ipc_counter[ipc_key] += 1
                
                # Only save necessary information (including title)
                raw_samples[ipc_key].append({
                    "abstract": abstract.strip(),
                    "title": title.strip() if isinstance(title, str) else "",
                    "year": year,
                    "word_count": len(abstract.strip().split())
                })
                
                # Debug information
                if len(raw_samples) <= 5:
                    print(f"Debug - Added sample with IPC key: {ipc_key}")
    
    print(f"   - Collected {len(raw_samples)} unique IPC combinations")
    print(f"   - IPC counter stats: {dict(list(ipc_counter.most_common(10)))}")
    
    # Second pass: build final groups (only process combinations with frequency > 1)
    valid_ipc_keys = {k for k, v in ipc_counter.items() if v > 1}
    final_groups = {}
    
    print(f"   - Processing {len(valid_ipc_keys)} valid IPC combinations...")
    for ipc_key in valid_ipc_keys:
        candidates = raw_samples[ipc_key]
        
        if len(candidates) <= 1:
            continue
        
        # Quality sorting: prefer samples with moderate word count
        candidates.sort(key=lambda x: abs(x["word_count"] - 100))  # Prefer samples around 100 words
        
        # Diversity sampling
        year_groups = defaultdict(list)
        for candidate in candidates:
            year_groups[candidate["year"]].append(candidate)
        
        selected_samples = []
        target_per_year = max(1, max_group_size // len(year_groups))
        
        for year, year_samples in year_groups.items():
            # Select at most target_per_year samples from each year
            selected_count = min(len(year_samples), target_per_year)
            # Ensure title information is saved
            for sample in year_samples[:selected_count]:
                if "title" not in sample:
                    # If title is missing, set default empty value
                    sample["title"] = ""
            selected_samples.extend(year_samples[:selected_count])
        
        # Final size control
        if len(selected_samples) > max_group_size:
            # Maintain diversity while limiting size
            random.shuffle(selected_samples)
            selected_samples = selected_samples[:max_group_size]
        
        final_groups[ipc_key] = selected_samples
    
    # Statistics
    total_samples = sum(len(v) for v in final_groups.values())
    avg_size = total_samples / len(final_groups) if final_groups else 0
    
    print(f"   - Built {len(final_groups)} IPC groups")
    print(f"   - Total samples: {total_samples}")
    
    if final_groups:
        print(f"   - Average group size: {avg_size:.1f}")
        size_dist = sorted([len(v) for v in final_groups.values()])
        print(f"   - Size distribution (first 10): {size_dist[:10]}")
        print(f"   - Sample IPC keys: {list(final_groups.keys())[:5]}")
    else:
        print("   - No valid IPC groups found!")
        
        # Debug information: check why no valid groups
        print("   - Debug: Checking ipc_counter...")
        if ipc_counter:
            print(f"   - Top 10 IPC combinations: {dict(list(ipc_counter.most_common(10)))}")
            single_occurrence = sum(1 for count in ipc_counter.values() if count == 1)
            multi_occurrence = sum(1 for count in ipc_counter.values() if count > 1)
            print(f"   - Single occurrence: {single_occurrence}, Multi occurrence: {multi_occurrence}")
        else:
            print("   - ipc_counter is empty!")
    
    # Cache results
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"ipc_groups_{start_year}_{end_year}_{max_group_size}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(final_groups, f)
        print(f"   - Cached results to {cache_file}")
    
    return final_groups



torch.backends.cuda.matmul.allow_tf32 = True

class LocalLLMParaphraser:
    """Use Qwen-Chat small model for technical text paraphrasing"""
    def __init__(
        self,
        model_name="Qwen/Qwen3-0.6B",
        device="cuda",
        batch_size=32,
    ):
        self.device        = device
        self.batch_size    = batch_size
        self.max_prompt    = 2048    # Maximum input length for Qwen Chat model
        self.max_new       = 512     # Maximum new tokens to generate

        # Quantization
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
            use_fast=True,
            )
        # Qwen Chat model comes with built-in prompt template

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def batch_paraphrase(self, texts, **gen_kw):
        results = []
        dl = DataLoader(texts, batch_size=self.batch_size,
                collate_fn=lambda x: x, shuffle=False)

        for batch in dl:
            # === 1) Construct Chat prompt (Qwen official template) ===
            # Qwen's apply_chat_template accepts [{'role':…, 'content':…}, …]
            convs = [
                [
                    {"role": "system", "content": "You are an expert technical writer."},
                    {"role": "user",
                     "content":
                     f"Paraphrase the following technical text while preserving its meaning:\n\n{t}\n\nParaphrase:"}
                ]
                for t in batch
            ]
            toks = self.tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=True,
                enable_thinking=False,
                truncation=True,
                padding=True,
                max_length=self.max_prompt,
                return_tensors="pt"
            ).to(self.model.device)

            # === 2) Generate ===
            outs = self.model.generate(
                toks,
                max_new_tokens=self.max_new,
                attention_mask=(toks != self.tokenizer.pad_token_id).to(self.model.device),
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # === 3) Parse output ===
            decoded = self.tokenizer.batch_decode(
                outs[:, toks.shape[-1]:],   # Only take generated part
                skip_special_tokens=True
            )

            for para, src in zip(decoded, batch):
                txt = para.strip().strip('"')
                # clean up the paraphrase output
                txt = re.sub(r"^(Paraphrase:|Rewritten:)\s*", "", txt).strip()
                ok = 0.3 * len(src) < len(txt) < 3 * len(src) and txt != src
                results.append(txt if ok else src)

        return results


def build_global_ipc_groups_once(data_args):
    """Build global IPC groups once"""
    ipc_groups = {}
    if "ipc8" in data_args.data_augmentation:
        cache_dir = os.path.join(data_args.train_dir, ".cache")

        print(f"-> Building global IPC groups for years {data_args.start_year}-{data_args.end_year}")
        ipc_groups = build_efficient_ipc_groups(
            data_args.start_year, data_args.end_year, data_args.train_dir, 
            cache_dir=cache_dir,
            max_group_size=300
        )
    return ipc_groups


def initialize_global_paraphraser(data_args):
    """Initialize paraphraser once"""
    paraphraser = None
    if "paraphrase" in data_args.data_augmentation:
        try:
            paraphraser = LocalLLMParaphraser(device="cuda" if torch.cuda.is_available() else "cpu")
            print("-> Global paraphraser initialized successfully")
        except Exception as e:
            print(f"-> Warning: Failed to initialize paraphraser: {e}")
            print("-> Falling back to dropout strategy for paraphrase augmentation")
            paraphraser = None
    return paraphraser


def batch_sentence_shuffle(examples, ctx, batch_size=32):
    """Batch sentence shuffle processing - supports formatted text, only shuffle section content"""
    results = []
    
    # Separate titles and content from all samples
    title_content_pairs = []
    for ex in examples:
        formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
        title, section_content, section_token = extract_content_for_augmentation(formatted_text, ctx.special_tokens_map or {})
        title_content_pairs.append((formatted_text, title, section_content, section_token))
    
    # Extract all section contents for spaCy processing
    section_contents = [pair[2] for pair in title_content_pairs]
    
    # Batch spaCy processing
    nlp = get_nlp()
    
    # Process in batches to avoid memory overflow
    for i in range(0, len(section_contents), batch_size):
        batch_contents = section_contents[i:i+batch_size]
        
        # Batch processing
        docs = list(nlp.pipe(batch_contents, batch_size=batch_size))
        
        for j, doc in enumerate(docs):
            original_idx = i + j
            formatted_text, title, section_content, section_token = title_content_pairs[original_idx]
            
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if len(sentences) > 1:
                shuffled_sentences = sentences.copy()
                ctx.rng.shuffle(shuffled_sentences)
                shuffled_content = " ".join(shuffled_sentences)
                
                # Reconstruct formatted text, keep title unchanged
                view2 = reconstruct_from_augmented_content(title, shuffled_content, section_token)
                
                if view2 != formatted_text:
                    results.append((formatted_text, view2, False))
                else:
                    results.append((formatted_text, formatted_text, True))
            else:
                results.append((formatted_text, formatted_text, True))
    
    return results


def batch_ipc8(examples, ctx):
    """Optimized batch IPC8 processing - supports formatted text"""
    results = []
    
    # Pre-filtering: group current batch samples by IPC key
    batch_ipc_groups = defaultdict(list)
    no_ipc_samples = []
    
    for idx, ex in enumerate(examples):
        ipc_labels = ex.get("ipcr_labels", [])
        if ipc_labels and len(ipc_labels) > 0:
            # Ensure IPC labels format is correct
            if hasattr(ipc_labels, '__iter__') and not isinstance(ipc_labels, str):
                try:
                    ipc_list = list(ipc_labels)
                    ipc_list = [ipc for ipc in ipc_list if ipc and str(ipc).strip()]
                    if ipc_list:
                        ipc_key = tuple(sorted(ipc_list))
                        batch_ipc_groups[ipc_key].append((idx, ex))
                    else:
                        no_ipc_samples.append((idx, ex))
                except:
                    no_ipc_samples.append((idx, ex))
            else:
                no_ipc_samples.append((idx, ex))
        else:
            no_ipc_samples.append((idx, ex))
    
    # Process samples without IPC labels
    for idx, ex in no_ipc_samples:
        formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
        results.append((idx, formatted_text, formatted_text, True))
    
    # Batch process each IPC group
    for ipc_key, group_samples in batch_ipc_groups.items():
        # Get global same-group sample pool
        global_pool = ctx.ipc_groups.get(ipc_key, [])
        
        if len(global_pool) <= 1:
            # Insufficient samples in global pool, all use dropout
            for idx, ex in group_samples:
                formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
                results.append((idx, formatted_text, formatted_text, True))
            continue
        
        # Pre-filter global pool - only keep good quality samples
        quality_pool = [
            candidate for candidate in global_pool
            if (len(candidate.get("abstract", "").strip()) > 50 and 
                len(candidate.get("abstract", "").strip().split()) > 15)
        ]
        
        if len(quality_pool) <= 1:
            # Insufficient samples after quality filtering
            for idx, ex in group_samples:
                formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
                results.append((idx, formatted_text, formatted_text, True))
            continue
        
        # Find matches for each sample in the group
        for idx, ex in group_samples:
            base = ex.get("abstract_formatted", ex.get("abstract", ""))
            
            # Select candidate samples from quality pool
            found_match = False
            for _ in range(5):  # Try at most 5 times
                candidate = ctx.rng.choice(quality_pool)
                
                # Construct formatted text for candidate sample
                candidate_text = candidate.get("abstract", "")
                candidate_title = candidate.get("title", "")
                
                if ctx.special_tokens_map and ctx.tokenizer:
                    view2 = concatenate_title_and_section(
                        candidate_title, candidate_text, "abstract", 
                        ctx.tokenizer, ctx.special_tokens_map
                    )
                else:
                    view2 = candidate_text
                
                if view2 != base and len(view2.strip()) > 50:
                    results.append((idx, base, view2, False))
                    found_match = True
                    break
            
            if not found_match:
                results.append((idx, base, base, True))
    
    # Sort by original index and return
    results.sort(key=lambda x: x[0])
    return [(r[1], r[2], r[3]) for r in results]  # Return (v1, v2, dropout_flag)


def concatenate_title_and_section(title: str, section_text: str, section: str, 
                                tokenizer, special_tokens_map: Dict[str, str]) -> str:
    """Concatenate title and section text via [SEP], format: Title [SEP] [section] Section content"""
    # Clean input
    title_clean = title.strip() if title else ""
    section_clean = section_text.strip() if section_text else ""
    
    # Get section token
    section_token = special_tokens_map.get(section, "")
    
    # Ensure sep_token has correct spacing
    sep_token = " [SEP] "  # Force standard format, ensure spaces on both sides
    
    # If both are empty
    if not title_clean and not section_clean:
        return ""
    
    # Only section content, no title
    elif not title_clean:
        if section_token:
            return f"{section_token} {section_clean}"
        else:
            return section_clean
    
    # Only title, no section content
    elif not section_clean:
        if section_token:
            # Title [SEP] [section] (empty content, but maintain structure)
            return f"{title_clean}{sep_token}{section_token}"
        else:
            return title_clean
    
    # Both exist: Title [SEP] [section] Section content
    else:
        if section_token:
            return f"{title_clean}{sep_token}{section_token} {section_clean}"
        else:
            return f"{title_clean}{sep_token}{section_clean}"


def extract_content_for_augmentation(formatted_text: str, special_tokens_map: Dict[str, str]) -> tuple:
    """Separate title and content from formatted text for data augmentation
    
    Input format: Title [SEP] [section] Section content
    Output: (title, section_content, section_token)
    
    This way we can only apply augmentation to section content while keeping title unchanged
    """
    text = formatted_text.strip()
    
    # Find [SEP] position
    sep_markers = [" [SEP] ", "[SEP]"]
    sep_pos = -1
    for sep in sep_markers:
        pos = text.find(sep)
        if pos != -1:
            sep_pos = pos
            sep_length = len(sep)
            break
    
    if sep_pos == -1:
        # No [SEP], check if it starts directly with section token
        for section, token in special_tokens_map.items():
            if text.startswith(token + " "):
                # Format: [section] content (no title)
                content = text[len(token):].strip()
                return "", content, token
        # No structure token at all, treat as plain text
        return "", text, ""
    
    # Has [SEP], separate title and section parts
    title = text[:sep_pos].strip()
    section_part = text[sep_pos + sep_length:].strip()
    
    # Extract section token and content from section part
    section_token = ""
    section_content = section_part
    
    for section, token in special_tokens_map.items():
        if section_part.startswith(token + " "):
            section_token = token
            section_content = section_part[len(token):].strip()
            break
    
    return title, section_content, section_token


def reconstruct_from_augmented_content(title: str, augmented_content: str, section_token: str) -> str:
    """Reconstruct complete formatted text from augmented content
    
    Input: (title, augmented_section_content, section_token)
    Output: Title [SEP] [section] Augmented content
    """
    if not title and not augmented_content:
        return ""
    
    if not title:
        # Only content, no title
        if section_token:
            return f"{section_token} {augmented_content}"
        else:
            return augmented_content
    
    if not augmented_content:
        # Only title, no content
        if section_token:
            return f"{title} [SEP] {section_token}"
        else:
            return title
    
    # Both exist
    if section_token:
        return f"{title} [SEP] {section_token} {augmented_content}"
    else:
        return f"{title} [SEP] {augmented_content}"


def extract_text_for_augmentation(formatted_text: str, special_tokens_map: Dict[str, str]) -> str:
    """Compatibility function: extract complete text for certain augmentation strategies (like sentence shuffle)
    
    Input format: Title [SEP] [section] Section content  
    Output: Title Section content (keep title, because sentence shuffle etc. need complete context)
    """
    text = formatted_text
    
    # Remove all section tokens
    for token in special_tokens_map.values():
        text = text.replace(token, "").strip()
    
    # Remove [SEP] token (if exists)
    text = text.replace(" [SEP] ", " ").replace("[SEP]", " ").strip()
    
    # Clean extra spaces
    text = " ".join(text.split())
    return text


def reconstruct_formatted_text(augmented_text: str, original_formatted: str, 
                             special_tokens_map: Dict[str, str]) -> str:
    """Reconstruct formatted text from augmented plain text
    
    Strategy: Parse original structure and apply to augmented text
    Format: Title [SEP] [section] Content
    """
    # Parse original structure
    original = original_formatted.strip()
    
    # Detect section token
    section_token = None
    for token in special_tokens_map.values():
        if token in original:
            section_token = token
            break
    
    # Detect [SEP] position
    sep_pos = -1
    for sep_variant in [" [SEP] ", "[SEP]"]:
        if sep_variant in original:
            sep_pos = original.find(sep_variant)
            break
    
    # Reconstruction logic
    if sep_pos == -1:
        # No [SEP], simple format
        if section_token:
            return f"{section_token} {augmented_text.strip()}"
        else:
            return augmented_text.strip()
    
    else:
        # Has [SEP], complex format: Title [SEP] [section] Content
        title_part = original[:sep_pos].strip()
        
        if section_token:
            return f"{title_part} [SEP] {section_token} {augmented_text.strip()}"
        else:
            return f"{title_part} [SEP] {augmented_text.strip()}"


def batch_paraphrase_augmentation(examples, ctx, paraphraser, batch_size=32):
    """Optimized batch processing of paraphrases for augmentation"""
    if not paraphraser:
        # Fallback: return original texts
        results = []
        for ex in examples:
            formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
            results.append((formatted_text, formatted_text, False))
        return results
    
    # Extract and prepare all content for paraphrasing
    extraction_data = []
    for ex in examples:
        formatted_text = ex.get("abstract_formatted", ex.get("abstract", ""))
        title, section_content, section_token = extract_content_for_augmentation(
            formatted_text, ctx.special_tokens_map or {}
        )
        extraction_data.append((formatted_text, title, section_content, section_token))
    
    # Collecter tous les section_contents non vides
    contents_to_paraphrase = []
    content_indices = []
    
    for i, (_, _, section_content, _) in enumerate(extraction_data):
        if section_content and len(section_content.strip()) > 20:
            contents_to_paraphrase.append(section_content)
            content_indices.append(i)
    
    # Paraphraser en lots
    if contents_to_paraphrase:
        paraphrased_contents = paraphraser.batch_paraphrase(contents_to_paraphrase)
    else:
        paraphrased_contents = []
    
    # Reconstruct the results
    results = []
    paraphrase_idx = 0
    
    for i, (formatted_text, title, section_content, section_token) in enumerate(extraction_data):
        if i in content_indices and paraphrase_idx < len(paraphrased_contents):
            # Use paraphrased content
            paraphrased_content = paraphrased_contents[paraphrase_idx]
            paraphrase_idx += 1
            
            # Reconstruct the formatted text
            view1 = formatted_text
            view2 = reconstruct_from_augmented_content(title, paraphrased_content, section_token)
            
            results.append((view1, view2, False))
        else:
            # No paraphrase available
            results.append((formatted_text, formatted_text, False))
    
    return results


def get_tokenized_dataset_path(year, data_args, model_args):
    """Generate hierarchical directory path based on configuration"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if data_args.train_dir is None:
        raise ValueError("train_dir must be specified to get tokenized dataset path.")
    
    # Build configuration string
    config_parts = []
    
    # 1. Data augmentation strategies
    aug_str = "_".join(sorted(set(data_args.data_augmentation))) if data_args.data_augmentation else "none"
    config_parts.append(f"aug-{aug_str}")
    
    # 2. Regularization strategies
    if model_args.regularization == "ipc4":
        config_parts.append(f"reg-{model_args.regularization}")
    
    # 3. Additional views
    if data_args.additional_views:
        views_str = "+".join(sorted(data_args.additional_views))
        config_parts.append(f"views-{views_str}")
    
    # 4. Other important parameters
    config_parts.append(f"maxlen-{data_args.max_seq_length}")
    if "rand_crop" in data_args.data_augmentation:
        config_parts.append(f"crop-{data_args.crop_ratio}")
    
    # Create configuration directory name
    config_dir_name = "_".join(config_parts)
    
    # Build complete path
    tokenized_data_dir = os.path.join(current_dir, "data", "tokenized", config_dir_name)
    os.makedirs(tokenized_data_dir, exist_ok=True)
    
    return os.path.join(tokenized_data_dir, f"year_{year}")


def stream_year_chunks_shuffled(year_range, data_args, model_args, seed=None):
    """yield each sample, with true global shuffling across all years"""
    if seed is not None:
        random.seed(seed)
    
    # 1. Collect all chunk paths from all years
    all_chunk_paths = []
    for y in year_range:
        base = Path(get_tokenized_dataset_path(y, data_args, model_args))
        chunk_paths = list(base.parent.glob(f"{base.name}_chunk*.pt"))
        
        if not chunk_paths:
            print(f"[WARNING] No tokenized chunks found for year {y}, skipping...")
            continue
            
        logger.debug(f"Found {len(chunk_paths)} chunks for year {y}")
        # Add year information to chunk paths for debugging
        for p in chunk_paths:
            all_chunk_paths.append((y, p))
    
    # 2. Globally shuffle all chunks (across years)
    random.shuffle(all_chunk_paths)
    print(f"[INFO] Shuffling {len(all_chunk_paths)} chunks across {len(year_range)} years")
    
    if len(all_chunk_paths) == 0:
        print("[ERROR] No chunks found for any year!")
        return
    
    # 3. Load and yield samples in shuffled order
    total_yielded = 0
    for year, chunk_path in all_chunk_paths:
        chunk = torch.load(chunk_path, weights_only=False)
        chunk_list = list(chunk)
        
        # 4. Shuffle samples within chunk
        random.shuffle(chunk_list)
        
        # 5. Yield each sample
        for ex in chunk_list:
            yield ex
            total_yielded += 1
            
        logger.debug(f"Processed chunk {chunk_path.name} from year {year}, yielded {len(chunk_list)} samples (total: {total_yielded})")
            
        # 6. Clean up memory
        del chunk, chunk_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stream_year_chunks_true_global_shuffle(year_range, data_args, model_args, seed=None, buffer_size=50000):
    """
    Sliding window shuffle - uses memory buffer for cross-chunk sample mixing
    
    WARNING: This is NOT truly global shuffle due to memory constraints.
    It provides better mixing than chunk-level shuffle but cannot achieve
    perfect global randomization for large datasets.
    
    Args:
        buffer_size: Memory buffer size for mixing samples (increased default to 50k)
    """
    if seed is not None:
        random.seed(seed)
    
    # 1. Collect all chunk paths from all years
    all_chunk_paths = []
    for y in year_range:
        base = Path(get_tokenized_dataset_path(y, data_args, model_args))
        chunk_paths = list(base.parent.glob(f"{base.name}_chunk*.pt"))
        
        if not chunk_paths:
            print(f"[WARNING] No tokenized chunks found for year {y}, skipping...")
            continue
            
        print(f"[DEBUG] Found {len(chunk_paths)} chunks for year {y}")
        for p in chunk_paths:
            all_chunk_paths.append((y, p))
    
    # 2. Globally shuffle all chunks (across years)
    random.shuffle(all_chunk_paths)
    print(f"[INFO] True global shuffle: {len(all_chunk_paths)} chunks, buffer_size={buffer_size}")
    
    if len(all_chunk_paths) == 0:
        print("[ERROR] No chunks found for any year!")
        return
    
    # 3. Use sliding buffer to achieve true global shuffle
    buffer = []
    chunk_iterator = iter(all_chunk_paths)
    total_yielded = 0
    
    # Initialize buffer
    while len(buffer) < buffer_size:
        try:
            year, chunk_path = next(chunk_iterator)
            chunk = torch.load(chunk_path, weights_only=False)
            chunk_list = list(chunk)
            
            # Add year information to each sample (for debugging)
            for sample in chunk_list:
                if isinstance(sample, dict):
                    sample['_debug_year'] = year
                    sample['_debug_chunk'] = chunk_path.name
                buffer.append(sample)
            
            logger.debug(f"Loaded chunk {chunk_path.name} from year {year}, buffer size: {len(buffer)}")
            
            # Clean up memory
            del chunk, chunk_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except StopIteration:
            break
    
    # Initial buffer shuffle
    random.shuffle(buffer)
    print(f"[INFO] Initial buffer loaded with {len(buffer)} samples")
    
    # 4. While continuing to process chunks, yield samples from buffer
    while buffer:
        # Randomly select and yield a sample from buffer
        if len(buffer) > 1:
            idx = random.randint(0, len(buffer) - 1)
            sample = buffer.pop(idx)
        else:
            sample = buffer.pop()
        
        yield sample
        total_yielded += 1
        
        # Try to refill buffer from next chunk
        if len(buffer) < buffer_size // 2:  # Refill when buffer is less than half
            try:
                year, chunk_path = next(chunk_iterator)
                chunk = torch.load(chunk_path, weights_only=False)
                chunk_list = list(chunk)
                
                # Add new samples to buffer
                for sample in chunk_list:
                    if isinstance(sample, dict):
                        sample['_debug_year'] = year
                        sample['_debug_chunk'] = chunk_path.name
                    buffer.append(sample)
                
                # Re-shuffle buffer
                random.shuffle(buffer)
                
                logger.debug(f"Refilled buffer with chunk {chunk_path.name} from year {year}, buffer size: {len(buffer)}")
                
                # Clean up memory
                del chunk, chunk_list
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except StopIteration:
                pass  # No more chunks


def stream_year_chunks_reservoir_shuffle(year_range, data_args, model_args, seed=None, sample_rate=1.0):
    """
    Reservoir sampling based shuffle - closer to true global shuffle
    
    This approach loads ALL samples into memory first, then shuffles globally.
    More memory intensive but provides true global randomization.
    
    Args:
        sample_rate: Fraction of data to use (1.0 = all data, 0.1 = 10% sample)
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"[INFO] Loading all samples for true global shuffle (sample_rate={sample_rate})")
    
    # 1. Load ALL samples from all chunks
    all_samples = []
    total_chunks = 0
    
    for y in year_range:
        base = Path(get_tokenized_dataset_path(y, data_args, model_args))
        chunk_paths = list(base.parent.glob(f"{base.name}_chunk*.pt"))
        
        if not chunk_paths:
            print(f"[WARNING] No tokenized chunks found for year {y}, skipping...")
            continue
        
        print(f"[INFO] Loading {len(chunk_paths)} chunks from year {y}")
        
        for chunk_path in chunk_paths:
            chunk = torch.load(chunk_path, weights_only=False)
            chunk_list = list(chunk)
            
            # Apply sampling if needed
            if sample_rate < 1.0:
                sample_size = int(len(chunk_list) * sample_rate)
                chunk_list = random.sample(chunk_list, sample_size)
            
            all_samples.extend(chunk_list)
            total_chunks += 1
            
            # Clean up memory
            del chunk, chunk_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"[INFO] Loaded {len(all_samples)} samples from {total_chunks} chunks")
    
    # 2. TRUE GLOBAL SHUFFLE
    random.shuffle(all_samples)
    print(f"[INFO] Performed true global shuffle on {len(all_samples)} samples")
    
    # 3. Yield all samples
    for sample in all_samples:
        yield sample
    
    # 4. Clean up
    del all_samples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stream_year_chunks_advanced_shuffle(year_range, data_args, model_args, seed=None, shuffle_strategy="sliding_window", buffer_size=50000):
    """
    Advanced shuffle strategies
    
    Args:
        shuffle_strategy: 
            - "true_global": TRUE global shuffle - loads all data into memory (memory intensive!)
            - "sliding_window": Sliding window shuffle with buffer (balanced memory vs randomness)
            - "global": Global chunk shuffle + intra-chunk shuffle (memory efficient)
            - "balanced": Ensure uniform year distribution in each epoch
            - "block": Shuffle within years, but maintain year block structure
        buffer_size: Memory buffer size for sliding_window strategy
    """
    if seed is not None:
        random.seed(seed)
    
    if shuffle_strategy == "true_global":
        # Use reservoir sampling for true global shuffle (memory intensive)
        yield from stream_year_chunks_reservoir_shuffle(year_range, data_args, model_args, seed)
        
    elif shuffle_strategy == "sliding_window":
        # Use sliding window shuffle (balanced memory vs randomness)
        yield from stream_year_chunks_true_global_shuffle(year_range, data_args, model_args, seed, buffer_size)
        
    elif shuffle_strategy == "global":
        # Use global chunk shuffle logic
        yield from stream_year_chunks_shuffled(year_range, data_args, model_args, seed)
        
    elif shuffle_strategy == "balanced":
        # Ensure uniform distribution of year data within each time window
        year_chunks = {}
        for y in year_range:
            base = Path(get_tokenized_dataset_path(y, data_args, model_args))
            chunk_paths = list(base.parent.glob(f"{base.name}_chunk*.pt"))
            if chunk_paths:
                year_chunks[y] = chunk_paths
        
        # Check if there are valid chunks
        if not year_chunks:
            print("[WARNING] No valid chunks found for any year in balanced strategy")
            return
        
        # Create round-robin iterator to ensure uniform year distribution
        max_chunks = max(len(chunks) for chunks in year_chunks.values())
        print(f"[INFO] Balanced shuffle: {len(year_chunks)} years, max {max_chunks} chunks per year")
        
        for round_idx in range(max_chunks):
            years_this_round = list(year_chunks.keys())
            random.shuffle(years_this_round)
            
            for year in years_this_round:
                if round_idx < len(year_chunks[year]):
                    chunk_path = year_chunks[year][round_idx]
                    chunk = torch.load(chunk_path, weights_only=False)
                    chunk_list = list(chunk)
                    random.shuffle(chunk_list)
                    
                    for ex in chunk_list:
                        yield ex
                    
                    del chunk, chunk_list
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    else:
        # Default fallback to sliding_window strategy
        print(f"[WARNING] Unknown shuffle strategy '{shuffle_strategy}', falling back to sliding_window")
        yield from stream_year_chunks_true_global_shuffle(year_range, data_args, model_args, seed, buffer_size)




