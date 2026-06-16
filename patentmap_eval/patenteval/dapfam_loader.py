"""
Loader for the DAPFAM patent retrieval benchmark.

Hugging Face dataset: ``datalyes/DAPFAM_patent`` (Ayaou et al. 2025,
arXiv:2506.22141). The dataset exposes three subsets:

* ``queries``   — 1,247 query patents.
* ``corpus``    — 45,336 target patents.
* ``relations`` — ~49,869 (query, target) pairs with binary
  ``relevance_score`` and a ``domain_rel`` tag in ``{IN, OUT, NC}`` where
  ``NC`` marks sampled non-citation negatives.

For our retrieval pipeline we only need the positive citations
(``relevance_score == 1``). They are pre-split into ``IN`` (same primary
IPC3 domain as the query) and ``OUT`` (cross-domain). We expose them as
three citation mappings: ``ALL`` (IN ∪ OUT), ``IN``, ``OUT``.

The function returns DataFrames whose shape mirrors the perf200 loader
used in ``evaluate.py``: ``index=patent_id`` and columns
``['title', 'abstract', 'ipc3']``. The ``ipc3`` column holds a
``set[str]`` of 3-char IPC codes, used downstream for IN/OUT candidate
pool filtering.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


SUBSETS = ("ALL", "IN", "OUT")


def load_dapfam(
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    """Load the DAPFAM benchmark from Hugging Face.

    Args:
        cache_dir: Optional directory passed through to ``datasets.load_dataset``.

    Returns:
        ``(queries_df, documents_df, citation_mapping_by_subset)`` where

        * ``queries_df``  — index=query_id, cols=['title','abstract','ipc3'].
        * ``documents_df`` — index=doc_id, cols=['title','abstract','ipc3'].
          ``ipc3`` values are ``set[str]`` of 3-char IPC codes
          (e.g. ``{'A22', 'A01'}``).
        * ``citation_mapping_by_subset`` — dict ``{subset: {q_id: [d_id, ...]}}``
          for subset ∈ {ALL, IN, OUT}. Only positives are included; ALL is the
          union of IN and OUT.
    """
    # Local import so importing this module does not require ``datasets``
    # at module load time (e.g. on a node without it installed).
    from datasets import load_dataset

    queries = load_dataset(
        "datalyes/DAPFAM_patent", "queries", split="train", cache_dir=cache_dir
    )
    corpus = load_dataset(
        "datalyes/DAPFAM_patent", "corpus", split="train", cache_dir=cache_dir
    )
    relations = load_dataset(
        "datalyes/DAPFAM_patent", "relations", split="train", cache_dir=cache_dir
    )

    queries_df = pd.DataFrame(
        {
            "title": queries["title_en"],
            "abstract": queries["abstract_en"],
            "claims_text": queries["claims_text"],
            "ipc3": [
                set(x or []) for x in queries["classifications_ipcr_list_first_three_chars_list"]
            ],
        },
        index=list(queries["query_id"]),
    )
    queries_df.index.name = "query_id"

    documents_df = pd.DataFrame(
        {
            "title": corpus["title_en"],
            "abstract": corpus["abstract_en"],
            "claims_text": corpus["claims_text"],
            "ipc3": [
                set(x or []) for x in corpus["classifications_ipcr_list_first_three_chars_list"]
            ],
        },
        index=list(corpus["relevant_id"]),
    )
    documents_df.index.name = "doc_id"

    citation_mapping_by_subset: Dict[str, Dict[str, List[str]]] = {
        s: {} for s in SUBSETS
    }
    for q_id, d_id, score, dom in zip(
        relations["query_id"],
        relations["relevant_id"],
        relations["relevance_score"],
        relations["domain_rel"],
    ):
        if float(score) != 1.0:
            continue  # skip sampled negatives (domain_rel == 'NC')
        citation_mapping_by_subset["ALL"].setdefault(q_id, []).append(d_id)
        if dom == "IN":
            citation_mapping_by_subset["IN"].setdefault(q_id, []).append(d_id)
        elif dom == "OUT":
            citation_mapping_by_subset["OUT"].setdefault(q_id, []).append(d_id)

    return queries_df, documents_df, citation_mapping_by_subset
