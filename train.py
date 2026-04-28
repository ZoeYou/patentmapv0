import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple

import torch

# show available devices
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

# clean cache to avoid memory error
torch.cuda.empty_cache()
import pandas as pd
import numpy as np


import json
import time
import dataclasses


from datasets import Dataset
from torch.utils.data import IterableDataset, Dataset as TorchDataset
from collections import defaultdict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    BertForPreTraining,
    TrainingArguments,
    set_seed,
)
from transformers.tokenization_utils_base import  PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process

from patentmap.models import BertForCL
from patentmap.trainers import CLTrainer
  
import warnings
# ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from utils import (
    EvaluateAtLogarithmicStepsCallback,
    aug_registry, Ctx,
    batch_sentence_shuffle,
    batch_ipc8,
    batch_paraphrase_augmentation,
    concatenate_title_and_section,
    extract_text_for_augmentation,
    IterableWithLen,
    get_tokenized_dataset_path,
    stream_year_chunks_advanced_shuffle,
    build_global_ipc_groups_once,
    initialize_global_paraphraser,
    create_adaptive_wrapper,
    Float32Encoder,
    shorten_task_id,
    create_probe_dataset,
    setup_patent_special_tokens,
)
import random
import gc


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # InfoNCE's arguments
    temperature: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmaxing logits in InfoNCE loss."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    mlp_only_train: bool = field(
        default=True,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_strategy: str = field(
        default="conditional",
        metadata={
            "help": "MLM strategy: 'single_view' (only first view), 'conditional' (both views when views are different, else single view)."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    span_masking: bool = field(
        default=False,
        metadata={
            "help": "Whether to use span masking for MLM."
        }
    )

    # regularization arguments
    regularization: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regularization technique to apply. Options: 'barlow_twins', 'vicreg'."
        }
    )
    regularization_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for the regularization loss."
        }
    )
    infonce_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the InfoNCE contrastive loss."
        }
    )
    # Barlow Twins specific hyperparameters
    barlow_twins_lambda: float = field(
        default=5e-3,
        metadata={
            "help": "Lambda parameter for Barlow Twins off-diagonal penalty."
        }
    )
    # VICReg specific hyperparameters  
    vicreg_force_fp32: bool = field(
        default=True,
        metadata={
            "help": "Force FP32 computation for VICReg projector (recommended for SwiGLU stability)."
        }
    )
    vicreg_variance_weight: float = field(
        default=30.0,
        metadata={
            "help": "Weight for VICReg variance loss component."
        }
    )
    vicreg_invariance_weight: float = field(
        default=15.0,
        metadata={
            "help": "Weight for VICReg invariance loss component."
        }
    )
    vicreg_covariance_weight: float = field(
        default=10.0,
        metadata={
            "help": "Weight for VICReg covariance loss component."
        }
    )
    vicreg_hidden_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Hidden dimension for VICReg MLP. If None, will be set to 3x input dimension."
        }
    )
    vicreg_output_dim: int = field(
        default=None,
        metadata={
            "help": "Output dimension for VICReg MLP. If None, will be set to vicreg_hidden_dim."
        }
    )
    vicreg_gamma: float = field(
        default=1.0,
        metadata={
            "help": "Gamma parameter for VICReg variance loss. Target minimum standard deviation for each embedding dimension."
        }
    )

    def __post_init__(self):
        # Check if regularization is valid
        valid_regularizations = ["barlow_twins", "vicreg"]
        if self.regularization is not None and self.regularization not in valid_regularizations:
            raise ValueError(f"Invalid regularization: {self.regularization}. Valid regularizations are: {', '.join(valid_regularizations)}")
        
        # Check if MLM strategy is valid
        valid_mlm_strategies = ["single_view", "conditional"]
        if self.mlm_strategy not in valid_mlm_strategies:
            raise ValueError(f"Invalid MLM strategy: {self.mlm_strategy}. Valid strategies are: {', '.join(valid_mlm_strategies)}")



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        # sys.maxsize is the maximum value of a variable of type Py_ssize_t.
        default=os.cpu_count() - 8 if os.cpu_count() > 1 else os.cpu_count(),
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # PatentMap's arguments
    train_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data directory (before tokenization)."}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    additional_views: List[str] = field(
        default_factory=list,
        metadata={
            "help": "The additional sections to use for training. Available sections: 'abstract', 'claim', 'summary', 'drawing', 'background', 'detailed_description'."
        }
    )
    start_year: Optional[int] = field(
        default=2010,
        metadata={
            "help": "The start year for filtering the hupd training data."
        }
    )
    end_year: Optional[int] = field(
        default=2018,
        metadata={
            "help": "The end year for filtering the hupd training data."
        }
    )
    shuffle_strategy: str = field(
        default="sliding_window",
        metadata={
            "help": "Shuffle strategy for training data. Options: 'true_global' (loads all data, memory intensive), 'sliding_window' (buffer-based cross-chunk mixing, recommended), 'global' (chunk-level shuffle, memory efficient), 'balanced' (year-balanced), 'block' (year-block structure)."
        }
    )
    shuffle_buffer_size: int = field(
        default=10000,
        metadata={
            "help": "Buffer size for sliding_window shuffle strategy. Larger values provide better randomness but use more memory."
        }
    )
    data_augmentation: List[str] = field(
        default_factory=lambda: ['dropout'],
        metadata={
            "help": "List of data augmentation techniques to apply. Options: 'section_pair', 'dropout', 'rand_crop', 'sentence_shuffle', 'ipc8', 'paraphrase'. Use 'adaptive_augmentation=True' for v1/v2 swapping with any method."
        }
    )
    adaptive_augmentation: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply adaptive mode (50% chance of swapping v1/v2) to all data augmentation methods (applying data augmentation method to negative examples as well)."
        }
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate to use for dropout data augmentation. Only effective when 'dropout' is in data_augmentation list."
        }
    )
    crop_ratio: float = field(
        default=0.1,
        metadata={
            "help": "Ratio of the original sentence length to use for random cropping."
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_dir is None:
            raise ValueError("Need either a dataset name or a training directory.")
        else:
            if self.train_dir is not None:
                # detect subfiles in the directory
                if os.path.isdir(self.train_dir):
                    # check if there are any files in the directory
                    if len(os.listdir(self.train_dir)) == 0:
                        raise ValueError(f"Directory {self.train_dir} is empty.")
                    else:
                        # check if there are any feather files in the directory
                        if not any(f.endswith(".feather") for f in os.listdir(self.train_dir)):
                            raise ValueError(f"No feather files found in directory {self.train_dir}.")
                else:
                    raise ValueError(f"Directory {self.train_dir} does not exist.")

        # Check if additional views are valid
        valid_views = ["abstract", "claim", "summary", "drawing", "background", "detailed_description"]
        for view in self.additional_views:
            if view not in valid_views:
                raise ValueError(f"Invalid additional view: {view}. Valid views are: {', '.join(valid_views)}")

        # check data_augmentation
        valid_augmentations = ["section_pair", "dropout", "rand_crop", "sentence_shuffle", "ipc8", "paraphrase"]
        for aug in self.data_augmentation:
            if aug not in valid_augmentations:
                raise ValueError(f"Invalid data augmentation: {aug}. Valid augmentations are: {', '.join(valid_augmentations)}")

        # check shuffle_strategy
        valid_shuffle_strategies = ["true_global", "sliding_window", "global", "balanced", "block"]
        if self.shuffle_strategy not in valid_shuffle_strategies:
            raise ValueError(f"Invalid shuffle strategy: {self.shuffle_strategy}. Valid strategies are: {', '.join(valid_shuffle_strategies)}")
        
        # check shuffle_buffer_size
        if self.shuffle_buffer_size <= 0:
            raise ValueError(f"shuffle_buffer_size must be positive, got {self.shuffle_buffer_size}")
        
        # check dropout_rate
        if not (0.0 < self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {self.dropout_rate}")
        
        # Buffer size warnings for strategies that use it
        if self.shuffle_strategy in ["true_global", "sliding_window"] and self.shuffle_buffer_size < 1000:
            print(f"[WARNING] shuffle_buffer_size={self.shuffle_buffer_size} is quite small for {self.shuffle_strategy} strategy. Consider using a larger value (e.g., 50000) for better randomness.")
        
        if self.shuffle_strategy == "true_global" and self.shuffle_buffer_size > 0:
            print(f"[INFO] Using true_global strategy - will load ALL data into memory. Buffer size setting is ignored for this strategy.")

        
@dataclass
class OurTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    do_eval_flag: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation after training."}
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use: 'steps' or 'epoch'."}
    )
    logging_steps: int = field(
        default=125, metadata={"help": "Log every X updates steps."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    eval_steps: int = field(
        default=250, metadata={"help": "Evaluate every X update steps."}
    )
    save_steps: int = field(
        default=250, metadata={"help": "Save checkpoint every X updates steps."}
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to use: 'steps' or 'epoch'."}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints."}
    )
    report_to: List[str] = field(default_factory=lambda: [])
    
    # Multi-metric support for best model selection
    metric_for_best_model_list: Optional[List[str]] = field(
        default_factory=lambda: ['eval_uniformity_global', 'eval_alignment_global'],
        metadata={
            "help": "List of metrics to combine for best model selection. "
            "If provided, this overrides metric_for_best_model. "
            "Example: ['eval_priorArt_recall@50_a2a', 'eval_ipc_cls_4_accuracy']"
        }
    )
    metric_combination_strategy: str = field(
        default="multiply",
        metadata={
            "help": "Strategy to combine multiple metrics: 'multiply', 'weighted_sum', or 'geometric_mean'. "
            "Default: 'multiply'"
        }
    )
    metric_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each metric when using weighted_sum strategy. "
            "Must have same length as metric_for_best_model_list. "
            "If None, equal weights are used."
        }
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether the metric is better when higher (True) or lower (False)."}
    )

    # Contrastive Learning Dynamic Diagnostics
    enable_diagnostics: bool = field(
        default=False,
        metadata={"help": "Whether to enable contrastive learning dynamic diagnostics during training."}
    )
    diagnostic_log_every: int = field(
        default=5,
        metadata={"help": "Log diagnostic metrics every X steps."}
    )
    diagnostic_max_batches: int = field(
        default=4,
        metadata={"help": "Maximum number of batches to use for diagnostic computation."}
    )
    diagnostic_probe_size: int = field(
        default=2000,
        metadata={"help": "Number of samples to use for diagnostic probe dataset."}
    )

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to deepspeed config file."}
    )

    def __post_init__(self):
        # Fix load_best_model_at_end compatibility issue
        # Since this setup doesn't use eval datasets during training, we need to ensure
        # that load_best_model_at_end is disabled when evaluation_strategy is "no"
        if self.evaluation_strategy == "no" and getattr(self, 'load_best_model_at_end', False):
            raise ValueError(
                "Configuration error: load_best_model_at_end=True requires evaluation during training, "
                "but evaluation_strategy='no'. Please either set evaluation_strategy to 'steps' or "
                "disable load_best_model_at_end."
            )
        
        # Set evaluation strategy before calling parent __post_init__
        # Only set evaluation strategy to steps if we actually have an eval dataset
        # Since this setup doesn't use eval datasets during training, keep it as "no"
        if hasattr(self, 'do_eval_flag') and self.do_eval_flag:
            if self.evaluation_strategy == "no":
                # Keep evaluation_strategy as "no" since we handle evaluation manually after training
                pass
        
        # First, let the parent class do its setup
        super().__post_init__()
        # If you had any additional setup to do after the parent initialization,
        # you could do it here.
        # Avoid redoing device setup here, since the parent class handles it.
        # If you used to rely on _setup_devices logic, remove it or adapt it.
        # The parent __post_init__() now correctly sets distributed_state and other attributes.


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.num_sections = len(data_args.additional_views) + 1 if data_args.additional_views else 1
    # Map the custom flag to the actual `do_eval` attribute
    training_args.do_eval = training_args.do_eval_flag
    
    # Handle evaluation strategy compatibility
    # Since we don't have an eval_dataset in this setup, ensure load_best_model_at_end is disabled
    # when evaluation_strategy is "no" to avoid configuration conflicts
    if training_args.evaluation_strategy == "no" and getattr(training_args, 'load_best_model_at_end', False):
        print("[WARNING] load_best_model_at_end=True conflicts with evaluation_strategy='no'.")
        print("[WARNING] Disabling load_best_model_at_end since no evaluation dataset is provided.")
        training_args.load_best_model_at_end = False

    # set output_dir using hyperparameters
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        f"batchsize_{training_args.per_device_train_batch_size * torch.cuda.device_count()}_da_{'_'.join(data_args.data_augmentation)}_lr-{training_args.learning_rate}_views-{'+'.join(data_args.additional_views)}_reg-{model_args.regularization}"
    )
    if model_args.do_mlm:
        training_args.output_dir += f"_mlm_weight-{model_args.mlm_weight}"
    if data_args.adaptive_augmentation:
        training_args.output_dir += "_adaptive_aug"
    if model_args.infonce_weight != 1.0:
        training_args.output_dir += f"_infonce_weight-{model_args.infonce_weight}"
    if model_args.regularization_weight != 0.1:
        training_args.output_dir += f"_reg_weight-{model_args.regularization_weight}"
    if model_args.regularization == "vicreg":
        training_args.output_dir += f"_vicreg_lambda-{model_args.vicreg_invariance_weight}_{model_args.vicreg_variance_weight}_{model_args.vicreg_covariance_weight}_hidden-{model_args.vicreg_hidden_dim}_gamma-{model_args.vicreg_gamma}"

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        print("output_dir", training_args.output_dir)
        print("os.listdir(training_args.output_dir)", os.listdir(training_args.output_dir))
        
        # detect last checkpoint from the output directory (directory named "checkpoint-xxx", where xxx is the largest number)
        for f in os.listdir(training_args.output_dir):
            if f.startswith("checkpoint-"):
                checkpoint_num = int(f.split("-")[-1])
                if last_checkpoint is None or checkpoint_num > last_checkpoint:
                    last_checkpoint = checkpoint_num
        if last_checkpoint is not None:
            last_checkpoint = os.path.join(training_args.output_dir, f"checkpoint-{last_checkpoint}")
            print("last_checkpoint", last_checkpoint)
        else:
            # if only the directory exists, but no checkpoint, then overwrite the directory
            logger.info(f"Output directory ({training_args.output_dir}) already exists but no checkpoint found.")
            logger.info(f"Files in the output directory: {os.listdir(training_args.output_dir)}")
            logger.info("Overwrite the output directory.")
            os.makedirs(training_args.output_dir, exist_ok=True)
    else:
        os.makedirs(training_args.output_dir, exist_ok=True)

    # Filter out non-serializable objects
    def filter_serializable(config):
        return {k: v for k, v in config.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
    training_args_dict = filter_serializable(vars(training_args))

    if is_main_process(training_args.local_rank):
        # project_id = f"patentmap-batchsize_{training_args.per_device_train_batch_size * torch.cuda.device_count()}"
        project_id = "diagnostic-patentmap"
        
        # Build loss weights string for task_id
        loss_weights = ""
        if model_args.infonce_weight != 1.0:
            loss_weights += f"_infonce-{model_args.infonce_weight}"
        if model_args.regularization_weight != 0.1:
            loss_weights += f"_regw-{model_args.regularization_weight}"
            
        # add timestamp to the task_id
        if model_args.regularization == "vicreg":
            task_id = f"da_{'_'.join(data_args.data_augmentation)}_views-{'+'.join(data_args.additional_views)}_mlm{model_args.do_mlm}_adaptive-aug-{data_args.adaptive_augmentation}{loss_weights}_vicreg_lambda-{model_args.vicreg_invariance_weight}-{model_args.vicreg_variance_weight}-{model_args.vicreg_covariance_weight}-{model_args.vicreg_hidden_dim}-gamma-{model_args.vicreg_gamma}_{time.strftime('%Y%m%d-%H%M%S')}"
        else:
            task_id = f"da_{'_'.join(data_args.data_augmentation)}_views-{'+'.join(data_args.additional_views)}_mlm{model_args.do_mlm}_adaptive-aug-{data_args.adaptive_augmentation}{loss_weights}_reg-{model_args.regularization}_{time.strftime('%Y%m%d-%H%M%S')}"
        
        # Shorten task_id to fit wandb's 128 character limit
        task_id = shorten_task_id(task_id)
        
        
    # Remove any existing logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        # transformers.utils.logging.enable_default_handler()
        # transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if training_args.per_device_train_batch_size > 32:
        config.gradient_checkpointing = True
    else:
        config.gradient_checkpointing = False
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Set tokenizer special tokens
    tokenizer, special_tokens_map = setup_patent_special_tokens(tokenizer, data_args.additional_views)

    # print the ids of the special tokens
    if data_args.additional_views is not None:
        for section in data_args.additional_views:
            print(f"{section}: {special_tokens_map[section]} -> {tokenizer.convert_tokens_to_ids(special_tokens_map[section])}")

    if last_checkpoint is not None:
        logger.info(f"Loading model from checkpoint {last_checkpoint}")
        if 'bert-for-patents' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                last_checkpoint,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                ignore_mismatched_sizes=True  if data_args.additional_views is not None else False
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
        model.resize_token_embeddings(len(tokenizer))
        
    else:
        # Load model from model_args.model_name_or_path
        if 'bert-for-patents' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
        
        model.resize_token_embeddings(len(tokenizer))
    
    # Ensure return_dict is True for consistent model outputs (fixes diagnostic issues)
    try:
        model.config.use_return_dict = True
        logger.info("Set model.config.use_return_dict = True for consistent model outputs")
    except AttributeError:
        # In newer transformers versions, use_return_dict might be read-only
        # or already set to True by default
        if hasattr(model.config, 'use_return_dict'):
            logger.info(f"model.config.use_return_dict is already set to: {model.config.use_return_dict}")
        else:
            logger.info("model.config.use_return_dict attribute not available - using default behavior")
    
    # Initialize the new tokens in the model ([drawing], [description] by [invention])
    if "drawing" in data_args.additional_views:
        token_id = tokenizer.convert_tokens_to_ids(special_tokens_map["drawing"])
        model.bert.embeddings.word_embeddings.weight.data[token_id] = model.bert.embeddings.word_embeddings.weight.data[tokenizer.convert_tokens_to_ids("[invention]")]
    if "detailed_description" in data_args.additional_views:
        token_id = tokenizer.convert_tokens_to_ids(special_tokens_map["detailed_description"])
        model.bert.embeddings.word_embeddings.weight.data[token_id] = model.bert.embeddings.word_embeddings.weight.data[tokenizer.convert_tokens_to_ids("[invention]")]

    # Data collator
    logger.info(f"Loading/creating tokenized dataset from {data_args.train_dir} with max_seq_length {data_args.max_seq_length}")
    train_views =  ["abstract"] + data_args.additional_views if data_args.additional_views else ["abstract"]
    target_view_indices = [i for i, view in enumerate(special_tokens_map.keys()) if view in train_views]
    

    # Build global resources once
    global_ipc_groups = {}
    global_paraphraser = None
    
    if "ipc8" in data_args.data_augmentation:
        print(f"-> Building global IPC groups for years {data_args.start_year}-{data_args.end_year}")
        global_ipc_groups = build_global_ipc_groups_once(data_args)
    if "paraphrase" in data_args.data_augmentation:
        print("-> Initializing global paraphraser...")
        global_paraphraser = initialize_global_paraphraser(data_args)

    logger.debug(f"Global resources initialized: ipc_groups={len(global_ipc_groups) if global_ipc_groups else 0}, paraphraser={global_paraphraser is not None}")

    def tokenize_and_cache_view(year, feather_dir=data_args.train_dir):
        target_columns = ["title", "ipcr_labels", "abstract"]
        if data_args.additional_views:
            target_columns.extend(data_args.additional_views)

        print(f"-> Processing feather files for {year}...")
        feather_path = os.path.join(feather_dir, f"patentmap_dataset_{year}.feather")
        
        print(f"   - Processing file: {os.path.basename(feather_path)}")
        df = pd.read_feather(feather_path, columns=target_columns)

        for section in train_views:
            if section not in df.columns:
                raise ValueError(f"Section {section} not found in dataset.")
            df[section] = df[section].fillna("").apply(lambda x: " ".join(x.split(" ")[:data_args.max_seq_length]))

        dataset = Dataset.from_pandas(df)
        print(f"-> Loaded dataset for {year} with {len(dataset)} samples.")

        def prepare_features(examples):
            """Optimized feature preparation function - batch processing same augmentation strategies, supporting title concatenation and section tokens"""
            # Create independent RNG for each worker
            worker_id = 0
            rng = random.Random(worker_id + os.getpid())
            
            valid_sections = ["abstract"]
            if data_args.additional_views:
                # Filter sections that actually exist in the data
                available_sections = [s for s in data_args.additional_views if s in df.columns]
                valid_sections.extend(available_sections)

            # Create context
            ctx = Ctx(
                rng=rng, 
                crop_ratio=data_args.crop_ratio,
                section_candidates=valid_sections,
                ipc_groups=global_ipc_groups if "ipc8" in data_args.data_augmentation else None,
                paraphraser=global_paraphraser if "paraphrase" in data_args.data_augmentation else None,
                tokenizer=tokenizer,
                special_tokens_map=special_tokens_map
            )
            
            # 1. Preprocessing: create formatted text for each sample (title + section + section_token)
            batch_size = len(examples["title"])
            enhanced_examples = []
            
            for i in range(batch_size):
                ex = {col: examples[col][i] for col in examples.keys()}
                
                # Only create formatted version for abstract (including title + [SEP])
                if "abstract" in ex:
                    title = ex.get("title", "")
                    abstract_text = ex.get("abstract", "")
                    
                    # Create formatted text containing title, abstract and section token
                    formatted_text = concatenate_title_and_section(
                        title, abstract_text, "abstract", tokenizer, special_tokens_map
                    )
                    ex["abstract_formatted"] = formatted_text
                
                # Other sections remain unchanged, no need for title and [SEP]
                # section_pair strategy will add section tokens for them when needed
                
                enhanced_examples.append(ex)
            
            # 2. Group processing by augmentation strategy
            examples_with_aug = []
            
            # Assign augmentation strategy for each sample
            for i, ex in enumerate(enhanced_examples):
                aug_name = rng.choice(data_args.data_augmentation)
                examples_with_aug.append((ex, aug_name))
            
            # 3. Group by strategy
            strategy_groups = defaultdict(list)
            for idx, (ex, aug_name) in enumerate(examples_with_aug):
                strategy_groups[aug_name].append((idx, ex))
            
            # 4. Batch process each strategy
            results = {}
            for aug_name, samples in strategy_groups.items():
                indices, batch_examples = zip(*samples)
                
                if aug_name == "sentence_shuffle":
                    batch_results = batch_sentence_shuffle(batch_examples, ctx)
                    # Apply adaptive swapping if enabled
                    if data_args.adaptive_augmentation:
                        batch_results = [
                            (v2, v1, need_dropout) if ctx.rng.random() < 0.5 else (v1, v2, need_dropout)
                            for v1, v2, need_dropout in batch_results
                        ]
                elif aug_name == "paraphrase":
                    # Use optimized batch processing
                    batch_results = batch_paraphrase_augmentation(batch_examples, ctx, global_paraphraser)
                    # Apply adaptive swapping if enabled
                    if data_args.adaptive_augmentation:
                        batch_results = [
                            (v2, v1, need_dropout) if ctx.rng.random() < 0.5 else (v1, v2, need_dropout)
                            for v1, v2, need_dropout in batch_results
                        ]
                elif aug_name == "ipc8":
                    batch_results = batch_ipc8(batch_examples, ctx)
                    # Apply adaptive swapping if enabled
                    if data_args.adaptive_augmentation:
                        batch_results = [
                            (v2, v1, need_dropout) if ctx.rng.random() < 0.5 else (v1, v2, need_dropout)
                            for v1, v2, need_dropout in batch_results
                        ]
                else:
                    # Get the base augmentation function
                    base_aug_fn = aug_registry[aug_name]
                    
                    # Apply adaptive wrapper if enabled
                    if data_args.adaptive_augmentation:
                        aug_fn = create_adaptive_wrapper(base_aug_fn)
                    else:
                        aug_fn = base_aug_fn
                    
                    batch_results = [aug_fn(ex, ctx) for ex in batch_examples]                        
                
                # Store results
                for idx, result in zip(indices, batch_results):
                    results[idx] = result

            # 5. Reorganize results in original order
            v1_list, v2_list, need_dropout = [], [], []
            for i in range(batch_size):
                if i in results:
                    v1, v2, dp_flag = results[i]
                    
                    # Quality check (based on pure text length)
                    v1_pure = extract_text_for_augmentation(v1, special_tokens_map)
                    v2_pure = extract_text_for_augmentation(v2, special_tokens_map)
                    
                    if len(v1_pure.strip().split()) < 15 or len(v2_pure.strip().split()) < 15:
                        continue
                        
                    v1_list.append(v1)
                    v2_list.append(v2)
                    need_dropout.append(dp_flag)
            
            # 6. Batch tokenization
            if not v1_list:  # If no valid samples
                return {
                    "input_ids_1": [], "attention_mask_1": [],
                    "input_ids_2": [], "attention_mask_2": [],
                    "need_dropout": []
                }
            
            tok1 = tokenizer(v1_list, truncation=True, max_length=data_args.max_seq_length,
                            padding="max_length" if data_args.pad_to_max_length else False)
            tok2 = tokenizer(v2_list, truncation=True, max_length=data_args.max_seq_length,
                            padding="max_length" if data_args.pad_to_max_length else False)

            features = {f"{k}_1": tok1[k] for k in tok1}
            features.update({f"{k}_2": tok2[k] for k in tok2})
            features["need_dropout"] = need_dropout
            return features


        tokenized = dataset.map(
            prepare_features,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            num_proc=1,
        )

        # Save only the raw dictionary of tensors
        tokenized.set_format("torch")
        ## NEW: Save in chunks
        chunk_size = 100_000  # Save every 100k examples
        num_chunks = (len(tokenized) + chunk_size - 1) // chunk_size
        save_base_path = get_tokenized_dataset_path(year, data_args, model_args) # e.g., tokenized_2004_512
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(tokenized))
            chunk = tokenized.select(range(start_idx, end_idx))
            chunk_save_path = f"{save_base_path}_chunk{chunk_idx}.pt"
            print(f"-> Saving chunk {chunk_idx} with {len(chunk)} samples to {chunk_save_path}")
            torch.save(chunk, chunk_save_path)

        # Clean up memory after processing each year
        del df, dataset, tokenized
        torch.cuda.empty_cache()
        gc.collect()

        return None
    
    # Tokenize and cache the dataset for each year
    def count_total_samples(year_range):
        total = 0
        for y in year_range:
            base = Path(get_tokenized_dataset_path(y, data_args, model_args))
            
            # Ensure parent directory exists
            if not base.parent.exists():
                print(f"[INFO] Tokenized data directory doesn't exist for year {y}, creating and tokenizing...")
                tokenize_and_cache_view(y)
            
            chunk_paths = list(base.parent.glob(f"{base.name}_chunk*.pt"))
            
            # If chunk files don't exist, tokenize first
            if not chunk_paths:
                print(f"[INFO] No tokenized chunks found for year {y}, tokenizing...")
                tokenize_and_cache_view(y)
                chunk_paths = list(base.parent.glob(f"{base.name}_chunk*.pt"))
            
            for p in chunk_paths:
                total += len(torch.load(p, weights_only=False))
        return total

    TOTAL_SAMPLES = count_total_samples(range(data_args.start_year, data_args.end_year + 1))
    print(f"[INFO] total samples = {TOTAL_SAMPLES:,}")
    
    if TOTAL_SAMPLES == 0:
        raise ValueError(f"No training samples found for years {data_args.start_year}-{data_args.end_year}. Please check your data directory and year range.")

    train_dataset = IterableWithLen(
        lambda: stream_year_chunks_advanced_shuffle(
            range(data_args.start_year, data_args.end_year + 1),
            data_args,
            model_args,
            shuffle_strategy=data_args.shuffle_strategy,
            buffer_size=data_args.shuffle_buffer_size
        ),
        TOTAL_SAMPLES,
    )
    print(f"[INFO] IterableDataset set – streaming mode enabled with '{data_args.shuffle_strategy}' shuffle strategy ✱")


    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        target_view_indices: list
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Limit batch size to avoid OOM
            if len(features) > 32:
                # Process in smaller sub-batches
                sub_batches = []
                for i in range(0, len(features), 16):
                    sub_batch = features[i:i+16]
                    sub_batches.append(self._process_batch(sub_batch))
                
                # Concatenate results
                batch = {}
                for key in sub_batches[0].keys():
                    batch[key] = torch.cat([sb[key] for sb in sub_batches], dim=0)
                return batch
            else:
                return self._process_batch(features)        
            
        def _process_batch(self, features):
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
  
            # Get batch sizes
            batch_size = len(features)
            if batch_size == 0:
                return {}

            num_views = 2  # Default to 2 views (v1, v2)
            seq_length = self.max_length
            
            # Check data format: new format vs old format
            is_new_format = 'input_ids_1' in features[0] if features[0] else False
            
            if is_new_format:
                # New format: process input_ids_1, input_ids_2 etc.
                batch = {
                    "input_ids": torch.full((batch_size, num_views, seq_length), self.tokenizer.pad_token_id, dtype=torch.long),
                    "attention_mask": torch.zeros((batch_size, num_views, seq_length), dtype=torch.long),
                }
                
                # Check if token_type_ids exist
                if any('token_type_ids_1' in feature for feature in features):
                    batch["token_type_ids"] = torch.zeros((batch_size, num_views, seq_length), dtype=torch.long)
                
                # Fill tensors - new format
                for i, feature in enumerate(features):
                    for view_idx, suffix in enumerate(['_1', '_2']):
                        for k in special_keys:
                            key_name = k + suffix
                            if key_name in feature:
                                tokens = feature[key_name]
                                if tokens is None:
                                    continue
                                if not isinstance(tokens, torch.Tensor):
                                    tokens = torch.tensor(tokens, dtype=torch.long)
                                length = min(len(tokens), seq_length)
                                batch[k][i, view_idx, :length] = tokens[:length]
            else:
                # Old format: process multi-view index format
                batch = {
                    "input_ids": torch.full((batch_size, num_views, seq_length), self.tokenizer.pad_token_id, dtype=torch.long),
                    "attention_mask": torch.zeros((batch_size, num_views, seq_length), dtype=torch.long),
                }
                if any('token_type_ids' in feature for feature in features):
                    batch["token_type_ids"] = torch.zeros((batch_size, num_views, seq_length), dtype=torch.long)

                # Fill tensors - old format
                for i, feature in enumerate(features):
                    for j, view_idx in enumerate(self.target_view_indices):
                        for k in special_keys:
                            if k in feature:
                                tokens = feature[k][view_idx] if isinstance(feature[k], list) else feature[k]
                                if tokens is None:
                                    continue
                                if not isinstance(tokens, torch.Tensor):
                                    tokens = torch.tensor(tokens, dtype=torch.long)
                                length = min(len(tokens), seq_length)
                                batch[k][i, j, :length] = tokens[:length]

            if model_args.do_mlm:
                # MLM strategy: control how MLM is applied
                if model_args.mlm_strategy == "single_view":
                    # Only create MLM masks for the first view to avoid redundancy
                    first_view_only = batch["input_ids"][:, 0:1, :]  # [B, 1, L] - only first view
                    mlm_input_ids, mlm_labels = self.mask_tokens(first_view_only)
                    
                    # Expand back to match both views format for consistency
                    batch["mlm_input_ids"] = mlm_input_ids.repeat(1, 2, 1)  # [B, 2, L]
                    batch["mlm_labels"] = mlm_labels.repeat(1, 2, 1)  # [B, 2, L]
                    
                    # But mask out the second view labels to avoid double loss computation
                    batch["mlm_labels"][:, 1, :] = -100  # Ignore second view in MLM loss
                    
                elif model_args.mlm_strategy == "conditional":
                    # Only do MLM when views are likely different (when need_dropout is False)
                    # OR fallback to single_view strategy when all samples use dropout
                    if 'need_dropout' in features[0]:
                        has_pre_augmented = any(not f.get('need_dropout', True) for f in features)
                        if has_pre_augmented:
                            # Some samples have pre-augmented different views - use both views
                            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])
                        else:
                            # All samples use dropout augmentation - fallback to single_view strategy
                            # to avoid over-masking identical views
                            logger.debug("No pre-augmented samples, falling back to single_view MLM strategy")
                            first_view_only = batch["input_ids"][:, 0:1, :]  # [B, 1, L] - only first view
                            mlm_input_ids, mlm_labels = self.mask_tokens(first_view_only)
                            
                            # Expand back to match both views format for consistency
                            batch["mlm_input_ids"] = mlm_input_ids.repeat(1, 2, 1)  # [B, 2, L]
                            batch["mlm_labels"] = mlm_labels.repeat(1, 2, 1)  # [B, 2, L]
                            
                            # Mask out the second view labels to avoid double loss computation
                            batch["mlm_labels"][:, 1, :] = -100  # Ignore second view in MLM loss
                    else:
                        # Fallback: use both views with standard probability (should rarely happen)
                        batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])
                else:
                    raise ValueError(f"Unknown MLM strategy: {model_args.mlm_strategy}")
                    
                # Debug info about MLM masking
                mlm_tokens_count = (batch['mlm_labels'] != -100).sum().item()
                total_tokens = batch['mlm_labels'].numel()
                masking_rate = mlm_tokens_count / total_tokens if total_tokens > 0 else 0
                logger.debug(f"MLM strategy: {model_args.mlm_strategy}, MLM tokens: {mlm_tokens_count}/{total_tokens} ({masking_rate:.3f})")

            return batch
        
        if model_args.span_masking: 
            def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                SpanBERT-style random span masking.
                Ensures all special tokens (including custom patent section tokens) are NOT masked.
                """
                labels = inputs.clone()
                batch_size, seq_len = inputs.size()
                mask_token_id = self.tokenizer.mask_token_id
                vocab_size = len(self.tokenizer)
                
                # Create comprehensive special tokens set
                all_special_token_ids = set()
                
                # Add standard special tokens
                if hasattr(self.tokenizer, 'all_special_ids'):
                    all_special_token_ids.update(self.tokenizer.all_special_ids)
                else:
                    standard_specials = [
                        self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, 
                        self.tokenizer.pad_token_id, self.tokenizer.mask_token_id,
                        self.tokenizer.unk_token_id
                    ]
                    all_special_token_ids.update([t for t in standard_specials if t is not None])
                
                # Add our custom patent section tokens
                if hasattr(self.tokenizer, 'additional_special_tokens') and self.tokenizer.additional_special_tokens:
                    custom_token_ids = [
                        self.tokenizer.convert_tokens_to_ids(token) 
                        for token in self.tokenizer.additional_special_tokens
                    ]
                    all_special_token_ids.update(custom_token_ids)
                
                # Set parameters
                p = 0.2  # geometric distribution parameter
                max_span_length = 10
                mask_ratio = self.mlm_probability
                
                for i in range(batch_size):
                    # Create mask for special tokens in this sequence
                    special_tokens_mask = torch.zeros(seq_len, dtype=torch.bool, device=inputs.device)
                    for token_id in all_special_token_ids:
                        if token_id is not None and token_id >= 0:
                            special_tokens_mask |= (inputs[i] == token_id)
                    
                    num_to_mask = int(mask_ratio * seq_len)
                    covered = set()
                    
                    while len(covered) < num_to_mask:
                        span_len = min(np.random.geometric(p), max_span_length)
                        span_start = np.random.randint(1, seq_len - span_len - 1)
                        span_end = span_start + span_len
                        
                        # Check if span overlaps with existing coverage or special tokens
                        if any(pos in covered for pos in range(span_start, span_end)):
                            continue
                        if any(special_tokens_mask[pos] for pos in range(span_start, span_end)):
                            continue  # Skip spans containing special tokens
                        
                        for pos in range(span_start, span_end):
                            if pos >= seq_len or pos in covered:
                                continue
                            # IMPORTANT: Store original token in labels BEFORE modifying inputs
                            labels[i, pos] = inputs[i, pos]
                            
                            # Now modify the input token
                            prob = np.random.rand()
                            if prob < 0.8:
                                inputs[i, pos] = mask_token_id
                            elif prob < 0.9:
                                inputs[i, pos] = torch.randint(vocab_size, (1,), dtype=torch.long)
                            # else: keep original
                            covered.add(pos)
                    
                    # For positions not masked, ignore in loss
                    for j in range(seq_len):
                        if j not in covered:
                            labels[i, j] = -100
                
                return inputs, labels
        else:
            def mask_tokens(
                self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
                Handles 3D tensors: (batch_size, num_views, seq_len)
                Ensures all special tokens (including custom patent section tokens) are NOT masked.
                """
                shape = inputs.shape  # (batch_size, num_views, seq_len)
                inputs = inputs.reshape(-1, shape[-1])  # flatten: (batch_size * num_views, seq_len)
                labels = inputs.clone()
                
                # Create comprehensive special tokens set
                all_special_token_ids = set()
                
                # Add standard special tokens
                if hasattr(self.tokenizer, 'all_special_ids'):
                    all_special_token_ids.update(self.tokenizer.all_special_ids)
                else:
                    # Fallback for older tokenizer versions
                    standard_specials = [
                        self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, 
                        self.tokenizer.pad_token_id, self.tokenizer.mask_token_id,
                        self.tokenizer.unk_token_id
                    ]
                    all_special_token_ids.update([t for t in standard_specials if t is not None])
                
                # Add our custom patent section tokens
                if hasattr(self.tokenizer, 'additional_special_tokens') and self.tokenizer.additional_special_tokens:
                    custom_token_ids = [
                        self.tokenizer.convert_tokens_to_ids(token) 
                        for token in self.tokenizer.additional_special_tokens
                    ]
                    all_special_token_ids.update(custom_token_ids)
                
                # Create mask for special tokens
                special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
                for token_id in all_special_token_ids:
                    if token_id is not None and token_id >= 0:
                        special_tokens_mask |= (inputs == token_id)
                
                # Create probability matrix for masking
                probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)
                
                # Exclude special tokens from masking (set probability to 0)
                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                
                # Sample tokens to mask
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = -100  # We only compute loss on masked tokens
                
                # 80% of the time, replace masked input tokens with [MASK]
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
                inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                
                # 10% of the time, replace masked input tokens with random word
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
                inputs[indices_random] = random_words[indices_random]
                
                # Restore to original 3D shape
                inputs = inputs.reshape(shape)
                labels = labels.reshape(shape)
                
                # The rest of the time (10% of the time) we keep the masked input tokens unchanged
                return inputs, labels
            

    data_collator = OurDataCollatorWithPadding(
        tokenizer=tokenizer,
        target_view_indices=target_view_indices,  # your selected indices
        padding="max_length" if data_args.pad_to_max_length else False,
        max_length=data_args.max_seq_length,
        mlm=(model_args.do_mlm),
        mlm_probability=data_args.mlm_probability,
    )

    # Create probe dataset for diagnostics if enabled
    probe_dataset = None
    if training_args.enable_diagnostics:
        logger.info("Diagnostic mode enabled - creating probe dataset...")
        probe_dataset = create_probe_dataset(
            data_args, model_args, tokenizer, special_tokens_map, get_tokenized_dataset_path, training_args.diagnostic_probe_size
        )
        if probe_dataset is None:
            logger.warning("Failed to create probe dataset. Disabling diagnostics.")
            training_args.enable_diagnostics = False
    else:
        logger.info("Diagnostic mode disabled")

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_args=model_args,
        data_args=data_args,
        # Diagnostic parameters
        probe_dataset=probe_dataset,
        enable_diagnostics=training_args.enable_diagnostics,
        diagnostic_log_every=training_args.diagnostic_log_every,
        diagnostic_max_batches=training_args.diagnostic_max_batches,
    )

    eval_callback = EvaluateAtLogarithmicStepsCallback(tolerance=1e-3)
    eval_callback.trainer = trainer  # Assign the trainer reference manually
    trainer.callback_handler.add_callback(eval_callback)


    if training_args.do_train:
        logging.info("*** Train ***")
        logging.info(f"Tokenizer special tokens: {tokenizer.additional_special_tokens}")
        logging.info(f"Tokenizer special tokens ids: {[tokenizer.convert_tokens_to_ids(tok) for tok in tokenizer.additional_special_tokens]}")
        
        # Check if the dataset contains samples
        if TOTAL_SAMPLES == 0:
            logging.error("No training samples found. Please check your data directory and year range.")
            raise ValueError("No training samples found. Exiting.")
        
        # Try to get a sample for logging
        sample = None
        try:
            dataset_iter = iter(train_dataset)
            sample = next(dataset_iter)
        except StopIteration:
            raise ValueError("Dataset iterator is empty - no training samples found. Check your data directory and year range.")
        except Exception as e:
            raise RuntimeError(f"Failed to load sample from dataset: {e}") from e
        
        # Log sample data (new format only)
        if 'input_ids_1' in sample and 'input_ids_2' in sample:
            logging.info(f"  View 1: {tokenizer.decode(sample['input_ids_1'])}")
            logging.info(f"  View 2: {tokenizer.decode(sample['input_ids_2'])}")
            logging.info(f"  Need dropout: {sample.get('need_dropout', 'N/A')}")
            logging.info(f"  Input IDs 1: {sample['input_ids_1']}")
            logging.info(f"  Input IDs 2: {sample['input_ids_2']}")
        else:
            logging.warning("  Unexpected data format in sample")
            logging.info(f"  Available keys: {list(sample.keys())}")



        # Explicitly log what you load
        if last_checkpoint:
            logger.info(f"Resuming training from {last_checkpoint}")

            # IMPORTANT: DO NOT manually load model optimizer here, let the trainer handle this
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            # # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            # trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            trainer_state_dict = dataclasses.asdict(trainer.state)
            trainer_state_path = os.path.join(training_args.output_dir, "trainer_state.json")
            with open(trainer_state_path, "w") as f:
                json.dump(trainer_state_dict, f, indent=2, sort_keys=True, cls=Float32Encoder)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate after training ***")
        results = trainer.evaluate(final_eval=True)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
    return results



if __name__ == "__main__":
    main()