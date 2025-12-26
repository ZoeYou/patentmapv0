# PatentMap v0

A patent document embedding and similarity learning framework that uses contrastive learning techniques, specifically designed for patent document understanding and retrieval tasks.


## Features

- **Patent-Specific Training**: Adapted for patent document structure including claims, descriptions, abstracts, etc.
- **Contrastive Learning Architecture**: Utilizes self-supervised contrastive learning for high-quality patent text embeddings.
- **Flexible Data Augmentation**: Supports different augmentation strategies.
- **Comprehensive Evaluation**: Built-in evaluation framework for patent representation learning tasks.

## Requirements

This project targets Python 3.8+ and CUDA-compatible GPUs for training (if you want to use GPU acceleration).
We recommend creating a conda environment as described below.

Key dependencies (pinned in `requirements.txt`) include:
- PyTorch 2.4.1
- Transformers 4.51.3
- DeepSpeed 0.17.1 (optional, for accelerated training)
- CUDA 12.x (for GPU acceleration, install via conda or appropriate wheel)

For a complete list of dependencies, see `requirements.txt`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ZoeYou/patentmapv0.git
cd patentmapv0
```

2. Install dependencies (recommended: conda + pip)

The canonical dependency list is `requirements.txt`. For GPU-enabled installations we recommend using conda to install PyTorch and FAISS first, then using pip to install Python packages:

```bash
# Create and activate a conda environment
conda create -n patentmap python=3.9
conda activate patentmap

# Install PyTorch (CUDA 12.1 example) - adjust CUDA version as needed. See PyTorch website for the correct channel and CUDA combo.
conda install -c pytorch -c nvidia pytorch==2.4.1 pytorch-cuda=12.1

# Install FAISS (CPU or GPU)
conda install -c conda-forge faiss-cpu
## For GPU-enabled FAISS (if available for your CUDA):
## conda install -c pytorch faiss-gpu

# Now install the rest of the Python dependencies
pip install -r requirements.txt
```

DeepSpeed is optional and used for accelerated or memory-efficient training. Installing DeepSpeed via `pip` is often sufficient, but sometimes the wheel must match your CUDA/PyTorch/OS combination.

Recommended quick install (pip):
```bash
pip install deepspeed==0.17.1
```

## Data Preparation

The project uses the HUPD (Harvard USPTO Patent Dataset) for training. Follow these steps to prepare the data:

### Step 1: Download the Dataset

Download the HUPD dataset from HuggingFace:

```bash
cd data

# Download the dataset (approximately 50GB compressed)
wget https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/all-years.tar

# Extract the tar file
tar -xf all-years.tar

# Rename the extracted directory to 'tar_data'
mv data tar_data
```

### Step 2: Create Raw Dataset

Run the data creation script to extract and structure patent data from the downloaded files:

```bash
python uspto_1data_creation.py
```

This script will:
- Process all `.tar.gz` files in `data/tar_data/`
- Extract patent information including: application_number, publication_number, decision, title, background, abstract, claims, summary, full_description, and ipcr_labels
- Save the processed data as feather files in `data/raw_data/`

### Step 3: Preprocess the Data

Run the preprocessing script to clean and prepare the data for training:

```bash
python uspto_2data_preprocessing.py \
    --input ./raw_data \
    --output ./preprocessed_data \
    --start_year 2010 \
    --end_year 2018
```

This script will:
- Normalize whitespace and reorganize claim sets
- Extract drawing descriptions and detailed descriptions from full descriptions
- Remove duplicate entries
- Truncate text to a maximum of 1000 words per section
- Save preprocessed data as feather files in `data/preprocessed_data/`

### Step 4: (Optional) Analyze the Dataset

You can analyze the prepared dataset using:

```bash
python data/uspto_3data_analysis.py
```

After completing these steps, your data will be ready for training in the `data/preprocessed_data/` directory.

### Step 5: Download downstream evaluation data

Before running training or evaluation scripts, you should download the downstream evaluation datasets used by the evaluation framework. These files are stored with Git LFS and can be fetched with the following commands:

```bash
cd patentmap_eval/data
git lfs install
git clone https://huggingface.co/datasets/ZoeYou/downstream
```

This will populate `patentmap_eval/data/downstream` with the datasets required for evaluation tasks (IPC-Classification, perf20, perf200).

## Training

### Quick Start

Use the provided shell script to start training with recommended settings:

```bash
bash run_example.sh
```

This script automatically:
- Sets up DeepSpeed for accelerated training
- Configures optimal training parameters for patent embeddings
- Uses the `anferico/bert-for-patents` model as the base
- Applies data augmentation strategies (e.g., dropout + section_pair_adaptive)
- Trains with the claim view as an additional perspective

### Custom Training

For more control, you can use the training script directly with custom parameters:

Note: The example below uses `deepspeed` — make sure you have DeepSpeed installed and `ds_config.json` configured for your setup.

```bash
deepspeed --num_gpus=1 train.py \
    --model_name_or_path anferico/bert-for-patents \
    --train_dir data/preprocessed_data/ \
    --output_dir results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 512 \
    --learning_rate 1e-5 \
    --max_seq_length 512 \
    --pooler_type cls \
    --temperature 0.05 \
    --fp16 \
    --deepspeed "./ds_config.json" \
    --do_train \
    --data_augmentation dropout section_pair \
    --additional_views claim
```

### Key Training Arguments

- `--model_name_or_path`: Base model (e.g., `anferico/bert-for-patents`)
- `--train_dir`: Directory containing training data
- `--pooler_type`: Pooling strategy (`cls`, `avg`, etc.)
- `--temperature`: Temperature for InfoNCE contrastive loss
- `--data_augmentation`: Augmentation strategies (e.g., `dropout`, `section_pair_adaptive`)
- `--additional_views`: Additional patent sections to use (e.g., `claim`)
- `--mlp_only_train`: Train only the MLP projection head

## Evaluation

The `baselines.py` script evaluates pretrained models on various patent-specific tasks without additional training.

### Evaluate a Pretrained Model

To evaluate any pretrained model (e.g., from HuggingFace or a local checkpoint):

```bash
python baselines.py \
    --model_name anferico/bert-for-patents \
    --output_dir ./baseline_results
```

### Evaluate Your Trained Model

After training with `train.py`, evaluate your model:

```bash
python baselines.py \
    --model_name ./results/checkpoint-final \
    --output_dir ./evaluation_results
```

### Evaluation Tasks

The evaluation framework automatically assesses models on multiple patent-specific tasks:

1. **IPC Classification**: Classifies patents into International Patent Classification categories
2. **Prior Art Search**: Retrieves relevant prior art patents (measures Recall@K, NDCG@K)
3. **Embedding Quality Metrics**:
   - Uniformity: How well embeddings spread across the representation space
   - Alignment: Semantic alignment between related patents
   - Topology: Structure preservation in embedding space

### Output

Results are saved in the specified output directory with detailed metrics for each task, including:
- Classification Precision@K (K=1,3,5)
- Retrieval metrics: Recall@K, NDCG@K (K=10,20,50,100)
- Embedding quality scores (alignment, uniformity, SSD, intra-document cohesion, etc.)
- Per-task performance breakdowns


## Project Structure

```
patentmapv0/
├── train.py              # Main training script
├── baselines.py           # Baseline evaluation script
├── utils.py              # Utility functions
├── run_example.sh        # Example training script
├── requirements.txt      # Python dependencies
├── ds_config.json        # DeepSpeed configuration
├── data/                 # Data processing scripts
│   ├── uspto_1data_creation.py
│   ├── uspto_2data_preprocessing.py
│   └── uspto_3data_analysis.py
├── patentmap/               # SimCSE module
│   ├── models.py         # Model definitions
│   └── trainers.py       # Training utilities
└── patentmap_eval/           # Evaluation framework
    ├── patenteval/       # Patent evaluation utilities
    └── data/downstream/     # Dataset for downstream evaluation

```


## Citation

If you use this code in your research, please consider citing:

```bibtex
@article{zuo2025patent,
  title={Patent Representation Learning via Self-supervision},
  author={Zuo, You and Gerdes, Kim and de La Clergerie, Eric Villemonte and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2511.10657},
  year={2025}
}
```

## License

CC BY-NC 4.0


## Contact

For questions or issues, please open an issue on GitHub.
