"""
Patent Data Analysis Script
Analyzes patent documents for:
1. Average word counts per section (excluding empty fields)
2. Word count distributions per section (including empty fields)
3. Lexical diversity using normalized unigram entropy
"""

import os
import glob
import pandas as pd
import numpy as np

from collections import Counter
from tqdm import tqdm
import argparse

import re
import warnings
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')


class PatentDataAnalyzer:
    def __init__(self, data_dir):
        """Initialize the analyzer with data directory."""
        self.data_dir = data_dir
        self.sections = ["title", "abstract", "claim", "summary", "background", 
                        "drawing", "detailed_description"]
        self.df = None
        
        # Initialize BERT model for patents
        print("Loading BERT-for-patents model...")
        self.tokenizer = None
        self.model = None
        
        # Setup device with detailed GPU information
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: {self.device} (GPU: {torch.cuda.get_device_name()})")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: {self.device} (CUDA not available)")
        
    def load_data(self, year_range=None):
        """Load patent data from feather files."""
        print("Loading patent data...")
        
        # Find all feather files
        feather_files = glob.glob(os.path.join(self.data_dir, "*.feather"))
        feather_files.sort()
        
        if year_range:
            # Filter files by year range
            filtered_files = []
            for file in feather_files:
                filename = os.path.basename(file)
                # Match year in format like "patentmap_dataset_2010.feather"
                year_match = re.search(r'dataset_(\d{4})', filename)
                if year_match:
                    year = int(year_match.group(1))
                    if year_range[0] <= year <= year_range[1]:
                        filtered_files.append(file)
            feather_files = filtered_files
        
        print(f"Found {len(feather_files)} files to analyze")
        
        # Load and concatenate data
        dfs = []
        for file in tqdm(feather_files, desc="Loading files"):
            try:
                df_temp = pd.read_feather(file)
                # Ensure all expected columns exist
                for col in self.sections:
                    if col not in df_temp.columns:
                        df_temp[col] = ""
                dfs.append(df_temp)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid data files found!")
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Fill NaN values with empty strings
        for col in self.sections:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")
        
        print(f"Loaded {len(self.df)} patents")
        return self.df
    
    def calculate_word_counts(self):
        """Calculate word counts for each section."""
        print("Calculating word counts...")
        
        for section in tqdm(self.sections, desc="Processing sections"):
            word_count_col = f'{section}_word_count'
            
            # Check if word count column already exists (from preprocessing)
            if word_count_col in self.df.columns:
                print(f"  Using existing word counts for {section}")
            else:
                # Handle special case: 'claim' in analysis but 'claims' in preprocessing
                if section == 'claim' and 'claims_word_count' in self.df.columns:
                    self.df['claim_word_count'] = self.df['claims_word_count']
                    print(f"  Using existing word counts for {section} (from claims_word_count)")
                elif section in self.df.columns:
                    # Calculate word counts if not already present
                    self.df[word_count_col] = self.df[section].apply(
                        lambda x: len(str(x).split()) if pd.notnull(x) and str(x).strip() else 0
                    )
        
        return self.df
    
    def analyze_average_word_counts(self):
        """Analyze average word counts per section (excluding empty fields)."""
        print("\n" + "="*60)
        print("1. AVERAGE WORD COUNTS PER SECTION (excluding empty fields)")
        print("="*60)
        
        results = {}
        
        for section in self.sections:
            word_count_col = f'{section}_word_count'
            if word_count_col in self.df.columns:
                # Filter out empty fields (word_count > 0)
                non_empty_mask = self.df[word_count_col] > 0
                non_empty_counts = self.df[non_empty_mask][word_count_col]
                
                if len(non_empty_counts) > 0:
                    avg_words = non_empty_counts.mean()
                    median_words = non_empty_counts.median()
                    std_words = non_empty_counts.std()
                    min_words = non_empty_counts.min()
                    max_words = non_empty_counts.max()
                    
                    results[section] = {
                        'avg': avg_words,
                        'median': median_words,
                        'std': std_words,
                        'min': min_words,
                        'max': max_words,
                        'non_empty_count': len(non_empty_counts)
                    }
                    
                    print(f"\n{section.upper()}:")
                    print(f"  Non-empty documents: {len(non_empty_counts):,}")
                    print(f"  Average words: {avg_words:.2f}")
                    print(f"  Median words: {median_words:.1f}")
                    print(f"  Std deviation: {std_words:.2f}")
                    print(f"  Range: {min_words} - {max_words}")
                else:
                    print(f"\n{section.upper()}: No non-empty documents found")
                    results[section] = None
        
        return results
    
    
    def tokenize_text(self, text):
        """Simple tokenization (split by whitespace and basic cleaning)."""
        if pd.isna(text) or not str(text).strip():
            return []
        
        # Convert to lowercase and split by whitespace
        tokens = str(text).lower().split()
        
        # Basic cleaning: remove punctuation and keep only words
        cleaned_tokens = []
        for token in tokens:
            # Remove punctuation from ends
            token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
            if token and len(token) > 1:  # Keep tokens with length > 1
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def calculate_lexical_diversity(self, max_docs_per_section=10000):
        """Calculate lexical diversity using normalized unigram entropy."""
        print("\n" + "="*60)
        print("3. LEXICAL DIVERSITY ANALYSIS (Normalized Unigram Entropy)")
        print("="*60)
        
        results = {}
        
        for section in tqdm(self.sections, desc="Calculating lexical diversity"):
            if section not in self.df.columns:
                continue
                
            print(f"\nProcessing {section}...")
            
            # Get non-empty documents
            non_empty_mask = self.df[section].apply(lambda x: bool(str(x).strip()))
            section_data = self.df[non_empty_mask][section]
            
            if len(section_data) == 0:
                print(f"  No non-empty documents found for {section}")
                continue
            
            # Sample if too many documents (for computational efficiency)
            if len(section_data) > max_docs_per_section:
                section_data = section_data.sample(n=max_docs_per_section, random_state=42)
                print(f"  Sampled {max_docs_per_section:,} documents from {len(self.df[non_empty_mask]):,}")
            else:
                print(f"  Processing all {len(section_data):,} documents")
            
            # Tokenize all documents and count word frequencies
            word_counter = Counter()
            total_tokens = 0
            
            for text in tqdm(section_data, desc=f"Tokenizing {section}", leave=False):
                tokens = self.tokenize_text(text)
                word_counter.update(tokens)
                total_tokens += len(tokens)
            
            if total_tokens == 0:
                print(f"  No tokens found for {section}")
                continue
            
            # Calculate probabilities
            vocab_size = len(word_counter)
            probabilities = np.array([count / total_tokens for count in word_counter.values()])
            
            # Calculate unigram entropy H1(s) = -Σ p(w) * log(p(w))
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Calculate maximum entropy log|Vs|
            max_entropy = np.log2(vocab_size) if vocab_size > 1 else 1
            
            # Calculate normalized entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Calculate template-ness using top-K n-gram coverage
            formulaicity = self.calculate_formulaicity(section_data)
            
            # Calculate semantic similarity using BERT and cache embeddings for later use
            print(f"  Calculating semantic similarity for {section}...")
            semantic_similarity, avg_encoded_tokens, cached_embeddings = self.calculate_semantic_similarity(
                section_data, section_name=section, cache_embeddings=True
            )
            
            results[section] = {
                'total_documents': len(section_data),
                'total_tokens': total_tokens,
                'vocabulary_size': vocab_size,
                'avg_tokens_per_doc': total_tokens / len(section_data),
                'entropy': entropy,
                'max_entropy': max_entropy,
                'normalized_entropy': normalized_entropy,
                'formulaicity': formulaicity,
                'semantic_similarity': semantic_similarity,
                'avg_encoded_tokens': avg_encoded_tokens
            }
            
            # Store embeddings in results for reuse in inter-section similarity
            if cached_embeddings is not None:
                results[section]['cached_embeddings'] = cached_embeddings
            
            print(f"  Results for {section}:")
            print(f"    Total tokens: {total_tokens:,}")
            print(f"    Vocabulary size: {vocab_size:,}")
            print(f"    Avg tokens per document: {total_tokens / len(section_data):.1f}")
            print(f"    Avg BERT-encoded tokens: {avg_encoded_tokens:.1f}")
            print(f"    Formulaicity (Top-K coverage): {formulaicity:.4f}")
            print(f"    Semantic similarity (BERT): {semantic_similarity:.4f}")
            print(f"    Unigram entropy H1: {entropy:.4f}")
            print(f"    Max entropy log|V|: {max_entropy:.4f}")
            print(f"    Normalized entropy H1_norm: {normalized_entropy:.4f}")
        
        return results
    
    def calculate_formulaicity(self, section_data, n=3, k=100):
        """
        Calculate template-ness/formulaicity using top-K n-gram coverage.
        
        Args:
            section_data: pandas Series of text documents
            n: n-gram size (default: 3 for trigrams)
            k: number of top n-grams to consider (default: 100)
            
        Returns:
            float: Coverage ratio (0-1), higher = more formulaic/template-like
        """
        from collections import defaultdict
        
        # Extract all n-grams from all documents
        ngram_counter = Counter()
        total_tokens = 0
        
        for text in section_data:
            tokens = self.tokenize_text(text)
            total_tokens += len(tokens)
            
            # Generate n-grams
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngram_counter[ngram] += 1
        
        if total_tokens == 0 or len(ngram_counter) == 0:
            return 0.0
        
        # Get top-K most frequent n-grams
        top_k_ngrams = ngram_counter.most_common(k)
        
        # Calculate total tokens covered by top-K n-grams
        covered_tokens = 0
        for ngram, count in top_k_ngrams:
            # Each n-gram occurrence covers n tokens
            covered_tokens += count * n
        
        # Calculate coverage ratio
        coverage = covered_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return min(coverage, 1.0)  # Cap at 1.0 to handle overlapping cases
    
    def calculate_inter_section_similarity(self, diversity_results, max_docs_per_section=200, n_pairs=1000):
        """Calculate cosine similarity between different sections using document sampling."""
        print("\n" + "="*60)
        print("INTER-SECTION SEMANTIC SIMILARITY ANALYSIS")
        print("="*60)
        
        # Use cached embeddings from diversity_results to avoid re-encoding
        section_embeddings = {}
        
        for section in self.sections:
            if section in diversity_results and 'cached_embeddings' in diversity_results[section]:
                embeddings = diversity_results[section]['cached_embeddings']
                if embeddings is not None and len(embeddings) > 0:
                    section_embeddings[section] = {
                        'embeddings': embeddings,
                        'count': len(embeddings)
                    }
                    print(f"Using cached embeddings for {section} ({len(embeddings)} documents)")
            else:
                print(f"Warning: No cached embeddings found for {section}, skipping...")
        
        # Calculate pairwise similarities between sections
        available_sections = list(section_embeddings.keys())
        n_sections = len(available_sections)
        similarity_matrix = np.zeros((n_sections, n_sections))
        
        for i, section1 in enumerate(available_sections):
            for j, section2 in enumerate(available_sections):
                if i == j:
                    # Use intra-section similarity from diversity_results
                    if section1 in diversity_results:
                        similarity_matrix[i, j] = diversity_results[section1]['semantic_similarity']
                    else:
                        similarity_matrix[i, j] = 0.0
                else:
                    # Calculate inter-section similarity by sampling document pairs
                    print(f"Computing similarity between {section1} and {section2}...")
                    
                    embeddings1 = section_embeddings[section1]['embeddings']
                    embeddings2 = section_embeddings[section2]['embeddings']
                    
                    similarities = []
                    for _ in range(min(n_pairs, len(embeddings1) * len(embeddings2))):
                        # Randomly sample one document from each section
                        idx1 = np.random.randint(0, len(embeddings1))
                        idx2 = np.random.randint(0, len(embeddings2))
                        
                        sim = cosine_similarity(
                            [embeddings1[idx1]], 
                            [embeddings2[idx2]]
                        )[0][0]
                        similarities.append(sim)
                    
                    similarity_matrix[i, j] = np.mean(similarities) if similarities else 0.0
        
        # Create DataFrame for better visualization
        similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=[s.title() for s in available_sections],
            columns=[s.title() for s in available_sections]
        )
        
        print("\nINTER-SECTION SIMILARITY MATRIX:")
        print("(Diagonal = intra-section similarity, Off-diagonal = inter-section similarity)")
        print("(Off-diagonal: Random pairs from different patents)")
        print(similarity_df.round(4).to_string())
        
        # Save to CSV
        similarity_df.to_csv('inter_section_similarity_matrix.csv')
        print(f"\nSaved inter-section similarity matrix to 'inter_section_similarity_matrix.csv'")
        
        return similarity_df
    
    def load_bert_model(self):
        """Load BERT-for-patents model lazily when needed."""
        if self.tokenizer is None or self.model is None:
            try:
                model_name = "anferico/bert-for-patents"
                print(f"Loading {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                
                # Move model to GPU with explicit memory management
                print(f"Moving model to {self.device}...")
                self.model.to(self.device)
                self.model.eval()
                
                # Clear GPU cache if using CUDA
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                
                print("BERT model loaded successfully!")
            except Exception as e:
                print(f"Error loading BERT model: {e}")
                print("Falling back to simplified similarity calculation...")
                return False
        return True
    
    def encode_text_batch(self, texts, max_length=512, batch_size=32, return_token_stats=False):
        """Encode multiple texts in batches for better GPU utilization."""
        if not self.load_bert_model():
            return (None, 0.0) if return_token_stats else None
            
        try:
            all_embeddings = []
            all_token_counts = []
            texts = [str(text) for text in texts if pd.notna(text) and str(text).strip()]
            
            # Process texts in batches
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", leave=False):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                # Move inputs to device (GPU/CPU)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Count actual tokens (non-padding) per document in batch
                if return_token_stats:
                    batch_token_counts = inputs['attention_mask'].sum(dim=1).cpu().numpy()
                    all_token_counts.extend(batch_token_counts)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token embeddings as sentence representations
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeddings.append(batch_embeddings)
                    
                    # Clear GPU cache periodically to prevent memory issues
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else None
            
            if return_token_stats:
                avg_tokens = np.mean(all_token_counts) if all_token_counts else 0.0
                return embeddings, avg_tokens
            else:
                return embeddings
                
        except Exception as e:
            print(f"Error in batch encoding: {e}")
            if self.device.type == 'cuda':
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                torch.cuda.empty_cache()
            return (None, 0.0) if return_token_stats else None

    def encode_text(self, text, max_length=512):
        """Encode text using BERT-for-patents model."""
        if not self.load_bert_model():
            return None
            
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )
            # Move inputs to device (GPU/CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings with memory management
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding as sentence representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Clear GPU cache periodically to prevent memory issues
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            return embeddings[0]  # Return single embedding vector
        except Exception as e:
            print(f"Error encoding text: {e}")
            if self.device.type == 'cuda':
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                torch.cuda.empty_cache()
            return None
    
    def calculate_semantic_similarity(self, section_data, max_pairs=10000, section_name=None, cache_embeddings=False):
        """Calculate average cosine similarity between document pairs in a section."""
        if len(section_data) < 2:
            return 0.0, 0.0, None
            
        # Use all sampled documents from section_data (already sampled in calculate_lexical_diversity)
        sampled_data = section_data
            
        print(f"    Encoding {len(sampled_data)} documents...")
        
        # Use batch encoding for better GPU utilization
        texts = sampled_data.tolist()
        batch_size = 16 if self.device.type == 'cuda' else 8  # Larger batches for GPU
        
        embeddings, avg_encoded_tokens = self.encode_text_batch(texts, batch_size=batch_size, return_token_stats=True)
        
        if embeddings is None or len(embeddings) < 2:
            return 0.0, 0.0, None
        
        print(f"    Average tokens encoded per document: {avg_encoded_tokens:.1f}")
        
        # Calculate pairwise similarities
        print(f"    Calculating similarities for {len(embeddings)} documents...")
        
        # Sample pairs if too many combinations
        n_docs = len(embeddings)
        total_pairs = n_docs * (n_docs - 1) // 2
        
        if total_pairs > max_pairs:
            # Randomly sample pairs
            pairs_to_sample = max_pairs
            similarities = []
            
            for _ in range(pairs_to_sample):
                i, j = np.random.choice(n_docs, 2, replace=False)
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        else:
            # Calculate all pairs
            similarity_matrix = cosine_similarity(embeddings)
            # Get upper triangle (excluding diagonal)
            similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        result_embeddings = embeddings if cache_embeddings else None
        return np.mean(similarities), avg_encoded_tokens, result_embeddings



    def generate_summary_report(self, avg_results, diversity_results):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*60)
        
        # Create summary table
        summary_data = []
        
        for section in self.sections:
            row = {'Section': section.title()}
            
            # Average word counts (non-empty)
            if section in avg_results and avg_results[section]:
                row['Avg_Words_NonEmpty'] = f"{avg_results[section]['avg']:.1f}"
                row['NonEmpty_Count'] = f"{avg_results[section]['non_empty_count']:,}"
            else:
                row['Avg_Words_NonEmpty'] = "N/A"
                row['NonEmpty_Count'] = "0"
            
            # Lexical diversity metrics
            if section in diversity_results:
                row['Vocab_Size'] = f"{diversity_results[section]['vocabulary_size']:,}"
                row['Avg_Encoded_Tokens'] = f"{diversity_results[section]['avg_encoded_tokens']:.1f}"
                row['Normalized_Entropy'] = f"{diversity_results[section]['normalized_entropy']:.3f}"
                row['Formulaicity'] = f"{diversity_results[section]['formulaicity']:.4f}"
                row['Semantic_Similarity'] = f"{diversity_results[section]['semantic_similarity']:.4f}"
            else:
                row['Vocab_Size'] = "N/A"
                row['Avg_Encoded_Tokens'] = "N/A"
                row['Normalized_Entropy'] = "N/A"
                row['Formulaicity'] = "N/A"
                row['Semantic_Similarity'] = "N/A"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\nSUMMARY TABLE:")
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('patent_analysis_summary.csv', index=False)
        print(f"\nSaved summary table to 'patent_analysis_summary.csv'")
        
        return summary_df


def main():
    parser = argparse.ArgumentParser(description="Analyze patent data")
    parser.add_argument("--data_dir", type=str, default="./preprocessed_data", 
                       help="Directory containing preprocessed feather files")
    parser.add_argument("--start_year", type=int, default=2010, 
                       help="Start year for analysis")
    parser.add_argument("--end_year", type=int, default=2018, 
                       help="End year for analysis")
    parser.add_argument("--max_docs_diversity", type=int, default=50000,
                       help="Maximum documents per section for lexical diversity calculation")

    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PatentDataAnalyzer(args.data_dir)
    
    # Load data
    year_range = None
    if args.start_year and args.end_year:
        year_range = (args.start_year, args.end_year)
    
    analyzer.load_data(year_range=year_range)
    
    # Calculate word counts
    analyzer.calculate_word_counts()
    
    # Run analyses
    print("Starting comprehensive patent data analysis...")
    
    # 1. Average word counts (excluding empty)
    avg_results = analyzer.analyze_average_word_counts()
    
    # 3. Lexical diversity
    diversity_results = analyzer.calculate_lexical_diversity(max_docs_per_section=args.max_docs_diversity)
    
    # Generate summary report
    summary_df = analyzer.generate_summary_report(avg_results, diversity_results)
    
    # Generate inter-section similarity matrix
    inter_section_similarity = analyzer.calculate_inter_section_similarity(diversity_results, max_docs_per_section=args.max_docs_diversity, n_pairs=args.max_docs_diversity//10)
    
    print(f"\nAnalysis complete! Generated reports:")
    print("- patent_analysis_summary.csv")
    print("- inter_section_similarity_matrix.csv")


if __name__ == "__main__":
    main()