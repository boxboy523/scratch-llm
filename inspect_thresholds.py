"""
Utility to sample documents near the quality threshold for each source.
Helps in manual verification and tuning of the CUT_PERCENTILE.
"""

import numpy as np
import torch
from datasets import load_dataset
from data.preprocess import get_quality_score
from data.dataset import get_filtered_culturax, compute_source_threshold
import config


def get_samples_near_threshold(dataset, threshold: float, n_samples: int = 10, tolerance: float = 0.05):
    """
    Samples documents whose quality score is within (threshold - tolerance, threshold + tolerance).
    Inputs: dataset, threshold (float), n_samples (int), tolerance (float)
    Outputs: list of sampled texts (list[str])
    """
    samples = []
    # Shuffle to get diverse samples
    shuffled_ds = dataset.shuffle(seed=config.SEED)
    
    for item in shuffled_ds:
        score = get_quality_score(item["text"])
        if threshold - tolerance <= score <= threshold + tolerance:
            samples.append((score, item["text"][:500])) # Show first 500 chars
        
        if len(samples) >= n_samples:
            break
            
    return samples


def analyze_thresholds():
    """
    Computes thresholds for each source and prints samples near them.
    """
    sources = {
        "Wikipedia": (load_dataset(config.WIKI_DATASET, config.WIKI_KO, split="train"), config.WIKI_CUT_RATIO),
        "NamuWiki": (load_dataset(config.NAMUWIKI_DATASET, split="train"), config.NAMU_CUT_RATIO),
        "CulturaX": (get_filtered_culturax(), config.CULTURAX_CUT_RATIO)
    }
    
    for name, (ds, ratio) in sources.items():
        threshold = compute_source_threshold(ds, config.SAMPLE_SIZE, ratio)
        print(f"\n{'='*20} Source: {name} (Ratio: {ratio}, Threshold: {threshold:.4f}) {'='*20}")
        
        samples = get_samples_near_threshold(ds, threshold)
        for i, (score, text) in enumerate(samples):
            print(f"\n--- Sample {i+1} (Score: {score:.4f}) ---")
            print(text.strip())
            print("-" * 50)


if __name__ == "__main__":
    analyze_thresholds()
