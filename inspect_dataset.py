"""
Utility to sample 100 documents from the final interleaved dataset.
Verifies that mixing, filtering, and cleaning are working as expected.
"""

import config
from data.dataset import load_multi_source_dataset


def inspect_dataset_samples(n_samples: int = 100):
    """
    Loads the multi-source dataset and prints a summary of samples.
    Inputs: n_samples (int)
    """
    print(f"🚀 Loading interleaved dataset (STREAMING={config.STREAMING})...")
    dataset = load_multi_source_dataset()
    
    print(f"📊 Sampling {n_samples} items for inspection...\n")
    
    count = 0
    # Works for both regular Dataset and IterableDataset
    for item in dataset:
        text = item["text"]
        count += 1
        
        # Print sample info
        print(f"[{count:03d}] Length: {len(text):4d} | Content: {text[:150].replace(chr(10), ' ')}...")
        
        if count >= n_samples:
            break
            
    print(f"\n✅ Inspection complete. Sampled {count} items.")


if __name__ == "__main__":
    inspect_dataset_samples(100)
