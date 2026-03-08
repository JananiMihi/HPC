#!/usr/bin/env python3
"""
Amazon Dataset Converter
Downloads Amazon review dataset and converts to CSV format for the recommendation system.
Source: https://nijianmo.github.io/amazon/index.html
"""

import json
import gzip
import urllib.request
import sys
from collections import defaultdict
from pathlib import Path

# Dataset options (small categories recommended for faster processing)
DATASETS = {
    'small_books': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz',
    'small_electronics': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz',
    'small_toys': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz',
}

def download_dataset(url, output_file):
    """Download gzipped JSON dataset"""
    print(f"Downloading dataset from: {url}")
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✓ Downloaded to: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Download error: {e}")
        return False

def parse_amazon_json(gz_file, limit_users=100, limit_items=500):
    """
    Parse Amazon JSON gz file and extract User-Item-Rating triples
    
    Args:
        gz_file: Path to gzipped JSON file
        limit_users: Maximum users to include
        limit_items: Maximum items to include
    
    Returns:
        Dictionary: user_id -> item_id -> rating
    """
    print(f"\nParsing dataset (limit: {limit_users} users, {limit_items} items)...")
    
    ratings_dict = defaultdict(dict)
    user_set = set()
    item_set = set()
    line_count = 0
    
    try:
        with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Extract fields
                user_id = record.get('reviewerID')
                product_id = record.get('asin')
                rating = record.get('overall')
                
                if not all([user_id, product_id, rating]):
                    continue
                
                # Skip if limits exceeded
                if len(user_set) >= limit_users and user_id not in user_set:
                    continue
                if len(item_set) >= limit_items and product_id not in item_set:
                    continue
                
                # Add to collections
                user_set.add(user_id)
                item_set.add(product_id)
                ratings_dict[user_id][product_id] = float(rating)
                
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"  Processed {line_count} reviews... ({len(user_set)} users, {len(item_set)} items)")
                
                if len(user_set) >= limit_users and len(item_set) >= limit_items:
                    break
        
        print(f"✓ Parsed {line_count} reviews")
        print(f"  Total users: {len(user_set)}")
        print(f"  Total items: {len(item_set)}")
        return ratings_dict, sorted(user_set), sorted(item_set)
        
    except Exception as e:
        print(f"✗ Parse error: {e}")
        return {}, [], []

def create_user_item_matrix(ratings_dict, users, items):
    """
    Convert user-item-rating dict to matrix format
    
    Returns:
        List of lists: rows = users, columns = items, values = ratings (0 if not rated)
    """
    print(f"\nCreating user-item matrix ({len(users)} x {len(items)})...")
    
    user_to_idx = {uid: i for i, uid in enumerate(users)}
    item_to_idx = {iid: j for j, iid in enumerate(items)}
    
    matrix = [[0.0 for _ in range(len(items))] for _ in range(len(users))]
    
    rating_count = 0
    for user_id, items_dict in ratings_dict.items():
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        
        for item_id, rating in items_dict.items():
            if item_id not in item_to_idx:
                continue
            item_idx = item_to_idx[item_id]
            matrix[user_idx][item_idx] = rating
            rating_count += 1
    
    # Calculate sparsity
    total_cells = len(users) * len(items)
    sparsity = (total_cells - rating_count) / total_cells * 100
    
    print(f"✓ Matrix created")
    print(f"  Total ratings: {rating_count}")
    print(f"  Sparsity: {sparsity:.2f}%")
    
    return matrix

def save_to_csv(matrix, output_file):
    """Save matrix to CSV file"""
    print(f"\nSaving to CSV: {output_file}")
    
    try:
        with open(output_file, 'w') as f:
            for row in matrix:
                # Format: round to 1 decimal, comma-separated
                formatted_row = ','.join(f'{val:.1f}' if val > 0 else '0' for val in row)
                f.write(formatted_row + '\n')
        
        print(f"✓ Saved to: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Save error: {e}")
        return False

def main():
    print("=" * 70)
    print("Amazon Dataset Converter for Recommendation System")
    print("=" * 70)
    
    # Dataset selection
    print("\nAvailable datasets:")
    for i, (name, url) in enumerate(DATASETS.items(), 1):
        print(f"  {i}. {name}")
    
    dataset_choice = input("\nSelect dataset (1-3) [default: 1]: ").strip()
    dataset_choice = dataset_choice or "1"
    
    try:
        dataset_idx = int(dataset_choice) - 1
        dataset_name = list(DATASETS.keys())[dataset_idx]
        dataset_url = list(DATASETS.values())[dataset_idx]
    except (ValueError, IndexError):
        print("Invalid choice. Using small_books...")
        dataset_name = 'small_books'
        dataset_url = DATASETS['small_books']
    
    # Configuration
    num_users = int(input("Number of users [default: 100]: ").strip() or "100")
    num_items = int(input("Number of items [default: 500]: ").strip() or "500")
    
    # File paths
    script_dir = Path(__file__).parent
    gz_file = script_dir / f"{dataset_name}.json.gz"
    csv_file = script_dir / "amazon_ratings.csv"
    
    # Download
    if not gz_file.exists():
        if not download_dataset(dataset_url, str(gz_file)):
            return
    else:
        print(f"✓ Using existing file: {gz_file}")
    
    # Parse
    ratings_dict, users, items = parse_amazon_json(str(gz_file), num_users, num_items)
    if not ratings_dict:
        print("✗ Failed to parse dataset")
        return
    
    # Create matrix
    matrix = create_user_item_matrix(ratings_dict, users, items)
    
    # Save
    if save_to_csv(matrix, str(csv_file)):
        print("\n" + "=" * 70)
        print("SUCCESS! Dataset ready to use.")
        print("=" * 70)
        print(f"\nCSV file: {csv_file}")
        print(f"Users: {len(users)}")
        print(f"Items: {len(items)}")
        print(f"\nUsage in C++:")
        print(f"  RecommendationSystem rec({len(users)}, {len(items)});")
        print(f"  rec.loadRatingsFromCSV(\"amazon_ratings.csv\");")

if __name__ == "__main__":
    main()
