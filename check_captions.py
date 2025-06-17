import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Read the TSV file
df = pd.read_csv('imagenet_gemma3_12b.tsv', sep='\t', header=None, names=['filename', 'caption'])

# Initialize Qwen3 0.6B tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

# Tokenize all captions and get their lengths (batch processing for speed)
captions_list = df['caption'].tolist()
tokenized = tokenizer(captions_list, add_special_tokens=True, padding=False, truncation=False)
caption_lengths = [len(tokens) for tokens in tokenized['input_ids']]

caption_lengths = np.array(caption_lengths)

# Calculate statistics
max_length = np.max(caption_lengths)
min_length = np.min(caption_lengths)
mean_length = np.mean(caption_lengths)
median_length = np.median(caption_lengths)

# Print statistics
print(f"Caption Length Statistics:")
print(f"Max: {max_length}")
print(f"Min: {min_length}")
print(f"Mean: {mean_length:.2f}")
print(f"Median: {median_length:.2f}")
print(f"Total captions: {len(caption_lengths)}")

# Plot histogram of caption lengths
plt.figure(figsize=(10, 6))
plt.hist(caption_lengths, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Caption Length (tokens)')
plt.ylabel('Frequency')
plt.title('Distribution of Caption Lengths (Tokenized with Qwen3-0.6B)')
plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.2f}')
plt.axvline(median_length, color='green', linestyle='--', label=f'Median: {median_length:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Additional plot: cumulative distribution
plt.figure(figsize=(10, 6))
sorted_lengths = np.sort(caption_lengths)
cumulative_pct = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
plt.plot(sorted_lengths, cumulative_pct)
plt.xlabel('Caption Length (tokens)')
plt.ylabel('Cumulative Percentage')
plt.title('Cumulative Distribution of Caption Lengths')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
