import re
import string
from pathlib import Path
from typing import List

import nltk
from nltk.corpus import brown
import numpy as np
from collections import Counter
import ssl



# Function to compute PMI
def compute_pmi(co_occurrences, word_counts, total_count):
    pmi_values = {}

    for pair, count in co_occurrences.items():
        w1, w2 = pair
        p_x1 = word_counts[w1]
        p_x2 = word_counts[w2]
        p_xy = count

        pmi = np.log((p_xy * total_count) / (p_x1 * p_x2))
        pmi_values[pair] = pmi

    return pmi_values

# Function to compute PPMI
def compute_ppmi(co_occurrences, word_counts, total_count):
    ppmi_values = {}

    for pair, count in co_occurrences.items():
        w1, w2 = pair
        p_x1 = word_counts[w1]
        p_x2 = word_counts[w2]
        p_xy = count

        # if p_xy > 0:  # Avoid log(0)
        ppmi = np.log((p_xy * total_count) / (p_x1 * p_x2))
        ppmi_values[pair] = max(ppmi, 0)  # Only keep non-negative PMI

    return ppmi_values


def step4 (corpus: List[str]):

    # Minimum word occurrence threshold
    threshold = 10

    # Count individual word frequencies
    word_counts = Counter(corpus)

    # Filter out words that occur less than the threshold
    filtered_words = {word for word, count in word_counts.items() if count >= threshold}

    # Initialize co-occurrence counts
    co_occurrences = Counter()

    # Count co-occurrences for successive pairs of words
    for i in range(len(corpus) - 1):
        if corpus[i] in filtered_words and corpus[i + 1] in filtered_words:
            pair = (corpus[i], corpus[i + 1])
            co_occurrences[pair] += 1

    # Total number of words in the corpus
    total_count = len(corpus)

    # Compute PMI values
    pmi_result = compute_pmi(co_occurrences, word_counts, total_count)

    # Get the top 20 pairs with highest and lowest PMI values
    top_20_pmi = sorted(pmi_result.items(), key=lambda item: item[1], reverse=True)[:20]
    lowest_20_pmi = sorted(pmi_result.items(), key=lambda item: item[1])[:20]

    # Print the results
    print("Top 20 PMI values:")
    for pair, pmi in top_20_pmi:
        print(f"{pair}: {pmi:.4f}")

    print("\nLowest 20 PMI values:")
    for pair, pmi in lowest_20_pmi:
        print(f"{pair}: {pmi:.4f}")

    # Compute PPMI values
    ppmi_result = compute_ppmi(co_occurrences, word_counts, total_count)

    # Get the top 20 and lowest 20 pairs with PPMI values
    top_20_ppmi = sorted(ppmi_result.items(), key=lambda item: item[1], reverse=True)[:20]
    lowest_20_ppmi = sorted(ppmi_result.items(), key=lambda item: item[1])[:20]

    # Print the results
    print("\nTop 20 PPMI values:")
    for pair, ppmi in top_20_ppmi:
        print(f"{pair}: {ppmi:.4f}")

    print("\nLowest 20 PPMI values:")
    for pair, ppmi in lowest_20_ppmi:
        print(f"{pair}: {ppmi:.4f}")


if __name__ == '__main__':
    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    #
    # nltk.download('brown')

    # Get the Brown corpus words
    not_punctuation = r"(\w+('s)?)|<s>|</s>|\$\d+"

    corpus = list(brown.words())
    corpus_removed_punctuation = [word for word in corpus if re.search(not_punctuation, word)]
    step4(corpus_removed_punctuation)

    lines = Path("brown_100.txt").read_text().splitlines()
    sentences = [line.strip().split() for line in lines]
    corpus = [word for sentence in sentences for word in sentence if re.search(not_punctuation, word)]
    print("====" * 30)
    step4(corpus)
    # assert all(is_valid(word) for sentence in sentences for word in sentence)
    #
    # counts = Counter(word for sentence in sentences for word in sentence)