import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def step1():
    _VALID_WORD_REGEX = r"(\w+('s)?)|<s>|</s>|\$\d+"
    _REMOVABLE_PUNCTUATION_REGEX = r"(``|''|--|[\.,\(\):])"

    is_valid = re.compile(_VALID_WORD_REGEX).match
    lines = Path("brown_100.txt").read_text().splitlines()

    sentences = [re.sub(_REMOVABLE_PUNCTUATION_REGEX,"",  line).split() for line in lines]
    assert all(is_valid(word) for sentence in sentences for word in sentence)

    counts = Counter(word for sentence in sentences for word in sentence)
    pairs = Counter(
        (w1, w2)
        for sentence in sentences for w1, w2 in zip(sentence, sentence[1:])
        if counts[w1] >= 10 and counts[w2] >= 10
    )
    n = len([w for w, c in counts.items() if c >= 10])

    pmi = lambda p: math.log(pairs[p] * n / (counts[p[0]] * counts[p[0]]))

    pairs_sorted = sorted(pairs, key=pmi)
    print("=" * 80)
    print("twenty best:")
    for w1, w2 in pairs_sorted[-20:][::-1]:
        print(f"'{w1} {w2}' : {pmi((w1, w2))}")

    print("=" * 80)
    print("twenty worst:")
    for w1, w2 in pairs_sorted[:20]:
        print(f"'{w1} {w2}' : {pmi((w1, w2))}")

    print(f"total pairs: {len(pairs_sorted)}")

'''
step2:

Unigram models operate under the independence assumption that the occurrence of a word is not influenced by 
the presence of other words. While this simplifies modeling and calculation, it often fails to capture significant 
linguistic phenomena, leading to limitations in performance.

Validity of the Independence Assumption
Ignoring Context: Unigram models treat each word in isolation, disregarding contextual dependencies. For example, 
the words "bank" (financial institution) and "riverbank" can lead to different meanings in context, 
which unigrams cannot capture.

Common Sequences: Words that often appear together, such as "peanut butter" or "ice cream," may be represented 
separately in a unigram model, losing the inherent relationship. A unigram might underestimate the likelihood 
of seeing "peanut" immediately followed by "butter" compared to a bigram model.

Polysemy and Homonymy: Consider the word "bark," which can mean the sound of a dog or the outer layer of a tree. 
In a unigram model, it fails to distinguish between these meanings as it lacks semantic context.

These examples demonstrate that while unigram models are computationally efficient, the independence assumption 
limits their ability to accurately represent language’s complexities and nuances.


Negative PMI values illustrate the limitations of the independence assumption in unigram models by revealing 
how the model fails to capture meaningful relationships between words. Here’s a breakdown of the concept:

1. Independence Assumption in Unigram Models
Unigram models treat each word as an independent entity. The probabilities of words occurring are considered 
without regard to their context or the presence of surrounding words. This means that:

Words are evaluated based only on their individual frequencies, not on their co-occurrence with other words.

2. Understanding PMI
Pointwise Mutual Information (PMI) measures the degree of association between two words compared to what would 
be expected if they were independently distributed. The formula is:

[
PMI(w_1, w_2) = \log\left(\frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}\right)
]

3. Negative PMI Implications
Low Co-occurrence: A negative PMI indicates that the co-occurrence of the two words is less frequent than would 
be expected if they were independent. This suggests a lack of meaningful relationship between them.
Violation of Independence: The presence of negative PMI values contradicts the independence assumption since 
it demonstrates that word pairs often don't occur together, contrary to what an independent model would suggest.

4. Examples:
"As a": If the phrase has a negative PMI, it suggests that "as" does not frequently appear right before "a," 
highlighting a syntactic relationship that is not captured by solely looking at individual word frequencies.
"Bank": The word "bank" could have multiple meanings (financial vs. riverbank). If its co-occurrences with 
specific words yield negative PMI, it reflects that the model lacks the ability to account for context in 
which "bank" appears.

Conclusion
Negative PMI values highlight the limitations of unigram models by demonstrating how they overlook important 
contextual information and language structures. This results in an incomplete understanding of word relationships, 
as significant connections could be masked by just counting individual word occurrences. Thus, negative PMI 
effectively illustrates the need for more sophisticated models that take context into account, such as bigram or 
higher-order n-gram models.

'''

def step3():
    ...

if __name__ == '__main__':
    step1()


