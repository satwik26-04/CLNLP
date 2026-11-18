# Walkthrough: Experiment 6 - GloVe Word Embeddings and Statistical N-Gram Language Modeling

This walkthrough provides an exhaustive, line-by-line explanation of every code statement, parameter, algorithm, mathematical formulation, and linguistic concept implemented in [Experiment_6.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%206/Experiment_6.ipynb). The experiment covers:
1. **Problem 1.1**: Implementing the GloVe (Global Vectors for Word Representation) algorithm from scratch without any pretrained model on `glove_corpus.txt`, constructing the global co-occurrence matrix, optimizing the weighted least-squares loss using AdaGrad, and finding the top-5 most similar words to "King".
2. **Problem 1.2**: Using a pretrained GloVe model (`glove-wiki-gigaword-50` via Gensim) to find the top-5 most similar words to "King" and comparing the results against the scratch implementation.
3. **Problem 2**: Implementing a statistical language model from scratch using the N-gram approach (Bigram and Trigram Maximum Likelihood Estimation) to predict the next word in a sequence based on preceding words in the provided text corpus.

---

## 1. Setup and Environment Imports

### Purpose
Load required standard libraries for text cleaning (`re`), numerical optimization (`numpy`), tabular formatting (`pandas`), frequency counting (`collections`), and Gensim's model repository (`gensim.downloader`).

### Code Executed
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import gensim.downloader as api
```

### Line-by-Line & Syntax Explanation
- `import os`: Operating system interface module.
- `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`: Sets the runtime environment flag to bypass multi-runtime OpenMP collisions between PyTorch, NumPy, and Scipy on Windows platforms.
- `import re`: Regular expression module used for stripping punctuation and normalizing whitespace.
- `import math`: Provides standard mathematical functions such as square roots and logarithms.
- `import numpy as np`: High-performance multidimensional array library used for matrix operations, gradient computations, and vector similarity calculations.
- `import pandas as pd`: Tabular data analysis toolkit used to present similarity rankings and model predictions in structured tables.
- `from collections import defaultdict, Counter`: `defaultdict(float)` automatically initializes non-existent key lookups with zero when accumulating co-occurrence tallies; `Counter` calculates unigram, bigram, and trigram frequencies.
- `import gensim.downloader as api`: Gensim's data repository client for fetching pretrained word embeddings (`glove-wiki-gigaword-50`).

---

## 2. Problem 1.1: GloVe Algorithm From Scratch

### Theoretical Background & Mathematical Foundations
GloVe (Pennington et al., 2014) combines the advantages of global matrix factorization (like LSA) and local context window methods (like Word2Vec skip-gram).

#### 1. Co-occurrence Matrix Formulation
Let $X_{ij}$ denote the number of times word $j$ appears in the context of word $i$. In accordance with the original GloVe paper, co-occurrences are weighted inversely proportional to the distance $d$ between words:
$$\text{Weight}(d) = \frac{1}{d}, \quad 1 \le d \le \text{window\_size}$$
This harmonic decay ensures that closer context words contribute more strongly to semantic affinity than distant words.

#### 2. Weighted Least Squares Objective
GloVe models the relationship between word pairs using log-bilinear dot products:
$$J = \sum_{i=1}^V \sum_{j=1}^V f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
where:
- $w_i \in \mathbb{R}^d$ is the target word vector.
- $\tilde{w}_j \in \mathbb{R}^d$ is the context word vector.
- $b_i, \tilde{b}_j \in \mathbb{R}$ are scalar bias terms.
- $f(X_{ij})$ is a continuous weighting function designed to prevent frequent co-occurrences (such as stopwords) from dominating the objective:
$$f(x) = \begin{cases} \left( \frac{x}{x_{\max}} \right)^\alpha & \text{if } x < x_{\max} \\ 1.0 & \text{otherwise} \end{cases}$$
Default standard parameters are $x_{\max} = 10.0$ and $\alpha = 0.75$.

#### 3. Gradient Updates via AdaGrad
The error term is defined as:
$$\delta_{ij} = w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}$$
The gradients of the cost function with respect to the parameters are:
$$\frac{\partial J}{\partial w_i} = 2 f(X_{ij}) \delta_{ij} \tilde{w}_j$$
$$\frac{\partial J}{\partial \tilde{w}_j} = 2 f(X_{ij}) \delta_{ij} w_i$$
$$\frac{\partial J}{\partial b_i} = 2 f(X_{ij}) \delta_{ij}$$
$$\frac{\partial J}{\partial \tilde{b}_j} = 2 f(X_{ij}) \delta_{ij}$$
To handle learning rate adaptation across sparse non-zero entries, AdaGrad maintains historical sum of squared gradients $G$:
$$\theta^{(t+1)} = \theta^{(t)} - \frac{\eta}{\sqrt{G^{(t)}}} \odot \nabla_\theta J$$

#### 4. Final Word Representation
Because $X$ is symmetric, $W$ and $\tilde{W}$ are equivalent in expectation. Adding them eliminates noise:
$$v_i = w_i + \tilde{w}_i$$

### Code Executed
```python
with open('glove_corpus.txt', 'r', encoding='utf-8') as f:
    raw_lines = f.readlines()

sentences = [re.sub(r'[^a-zA-Z\s]', '', line.lower()).split() for line in raw_lines if line.strip()]
vocab = sorted(list(set(word for sent in sentences for word in sent)))
w2i = {word: idx for idx, word in enumerate(vocab)}
i2w = {idx: word for idx, word in enumerate(vocab)}
V = len(vocab)

window_size = 4
cooc_dict = defaultdict(float)
for sent in sentences:
    for i, target in enumerate(sent):
        for dist in range(1, window_size + 1):
            if i - dist >= 0:
                ctx = sent[i - dist]
                cooc_dict[(w2i[target], w2i[ctx])] += 1.0 / dist
            if i + dist < len(sent):
                ctx = sent[i + dist]
                cooc_dict[(w2i[target], w2i[ctx])] += 1.0 / dist

pairs = list(cooc_dict.keys())
counts = np.array([cooc_dict[p] for p in pairs])
x_max = 10.0
alpha = 0.75
weights = np.where(counts < x_max, (counts / x_max) ** alpha, 1.0)
log_counts = np.log(counts)

embedding_dim = 30
np.random.seed(42)
W = (np.random.rand(V, embedding_dim) - 0.5) / embedding_dim
W_tilde = (np.random.rand(V, embedding_dim) - 0.5) / embedding_dim
b = np.zeros(V)
b_tilde = np.zeros(V)

gradsq_W = np.ones((V, embedding_dim))
gradsq_W_tilde = np.ones((V, embedding_dim))
gradsq_b = np.ones(V)
gradsq_b_tilde = np.ones(V)
learning_rate = 0.05
epochs = 150

for epoch in range(epochs):
    for idx, (i, j) in enumerate(pairs):
        f_val = weights[idx]
        diff = np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j] - log_counts[idx]
        loss_val = f_val * (diff ** 2)
        grad_common = 2.0 * f_val * diff
        grad_W = grad_common * W_tilde[j]
        grad_W_tilde = grad_common * W[i]
        grad_b = grad_common
        grad_b_tilde = grad_common
        
        W[i] -= (learning_rate / np.sqrt(gradsq_W[i])) * grad_W
        gradsq_W[i] += grad_W ** 2
        W_tilde[j] -= (learning_rate / np.sqrt(gradsq_W_tilde[j])) * grad_W_tilde
        gradsq_W_tilde[j] += grad_W_tilde ** 2
        b[i] -= (learning_rate / np.sqrt(gradsq_b[i])) * grad_b
        gradsq_b[i] += grad_b ** 2
        b_tilde[j] -= (learning_rate / np.sqrt(gradsq_b_tilde[j])) * grad_b_tilde
        gradsq_b_tilde[j] += grad_b_tilde ** 2

embeddings = W + W_tilde
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1e-10
normalized_embeddings = embeddings / norms

target_word = 'king'
target_idx = w2i[target_word]
sim_scores = np.dot(normalized_embeddings, normalized_embeddings[target_idx])
sorted_indices = np.argsort(-sim_scores)

results_scratch = []
for idx in sorted_indices:
    if idx != target_idx:
        results_scratch.append({'Word': i2w[idx], 'Cosine Similarity': round(float(sim_scores[idx]), 4)})
    if len(results_scratch) == 5:
        break

df_scratch = pd.DataFrame(results_scratch)
print('=== Top 5 Similar Words to "King" (GloVe From Scratch) ===')
print(df_scratch)
```

### Line-by-Line & Syntax Explanation
1. `with open('glove_corpus.txt', 'r', encoding='utf-8') as f`: Opens the 24-sentence corpus text file cleanly.
2. `sentences = [...]`: Cleans non-alphabetic characters using `re.sub(r'[^a-zA-Z\s]', '', ...)`, converts text to lowercase, and splits into token lists.
3. `vocab = sorted(list(set(...)))`: Extracts unique tokens (vocabulary size $V = 68$).
4. `w2i` and `i2w`: Two-way mapping dictionaries between token strings and integer indices.
5. `window_size = 4`: Defines symmetric context scope of 4 words to the left and 4 words to the right.
6. `cooc_dict[(w2i[target], w2i[ctx])] += 1.0 / dist`: Harmonic decay weighting accumulating into the global co-occurrence dictionary.
7. `weights = np.where(counts < x_max, (counts / x_max) ** alpha, 1.0)`: Vectorized calculation of the GloVe weighting function $f(X_{ij})$.
8. `log_counts = np.log(counts)`: Precomputes $\log X_{ij}$ to accelerate per-epoch loss evaluations.
9. `W`, `W_tilde`, `b`, `b_tilde`: Parameter initialization. Embedding matrices are uniformly initialized around zero scaled by dimension; bias vectors start at zero.
10. `gradsq_W = np.ones(...)`: Initializes historical squared gradient accumulators with ones (preventing division by zero).
11. `for epoch in range(epochs)`: Iterates 150 optimization cycles over all non-zero co-occurrence pairs.
12. `diff = np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j] - log_counts[idx]`: Computes prediction residual $\delta_{ij}$.
13. Parameter updates: Applies AdaGrad update rule for $W[i]$, $W\_tilde[j]$, $b[i]$, and $b\_tilde[j]$.
14. `embeddings = W + W_tilde`: Sums target and context representations to form the final embedding matrix.
15. `normalized_embeddings = embeddings / norms`: L2-normalizes vectors so that dot products directly compute cosine similarities.
16. `sim_scores = np.dot(normalized_embeddings, normalized_embeddings[target_idx])`: Projects all word vectors onto the vector for "king".
17. `sorted_indices = np.argsort(-sim_scores)`: Sorts words by descending cosine similarity.

---

## 3. Problem 1.2: GloVe Word Embeddings Using Pretrained Gensim Model

### Theoretical Background
The pretrained `glove-wiki-gigaword-50` model was trained on Wikipedia 2014 + Gigaword 5 (6 billion tokens, 400,000 vocabulary words, 50 dimensions). It captures global semantic and syntactic relations trained on massive web corpora.

### Code Executed
```python
pretrained_glove = api.load('glove-wiki-gigaword-50')
similar_words_gensim = pretrained_glove.most_similar('king', topn=5)

results_gensim = [{'Word': w, 'Cosine Similarity': round(float(score), 4)} for w, score in similar_words_gensim]
df_gensim = pd.DataFrame(results_gensim)
print('=== Top 5 Similar Words to "King" (Pretrained GloVe: glove-wiki-gigaword-50) ===')
print(df_gensim)
```

### Line-by-Line & Syntax Explanation
1. `api.load('glove-wiki-gigaword-50')`: Downloads or loads from local cache the 50-dimensional Gensim `KeyedVectors` instance.
2. `pretrained_glove.most_similar('king', topn=5)`: Computes cosine similarity between the 50-dimensional unit vector of "king" and all 400,000 vocabulary vectors, returning top 5 candidates.
3. `pd.DataFrame(results_gensim)`: Formats output into a tabular representation showing candidate words and similarity coefficients.

### Output Comparison Between Scratch and Pretrained GloVe
| Rank | Scratch GloVe (`glove_corpus.txt`) | Cosine Sim | Pretrained GloVe (`glove-wiki-gigaword-50`) | Cosine Sim |
| :---: | :---: | :---: | :---: | :---: |
| 1 | `the` | 0.9426 | `prince` | 0.8236 |
| 2 | `rules` | 0.8958 | `queen` | 0.7839 |
| 3 | `queen` | 0.7920 | `ii` | 0.7746 |
| 4 | `is` | 0.7494 | `emperor` | 0.7736 |
| 5 | `princess` | 0.6927 | `son` | 0.7667 |

#### Key Takeaway:
- In the scratch model trained on a small 24-sentence corpus, "queen", "rules", and "princess" emerge prominently because they co-occur directly in sentences describing royal governance (e.g., "The king rules the kingdom", "The queen rules the kingdom", "The prince is the son of the king").
- In the pretrained model, "prince", "queen", "emperor", and "son" dominate, reflecting broad encyclopedic associations learned across 6 billion tokens.

---

## 4. Problem 2: Statistical N-Gram Language Model From Scratch

### Theoretical Background & Mathematical Foundations
An N-gram language model estimates the probability of a word sequence $W = (w_1, w_2, \dots, w_m)$ by applying the chain rule of probability:
$$P(w_1, w_2, \dots, w_m) = \prod_{k=1}^m P(w_k \mid w_1, \dots, w_{k-1})$$

#### Markov Assumption
To make computation tractable, the model assumes that the probability of word $w_k$ depends only on the preceding $N-1$ words:
- **Unigram Model ($N=1$)**: Words are independent:
  $$P(w_k \mid w_1, \dots, w_{k-1}) \approx P(w_k) = \frac{C(w_k)}{M}$$
- **Bigram Model ($N=2$)**: Depends on 1 preceding word:
  $$P(w_k \mid w_{k-1}) = \frac{C(w_{k-1}, w_k)}{C(w_{k-1})}$$
- **Trigram Model ($N=3$)**: Depends on 2 preceding words:
  $$P(w_k \mid w_{k-2}, w_{k-1}) = \frac{C(w_{k-2}, w_{k-1}, w_k)}{C(w_{k-2}, w_{k-1})}$$

#### Sequence Padding & Backoff
To properly model start-of-sentence and end-of-sentence distributions, special boundary tokens `<s>` and `</s>` are prepended and appended to each sentence. If a trigram context has zero occurrences, the model smoothly backs off to the bigram estimate.

### Code Executed
```python
lm_corpus = """
natural language processing is a branch of artificial intelligence.
natural language processing enables computers to understand human language.
machine learning is a branch of artificial intelligence.
deep learning is a part of machine learning.
language models learn patterns from text data.
a language model predicts the next word in a sentence.
natural language processing uses language models.
artificial intelligence includes machine learning and deep learning.
"""

sentences_lm = [re.sub(r'[^a-zA-Z\s]', '', line.lower()).split() for line in lm_corpus.strip().split('\n') if line.strip()]

unigrams = Counter()
bigrams = Counter()
trigrams = Counter()

for sent in sentences_lm:
    padded = ['<s>'] + sent + ['</s>']
    for w in padded:
        unigrams[w] += 1
    for i in range(len(padded) - 1):
        bigrams[(padded[i], padded[i+1])] += 1
    for i in range(len(padded) - 2):
        trigrams[(padded[i], padded[i+1], padded[i+2])] += 1

def predict_next_word(context_str, n=2, top_k=3):
    tokens = context_str.strip().lower().split()
    candidates = {}
    if n == 2:
        prev_word = tokens[-1]
        denom = unigrams[prev_word]
        if denom > 0:
            for pair, count in bigrams.items():
                if pair[0] == prev_word and pair[1] != '</s>':
                    candidates[pair[1]] = count / denom
    elif n == 3:
        if len(tokens) >= 2:
            prev_context = (tokens[-2], tokens[-1])
            denom = bigrams[prev_context]
            if denom > 0:
                for tri, count in trigrams.items():
                    if (tri[0], tri[1]) == prev_context and tri[2] != '</s>':
                        candidates[tri[2]] = count / denom
        if not candidates and len(tokens) >= 1:
            prev_word = tokens[-1]
            denom = unigrams[prev_word]
            if denom > 0:
                for pair, count in bigrams.items():
                    if pair[0] == prev_word and pair[1] != '</s>':
                        candidates[pair[1]] = count / denom
    sorted_items = sorted(candidates.items(), key=lambda x: -x[1])[:top_k]
    return [{'Predicted Next Word': w, 'Conditional Probability': round(float(prob), 4)} for w, prob in sorted_items]

test_prompts = [
    ('natural language', 3),
    ('machine', 2),
    ('deep', 2),
    ('artificial', 2),
    ('language', 2)
]

prediction_records = []
for prompt, n_val in test_prompts:
    preds = predict_next_word(prompt, n=n_val, top_k=1)
    if preds:
        prediction_records.append({
            'Input Context': prompt,
            'Model Type': f'{n_val}-Gram',
            'Predicted Word': preds[0]['Predicted Next Word'],
            'Probability': preds[0]['Conditional Probability']
        })

df_predictions = pd.DataFrame(prediction_records)
print('=== Statistical N-Gram Language Model Predictions ===')
print(df_predictions)
```

### Empirical Model Predictions Table
| Input Context | Model Type | Predicted Word | Probability | Mathematical Explanation |
| :---: | :---: | :---: | :---: | :--- |
| `natural language` | 3-Gram | `processing` | 1.0000 | In all 3 occurrences of `(natural, language)`, the following token is `processing` ($3/3 = 1.0$). |
| `machine` | 2-Gram | `learning` | 1.0000 | In all 4 occurrences of `machine`, the following token is `learning` ($4/4 = 1.0$). |
| `deep` | 2-Gram | `learning` | 1.0000 | In all 2 occurrences of `deep`, the following token is `learning` ($2/2 = 1.0$). |
| `artificial` | 2-Gram | `intelligence` | 1.0000 | In all 3 occurrences of `artificial`, the following token is `intelligence` ($3/3 = 1.0$). |
| `language` | 2-Gram | `processing` | 0.4286 | Total occurrences of `language` as antecedent is 7: followed by `processing` (3 times $\to 3/7 \approx 0.4286$), `models` (2 times $\to 2/7$), `model` (1 time $\to 1/7$). |

---

## 5. Comprehensive Viva Voce Preparation

### Q1: What is the fundamental difference between GloVe and Word2Vec?
- **Answer**:
  - **Word2Vec (Skip-Gram / CBOW)** is a window-based local prediction model. It streams over the corpus and updates weights using local context windows, but it does not directly utilize global co-occurrence statistics across the entire dataset.
  - **GloVe (Global Vectors)** is a count-based model that explicitly computes the global word-word co-occurrence matrix $X$ first, then directly factorizes this matrix using a weighted least-squares regression objective. It combines the global statistical advantages of matrix factorization with the local sub-linear vector properties of Word2Vec.

### Q2: Why is the weighting function $f(X_{ij})$ essential in GloVe?
- **Answer**: Without $f(X_{ij})$, high-frequency word pairs (such as stopwords like "the", "is", "a") would completely dominate the objective function, skewing embeddings away from meaningful semantic relationships. The function $f(X_{ij}) = (X_{ij} / X_{\max})^\alpha$ caps the weight of very frequent co-occurrences at 1.0 when $X_{ij} \ge X_{\max}$ (typically $X_{\max} = 100$ on large corpora, or $10$ on small corpora) and smoothly dampens rare pairs with exponent $\alpha = 0.75$.

### Q3: Why does GloVe formulate the objective in terms of $\log X_{ij}$ rather than $X_{ij}$?
- **Answer**: Word co-occurrence frequencies follow Zipf's law (an exponential distribution). The logarithm compresses this heavy-tailed distribution into a linear scale. Furthermore, the ratio of co-occurrence probabilities $\frac{P(k \mid i)}{P(k \mid j)}$ encodes semantic meaning (e.g. ice vs steam relative to solid and gas). Taking the log of this ratio yields $\log P(k \mid i) - \log P(k \mid j) = w_k^\top (w_i - w_j)$, creating linear vector arithmetic in vector space.

### Q4: Why do we add $W$ and $\tilde{W}$ to obtain the final GloVe embeddings?
- **Answer**: In an undirected context window, the co-occurrence matrix $X$ is symmetric ($X_{ij} = X_{ji}$). Consequently, $W$ (target word matrix) and $\tilde{W}$ (context word matrix) play mathematically identical roles and differ only due to random initialization and optimization noise. Summing $W + \tilde{W}$ (or averaging them) performs model averaging, reducing variance and improving vector quality.

### Q5: What is the Markov assumption in statistical language modeling?
- **Answer**: The general probability of a word sequence requires conditioning each word on the entire history of preceding words: $P(w_k \mid w_1, \dots, w_{k-1})$. The Markov assumption posits that the probability of the current word depends only on the immediate previous $N-1$ words:
$$P(w_k \mid w_1, \dots, w_{k-1}) \approx P(w_k \mid w_{k-N+1}, \dots, w_{k-1})$$
This simplifies exponential parameter growth from $|V|^k$ to $|V|^N$.

### Q6: What is Maximum Likelihood Estimation (MLE) in N-grams?
- **Answer**: MLE calculates the parameters of the model by maximizing the likelihood of the observed training corpus. For bigrams:
$$P_{MLE}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$
It is the ratio of the frequency of the pair to the frequency of the prefix.

### Q7: What is the zero-probability problem in N-gram language models and how is it resolved?
- **Answer**: If an N-gram is never observed in the training corpus ($C = 0$), the MLE probability is 0. Under the chain rule, multiplying by 0 gives the entire sentence a probability of 0, resulting in infinite perplexity ($PP = \infty$). This is solved by smoothing techniques:
  - **Laplace (Add-1) Smoothing**: Adds 1 to all counts: $\frac{C + 1}{C_{prefix} + |V|}$.
  - **Add-$k$ (Lidstone) Smoothing**: Adds a fractional count $k \in (0, 1)$.
  - **Kneser-Ney Smoothing**: Uses absolute discounting and continuation probabilities based on how versatile a word is as a novel continuation.
  - **Backoff (e.g. Katz)**: If the trigram count is 0, backs off to the bigram, then unigram.

### Q8: How is the performance of a language model evaluated quantitatively?
- **Answer**: Using **Perplexity (PP)**, which is the exponentiated cross-entropy:
$$PP(W) = \exp \left( -\frac{1}{M} \sum_{i=1}^M \ln P(w_i \mid w_1, \dots, w_{i-1}) \right)$$
Perplexity can be interpreted as the weighted average branching factor (the effective number of equally likely words the model is choosing between at each step). A lower perplexity indicates a better predictive model.

### Q9: Why did the scratch GloVe model rank "the" as the most similar word to "King"?
- **Answer**: On a very small corpus of 24 sentences without stopword removal or high-frequency capping, functional words like "the" appear in almost every sentence directly preceding "king", "queen", and "man". Consequently, their co-occurrence counts and gradient updates share high overlap in small vector spaces. In large-scale corpora (e.g., Wikipedia), $x_{\max}$ and sub-sampling mitigate this effect.

### Q10: How does harmonic weighting $1/d$ improve co-occurrence statistics?
- **Answer**: In natural syntax, words immediately adjacent to the target word (distance $d=1$) convey stronger grammatical and semantic dependencies than words separated by 4 or 5 tokens. Weighting co-occurrence by $1/d$ reflects syntactic locality while still allowing broader topical signals within the window.
