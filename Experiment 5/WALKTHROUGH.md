# Walkthrough: Experiment 5 - Subword-Level Tokenization and POS Tagging

## Experiment 5.1: Subword-Level Tokenization Using Byte Pair Encoding (BPE)

This walkthrough provides an in-depth, line-by-line explanation of the algorithms, code statements, syntax, and NLP theory for **Experiment 5.1**, executed on the provided dataset [input_sub_word_data.txt](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%205/input_sub_word_data.txt).

---

## 1. Theoretical Foundations of Byte Pair Encoding (BPE)

### Why Subword Tokenization?
Traditional NLP tokenization approaches face a fundamental trade-off:
- **Word-level Tokenization**: Suffers from the **Out-of-Vocabulary (OOV)** problem. If a model encounters a word not present in its fixed training dictionary (e.g., `unhappiness`, `bioinformatics`), it must replace it with an uninformative `<UNK>` token. Furthermore, the vocabulary size grows into hundreds of thousands of words.
- **Character-level Tokenization**: Solves OOV (since any word can be spelled with letters), but the sequence length increases significantly, and individual characters carry little standalone semantic meaning.
- **Subword-level Tokenization (BPE)**: Provides an optimal middle ground. Common words remain intact as single tokens (e.g., `the`, `language`), while rare, compound, or morphological words are decomposed into familiar subword pieces (e.g., `un` + `happy`, `play` + `ing`).

---

### How the BPE Algorithm Operates (4 Steps)

```text
Step 1: Initialize Vocabulary with single characters + end-of-word marker </w>
        Example: "low </w>", "lowest </w>", "newer </w>", "wider </w>"

Step 2: Count adjacent symbol pair frequencies across the entire corpus.
        Pairs: ('e', 'r'), ('l', 'o'), ('o', 'w'), ('s', 't'), ...

Step 3: Find the most frequent pair and merge into a new single symbol.
        Merge 1: ('e', 'r') -> 'er'
        Merge 2: ('e', 's') -> 'es'
        Merge 3: ('es', 't') -> 'est'

Step 4: Repeat for K merge iterations to build the final subword vocabulary.
```

---

## 2. Part 5.1.a: Subword Tokenization Using a Pretrained BPE Model

### Code Executed
```python
import tiktoken

with open('input_sub_word_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

sentence = "Natural language processing is a branch of artificial intelligence that enables computers to understand, interpret, and generate human language."

enc = tiktoken.get_encoding("gpt2")
token_ids = enc.encode(sentence)
tokens = [enc.decode([tid]) for tid in token_ids]

print("=== 5.1.a: Pretrained BPE Tokenization (GPT-2) ===")
print(f"Input Sentence: {sentence}\n")
for token, tid in zip(tokens, token_ids):
    print(f"{token:25} -> Token ID: {tid}")
```

### Line-by-Line & Syntax Explanation

**Line 1: `import tiktoken`**
- **Syntax**: `import <library>`
- **Explanation**: Imports OpenAI's `tiktoken` library, a high-performance Byte Pair Encoding tokenizer written in Rust for models such as GPT-2, GPT-3.5, and GPT-4.

**Lines 3-4: `with open('input_sub_word_data.txt', 'r', encoding='utf-8') as f:` & `text = f.read()`**
- **Syntax**: Context manager opening the dataset in read-only UTF-8 mode.
- **Explanation**: Reads the complete text corpus provided for Experiment 5.

**Line 6: `sentence = "..."`**
- **Explanation**: Defines the target input sentence extracted directly from the first paragraph of `input_sub_word_data.txt` to be tokenized into subword units.

**Line 8: `enc = tiktoken.get_encoding("gpt2")`**
- **Syntax**: `tiktoken.get_encoding(encoding_name: str) -> Encoding`
- **Explanation**: Loads the standard pre-trained GPT-2 BPE vocabulary (vocabulary size: 50,257 tokens). In this pre-trained model, byte-level BPE is applied, mapping all common English words and subwords to fixed integer IDs.

**Line 9: `token_ids = enc.encode(sentence)`**
- **Syntax**: `Encoding.encode(text: str) -> list[int]`
- **Explanation**: Passes the string through the pre-trained BPE merge rules. It decomposes the sentence into subwords and converts each subword into its corresponding numerical token ID.

**Line 10: `tokens = [enc.decode([tid]) for tid in token_ids]`**
- **Syntax**: List comprehension over `Encoding.decode(tokens: list[int]) -> str`
- **Explanation**: Decodes each individual token ID back into its exact string representation so we can inspect how words were split. Notice that leading spaces are preserved as part of the token (e.g., `' language'` rather than `'language'`).

**Lines 12-15: Display Output**
- Formats each subword token alongside its integer token ID.

---

## 3. Part 5.1.b: BPE Tokenization From Scratch (Without Pretrained Model)

### Code Executed
```python
import string
from collections import Counter, defaultdict

with open('input_sub_word_data.txt', 'r', encoding='utf-8') as f:
    full_text = f.read()

sentence = "Natural language processing is a branch of artificial intelligence that enables computers to understand, interpret, and generate human language."

words = full_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
vocab = Counter([' '.join(list(w)) + ' </w>' for w in words])

def get_stats(v):
    pairs = defaultdict(int)
    for word, freq in v.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, v):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in v:
        v_out[word.replace(bigram, replacement)] = v[word]
    return v_out

num_merges = 30
merges = []
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    merges.append(best_pair)

subwords = set()
for word in vocab:
    for sym in word.split():
        subwords.add(sym)
token_to_id = {sym: idx for idx, sym in enumerate(sorted(list(subwords)), 1)}

def tokenize_word(w, learned_merges):
    symbols = list(w) + ['</w>']
    for pair in learned_merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                symbols = symbols[:i] + [pair[0] + pair[1]] + symbols[i+2:]
            else:
                i += 1
    return symbols

clean_words = sentence.lower().translate(str.maketrans('', '', string.punctuation)).split()
scratch_tokens = []
for w in clean_words:
    scratch_tokens.extend(tokenize_word(w, merges))

print("=== 5.1.b: BPE From Scratch ===")
print(f"Total Merges Learned: {len(merges)}")
print("Sample Merges Learned:", merges[:5])
print("\nGenerated Subword Tokens & Token IDs:")
for tok in scratch_tokens:
    print(f"{tok:20} -> Token ID: {token_to_id.get(tok, -1)}")
```

---

### Line-by-Line & Algorithmic Breakdown

#### 1. Corpus Preprocessing & Vocabulary Initialization
```python
words = full_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
vocab = Counter([' '.join(list(w)) + ' </w>' for w in words])
```
- Converts the training corpus `input_sub_word_data.txt` to lowercase and deletes punctuation.
- For every word `w`, splits its characters with spaces and attaches the special end-of-word delimiter `'</w>'`.
  - For example, the word `"natural"` becomes `'n a t u r a l </w>'`.
- `Counter(...)` aggregates identical word patterns with their frequency count in the corpus.

---

#### 2. Pair Frequency Statistics Function (`get_stats`)
```python
def get_stats(v):
    pairs = defaultdict(int)
    for word, freq in v.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs
```
- Iterates over every word in the vocabulary.
- `symbols = word.split()` splits the word string into its current constituent symbols.
- For each adjacent pair `(symbols[i], symbols[i+1])`, increments the pair count by the word's frequency `freq`.
- Returns a frequency dictionary of all adjacent symbol pairs.

---

#### 3. Vocabulary Merge Function (`merge_vocab`)
```python
def merge_vocab(pair, v):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in v:
        v_out[word.replace(bigram, replacement)] = v[word]
    return v_out
```
- Takes the winning candidate pair (e.g., `('i', 'n')`).
- `bigram = 'i n'` and `replacement = 'in'`.
- Replaces occurrences of the space-separated bigram `'i n'` with the fused token `'in'` across all words in the vocabulary.

---

#### 4. The Iterative Training Loop
```python
num_merges = 30
merges = []
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    merges.append(best_pair)
```
- Runs 30 merge iterations.
- In each round:
  1. Computes pair frequencies across the updated vocabulary.
  2. Identifies `best_pair = max(pairs, key=pairs.get)`.
  3. Updates the vocabulary with `merge_vocab`.
  4. Records `best_pair` in the sequential `merges` rulebook.

---

#### 5. Assigning Unique Token IDs
```python
subwords = set()
for word in vocab:
    for sym in word.split():
        subwords.add(sym)
token_to_id = {sym: idx for idx, sym in enumerate(sorted(list(subwords)), 1)}
```
- Collects all unique subword units remaining in the vocabulary (base characters plus newly merged subwords).
- Creates a 1-indexed dictionary mapping every subword to an integer `Token ID`.

---

#### 6. Inference / Tokenization Function (`tokenize_word`)
```python
def tokenize_word(w, learned_merges):
    symbols = list(w) + ['</w>']
    for pair in learned_merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                symbols = symbols[:i] + [pair[0] + pair[1]] + symbols[i+2:]
            else:
                i += 1
    return symbols
```
- Takes an arbitrary input word `w`.
- Starts by splitting it into characters: `['n', 'a', 't', 'u', 'r', 'a', 'l', '</w>']`.
- Loops through the learned `merges` in the exact chronological order they were discovered during training.
- Whenever adjacent symbols match `pair`, it fuses them.
- Returns the final subword breakdown for that word.

---

## 4. Key Differences Summary (Pretrained vs. Scratch)

| Aspect | Pretrained Model (5.1.a) | From Scratch (5.1.b) |
| :--- | :--- | :--- |
| **Model / Architecture** | GPT-2 Byte-Level BPE (`tiktoken` / `transformers`) | Algorithmic BPE trained directly on `input_sub_word_data.txt` |
| **Vocabulary Size** | 50,257 tokens (trained on massive WebText) | Custom vocabulary based on the local corpus + merge iterations |
| **Space Encoding** | Byte-level prefix encoding (e.g. `' '` represented as `Ġ` or whitespace) | End-of-word marker `</w>` appended to words |
| **Primary Purpose** | Production inference using established LLM tokenizers | Understanding internal mechanics and merge dynamics of BPE |

---

## 5. Viva / Evaluation Preparation Q&A

### Q1: Why do we append `</w>` to words before training BPE?
- **Answer**: `</w>` serves as an explicit word boundary marker. It allows the tokenizer to distinguish between a character sequence that appears at the end of a word (e.g., `ing</w>`) versus one that appears in the middle of a word (e.g., `ing` in `ginger`).

### Q2: Is BPE deterministic during tokenization?
- **Answer**: Yes. Because merge rules are applied strictly in the sequential order they were learned during training, the exact same word will always produce the identical subword sequence.

### Q3: What is the computational complexity of finding the best pair in BPE?
- **Answer**: Counting adjacent pairs requires a single pass over the unique vocabulary $O(V \times L)$, where $V$ is vocabulary size and $L$ is average word length. Finding the maximum pair frequency takes $O(P)$ where $P$ is the number of unique pairs.
