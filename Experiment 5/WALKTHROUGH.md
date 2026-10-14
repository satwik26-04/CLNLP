# Walkthrough: Experiment 5 - Subword Tokenization, Part-of-Speech Tagging, and Word2Vec Embeddings

This walkthrough provides an exhaustive, line-by-line explanation of every code statement, parameter, algorithm, mathematical formulation, and linguistic concept implemented in [Experiment_5.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%205/Experiment_5.ipynb). The experiment covers:
1. **Experiment 5.1**: Subword-level tokenization using Byte Pair Encoding (BPE) (Pretrained GPT-2 & Custom BPE from scratch).
2. **Experiment 5.2**: Subword-level tokenization using SentencePiece (Pretrained T5-Small & Custom BPE model trained on `input_sub_word_data.txt`).
3. **Experiment 5.3**: Sentence-level Part-of-Speech (POS) tagging on a target sentence using both spaCy and NLTK with human-readable tag descriptions.
4. **Experiment 5.4**: Corpus-level Part-of-Speech (POS) tagging and frequency profiling on `input_sub_word_data.txt` using spaCy.
5. **Experiment 5.5**: Word2Vec Continuous Bag-of-Words (CBOW) model training on `corpus.txt`, vector extraction, and top-3 most similar words.
6. **Experiment 5.6**: Word2Vec Skip-Gram model training on `corpus.txt`, vector retrieval for "language", and top-5 most similar words.
7. **Experiment 5.7**: Cosine similarity calculation between specific word pairs (`machine` & `learning`, `natural` & `language`, `deep` & `learning`) using Word2Vec.

---

## 1. Setup and Environment Imports

### Purpose
Load required standard libraries (`string`, `collections`), table formatting tools (`pandas`), modern subword tokenizer frameworks (`tiktoken`, `transformers`, `sentencepiece`), NLP toolkits (`spacy`, `nltk`), and Gensim's Word2Vec architecture.

### Code Executed
```python
import string
import pandas as pd
from collections import Counter, defaultdict
import spacy
import nltk
import tiktoken
from transformers import AutoTokenizer
import sentencepiece as spm
from gensim.models import Word2Vec

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
```

### Line-by-Line & Syntax Explanation
- `import string`: Provides `string.punctuation` containing all 32 standard ASCII punctuation symbols (`!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`) used to clean punctuation from raw text.
- `import pandas as pd`: Imports the Pandas library under alias `pd` to construct DataFrames and present token-ID tables and POS distributions cleanly.
- `from collections import Counter, defaultdict`: Imports `Counter` for frequency mapping and `defaultdict` for collecting adjacent bigram symbol frequencies without raising `KeyError` on unseen pairs.
- `import spacy` & `nlp = spacy.load('en_core_web_sm')`: Loads spaCy's pre-trained small English pipeline (`en_core_web_sm`), providing tokenization, dependency parsing, and Universal POS tagging.
- `import nltk`: Imports the Natural Language Toolkit for rule-based and statistical NLP tasks.
- `import tiktoken`: Imports OpenAI's fast BPE tokenizer library, used for GPT-2, GPT-3.5, and GPT-4 tokenization.
- `from transformers import AutoTokenizer`: Imports HuggingFace's `AutoTokenizer` factory to load pretrained SentencePiece tokenizers (such as `t5-small`).
- `import sentencepiece as spm`: Imports Google's language-independent subword tokenizer trainer and processor.
- `from gensim.models import Word2Vec`: Imports Gensim's implementation of the Word2Vec neural embedding architecture (CBOW and Skip-Gram).
- `nltk.download(...)`: Downloads NLTK's Punkt tokenization data and Averaged Perceptron POS tagger models quietly.

---

## Experiment 5.1: Subword Tokenization Using BPE (Byte Pair Encoding)

### Background & Concepts
- **BPE Principles**: Starts with a character-level vocabulary and repeatedly merges the most frequent adjacent symbol pair across the corpus until reaching the target vocabulary size.
- **Out-of-Vocabulary (OOV) Handling**: Rare or unseen words are split into known subword units rather than being replaced with `<UNK>`.

### Code Executed
```python
with open('input_sub_word_data.txt', 'r', encoding='utf-8') as f:
    text_5 = f.read()

sample_sentence = "Natural language processing is a branch of artificial intelligence that enables computers to understand, interpret, and generate human language."

enc = tiktoken.get_encoding("gpt2")
token_ids_5_1_a = enc.encode(sample_sentence)
tokens_5_1_a = [enc.decode([tid]) for tid in token_ids_5_1_a]

df_5_1_a = pd.DataFrame({'Token': tokens_5_1_a, 'Token ID': token_ids_5_1_a})
print("=== 5.1.a: Pretrained BPE Tokenization (GPT-2) ===")
print(df_5_1_a.head(10))

words = text_5.lower().translate(str.maketrans('', '', string.punctuation)).split()
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

clean_words = sample_sentence.lower().translate(str.maketrans('', '', string.punctuation)).split()
scratch_tokens = []
for w in clean_words:
    scratch_tokens.extend(tokenize_word(w, merges))

scratch_token_ids = [token_to_id.get(tok, -1) for tok in scratch_tokens]
df_5_1_b = pd.DataFrame({'Token': scratch_tokens, 'Token ID': scratch_token_ids})

print("\n=== 5.1.b: BPE From Scratch (Trained on input_sub_word_data.txt) ===")
print(df_5_1_b.head(10))
```

### Line-by-Line Breakdown
- **5.1.a (Pretrained GPT-2 BPE)**:
  1. `enc = tiktoken.get_encoding("gpt2")`: Loads the pretrained GPT-2 BPE vocabulary (50,257 tokens).
  2. `token_ids_5_1_a = enc.encode(sample_sentence)`: Converts input sentence into numerical token IDs.
  3. `tokens_5_1_a = [enc.decode([tid]) for tid in token_ids_5_1_a]`: Decodes each ID to its exact subword piece (e.g. `'Natural'`, `' language'`, `' processing'`).
- **5.1.b (From Scratch BPE on `input_sub_word_data.txt`)**:
  1. `vocab = Counter([' '.join(list(w)) + ' </w>' for w in words])`: Initializes words as space-separated characters with end-of-word tag `</w>`.
  2. `get_stats(v)`: Counts adjacent symbol pair frequencies across the training vocabulary.
  3. `merge_vocab(best_pair, v)`: Replaces the winning bigram pair with the fused symbol.
  4. `tokenize_word(w, merges)`: Re-applies the learned merges in chronological order to tokenize new input words.

---

## Experiment 5.2: Subword Tokenization Using SentencePiece

### Background & Concepts
- **SentencePiece**: A language-independent subword tokenizer developed by Google.
- Unlike traditional BPE (which requires pre-tokenization by whitespace), SentencePiece treats the raw input text as a **pure sequence of characters/bytes**, encoding whitespace as a visible meta-symbol (`_` / `\u2581`).
- This makes SentencePiece completely lossless (reversible without heuristic whitespace rules) and ideal for unsegmented languages (Chinese, Japanese) and multilingual LLMs (T5, LLaMA, XLM-RoBERTa).

### Code Executed
```python
t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
t5_token_ids = t5_tokenizer.encode(sample_sentence)
t5_tokens = t5_tokenizer.convert_ids_to_tokens(t5_token_ids)

df_5_2_a = pd.DataFrame({'Token': t5_tokens, 'Token ID': t5_token_ids})
print("=== 5.2.a: Pretrained SentencePiece Tokenization (T5-Small) ===")
print(df_5_2_a.head(10))

spm.SentencePieceTrainer.train(
    input='input_sub_word_data.txt',
    model_prefix='spm_custom',
    vocab_size=300,
    model_type='bpe',
    character_coverage=1.0
)

sp_processor = spm.SentencePieceProcessor(model_file='spm_custom.model')
sp_tokens = sp_processor.encode_as_pieces(sample_sentence)
sp_token_ids = sp_processor.encode_as_ids(sample_sentence)

df_5_2_b = pd.DataFrame({'Token': sp_tokens, 'Token ID': sp_token_ids})
print("\n=== 5.2.b: SentencePiece From Scratch (Trained on input_sub_word_data.txt) ===")
print(df_5_2_b.head(10))
```

### Line-by-Line Breakdown
- **5.2.a (Pretrained T5 SentencePiece)**:
  1. `AutoTokenizer.from_pretrained('t5-small')`: Loads Google's T5 SentencePiece model.
  2. `t5_tokens = t5_tokenizer.convert_ids_to_tokens(...)`: Inspects tokens. Notice words with leading spaces are prefixed with ` ` (e.g. ` Natural`, ` language`).
- **5.2.b (Custom Trained SentencePiece on `input_sub_word_data.txt`)**:
  1. `spm.SentencePieceTrainer.train(...)`: Trains a custom SentencePiece model with vocabulary size 300 directly on `input_sub_word_data.txt`.
  2. `sp_processor.encode_as_pieces(sample_sentence)`: Encodes sentence into subword pieces (`[' Natural', ' language', ' processing', ...]`).
  3. `sp_processor.encode_as_ids(sample_sentence)`: Encodes sentence into integer token IDs.

---

## Experiment 5.3: Sentence-Level POS Tagging (spaCy vs NLTK)

### Target Sentence
> *"The young student is reading an interesting book in the library."*

### Code Executed
```python
sentence_5_3 = "The young student is reading an interesting book in the library."

doc_5_3 = nlp(sentence_5_3)
spacy_results = [
    {'Unique Token': token.text, 'POS Tag': token.pos_, 'Description': spacy.explain(token.pos_)}
    for token in doc_5_3 if not token.is_punct
]
df_5_3_spacy = pd.DataFrame(spacy_results).drop_duplicates(subset=['Unique Token'])
print("=== 5.3.a: POS Tagging Using spaCy ===")
print(df_5_3_spacy)

tokens_nltk = nltk.word_tokenize(sentence_5_3)
nltk_tags = nltk.pos_tag(tokens_nltk)

nltk_tag_descriptions = {
    'DT': 'determiner',
    'JJ': 'adjective',
    'NN': 'noun, singular',
    'VBZ': 'verb, 3rd person singular present',
    'VBG': 'verb, gerund or present participle',
    'IN': 'preposition or conjunction, subordinating',
    '.': 'punctuation'
}

nltk_results = [
    {'Unique Token': word, 'POS Tag': tag, 'Description': nltk_tag_descriptions.get(tag, tag)}
    for word, tag in nltk_tags if tag != '.'
]
df_5_3_nltk = pd.DataFrame(nltk_results).drop_duplicates(subset=['Unique Token'])
print("\n=== 5.3.b: POS Tagging Using NLTK ===")
print(df_5_3_nltk)
```

### POS Tagging Results Comparison Table

| Word | spaCy Tag | spaCy Description (Universal POS) | NLTK Tag | NLTK Description (Penn Treebank) |
| :--- | :--- | :--- | :--- | :--- |
| **The** | `DET` | determiner | `DT` | determiner |
| **young** | `ADJ` | adjective | `JJ` | adjective |
| **student** | `NOUN` | noun | `NN` | noun, singular |
| **is** | `AUX` | auxiliary | `VBZ` | verb, 3rd person singular present |
| **reading** | `VERB` | verb | `VBG` | verb, gerund / present participle |
| **an** | `DET` | determiner | `DT` | determiner |
| **interesting** | `ADJ` | adjective | `JJ` | adjective |
| **book** | `NOUN` | noun | `NN` | noun, singular |
| **in** | `ADP` | adposition | `IN` | preposition or subordinating conjunction |
| **the** | `DET` | determiner | `DT` | determiner |
| **library** | `NOUN` | noun | `NN` | noun, singular |

---

## Experiment 5.4: Corpus-Level POS Tagging & Frequency Profiling

### Goal
Perform POS tagging across the complete text file `input_sub_word_data.txt` using spaCy and calculate the occurrence frequency of each unique `(Token, POS Tag)` pair.

### Code Executed
```python
doc_5_4 = nlp(text_5)

token_pos_pairs = [
    (token.text.lower(), token.pos_, spacy.explain(token.pos_))
    for token in doc_5_4 if token.is_alpha
]

pair_counts = Counter(token_pos_pairs)

df_5_4 = pd.DataFrame([
    {'Unique Token': token, 'POS Tag': pos, 'Description': desc, 'Frequency': freq}
    for (token, pos, desc), freq in pair_counts.most_common()
])

print("=== 5.4: Corpus-Level POS Tagging & Frequency (Top 15 Most Frequent Pairs) ===")
print(df_5_4.head(15))
```

### Top 10 Most Frequent (Token, POS Tag) Pairs in `input_sub_word_data.txt`:
1. `('the', 'DET')`: 47 occurrences
2. `('a', 'DET')`: 28 occurrences
3. `('and', 'CCONJ')`: 27 occurrences
4. `('can', 'AUX')`: 23 occurrences
5. `('of', 'ADP')`: 22 occurrences
6. `('be', 'AUX')`: 15 occurrences
7. `('is', 'AUX')`: 14 occurrences
8. `('as', 'ADP')`: 13 occurrences
9. `('bpe', 'PROPN')`: 13 occurrences
10. `('language', 'NOUN')`: 12 occurrences

---

## Experiment 5.5: Word2Vec Continuous Bag-of-Words (CBOW) Model

### Background & Concepts
- **Word2Vec (Mikolov et al., 2013)**: Represents words as dense, real-valued vectors in a continuous vector space where semantically similar words are located close to each other.
- **Continuous Bag-of-Words (CBOW)**:
  - **Objective**: Predicts the target word $w_t$ given its surrounding context words $w_{t-c}, \dots, w_{t+c}$.
  - **Architecture**: A shallow two-layer neural network with projection and output layers. Context word vectors are averaged in the projection layer.
  - **Speed & Properties**: Faster to train than Skip-Gram and slightly better representations for frequent words.
  - **In Gensim**: Controlled by parameter `sg=0` (`sg` stands for skip-gram; `0` specifies CBOW).

### Input Corpus (`corpus.txt`)
```text
Natural language processing is interesting
Natural language processing is useful
I love machine learning
Machine learning is useful
Deep learning is a part of machine learning
Artificial intelligence is changing the world
Natural language processing uses machine learning
Deep learning and artificial intelligence are related
I study natural language processing
Machine learning helps artificial intelligence
```

### Code Executed
```python
with open('corpus.txt', 'r', encoding='utf-8') as f:
    corpus_sentences = [line.strip().lower().split() for line in f if line.strip()]

cbow_model = Word2Vec(corpus_sentences, vector_size=50, window=3, min_count=1, sg=0, epochs=100, seed=42)

selected_word = 'learning'
cbow_vector = cbow_model.wv[selected_word]

print("=== 5.5: Word2Vec CBOW Model ===")
print(f"Selected Word: '{selected_word}'")
print(f"Vector Representation (Dimension: {len(cbow_vector)}):")
print(cbow_vector)

print(f"\nThree Most Similar Words to '{selected_word}':")
similar_words_cbow = cbow_model.wv.most_similar(selected_word, topn=3)
for word, score in similar_words_cbow:
    print(f"  {word:15} -> Cosine Similarity: {score:.4f}")
```

### Line-by-Line & Syntax Explanation
1. `corpus_sentences = [line.strip().lower().split() for line in f if line.strip()]`: Reads lines from `corpus.txt`, strips whitespace, converts to lowercase, and splits into token lists.
2. `Word2Vec(...)`:
   - `sentences`: Input list of tokenized sentences.
   - `vector_size=50`: Dimensionality of the learned dense embedding vectors (each word is a 50-dimensional point).
   - `window=3`: Maximum distance between the current and predicted word within a sentence.
   - `min_count=1`: Ignores all words with total frequency lower than this. Setting `1` ensures every word in our small corpus is assigned an embedding.
   - `sg=0`: Selects the Continuous Bag-of-Words (CBOW) training algorithm.
   - `epochs=100`: Number of training iterations over the corpus (needed for small corpora to converge).
   - `seed=42`: Random seed for deterministic reproducibility.
3. `cbow_vector = cbow_model.wv[selected_word]`: Accesses the KeyedVectors collection (`.wv`) and retrieves the 50-dimensional floating-point numpy array for `'learning'`.
4. `cbow_model.wv.most_similar(selected_word, topn=3)`: Computes cosine similarity between the vector of `'learning'` and all other vectors in the vocabulary, returning the top 3 highest-scoring words.

---

## Experiment 5.6: Word2Vec Skip-Gram Model

### Background & Concepts
- **Skip-Gram Architecture**:
  - **Objective**: Given a center target word $w_t$, the model predicts the surrounding context words within a window $c$.
  - **In Gensim**: Controlled by parameter `sg=1` (`sg` stands for Skip-Gram).
  - **Strengths**: Works well with small datasets and creates superior representations for rare words compared to CBOW.

### Code Executed
```python
sg_model = Word2Vec(corpus_sentences, vector_size=50, window=3, min_count=1, sg=1, epochs=100, seed=42)

target_word = 'language'
sg_vector = sg_model.wv[target_word]

print("=== 5.6: Word2Vec Skip-Gram Model ===")
print(f"Target Word: '{target_word}'")
print(f"Vector Representation (Dimension: {len(sg_vector)}):")
print(sg_vector)

print(f"\nFive Most Similar Words to '{target_word}':")
similar_words_sg = sg_model.wv.most_similar(target_word, topn=5)
for word, score in similar_words_sg:
    print(f"  {word:15} -> Cosine Similarity: {score:.4f}")
```

### Line-by-Line & Syntax Explanation
1. `Word2Vec(..., sg=1)`: Instantiates and trains the Skip-Gram variant (`sg=1`) using the same 10 sentences.
2. `sg_vector = sg_model.wv[target_word]`: Retrieves the 50-dimensional vector for `"language"`.
3. `sg_model.wv.most_similar(target_word, topn=5)`: Extracts the 5 words having the highest cosine similarity to `"language"` in the Skip-Gram embedding space.

---

## Experiment 5.7: Cosine Similarity Between Word Pairs

### Background & Mathematical Foundations
Cosine similarity measures the orientation of two vectors in multi-dimensional space, independent of their magnitudes:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

- **Value Range**: $[-1.0, 1.0]$.
  - $+1.0$: Exactly the same direction (semantically identical / collinear).
  - $0.0$: Orthogonal vectors (no relationship / independent).
  - $-1.0$: Opposite direction.

### Code Executed
```python
word_pairs = [
    ('machine', 'learning'),
    ('natural', 'language'),
    ('deep', 'learning')
]

print("=== 5.7: Cosine Similarity Between Word Pairs ===")
similarity_records = []
for w1, w2 in word_pairs:
    sim_score = sg_model.wv.similarity(w1, w2)
    similarity_records.append({'Word 1': w1, 'Word 2': w2, 'Cosine Similarity': round(float(sim_score), 4)})

df_sim = pd.DataFrame(similarity_records)
print(df_sim)
```

### Line-by-Line & Syntax Explanation
1. `word_pairs = [...]`: Defines the three required evaluation pairs from the experiment sheet:
   - `machine` and `learning`
   - `natural` and `language`
   - `deep` and `learning`
2. `sg_model.wv.similarity(w1, w2)`: Computes the dot product of normalized vectors for `w1` and `w2`.
3. `pd.DataFrame(similarity_records)`: Tabulates the calculated pairwise similarity scores.

---

## Evaluation Viva Questions & Answers

### Q1: What is the main difference between CBOW and Skip-Gram in Word2Vec?
- **Answer**:
  - **CBOW (Continuous Bag-of-Words)**: Predicts the target word from its context words. It averages context vectors, making it faster to train and effective for frequent words.
  - **Skip-Gram**: Predicts context words from a given target word. It treats each context-target pair independently, making it more effective for smaller datasets and rare words.

### Q2: Why is Word2Vec superior to One-Hot Encoding?
- **Answer**:
  - **One-Hot Encoding**: Sparse, high-dimensional ($V$-dimensional where $V$ is vocabulary size), and has zero semantic distance between words (all dot products are 0).
  - **Word2Vec**: Dense, low-dimensional (e.g. 50-300 dimensions), and encodes semantic relationships (words appearing in similar contexts share similar geometric directions).

### Q3: What is the difference between BPE and SentencePiece?
- **Answer**: Standard BPE requires the input text to be pre-tokenized on whitespaces first (which assumes language has spaces and discards exact whitespace details). SentencePiece treats raw input as a continuous character stream, treating whitespace as a regular character (` `), making it completely language-independent and lossless.

### Q4: What is the difference between Universal POS tags (spaCy) and Penn Treebank tags (NLTK)?
- **Answer**:
  - **Universal POS (UPOS)** in spaCy uses a standardized set of 17 coarse-grained tags (`NOUN`, `VERB`, `ADJ`, `AUX`, `DET`, `ADP`) consistent across languages.
  - **Penn Treebank** in NLTK uses a fine-grained tagset of 36 tags specific to English grammar (`NN`, `NNS`, `NNP`, `VB`, `VBZ`, `VBG`, `VBD`).

### Q5: Why is `min_count=1` essential when training Word2Vec on `corpus.txt`?
- **Answer**: In Gensim, the default `min_count` is 5 (words occurring fewer than 5 times are discarded). Because `corpus.txt` is a small training corpus of 10 sentences, most words appear fewer than 5 times. Setting `min_count=1` ensures that all words are retained in the model vocabulary.
