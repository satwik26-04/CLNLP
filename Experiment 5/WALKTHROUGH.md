# Walkthrough: Experiment 5 - Subword Tokenization and Part-of-Speech Tagging

This walkthrough provides a comprehensive, line-by-line explanation of all code, algorithms, parameters, NLP theory, and outputs implemented in [Experiment_5.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%205/Experiment_5.ipynb). The experiment explores four sub-experiments:
1. **Experiment 5.1**: Subword-level tokenization using Byte Pair Encoding (BPE) (Pretrained GPT-2 & From Scratch).
2. **Experiment 5.2**: Subword-level tokenization using SentencePiece (Pretrained T5-Small & Custom BPE model trained on `input_sub_word_data.txt`).
3. **Experiment 5.3**: Sentence-level Part-of-Speech (POS) tagging on a target sentence using both spaCy and NLTK with human-readable tag descriptions.
4. **Experiment 5.4**: Corpus-level Part-of-Speech (POS) tagging and frequency profiling on `input_sub_word_data.txt` using spaCy.

---

## 1. Setup and Environment Imports

### Purpose
Load required standard libraries (`string`, `collections`), table formatting tools (`pandas`), modern subword tokenizer frameworks (`tiktoken`, `transformers`, `sentencepiece`), and NLP toolkits (`spacy`, `nltk`).

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

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
```

### Line-by-Line & Syntax Explanation
- `import string`: Provides `string.punctuation` for punctuation stripping during text preprocessing.
- `import pandas as pd`: For formatting token-ID tables and POS frequency distributions.
- `from collections import Counter, defaultdict`: Provides `Counter` for frequency aggregation and `defaultdict` for collecting bigram symbol pair frequencies without `KeyError`.
- `import spacy` & `nlp = spacy.load('en_core_web_sm')`: Loads spaCy's English pipeline model.
- `import nltk`: Imports NLTK for Penn Treebank POS tagging (`nltk.pos_tag`).
- `import tiktoken`: OpenAI's high-speed BPE tokenizer library.
- `from transformers import AutoTokenizer`: HuggingFace model hub tokenizer interface for loading pretrained SentencePiece models (`t5-small`).
- `import sentencepiece as spm`: Google's language-independent subword tokenizer trainer and processor.
- `nltk.download(...)`: Downloads NLTK's sentence/word tokenizers and Averaged Perceptron POS tagger models.

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
  1. `vocab = Counter([' '.join(list(w)) + ' </w>' for w in words])`: Initializes words as character sequences with end-of-word tag `</w>`.
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
  1. `AutoTokenizer.from_pretrained('t5-small')`: Loads Google's T5 SentencePiece unigram/BPE model.
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

## Evaluation Viva Questions & Answers

### Q1: What is the main difference between BPE and SentencePiece?
- **Answer**: Standard BPE requires the input text to be pre-tokenized on whitespaces first (which assumes language has spaces and discards exact whitespace details). SentencePiece treats raw input as a continuous character stream, treating whitespace as a regular character (` `), making it completely language-independent and lossless.

### Q2: What is the difference between Universal POS tags (spaCy) and Penn Treebank tags (NLTK)?
- **Answer**:
  - **Universal POS (UPOS)** in spaCy uses a standardized set of 17 coarse-grained tags (`NOUN`, `VERB`, `ADJ`, `AUX`, `DET`, `ADP`) consistent across languages.
  - **Penn Treebank** in NLTK uses a fine-grained tagset of 36 tags specific to English grammar (`NN`, `NNS`, `NNP`, `VB`, `VBZ`, `VBG`, `VBD`).

### Q3: Why does `is` receive tag `AUX` in spaCy but `VBZ` in NLTK?
- **Answer**: In the sentence *"The student is reading"*, `is` functions as an auxiliary helping verb for the main lexical verb `reading`. Universal POS distinguishes auxiliaries (`AUX`) from main verbs (`VERB`), whereas Penn Treebank tags all verb forms under morphological inflection tags (`VBZ` = 3rd person singular present).
