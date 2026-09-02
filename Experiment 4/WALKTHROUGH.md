# Walkthrough: Experiment 4 - Term Frequency Analysis, Named Entity Recognition (NER), and TF-IDF

This walkthrough provides a comprehensive, line-by-line explanation of all code, functions, mathematical formulas, and NLP concepts implemented in [Experiment_4.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%204/Experiment_4.ipynb). The experiment explores three core areas of computational linguistics:
1. **Experiment 4.1**: Term Frequency (TF) Analysis and Named Entity Recognition (NER) using an NLP toolkit (`spaCy`).
2. **Experiment 4.2**: Term Frequency (TF) Analysis without using any external NLP toolkit (Pure Python string manipulation and dictionaries).
3. **Experiment 4.3**: End-to-end mathematical calculation of Term Frequency (TF), Document Frequency (DF), Inverse Document Frequency (IDF), and TF-IDF weights across multiple documents completely from scratch.

---

## 1. Setup and Environment Imports

### Purpose
Load the necessary Python standard library modules, data structures, tabular formatting tools, and spaCy's pre-trained English language pipeline.

### Code Executed
```python
import string
import math
import pandas as pd
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')
```

### Line-by-Line & Syntax Explanation

**Line 1: `import string`**
- **Syntax**: `import <module_name>`
- **Explanation**: Imports Python's built-in string module to access the `string.punctuation` constant containing all 32 standard ASCII punctuation marks (`!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~`). Used in pure Python text cleaning.

**Line 2: `import math`**
- **Syntax**: `import <module_name>`
- **Explanation**: Imports Python's standard math library to compute natural logarithms (`math.log()`) for the Inverse Document Frequency (IDF) formula.

**Line 3: `import pandas as pd`**
- **Syntax**: `import <library> as <alias>`
- **Explanation**: Imports **pandas** for building tabular DataFrames, sorting frequency tables, and exporting results to structured CSV files.

**Line 4: `from collections import Counter`**
- **Syntax**: `from <module> import <class>`
- **Explanation**: Imports `Counter`, a high-performance dictionary subclass specifically designed for counting hashable objects. `Counter.most_common()` quickly extracts the top $K$ frequent items in $O(N \log K)$ time.

**Lines 5-7: `import spacy` & `nlp = spacy.load('en_core_web_sm')`**
- **Syntax**: `spacy.load(model_name: str) -> Language`
- **Explanation**: Loads spaCy's trained English pipeline model (`en_core_web_sm`), which includes tokenization, part-of-speech tagging, dependency parsing, and Named Entity Recognition (NER).

---

## Experiment 4.1: Term Frequency & Named Entity Recognition (Toolkit)

### Background & Concepts
- **Term Frequency (TF)**: Counts the number of times each word appears in the text.
- **Named Entity Recognition (NER)**: A subtask of information extraction that identifies and classifies proper nouns / entities into predefined categories (e.g., `PERSON`, `ORG`, `GPE` for geopolitical entities, `LANGUAGE`, `DATE`, `NORP`).

### Code Executed
```python
with open('4.1_4.2_input.txt', 'r', encoding='utf-8') as f:
    text_4_1 = f.read()

doc_4_1 = nlp(text_4_1)
tokens_toolkit = [token.text.lower() for token in doc_4_1 if token.is_alpha]
tf_toolkit = Counter(tokens_toolkit)

df_tf_toolkit = pd.DataFrame(tf_toolkit.most_common(), columns=['Term', 'Frequency'])
df_tf_toolkit.to_csv('4.1_term_frequency_toolkit.csv', index=False)

print("Top 10 Most Frequent Terms (Toolkit):")
print(df_tf_toolkit.head(10))

entities_list = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc_4_1.ents]
df_ner = pd.DataFrame(entities_list, columns=['Entity', 'Label', 'Description']).drop_duplicates()

print("\nNamed Entity Recognition (NER) Results Sample:")
print(df_ner.head(10))
```

### Line-by-Line & Syntax Explanation
1. `with open('4.1_4.2_input.txt', 'r', encoding='utf-8') as f:`: Opens the input text file safely using a context manager.
2. `text_4_1 = f.read()`: Reads the entire text content into memory.
3. `doc_4_1 = nlp(text_4_1)`: Processes the raw text through spaCy's pipeline, generating token annotations, dependency parses, and named entity spans.
4. `tokens_toolkit = [token.text.lower() for token in doc_4_1 if token.is_alpha]`:
   - `token.text.lower()`: Converts token to lowercase.
   - `if token.is_alpha`: Filters out punctuation marks, numbers, and whitespace, keeping only alphabetic words.
5. `tf_toolkit = Counter(tokens_toolkit)`: Counts the frequency of each unique word token.
6. `df_tf_toolkit = pd.DataFrame(tf_toolkit.most_common(), columns=['Term', 'Frequency'])`: Converts frequency pairs sorted in descending order into a DataFrame.
7. `df_tf_toolkit.to_csv('4.1_term_frequency_toolkit.csv', index=False)`: Exports the full term-frequency table to a CSV file.
8. `entities_list = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc_4_1.ents]`:
   - `doc_4_1.ents`: Iterates over the named entity spans (`Span` objects) identified by spaCy's statistical NER model.
   - `ent.text`: The text string of the entity (e.g., `"NLP"`, `"Hindi"`, `"India"`).
   - `ent.label_`: The entity category tag (e.g., `ORG`, `LANGUAGE`, `GPE`).
   - `spacy.explain(ent.label_)`: Built-in helper that returns human-readable descriptions of the label tag.
9. `df_ner = pd.DataFrame(...).drop_duplicates()`: Tabulates unique entities and displays the results.

### Output Results
- **Top 10 Terms (Toolkit)**:
  1. `a`: 88
  2. `and`: 86
  3. `the`: 75
  4. `of`: 42
  5. `to`: 42
  6. `in`: 42
  7. `can`: 42
  8. `is`: 37
  9. `may`: 30
  10. `word`: 30
- **Identified Entities**:
  - `NLP` $\to$ `ORG` (Companies, agencies, institutions)
  - `English` $\to$ `LANGUAGE` (Any named language)
  - `Hindi`, `Tamil`, `Assamese`, `Bengali` $\to$ `GPE` / `NORP`
  - `twenty` $\to$ `CARDINAL` (Numerals that do not fall under another type)

---

## Experiment 4.2: Term Frequency Analysis (Pure Python)

### Background & Motivation
In resource-constrained environments or lightweight scripts, performing term-frequency counting without external libraries demonstrates how basic dictionary data structures and string transformations work under the hood.

### Code Executed
```python
with open('4.1_4.2_input.txt', 'r', encoding='utf-8') as f:
    text_4_2 = f.read()

clean_text_4_2 = text_4_2.lower().translate(str.maketrans('', '', string.punctuation))
words_pure = [w for w in clean_text_4_2.split() if w.isalpha()]

tf_dict_pure = {}
for word in words_pure:
    tf_dict_pure[word] = tf_dict_pure.get(word, 0) + 1

sorted_tf_pure = sorted(tf_dict_pure.items(), key=lambda item: item[1], reverse=True)
df_tf_pure = pd.DataFrame(sorted_tf_pure, columns=['Term', 'Frequency'])
df_tf_pure.to_csv('4.2_term_frequency_pure_python.csv', index=False)

print("Top 10 Most Frequent Terms (Pure Python):")
print(df_tf_pure.head(10))
```

### Line-by-Line & Syntax Explanation
1. `clean_text_4_2 = text_4_2.lower().translate(str.maketrans('', '', string.punctuation))`:
   - Converts the text to lowercase.
   - `str.maketrans('', '', string.punctuation)` creates a C-level deletion table mapping every punctuation character to `None`.
   - `.translate()` executes character stripping in $O(N)$ single-pass time.
2. `words_pure = [w for w in clean_text_4_2.split() if w.isalpha()]`:
   - `.split()` breaks text on any whitespace (spaces, tabs, newlines).
   - `w.isalpha()` ensures only alphabetic tokens are retained.
3. `for word in words_pure: tf_dict_pure[word] = tf_dict_pure.get(word, 0) + 1`:
   - Classic Python frequency counting using `dict.get(key, default_value)`. If `word` is not yet in the dictionary, `get()` returns `0`, then adds `1`.
4. `sorted(tf_dict_pure.items(), key=lambda item: item[1], reverse=True)`:
   - Sorts the `(word, count)` key-value pairs by count (index `1`) in descending (`reverse=True`) order.
5. `df_tf_pure.to_csv('4.2_term_frequency_pure_python.csv', index=False)`: Exports the pure Python frequency table to CSV.

---

## Experiment 4.3: Manual Multi-Document TF-IDF Pipeline

### Background & Mathematical Foundations

#### 1. Term Frequency (TF)
Measures the local importance of term $t$ in a specific document $d$:
$$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total number of words in document } d}$$

#### 2. Document Frequency (DF)
Measures how many documents in the collection contain term $t$:
$$\text{DF}(t) = \text{Count of documents } d \in D \text{ where } t \in d$$

#### 3. Inverse Document Frequency (IDF)
Measures the global specificity / informativeness of term $t$ across the entire corpus:
$$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)$$
- If a word appears in every document ($\text{DF} = N$), $\text{IDF} = \log(1) = 0$ (meaning it provides zero discriminative power).
- If a word appears in only 1 document ($\text{DF} = 1$), $\text{IDF} = \log(N/1)$ is maximized.

#### 4. TF-IDF Score
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
A high TF-IDF score is reached when a word occurs frequently inside a specific document (high TF) but rarely across the rest of the corpus (high IDF).

---

### Code Executed
```python
documents = [
    "Natural language processing is a field of artificial intelligence.",
    "Natural language processing helps computers understand human language.",
    "Machine learning is an important part of artificial intelligence."
]

cleaned_docs = []
for doc in documents:
    no_punct = doc.lower().translate(str.maketrans('', '', string.punctuation))
    cleaned_docs.append(no_punct.split())

vocabulary = sorted(list(set(term for doc in cleaned_docs for term in doc)))
num_docs = len(documents)

df_counts = {}
for term in vocabulary:
    df_counts[term] = sum(1 for doc in cleaned_docs if term in doc)

idf_values = {}
for term in vocabulary:
    idf_values[term] = math.log(num_docs / df_counts[term])

df_stats = pd.DataFrame({
    'Term': vocabulary,
    'DF (Doc Frequency)': [df_counts[t] for t in vocabulary],
    'IDF (Inverse Doc Frequency)': [round(idf_values[t], 4) for t in vocabulary]
})

print("1. Document Frequency (DF) and Inverse Document Frequency (IDF):")
print(df_stats)

tfidf_records = []
for doc_idx, doc_tokens in enumerate(cleaned_docs, 1):
    doc_len = len(doc_tokens)
    term_counts = Counter(doc_tokens)
    for term in vocabulary:
        count = term_counts[term]
        tf = count / doc_len
        idf = idf_values[term]
        tfidf = tf * idf
        if count > 0:
            tfidf_records.append({
                'Document': f'Doc {doc_idx}',
                'Term': term,
                'Count': count,
                'TF': round(tf, 4),
                'IDF': round(idf, 4),
                'TF-IDF Score': round(tfidf, 4)
            })

df_tfidf_results = pd.DataFrame(tfidf_records)
print("\n2. Term Frequency (TF) and TF-IDF Scores for Present Terms:")
print(df_tfidf_results)

print("\n3. Top Terms with Highest TF-IDF Score Per Document:")
for doc_idx in range(1, num_docs + 1):
    doc_name = f'Doc {doc_idx}'
    subset = df_tfidf_results[df_tfidf_results['Document'] == doc_name]
    top_terms = subset.sort_values(by='TF-IDF Score', ascending=False)
    print(f"\n--- {doc_name} Top Terms ---")
    print(top_terms[['Term', 'TF-IDF Score']].head(5))
```

---

### Step-by-Step Mathematical Calculation

#### Corpus Documents:
- **Doc 1** (9 words): `['natural', 'language', 'processing', 'is', 'a', 'field', 'of', 'artificial', 'intelligence']`
- **Doc 2** (8 words): `['natural', 'language', 'processing', 'helps', 'computers', 'understand', 'human', 'language']` *(Note: 'language' appears twice)*
- **Doc 3** (9 words): `['machine', 'learning', 'is', 'an', 'important', 'part', 'of', 'artificial', 'intelligence']`

#### Vocabulary & DF / IDF Table ($N = 3$):
| Term | Documents Containing Term | DF | IDF = $\log(3 / \text{DF})$ |
| :--- | :--- | :--- | :--- |
| `a` | Doc 1 | 1 | $\log(3/1) \approx 1.0986$ |
| `an` | Doc 3 | 1 | $\log(3/1) \approx 1.0986$ |
| `artificial` | Doc 1, Doc 3 | 2 | $\log(3/2) \approx 0.4055$ |
| `computers` | Doc 2 | 1 | $\log(3/1) \approx 1.0986$ |
| `field` | Doc 1 | 1 | $\log(3/1) \approx 1.0986$ |
| `helps` | Doc 2 | 1 | $\log(3/1) \approx 1.0986$ |
| `human` | Doc 2 | 1 | $\log(3/1) \approx 1.0986$ |
| `important` | Doc 3 | 1 | $\log(3/1) \approx 1.0986$ |
| `intelligence` | Doc 1, Doc 3 | 2 | $\log(3/2) \approx 0.4055$ |
| `is` | Doc 1, Doc 3 | 2 | $\log(3/2) \approx 0.4055$ |
| `language` | Doc 1, Doc 2 | 2 | $\log(3/2) \approx 0.4055$ |
| `learning` | Doc 3 | 1 | $\log(3/1) \approx 1.0986$ |
| `machine` | Doc 3 | 1 | $\log(3/1) \approx 1.0986$ |
| `natural` | Doc 1, Doc 2 | 2 | $\log(3/2) \approx 0.4055$ |
| `of` | Doc 1, Doc 3 | 2 | $\log(3/2) \approx 0.4055$ |
| `part` | Doc 3 | 1 | $\log(3/1) \approx 1.0986$ |
| `processing` | Doc 1, Doc 2 | 2 | $\log(3/2) \approx 0.4055$ |
| `understand` | Doc 2 | 1 | $\log(3/1) \approx 1.0986$ |

---

### Top Terms with Highest TF-IDF Score by Document

#### Document 1:
- Highest score: `a`, `field` with $\text{TF-IDF} = \frac{1}{9} \times 1.0986 \approx \mathbf{0.1221}$ (Unique to Doc 1).

#### Document 2:
- Highest score: `language` with $\text{TF-IDF} = \frac{2}{8} \times 0.4055 \approx \mathbf{0.1014}$, and unique words `computers`, `helps`, `human`, `understand` with $\text{TF-IDF} = \frac{1}{8} \times 1.0986 \approx \mathbf{0.1373}$.

#### Document 3:
- Highest score: `machine`, `learning`, `an`, `important`, `part` with $\text{TF-IDF} = \frac{1}{9} \times 1.0986 \approx \mathbf{0.1221}$ (Unique to Doc 3).

---

## Evaluation Viva Questions & Answers

### Q1: Why is Term Frequency (TF) alone insufficient for document retrieval?
- **Answer**: TF measures raw word counts within a single document. However, high-frequency grammatical words (stop words like *the, is, and*) will always dominate TF counts without offering any topical distinction. TF-IDF penalizes words that appear across all documents using the IDF multiplier, surfacing true topical keywords.

### Q2: What happens to IDF when a word appears in every document in the corpus?
- **Answer**: When $\text{DF}(t) = N$, the argument to the logarithm becomes $N/N = 1$. Since $\log(1) = 0$, the $\text{IDF} = 0$, and consequently $\text{TF-IDF} = 0$. This automatically eliminates universally occurring non-discriminative words.

### Q3: What is the difference between Token-level and Span-level representation in NER?
- **Answer**: In NER, entities frequently span multiple consecutive tokens (e.g., `"Artificial Intelligence"` or `"Dr. A. P. J. Abdul Kalam"`). Token-level taggers use schemes like **BIO/BILUO** (`B-ORG`, `I-ORG`), whereas high-level frameworks like spaCy represent entities as continuous `Span` objects with a single entity label.
