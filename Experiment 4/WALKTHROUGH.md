# Walkthrough: Experiment 4 - Term Frequency, NER, and TF-IDF

This walkthrough provides a clear, step-by-step explanation of [Experiment_4.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%204/Experiment_4.ipynb). The code is structured to be clean, readable, and easy to explain during a lab evaluation or viva.

---

## 1. Setup and Environment Imports

### Code
```python
import string
import math
import pandas as pd
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')
```

### What to explain to the examiner:
- `import string`: Gives us `string.punctuation` (a string of all 32 punctuation symbols `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`) to clean text.
- `import math`: Used for `math.log()` to compute the natural logarithm for Inverse Document Frequency (IDF).
- `import pandas as pd`: Used to format frequency tables into clean DataFrames and export them to `.csv`.
- `from collections import Counter`: Python's built-in counting dictionary to count word frequencies with `Counter(words)`.
- `import spacy` & `nlp = spacy.load('en_core_web_sm')`: Loads spaCy's pre-trained small English pipeline model for tokenization and Named Entity Recognition (NER).

---

## Experiment 4.1: Term-Frequency Analysis & Named Entity Recognition (Using Toolkit)

### Goal
1. Read the text file `4.1_4.2_input.txt`.
2. Tokenize and count the frequency of each alphabetic word using spaCy.
3. Save the full frequency table to `4.1_term_frequency_toolkit.csv` and display the top 10 terms.
4. Extract Named Entities (entities like names, organizations, languages, locations) using spaCy's `doc.ents`.

### Code
```python
with open('4.1_4.2_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

doc = nlp(text)
words = [token.text.lower() for token in doc if token.is_alpha]
tf = Counter(words)

df_tf_toolkit = pd.DataFrame(tf.most_common(), columns=['Term', 'Frequency'])
df_tf_toolkit.to_csv('4.1_term_frequency_toolkit.csv', index=False)

print("Top 10 Most Frequent Terms (Toolkit):")
print(df_tf_toolkit.head(10))

print("\nSample Named Entities:")
seen_entities = set()
for ent in doc.ents:
    if ent.text not in seen_entities:
        print(f"{ent.text:25} -> {ent.label_:10} ({spacy.explain(ent.label_)})")
        seen_entities.add(ent.text)
    if len(seen_entities) >= 10:
        break
```

### Line-by-Line Breakdown
1. `doc = nlp(text)`: Passes text through spaCy to produce a `Doc` object containing tokens and entity spans.
2. `words = [token.text.lower() for token in doc if token.is_alpha]`: Extracts each token in lowercase, keeping only alphabetic words (`token.is_alpha` ignores punctuation and numbers).
3. `tf = Counter(words)`: Automatically creates a frequency map `{word: count}`.
4. `df_tf_toolkit.to_csv('4.1_term_frequency_toolkit.csv', index=False)`: Exports the table to CSV without row index numbers.
5. `doc.ents`: Iterates through the detected named entities.
   - `ent.text`: The text string (e.g., `"NLP"`, `"English"`, `"India"`).
   - `ent.label_`: The entity category tag (e.g., `ORG`, `LANGUAGE`, `GPE`).
   - `spacy.explain(ent.label_)`: Returns the explanation of the category tag (e.g., `"Companies, agencies, institutions"`).

---

## Experiment 4.2: Term-Frequency Analysis Without Any Toolkit (Pure Python)

### Goal
Calculate word frequencies using basic Python string methods (`lower()`, `translate()`, `split()`) and standard dictionaries without importing NLTK or spaCy.

### Code
```python
with open('4.1_4.2_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))
words = clean_text.split()

tf_dict = {}
for word in words:
    tf_dict[word] = tf_dict.get(word, 0) + 1

sorted_tf = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)
df_tf_pure = pd.DataFrame(sorted_tf, columns=['Term', 'Frequency'])
df_tf_pure.to_csv('4.2_term_frequency_pure_python.csv', index=False)

print("Top 10 Most Frequent Terms (Pure Python):")
print(df_tf_pure.head(10))
```

### Line-by-Line Breakdown
1. `clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))`:
   - Converts the text to lowercase.
   - `str.maketrans('', '', string.punctuation)` creates a translation table marking all punctuation characters for deletion.
   - `.translate()` deletes all punctuation marks in a single fast pass.
2. `words = clean_text.split()`: Splits the text by whitespace into a list of words.
3. `tf_dict[word] = tf_dict.get(word, 0) + 1`:
   - `dict.get(word, 0)` returns the current count of `word`, or `0` if it has not been seen yet, then adds `1`.
4. `sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)`:
   - Sorts the `(word, count)` items based on the count (`x[1]`) in descending order (`reverse=True`).
5. `df_tf_pure.to_csv('4.2_term_frequency_pure_python.csv', index=False)`: Exports the pure Python frequency table to CSV.

---

## Experiment 4.3: Manual Multi-Document TF-IDF Pipeline (From Scratch)

### Goal
Calculate **Term Frequency (TF)**, **Document Frequency (DF)**, **Inverse Document Frequency (IDF)**, and **TF-IDF** for 3 documents completely from scratch using standard Python loops and `math.log()`.

### The 3 Documents
- **Doc 1**: `"Natural language processing is a field of artificial intelligence."`
- **Doc 2**: `"Natural language processing helps computers understand human language."`
- **Doc 3**: `"Machine learning is an important part of artificial intelligence."`

---

### Step-by-Step Mathematical Formulas to Memorize

1. **Term Frequency (TF)**:
   $$\text{TF}(t, d) = \frac{\text{Count of word } t \text{ in document } d}{\text{Total words in document } d}$$
   - *Example*: In Doc 1, the word `natural` appears 1 time out of 9 total words $\to \text{TF} = 1/9 \approx 0.1111$.
   - In Doc 2, the word `language` appears 2 times out of 8 total words $\to \text{TF} = 2/8 = 0.2500$.

2. **Document Frequency (DF)**:
   $$\text{DF}(t) = \text{Number of documents that contain word } t$$
   - *Example*: `language` appears in Doc 1 and Doc 2 $\to \text{DF} = 2$.
   - `field` appears only in Doc 1 $\to \text{DF} = 1$.

3. **Inverse Document Frequency (IDF)**:
   $$\text{IDF}(t) = \log_e\left(\frac{N}{\text{DF}(t)}\right)$$
   - where $N = 3$ (total number of documents).
   - If $\text{DF} = 1$ (rare word): $\text{IDF} = \log(3/1) \approx 1.0986$.
   - If $\text{DF} = 2$ (common word): $\text{IDF} = \log(3/2) \approx 0.4055$.
   - If a word appeared in all 3 documents ($\text{DF} = 3$): $\text{IDF} = \log(3/3) = \log(1) = 0$.

4. **TF-IDF Score**:
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

---

### Code
```python
documents = [
    "Natural language processing is a field of artificial intelligence.",
    "Natural language processing helps computers understand human language.",
    "Machine learning is an important part of artificial intelligence."
]

cleaned_docs = []
for doc in documents:
    clean_text = doc.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = clean_text.split()
    cleaned_docs.append(tokens)

vocab = sorted(list(set(word for doc in cleaned_docs for word in doc)))
N = len(documents)

df_dict = {}
for word in vocab:
    count = 0
    for doc in cleaned_docs:
        if word in doc:
            count += 1
    df_dict[word] = count

idf_dict = {}
for word in vocab:
    idf_dict[word] = math.log(N / df_dict[word])

df_idf_table = pd.DataFrame({
    'Term': vocab,
    'DF': [df_dict[w] for w in vocab],
    'IDF': [round(idf_dict[w], 4) for w in vocab]
})

print("=== 1. Vocabulary, Document Frequency (DF) & Inverse Document Frequency (IDF) ===")
print(df_idf_table)

for i, doc in enumerate(cleaned_docs, 1):
    doc_len = len(doc)
    doc_scores = []
    
    for word in vocab:
        word_count = doc.count(word)
        if word_count > 0:
            tf = word_count / doc_len
            idf = idf_dict[word]
            tfidf = tf * idf
            doc_scores.append({
                'Term': word,
                'Count': word_count,
                'TF': round(tf, 4),
                'DF': df_dict[word],
                'IDF': round(idf, 4),
                'TF-IDF': round(tfidf, 4)
            })
            
    df_doc_table = pd.DataFrame(doc_scores)
    print(f"\n=== Document {i} TF & TF-IDF Scores ===")
    print(df_doc_table)
    
    top_terms = df_doc_table.sort_values(by='TF-IDF', ascending=False)
    print(f"\nTop Terms for Document {i}:")
    print(top_terms[['Term', 'TF-IDF']].head(3))
```

---

### Key Viva Questions & Direct Answers

#### Q1: What is the main intuition behind TF-IDF?
- **Answer**: 
  - **TF** gives higher weight to words that appear frequently within a document.
  - **IDF** penalizes words that appear everywhere across the entire corpus.
  - Combined, **TF-IDF** highlights words that are **uniquely characteristic** of a specific document (i.e. keywords).

#### Q2: What happens if a word appears in every document in the corpus?
- **Answer**: Its Document Frequency $\text{DF} = N$. Thus, $\text{IDF} = \log(N/N) = \log(1) = 0$. Consequently, its $\text{TF-IDF} = \text{TF} \times 0 = 0$. This mathematically eliminates ubiquitous non-informative words.

#### Q3: How is Named Entity Recognition (NER) different from Part-of-Speech (POS) tagging?
- **Answer**:
  - **POS Tagging** assigns grammatical roles to individual words (e.g. *Noun, Verb, Adjective*).
  - **NER** identifies real-world semantic entities and categorizes them into entity types (e.g. *Person, Organization, Location, Language, Date*), often spanning multiple words (e.g., `"Natural Language Processing"` $\to$ `ORG`).

#### Q4: Why do we use natural log `math.log()` in the IDF formula?
- **Answer**: As the corpus size $N$ grows large (e.g., thousands or millions of documents), raw ratios like $N/\text{DF}$ would explode and overpower the TF term. Taking the logarithm dampens the scale, keeping TF and IDF balanced.
