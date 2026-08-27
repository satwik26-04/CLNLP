# Walkthrough: Experiment 3 - Stemming, Lemmatization, and Regular Expressions

This walkthrough provides an exhaustive, line-by-line explanation of all code, algorithms, parameters, regular expressions, and linguistic theory implemented in [Experiment_3.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%203/Experiment_3.ipynb). The experiment explores word normalization via the Porter Stemmer, morphological dictionary reduction via the WordNet Lemmatizer (with default noun and POS-aware modes), and pattern-based information extraction using Python's regular expressions engine.

---

## 1. Setup and Library Imports

### Purpose
Before executing morphological normalization and pattern matching, we load the required Python standard libraries, data manipulation tools, and NLTK algorithms.

### Code Executed
```python
import re
import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```

### Line-by-Line & Syntax Explanation

**Line 1: `import re`**
- **Syntax**: `import <module_name>`
- **Explanation**: Imports Python's built-in regular expression engine. The `re` module allows pattern-based text searching, extraction, and substitution using deterministic finite automata (DFA) under the hood. In this experiment, it is used to extract structured metadata (emails, URLs, phone numbers, hashtags, mentions) from raw natural language strings.

**Line 2: `import pandas as pd`**
- **Syntax**: `import <library> as <alias>`
- **Explanation**: Imports the **pandas** library under the alias `pd`. Pandas provides DataFrame data structures to neatly tabulate, format, and display word transformations side-by-side (original word, stem, and lemma).

**Line 3: `import nltk`**
- **Syntax**: `import <library>`
- **Explanation**: Imports the **Natural Language Toolkit (NLTK)**, providing algorithmic implementations of linguistic rules, corpora, and lexical databases.

**Line 4: `from nltk.stem import PorterStemmer, WordNetLemmatizer`**
- **Syntax**: `from <package>.<submodule> import <class1>, <class2>`
- **Explanation**: Imports two key normalization classes:
  - `PorterStemmer`: Implements Martin Porter's 1980 rule-based suffix-stripping algorithm.
  - `WordNetLemmatizer`: Implements morphological lemmatization using the Princeton WordNet lexical database.

**Line 5: `from nltk.corpus import wordnet`**
- **Syntax**: `from <package>.<submodule> import <corpus_reader>`
- **Explanation**: Imports the `wordnet` corpus interface, providing Part-of-Speech (POS) constants: `wordnet.NOUN` (`'n'`), `wordnet.VERB` (`'v'`), `wordnet.ADJ` (`'a'`), and `wordnet.ADV` (`'r'`).

**Lines 6-7: `stemmer = PorterStemmer()` & `lemmatizer = WordNetLemmatizer()`**
- **Syntax**: `<variable> = <ClassName>()`
- **Explanation**: Instantiates objects for the stemmer and lemmatizer. `PorterStemmer()` initializes the cascading condition-action rule sets, and `WordNetLemmatizer()` loads the interface to the WordNet morphological lookup processor (`Morphy`).

---

## Experiment 3.1: Stemming Using the Porter Stemmer

### Background & Linguistic Theory
**Stemming** is a crude heuristic process that chops off the ends of words (affixes) in the hope of achieving the goal of reducing words to their root stem. The Porter Stemmer (Martin Porter, 1980) applies 5 sequential phases of algorithmic rewrite rules based on consonant-vowel patterns (measure $m$).

Because it operates purely syntactically without a dictionary:
- **Valid base words are not guaranteed**: Stems like `studi` or `comput` are not real English dictionary words.
- **Over-stemming**: When distinct words are erroneously reduced to the same stem (e.g., `universe` and `university` both becoming `univers`).
- **Under-stemming**: When words that should share a stem are reduced to different stems (e.g., `alumnus`, `alumni`, `alumnae`).

### Code Executed
```python
words_to_stem = [
    'playing', 'played', 'plays', 
    'studies', 'studying', 
    'connected', 'connection', 
    'computers'
]

stem_data = []
for word in words_to_stem:
    stem_data.append({
        'Original Word': word,
        'Porter Stem': stemmer.stem(word)
    })

df_stemming = pd.DataFrame(stem_data)
df_stemming
```

### Line-by-Line & Syntax Explanation
1. `words_to_stem = [...]`: Defines a list of 8 test words displaying inflectional and derivational variations.
2. `stem_data = []`: Initializes an empty list to accumulate dictionary records for tabular visualization.
3. `for word in words_to_stem:`: Iterates through each target word string.
4. `stemmer.stem(word)`:
   - **Syntax**: `PorterStemmer.stem(token: str) -> str`
   - **Explanation**: Applies the 5-phase Porter rewrite rules to the input token string.
   - For `playing`, `played`, `plays`: Strips inflectional suffixes (`-ing`, `-ed`, `-s`) $\to$ `play`.
   - For `studies`, `studying`: Replaces `-ies` and `-ying` $\to$ `studi`.
   - For `connected`, `connection`: Strips `-ed` and replaces `-tion` with `-t` $\to$ `connect`.
   - For `computers`: Strips plural `-s` and suffix `-er` $\to$ `comput`.
5. `df_stemming = pd.DataFrame(stem_data)`: Converts the list of dictionaries into a 2-column DataFrame (`Original Word`, `Porter Stem`).

### Output Table & Analysis
| Original Word | Porter Stem | Analysis / Linguistic Note |
| :--- | :--- | :--- |
| **playing** | `play` | Inflectional suffix `-ing` removed cleanly. |
| **played** | `play` | Past-tense inflection `-ed` removed cleanly. |
| **plays** | `play` | Third-person singular `-s` removed cleanly. |
| **studies** | `studi` | Rule replaces `-ies` with `-i` (not a valid dictionary word). |
| **studying** | `studi` | Rule reduces `-ying` to `-i` (consistent with `studies`). |
| **connected** | `connect` | Past participle suffix `-ed` removed. |
| **connection** | `connect` | Derivational noun suffix `-ion` reduced to verb base. |
| **computers** | `comput` | Suffix `-er` and `-s` stripped, leaving non-word root `comput`. |

---

## Experiment 3.2: Lemmatization Using WordNet Lemmatizer

### Background & Linguistic Theory
**Lemmatization** is a morphologically sophisticated process that uses a complete lexical database (WordNet) and morphological analysis (`Morphy`) to reduce an inflected word to its canonical base form, known as the **lemma**.

Key distinctions from stemming:
1. **Guaranteed Dictionary Form**: Lemmatization always produces a valid dictionary base word (canonical form).
2. **Part-of-Speech (POS) Sensitivity**: By default, `WordNetLemmatizer.lemmatize(word)` assumes the word is a **Noun** (`pos='n'`). If a word is a verb (e.g., `running`, `ate`, `went`) or adjective (e.g., `better`), passing the correct POS tag is essential for accurate morphological reduction.

### Code Executed
```python
words_to_lemmatize = [
    'cats', 'dogs', 'running', 'runs', 'ran', 
    'studies', 'studying', 'better', 'children', 
    'mice', 'went', 'ate', 'leaves', 'caring'
]

pos_mapping = {
    'cats': wordnet.NOUN,
    'dogs': wordnet.NOUN,
    'running': wordnet.VERB,
    'runs': wordnet.VERB,
    'ran': wordnet.VERB,
    'studies': wordnet.VERB,
    'studying': wordnet.VERB,
    'better': wordnet.ADJ,
    'children': wordnet.NOUN,
    'mice': wordnet.NOUN,
    'went': wordnet.VERB,
    'ate': wordnet.VERB,
    'leaves': wordnet.NOUN,
    'caring': wordnet.VERB
}

lemma_data = []
for word in words_to_lemmatize:
    default_lemma = lemmatizer.lemmatize(word)
    pos_tag = pos_mapping[word]
    pos_lemma = lemmatizer.lemmatize(word, pos=pos_tag)
    lemma_data.append({
        'Original Word': word,
        'Assigned POS': pos_tag,
        'Default Lemma (Noun)': default_lemma,
        'POS-Aware Lemma': pos_lemma
    })

df_lemmatization = pd.DataFrame(lemma_data)
df_lemmatization
```

### Line-by-Line & Syntax Explanation
1. `words_to_lemmatize = [...]`: 14 diverse words including regular plurals (`cats`, `dogs`), irregular plurals (`children`, `mice`, `leaves`), regular verbs (`running`, `runs`, `studies`, `caring`), irregular past-tense verbs (`ran`, `went`, `ate`), and irregular adjectives (`better`).
2. `pos_mapping = {...}`: Maps each word to its appropriate WordNet POS constant (`wordnet.NOUN = 'n'`, `wordnet.VERB = 'v'`, `wordnet.ADJ = 'a'`).
3. `default_lemma = lemmatizer.lemmatize(word)`:
   - **Syntax**: `WordNetLemmatizer.lemmatize(word: str, pos: str = 'n') -> str`
   - **Behavior**: Uses the default POS parameter `pos='n'` (Noun). Verbs like `running`, `ran`, `went`, `ate` are not reduced because WordNet looks for them in noun tables.
4. `pos_lemma = lemmatizer.lemmatize(word, pos=pos_tag)`:
   - **Behavior**: Passes the explicit POS tag. WordNet successfully reduces irregular verbs (`went` $\to$ `go`, `ate` $\to$ `eat`, `ran` $\to$ `run`) and irregular adjectives (`better` $\to$ `good`).
5. `df_lemmatization = pd.DataFrame(...)`: Combines results into a clear 4-column comparison table.

### Output Table & Analysis
| Original Word | Assigned POS | Default Lemma (Noun) | POS-Aware Lemma | Linguistic Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **cats** | Noun (`n`) | `cat` | `cat` | Regular plural noun stripped of `-s`. |
| **dogs** | Noun (`n`) | `dog` | `dog` | Regular plural noun stripped of `-s`. |
| **running** | Verb (`v`) | `running` | `run` | Default fails (not a noun); Verb POS correctly identifies base form `run`. |
| **runs** | Verb (`v`) | `run` | `run` | WordNet recognizes `runs` as a noun plural and a verb; both yield `run`. |
| **ran** | Verb (`v`) | `ran` | `run` | Irregular past tense; only resolved when `pos='v'` is supplied. |
| **studies** | Verb (`v`) | `study` | `study` | Morphological change `-ies` $\to$ `study`. |
| **studying** | Verb (`v`) | `studying` | `study` | Participle form resolved to `study` under verb POS. |
| **better** | Adjective (`a`) | `better` | `good` | Irregular comparative adjective resolved to base `good`. |
| **children** | Noun (`n`) | `child` | `child` | Irregular plural noun correctly resolved to `child`. |
| **mice** | Noun (`n`) | `mouse` | `mouse` | Irregular plural noun correctly resolved to `mouse`. |
| **went** | Verb (`v`) | `went` | `go` | Suppletive irregular past tense of `go`; requires `pos='v'`. |
| **ate** | Verb (`v`) | `ate` | `eat` | Irregular past tense of `eat`; requires `pos='v'`. |
| **leaves** | Noun (`n`) | `leaf` | `leaf` | Irregular plural noun (`-ves` $\to$ `leaf`). |
| **caring** | Verb (`v`) | `caring` | `care` | Present participle resolved to root `care`. |

---

## Experiment 3.3: Information Extraction Using Regular Expressions (Regex)

### Background & Regex Syntax Rules
Regular Expressions (Regex) allow deterministic pattern matching over text. We define robust patterns for 5 distinct structural entities.

### Code Executed
```python
input_text = """Welcome to the Natural Language Processing workshop! For more information, visit https://www.nlpworkshop.com or www.python.org. You can contact the coordinator at nlpworkshop@gmail.com or support@python.org. For registration, call +91-9876543210 or 9123456789. Follow us on social media @NLPWorkshop and @PythonLearner. Share your experience using #NLP, #Python, and #MachineLearning. You can also visit https://github.com/NLPWorkshop for the latest updates."""

emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', input_text)
raw_urls = re.findall(r'https?://[^\s,]+|www\.[^\s,]+(?<!\.)', input_text)
urls = [u.rstrip('.') for u in raw_urls]
mobile_numbers = re.findall(r'\+91-\d{10}|\b\d{10}\b', input_text)
hashtags = re.findall(r'#\w+', input_text)
mentions = re.findall(r'(?<!\S)@\w+', input_text)

print(f"1. Email Addresses ({len(emails)}):")
for item in emails:
    print(f"   - {item}")

print(f"\n2. URLs ({len(urls)}):")
for item in urls:
    print(f"   - {item}")

print(f"\n3. Mobile Numbers ({len(mobile_numbers)}):")
for item in mobile_numbers:
    print(f"   - {item}")

print(f"\n4. Hashtags ({len(hashtags)}):")
for item in hashtags:
    print(f"   - {item}")

print(f"\n5. Mentions ({len(mentions)}):")
for item in mentions:
    print(f"   - {item}")
```

### In-Depth Pattern Breakdown & Syntax Explanation

#### 1. Email Addresses Pattern
- **Pattern**: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`
  - `\b`: Word boundary ensuring the match starts at the beginning of an identifier.
  - `[A-Za-z0-9._%+-]+`: Matches one or more username characters (alphanumerics and common email special characters).
  - `@`: Matches the literal at-sign.
  - `[A-Za-z0-9.-]+`: Matches the domain host name (e.g., `gmail`, `python`).
  - `\.`: Matches the literal period separating domain and top-level domain.
  - `[A-Za-z]{2,}`: Matches top-level domains (e.g., `com`, `org`, `edu`) requiring at least 2 alphabetic letters.
- **Extracted**:
  - `nlpworkshop@gmail.com`
  - `support@python.org`

#### 2. URLs Pattern
- **Pattern**: `https?://[^\s,]+|www\.[^\s,]+` (with trailing period stripped)
  - `https?://`: Matches `http://` or `https://` (`s?` makes the 's' optional).
  - `[^\s,]+`: Matches all contiguous non-whitespace, non-comma characters forming the URL path.
  - `|`: Alternation operator allowing matching of domain URLs starting with `www.`.
  - `u.rstrip('.')`: Strips terminal sentence periods without corrupting intra-URL dots.
- **Extracted**:
  - `https://www.nlpworkshop.com`
  - `www.python.org`
  - `https://github.com/NLPWorkshop`

#### 3. Mobile Numbers Pattern
- **Pattern**: `\+91-\d{10}|\b\d{10}\b`
  - `\+91-\d{10}`: Matches Indian country code prefix `+91-` followed by exactly 10 consecutive digits (`\d{10}`).
  - `|`: Alternation operator.
  - `\b\d{10}\b`: Matches standalone 10-digit mobile numbers bounded by word boundaries.
- **Extracted**:
  - `+91-9876543210`
  - `9123456789`

#### 4. Hashtags Pattern
- **Pattern**: `#\w+`
  - `#`: Matches the literal hashtag symbol.
  - `\w+`: Matches one or more word characters (letters, digits, underscore).
- **Extracted**:
  - `#NLP`
  - `#Python`
  - `#MachineLearning`

#### 5. Mentions Pattern
- **Pattern**: `(?<!\S)@\w+`
  - `(?<!\S)`: Negative lookbehind asserting that `@` is preceded by whitespace or start of line (preventing false capture of `@gmail` inside email addresses).
  - `@`: Matches the literal mention symbol.
  - `\w+`: Matches the username handle.
- **Extracted**:
  - `@NLPWorkshop`
  - `@PythonLearner`

---

## Comparison: Stemming vs Lemmatization

| Feature | Stemming (Porter Stemmer) | Lemmatization (WordNet Lemmatizer) |
| :--- | :--- | :--- |
| **Approach** | Rule-based heuristic suffix removal. | Lexical database lookup and morphological analysis. |
| **Output Validity** | May produce non-words (`studi`, `comput`). | Always produces valid dictionary lemmas (`study`, `good`). |
| **POS Sensitivity** | Context-free (ignores Part-of-Speech). | POS-sensitive (`went` $\to$ `go` only with `pos='v'`). |
| **Irregular Words** | Fails on irregulars (`ate` $\to$ `ate`, `went` $\to$ `went`). | Resolves irregulars (`ate` $\to$ `eat`, `better` $\to$ `good`). |
| **Speed / Cost** | Very fast ($O(N)$ string slicing, low memory). | Slower ($O(N)$ hash lookups + morphological rules). |
| **Typical Use Case** | Search indexing, Bag-of-Words classification. | QA systems, Sentiment analysis, Machine translation. |

---

## Notebook Compliance Verification
- **Code comments**: `0` (Zero comments present in code cells).
- **Markdown cells**: Structured, concise headers preceding each code cell.
- **Execution state**: All cells fully executed with inline outputs and tables rendered.
