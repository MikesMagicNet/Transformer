# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  tokenizer.py  –  Word-Level Tokenizer for our Transformer                ║
# ║                                                                            ║
# ║  WHAT IS A TOKENIZER?                                                      ║
# ║  A tokenizer converts raw text (strings) into numbers (integers) that a    ║
# ║  neural network can process.  Our Transformer's Inputs layer               ║
# ║  (nn.Embedding) expects integer IDs, not words — so the tokenizer is the   ║
# ║  bridge between human language and the model.                              ║
# ║                                                                            ║
# ║  WHY "WORD-LEVEL"?                                                         ║
# ║  Each unique word in the training data gets its own integer ID.            ║
# ║  Example:  "the cat sat" → [4, 127, 853]                                  ║
# ║                                                                            ║
# ║  HOW IT CONNECTS TO model.py:                                              ║
# ║    Raw text                                                                ║
# ║       ↓  tokenizer.encode()     ← THIS FILE                               ║
# ║    List of integer IDs                                                     ║
# ║       ↓  Inputs (nn.Embedding)  ← model.py line 5                         ║
# ║    Dense vectors (dModel=512)                                              ║
# ║       ↓  PositionalEncoding     ← model.py line 15                        ║
# ║    Vectors + position info                                                 ║
# ║       ↓  Encoder / Decoder      ← model.py lines 120-169                  ║
# ║       ↓  ProjectionLayer        ← model.py line 171                       ║
# ║    Probability over vocab                                                  ║
# ║       ↓  tokenizer.decode()     ← THIS FILE                               ║
# ║    Human-readable text                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import re
import json
import os
from collections import Counter
from datasets import load_dataset


# ── Special Tokens ────────────────────────────────────────────────────────────
# These are reserved words that the model uses for structure, not meaning.
#
#   <PAD>  – padding; fills empty slots so every sequence is the same length.
#            The Transformer's mask will ignore these positions.
#
#   <UNK>  – unknown; replaces any word not seen during training.
#
#   <SOS>  – start-of-sequence; tells the decoder "begin generating here."
#            Fed as the very first token to the decoder input.
#
#   <EOS>  – end-of-sequence; tells the decoder "stop generating."
#            The model learns to output this when the sentence is done.

PAD_TOKEN = "<PAD>"   # ... ID will always be 0
UNK_TOKEN = "<UNK>"   # ... ID will always be 1
SOS_TOKEN = "<SOS>"   # ... ID will always be 2
EOS_TOKEN = "<EOS>"   # ... ID will always be 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


# ═══════════════════════════════════════════════════════════════════════════════
#  TOKENIZATION HELPER  – splitting raw text into a list of words
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(text):
    """
    Splits a raw string into a list of lowercase word tokens.

    Improved for the Claude Opus reasoning dataset:
      1. Lowercase everything         → "The Cat!" becomes "the cat!"
      2. Keep letters, digits, and key punctuation that carries meaning
         for code/math/reasoning (e.g. parentheses, colons, equals, etc.)
      3. Split on whitespace           → "the cat" becomes ["the", "cat"]

    Args:
        text (str): Any raw string, e.g. a conversation or article.

    Returns:
        list[str]: Individual word tokens.
                   Example: ["the", "cat", "sat", "on", "the", "mat"]
    """
    text = text.lower()                                 # ... normalize case
    # Keep alphanumeric, basic punctuation meaningful for code/math/reasoning
    # This preserves: . , : ; ( ) [ ] { } = + - * / < > ? ! @ # $ % & _ | ~ ^ '
    text = re.sub(r"[^\w\s.,;:!?(){}[\]=+\-*/<>@#$%&|~^'\"\\]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()                                # ... split on whitespace
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
#  WORD-LEVEL TOKENIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class WordTokenizer:
    """
    A word-level tokenizer that maps words ↔ integer IDs.

    Vocabulary is built from a training corpus. Words that appear fewer than
    `minFrequency` times are excluded (they become <UNK> at encode time).

    Attributes:
        vocabSize (int):  Total number of unique tokens (words + special tokens).
                          This value is passed to model.py's:
                            - Inputs(dModel, vocabSize)        → embedding table
                            - ProjectionLayer(dModel, vocabSize) → output layer

        word2id (dict):   Maps a word string → integer ID.
                          Example: {"<PAD>": 0, "the": 4, "cat": 127, ...}

        id2word (dict):   Maps an integer ID → word string  (reverse lookup).
                          Example: {0: "<PAD>", 4: "the", 127: "cat", ...}
    """

    def __init__(self, minFrequency=5):
        """
        Args:
            minFrequency (int): Minimum number of times a word must appear in
                                the training data to earn its own ID. Rare words
                                below this count are mapped to <UNK>.
                                Higher = smaller vocab (faster, less memory).
                                Lower  = bigger vocab  (captures rare words).
        """
        self.minFrequency = minFrequency  # ... threshold for keeping a word

        # --- Initialize vocabulary with special tokens first ---
        # Special tokens always occupy the first IDs: 0, 1, 2, 3
        self.word2id = {}    # ... word string → integer ID
        self.id2word = {}    # ... integer ID → word string
        for i, token in enumerate(SPECIAL_TOKENS):
            self.word2id[token] = i
            self.id2word[i] = token

        self.vocabSize = len(SPECIAL_TOKENS)  # ... starts at 4, grows as we add words


    # ─── BUILD VOCABULARY ─────────────────────────────────────────────────────

    def buildVocab(self, texts):
        """
        Scans a list of raw text strings to discover all words and their
        frequencies, then creates the word ↔ ID mappings.

        How it works:
          1. Tokenize every text into words.
          2. Count how often each word appears (Counter).
          3. Keep only words that appear >= minFrequency times.
          4. Assign each surviving word a unique integer ID.
             Words are sorted by frequency (most common first) for
             better cache locality during training.

        Args:
            texts (list[str]): A list of raw strings (e.g., Wikipedia articles).

        After calling this:
            - self.word2id  is fully populated
            - self.id2word  is fully populated
            - self.vocabSize reflects the final vocabulary size
        """
        print("Building vocabulary...")

        # Step 1 & 2: Count every word across all texts
        wordCounts = Counter()  # ... dict-like: {"the": 98234, "cat": 412, ...}
        totalTokens = 0
        for i, text in enumerate(texts):
            words = tokenize(text)
            wordCounts.update(words)  # ... adds each word's count
            totalTokens += len(words)
            if (i + 1) % 2000 == 0:
                print(f"  Scanned {i+1:,}/{len(texts):,} texts...")

        # Step 3 & 4: Filter by frequency, assign IDs
        # Sort by frequency descending — most common words get lowest IDs
        # This improves cache locality during embedding lookups
        nextId = len(SPECIAL_TOKENS)  # ... = 4
        keptCount = 0
        droppedCount = 0

        for word, count in wordCounts.most_common():
            if count >= self.minFrequency:
                self.word2id[word] = nextId
                self.id2word[nextId] = word
                nextId += 1
                keptCount += 1
            else:
                droppedCount += 1

        self.vocabSize = nextId  # ... total tokens = special + kept words

        # Compute UNK rate — what fraction of tokens will become <UNK>
        unkTokens = sum(c for w, c in wordCounts.items() if c < self.minFrequency)
        unkRate = unkTokens / max(totalTokens, 1) * 100

        print(f"  Total tokens scanned: {totalTokens:,}")
        print(f"  Words seen:    {len(wordCounts):,}")
        print(f"  Words kept:    {keptCount:,}  (appeared >= {self.minFrequency} times)")
        print(f"  Words dropped: {droppedCount:,}  (too rare → mapped to <UNK>)")
        print(f"  UNK rate:      {unkRate:.1f}% of tokens will map to <UNK>")
        print(f"  Final vocabSize: {self.vocabSize:,}")
        print(f"  (This number goes into Inputs(dModel, vocabSize) in model.py)\n")


    # ─── ENCODE: text → list of integer IDs ───────────────────────────────────

    def encode(self, text, maxLength=None, addSpecialTokens=True):
        """
        Converts a raw text string into a list of integer IDs that the
        Transformer can process.

        Steps:
          1. Tokenize the text into words.
          2. Optionally prepend <SOS> and append <EOS>.
          3. Look up each word in word2id (unknown words → <UNK> ID).
          4. Optionally truncate or pad to a fixed maxLength.

        Args:
            text (str):              Raw input string.
            maxLength (int or None): If set, sequences are truncated or padded
                                     to exactly this length. This must match the
                                     sequenceLength used in PositionalEncoding
                                     (model.py line 17).
            addSpecialTokens (bool): If True, wraps the sequence with
                                     <SOS> ... <EOS>.

        Returns:
            list[int]: Integer IDs ready to become a torch.Tensor and be fed
                       into the model.

        Example:
            >>> tok.encode("The cat sat")
            [2, 4, 127, 853, 3]       # [<SOS>, the, cat, sat, <EOS>]

            >>> tok.encode("The cat sat", maxLength=8)
            [2, 4, 127, 853, 3, 0, 0, 0]  # padded with <PAD> to length 8
        """
        words = tokenize(text)

        # Look up each word → its integer ID (or <UNK>'s ID if not in vocab)
        unkId = self.word2id[UNK_TOKEN]  # ... = 1
        ids = [self.word2id.get(word, unkId) for word in words]
        # ... .get(word, unkId) returns the word's ID if found, else unkId

        # Wrap with <SOS> at the start and <EOS> at the end
        if addSpecialTokens:
            ids = [self.word2id[SOS_TOKEN]] + ids + [self.word2id[EOS_TOKEN]]
            # ... now ids looks like: [2, ...word_ids..., 3]

        # Truncate if too long (must fit within model's sequenceLength)
        if maxLength is not None:
            ids = ids[:maxLength]  # ... chop off anything beyond maxLength

        # Pad if too short (fill remaining slots with <PAD> = 0)
        if maxLength is not None:
            padId = self.word2id[PAD_TOKEN]  # ... = 0
            paddingNeeded = maxLength - len(ids)
            ids = ids + [padId] * paddingNeeded
            # ... e.g., [2, 4, 127, 3] + [0, 0, 0, 0] for maxLength=8

        return ids


    # ─── DECODE: list of integer IDs → text ───────────────────────────────────

    def decode(self, ids, skipSpecialTokens=True):
        """
        Converts a list of integer IDs back into a human-readable string.
        This is the reverse of encode() — used to read the model's output.

        Args:
            ids (list[int]):           Token IDs from the model.
            skipSpecialTokens (bool):  If True, removes <PAD>, <SOS>, <EOS>, <UNK>
                                       from the output.

        Returns:
            str: The decoded text.

        Example:
            >>> tok.decode([2, 4, 127, 853, 3, 0, 0])
            "the cat sat"
        """
        words = []
        specialIds = set(range(len(SPECIAL_TOKENS)))  # ... {0, 1, 2, 3}

        for tokenId in ids:
            if skipSpecialTokens and tokenId in specialIds:
                continue  # ... skip <PAD>, <UNK>, <SOS>, <EOS>
            word = self.id2word.get(tokenId, UNK_TOKEN)
            # ... look up the integer → word, fallback to "<UNK>"
            words.append(word)

        return " ".join(words)  # ... glue words back together with spaces


    # ─── SAVE / LOAD  –  persist the vocabulary to disk ───────────────────────

    def save(self, path):
        """
        Saves the vocabulary to a JSON file so you don't have to rebuild
        it every time. The file stores word2id, id2word, and minFrequency.

        Args:
            path (str): File path, e.g. "vocab.json"
        """
        data = {
            "minFrequency": self.minFrequency,
            "vocabSize": self.vocabSize,
            "word2id": self.word2id,
            # ... id2word keys must be strings in JSON, so we convert
            "id2word": {str(k): v for k, v in self.id2word.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Vocabulary saved to {path}  ({self.vocabSize:,} tokens)")


    @classmethod
    def load(cls, path):
        """
        Loads a previously saved vocabulary from a JSON file.

        Args:
            path (str): File path to the saved vocab, e.g. "vocab.json"

        Returns:
            WordTokenizer: A fully initialized tokenizer ready to encode/decode.
        """
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(minFrequency=data["minFrequency"])
        tokenizer.word2id = data["word2id"]
        # ... JSON keys are always strings, so convert back to int
        tokenizer.id2word = {int(k): v for k, v in data["id2word"].items()}
        tokenizer.vocabSize = data["vocabSize"]

        print(f"Vocabulary loaded from {path}  ({tokenizer.vocabSize:,} tokens)")
        return tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING  –  Fetch Wikipedia articles from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════

def loadWikipediaDataset(language="en", date="20231101", numArticles=50000):
    """
    Downloads Wikipedia articles from HuggingFace and returns them as a
    list of raw text strings.

    The dataset: https://huggingface.co/datasets/wikimedia/wikipedia
      - Each row has: id, url, title, text
      - We only need the 'text' field (the article body)

    Args:
        language (str):     Wikipedia language code.
                            "en" = English, "fr" = French, "es" = Spanish, etc.
        date (str):         Wikipedia dump date. "20231101" is the latest.
        numArticles (int):  How many articles to load. The full English Wikipedia
                            has ~6.4 million articles — start small for testing!
                            50,000 is a good starting point.

    Returns:
        list[str]: Raw article texts.
    """
    # The config name combines the date and language: "20231101.en"
    configName = f"{date}.{language}"
    print(f"Loading Wikipedia dataset: wikimedia/wikipedia [{configName}]")
    print(f"  Requesting {numArticles:,} articles...")

    # load_dataset downloads & caches the data from HuggingFace
    # streaming=True means we don't download the entire 71.8 GB at once —
    # instead it streams articles one at a time (much friendlier on RAM/disk)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        configName,
        split="train",
        streaming=True       # ... stream instead of downloading everything
    )

    # Collect the article text from each row
    texts = []
    for i, article in enumerate(dataset):
        if i >= numArticles:
            break  # ... we have enough articles
        texts.append(article["text"])  # ... each article is a dict with 'text' key

        # Print progress every 10,000 articles
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1:,} articles...")

    print(f"  Done! Loaded {len(texts):,} articles.\n")
    return texts


def loadClaudeOpusDataset(datasetName="Roman1111111/claude-opus-4.6-10000x", numRows=9633):
    """
    Loads the Claude Opus 4.6 reasoning dataset from HuggingFace.
    
    This dataset has conversational format with reasoning traces:
      - Each row has 'messages': [{role, content, reasoning?}, ...]
      - We extract text with structural markers preserved so the model
        learns the conversation format:
          [SYSTEM] ... [USER] ... [THINK] ... [/THINK] [ASSISTANT] ...

    This teaches the model structured reasoning patterns:
      question → thinking → answer

    Args:
        datasetName (str): HuggingFace dataset identifier.
        numRows (int):     How many rows to load.

    Returns:
        list[str]: Extracted text from each conversation.
    """
    print(f"Loading dataset: {datasetName}")
    print(f"  Requesting {numRows:,} rows...")

    dataset = load_dataset(datasetName, split="train", streaming=True)

    texts = []
    skipped = 0
    for i, row in enumerate(dataset):
        if i >= numRows:
            break

        # Extract text with role markers for conversation structure
        parts = []
        for msg in row.get("messages", []):
            role = msg.get("role", "").lower()
            content = msg.get("content", "").strip()
            reasoning = msg.get("reasoning", "").strip()

            # Extract <reasoning> blocks embedded inside content
            if not reasoning and content:
                # Some rows embed reasoning inside <reasoning>...</reasoning> tags
                match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    # Remove the reasoning block from content to avoid duplication
                    content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL).strip()

            # Build structured text with role markers
            if role == "system" and content:
                parts.append(f"[SYSTEM] {content}")
            elif role == "user" and content:
                parts.append(f"[USER] {content}")
            elif role == "assistant":
                if reasoning:
                    parts.append(f"[THINK] {reasoning} [/THINK]")
                if content:
                    parts.append(f"[ASSISTANT] {content}")

        if parts:
            combined = " ".join(parts)
            # Skip very short conversations (< 20 tokens) — likely garbage
            if len(combined.split()) >= 20:
                texts.append(combined)
            else:
                skipped += 1
        else:
            skipped += 1

        if (i + 1) % 2000 == 0:
            print(f"  Loaded {i + 1:,} rows...")

    print(f"  Done! Loaded {len(texts):,} rows (skipped {skipped} too-short rows).\n")
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  –  Build the tokenizer from Wikipedia and save it
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    from config import TOKENIZER, PATHS, DATA

    # ── Configuration pulled from config.py ───────────────────────────────────
    VOCAB_PATH   = PATHS["vocab"]
    NUM_ARTICLES = TOKENIZER["vocabArticles"]
    MIN_FREQ     = TOKENIZER["minFrequency"]
    SEQ_LENGTH   = TOKENIZER["seqLengthForTest"]

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  STEP 1: Loading Data ({DATA['source']})")
    print("=" * 60)

    if os.path.exists(VOCAB_PATH):
        print(f"  Found existing vocab at '{VOCAB_PATH}', loading it instead.\n")
        tokenizer = WordTokenizer.load(VOCAB_PATH)
    else:
        if DATA["source"] == "claude-opus":
            texts = loadClaudeOpusDataset(
                datasetName=DATA["claudeDataset"],
                numRows=NUM_ARTICLES,
            )
        else:
            texts = loadWikipediaDataset(numArticles=NUM_ARTICLES)

        # ── Step 2: Build vocabulary ──────────────────────────────────────────
        print("=" * 60)
        print("  STEP 2: Building Vocabulary")
        print("=" * 60)

        tokenizer = WordTokenizer(minFrequency=MIN_FREQ)
        tokenizer.buildVocab(texts)

        # ── Step 3: Save vocabulary ───────────────────────────────────────────
        print("=" * 60)
        print("  STEP 3: Saving Vocabulary")
        print("=" * 60)

        tokenizer.save(VOCAB_PATH)

    # ── Step 4: Test encode / decode ──────────────────────────────────────────
    print("=" * 60)
    print("  STEP 4: Testing Encode / Decode")
    print("=" * 60)

    testSentence = "The Transformer model was introduced in the paper Attention Is All You Need"
    print(f"\n  Input:    \"{testSentence}\"")

    encoded = tokenizer.encode(testSentence, maxLength=SEQ_LENGTH)
    print(f"  Encoded:  {encoded[:20]}...  (showing first 20 of {len(encoded)} IDs)")

    decoded = tokenizer.decode(encoded)
    print(f"  Decoded:  \"{decoded}\"")

    # Show vocab size — this is what you pass to buildTransformer()
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  vocabSize = {tokenizer.vocabSize:<10,}                    │")
    print(f"  │  Use this value in buildTransformer():       │")
    print(f"  │    source_vocabSize = {tokenizer.vocabSize:<10,}           │")
    print(f"  │    target_vocabSize = {tokenizer.vocabSize:<10,}           │")
    print(f"  └─────────────────────────────────────────────┘")
    print()
