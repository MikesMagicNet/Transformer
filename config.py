# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  config.py  –  Single source of truth for ALL project settings             ║
# ║                                                                            ║
# ║  Edit this file to change anything about training, data, model size, etc.  ║
# ║  Every other file imports from here — you never need to hunt for settings. ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
#  These define the size/capacity of the Transformer.
#  Changing these creates a NEW model — old checkpoints won't be compatible.
# ══════════════════════════════════════════════════════════════════════════════

MODEL = {
    "dModel": 200,      # ... embedding dimension (paper=512, smaller = faster)
    "dFF": 512,         # ... feed-forward hidden size (paper=2048, typically 4x dModel)
    "h": 4,             # ... number of attention heads (must divide dModel evenly)
    "N": 4,             # ... number of encoder & decoder layers (paper=6, was 2)
    "dropout": 0.1,     # ... dropout probability (0.1 = drop 10% of connections)
    "seqLength": 85,    # ... max tokens per sequence (longer for reasoning text)
}


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
#  These control HOW the model learns. Safe to change between runs.
# ══════════════════════════════════════════════════════════════════════════════

TRAINING = {
    "epochs": 5,            # ... number of full passes through the data
    "batchSize": 30,        # ... samples per training step
    "maxLR": 4e-5,          # ... peak learning rate (lower for deeper model)
    "minLR": 1e-3,          # ... floor learning rate (end of cosine decay)
    "warmupSteps": 500,     # ... more warmup for deeper model (prevents early instability)
    "labelSmoothing": 0.1,  # ... prevents overconfidence (from the paper)
    "gradClipNorm": 1.2,    # ... max gradient norm (prevents exploding gradients)
    "logInterval": 10,      # ... print + save metrics every N batches (5=verbose, 25=normal, 100=quiet)
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
#  Controls what data the model trains on.
# ══════════════════════════════════════════════════════════════════════════════

DATA = {
    "source": "claude-opus",          # ... "wikipedia" or "claude-opus"
    "numArticles": 7000,              # ... all rows for claude-opus (or article count for wiki)
    "valSplit": 0.1,                  # ... fraction reserved for validation (0.1 = 10%)
    # Wikipedia settings (used when source = "wikipedia")
    "wikiDataset": "wikimedia/wikipedia",
    "wikiConfig": "20231101.en",
    # Claude Opus settings (used when source = "claude-opus")
    "claudeDataset": "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k", #updated to more better data to see if we can improve the model
}


# ══════════════════════════════════════════════════════════════════════════════
#  TOKENIZER
#  Controls vocabulary building. Rebuild vocab.json after changing these.
# ══════════════════════════════════════════════════════════════════════════════

TOKENIZER = {
    "minFrequency": 15,         # ... minimum word count (lower for smaller datasets)
    "vocabArticles": 7000,      # ... rows to scan (use all Claude Opus data)
    "seqLengthForTest": 512,    # ... sequence length used during tokenizer test
}


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
#  Where files get saved. All relative to the project root.
# ══════════════════════════════════════════════════════════════════════════════

PATHS = {
    "vocab": "vocab.json",            # ... tokenizer vocabulary
    "checkpoint": "checkpoint.pt",    # ... trained model weights
    "metrics": "metrics.json",        # ... dashboard metrics (auto-generated)
}


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME / FINE-TUNE
#  Controls whether training starts fresh or continues from a checkpoint.
# ══════════════════════════════════════════════════════════════════════════════

RESUME = {
    "enabled": True,       # ... False = train from scratch with new dataset
                            # ... True = load checkpoint if it exists
                            # ... Set to False + delete checkpoint.pt for a clean start

    "loadOptimizer": False, # ... True = resume optimizer state (same data, more epochs)
                            # ... False = fresh optimizer (fine-tuning on new/different data)
}


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD = {
    "port": 8080,           # ... http://localhost:8080
    "pollInterval": 2000,   # ... ms between metric updates
}
