# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  train.py  –  Training & Validation Loop + Live Dashboard                  ║
# ║                                                                            ║
# ║  HOW TO RUN:                                                               ║
# ║    python3 train.py                                                        ║
# ║    Then open http://localhost:8080 in your browser for the live dashboard.  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import torch
import torch.nn as nn
import math
import os
import json
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from torch.utils.data import Dataset, DataLoader

from model import buildTransformer
from tokenizer import WordTokenizer, tokenize, loadWikipediaDataset, loadClaudeOpusDataset
from config import MODEL, TRAINING, DATA, PATHS, RESUME, DASHBOARD

# ── BUILD RUNTIME CONFIG from config.py ───────────────────────────────────────
# ... Flattened for easy access throughout this file.
CONFIG = {
    **MODEL,
    "epochs": TRAINING["epochs"],
    "batchSize": TRAINING["batchSize"],
    "maxLR": TRAINING["maxLR"],
    "minLR": TRAINING["minLR"],
    "warmupSteps": TRAINING["warmupSteps"],
    "labelSmoothing": TRAINING["labelSmoothing"],
    "gradClipNorm": TRAINING["gradClipNorm"],
    "logInterval": TRAINING.get("logInterval", 25),
    "numArticles": DATA["numArticles"],
    "valSplit": DATA["valSplit"],
    "vocabPath": PATHS["vocab"],
    "metricsPath": PATHS["metrics"],
    "checkpointPath": PATHS["checkpoint"],
    "dashboardPort": DASHBOARD["port"],
    "totalSteps": 0,  # ... computed at runtime
}

# ── SPECIAL TOKEN IDS ─────────────────────────────────────────────────────────
PAD_ID = 0  # ... <PAD>
UNK_ID = 1  # ... <UNK>
SOS_ID = 2  # ... <SOS>
EOS_ID = 3  # ... <EOS>


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET  –  Turns Wikipedia text into (source, target) training pairs
# ═══════════════════════════════════════════════════════════════════════════════

class WikiTextDataset(Dataset):
    """
    Converts raw Wikipedia articles into training samples for the Transformer.

    Each sample is a pair:
        encoder_input  =  first half of a text chunk   (what the model reads)
        decoder_input  =  [SOS] + second half[:-1]     (teacher forcing input)
        label          =  second half[:-1] + [EOS]     (what decoder must predict)

    The encoder sees context, the decoder learns to continue it.
    """

    def __init__(self, texts, tokenizer, seqLength):
        self.tokenizer = tokenizer
        self.seqLength = seqLength
        self.samples = []  # ... list of (encoder_ids, target_ids) tuples

        for text in texts:
            # Tokenize text → list of word strings → list of integer IDs
            words = tokenize(text)
            unkId = tokenizer.word2id.get("<UNK>", UNK_ID)
            ids = [tokenizer.word2id.get(w, unkId) for w in words]

            # Slide a window across the article, creating training pairs
            # ... each window is 2*seqLength tokens long
            # ... first half → encoder, second half → decoder
            windowSize = seqLength * 2
            for i in range(0, len(ids) - windowSize, seqLength):
                src = ids[i : i + seqLength]
                tgt = ids[i + seqLength : i + windowSize]
                self.samples.append((src, tgt))

        print(f"  Created {len(self.samples):,} training samples "
              f"from {len(texts):,} articles")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.samples[idx]
        sl = self.seqLength  # ... shorthand

        # ── Build encoder input (what the encoder reads) ──────────────────
        # Shape: (seqLength,)
        encoder_input = torch.tensor(src_ids[:sl], dtype=torch.long)

        # ── Build decoder input (teacher forcing: shifted right) ──────────
        # [SOS, tok1, tok2, ..., tok_{sl-2}]  → length = seqLength
        decoder_input = torch.tensor(
            [SOS_ID] + tgt_ids[:sl - 1], dtype=torch.long
        )

        # ── Build label (what the decoder should output) ──────────────────
        # [tok1, tok2, ..., tok_{sl-1}, EOS]  → length = seqLength
        label = torch.tensor(
            tgt_ids[:sl - 1] + [EOS_ID], dtype=torch.long
        )

        # ── Build masks ───────────────────────────────────────────────────
        # Encoder mask: 1 where NOT padding, 0 where padding
        # ... shape (1, 1, seqLength) — broadcasts over (batch, heads, query, key)
        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()

        # Decoder mask: causal (can only see past tokens) AND not padding
        # ... shape (1, seqLength, seqLength)
        decoder_padding = (decoder_input != PAD_ID).unsqueeze(0).int()
        # ... causal_mask[i][j] = 1 if j <= i (can attend to current + past)
        causal_mask = torch.tril(torch.ones(sl, sl, dtype=torch.int))
        decoder_mask = decoder_padding & causal_mask  # ... combine both

        return {
            "encoder_input": encoder_input,    # (seqLength,)
            "decoder_input": decoder_input,    # (seqLength,)
            "encoder_mask": encoder_mask,      # (1, 1, seqLength)
            "decoder_mask": decoder_mask.unsqueeze(0),  # (1, seqLength, seqLength)
            "label": label,                    # (seqLength,)
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  LEARNING RATE SCHEDULER  –  Cosine annealing with linear warmup
# ═══════════════════════════════════════════════════════════════════════════════

def getLearningRate(step, maxLR, minLR, warmupSteps, totalSteps):
    """
    Cosine annealing with linear warmup — much more stable than the paper's
    inverse-sqrt schedule, especially for small models.

    Phase 1 (warmup):  LR climbs linearly from 0 → maxLR
    Phase 2 (decay):   LR follows a cosine curve from maxLR → minLR

    This prevents the LR from crashing too fast (the old schedule's problem)
    and gives the model a smooth, predictable learning rate throughout training.
    """
    if step < warmupSteps:
        # ... linear warmup: 0 → maxLR over warmupSteps
        return maxLR * (step / max(warmupSteps, 1))
    else:
        # ... cosine decay: maxLR → minLR over remaining steps
        progress = (step - warmupSteps) / max(totalSteps - warmupSteps, 1)
        progress = min(progress, 1.0)  # ... clamp to [0, 1]
        return minLR + (maxLR - minLR) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS TRACKER  –  Records everything the dashboard needs
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """Collects training stats and saves them to a JSON file for the dashboard."""

    def __init__(self, config, metricsPath):
        self.path = metricsPath
        self.data = {
            "config": config,
            "status": "initializing",
            "currentEpoch": 0,
            "totalEpochs": config["epochs"],
            "currentBatch": 0,
            "totalBatches": 0,
            "globalStep": 0,
            "trainLosses": [],        # ... loss each batch
            "smoothedLoss": [],       # ... exponential moving avg of loss
            "valLosses": [],          # ... avg loss each epoch
            "learningRates": [],      # ... lr each batch
            "gradNorms": [],          # ... gradient L2 norm each batch
            "trainAccuracies": [],    # ... token accuracy each batch
            "epochTrainLosses": [],   # ... avg train loss each epoch
            "epochValAccuracies": [], # ... token accuracy each epoch
            "perplexities": [],       # ... e^loss each epoch
            "attentionWeights": [],   # ... 2D list for heatmap
            "attentionSrcTokens": [], # ... token labels for heatmap x-axis
            "attentionTgtTokens": [], # ... token labels for heatmap y-axis
            "predictions": [],        # ... sample outputs
            "embeddingNodes": [],     # ... [{word, x, y}, ...] for word map
            "embeddingEdges": [],     # ... [[i,j], ...] nearest-neighbor links
            "tokensPerSecond": 0,
            "elapsedSeconds": 0,
        }
        self.save()

    def update(self, **kwargs):
        self.data.update(kwargs)

    def save(self):
        """Write metrics to disk so the dashboard can read them."""
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f)
        except Exception:
            pass  # ... don't crash training if file write fails


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP  –  One epoch of training
# ═══════════════════════════════════════════════════════════════════════════════

def trainOneEpoch(model, dataloader, optimizer, criterion, device, metrics, epoch):
    """
    Runs one full pass through the training data.

    For each batch:
      1. Feed encoder_input through the encoder  → context vectors
      2. Feed decoder_input + context through the decoder → predictions
      3. Compare predictions to labels using CrossEntropyLoss
      4. Backpropagate the error and update weights
    """
    model.train()  # ... enable dropout, batch norm, etc.
    totalLoss = 0
    batchCount = len(dataloader)
    startTime = time.time()
    tokenCount = 0

    for batchIdx, batch in enumerate(dataloader):
        # Move data to the right device (CPU or GPU)
        enc_input = batch["encoder_input"].to(device)   # (B, seqLen)
        dec_input = batch["decoder_input"].to(device)   # (B, seqLen)
        enc_mask = batch["encoder_mask"].to(device)      # (B, 1, 1, seqLen)
        dec_mask = batch["decoder_mask"].to(device)      # (B, 1, seqLen, seqLen)
        label = batch["label"].to(device)                # (B, seqLen)

        # ── Forward pass ──────────────────────────────────────────────────
        # Step 1: Encode the source sequence
        encoderOut = model.encode(enc_input, enc_mask)

        # Step 2: Decode using encoder output + decoder input
        decoderOut = model.decode(encoderOut, enc_mask, dec_input, dec_mask)

        # Step 3: Project to vocabulary probabilities
        projOut = model.projection(decoderOut)  # (B, seqLen, vocabSize)

        # ── Compute loss ──────────────────────────────────────────────────
        # Reshape for CrossEntropyLoss: (B*seqLen, vocabSize) vs (B*seqLen,)
        loss = criterion(
            projOut.view(-1, projOut.size(-1)),  # ... flatten predictions
            label.view(-1)                       # ... flatten labels
        )

        # ── Backward pass ─────────────────────────────────────────────────
        optimizer.zero_grad()  # ... clear old gradients
        loss.backward()        # ... compute new gradients

        # Measure gradient norm BEFORE clipping (diagnostic metric)
        # ... if this number explodes, training is unstable
        gradNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["gradClipNorm"])
        optimizer.step()       # ... update weights

        # ── Update learning rate (cosine annealing with warmup) ───────────
        globalStep = metrics.data["globalStep"] + 1
        lr = getLearningRate(
            globalStep,
            CONFIG["maxLR"], CONFIG["minLR"],
            CONFIG["warmupSteps"], CONFIG["totalSteps"]
        )
        for paramGroup in optimizer.param_groups:
            paramGroup["lr"] = lr

        # ── Compute token accuracy ────────────────────────────────────────
        # ... what % of predicted tokens exactly match the label?
        predicted = projOut.argmax(dim=-1)        # (B, seqLen)
        nonPadMask = (label != PAD_ID)             # ignore <PAD> positions
        correct = (predicted == label) & nonPadMask
        accuracy = correct.sum().item() / max(nonPadMask.sum().item(), 1)

        # ── Track metrics ─────────────────────────────────────────────────
        batchLoss = loss.item()
        totalLoss += batchLoss
        tokenCount += enc_input.numel()
        elapsed = time.time() - startTime
        tokPerSec = tokenCount / max(elapsed, 1)

        # Exponential moving average of loss (smoothed trend line)
        # ... alpha=0.05 means ~20-batch window, smooths out the noise
        prevSmoothed = metrics.data["smoothedLoss"]
        if prevSmoothed:
            ema = 0.95 * prevSmoothed[-1] + 0.05 * batchLoss
        else:
            ema = batchLoss

        metrics.update(
            currentBatch=batchIdx + 1,
            totalBatches=batchCount,
            globalStep=globalStep,
            tokensPerSecond=round(tokPerSec),
            status="training",
        )
        metrics.data["trainLosses"].append(round(batchLoss, 4))
        metrics.data["smoothedLoss"].append(round(ema, 4))
        metrics.data["learningRates"].append(round(lr, 8))
        metrics.data["gradNorms"].append(round(gradNorm.item(), 4))
        metrics.data["trainAccuracies"].append(round(accuracy, 4))

        # Save metrics every 5 batches so dashboard stays responsive
        if (batchIdx + 1) % CONFIG["logInterval"] == 0 or batchIdx == batchCount - 1:
            metrics.save()
            print(f"  Epoch {epoch+1} | Batch {batchIdx+1}/{batchCount} "
                  f"| Loss: {batchLoss:.4f} | LR: {lr:.6f} "
                  f"| {tokPerSec:,.0f} tok/s")

    avgLoss = totalLoss / max(batchCount, 1)
    return avgLoss


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATION LOOP  –  Evaluate without updating weights
# ═══════════════════════════════════════════════════════════════════════════════

def validate(model, dataloader, criterion, device, tokenizer, metrics):
    """
    Runs the model on validation data WITHOUT updating weights.
    Also captures attention weights and sample predictions for the dashboard.
    """
    model.eval()  # ... disable dropout
    totalLoss = 0
    totalCorrect = 0
    totalTokens = 0
    batchCount = len(dataloader)

    with torch.no_grad():  # ... don't compute gradients (saves memory + speed)
        for batchIdx, batch in enumerate(dataloader):
            enc_input = batch["encoder_input"].to(device)
            dec_input = batch["decoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoderOut = model.encode(enc_input, enc_mask)
            decoderOut = model.decode(encoderOut, enc_mask, dec_input, dec_mask)
            projOut = model.projection(decoderOut)

            loss = criterion(
                projOut.view(-1, projOut.size(-1)),
                label.view(-1)
            )
            totalLoss += loss.item()

            # ── Token accuracy ────────────────────────────────────────────
            predicted = projOut.argmax(dim=-1)
            nonPadMask = (label != PAD_ID)
            totalCorrect += ((predicted == label) & nonPadMask).sum().item()
            totalTokens += nonPadMask.sum().item()

            # ── Capture attention & predictions from the FIRST batch ──────
            if batchIdx == 0:
                captureAttention(model, enc_input, dec_input, tokenizer, metrics)
                capturePredictions(projOut, enc_input, label, tokenizer, metrics)
                captureEmbeddings(model, tokenizer, metrics)

    avgLoss = totalLoss / max(batchCount, 1)
    valAccuracy = totalCorrect / max(totalTokens, 1)
    metrics.data["epochValAccuracies"].append(round(valAccuracy, 4))
    return avgLoss


def captureAttention(model, enc_input, dec_input, tokenizer, metrics):
    """
    Extracts attention weights from the first encoder layer for visualization.
    The attention heatmap shows which input words the model focuses on.
    """
    try:
        # Get attention from the first encoder layer, first sample in batch
        attn = model.encoder.layers[0].selfAttentionBlock.attention_scores
        # ... shape: (batch, heads, seqLen, seqLen)
        attn = attn[0].mean(dim=0)  # ... average across heads → (seqLen, seqLen)

        # Only show the first 16 tokens for a readable heatmap
        size = min(16, attn.size(0))
        attnSlice = attn[:size, :size].cpu().tolist()

        # Get token labels for the axes
        srcIds = enc_input[0][:size].cpu().tolist()
        srcTokens = [tokenizer.id2word.get(i, "?") for i in srcIds]

        metrics.update(
            attentionWeights=[[round(v, 4) for v in row] for row in attnSlice],
            attentionSrcTokens=srcTokens,
            attentionTgtTokens=srcTokens,  # ... self-attention: same tokens
        )
    except Exception:
        pass  # ... attention may not be available on first call


def capturePredictions(projOut, enc_input, label, tokenizer, metrics):
    """
    Grabs a few sample predictions to display on the dashboard so you can
    see what the model is actually generating vs what it should generate.
    """
    # Get predicted token IDs (highest probability token at each position)
    predicted = projOut.argmax(dim=-1)  # ... (batch, seqLen)

    samples = []
    numSamples = min(10, predicted.size(0))
    for i in range(numSamples):
        srcIds = enc_input[i].cpu().tolist()
        lblIds = label[i].cpu().tolist()
        predIds = predicted[i].cpu().tolist()

        samples.append({
            "source": tokenizer.decode(srcIds),
            "target": tokenizer.decode(lblIds),
            "predicted": tokenizer.decode(predIds),
        })

    metrics.update(predictions=samples)


def captureEmbeddings(model, tokenizer, metrics, numWords=80, kNeighbors=3):
    """
    Projects the model's learned word embeddings into 2D for visualization.
    This shows HOW the model organizes words internally — similar words
    should cluster together as training progresses.

    Steps:
      1. Grab embedding vectors for the most common words
      2. Project from dModel dimensions → 2D using PCA (SVD)
      3. Find k nearest neighbors for each word (cosine similarity)
      4. Send coordinates + connections to the dashboard
    """
    try:
        # Get the embedding weight matrix: shape (vocabSize, dModel)
        embMatrix = model.sourceEmbed.embedding.weight.detach().cpu()

        # Pick the first numWords non-special tokens (IDs 4 onward)
        # These are the most common words since they were added first
        maxId = min(numWords + 4, embMatrix.size(0))
        ids = list(range(4, maxId))  # ... skip PAD, UNK, SOS, EOS
        subset = embMatrix[ids]       # ... (numWords, dModel)
        words = [tokenizer.id2word.get(i, "?") for i in ids]

        # ── PCA via SVD: project dModel dims → 2D ─────────────────────────
        centered = subset - subset.mean(dim=0)   # ... center the data
        U, S, V = torch.svd(centered)
        coords = (centered @ V[:, :2]).tolist()   # ... project to 2D

        # Normalize coordinates to [-1, 1] range for the canvas
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        xMin, xMax = min(xs), max(xs)
        yMin, yMax = min(ys), max(ys)
        xRange = max(xMax - xMin, 1e-6)
        yRange = max(yMax - yMin, 1e-6)

        nodes = []
        for i, word in enumerate(words):
            nx = (coords[i][0] - xMin) / xRange * 2 - 1  # ... map to [-1, 1]
            ny = (coords[i][1] - yMin) / yRange * 2 - 1
            nodes.append({"word": word, "x": round(nx, 4), "y": round(ny, 4)})

        # ── Find k nearest neighbors (cosine similarity) ──────────────────
        norms = subset.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = subset / norms
        similarity = normalized @ normalized.T  # ... (numWords, numWords)

        edges = []
        for i in range(len(ids)):
            sim_row = similarity[i].clone()
            sim_row[i] = -1  # ... exclude self
            _, topk = sim_row.topk(kNeighbors)
            for j in topk.tolist():
                if [j, i] not in edges:  # ... avoid duplicate edges
                    edges.append([i, j])

        metrics.update(embeddingNodes=nodes, embeddingEdges=edges)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD SERVER  –  Serves the live HTML visualizer
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler that serves the dashboard and metrics API."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/dashboard.html"
        elif self.path == "/api/metrics":
            self.sendMetrics()
            return
        return super().do_GET()

    def sendMetrics(self):
        """Serve the metrics JSON file as an API response."""
        try:
            with open(CONFIG["metricsPath"], "r") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
        except Exception:
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # ... suppress console spam from HTTP requests


def startDashboardServer(port):
    """Launch the dashboard web server in a background thread."""
    server = HTTPServer(("", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"\n  🌐 Dashboard running at: http://localhost:{port}")
    print(f"     Open this URL in your browser to see live training stats!\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  –  Orchestrates everything
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🧠 TRANSFORMER TRAINING")
    print("=" * 60)

    # ── Device selection ──────────────────────────────────────────────────
    # ... Use MPS (Apple Silicon GPU), CUDA (NVIDIA GPU), or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU (training will be slow)")

    # ── Load tokenizer ────────────────────────────────────────────────────
    print(f"\n  Loading tokenizer from {CONFIG['vocabPath']}...")
    tokenizer = WordTokenizer.load(CONFIG["vocabPath"])

    # ── Load training data ─────────────────────────────────────────────────
    print(f"\n  Data source: {DATA['source']}")
    if DATA["source"] == "claude-opus":
        print(f"  Loading {CONFIG['numArticles']:,} rows from Claude Opus dataset...")
        texts = loadClaudeOpusDataset(
            datasetName=DATA["claudeDataset"],
            numRows=CONFIG["numArticles"],
        )
    else:
        print(f"  Loading {CONFIG['numArticles']:,} Wikipedia articles...")
        texts = loadWikipediaDataset(numArticles=CONFIG["numArticles"])

    # ── Split into train / validation ─────────────────────────────────────
    splitIdx = int(len(texts) * (1 - CONFIG["valSplit"]))
    trainTexts = texts[:splitIdx]
    valTexts = texts[splitIdx:]
    print(f"  Train articles: {len(trainTexts):,}")
    print(f"  Val articles:   {len(valTexts):,}")

    # ── Create datasets and dataloaders ───────────────────────────────────
    print(f"\n  Building training dataset...")
    trainDataset = WikiTextDataset(trainTexts, tokenizer, CONFIG["seqLength"])
    valDataset = WikiTextDataset(valTexts, tokenizer, CONFIG["seqLength"])

    trainLoader = DataLoader(
        trainDataset,
        batch_size=CONFIG["batchSize"],
        shuffle=True,   # ... randomize order each epoch
        drop_last=True,  # ... skip incomplete last batch
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=CONFIG["batchSize"],
        shuffle=False,
        drop_last=True,
    )

    # ── Build the Transformer ─────────────────────────────────────────────
    print(f"\n  Building Transformer model...")
    model = buildTransformer(
        source_vocabSize=tokenizer.vocabSize,
        target_vocabSize=tokenizer.vocabSize,
        source_sequenceLength=CONFIG["seqLength"],
        target_sequenceLength=CONFIG["seqLength"],
        N=CONFIG["N"],
        dModel=CONFIG["dModel"],
        dFF=CONFIG["dFF"],
        h=CONFIG["h"],
        dropout=CONFIG["dropout"],
    ).to(device)

    # Count total parameters
    paramCount = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {paramCount:,}")

    # ── Compute total training steps (needed for LR scheduler) ─────────
    CONFIG["totalSteps"] = len(trainLoader) * CONFIG["epochs"]
    print(f"  Total training steps: {CONFIG['totalSteps']:,}")

    # ── Optimizer & Loss ──────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["maxLR"],
        betas=(0.9, 0.98),   # ... from the paper
        eps=1e-9,
    )
    # CrossEntropyLoss: compares predicted token probabilities vs actual tokens
    # ... ignore_index=PAD_ID means we don't penalize predictions on <PAD> tokens
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=CONFIG["labelSmoothing"],
    )

    # ── Fine-tune / Resume from checkpoint ─────────────────────────────────
    # If a checkpoint exists, load the trained weights + optimizer state.
    # This lets you:
    #   - Continue training where you left off
    #   - Fine-tune on new/different data
    #   - Add more epochs on top of existing training
    startEpoch = 0
    bestValLoss = float("inf")
    if RESUME["enabled"] and os.path.exists(CONFIG["checkpointPath"]):
        print(f"\n  📂 Found existing checkpoint: {CONFIG['checkpointPath']}")
        checkpoint = torch.load(CONFIG["checkpointPath"], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        if RESUME["loadOptimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"  Loaded optimizer state (full resume mode)")
        else:
            print(f"  Fresh optimizer (fine-tune mode)")
        startEpoch = checkpoint.get("epoch", 0) + 1
        bestValLoss = checkpoint.get("val_loss", float("inf"))
        print(f"  Resuming from epoch {startEpoch} (best val_loss={bestValLoss:.4f})")
        print(f"  Training epochs {startEpoch+1} → {CONFIG['epochs']}")
    else:
        print(f"\n  🆕 Training from scratch.")

    # ── Metrics tracker ───────────────────────────────────────────────────
    metrics = MetricsTracker(CONFIG, CONFIG["metricsPath"])

    # ── Start the live dashboard ──────────────────────────────────────────
    startDashboardServer(CONFIG["dashboardPort"])

    # ── Training loop ─────────────────────────────────────────────────────
    startTime = time.time()

    for epoch in range(startEpoch, CONFIG["epochs"]):
        print(f"\n{'─' * 60}")
        print(f"  EPOCH {epoch + 1} / {CONFIG['epochs']}")
        print(f"{'─' * 60}")

        metrics.update(currentEpoch=epoch + 1, status="training")
        metrics.save()

        # ── Train ─────────────────────────────────────────────────────────
        avgTrainLoss = trainOneEpoch(
            model, trainLoader, optimizer, criterion, device, metrics, epoch
        )

        # ── Validate ──────────────────────────────────────────────────────
        metrics.update(status="validating")
        metrics.save()
        avgValLoss = validate(
            model, valLoader, criterion, device, tokenizer, metrics
        )

        # ── Record epoch-level metrics ────────────────────────────────────
        perplexity = math.exp(min(avgValLoss, 20))  # ... cap to avoid overflow
        elapsed = time.time() - startTime

        metrics.data["epochTrainLosses"].append(round(avgTrainLoss, 4))
        metrics.data["valLosses"].append(round(avgValLoss, 4))
        metrics.data["perplexities"].append(round(perplexity, 2))
        metrics.update(elapsedSeconds=round(elapsed))
        metrics.save()

        print(f"\n  ── Epoch {epoch+1} Summary ──")
        print(f"     Train Loss:  {avgTrainLoss:.4f}")
        print(f"     Val Loss:    {avgValLoss:.4f}")
        print(f"     Perplexity:  {perplexity:.2f}")
        print(f"     Elapsed:     {elapsed/60:.1f} min")

        # ── Save best checkpoint ──────────────────────────────────────────
        if avgValLoss < bestValLoss:
            bestValLoss = avgValLoss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avgValLoss,
            }, CONFIG["checkpointPath"])
            print(f"     ✅ Best model saved! (val_loss={avgValLoss:.4f})")

    # ── Done! ─────────────────────────────────────────────────────────────
    metrics.update(status="complete")
    metrics.save()
    print(f"\n{'=' * 60}")
    print(f"  ✅ Training complete!")
    print(f"  Dashboard still running at http://localhost:{CONFIG['dashboardPort']}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'=' * 60}")

    # Keep the server running so user can inspect the dashboard
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
