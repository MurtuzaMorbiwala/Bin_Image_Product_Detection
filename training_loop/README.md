# Training Loop Scripts

Automated training loop that runs multiple rounds of:
1. **Train YOLO** - Object detection
2. **Build CLIP** - Product embeddings  
3. **Auto-label** - Expand dataset using metadata validation
4. **Repeat** - Until no new labels

## Quick Start

```bash
# Deploy everything to Kaggle (dataset + kernel)
python training_loop/deploy.py

# Check status
python training_loop/deploy.py --status

# Download results when done
python training_loop/deploy.py --download
```

## Output:
```
Round  Labels   Products   Precision  Recall   New
--------------------------------------------------
1      354      294        0.9971     0.9915   71
2      425      340        0.9985     0.9932   45
3      470      380        0.9990     0.9945   23
...
```

## Configuration

Edit top of `kaggle_loop.py`:
```python
MAX_ROUNDS = 10      # Max training rounds
YOLO_EPOCHS = 100    # Epochs per round
YOLO_CONF = 0.5      # Min YOLO confidence
CLIP_CONF = 0.7      # Min CLIP similarity
MIN_NEW_LABELS = 5   # Stop if fewer new labels
```

## Files

| File | Description |
|------|-------------|
| `deploy.py` | Kaggle deployment script |
| `kaggle_loop.py` | Training loop (runs on Kaggle) |
| `prepare_dataset.py` | Manual dataset preparation |

## Deploy Options

```bash
python training_loop/deploy.py              # Full deploy
python training_loop/deploy.py --prepare-only  # Only prepare dataset
python training_loop/deploy.py --push-only     # Only push kernel
python training_loop/deploy.py --status        # Check status
python training_loop/deploy.py --download      # Download results
```

## Kaggle Output

- `best.pt` - Final trained model
- Training logs per round
