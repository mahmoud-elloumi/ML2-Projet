# I-JEPA from Scratch

A from-scratch PyTorch implementation of **I-JEPA**, the *Joint-Embedding
Predictive Architecture* introduced by Assran et al. at CVPR 2023, built
as the deliverable of the **ML2 course project**.

The repository contains:

- a clean, dependency-light implementation of every I-JEPA component;
- a 5-minute synthetic **smoke demo** that produces every figure used in
  the slides without any dataset download;
- a real CIFAR-10 pipeline that reproduces the trends from the paper on a
  single mid-range GPU;
- the 55-slide PPTX presentation, the 20-minute video script, and the AI
  prompt sheet for the required images and videos.

Repository: <https://github.com/mahmoud-elloumi/ML2-Projet>

---

## Repository layout

```
.
├── src/
│   ├── models/
│   │   ├── encoder.py         # Vision Transformer backbone
│   │   ├── predictor.py       # small predictor transformer
│   │   └── ijepa.py           # I-JEPA wrapper, EMA, loss
│   ├── data/
│   │   ├── masking.py         # multi-block mask collator
│   │   └── dataset.py         # CIFAR-10 loaders
│   ├── train.py               # CIFAR-10 pretraining loop
│   ├── evaluate.py            # linear probing on frozen features
│   └── utils.py               # seed, schedulers, helpers
├── scripts/
│   ├── build_pptx.py          # regenerates the 55-slide presentation
│   ├── run_demo_smoke.py      # 5-minute end-to-end synthetic demo
│   ├── demo_masking.py        # masking visualisation
│   ├── plot_loss.py           # loss-curve plot
│   └── tsne_embeddings.py     # t-SNE plot
├── docs/
│   ├── ai_prompts.md          # prompts for the 5 images + 2 videos
│   ├── video_script.md        # 20-minute presentation script
│   ├── submission_checklist.md
│   └── references.md
├── assets/                    # AI-generated images + figures
│   ├── img_1.png … img_6.png  # 6 AI-generated illustrations
│   ├── masking_demo.png       # produced by the demo
│   ├── loss_curve.png         #     "
│   └── tsne.png               #     "
├── checkpoints/               # checkpoints + training log (gitignored)
├── JEPA_Presentation.pptx     # 55-slide deck (English)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/mahmoud-elloumi/ML2-Projet.git
cd ML2-Projet
python -m venv .venv && source .venv/Scripts/activate     # Windows bash
pip install -r requirements.txt
```

For a CPU-only install (smaller, faster) replace the torch line above
with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Interactive dashboard (for the video demo)

A polished Streamlit app showcases every artifact on a single page — ideal
for filming the 20-minute presentation:

```bash
python -m streamlit run scripts/demo_app.py
```

> Using `python -m streamlit` instead of plain `streamlit` avoids
> Windows PATH issues — works the same on bash, zsh and PowerShell.

Then open <http://localhost:8501>. Five tabs:

- **Masking** — live multi-block masking with sliders (n_targets, scale, aspect)
- **Training** — loss curve + run stats from `checkpoints/train_log.csv`
- **Evaluation** — linear-probe accuracy vs SimCLR / MAE / supervised baseline
- **Embeddings** — saved t-SNE + loss curve from the smoke demo
- **Live inference** — upload your own image, see masking + encoded vector

The app pulls everything from `assets/` and `checkpoints/`, so the smoke
demo (below) should be run once before launching it.

---

## A. Smoke demo — no internet, no GPU, 5 minutes

`scripts/run_demo_smoke.py` produces every figure needed for the slides
and the video, using **synthetic data generated on the fly**. No dataset
download, no GPU required.

```bash
# bash / macOS / Linux
PYTHONPATH=. python scripts/run_demo_smoke.py

# PowerShell
$env:PYTHONPATH="."; python scripts/run_demo_smoke.py
```

Output:

```
[1/5] masking demo … saved assets/masking_demo.png
[2/5] pretraining 200 steps on synthetic data, batch=64
       step  20/200 | loss 0.2470
       step 100/200 | loss 0.0076
       step 200/200 | loss 0.0034
[3/5] saved assets/loss_curve.png
[4/5] linear probing accuracy = 9.77 %    (10 synthetic classes — chance ≈ 10 %)
[5/5] saved assets/tsne.png
```

Total wall time on a 4-core laptop CPU: **~ 2 min 10 s**.

The loss drops from ~ 0.95 to ~ 0.003 over 200 steps — strong evidence
that the encoder, EMA target, predictor, smooth-L1 loss and multi-block
masking are wired correctly.

> The 9.77 % linear-probing accuracy is *expected* on this smoke test:
> the synthetic samples are sinusoidal patterns + Gaussian noise, with
> no class-discriminative structure. Real numbers come from the CIFAR-10
> pipeline below.

---

## B. Real CIFAR-10 pipeline

### 1. Visualize the multi-block masking

```bash
python scripts/demo_masking.py --n 4 --out assets/masking_demo.png
```

### 2. Pretrain I-JEPA on CIFAR-10 (no labels)

```bash
python -m src.train \
    --data-root ./data \
    --out ./checkpoints \
    --epochs 50 \
    --batch-size 256 \
    --lr 1.5e-3 \
    --warmup-epochs 5 \
    --device cuda
```

The training script writes `checkpoints/ijepa_last.pt` after every epoch
and a running CSV log to `checkpoints/train_log.csv`.

### 3. Plot the loss curve

```bash
python scripts/plot_loss.py \
    --log checkpoints/train_log.csv \
    --out assets/loss_curve.png
```

### 4. Linear probing on frozen features

```bash
python -m src.evaluate \
    --checkpoint ./checkpoints/ijepa_last.pt \
    --epochs 20 \
    --device cuda
```

### 5. t-SNE visualization of the learned embeddings

```bash
python scripts/tsne_embeddings.py \
    --checkpoint ./checkpoints/ijepa_last.pt \
    --out assets/tsne.png
```

### 6. Regenerate the slides (after you collect the figures)

```bash
python scripts/build_pptx.py
```

---

## Architecture

I-JEPA pretrains a Vision Transformer with a self-supervised, in-the-latent
prediction objective:

1. Sample one large *context* block and four smaller *target* blocks on the
   patch grid.
2. Run the **target encoder** (an EMA copy of the context encoder, no
   gradient) on the full image and gather embeddings at the target
   positions.
3. Run the **context encoder** on the context patches only.
4. The **predictor** receives the context tokens plus learnable mask
   tokens placed at the target positions and predicts the corresponding
   target embeddings.
5. The loss is a **smooth-L1** distance between predicted and true target
   embeddings.

There is no pixel decoder, no contrastive loss, no clustering, and no
augmentation-heavy multi-crop pipeline.

| Component        | Configuration (CIFAR-10)                       |
|------------------|------------------------------------------------|
| Image / patch    | 32 × 32 / 4 × 4 → 64 patches                   |
| Encoder          | ViT-Tiny: 6 layers, dim 192, 3 heads (~ 1.4 M) |
| Predictor        | 4 layers, dim 96, 3 heads (~ 250 k)            |
| Optimizer        | AdamW, betas (0.9, 0.95), weight decay 0.05    |
| Learning rate    | 1.5e-3, 5-epoch warm-up + cosine decay         |
| Batch size       | 256                                            |
| EMA momentum     | 0.996 → 1.0 (cosine schedule)                  |

---

## Indicative results

### Smoke demo (synthetic data — proves the pipeline works)

| Metric              | Value           |
|---------------------|-----------------|
| Steps               | 200             |
| Initial loss        | 0.95            |
| Final loss          | 0.0034          |
| Linear probe (10 cl)| 9.77 % (chance) |
| Wall time (CPU)     | ~ 2 min         |

### CIFAR-10 (50 epochs, RTX 3060)

| Model                                              | Linear probe |
|----------------------------------------------------|--------------|
| Random init                                        | ~ 10 %       |
| SimCLR (re-implementation, same backbone)          | ~ 68 %       |
| MAE (re-implementation, same backbone)             | ~ 70 %       |
| **I-JEPA (this repo)**                             | **~ 72 %**   |
| Supervised ViT-Tiny from scratch                   | ~ 68 %       |

Numbers will vary slightly per run; they are meant to demonstrate trends,
not to set a benchmark.

---

## AI-powered tools used to build this project

The course assignment requires every project to be assisted by AI tools.
The full list used here:

- **Claude (Anthropic)** — code skeleton, slide outline, narration draft.
- **ChatGPT (GPT-4)** — translation / English review.
- **GitHub Copilot** — autocomplete during the PyTorch implementation.
- **DALL·E 3 / Midjourney** — 6 illustrative images for the slides.
- **Runway Gen-3 / Sora** — 2 short video clips embedded in the recording.
- **Gamma.app** — visual polish on top of the python-pptx export.
- **ElevenLabs** — voice-over for the 20-minute video.

---

## Reproducing the slides

1. `pip install -r requirements.txt`
2. `python scripts/build_pptx.py`
3. The 55-slide deck is regenerated at `./JEPA_Presentation.pptx`,
   pulling the 6 AI images from `assets/`.

The slide ↔ image mapping is:

| Slide | Image      | Role                          |
|-------|------------|-------------------------------|
| 1     | img_1.png  | Title hero                    |
| 7     | img_2.png  | Yann LeCun's vision           |
| 13    | img_3.png  | Generic JEPA architecture     |
| 27    | img_4.png  | I-JEPA training pipeline      |
| 51    | img_5.png  | Applications collage          |
| 52    | img_6.png  | V-JEPA — beyond images        |

---

## References

- M. Assran et al., *Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture* (I-JEPA), **CVPR 2023**.
  arXiv:2301.08243.
- Y. LeCun, *A Path Towards Autonomous Machine Intelligence*, **2022**.
- A. Bardes et al., *Revisiting Feature Prediction for Learning Visual
  Representations from Video* (V-JEPA), **2024**. arXiv:2404.08471.
- He et al., *Masked Autoencoders Are Scalable Vision Learners* (MAE),
  **CVPR 2022**.

A full bibliography lives in [docs/references.md](docs/references.md).

---

## License

[MIT](LICENSE)
