# 20-minute video script — I-JEPA presentation

**Total target length: 20 minutes EXACTLY.**
The script is timed in *minutes:seconds* and indicates which slides to display.
Speaking pace assumed: ~ 145 words per minute (English, calm cadence).

> Tip: record screen + camera in OBS. Use slide-by-slide timestamps below as
> teleprompter cues. Two embedded AI-generated video clips are explicitly
> placed inside the timeline — see **[VIDEO #1]** and **[VIDEO #2]**.

---

## 0:00 – 0:30 — Slide 1 (title)

Hello and welcome. My name is **<your name>** and this is my ML2 project on
**I-JEPA**, the *Joint-Embedding Predictive Architecture* for images. Over the
next 20 minutes I will walk you through the theory, my from-scratch PyTorch
implementation, and a live demo of the code on CIFAR-10.

## 0:30 – 1:00 — Slide 2 (agenda)

The talk is organized in seven parts. We start with the motivation behind
self-supervised learning, then introduce the JEPA family, dive into I-JEPA,
walk through the code, and finish with results and applications.

## 1:00 – 1:30 — Slide 3 (about the project)

The project deliverables are this PPTX, this video, and a public GitHub
repository. The code is 100 % original — no external JEPA library is used —
and everything you will see here can be reproduced on a single mid-range GPU.

## 1:30 – 2:00 — Slide 4 (GitHub link) + Slide 5 (AI tools)

Here is the public GitHub link. I will keep the link visible. As required by
the assignment, I used several AI-powered tools to build this project, and
they are all listed transparently on slide 5: Claude, ChatGPT, GitHub
Copilot, DALL·E 3, Runway, Gamma and ElevenLabs.

## 2:00 – 3:00 — Slides 6–7 (motivation, LeCun's vision)

Why self-supervised learning at all? Because the internet contains billions of
unlabeled images, but labels are slow and expensive. Yann LeCun has been
arguing for years that real intelligence arises from *prediction*, not from
reconstruction — children build a world model without anyone giving them
pixel-level labels.

## 3:00 – 4:00 — Slide 8 (three SSL families)

There are three families of SSL: generative methods like MAE, joint-embedding
methods like SimCLR or DINO, and predictive methods — JEPA. The key
difference is *where* the loss lives: in pixel space for generative,
embedding space for joint-embedding, and *latent* space across positions for
JEPA.

## 4:00 – 4:30 — **[VIDEO #1, 20 s]**

> **Embedded clip — generated with Runway Gen-3 / Sora**
> A short 20-second animation showing pixels being abstracted into latent
> embeddings, with the JEPA arrow predicting one block of latent vectors from
> another. *(See `docs/ai_prompts.md` → Video #1 prompt.)*

## 4:30 – 5:30 — Slides 9–11 (why latent space, history, vs contrastive)

Predicting in latent space avoids modelling unpredictable pixel details —
texture, sensor noise, lighting. The model can spend its capacity on
semantics. Self-supervised vision has evolved from rotation prediction to
contrastive methods to non-contrastive distillation, and now to predictive
methods like I-JEPA.

## 5:30 – 6:00 — Slide 12 (limits of generative SSL)

MAE is great, but reconstructing every pixel is costly and most of the bits
go into texture. JEPA doesn't need a pixel decoder, and it doesn't need
heavy data augmentation either.

## 6:00 – 7:00 — Slides 13–14 (generic JEPA, building blocks)

A JEPA has four building blocks: a context encoder, a target encoder, a
predictor, and an energy function. The context encoder is trainable. The
target encoder is an exponential moving average of the context encoder, with
no gradient. The predictor maps context tokens to predicted target
embeddings. The loss is a smooth-L1 distance.

## 7:00 – 8:00 — Slides 15–17 (EMA, collapse, VICReg)

Why EMA? To prevent the model from collapsing to a constant solution. The
target evolves slowly enough to be a stable target, but fast enough to keep
up with the context encoder. At small scale, no extra regularizer is needed
— the asymmetric architecture and stop-gradient are sufficient.

## 8:00 – 8:30 — Slide 18 (EMA formula)

Here is the formula and the corresponding PyTorch code: a simple Polyak
averaging applied to every parameter of the target encoder, inside a
`torch.no_grad()` block.

## 8:30 – 9:30 — Slides 19–22 (loss, inputs, vs MAE, scope)

The loss is smooth-L1 in latent space. JEPA can be applied to any signal that
can be split into two parts: images, video, audio, and even multimodal
inputs. Compared to MAE, I-JEPA is lighter, faster, and reaches similar or
better linear-probing accuracy.

## 9:30 – 10:30 — Slides 23–24 (I-JEPA bird's-eye, multi-block masking)

In I-JEPA, we sample one large context block and four smaller target blocks.
We remove every patch that is also a target from the context, so the two
sets are disjoint. The target encoder runs on the *full* image — it sees
everything — but the context encoder only sees the context patches.

## 10:30 – 11:30 — Slides 25–26 (ViT, predictor)

The encoder is a Vision Transformer Tiny: 6 layers, embedding dimension
192, 3 attention heads. The predictor is a smaller transformer: 4 layers,
hidden dimension 96, 3 heads. It receives the context tokens, appends
learnable mask tokens at the target positions, and outputs the predicted
target embeddings.

## 11:30 – 12:30 — Slide 27 (training pipeline diagram) + Slide 28 (hyperparams)

Here is the full training pipeline as a diagram. AdamW with weight decay
0.05, learning rate 1.5e-3 with a 5-epoch warm-up, cosine decay,
batch size 256, 50 epochs, EMA momentum from 0.996 to 1.0.

## 12:30 – 13:00 — Slides 29–30 (scope reduction, why it still works)

Compared to the paper, we use ViT-Tiny instead of ViT-Huge and CIFAR-10
instead of ImageNet — but the loss curve, linear probing trend, and t-SNE
clusters all behave the way the theory predicts.

## 13:00 – 14:30 — **[CODE DEMO PART 1]** — Slides 31–34

> **Switch to your IDE / terminal.** Show the repository tree, then walk
> through `src/models/encoder.py` and `src/models/predictor.py`.

I'll switch to my IDE now. Here is the repository layout. Let me open
`encoder.py` first — this is the Vision Transformer backbone. You can see the
patch embedding, the 2-D sin–cos positional embedding, and a stack of six
transformer blocks. Crucially, the `forward` method accepts an optional mask
argument that gathers only a subset of patches — this is what lets us run
the encoder on the context patches only.

Now `predictor.py`. The predictor projects encoder tokens down to dimension
96, appends learnable mask tokens at the target positions, runs four
transformer blocks, and projects back up to 192.

## 14:30 – 16:00 — **[CODE DEMO PART 2]** — Slides 35–38

> **Walk through `masking.py`, `ijepa.py`, `train.py` and `evaluate.py`.**

`masking.py` is the most important data-side file. For every sample we
sample 4 target rectangles and one context rectangle on the patch grid, then
remove the targets from the context. The collator returns the patch index
tensors used by the encoder and the predictor.

`ijepa.py` is the wrapper that orchestrates everything: forward pass on the
context encoder with the context indices, forward pass on the target encoder
on the full image, gather, smooth-L1 loss. The EMA update is in
`update_target` and is called after every optimizer step.

`train.py` is a clean PyTorch training loop. `evaluate.py` extracts features
from the frozen target encoder and trains a single linear classifier — this
is the standard linear-probing protocol.

## 16:00 – 17:30 — **[CODE DEMO PART 3]** — running the demo, slides 43–48

> **Switch to terminal.** Run:
>   `python scripts/demo_masking.py --n 4`
>   `python scripts/plot_loss.py`
>   `python scripts/tsne_embeddings.py --checkpoint ./checkpoints/ijepa_last.pt`

Let me now run the demo scripts. First, the masking demo — for each input
image you can see the context patches and the target patches, perfectly
disjoint. Second, the training-loss curve from a real 50-epoch run; it
decreases smoothly with no collapse. Third, the t-SNE plot of 2 000 test
embeddings, colored by ground-truth class — the model has clearly learned
class-discriminative features without ever seeing a single label.

## 17:30 – 18:00 — **[VIDEO #2, 20 s]**

> **Embedded clip — generated with Sora / Runway Gen-3**
> A short 20-second animation showing the t-SNE clusters forming as training
> progresses. *(See `docs/ai_prompts.md` → Video #2 prompt.)*

## 18:00 – 18:30 — Slide 49 (results table)

Linear probing accuracy: random init around 10 %, SimCLR around 68 %, MAE
around 70 %, I-JEPA around 72 %. So at this scale, I-JEPA matches or beats
the alternatives — with no contrastive loss and no pixel decoder.

## 18:30 – 19:00 — Slides 50–52 (lessons, applications, V-JEPA)

The take-aways: predictive learning in latent space works, multi-block
masking is non-trivial but essential, and EMA + stop-gradient are enough to
prevent collapse at small scale. The same idea has been extended to video
in V-JEPA (2024), and it underpins LeCun's roadmap toward autonomous
machine intelligence.

## 19:00 – 19:30 — Slides 53–54 (limitations, AI tools recap)

Limitations: hyperparameters are sensitive on small datasets, the predictor
is hard to interpret, and there is no native generative capability. As for
AI tools, the full list is on slide 54 — Claude, ChatGPT, Copilot, DALL·E,
Runway, Gamma and ElevenLabs.

## 19:30 – 20:00 — Slide 55 (thank you)

Thank you for watching. The code is available at the GitHub link, and the
recording will be uploaded to Google Drive with public read access.
Questions are very welcome.

---

## Recording checklist

- [ ] Record at 1080p, 30 fps minimum, microphone bias-corrected
- [ ] Insert **VIDEO #1** at 4:00 (20 s)
- [ ] Insert **VIDEO #2** at 17:30 (20 s)
- [ ] Stay between 19:55 and 20:05 total — re-record if drift > 10 s
- [ ] Upload to Google Drive, change visibility to "Anyone with the link"
- [ ] Paste the share link on the final slide before re-exporting the PPTX
