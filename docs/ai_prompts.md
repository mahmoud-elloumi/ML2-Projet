# AI-generation prompts

Copy-paste these prompts into the corresponding AI tool to produce the
visual assets required by the assignment.

> Required: at least **5 AI-generated images** in the PPTX and at least
> **2 AI-generated videos** in the recording.

---

## Images (5+) — DALL·E 3 / Midjourney

### Image #1 — *Hero image for slide 1 (title)*

> Tool: **DALL·E 3** (16:9, photorealistic). Place behind the "I-JEPA"
> title.

```
A clean, modern AI / neural-network hero illustration in a dark navy and
emerald green palette. Two abstract glowing brain-like neural structures
face each other; arrows of light flow from the left structure into a
floating latent space made of small green geometric tokens, where one
token is being predicted from another. Deep cinematic lighting, subtle
particle effects, no text, no logos, very high detail, 16:9 aspect ratio.
```

### Image #2 — *Slide 7 — LeCun's vision*

> Tool: **Midjourney**. Drop on the right half of slide 7.

```
A photorealistic split-screen scene. Left side: a young child watching a
ball roll off a table, intuitively predicting where it will land — soft
warm sunlight, depth-of-field. Right side: an abstract neural network of
glowing green nodes producing a "predicted future state" cloud. The two
halves connect via a thin luminous arrow. Editorial illustration style,
muted palette, 16:9, no text.
```

### Image #3 — *Slide 13 / 14 — Generic JEPA architecture*

> Tool: **DALL·E 3**. Use as the background of slide 13 or 14, faded.

```
A clean technical schematic on a light slate background. Two neural-network
encoders drawn as stacks of translucent transformer blocks. The left
encoder f_theta receives an "x" patch image; the right encoder f_target
receives a "y" patch image. Both produce vector embeddings drawn as small
green spheres. A predictor module g_phi fuses the left embedding with a
positional descriptor z and outputs a predicted embedding. A dashed
"stop-gradient" symbol is drawn between the right encoder and the loss.
Vector-art style, minimal palette of navy + emerald, 16:9, labeled but
clean.
```

### Image #4 — *Slide 27 — Training pipeline (decorative side image)*

> Tool: **Midjourney**. Place as a small decorative figure, not the main
> diagram (the main diagram is built in the slide).

```
A futuristic data-flow illustration: glowing image patches enter from the
left, half of them diverted into a "context encoder" cube and half into a
"target encoder" cube. The output of the context encoder flows into a
smaller "predictor" cube, which produces predicted vectors that are
matched against the target encoder vectors via a glowing comparison node
labeled "Smooth-L1". Cinematic, dark teal background, neon-green accents,
isometric perspective, 16:9, no text inside the cubes.
```

### Image #5 — *Slide 51 — Applications collage*

> Tool: **DALL·E 3**. 4-tile collage for slide 51.

```
A 2x2 collage of photorealistic images, each tile sharing a unified emerald
green tint and dark navy borders:
1. Top-left: a robotic arm grasping a household object on a kitchen counter.
2. Top-right: a medical CT scan being analyzed by an AI model.
3. Bottom-left: a satellite view of a city overlaid with AI segmentation
   masks.
4. Bottom-right: an autonomous car perceiving its surroundings, with
   bounding boxes around pedestrians and vehicles.
Each tile uses the same color grading. 16:9 final aspect ratio.
```

### *(Optional)* Image #6 — *Slide 52 — V-JEPA video extension*

> Tool: **Midjourney**. Bonus image for slide 52.

```
A futuristic concept art of a deep-learning model watching a sequence of
short video frames. The frames float in 3-D space, with a glowing
spatio-temporal mask cube highlighting four target tubes inside the
sequence. Behind, an abstract latent space of green vectors is being
predicted. Cinematic lighting, navy + emerald palette, 16:9.
```

---

## Videos (2+) — Runway Gen-3 / Sora / Pika

### Video #1 — *Inserted at 4:00 in the recording (~ 20 s)*

> Tool: **Runway Gen-3** or **Sora**. Length: 20 seconds. Resolution:
> 1080p.

**Prompt**:
```
A 20-second animated explainer in a clean, modern motion-graphics style.
Pixels of a real photograph (e.g. a dog) gradually decompose into glowing
patches, then each patch is abstracted into a green latent vector. The
camera pans across two groups of vectors: a larger "context" group on the
left, a smaller "targets" group on the right. An animated arrow from the
context group reaches across and predicts each target vector with a soft
glowing burst. Background: deep navy with subtle particle effects.
Smooth, professional motion-graphics, no on-screen text, 16:9, 1080p,
ambient soft synth audio (or silent).
```

### Video #2 — *Inserted at 17:30 in the recording (~ 20 s)*

> Tool: **Sora** or **Runway Gen-3**. Length: 20 seconds. Resolution:
> 1080p.

**Prompt**:
```
A 20-second animation showing a t-SNE embedding space evolving over time.
At t=0 the points are a chaotic random cloud. As an invisible "training
clock" advances, the points drift apart and self-organize into ten
distinct, color-coded clusters labeled implicitly by shape. The camera
slowly orbits the cloud. Style: clean scientific visualization, dark navy
background, ten neon-bright cluster colors, soft motion-blur trails when
points move, 16:9, 1080p, silent (or soft cinematic synth).
```

### *(Optional)* Video #3 — *Bonus B-roll for slide 1 / outro*

> Tool: **Pika** or **Runway Gen-3**. Length: 10 seconds.

**Prompt**:
```
A slow camera push-in on the words "I-JEPA" written in glowing emerald
green vector lines, slowly assembling from floating transformer blocks
inside a dark navy room. Cinematic, slow, calm. 16:9, 1080p, no on-screen
text other than the gradually forming letters.
```

---

## How to use these prompts

1. **Open the AI tool** of your choice (DALL·E inside ChatGPT, Midjourney
   on Discord, Runway / Sora in their web app).
2. **Paste the prompt** exactly. Add `--ar 16:9 --v 6` for Midjourney.
3. **Download the result** as a PNG (images) or MP4 (videos).
4. **Save** all images to `assets/img_1.png` … `assets/img_5.png` and
   videos to `assets/video_1.mp4`, `assets/video_2.mp4`.
5. **Insert** them into the corresponding slides of `JEPA_Presentation.pptx`
   (or its Gamma-polished version) and into the video timeline.

Keep the original tool name in the slide footnote — your professor can ask
you which tool produced which asset.
