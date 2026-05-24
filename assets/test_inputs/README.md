# Test input images

Six 256×256 placeholder photographs ready to drag-and-drop into the
*Live inference* tab of the Streamlit dashboard.

| File         | Source                                              |
|--------------|-----------------------------------------------------|
| cat.jpg      | https://picsum.photos/seed/cat/256/256              |
| dog.jpg      | https://picsum.photos/seed/dog/256/256              |
| car.jpg      | https://picsum.photos/seed/car/256/256              |
| plane.jpg    | https://picsum.photos/seed/plane/256/256            |
| ship.jpg     | https://picsum.photos/seed/ship/256/256             |
| horse.jpg    | https://picsum.photos/seed/horse/256/256            |

> Lorem Picsum returns random Unsplash photos seeded by the URL; the
> file names are just convenient labels — the actual content is
> arbitrary scenery / objects. The point is to show that the I-JEPA
> inference pipeline accepts any RGB photo (and that it is then
> resized to 32×32 before being patchified).

## Usage during the video demo

1. Launch the dashboard:
   ```bash
   python -m streamlit run scripts/demo_app.py
   ```
2. Open the **🧪 Live inference** tab.
3. Drag any file from this folder into the uploader.
4. The page shows: input → context patches → target patches →
   192-D encoded vector (magma heatmap).
