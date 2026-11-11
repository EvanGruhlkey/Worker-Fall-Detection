# Data sources

Third-party data used to train or evaluate the fall classifier, with the
attribution its license requires. Raw downloads live in `data/external/`
(not versioned); derived label sheets and pose files are versioned.

## CAUCAFall

- **Authors:** Jose Camilo Eraso, Elena Muñoz, Mariela Muñoz, Jesus Pinto
  (Universidad del Cauca; Universidad de la Amazonia)
- **Title:** CAUCAFall, Version 4, Mendeley Data, published 12 July 2022
- **DOI:** [10.17632/7w7fccy7ky.4](https://doi.org/10.17632/7w7fccy7ky.4)
- **Paper:** "Dataset for human fall recognition in an uncontrolled
  environment", *Data in Brief* 45 (2022) 108610
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **What we use:** the 100 `.avi` videos and the per-frame YOLO label files
  (not the PNG frames).
- **Changes we made:** fall start and impact times are derived from the
  labeled boxes (`scripts/import_caucafall.py`); poses are extracted with our
  pose pipeline and resampled to 12 fps. The original labels mark every frame
  from mid-fall to the end of the clip as "fall".

## Rig fall clips (internal)

AI-generated clips of oil rig workers in `data/input/rig_falls` and
`data/input/normal_rig_work`, labeled in `data/labels/clips.csv`. Owned by the
project team.
