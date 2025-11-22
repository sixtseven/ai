# Video segmentation + descriptor pipeline (SAM + detector)

This repository provides a starter pipeline to segment people and luggage in a video using a detection model + Segment Anything (SAM). It outputs per-frame masks, counts of luggage items, and non-sensitive descriptors for people (dominant clothing color, accessories, bounding boxes). A visualization video and a JSON report are produced.

Important safety note
--
This project explicitly does NOT implement or support inferring protected attributes such as race/ethnicity, religion, or political beliefs from images. It also avoids producing automated determinations of gender, age, or socioeconomic status because such inferences are highly error-prone, ethically fraught, and can cause harm. If you need higher-level analytics, prefer non-sensitive descriptors (clothing color, accessories, pose, groupings) and obtain informed consent where required.

What this repo provides
--
- `scripts/process_video.py`: main script to process a video, segment people and luggage, count bags, and produce a JSON report and optional visualization.
- `requirements.txt`: Python dependencies (install in a virtual environment).

How it works (high level)
--
1. A pretrained detector (Faster R-CNN) finds candidate bounding boxes for people and luggage-related COCO classes.
2. For each box, SAM refines a high-quality segmentation mask.
3. For people, we compute non-sensitive descriptors such as dominant clothing color and detected accessories (backpack, handbag, suitcase). We also group nearby people as likely travel groups using simple heuristics.
4. The script writes a JSON with counts and per-person+bag data and optionally saves a visualization video.

Setup
--
1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Obtain a SAM checkpoint file. Download a SAM checkpoint (with `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`) and place it somewhere accessible. Pass its path to the script via `--sam-checkpoint`.

Usage
--
Process a video (example):

```bash
python scripts/process_video.py --video input.mp4 --sam-checkpoint /path/to/sam_vit_h.pth --out report.json --vis out.mp4 --frame-skip 5
```

Notes and next steps
--
- This is a starter scaffold — accuracy depends on the detector, SAM checkpoint, and video quality. Improve by using a stronger detector (e.g., YOLOv8) and by adding a keypoint model for richer pose analysis.
- If you need face-level analytics, add explicit consent and strong privacy protections (face blurring, on-device processing).

License
--
Use according to the licenses of the included models and libraries.
