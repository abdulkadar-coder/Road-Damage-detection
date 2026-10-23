# AI-Based Road Damage Detection from Uploaded Images and Videos

A local Streamlit application that analyzes uploaded road images and videos with a YOLOv8n object detector fine-tuned on RDD2022. It accepts JPG, JPEG, and PNG images plus MP4, AVI, and MOV videos; annotates damage detections; and exports annotated media and per-detection CSV files.

The repository does not bundle dataset files or trained weights. Generic COCO-pretrained YOLOv8n weights are only a training initialization and are never used by app inference. Fine-tune RDD2022 to create the trained checkpoint before running predictions.

## Features

- Upload a single JPG, JPEG, or PNG image, or an MP4, AVI, or MOV video. There is no webcam or live-stream support.
- Adjustable confidence threshold, frame interval, and inference image size.
- Annotated image preview/download and annotated video preview/download with official RDD class codes, confidence, and bounding boxes. Video output preserves input duration and frame rate; frames between sampled frames pass through without annotations.
- Detection counts and class, severity, and timeline charts. Counts represent per-frame observations, not unique physical sites.
- CSV export includes time in seconds (blank for an image), frame index (blank for an image), class, confidence, bounding-box coordinates, normalized box area, and heuristic severity.
- Temporary video files are removed after processing, including when an error occurs. Uploaded images are decoded in memory.


## Model architecture and checkpoint contract

The training pipeline builds an Ultralytics YOLOv8n object detector from the generic yolov8n.pt initialization and fine-tunes it for four classes in ID order: 0=D00, 1=D10, 2=D20, 3=D40. App inference requires the trained checkpoint at models/road_damage_yolov8n.pt. scripts/train.py validates the saved best.pt class map and copies it to this path after successful training. A COCO checkpoint, segmentation/classification model, empty or corrupted file, or different class map is rejected. No random or generic-weight fallback is used.

OpenCV decodes images and video frames into BGR uint8 HWC arrays. The app passes those arrays to Ultralytics YOLO.predict(); Ultralytics handles letterbox resizing, channel conversion, tensor creation, and input scaling. Predictions use Results.boxes.xyxy, conf, and cls, with boxes in original-image pixel coordinates. Results.plot() creates the annotated BGR image. This follows the official Ultralytics prediction API: https://docs.ultralytics.com/modes/predict/. The app offers Auto, CPU, and CUDA device 0; Auto leaves device choice to Ultralytics.

## Damage classes and severity estimate

The detector uses the four CRDDC2022 categories documented by the RDD2022 release:

| Code | Class |
| --- | --- |
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

Severity is a **heuristic estimate**, not an engineering condition assessment. Its visible rule is: potholes and alligator cracks receive a one-point class prior; normalized box area adds 0 points below 0.5%, 1 point from 0.5% to below 3%, and 2 points at 3% or above. A score of 0 is Minor, 1 is Moderate, and 2 or more is Severe. Box size depends on camera distance and perspective, so these labels must not be used as a substitute for field inspection.

Repeated detections of the same damage in different sampled frames count as separate observations. No tracking or unique-site deduplication is performed.

## Windows setup and run

Use CPython 3.12 on Windows. Install it from python.org and enable the PATH option. PyTorch currently lists Windows support for Python 3.9–3.12: https://docs.pytorch.org/get-started/locally/. The py launcher is optional. In PowerShell from the project directory, run:

```powershell
python --version
# Confirm this reports Python 3.12.x, then create the environment:
python -m venv .venv312
# If the py launcher is missing and python is not Python 3.12, use the full path to the installed Python 3.12 executable:
# & "C:\Path\To\Python312\python.exe" -m venv .venv312
.\.venv312\Scripts\Activate.ps1
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501`. The model checkpoint must be present at:

```text
models/road_damage_yolov8n.pt
```

If the trained checkpoint is absent, the app starts, explains the training steps, and disables analysis. The training script validates best.pt and installs it at this path after training. Model weights and dataset files are excluded from Git. Use a clean Python 3.12 environment for training and inference; see Troubleshooting if your existing venv was created with another Python version.


## Interface and motion references

The interface uses a high-contrast editorial layout with a field-monitor / system-log feel, taking visual direction from [NoGood Studio](https://nogood.studio/) without reusing its branding or imagery. GSAP animates the class rail and hover tilt, and Lenis enhances its horizontal scrolling inside an isolated Streamlit component. Both load from their public CDNs; the rail falls back to native browser scrolling if the scripts are unavailable. These scripts do not receive uploaded media.


[React Bits](https://github.com/DavidHDev/react-bits) targets React components and does not plug directly into this Python/Streamlit app. Its motion and interaction ideas informed the small hover/reveal effects, implemented locally without copying React Bits component source. Review each upstream project's current license before redistributing or modifying their code.

## Dataset setup and training

RDD2022 is distributed by the dataset authors with Pascal VOC XML bounding boxes for labeled training images. The official release provides country-specific downloads, including Japan, India, Czech Republic, Norway, United States, and China. The official CRDDC class mapping is D00, D10, D20, and D40. The release states that its images are under **CC BY-SA 4.0**; follow that license for any redistributed data or adapted dataset material, give attribution, and include the license. Cite the RDD2022 paper/release in academic reports. See the [official RoadDamageDetector repository](https://github.com/sekilab/RoadDamageDetector) for current downloads, license details, and citation information.

1. Download the RDD2022 country archives you need from the official release and extract the labeled training data to a local folder. Do not use the unlabeled challenge test images for training or report validation metrics on them.
2. Convert the Pascal VOC XML annotations and create a deterministic image-level train/validation split (default 80/20):

   ```powershell
   .\.venv312\Scripts\python.exe scripts\prepare_dataset.py --source "D:\datasets\RDD2022" --output "data\rdd2022_yolo"
   ```

   The script maps the four official classes, writes normalized YOLO boxes, reports skipped/unmatched annotations and per-class counts, and creates `data.yaml`. It refuses to overwrite a non-empty output directory. Inspect converted images and labels before training. If you prefer to copy images instead of hard-linking where supported, add `--copy-images`.

3. Fine-tune the lightweight YOLOv8n model. Ultralytics downloads its generic `yolov8n.pt` initialization weights on first use; network access is needed for that one-time download. Training defaults to 50 epochs, 640-pixel images, batch size 8, early stopping, and automatic device selection. Adjust for available memory:

   ```powershell
   .\.venv312\Scripts\python.exe scripts\train.py --data "data\rdd2022_yolo\data.yaml" --epochs 50 --batch 8 --imgsz 640
   ```

   For a CUDA GPU, specify its index (usually 0); to force CPU, use --device cpu. Training run files stay under runs\road_damage\yolov8n_rdd2022\weights\. After checking the saved task and class map, the script copies best.pt to models\road_damage_yolov8n.pt.

4. Evaluate on the held-out validation split and use only metrics produced by this run in your report:

   ```powershell
   .\.venv312\Scripts\python.exe scripts\validate.py --weights "runs\road_damage\yolov8n_rdd2022\weights\best.pt" --data "data\rdd2022_yolo\data.yaml"
   ```

5. Training automatically installs the validated checkpoint. For a separately trained compatible checkpoint, copy it manually:

   ```powershell
   New-Item -ItemType Directory -Force models
   Copy-Item "D:\path\to\compatible\best.pt" "models\road_damage_yolov8n.pt"
   ```

Refresh Streamlit after installing it. Use a real trained four-class detection checkpoint.


For a genuine image-inference smoke run after installing the model, use a local road image:

```powershell
.\.venv312\Scripts\python.exe scripts\predict.py --source "D:\images\road.jpg" --weights "models\road_damage_yolov8n.pt" --conf 0.25 --imgsz 640 --device auto
```

The CLI writes runs\predict\annotated.png and a matching detection CSV. It prints each real class, confidence, and source-image pixel box. If the checkpoint returns zero detections, the CLI reports zero; it never substitutes sample output.

## CSV columns

`timestamp_seconds`, `frame_number`, `class_code`, `class_name`, `confidence`, `severity`, `severity_score`, `area_fraction`, `x1`, `y1`, `x2`, and `y2`. Coordinates are pixel locations in the original frame; `area_fraction` is the bounding-box area divided by frame area.

## Checks

Run the model metadata and missing-checkpoint checks with:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
```

A real prediction additionally requires the trained checkpoint and a local road image.

## Reference projects and licensing

- [RoadDamageDetection](https://github.com/Srusti-26/RoadDamageDetection) informed the Streamlit video-analysis and analytics flow. The repository page does not list an explicit software license, so no source code or assets are copied from it.
- [RDD2022-Road-Damage-Detection](https://github.com/nehaksharma11/RDD2022-Road-Damage-Detection) is listed under MIT and informed the YOLO training workflow. This project does not copy its implementation or report its claimed metrics as results for this application.
- RDD2022 dataset images have their own CC BY-SA 4.0 terms, separate from software licenses. Check the official dataset page before downloading and redistributing any dataset material.
- Ultralytics and PyTorch are third-party dependencies. Review their current license terms for your intended use.

## Troubleshooting

- **PyTorch DLL or Python ABI error:** use a fresh environment made with CPython 3.12; do not reuse site-packages from Python 3.13. Stop the old Streamlit process, create the documented .venv312, install requirements there, and run the app with that environment's Python. An error such as WinError 126 while importing torch indicates incompatible/missing native PyTorch DLLs, not a road-image issue.


- **Checkpoint missing:** run `scripts\train.py` after preparing the RDD2022 data; the script validates `best.pt` and copies it to `models/road_damage_yolov8n.pt`. For an existing compatible checkpoint, copy it manually to that path.
- **Cannot decode upload:** confirm the file is a valid MP4, AVI, or MOV and can be opened in a media player; re-encode it if needed.
- **MP4 writer/codec error:** install the standard `opencv-python` wheel in the active virtual environment and retry. Some custom OpenCV builds omit an MP4 writer.
- **Slow processing:** increase “Analyze every Nth frame” or lower inference image size. Unsampled frames remain in the output without boxes.
- **CUDA memory issue:** lower `--batch` during training or use `--device cpu`; for app inference lower image size.
