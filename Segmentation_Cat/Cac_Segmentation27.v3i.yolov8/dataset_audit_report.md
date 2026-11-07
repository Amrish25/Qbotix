# Comprehensive Dataset Audit Report: Cac_Segmentation27 (Caterpillar27)

**Audit Date:** October 27, 2026  
**Dataset Workspace:** `c:\Users\AMRISH A\Downloads\Cac_Segmentation27.v3i.yolov8`  
**Archive Source:** `C:\Users\AMRISH A\Downloads\Cac_Segmentation27.v3i.yolov8.zip` (426,542,339 bytes / ~406.8 MB)  
**Roboflow Origin:** `https://universe.roboflow.com/bunnys-space/cac_segmentation27/dataset/3`  
**Target Task:** YOLOv8 Instance Segmentation  

---

## Executive Summary & Final Verdict

### Final Verdict: **NOT READY FOR TRAINING**

While the dataset files are uncorrupted, perfectly paired (4,662 images to 4,662 labels), and normalized strictly between `0.0` and `1.0`, the dataset possesses **severe structural, semantic, and data split flaws** that prevent it from being trained directly:

1. **Extreme Data Leakage Across Splits (87% Contamination)**:  
   **86.6% of the validation split** (284 of 328 images) and **88.8% of the test split** (151 of 170 images) are augmented copies (rotations, flips, warps) of base images in the training set. Model evaluation on validation/test will yield artificially inflated, invalid performance metrics.
2. **5-Token Detection Bounding Box Contamination (539 Errors)**:  
   539 instances in class 5 (`Rock`) are formatted as standard detection bounding boxes (`class x_center y_center width height` — 5 tokens), which violates the YOLOv8 instance segmentation requirement (polygons must have $\ge 6$ coordinate values / $\ge 7$ tokens).
3. **Pervasive Axis-Aligned Bounding Boxes Disguised as Polygons (21,441 Instances / 43.1%)**:  
   More than 43% of all annotations (including 57.0% of `Crater` and 37.1% of `Rock`) are 4-point rectangular bounding boxes rather than true segmentation contours.
4. **Corrupted & Orphan Class Definitions**:  
   - In `data.yaml`, Class 1 is named `'1 0 0 0 1 1 1 1 0 0 0'` (a raw binary string or malformed export).
   - Classes 2 and 3 contain only 6 instances each in the entire 4,662-image dataset (0 in validation, 0 in test), making them unlearnable and unevaluable.
5. **Domain Fracture ("Frankenstein" Dataset)**:  
   The dataset is an unharmonized composite of two disparate domains: 2,059 video frame images (grayscale lunar/martian orbital/descent frames annotated with classes `0`, `1`, `2`, `3`) and 2,603 planetary surface images (Mars Mastcam and synthetic sand terrain annotated with `Crater` and `Rock`). There is 0% class overlap between the two domains.
6. **Dataset Identity Mismatch**:  
   The task prompt refers to "Caterpillar27", but the dataset is `Cac_Segmentation27` (planetary craters, rocks, and surface terrain). Zero caterpillars or machinery appear in the dataset.

---

## 1. Dataset Inspection & Integrity

### 1.1 Archive and File Verification
- **ZIP File Verification:** `C:\Users\AMRISH A\Downloads\Cac_Segmentation27.v3i.yolov8.zip` exists and was verified using Python's `zipfile.testzip()`. The archive CRC32 checksums are intact with 0 corrupt entries.
- **Image Corruption Check:** All 4,662 image files across train, valid, and test splits were opened, verified, and parsed through PIL. **0 corrupt or unreadable image files detected**.
- **Extension Consistency:** 100% of images are `.jpg` files in `RGB` color mode.
- **Label Encoding:** 100% of labels are UTF-8 text files.

### 1.2 Directory & Pairing Overview
Each split contains dedicated `images/` and `labels/` subdirectories. Image-to-label filename matching is 100% complete with 0 missing labels and 0 orphan labels.

| Split | Images | Label Files | Missing Labels | Orphan Labels | Empty Labels (Background) | Non-Empty Labels | Annotated Instances |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4,164 | 4,164 | 0 | 0 | 229 (5.5%) | 3,935 (94.5%) | 43,717 |
| **Valid** | 328 | 328 | 0 | 0 | 20 (6.1%) | 308 (93.9%) | 3,706 |
| **Test** | 170 | 170 | 0 | 0 | 20 (11.8%) | 150 (88.2%) | 2,347 |
| **Total** | **4,662** | **4,662** | **0** | **0** | **269 (5.8%)** | **4,393 (94.2%)** | **49,770** |

### 1.3 `data.yaml` Configuration Audit
Current contents of `data.yaml`:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['1', '1 0 0 0 1 1 1 1 0 0 0', '2', '3', 'Crater', 'Rock']
```

#### Identified `data.yaml` Issues:
1. **Relative Path Bug:** The paths use `../train/images` instead of `train/images` or specifying a root `path: .`. When invoking `yolo segment train data=data.yaml` from the dataset folder, YOLO will search in the parent directory (`Downloads/train/images`), causing immediate dataset loader failure.
2. **Corrupted Class Name String:** Class index 1 is named `'1 0 0 0 1 1 1 1 0 0 0'`.
3. **Numeric Class Names:** Classes 0, 2, and 3 are named `'1'`, `'2'`, `'3'`.

---

## 2. Segmentation Annotation Validation

Every annotation row across all 4,662 files was audited for YOLOv8 segmentation compliance.

### 2.1 Geometry & Syntax Breakdown

| Check Type | Count | % of Total | Severity | Description / Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Total Annotation Rows** | **49,770** | 100.0% | Info | Total instances across all splits |
| **Valid Coordinate Pairs** | 49,770 | 100.0% | Pass | All rows have an even number of coordinates (no orphan tokens) |
| **Normalized Range [0.0, 1.0]** | 49,770 | 100.0% | Pass | All coordinates are finite floats within `[0.0, 1.0]` (0 out-of-bounds) |
| **Detection Bounding Boxes (5 tokens)** | **539** | **1.08%** | **ERROR** | Formatted as `class xc yc w h` instead of polygon. Causes parser error or degenerate points |
| **Axis-Aligned Bounding Box Polygons** | **21,441** | **43.08%** | **CONCERN** | 4-point rectangular boxes saved as polygons; lacks contour segmentation |
| **True Polygon Segmentations** | **27,790** | **55.84%** | Pass | True multi-vertex contour polygons |
| **Duplicate Closing Vertices** | 39,856 | 80.08% | Info | Explicit closing point `(x_n == x_0)`. Redundant but harmless in YOLO |
| **Self-Intersecting Polygons** | 85 | 0.17% | **WARNING** | Complex bowtie polygons where polygon perimeter edges cross themselves |
| **Suspiciously Small (Micro) Polygons** | 9 | 0.02% | **WARNING** | Area $< 9\text{ px}^2$ (e.g. 1-2 pixel slivers on image borders) |
| **Suspiciously Large (Giant) Polygons** | 39 | 0.08% | **WARNING** | Polygons covering $> 95\%$ of entire image |

### 2.2 Detailed Breakdown of the 539 Bounding Box Errors
All 539 bounding boxes belong to Class 5 (`Rock`) and are distributed across all splits:
- **Train:** 396 instances
- **Valid:** 82 instances
- **Test:** 61 instances

*Example faulty row from `train/labels/331_..._rcw_png...txt`:*
`5 0.9140625 0.36328125 0.05078125 0.05078125`

### 2.3 Detailed Breakdown of the 21,441 Disguised Bounding Boxes
- **Class 4 (`Crater`):** 10,796 instances (57.0% of all craters are rectangular boxes)
- **Class 5 (`Rock`):** 10,644 instances (37.1% of all rocks are rectangular boxes)
- **Class 0 (`1`):** 1 instance

---

## 3. Class Distribution & Semantics

### 3.1 Class Frequency Table
*(Generated from [class_distribution.csv](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/class_distribution.csv))*

| ID | Class Name in `data.yaml` | Train | Valid | Test | Total Instances | Image Count | BBox Errors | Overall % | Imbalance Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `'1'` | 1,570 | 121 | 43 | 1,734 | 1,730 | 0 | 3.48% | Moderate |
| **1** | `'1 0 0 0 1 1 1 1 0 0 0'` | 384 | 30 | 15 | 429 | 428 | 0 | 0.86% | Severe Underrepresentation |
| **2** | `'2'` | 6 | 0 | 0 | 6 | 6 | 0 | 0.01% | **Critically Deficient (0 in Val/Test)** |
| **3** | `'3'` | 6 | 0 | 0 | 6 | 6 | 0 | 0.01% | **Critically Deficient (0 in Val/Test)** |
| **4** | `Crater` | 16,993 | 1,277 | 665 | 18,935 | 2,031 | 0 | 38.05% | Dominant Class |
| **5** | `Rock` | 24,758 | 2,278 | 1,624 | 28,660 | 2,278 | 539 | 57.58% | Dominant Class |
| **Total**| — | **43,717**| **3,706**| **2,347**| **49,770** | **4,393** | **539** | **100%** | **Max Ratio: 4,776 : 1** |

### 3.2 Visual Content Analysis: Two Disjoint Datasets

By correlating image filename patterns, visual contents, and class annotations, we reveal that `Cac_Segmentation27` is a forced union of two completely unrelated datasets:

```
                               Cac_Segmentation27 Dataset (4,662 Images)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     Domain A: Orbital/Lander Descent                              Domain B: Surface Geology
     (2,059 images, 640x640 grayscale)                       (2,603 images, mixed resolutions)
     • Video frames (frame_0 ... frame_269)                  • NASA Curiosity Mastcam (512x512)
     • Exclusively Classes 0, 1, 2, 3                        • Leonardo AI Sand Sim (1472x832)
     • Annotates crater rims & dark terrain patches          • General crater photos (1792x1792)
     • ZERO Crater (4) or Rock (5) instances                 • Exclusively Crater (4) & Rock (5)
                                                             • ZERO Class 0, 1, 2, or 3 instances
```

---

## 4. Data Leakage & Split Quality

### 4.1 Exact File Duplicates (Identical SHA-256)
- **Total Duplicate Groups:** 58 groups (116 total files)
- **Within-Split Duplication:** All 58 pairs exist entirely within the `train` split (duplicate Roboflow export hashes of the identical JPEG file).
- **Cross-Split Exact Duplicates:** 0 exact byte-level matches between train and valid/test.

### 4.2 Augmentation Data Leakage (Critical Defect)
In standard machine learning workflows, dataset splitting must be conducted **before** data augmentation. In this dataset, raw source images were augmented (horizontal flips `_hf`, rotations `_rcw`/`_rccw`, translations `_tleft`/`_tup`, shear down `_sdown`) and then randomly partitioned into `train`, `valid`, and `test`.

- **Total Unique Base Images:** 966
- **Base Images Present in Train:** 966
- **Validation Split Leakage:** **284 out of 328 validation images (86.6%)** originate from base images present in the training set!
- **Test Split Leakage:** **151 out of 170 test images (88.8%)** originate from base images present in the training set!
- **Validation-Test Overlap:** 44 base image IDs appear in both validation and test.

#### Impact of Split Contamination:
Any model trained on this dataset will achieve artificially near-perfect validation and test scores because the evaluation sets are simply rotated or flipped versions of the training images. It provides zero measure of generalizability.

---

## 5. Visual Overlays & Problem Case Gallery

Sample overlay images have been generated and saved to [`sample_overlays/`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays).

### Key Visual Evidence:

1. **Data Leakage Proof:** [`overlay_leakage_comparison.jpg`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays/overlay_leakage_comparison.jpg)
   - Left: Train image `331_..._rcw_png` (rotated clockwise).
   - Right: Validation image `331_..._hf_png` (horizontal flip).
   - Shows the exact same rock outcrop present in both splits.
2. **5-Token Bounding Box Issue:** [`overlay_train_problem_bbox_mars.jpg`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays/overlay_train_problem_bbox_mars.jpg)
   - Demonstrates red bounding boxes (`BOX: 5: Rock`) mixed alongside green polygon contours on the same image.
3. **Axis-Aligned Bounding Box Disguise:** [`overlay_train_crater_rock_leonardo.jpg`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays/overlay_train_crater_rock_leonardo.jpg)
   - Demonstrates that 100% of the sand craters (green) and rocks (red) in this synthetic scene are rectangular boxes rather than true segmentation contours.
4. **Domain A Video Frames:**
   - [`overlay_train_class0_frame0.jpg`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays/overlay_train_class0_frame0.jpg): Class 0 polygon on orbital lunar/martian surface.
   - [`overlay_train_class1_frame100.jpg`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays/overlay_train_class1_frame100.jpg): Class 1 (`'1 0 0 0...'`) polygon tracing a crater rim.
   - [`overlay_train_class2_frame253.jpg`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays/overlay_train_class2_frame253.jpg): Class 2 and Class 1 instances.
   - [`overlay_train_class3_frame240.jpg`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays/overlay_train_class3_frame240.jpg): Class 3 instance.

---

## 6. Required Fixes & Recommendations

To bring this dataset to a state that is **READY FOR TRAINING**, the following remediation steps are required:

### Step 1: Fix `data.yaml` Paths and Class Names
- Change `train: ../train/images` to `train: train/images` (or add `path: .`).
- Fix class definitions:
  - Decide whether this is a **2-Class Surface Model** (`['Crater', 'Rock']`) or an **Orbital Surface Model**.
  - If retaining classes 0-3, rename them to descriptive semantic names instead of `'1'`, `'1 0 0 0 1 1 1 1 0 0 0'`, `'2'`, `'3'`.

### Step 2: Resolve the 539 Bounding Box Errors
- Convert each 5-token bounding box `class xc yc w h` into a 4-point polygon:
  $x_1 = xc - w/2, y_1 = yc - h/2$
  $x_2 = xc + w/2, y_2 = yc - h/2$
  $x_3 = xc + w/2, y_3 = yc + h/2$
  $x_4 = xc - w/2, y_4 = yc + h/2$
- Format: `5 x1 y1 x2 y2 x3 y3 x4 y4` (9 tokens).

### Step 3: Eliminate Data Leakage by Group Splitting
- Extract root base image IDs (stripping `_hf`, `_rcw`, `_rccw`, `_sdown`, `_sup`, `_tleft`, etc., and `.rf.*`).
- Perform **GroupKFold / Group Shuffle Split** at the base image level:
  - 70% of base images exclusively to `train` (including all their augmented versions).
  - 15% of base images exclusively to `valid`.
  - 15% of base images exclusively to `test`.
- Ensure 0% base-image overlap between train, valid, and test.

### Step 4: Split or Harmonize the Two Domains
- Split the dataset into two clean independent projects:
  - **Dataset 1 (Planetary Surface):** 2,603 images annotated with `Crater` and `Rock`.
  - **Dataset 2 (Orbital / Video Frames):** 2,059 images annotated with descent terrain classes.
- Or prune the 12 orphan instances of Class 2 and Class 3 and re-annotate the video frames consistently.

---

## 7. Deliverables Index

1. **Full Audit Report:** [`dataset_audit_report.md`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/dataset_audit_report.md)
2. **Detailed Issue Log (22,113 issues recorded):** [`annotation_issues.csv`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/annotation_issues.csv)
3. **Class Distribution Breakdown:** [`class_distribution.csv`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/class_distribution.csv)
4. **Visual Sample Overlays:** Directory [`sample_overlays/`](file:///c:/Users/AMRISH%20A/Downloads/Cac_Segmentation27.v3i.yolov8/sample_overlays)
