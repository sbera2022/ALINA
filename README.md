ALINA: Automated Taxiway Line Identification Using AI/ML and Computer Vision

Project Overview

This project extends the ALINA (Automated Line Identification and Annotation) pipeline by exploring methods for automatically selecting a Region of Interest (ROI) in airport taxiway images.

The original ALINA workflow requires users to manually select a region before applying perspective transformation and line detection. This project investigates whether machine learning, semantic segmentation, and traditional computer vision techniques can automate that process.

Five approaches were implemented and evaluated on a 50-frame image dataset.

Original Repository: ALINA

Project Objectives

The primary objectives are to:

Automate ROI selection from the initial video frame.

Compare five different ROI-selection techniques.

Integrate generated ROIs with ALINA's existing perspective transformation and labeling pipeline.

Evaluate ROI selection using Intersection over Union (IoU).

Measure ALINA's reported labeling coverage and processing runtime.

Identify successful detections, false positives, and limitations.

Implemented Methods

Method 1: K-Means Color Clustering

Uses unsupervised K-Means clustering in Lab color space to segment visually similar image regions.

The algorithm identifies a pavement-like cluster, applies morphological filtering, and constructs a quadrilateral ROI based on the selected region.

Parameters: K = 4, random seed = 42.

Method 2: Canny Edge Detection + Hough Transform

Uses Canny edge detection to identify boundaries and the probabilistic Hough Transform to identify line segments.

Candidate line segments are filtered based on orientation and length. The selected segment is used to estimate the ROI geometry.

Parameters: Canny thresholds = 50 and 150; Hough threshold = 25; minimum line length = 25; maximum line gap = 20.

Method 3: HSV Color Thresholding

Converts the image to HSV color space and identifies pixels corresponding to yellow taxiway markings.

Connected-component analysis selects a candidate marking, and line fitting determines the orientation of the automatic ROI.

HSV thresholds: Lower = (15, 60, 70); Upper = (40, 255, 255).

Method 4: SegFormer Semantic Segmentation

Uses a pretrained SegFormer-B0 neural network to predict semantic categories for individual image pixels.

The model identifies pavement-like regions, and geometric heuristics estimate the ROI position.

Pretrained model: nvidia/segformer-b0-finetuned-ade-512-512

The model was pretrained on ADE20K, rather than airport-specific imagery.

Method 5: Hybrid SegFormer + HSV

Combines semantic segmentation with color-based detection.

SegFormer provides pavement-region information, while HSV thresholding identifies yellow markings. The two sources are combined to guide line fitting and determine ROI boundaries.

This approach evaluates whether combining learned semantic features with traditional computer vision improves ROI selection.

Project Structure

ALINA/
├── alina/
│   ├── config.py
│   ├── pipeline.py
│   └── roi.py
├── experiments/
│   ├── kmeans_roi.py
│   ├── canny_hough_roi.py
│   ├── hsv_roi.py
│   ├── segformer_roi.py
│   ├── hybrid_roi.py
│   ├── run_kmeans_alina.py
│   ├── run_canny_hough_alina.py
│   ├── run_hsv_alina.py
│   ├── run_segformer_alina.py
│   ├── run_hybrid_alina.py
│   ├── manual_reference_roi.py
│   ├── evaluate_roi.py
│   └── plot_results.py
├── data/
│   └── Raw_Data/
│       └── vidd_1/
├── outputs/
│   ├── kmeans/
│   ├── canny_hough/
│   ├── hsv/
│   ├── segformer/
│   ├── hybrid/
│   └── evaluation/
├── cli.py
└── README.md

Directory contents may vary depending on which outputs have been generated.

Installation and Setup

1. Clone the repository

git clone https://github.com/sbera2022/ALINA.git
cd ALINA

2. Create a virtual environment

Python 3.12 is recommended for compatibility with the pretrained segmentation dependencies.

On Windows:

py -3.12 -m venv venv_segformer
venv_segformer\Scripts\activate

3. Install dependencies

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install torch torchvision transformers pillow matplotlib

The first SegFormer run downloads the pretrained model. An internet connection is required for that initial download.

4. Prepare the dataset

Place the evaluation frames inside:

data/Raw_Data/vidd_1/

The experiments use 00001.jpg as the initial frame for automatic ROI generation.

Running the Experiments

Run the following commands from the project root.

Generate automatic ROIs

python -m experiments.kmeans_roi
python -m experiments.canny_hough_roi
python -m experiments.hsv_roi
python -m experiments.segformer_roi
python -m experiments.hybrid_roi

Run SegFormer before Hybrid because the hybrid implementation reuses the saved semantic predictions.

Each method generates an ALINA-compatible ROI saved as alina_roi.npy.

Run ALINA with the generated ROIs

python -m experiments.run_kmeans_alina
python -m experiments.run_canny_hough_alina
python -m experiments.run_hsv_alina
python -m experiments.run_segformer_alina
python -m experiments.run_hybrid_alina

The primary experiments use the same downstream ALINA configuration, including mask_ignore_left_columns=300.

Evaluate ROI overlap

Create the manual reference:

python -m experiments.manual_reference_roi

Then calculate IoU:

python -m experiments.evaluate_roi

Generate evaluation graphs

python -m experiments.plot_results

The evaluation script generates:

final_results.csv

labeling_rate.png

roi_iou.png

runtime.png

Evaluation Metrics

Reported Labeling Rate

The percentage of frames for which ALINA reports a labeling result.

[
\text{Labeling Rate} = \frac{\text{Frames Reported Labeled}}{\text{Total Frames}} \times 100
]

This metric does not necessarily measure correct line detection.

Intersection over Union (IoU)

Measures the geometric overlap between an automatically generated ROI and a manually selected reference ROI.

[
IoU = \frac{|ROI_{automatic} \cap ROI_{manual}|}{|ROI_{automatic} \cup ROI_{manual}|}
]

Processing Runtime

Measures the average time ALINA takes to process an image after the ROI has been generated. ROI-generation and model-inference times are not included in the reported ALINA runtimes.

Experimental Results

All five methods were evaluated on the same 50-frame batch.

Method

Frames Labeled

Labeling Rate

ROI IoU

Average ALINA Time

K-Means

26/50

52%

0.4931

20.6368 s

Canny + Hough

0/50

0%

0.5376

0.2875 s

HSV

10/50

20%

0.5705

0.3669 s

SegFormer

2/50

4%

0.3788

0.6018 s

Hybrid

20/50

40%

0.4329

8.3619 s

Labeling Rate Comparison



ROI IoU Comparison



Processing Runtime Comparison



Key Findings

K-Means achieved the highest reported labeling rate at 52%.

HSV achieved the highest ROI IoU at 0.5705.

Hybrid produced more labeled frames than HSV and SegFormer individually.

SegFormer showed limited transferability from general-purpose semantic segmentation to airport taxiway imagery.

The highest labeling rate did not necessarily correspond to the most accurate visual detections.

Canny + Hough Sensitivity Experiment

A separate experiment changed ALINA's mask_ignore_left_columns setting from 300 to 0.

Mask Setting

Frames Labeled

Rate

Average Time

300

0/50

0%

0.2875 s

0

20/50

40%

3.2814 s

The modified configuration improved reported labeling coverage and produced a visually aligned partial detection on frame 00001.jpg.

This result demonstrates that downstream masking parameters significantly affect ALINA's behavior.

The modified configuration is treated as a sensitivity experiment rather than directly replacing the primary baseline.

Qualitative Evaluation

Representative annotated images were visually inspected to determine whether ALINA's reported detections followed the actual yellow taxiway markings.

Method

Frame

Observation

K-Means

00001.jpg

Partial detection of a short yellow-line segment

Canny + Hough (mask = 0)

00001.jpg

Strong partial detection following much of the right-hand marking

HSV

00001.jpg

Partial detection along a substantial straight segment

SegFormer

01060.jpg

False positive near the far-right image boundary

Hybrid

01061.jpg

False positive near the far-right image boundary

These examples demonstrate that an ALINA [Labeled] result does not necessarily represent a correct detection.

Limitations

Fixed ROI: Each automatic ROI was generated from the initial frame and reused throughout the evaluation batch.

Limited reference annotations: ROI IoU was evaluated against one manually selected reference ROI.

False positives: Some frames reported as labeled contained annotations outside the intended taxiway marking.

Semantic domain mismatch: The pretrained SegFormer model was not specialized for airport taxiway imagery.

Parameter sensitivity: Downstream masking settings significantly affected the results.

Runtime variation: ALINA processing time varied considerably with the amount of candidate line data processed.

Future Improvements

Possible extensions include:

Generating a new ROI for every frame.

Using airport-specific segmentation datasets.

Fine-tuning semantic segmentation models for taxiway markings.

Evaluating against manually annotated line masks.

Measuring precision, recall, and F1-score for actual line detection.

Improving runtime through optimized line-processing algorithms.

Reproducibility

The experiments use deterministic seeds where applicable, including seed 42 for K-Means clustering and the SegFormer experiment.

For consistent results, preserve the same input images, ROI-generation parameters, model checkpoint, and ALINA configuration.

The main experiment uses mask_ignore_left_columns=300; the Canny + Hough sensitivity experiment uses 0.

References

ALINA GitHub Repository

SegFormer-B0 ADE20K Pretrained Model
