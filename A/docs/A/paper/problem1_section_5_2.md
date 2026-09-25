# 5.2 Problem 1: Model and Solution

## 5.2.1 Problem description and modeling objective

For each sample, use `video_id` and `clip_id` as the unique key. Extract a text
sequence `X_t`, an audio sequence `X_a` and a visual sequence `X_v`, and map
them to a common index length of 50:

`Phi(T_i, A_i, F_i) = (X_i^t, X_i^a, X_i^v)`.

The dimensions are 768 for text, 25 for audio and 20 for vision. Problem 1 is
an independent feature extraction and alignment task; it does not use the
features in Attachments 2-4 for prediction.

## 5.2.2 Feature models

### Text

Use the transcribed text in `label-100.xlsx`, tokenize it with
`bert-base-uncased`, truncate or pad to 50 positions, and retain the final
hidden state:

`X_t = BERT(Tokenizer(T)) in R^(50 x 768)`.

The attention mask sum is the valid token length.

### Audio

Extract 16 kHz mono audio with ffmpeg. Apply openSMILE eGeMAPSv02 at the
low-level descriptor level. The raw sequence is `(n_frames, 25)`. For segment
`k`, average all frames in `[0.5k, 0.5(k+1))`. Zero-pad to 50 segments.

### Vision

Run OpenFace FeatureExtraction with action-unit and head-pose outputs. Select
17 AU intensity columns and `pose_Rx`, `pose_Ry`, `pose_Rz`, giving 20
dimensions. Average frames in each 0.5 second interval and zero-pad to 50
segments.

### Alignment

Audio and vision share a 0.5 second, 50-position time axis. Text uses token
positions. Their valid positions and mappings are stored separately so that
the distinction between temporal alignment and token alignment remains
auditable.

## 5.2.3 Solution procedure

1. Read `label-100.xlsx` and verify sample identifiers.
2. Extract and align audio features.
3. Extract and align visual features.
4. Encode text with BERT.
5. Merge modalities, export `final_features.pkl`, generate the full summary
   and export both alignment mapping files.

## 5.2.4 Results and analysis

All 100 samples are covered. The merged feature shapes are:

| Modality | Shape |
|---|---|
| Audio | `(100, 50, 25)` |
| Vision | `(100, 50, 20)` |
| Text | `(100, 50, 768)` |

The full result summary has 300 rows. The time mapping has 10,000 rows and the
token mapping has 5,000 rows.

Insert the typical sample panel and t-SNE figure from `outputs/evidence/`.
Describe audio/vision valid-length consistency and the modality-level
Kruskal-Wallis and Pearson results. State clearly that limited single-modality
separation is motivation for multimodal fusion.

## 5.2.5 Reproducibility

Record Python and package versions, OpenFace model configuration, BERT path,
input paths, alignment parameters and execution command. The reproducible
entry point is:

```bash
python scripts/run_all.py
```

Known boundaries are listed in `outputs/logs/processing_log.md`.

