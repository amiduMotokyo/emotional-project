# Problem 1 Experiment Notes

## Data flow

```text
Attachment 1 video
  -> audio track -> openSMILE -> (n_frames, 25) -> 0.5 s mean -> (50, 25)
  -> OpenFace   -> (n_frames, 20) -> 0.5 s mean -> (50, 20)
label-100.xlsx text -> BERT -> (50, 768)
  -> final_features.pkl
```

## Actual artifacts

- 100 audio files, each `(50, 25)`.
- 100 visual files, each `(50, 20)`.
- 100 text files, each `(50, 768)`.
- `final_features.pkl` with keys `audio`, `text`, `vision`, `id`,
  `text_len`, `audio_len`, `vision_len`.
- 10,000 rows of audio/vision time mapping.
- 5,000 rows of text token mapping.

## Quality evidence

- Typical sample: `-UuX1xuaiiE_0`.
- Typical sample figure: `outputs/evidence/typical_sample_-UuX1xuaiiE_0_panel.png`.
- t-SNE figure: `outputs/evidence/tsne_three_modalities.png`.
- Feature-quality summary: `outputs/evidence/feature_quality_summary.csv`.

## Known limitations

- Text positions are token positions, not timestamps.
- Sample `-yRb-Jum7EQ_1` exceeds 25 seconds and is truncated by the fixed rule.
- openSMILE uses `-201` as an unavailable-value sentinel for some formant
  estimates; the original feature file preserves these values.
- OpenFace follows the primary detected face in videos containing multiple faces.

