# Verification Report

- sample_count: 100
- feature_manifest_rows: 300
- summary_rows: 300
- time_mapping_rows: 10000
- token_mapping_rows: 5000
- audio_shape: [100, 50, 25]
- vision_shape: [100, 50, 20]
- text_shape: [100, 50, 768]
- audio_nan: 0
- vision_nan: 0
- text_nan: 0

All 100 label samples have audio, vision and text feature files.
Feature manifest: outputs/summary/file_manifest.csv.
Time mapping: outputs/alignment/alignment_mapping_time.csv.
Token mapping: outputs/alignment/alignment_mapping_token.csv.
