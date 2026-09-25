# Problem 1 Requirements Checklist

## Required outputs

- Feature extraction and temporal alignment scheme for text, audio and vision.
- A self-generated feature file covering all 100 samples from Attachment 1.
- A full result summary containing sample id, modality, valid duration or length,
  feature dimension and alignment granularity.
- At least one typical sample showing text, audio time ranges, video frames and
  the relation among the three feature sequences.
- Reproduction notes including tool versions, parameters and execution order.

## Coverage checks

- 100 samples.
- 300 modality records.
- Audio feature shape: `(100, 50, 25)`.
- Vision feature shape: `(100, 50, 20)`.
- Text feature shape: `(100, 50, 768)`.
- Time mapping rows: `100 * 2 * 50 = 10000`.
- Token mapping rows: `100 * 50 = 5000`.

## Evidence interface

The B-side evidence interface uses:

- `start_sec` and `end_sec` in seconds;
- three decimal places, i.e. millisecond display precision;
- `sample_id = video_id + "_" + clip_id`;
- `is_valid` to distinguish real segments from zero padding.

