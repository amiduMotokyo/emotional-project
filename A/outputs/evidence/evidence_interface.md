# Evidence Locator Interface

## Purpose

This interface maps each aligned audio or visual feature position back to the
original time range of its source video.

## Time mapping

File: `outputs/alignment/alignment_mapping_time.csv`

| Field | Meaning |
|---|---|
| `sample_id` | `video_id + "_" + clip_id` |
| `modality` | `audio` or `vision` |
| `position` | Zero-based segment index, 0-49 |
| `start_sec` | Segment start time in seconds |
| `end_sec` | Segment end time in seconds |
| `is_valid` | Whether this position is a real segment rather than zero padding |

Rules:

- Time unit: seconds.
- Display precision: 0.001 second.
- Segment length: 0.5 second.
- Maximum positions: 50, covering `[0, 25)` seconds.
- The source video is determined directly by `sample_id`.

## Text mapping

File: `outputs/alignment/alignment_mapping_token.csv`

Text is aligned by token position, not by word timestamps. The field `token`
records the BERT token at each of the 50 positions and `is_valid` indicates
whether the position is a real token.

## Sample identity

For a row in `label-100.xlsx`:

```text
sample_id = str(video_id) + "_" + str(clip_id)
```

The corresponding original video is:

```text
{video_root}/{video_id}/{clip_id}.mp4
```

