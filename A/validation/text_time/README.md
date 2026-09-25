# Text-Time Mapping Results

Download the Whisper output directory from AutoDL into this folder, so the
local structure becomes:

```text
validation/text_time/
└── words/
    ├── sample_1_words.json
    ├── sample_2_words.json
    └── whisper_words.jsonl
```

Then run from the `A/` directory:

```powershell
python validation/autodl/map_text_time.py `
  --words-dir validation/text_time/words `
  --label-file "E:\E题\E题数据\label-100.xlsx" `
  --bert-path "E:\E题\bert-base-uncased" `
  --output-dir validation/text_time
```

The mapping has been completed locally.

Generated files:

- `text_time_alignment.csv`: all `100 x 50` BERT token positions with source
  word, time interval and timestamp availability.
- `text_time_coverage.csv`: word matching and token timestamp coverage for each
  sample.
- `text_time_review.csv`: samples with token timestamp coverage below 80%.
- `review_sample_ids.txt`: sample list for a higher-accuracy Whisper rerun.

Recommended rerun on AutoDL for the 16 low-coverage samples:

```bash
python validation/autodl/autodl_whisper_words.py \
  --audio-dir /root/autodl-tmp/audio \
  --output-dir /root/autodl-tmp/A/validation/text_time/words_small \
  --sample-list /root/autodl-tmp/A/validation/text_time/review_sample_ids.txt \
  --model small
```

The script used `transformers` when available. In this environment it used the
local `vocab.txt` WordPiece fallback, so no network access was required.
