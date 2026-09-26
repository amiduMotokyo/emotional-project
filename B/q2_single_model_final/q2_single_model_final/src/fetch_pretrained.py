"""Download only general-purpose pretrained encoders, never sentiment datasets."""
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
snapshot_download('bert-base-uncased',revision='86b5e0934494bd15c9632b12f734a8a67f723594',local_dir=str(ROOT/'models/bert-base-uncased'),allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt'])
print('Frozen BERT ready')
