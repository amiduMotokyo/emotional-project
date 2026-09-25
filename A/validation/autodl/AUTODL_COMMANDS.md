# AutoDL Commands

Run these commands from the local `A/` directory. SSH and SCP will request the
server password interactively; do not place the password in a script.

## 1. Upload validation files

Upload only the AutoDL scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\validation\autodl\upload_autodl_files.ps1
```

Or upload the complete validation directory:

```bash
scp -P 10128 -r validation root@connect.nmb2.seetacloud.com:/root/autodl-tmp/A/
scp -P 10128 -r "E:/E题/E题数据/问题1/音频特征提取/音频文件" root@connect.nmb2.seetacloud.com:/root/autodl-tmp/audio
scp -P 10128 "E:/E题/E题数据/label-100.xlsx" root@connect.nmb2.seetacloud.com:/root/autodl-tmp/
```

The BERT model directory should also be copied to:

```text
/root/autodl-tmp/bert-base-uncased
```

The word-to-BERT mapping only needs the tokenizer files:

```powershell
powershell -ExecutionPolicy Bypass -File .\validation\autodl\upload_bert_tokenizer.ps1
```

This uploads `vocab.txt`, `tokenizer_config.json` and `tokenizer.json`. It does
not upload the 440 MB model weights because the mapping step does not run BERT.

## 2. Install dependencies on AutoDL

```bash
ssh -p 10128 root@connect.nmb2.seetacloud.com
cd /root/autodl-tmp/A
pip install openai-whisper transformers pandas openpyxl
python -c "import torch; print(torch.cuda.is_available())"
```

The last command should print `True`.

The updated Whisper script reads the extracted 16 kHz mono WAV files with the
Python standard `wave` module. It does not require the system `ffmpeg`
executable.

## 3. Generate word timestamps

```bash
python validation/autodl/autodl_whisper_words.py \
  --audio-dir /root/autodl-tmp/audio \
  --output-dir /root/autodl-tmp/A/validation/text_time/words \
  --model base
```

If an older version of the script still reports `No such file or directory:
'ffmpeg'`, either re-upload the updated script:

```powershell
powershell -ExecutionPolicy Bypass -File .\validation\autodl\upload_autodl_files.ps1
```

or install ffmpeg on the server:

```bash
apt-get update
apt-get install -y ffmpeg
```

Higher-accuracy option:

```bash
python validation/autodl/autodl_whisper_words.py \
  --audio-dir /root/autodl-tmp/audio \
  --output-dir /root/autodl-tmp/A/validation/text_time/words \
  --model small
```

If GPU memory is tight:

```bash
python validation/autodl/autodl_whisper_words.py \
  --audio-dir /root/autodl-tmp/audio \
  --output-dir /root/autodl-tmp/A/validation/text_time/words \
  --model tiny
```

## 4. Map words to BERT token positions

```bash
python validation/autodl/map_text_time.py \
  --words-dir /root/autodl-tmp/A/validation/text_time/words \
  --label-file /root/autodl-tmp/label-100.xlsx \
  --bert-path /root/autodl-tmp/bert-base-uncased \
  --output-dir /root/autodl-tmp/A/validation/text_time
```

## 5. Download results

```bash
scp -P 10128 -r root@connect.nmb2.seetacloud.com:/root/autodl-tmp/A/validation/text_time validation/
```

The result contains:

- `text_time_alignment.csv`: word time interval for every BERT subword token;
- `text_time_coverage.csv`: timestamped word count and coverage for every sample.
