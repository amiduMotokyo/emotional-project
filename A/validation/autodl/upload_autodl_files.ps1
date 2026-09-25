$ErrorActionPreference = "Stop"

$Server = "root@connect.nmb2.seetacloud.com"
$Port = "10128"
$RemoteDir = "/root/autodl-tmp/A/validation/autodl"
$LocalDir = $PSScriptRoot

Write-Host "Creating remote directory..."
ssh -p $Port $Server "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directory."
}

$Files = @(
    (Join-Path $LocalDir "autodl_whisper_words.py"),
    (Join-Path $LocalDir "map_text_time.py"),
    (Join-Path $LocalDir "AUTODL_COMMANDS.md"),
    (Join-Path $LocalDir "upload_bert_tokenizer.ps1")
)

foreach ($File in $Files) {
    if (-not (Test-Path -LiteralPath $File)) {
        throw "Local file not found: $File"
    }
    Write-Host "Uploading $File"
    scp -P $Port $File "${Server}:${RemoteDir}/"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload: $File"
    }
}

Write-Host "Upload complete:"
ssh -p $Port $Server "ls -l '$RemoteDir'"
