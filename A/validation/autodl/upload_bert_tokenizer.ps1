$ErrorActionPreference = "Stop"

$Server = "root@connect.nmb2.seetacloud.com"
$Port = "10128"
$RemoteDir = "/root/autodl-tmp/bert-base-uncased"
$LocalDir = "E:\E题\bert-base-uncased"

$RequiredFiles = @(
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json"
)

foreach ($Name in $RequiredFiles) {
    $Path = Join-Path $LocalDir $Name
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Local tokenizer file not found: $Path"
    }
}

Write-Host "Creating remote BERT tokenizer directory..."
ssh -p $Port $Server "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directory."
}

foreach ($Name in $RequiredFiles) {
    $Path = Join-Path $LocalDir $Name
    Write-Host "Uploading $Path"
    scp -P $Port $Path "${Server}:${RemoteDir}/"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload: $Path"
    }
}

Write-Host "Upload complete:"
ssh -p $Port $Server "ls -lh '$RemoteDir'"
