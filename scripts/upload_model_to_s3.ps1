param(
    [string]$ModelVersion = "2026-04-14-001",
    [string]$Bucket = "capstone-gachon",
    [string]$Prefix = "models",
    [string]$Region = "ap-northeast-2"
)

$localPath = Join-Path "upload_model" $ModelVersion
$s3Uri = "s3://$Bucket/$Prefix/$ModelVersion/"

if (-not (Test-Path $localPath)) {
    Write-Error "Local artifact directory not found: $localPath"
    exit 1
}

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Error "AWS CLI is not installed or not available in PATH."
    exit 1
}

Write-Host "Uploading local artifact directory..."
Write-Host "  Local Path : $localPath"
Write-Host "  Target S3  : $s3Uri"
Write-Host "  AWS Region : $Region"

aws s3 cp $localPath $s3Uri `
    --recursive `
    --region $Region `
    --only-show-errors

if ($LASTEXITCODE -ne 0) {
    Write-Error "S3 upload failed."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "S3 upload completed."
Write-Host "Final S3 target path: $s3Uri"
