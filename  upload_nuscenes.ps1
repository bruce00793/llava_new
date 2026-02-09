# ============================================
# nuScenes 数据上传脚本 (SCP)
# 在 Windows PowerShell 中运行
# ============================================

# ⚠️ 请修改服务器IP地址
$SERVER = "192.168.1.84"
$USER = "cly"
$REMOTE_PATH = "/home/cly/auto/llava_test/LLaVA/data/nuscenes/"
$LOCAL_PATH = "D:\下载包"

# 要上传的文件夹（蓝色高亮的）
$FOLDERS = @(
    "v1.0-trainval01_blobs",
    "v1.0-trainval03_blobs",
    "v1.0-trainval04_blobs",
    "v1.0-trainval07_blobs",
    "v1.0-trainval08_blobs"
)

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    nuScenes 数据上传脚本               ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "服务器: $USER@$SERVER" -ForegroundColor Yellow
Write-Host "远程路径: $REMOTE_PATH" -ForegroundColor Yellow
Write-Host "本地路径: $LOCAL_PATH" -ForegroundColor Yellow
Write-Host ""

$total = $FOLDERS.Count
$current = 0

foreach ($folder in $FOLDERS) {
    $current++
    $sourcePath = "$LOCAL_PATH\$folder"
    
    if (Test-Path $sourcePath) {
        Write-Host "[$current/$total] 上传: $folder" -ForegroundColor Green
        Write-Host "────────────────────────────────────────" -ForegroundColor DarkGray
        
        scp -r "$sourcePath" "${USER}@${SERVER}:${REMOTE_PATH}"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 完成!" -ForegroundColor Green
        } else {
            Write-Host "✗ 失败!" -ForegroundColor Red
        }
    } else {
        Write-Host "[$current/$total] 跳过: $folder (不存在)" -ForegroundColor Yellow
    }
    Write-Host ""
}

Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    全部任务完成!                       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan