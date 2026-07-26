# ConvoPeq AI Profile — headroom + context-mode + RTK(WSL) 統合設定
# 使用法:  . .\tools\convopeq-ai-profile.ps1
# またはプロファイルに追加:  Add-Content $PROFILE ". `"$env:ConvoPeqRoot\tools\convopeq-ai-profile.ps1`""

$script:ConvoPeqRoot = if ($env:ConvoPeqRoot) { $env:ConvoPeqRoot } else { "C:\VSC_Project\ConvoPeq" }

# ============================================================
# RTK (WSL版) 便利関数
# ============================================================

# RTK ultra-compact: 最大圧縮のラッパー
function rtk-u {
    param([Parameter(ValueFromRemainingArguments = $true)]$Args)
    $cmd = "cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk-u $($Args -join ' ')"
    wsl bash -c $cmd
}

# RTK 標準版ラッパー
function rtk {
    param([Parameter(ValueFromRemainingArguments = $true)]$Args)
    $cmd = "cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk $($Args -join ' ')"
    wsl bash -c $cmd
}

# RTK 節約統計表示
function rtk-gain {
    wsl bash -c "cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk gain"
}

# ============================================================
# context-mode 便利関数
# ============================================================

# context-mode 統計表示
function ctx-stats {
    context-mode --version
}

# ============================================================
# 3層パイプラインの状態表示
# ============================================================

function Show-AiPipelineStatus {
    Write-Host "=== ConvoPeq AI 3-Layer Pipeline Status ===" -ForegroundColor Cyan

    # headroom
    Write-Host "`n[headroom MCP]" -ForegroundColor Yellow
    try {
        $v = & "C:\Users\user\AppData\Roaming\Python\Python314\Scripts\headroom.exe" --version 2>&1
        Write-Host "  Version: $v" -ForegroundColor Green
    } catch {
        Write-Host "  Not available" -ForegroundColor Red
    }

    # context-mode
    Write-Host "`n[context-mode MCP]" -ForegroundColor Yellow
    try {
        $v = context-mode --version 2>&1 | Select-Object -First 1
        Write-Host "  $v" -ForegroundColor Green
    } catch {
        Write-Host "  Not available" -ForegroundColor Red
    }

    # RTK
    Write-Host "`n[RTK (WSL版)]" -ForegroundColor Yellow
    try {
        $v = wsl bash -c "~/.local/bin/rtk --version 2>&1"
        Write-Host "  Version: $v" -ForegroundColor Green
    } catch {
        Write-Host "  Not available" -ForegroundColor Red
    }

    # RTK savings
    Write-Host "`n[RTK Savings]"
    wsl bash -c "cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk gain 2>&1 | head -20"
}
