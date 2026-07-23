$url = "http://127.0.0.1:8787"
$headroomExe = "C:\VSC_Project\ConvoPeq\.venv\Scripts\headroom.exe"
$totalBefore = 0; $totalAfter = 0
$startedByUs = $false

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Headroom Proxy 動作確認ベンチマーク" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ===== 事前チェック + 自動起動 =====
Write-Host "[Pre] Checking proxy..."
$proxyReady = $false
try {
    $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 3
    $c = $r.Content | ConvertFrom-Json
    Write-Host "  Already running: $( $c.status ), v$( $c.version )" -ForegroundColor Green
    $proxyReady = $true
} catch {
    Write-Host "  Not running. Starting proxy..." -ForegroundColor Yellow
    $null = Start-Process -FilePath $headroomExe -ArgumentList "proxy","--port","8787","--host","127.0.0.1" -WindowStyle Hidden -PassThru
    $startedByUs = $true
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 2
        try {
            $null = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 3
            $proxyReady = $true
            Write-Host "  Ready ( $(($i+1)*2)s )" -ForegroundColor Green
            break
        } catch {}
    }
}
if (-not $proxyReady) { Write-Host "  FAIL: proxy did not start" -ForegroundColor Red; exit 1 }

# ===== テスト1: JSON検索結果 =====
Write-Host "`n[Test 1] JSON search results (SmartCrusher)"
$results = @()
foreach ($j in 0..99) {
    $results += @{
        title = "Research Paper $j"
        authors = @("Author A", "Author B", "Author C")
        year = 2020 + ($j % 5)
        citations = [math]::Floor((100 - $j) * 2.5)
        score = [math]::Round((100 - $j) * 0.95, 2)
    }
}
$toolResult = @{ results = $results; total = 100; page = 1; query = "deep learning audio" }
$payload = @(
    @{ role = "system"; content = "Research assistant." }
    @{ role = "user"; content = "Summarize top papers." }
    @{ role = "assistant"; content = $null; tool_calls = @(@{ id = "c1"; type = "function"; function = @{ name = "search"; arguments = '{"q":"audio"}' } }) }
    @{ role = "tool"; tool_call_id = "c1"; content = ($toolResult | ConvertTo-Json -Depth 10) }
)
$body = @{ messages = $payload; model = "gpt-4o" } | ConvertTo-Json -Depth 10
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$cr = Invoke-WebRequest -Uri "$url/v1/compress" -Method Post -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 30
$sw.Stop()
$cc = $cr.Content | ConvertFrom-Json
$pct1 = [math]::Round((1 - $cc.compression_ratio) * 100, 1)
$totalBefore += $cc.tokens_before; $totalAfter += $cc.tokens_after
Write-Host "  $($cc.tokens_before) → $($cc.tokens_after) tokens  (-$pct1%)" -ForegroundColor $(if ($pct1 -gt 10) { "Green" } else { "White" })
Write-Host "  Time: $($sw.ElapsedMilliseconds)ms | Xform: $($cc.transforms_applied -join ', ')"

# ===== テスト2: C++コード =====
Write-Host "`n[Test 2] C++ source code (CodeCompressor)"
$codeText = @'
```cpp
class AudioProcessor {
public:
    void process(float* input, float* output, int n) {
        for (int i = 0; i < n; i++) fftBuffer[i] = input[i];
        fftwf_execute(fftPlan);
        for (int i = 0; i < n/2; i++) {
            float m = sqrt(spectrum[i][0]*spectrum[i][0] + spectrum[i][1]*spectrum[i][1]);
            float g = pow(10.0f, (compressDb(20*log10(m+1e-10f))-20*log10(m+1e-10f))/20);
            spectrum[i][0] *= g; spectrum[i][1] *= g;
        }
        fftwf_execute(ifftPlan);
        for (int i = 0; i < n; i++) output[i] = overlap[i] + fftBuffer[i];
    }
    float compressDb(float d) {
        float k = 6, t = -24, r = 4;
        if (d < t-k/2) return d;
        if (d < t+k/2) return d + (1/r-1)*(d-t+k/2)*(d-t+k/2)/(2*k);
        return t + (d-t)/r;
    }
};
```
'@
$payload2 = @(
    @{ role = "system"; content = "C++ expert." }
    @{ role = "user"; content = "Review this code." }
    @{ role = "assistant"; content = $null; tool_calls = @(@{ id = "c2"; type = "function"; function = @{ name = "read"; arguments = '{}' } }) }
    @{ role = "tool"; tool_call_id = "c2"; content = $codeText }
)
$body2 = @{ messages = $payload2; model = "gpt-4o" } | ConvertTo-Json -Depth 10
$sw2 = [System.Diagnostics.Stopwatch]::StartNew()
$cr2 = Invoke-WebRequest -Uri "$url/v1/compress" -Method Post -Body $body2 -ContentType "application/json" -UseBasicParsing -TimeoutSec 30
$sw2.Stop()
$cc2 = $cr2.Content | ConvertFrom-Json
$pct2 = [math]::Round((1 - $cc2.compression_ratio) * 100, 1)
$totalBefore += $cc2.tokens_before; $totalAfter += $cc2.tokens_after
Write-Host "  $($cc2.tokens_before) → $($cc2.tokens_after) tokens  (-$pct2%)" -ForegroundColor $(if ($pct2 -gt 10) { "Green" } else { "White" })
Write-Host "  Time: $($sw2.ElapsedMilliseconds)ms | Xform: $($cc2.transforms_applied -join ', ')"

# ===== テスト3: ビルドログ =====
Write-Host "`n[Test 3] Build log (LogCompressor)"
$logLines = @()
foreach ($i in 0..49) {
    $level = if ($i -lt 10) { "INFO" } elseif ($i -lt 20) { "WARN" } else { "ERROR" }
    $msg = if ($i -lt 10) { "Compiling AudioProcessor.cpp..." }
           elseif ($i -lt 20) { "AudioProcessor.cpp:142: signed/unsigned comparison" }
           else { "Build failed: AudioProcessor.cpp:42: undefined reference" }
    $logLines += "[2026-07-23 10:00:$($i.ToString('D2'))] $level  $msg"
}
$logOutput = $logLines -join "`n"
$payload3 = @(
    @{ role = "system"; content = "Build engineer." }
    @{ role = "user"; content = "What went wrong?" }
    @{ role = "assistant"; content = $null; tool_calls = @(@{ id = "c3"; type = "function"; function = @{ name = "read_log"; arguments = '{}' } }) }
    @{ role = "tool"; tool_call_id = "c3"; content = $logOutput }
)
$body3 = @{ messages = $payload3; model = "gpt-4o" } | ConvertTo-Json -Depth 10
$sw3 = [System.Diagnostics.Stopwatch]::StartNew()
$cr3 = Invoke-WebRequest -Uri "$url/v1/compress" -Method Post -Body $body3 -ContentType "application/json" -UseBasicParsing -TimeoutSec 30
$sw3.Stop()
$cc3 = $cr3.Content | ConvertFrom-Json
$pct3 = [math]::Round((1 - $cc3.compression_ratio) * 100, 1)
$totalBefore += $cc3.tokens_before; $totalAfter += $cc3.tokens_after
Write-Host "  $($cc3.tokens_before) → $($cc3.tokens_after) tokens  (-$pct3%)" -ForegroundColor $(if ($pct3 -gt 10) { "Green" } else { "White" })
Write-Host "  Time: $($sw3.ElapsedMilliseconds)ms | Xform: $($cc3.transforms_applied -join ', ')"

# ===== サマリー =====
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  サマリー" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
$totalSaved = $totalBefore - $totalAfter
$totalPct = if ($totalBefore -gt 0) { [math]::Round((1 - $totalAfter / $totalBefore) * 100, 1) } else { 0 }
Write-Host "  Total: $totalBefore → $totalAfter tokens" -ForegroundColor Cyan
Write-Host "  Saved: $totalSaved tokens (-$totalPct%)" -ForegroundColor Cyan
if ($startedByUs) { taskkill /F /IM headroom.exe *>$null; Write-Host "  Proxy stopped." -ForegroundColor DarkGray }
Write-Host "================================================" -ForegroundColor Cyan
