$url = "http://127.0.0.1:8787"

Write-Host "=== Headroom Proxy 動作確認 ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Health Check" -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 5
    $c = $r.Content | ConvertFrom-Json
    $ok = if ($c.status -eq "healthy") { "OK" } else { "UNEXPECTED" }
    Write-Host "  Status:   $($c.status) [$ok]"
    Write-Host "  Version:  $($c.version)"
    Write-Host "  Uptime:   $($c.uptime_seconds)s"
    Write-Host "  RustCore: $($c.rust_core)"
    Write-Host "  PID:      $($c.pid)"
} catch {
    Write-Host "  FAIL: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/3] Stats Overview" -ForegroundColor Yellow
try {
    $s = (Invoke-WebRequest -Uri "$url/stats" -UseBasicParsing -TimeoutSec 5).Content | ConvertFrom-Json
    Write-Host "  Mode:          $($s.summary.mode)"
    Write-Host "  Optimize:      $($s.config.optimize)"
    Write-Host "  Cache Enabled: $($s.config.cache)"
    Write-Host "  Requests:      $($s.requests.total)"
    Write-Host "  Failures:      $($s.requests.failed)"
    Write-Host "  Tokens Saved:  $($s.tokens.saved)"
} catch {
    Write-Host "  FAIL: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "[3/3] Compression Endpoint" -ForegroundColor Yellow
try {
    $payload = @{
        messages = @(
            @{ role = "user"; content = "Hello" },
            @{ role = "assistant"; content = "Hi there!" }
        )
        model = "gpt-4o"
    } | ConvertTo-Json

    $cr = Invoke-WebRequest -Uri "$url/v1/compress" -Method Post -Body $payload -ContentType "application/json" -UseBasicParsing -TimeoutSec 10
    $cc = $cr.Content | ConvertFrom-Json
    Write-Host "  Before:    $($cc.tokens_before) tokens"
    Write-Host "  After:     $($cc.tokens_after) tokens"
    Write-Host "  Saved:     $($cc.tokens_saved) tokens"
    Write-Host "  Ratio:     $($cc.compression_ratio)"
    Write-Host "  Transforms: $($cc.transforms_applied -join ', ')"
} catch {
    Write-Host "  FAIL: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 完了 ===" -ForegroundColor Green
