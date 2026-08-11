$path = '^src/tests/AudioEngineHarness/DeferredFlowIntegrationTests\.cpp$'
Write-Host "path=[$path]"
if ($path -match '^src/tests/') { Write-Host 'MATCHED' } else { Write-Host 'NOT MATCHED' }

# 実際のポリシーから読み込んで確認
$policy = Get-Content -LiteralPath '.github/isr-ai-governance-policy.json' -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($allow in @($policy.requestRebuildDirectCall.allowlist)) {
    $allowPathRegex = "$($allow.pathRegex)"
    $m = $allowPathRegex -match '^src/tests/'
    Write-Host "allowlist path=[$allowPathRegex] src/tests match=$m"
}
