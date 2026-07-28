# Run ISR Bridge Runtime verifiers (matches isr-verification.yml step)
$ErrorActionPreference = 'Continue'
$failed = $false
$env:PYTHONIOENCODING = 'utf-8'

$verifiers = @(
    'tools\coverage_verifier.py'
    'tools\runtime_graph_authority_verifier.py|--mode baseline'
    'tools\capture_session_id_verifier.py'
    'tools\identity_authority_verifier.py'
    'tools\engine_runtime_authority_verifier.py'
    'tools\non_authoritative_observe_verifier.py'
    'tools\retire_authority_verifier.py'
    'tools\snapshot_authority_usage_verifier.py'
    'tools\authority_source_count_verifier.py'
    'tools\publication_authority_verifier.py'
    'tools\generate_publication_manifest.py|--verify --repo-root .'
    'tools\detect_publication_mutation.py'
    'tools\retire_ordering_verifier.py'
    'tools\authority_inventory_verifier.py'
    'tools\authority_duplication_verifier.py'
    'tools\projection_origin_verifier.py'
    'tools\diagnostic_field_verifier.py'
)

Write-Host "=== ISR Bridge Runtime Verifiers ==="
foreach ($v in $verifiers) {
    $parts = $v -split '\|', 2
    $script = $parts[0]
    $argsStr = if ($parts.Length -gt 1) { $parts[1] } else { '' }
    Write-Host "::group::Running $script $argsStr"
    if ($argsStr -ne '') {
        $argArray = $argsStr -split ' '
        python $script @argArray 2>&1
    } else {
        python $script 2>&1
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "::error::$script failed (exit=$exitCode)"
        $failed = $true
    } else {
        Write-Host "PASSED: $script"
    }
    Write-Host "::endgroup::"
}

if ($failed) {
    Write-Host "ONE OR MORE VERIFIERS FAILED"
    exit 1
} else {
    Write-Host "ALL VERIFIERS PASSED"
    exit 0
}
