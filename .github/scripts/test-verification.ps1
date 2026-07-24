# Test ISR verification scripts locally
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

foreach ($v in $verifiers) {
    $parts = $v -split '\|', 2
    $script = $parts[0]
    $argsArr = if ($parts.Length -gt 1) { $parts[1] -split ' ' } else { @() }
    Write-Host "Running $v..."
    python $script @argsArr 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $v (exit=$LASTEXITCODE)"
        $failed = $true
    } else {
        Write-Host "PASSED: $v"
    }
}

if ($failed) {
    Write-Host 'One or more verifiers failed.'
    exit 1
}
Write-Host 'All verifiers passed.'
exit 0
