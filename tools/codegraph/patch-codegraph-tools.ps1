param(
    [switch]$DryRun,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info {
    param([string]$Message)
    Write-Host "[CodeGraphPatch] $Message"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$toolsTargetPath = Join-Path $repoRoot ".venv-codegraph\Lib\site-packages\codegraph_mcp\mcp\tools.py"
$indexerTargetPath = Join-Path $repoRoot ".venv-codegraph\Lib\site-packages\codegraph_mcp\core\indexer.py"

if (-not (Test-Path -LiteralPath $toolsTargetPath)) {
    throw "Target file not found: $toolsTargetPath"
}

if (-not (Test-Path -LiteralPath $indexerTargetPath)) {
    throw "Target file not found: $indexerTargetPath"
}

$toolsOriginalContent = Get-Content -LiteralPath $toolsTargetPath -Raw -Encoding UTF8
$toolsNewline = if ($toolsOriginalContent.Contains("`r`n")) { "`r`n" } else { "`n" }
$toolsContent = $toolsOriginalContent -replace "`r?`n", "`n"
$toolsPatched = $toolsContent
$toolsChanged = $false

$helperBlock = @'
def _build_file_lookup_candidates(path: "Path", repo_root: "Path") -> list[str]:
    """Build normalized lookup candidates for file_path DB matching."""
    repo_resolved = repo_root.resolve()
    resolved = path.resolve()

    candidates: set[str] = set()

    def add_candidate(value: str) -> None:
        if not value:
            return

        candidates.add(value)
        candidates.add(value.replace("\\", "/"))

    add_candidate(str(resolved))

    try:
        add_candidate(str(resolved.relative_to(repo_resolved)))
    except ValueError:
        pass

    return sorted({candidate.lower() for candidate in candidates})
'@

$oldAnalyze = @'
async def _handle_analyze_module(
    args: dict[str, Any],
    engine: Any,
    config: Config,
) -> dict[str, Any]:
    """Handle analyze_module_structure tool."""
    file_path = args["file_path"]

    cursor = await engine._connection.execute(
        "SELECT type, name, start_line, end_line FROM entities WHERE file_path = ?",
        (file_path,),
    )
    rows = await cursor.fetchall()

    return {
        "file": file_path,
        "entities": [
            {"type": row[0], "name": row[1], "lines": f"{row[2]}-{row[3]}"}
            for row in rows
        ],
    }
'@

$newAnalyze = @'
async def _handle_analyze_module(
    args: dict[str, Any],
    engine: Any,
    config: Config,
) -> dict[str, Any]:
    """Handle analyze_module_structure tool."""
    validated_path = _validate_path(args["file_path"], config.repo_path)
    if validated_path is None:
        return {"error": "Invalid path: access denied outside repository"}

    lookup_candidates = _build_file_lookup_candidates(validated_path, config.repo_path)
    if not lookup_candidates:
        return {"file": str(validated_path), "entities": []}

    placeholders = ", ".join("?" for _ in lookup_candidates)

    cursor = await engine._connection.execute(
        (
            "SELECT type, name, start_line, end_line FROM entities "
            f"WHERE LOWER(file_path) IN ({placeholders}) "
            "ORDER BY start_line, end_line, name"
        ),
        tuple(lookup_candidates),
    )
    rows = await cursor.fetchall()

    return {
        "file": str(validated_path),
        "entities": [
            {"type": row[0], "name": row[1], "lines": f"{row[2]}-{row[3]}"}
            for row in rows
        ],
    }
'@

$oldGetFileStructure = @'
async def _handle_get_file_structure(
    args: dict[str, Any],
    engine: Any,
    config: Config,
) -> dict[str, Any]:
    """Handle get_file_structure tool with path validation."""
    file_path = _validate_path(args["file_path"], config.repo_path)
    if file_path is None:
        return {"error": "Invalid path: access denied outside repository"}

    # Use validated path for analysis
    validated_args = {**args, "file_path": str(file_path.relative_to(config.repo_path))}
    return await _handle_analyze_module(validated_args, engine, config)
'@

$newGetFileStructure = @'
async def _handle_get_file_structure(
    args: dict[str, Any],
    engine: Any,
    config: Config,
) -> dict[str, Any]:
    """Handle get_file_structure tool with path validation."""
    return await _handle_analyze_module(args, engine, config)
'@

if ($toolsPatched -notmatch 'def _build_file_lookup_candidates\(') {
    $toolsPatched = [regex]::Replace(
        $toolsPatched,
        "logger = get_logger\(__name__\)\n\n",
        "logger = get_logger(__name__)`n`n$helperBlock`n",
        1
    )

    if ($toolsPatched -eq $toolsContent) {
        throw "Failed to inject _build_file_lookup_candidates helper."
    }

    $toolsChanged = $true
}

if ($toolsPatched -notmatch 'lookup_candidates = _build_file_lookup_candidates\(') {
    if ($toolsPatched.Contains($oldAnalyze)) {
        $toolsPatched = $toolsPatched.Replace($oldAnalyze, $newAnalyze)
        $toolsChanged = $true
    }
    else {
        throw "Unsupported _handle_analyze_module layout. Please inspect tools.py manually."
    }
}

if ($toolsPatched -notmatch 'return await _handle_analyze_module\(args, engine, config\)') {
    if ($toolsPatched.Contains($oldGetFileStructure)) {
        $toolsPatched = $toolsPatched.Replace($oldGetFileStructure, $newGetFileStructure)
        $toolsChanged = $true
    }
    else {
        throw "Unsupported _handle_get_file_structure layout. Please inspect tools.py manually."
    }
}

$expectedMarkers = @(
    'def _build_file_lookup_candidates(',
    'lookup_candidates = _build_file_lookup_candidates(validated_path, config.repo_path)',
    'WHERE LOWER(file_path) IN (',
    'return await _handle_analyze_module(args, engine, config)'
)

$missingMarkers = @($expectedMarkers | Where-Object { $toolsPatched.IndexOf($_, [System.StringComparison]::Ordinal) -lt 0 })
if ($missingMarkers.Count -gt 0) {
    throw "Patch verification failed. Missing markers: $($missingMarkers -join ', ')"
}

$indexerOriginalContent = Get-Content -LiteralPath $indexerTargetPath -Raw -Encoding UTF8
$indexerNewline = if ($indexerOriginalContent.Contains("`r`n")) { "`r`n" } else { "`n" }
$indexerContent = $indexerOriginalContent -replace "`r?`n", "`n"
$indexerPatched = $indexerContent
$indexerChanged = $false

if ($indexerPatched.IndexOf('".venv", "target", "dist", "build", ".codegraph",', [System.StringComparison]::Ordinal) -ge 0 -and
    $indexerPatched.IndexOf('".venv-codegraph", "target", "dist", "build", ".codegraph",', [System.StringComparison]::Ordinal) -lt 0) {
    $indexerPatched = $indexerPatched.Replace(
        '".venv", "target", "dist", "build", ".codegraph",',
        '".venv", ".venv-codegraph", "target", "dist", "build", ".codegraph",'
    )
    $indexerChanged = $true
}

if ($indexerPatched.IndexOf('exclude_dir_prefixes = (".venv-",)', [System.StringComparison]::Ordinal) -lt 0) {
    $indexerForLoopReplacement = @'
        exclude_dir_prefixes = (".venv-",)

        for path in repo_path.rglob("*"):
'@

    $indexerPatched = $indexerPatched.Replace(
        '        for path in repo_path.rglob("*"):',
        $indexerForLoopReplacement
    )
    $indexerChanged = $true
}

if ($indexerPatched.IndexOf('if any(ex in path.parts for ex in exclude_dirs):', [System.StringComparison]::Ordinal) -ge 0) {
    $indexerIfReplacement = @'
if any(
                    (part in exclude_dirs) or any(part.startswith(prefix) for prefix in exclude_dir_prefixes)
                    for part in path.parts
                ):
'@

    $indexerPatched = $indexerPatched.Replace(
        'if any(ex in path.parts for ex in exclude_dirs):',
        $indexerIfReplacement
    )
    $indexerChanged = $true
}

$indexerExpectedMarkers = @(
    '".venv-codegraph", "target", "dist", "build", ".codegraph",',
    'exclude_dir_prefixes = (".venv-",)',
    '(part in exclude_dirs) or any(part.startswith(prefix) for prefix in exclude_dir_prefixes)'
)

$missingIndexerMarkers = @($indexerExpectedMarkers | Where-Object { $indexerPatched.IndexOf($_, [System.StringComparison]::Ordinal) -lt 0 })
if ($missingIndexerMarkers.Count -gt 0) {
    throw "Indexer patch verification failed. Missing markers: $($missingIndexerMarkers -join ', ')"
}

if ($VerifyOnly) {
    Write-Info "Patch markers verified in:"
    Write-Info " - $toolsTargetPath"
    Write-Info " - $indexerTargetPath"
    exit 0
}

if (-not ($toolsChanged -or $indexerChanged)) {
    Write-Info "Patch already applied. No changes required."
    exit 0
}

if ($DryRun) {
    Write-Info "Dry run successful. Patch would be applied to:"
    Write-Info " - $toolsTargetPath"
    Write-Info " - $indexerTargetPath"
    exit 0
}

if ($toolsChanged) {
    $toolsFinalContent = $toolsPatched -replace "`n", $toolsNewline
    Set-Content -LiteralPath $toolsTargetPath -Value $toolsFinalContent -Encoding UTF8 -NoNewline
    Write-Info "Patch applied to $toolsTargetPath"
}
else {
    Write-Info "tools.py already patched."
}

if ($indexerChanged) {
    $indexerFinalContent = $indexerPatched -replace "`n", $indexerNewline
    Set-Content -LiteralPath $indexerTargetPath -Value $indexerFinalContent -Encoding UTF8 -NoNewline
    Write-Info "Patch applied to $indexerTargetPath"
}
else {
    Write-Info "indexer.py already patched."
}
