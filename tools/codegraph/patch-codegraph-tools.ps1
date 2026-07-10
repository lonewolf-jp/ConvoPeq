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
$graphTargetPath = Join-Path $repoRoot ".venv-codegraph\Lib\site-packages\codegraph_mcp\core\graph.py"
$indexerTargetPath = Join-Path $repoRoot ".venv-codegraph\Lib\site-packages\codegraph_mcp\core\indexer.py"

if (-not (Test-Path -LiteralPath $toolsTargetPath)) {
    throw "Target file not found: $toolsTargetPath"
}

if (-not (Test-Path -LiteralPath $graphTargetPath)) {
    throw "Target file not found: $graphTargetPath"
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


def _build_symbol_lookup_keys(entity_id: str) -> list[str]:
    """Build normalized symbol lookup keys for unresolved target fallback."""
    if not entity_id:
        return []

    raw = entity_id.strip()
    if not raw:
        return []

    paren_index = raw.find("(")
    if paren_index >= 0:
        raw = raw[:paren_index]

    keys: set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        if not value:
            return
        keys.add(value)

    add(raw)

    if "::" in raw:
        add(raw.rsplit("::", 1)[-1])

    if "/" in raw:
        add(raw.rsplit("/", 1)[-1])

    if "\\" in raw:
        add(raw.rsplit("\\", 1)[-1])

    return sorted(keys)
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

$oldFindCallers = @'
async def _handle_find_callers(
    args: dict[str, Any],
    engine: Any,
    config: Config,
) -> dict[str, Any]:
    """Handle find_callers tool."""
    entities = await engine.find_callers(args["entity_id"])
    return {
        "callers": [
            {"id": e.id, "name": e.name, "type": e.type.value}
            for e in entities
        ]
    }
'@

$newFindCallers = @'
async def _handle_find_callers(
    args: dict[str, Any],
    engine: Any,
    config: Config,
) -> dict[str, Any]:
    """Handle find_callers tool."""
    requested_entity = args["entity_id"]
    direct_entities = await engine.find_callers(requested_entity)

    callers: dict[str, dict[str, Any]] = {}
    for e in direct_entities:
        callers[e.id] = {
            "id": e.id,
            "name": e.name,
            "type": e.type.value,
            "resolution": "direct",
        }

    # Fallback for unresolved::<symbol> call targets generated by the parser.
    keys = _build_symbol_lookup_keys(requested_entity)
    unresolved_targets = [f"unresolved::{key}" for key in keys]

    if unresolved_targets:
        placeholders = ", ".join("?" for _ in unresolved_targets)
        cursor = await engine._connection.execute(
            (
                "SELECT DISTINCT e.id, e.name, e.type "
                "FROM entities e "
                "JOIN relations r ON e.id = r.source_id "
                f"WHERE r.type = 'calls' AND r.target_id IN ({placeholders})"
            ),
            tuple(unresolved_targets),
        )
        rows = await cursor.fetchall()
        for row in rows:
            entity_id = row[0]
            if entity_id not in callers:
                callers[entity_id] = {
                    "id": entity_id,
                    "name": row[1],
                    "type": row[2],
                    "resolution": "unresolved-target-fallback",
                }

    return {
        "callers": sorted(callers.values(), key=lambda item: (item["name"], item["id"])),
        "lookup": {
            "requested": requested_entity,
            "symbolKeys": keys,
            "unresolvedTargets": unresolved_targets,
        },
    }
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

if ($toolsPatched -notmatch 'def _build_symbol_lookup_keys\(') {
    $symbolHelperOnly = @'


def _build_symbol_lookup_keys(entity_id: str) -> list[str]:
    """Build normalized symbol lookup keys for unresolved target fallback."""
    if not entity_id:
        return []

    raw = entity_id.strip()
    if not raw:
        return []

    paren_index = raw.find("(")
    if paren_index >= 0:
        raw = raw[:paren_index]

    keys: set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        if not value:
            return
        keys.add(value)

    add(raw)

    if "::" in raw:
        add(raw.rsplit("::", 1)[-1])

    if "/" in raw:
        add(raw.rsplit("/", 1)[-1])

    if "\\" in raw:
        add(raw.rsplit("\\", 1)[-1])

    return sorted(keys)
'@

    $toolsPatchedWithSymbolHelper = [regex]::Replace(
        $toolsPatched,
        '(return sorted\(\{candidate\.lower\(\) for candidate in candidates\}\)\n)',
        ('$1' + $symbolHelperOnly + "`n"),
        1
    )

    if ($toolsPatchedWithSymbolHelper -eq $toolsPatched) {
        throw "Failed to inject _build_symbol_lookup_keys helper."
    }

    $toolsPatched = $toolsPatchedWithSymbolHelper
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

if ($toolsPatched -notmatch 'resolution": "unresolved-target-fallback"') {
    $callersPattern = 'async def _handle_find_callers\([\s\S]*?\n\nasync def _handle_find_callees\('
    $callersReplacement = "$newFindCallers`n`nasync def _handle_find_callees("
    $patchedCallers = [regex]::Replace($toolsPatched, $callersPattern, $callersReplacement, 1)
    if ($patchedCallers -ne $toolsPatched) {
        $toolsPatched = $patchedCallers
        $toolsChanged = $true
    }
    else {
        throw "Unsupported _handle_find_callers layout. Please inspect tools.py manually."
    }
}

$expectedMarkers = @(
    'def _build_file_lookup_candidates(',
    'def _build_symbol_lookup_keys(',
    'lookup_candidates = _build_file_lookup_candidates(validated_path, config.repo_path)',
    'resolution": "unresolved-target-fallback"',
    'WHERE LOWER(file_path) IN (',
    'return await _handle_analyze_module(args, engine, config)'
)

$missingMarkers = @($expectedMarkers | Where-Object { $toolsPatched.IndexOf($_, [System.StringComparison]::Ordinal) -lt 0 })
if ($missingMarkers.Count -gt 0) {
    throw "Patch verification failed. Missing markers: $($missingMarkers -join ', ')"
}

$graphOriginalContent = Get-Content -LiteralPath $graphTargetPath -Raw -Encoding UTF8
$graphNewline = if ($graphOriginalContent.Contains("`r`n")) { "`r`n" } else { "`n" }
$graphContent = $graphOriginalContent -replace "`r?`n", "`n"
$graphPatched = $graphContent
$graphChanged = $false

if ($graphPatched -notmatch 'def _build_symbol_lookup_keys\(') {
    $graphHelperBlock = @'

    def _build_symbol_lookup_keys(self, entity_id: str) -> list[str]:
        """Build normalized symbol lookup keys for unresolved target fallback."""
        if not entity_id:
            return []

        raw = entity_id.strip()
        if not raw:
            return []

        paren_index = raw.find("(")
        if paren_index >= 0:
            raw = raw[:paren_index]

        keys: set[str] = set()

        def add(value: str) -> None:
            value = value.strip()
            if not value:
                return
            keys.add(value)

        add(raw)

        if "::" in raw:
            add(raw.rsplit("::", 1)[-1])

        if "/" in raw:
            add(raw.rsplit("/", 1)[-1])

        if "\\" in raw:
            add(raw.rsplit("\\", 1)[-1])

        return sorted(keys)

'@

    $graphPatchedWithHelper = [regex]::Replace(
        $graphPatched,
        '(\n\s*if row:\n\s*return self\._row_to_entity\(row\)\n\s*return None\n)',
        ('$1' + $graphHelperBlock),
        1
    )

    if ($graphPatchedWithHelper -eq $graphPatched) {
        throw "Failed to inject _build_symbol_lookup_keys helper into graph.py."
    }

    $graphPatched = $graphPatchedWithHelper
    $graphChanged = $true
}

if ($graphPatched -notmatch 'unresolvedTargets') {
    $newFindDependencies = @'
    async def find_dependencies(
        self, entity_id: str, depth: int = 1
    ) -> QueryResult:
        """
        Find dependencies of an entity up to given depth.

        Supports partial entity_id matching.

        Requirements: REQ-TLS-002
        """
        # Resolve partial entity_id (direct match first)
        resolved_id = await self.resolve_entity_id(entity_id)

        root_ids: list[str] = []
        unresolved_targets: list[str] = []
        symbol_keys: list[str] = []

        if resolved_id:
            root_ids.append(resolved_id)
        else:
            # Fallback for unresolved::<symbol> call targets generated by parser
            symbol_keys = self._build_symbol_lookup_keys(entity_id)
            unresolved_targets = [f"unresolved::{key}" for key in symbol_keys]

            if unresolved_targets:
                placeholders = ", ".join("?" for _ in unresolved_targets)
                cursor = await self._connection.execute(
                    (
                        "SELECT DISTINCT source_id FROM relations "
                        f"WHERE type = 'calls' AND target_id IN ({placeholders})"
                    ),
                    tuple(unresolved_targets),
                )
                root_ids.extend(row[0] for row in await cursor.fetchall())

        if not root_ids:
            return QueryResult(
                entities=[],
                relations=[],
                metadata={
                    "requested": entity_id,
                    "resolvedRootIds": [],
                    "symbolKeys": symbol_keys,
                    "unresolvedTargets": unresolved_targets,
                },
            )

        # Preserve stable traversal order and avoid duplicate roots
        root_ids = list(dict.fromkeys(root_ids))

        visited: set[str] = set()
        entities: list[Entity] = []
        relations: list[Relation] = []

        async def traverse(eid: str, current_depth: int) -> None:
            if eid in visited or current_depth > depth:
                return
            visited.add(eid)

            entity = await self.get_entity(eid)
            if entity:
                entities.append(entity)

            cursor = await self._connection.execute(
                """
                SELECT target_id, type, weight FROM relations
                WHERE source_id = ? AND type IN ('imports', 'calls', 'uses')
                """,
                (eid,),
            )
            for row in await cursor.fetchall():
                rel = Relation(
                    source_id=eid,
                    target_id=row[0],
                    type=RelationType(row[1]),
                    weight=row[2],
                )
                relations.append(rel)
                await traverse(row[0], current_depth + 1)

        for root_id in root_ids:
            await traverse(root_id, 0)

        return QueryResult(
            entities=entities,
            relations=relations,
            metadata={
                "requested": entity_id,
                "resolvedRootIds": root_ids,
                "symbolKeys": symbol_keys,
                "unresolvedTargets": unresolved_targets,
            },
        )
'@

    $depsPattern = 'async def find_dependencies\([\s\S]*?\n\s*return QueryResult\(entities=entities, relations=relations\)\n'
    $graphPatchedWithDeps = [regex]::Replace($graphPatched, $depsPattern, ($newFindDependencies + "`n"), 1)
    if ($graphPatchedWithDeps -eq $graphPatched) {
        throw "Unsupported find_dependencies layout. Please inspect graph.py manually."
    }

    $graphPatched = $graphPatchedWithDeps
    $graphChanged = $true
}

$graphExpectedMarkers = @(
    'def _build_symbol_lookup_keys(self, entity_id: str) -> list[str]:',
    'unresolvedTargets',
    'SELECT DISTINCT source_id FROM relations',
    'resolvedRootIds'
)

$missingGraphMarkers = @($graphExpectedMarkers | Where-Object { $graphPatched.IndexOf($_, [System.StringComparison]::Ordinal) -lt 0 })
if ($missingGraphMarkers.Count -gt 0) {
    throw "Graph patch verification failed. Missing markers: $($missingGraphMarkers -join ', ')"
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

if ($indexerPatched.IndexOf('".musubi", "storage",', [System.StringComparison]::Ordinal) -lt 0) {
    if ($indexerPatched.IndexOf('".venv", ".venv-codegraph", "target", "dist", "build", ".codegraph",', [System.StringComparison]::Ordinal) -ge 0) {
        $indexerPatched = $indexerPatched.Replace(
            '".venv", ".venv-codegraph", "target", "dist", "build", ".codegraph",',
            '".venv", ".venv-codegraph", "target", "dist", "build", ".codegraph", ".musubi", "storage",'
        )
        $indexerChanged = $true
    }
    elseif ($indexerPatched.IndexOf('".venv", "target", "dist", "build", ".codegraph",', [System.StringComparison]::Ordinal) -ge 0) {
        $indexerPatched = $indexerPatched.Replace(
            '".venv", "target", "dist", "build", ".codegraph",',
            '".venv", "target", "dist", "build", ".codegraph", ".musubi", "storage",'
        )
        $indexerChanged = $true
    }
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
    '".musubi", "storage",',
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
    Write-Info " - $graphTargetPath"
    Write-Info " - $indexerTargetPath"
    exit 0
}

if (-not ($toolsChanged -or $graphChanged -or $indexerChanged)) {
    Write-Info "Patch already applied. No changes required."
    exit 0
}

if ($DryRun) {
    Write-Info "Dry run successful. Patch would be applied to:"
    Write-Info " - $toolsTargetPath"
    Write-Info " - $graphTargetPath"
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

if ($graphChanged) {
    $graphFinalContent = $graphPatched -replace "`n", $graphNewline
    Set-Content -LiteralPath $graphTargetPath -Value $graphFinalContent -Encoding UTF8 -NoNewline
    Write-Info "Patch applied to $graphTargetPath"
}
else {
    Write-Info "graph.py already patched."
}
