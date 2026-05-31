param(
    [string]$RepoRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info {
    param([string]$Message)
    Write-Host "[CodeGraphPrune] $Message"
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

$dbPath = Join-Path $RepoRoot ".codegraph\graph.db"
if (-not (Test-Path -LiteralPath $dbPath)) {
    throw "CodeGraph DB not found: $dbPath"
}

$py = @'
import sqlite3
import sys


def main(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    where_venv = (
        "(lower(file_path) like '%/.venv-codegraph/%' "
        "or lower(file_path) like '%\\.venv-codegraph\\%' "
        "or lower(file_path) like '%/.venv/%' "
        "or lower(file_path) like '%\\.venv\\%')"
    )
    where_codeql = (
        "(lower(file_path) like '%/storage/codeql/%' "
        "or lower(file_path) like '%\\storage\\codeql\\%')"
    )
    where_vendor = (
        "(lower(file_path) like '%/juce/%' "
        "or lower(file_path) like '%\\juce\\%' "
        "or lower(file_path) like '%/r8brain-free-src/%' "
        "or lower(file_path) like '%\\r8brain-free-src\\%')"
    )
    where_noise = f"({where_venv} or {where_codeql} or {where_vendor})"

    where_venv_files = (
        "(lower(path) like '%/.venv-codegraph/%' "
        "or lower(path) like '%\\.venv-codegraph\\%' "
        "or lower(path) like '%/.venv/%' "
        "or lower(path) like '%\\.venv\\%')"
    )
    where_codeql_files = (
        "(lower(path) like '%/storage/codeql/%' "
        "or lower(path) like '%\\storage\\codeql\\%')"
    )
    where_vendor_files = (
        "(lower(path) like '%/juce/%' "
        "or lower(path) like '%\\juce\\%' "
        "or lower(path) like '%/r8brain-free-src/%' "
        "or lower(path) like '%\\r8brain-free-src\\%')"
    )
    where_noise_files = f"({where_venv_files} or {where_codeql_files} or {where_vendor_files})"

    total_before = cur.execute("select count(*) from entities").fetchone()[0]
    noise_before = cur.execute(f"select count(*) from entities where {where_noise}").fetchone()[0]

    cur.execute("BEGIN IMMEDIATE")
    cur.execute(f"create temp table _drop_ids as select id from entities where {where_noise}")
    cur.execute("delete from relations where source_id in (select id from _drop_ids) or target_id in (select id from _drop_ids)")
    deleted_relations = cur.rowcount
    cur.execute("delete from entities where id in (select id from _drop_ids)")
    deleted_entities = cur.rowcount
    cur.execute(f"delete from files where {where_noise_files}")
    deleted_files = cur.rowcount
    cur.execute("drop table _drop_ids")
    con.commit()

    total_after = cur.execute("select count(*) from entities").fetchone()[0]
    noise_after = cur.execute(f"select count(*) from entities where {where_noise}").fetchone()[0]

    print(f"total_before={total_before}")
    print(f"noise_before={noise_before}")
    print(f"deleted_entities={deleted_entities}")
    print(f"deleted_relations={deleted_relations}")
    print(f"deleted_files={deleted_files}")
    print(f"total_after={total_after}")
    print(f"noise_after={noise_after}")

    con.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: prune.py <graph.db>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
'@

Write-Info "Pruning CodeGraph DB noise from venv/codeql/vendor paths..."
$tmpPy = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".py")
try {
    Set-Content -LiteralPath $tmpPy -Value $py -Encoding UTF8 -NoNewline
    python "$tmpPy" "$dbPath"
}
finally {
    if (Test-Path -LiteralPath $tmpPy) {
        Remove-Item -LiteralPath $tmpPy -Force -ErrorAction SilentlyContinue
    }
}

if ($LASTEXITCODE -ne 0) {
    throw "Noise pruning failed with exit code $LASTEXITCODE"
}

Write-Info "Prune completed: $dbPath"
