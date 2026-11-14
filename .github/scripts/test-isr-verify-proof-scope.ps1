#requires -Version 7.0
<#
.SYNOPSIS
    Self-test for isr-verify-proof-scope.ps1.

.DESCRIPTION
    Builds fixture repositories under a temporary directory and checks that the
    proof-scope verifier accepts the two supported implementations (legacy R25
    suppression and the B-11 force-evidence variant) and rejects the four
    regressions that would weaken the contract.

    The verifier is executed as a child process against each fixture root, so
    the test exercises exactly what CI runs (same script, same exit codes).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$verifier = Join-Path $PSScriptRoot 'isr-verify-proof-scope.ps1'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$exporterSource = Join-Path $repoRoot 'src\audioengine\ISREvidenceExporter.cpp'

if (-not (Test-Path $verifier)) { throw "verifier not found: $verifier" }
if (-not (Test-Path $exporterSource)) { throw "exporter source not found: $exporterSource" }

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("proof-scope-selftest-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot | Out-Null

function New-Fixture {
    param(
        [string]$Name,
        [string]$DebugRuntimeSource
    )

    $root = Join-Path $tempRoot $Name
    $audioengine = Join-Path $root 'src\audioengine'
    New-Item -ItemType Directory -Path $audioengine -Force | Out-Null
    Copy-Item -Path $exporterSource -Destination (Join-Path $audioengine 'ISREvidenceExporter.cpp')
    Set-Content -Path (Join-Path $audioengine 'ISRDebugRuntime.cpp') -Value $DebugRuntimeSource -Encoding utf8NoBOM
    return $root
}

function Invoke-Verifier {
    param([string]$Root)

    $output = & pwsh -NoProfile -ExecutionPolicy Bypass -File $verifier -Root $Root 2>&1
    return [PSCustomObject]@{
        ExitCode = $LASTEXITCODE
        Output   = ($output | Out-String).Trim()
    }
}

$commonTail = @'

void DebugRuntime::emitFinalViolationReport()
{
    const auto path = std::filesystem::current_path() / "evidence" / "hb_violation_report.json";
    hbVerifier_.setTraceRuntime(&hbTraceRuntime_);
    hbVerifier_.emitViolationReport(path);
}
} // namespace convo::isr
'@

$cases = @()

# 1) Legacy R25 implementation: plain Release suppression, CI-only trace emission.
$cases += [PSCustomObject]@{
    Name     = 'legacy-release-suppression'
    Expected = 0
    Source   = @'
namespace convo::isr
{
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release") {
        return;
    }
    if (mode == "CI") {
        emitHBTrace();
    }
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release") {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 2) B-11 implementation: force-evidence switch keeps Release evidence opt-in.
$cases += [PSCustomObject]@{
    Name     = 'b11-force-evidence'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    static const bool enabled = std::getenv("CONVOPEQ_ISR_FORCE_EVIDENCE") != nullptr;
    return enabled;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 3) Release emits unconditionally (no Release early-return at all).
$cases += [PSCustomObject]@{
    Name     = 'release-emits-unconditionally'
    Expected = 1
    Source   = @'
namespace convo::isr
{
void DebugRuntime::emitCIArtifacts()
{
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release") {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 4) Force-evidence switch exists but the Release guard ignores it.
$cases += [PSCustomObject]@{
    Name     = 'release-guard-ignores-force'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    static const bool enabled = std::getenv("CONVOPEQ_ISR_FORCE_EVIDENCE") != nullptr;
    return enabled;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release") {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 5) CI/Debug trace emission removed from emitCIArtifacts().
$cases += [PSCustomObject]@{
    Name     = 'ci-trace-emission-removed'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    static const bool enabled = std::getenv("CONVOPEQ_ISR_FORCE_EVIDENCE") != nullptr;
    return enabled;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitShadowCompareCadenceReport();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 6) emitHBTrace() lost its own Release suppression.
$cases += [PSCustomObject]@{
    Name     = 'hbtrace-release-suppression-removed'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    static const bool enabled = std::getenv("CONVOPEQ_ISR_FORCE_EVIDENCE") != nullptr;
    return enabled;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 7) A line comment marker inside a normal string literal must not swallow the code
#    that follows it.
$cases += [PSCustomObject]@{
    Name     = 'slash-slash-inside-string'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    const char* url = "https://example.com/hb";
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 8) A block comment opener inside a normal string literal must not erase code.
$cases += [PSCustomObject]@{
    Name     = 'slash-star-inside-string'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    const char* glob = "/* not a comment */";
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 9) Comment markers inside a raw string literal.
$cases += [PSCustomObject]@{
    Name     = 'comment-markers-inside-raw-string'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    const char* doc = R"hb(/* not a comment */ // nor this)hb";
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 10) A character literal holding a brace ('}') must not close the function early.
$cases += [PSCustomObject]@{
    Name     = 'char-literal-brace'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    const char closing = '}';
    (void) closing;
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 11) Source text that only *looks* like the Release guard, inside a string literal.
#     The verifier must not accept it as code.
$cases += [PSCustomObject]@{
    Name     = 'release-guard-only-inside-string'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const char* fake = "if (mode == \"Release\") { return; }";
    (void) fake;
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 12) emitViolationReport mentioned only inside a string literal must not satisfy A(D1).
$cases += [PSCustomObject]@{
    Name     = 'ad1-call-only-inside-string'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}

void DebugRuntime::emitFinalViolationReport()
{
    const char* doc = "hbVerifier_.emitViolationReport(path);";
    (void) doc;
}
} // namespace convo::isr
'@
}
# 13) Exporter contract must still be enforced.
$cases += [PSCustomObject]@{
    Name     = 'exporter-contract-broken'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
    MutateExporter = $true
}

# 14) A(D1) call removed from emitFinalViolationReport(), but present in a later function.
#    A whole-file "find the call somewhere after the function" check would pass; the
#    function-body check must fail.
$cases += [PSCustomObject]@{
    Name     = 'ad1-call-removed-from-body'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}

void DebugRuntime::emitFinalViolationReport()
{
    hbVerifier_.setTraceRuntime(&hbTraceRuntime_);
}

void DebugRuntime::emitSomethingElse()
{
    hbVerifier_.emitViolationReport("/tmp/other.json");
}
} // namespace convo::isr
'@
}

# 15) A numeric digit separator (50'000'000) must not be scanned as a character literal.
$cases += [PSCustomObject]@{
    Name     = 'numeric-digit-separator'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    constexpr long long kThreshold = 50'000'000;
    (void) kThreshold;
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 18) Escaped quote followed by // and /* inside the same literal.
$cases += [PSCustomObject]@{
    Name     = 'escaped-quote-with-comment-markers'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    const char* s = "escaped quote: \" // still inside /* here */";
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 19) CRLF line-continued string literal containing a comment marker.
$src19 = @"
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    const char* s = "abc\`r`n// not a comment";
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}

void DebugRuntime::emitFinalViolationReport()
{
    const auto path = std::filesystem::current_path() / "evidence" / "hb_violation_report.json";
    hbVerifier_.setTraceRuntime(&hbTraceRuntime_);
    hbVerifier_.emitViolationReport(path);
}
} // namespace convo::isr
"@
$cases += [PSCustomObject]@{
    Name     = 'crlf-line-continued-string'
    Expected = 0
    Source   = $src19
}

# 16) Prefixed raw string literal (u8R"...") must be scanned as one literal.
$cases += [PSCustomObject]@{
    Name     = 'prefixed-raw-string'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    const char* doc = u8R"hb(/* not a comment */ // nor this )hb";
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 17) Braces inside string literals must not break the function-body extraction.
$cases += [PSCustomObject]@{
    Name     = 'braces-inside-string-literal'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    const char* note = "} { }";
    hbTraceRuntime_.emitTrace(path);
}

void DebugRuntime::emitFinalViolationReport()
{
    const char* note = "} } {";
    const auto path = std::filesystem::current_path() / "evidence" / "hb_violation_report.json";
    hbVerifier_.setTraceRuntime(&hbTraceRuntime_);
    hbVerifier_.emitViolationReport(path);
}
} // namespace convo::isr
'@
}

# 20) Escaped single quote inside a character literal in the function body.
$cases += [PSCustomObject]@{
    Name     = 'char-literal-escaped-quote'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    // エスケープされた単一引用符のあとに、文字リテラルを含む小さなブロックを同じ行へ置く。
    // 文字リテラルの走査が崩れると 'a' の引用符までを 1 リテラルとして飲み込み、
    // ブロックの開き波括弧が数えられなくなる。閉じ波括弧は次行にあるため、
    // 関数本体がそこで終わり、下の Release ガードと emitHBTrace() が範囲外になる。
    const char quote = '\''; { const char letter = 'a';
    }
    (void) quote;
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    const char quote = '\'';
    (void) quote;
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 21) Release guard joined with || (suppresses forced evidence as well).
$cases += [PSCustomObject]@{
    Name     = 'release-guard-with-or'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" || ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 22) The HB trace call exists only inside a string literal.
$cases += [PSCustomObject]@{
    Name     = 'hbtrace-call-only-inside-string'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    const char* note = "emitHBTrace();";
    (void) note;
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 23) emitHBTrace() emits before its Release suppression.
$cases += [PSCustomObject]@{
    Name     = 'hbtrace-emit-before-guard'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    hbTraceRuntime_.emitTrace(path);
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
}
'@ + $commonTail
}

# 24) The Release guard is commented out with a line-continuation backslash.
$cases += [PSCustomObject]@{
    Name     = 'release-guard-commented-out-with-continuation'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    // temporarily disabled \
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 25) A continued comment that does not hide the guard must still pass.
$cases += [PSCustomObject]@{
    Name     = 'comment-continuation-without-hiding-guard'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    // note: the guard below is real, not commented \
    // (this line continues the comment)
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 26) The Release test itself is negated: !(mode == "Release") && ! force.
$cases += [PSCustomObject]@{
    Name     = 'release-guard-negated-mode'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (!(mode == "Release") && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 27) The same contract written with parentheses around each term.
$cases += [PSCustomObject]@{
    Name     = 'release-guard-parenthesized'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if ((mode == "Release") && (! isForceEvidenceEmission())) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if ((mode == "Release") && (! isForceEvidenceEmission())) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 28) A definition disabled by #if 0 comes first; the active definition has no guard.
$cases += [PSCustomObject]@{
    Name     = 'if0-disabled-definition-before-active'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

#if 0
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}
#endif

void DebugRuntime::emitCIArtifacts()
{
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 29) The Release literal is spelled in lower case, so it never matches at runtime.
$cases += [PSCustomObject]@{
    Name     = 'release-literal-lowercase'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 30) The guard sits inside a block that never runs.
$cases += [PSCustomObject]@{
    Name     = 'nested-release-guard'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (false) {
        if (mode == "Release" && ! isForceEvidenceEmission()) {
            return;
        }
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 31) The trace call is reachable only on the Release path.
$cases += [PSCustomObject]@{
    Name     = 'hbtrace-inside-release-block'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    if (mode == "Release") {
        emitHBTrace();
    }
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 32) emitHBTrace() keeps the guard but the actual trace emission is gone.
$cases += [PSCustomObject]@{
    Name     = 'emittrace-call-missing'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
}
'@ + $commonTail
}

# 33) A bad definition under #if 0 must be ignored, not rejected.
$cases += [PSCustomObject]@{
    Name     = 'if0-disabled-bad-definition'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

#if 0
void DebugRuntime::emitCIArtifacts()
{
    emitHBTrace();
}
#endif

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 34) A bare compound block around the trace call still runs unconditionally.
$cases += [PSCustomObject]@{
    Name     = 'hbtrace-in-bare-block'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    {
        emitHBTrace();
    }
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 35) Two definitions under conditional compilation: which one is active cannot be decided.
$cases += [PSCustomObject]@{
    Name     = 'definition-under-conditional-compilation'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

#if defined(CONVOPEQ_ALT_TRACE)
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}
#else
void DebugRuntime::emitCIArtifacts()
{
}
#endif

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 36) The Release guard lives inside an unresolved conditional-compilation block.
$cases += [PSCustomObject]@{
    Name     = 'guard-inside-conditional-compilation'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
#if defined(CONVOPEQ_TRACE_GUARD)
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#endif
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 37) The trace call lives inside an unresolved conditional-compilation block.
$cases += [PSCustomObject]@{
    Name     = 'hbtrace-inside-conditional-compilation'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#if defined(CONVOPEQ_TRACE)
    emitHBTrace();
#endif
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 38) The actual trace emission lives inside an unresolved conditional-compilation block.
$cases += [PSCustomObject]@{
    Name     = 'emittrace-inside-conditional-compilation'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#if defined(CONVOPEQ_TRACE)
    hbTraceRuntime_.emitTrace(path);
#endif
}
'@ + $commonTail
}

# 39) The violation report call lives inside an unresolved conditional-compilation block.
$cases += [PSCustomObject]@{
    Name     = 'violation-report-inside-conditional-compilation'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}

void DebugRuntime::emitFinalViolationReport()
{
#if defined(CONVOPEQ_VIOLATION_REPORT)
    const auto path = std::filesystem::current_path() / "evidence" / "hb_violation_report.json";
    hbVerifier_.setTraceRuntime(&hbTraceRuntime_);
    hbVerifier_.emitViolationReport(path);
#endif
}
} // namespace convo::isr
'@
}

# 40) A guard disabled by #if 0 must not count as a guard.
$cases += [PSCustomObject]@{
    Name     = 'guard-disabled-by-if0'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
#if 0
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#endif
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 41) A conditional guard next to a valid unconditional guard is still fine.
$cases += [PSCustomObject]@{
    Name     = 'conditional-guard-with-unconditional-guard'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#if defined(CONVOPEQ_EXTRA_GUARD)
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#endif
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 42) Conditional compilation that does not touch the contract items must still pass.
$cases += [PSCustomObject]@{
    Name     = 'conditional-compilation-elsewhere'
    Expected = 0
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

#if defined(CONVOPEQ_TRACE_HELPER)
std::string traceFileName()
{
    return "evidence/hb_graph_trace.json";
}
#endif

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 43) A non-constant #if expression (1 && 0) must be unknown, not an active branch.
$cases += [PSCustomObject]@{
    Name     = 'if-expression-not-constant'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
#if 1 && 0
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#endif
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 44) Preprocessor directives inside a block comment must not create regions.
$cases += [PSCustomObject]@{
    Name     = 'directive-inside-block-comment'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    /* #if 0 */
    hbTraceRuntime_.emitTrace(path);
    /* #endif */
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 45) A #define replacement list must not satisfy the guard checks.
$cases += [PSCustomObject]@{
    Name     = 'guard-macro-unused'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
#define UNUSED_GUARD if (mode == "Release" && ! isForceEvidenceEmission()) { return; }
    const std::string mode = currentBuildMode();
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

# 46) A backslash-continued #define whose continuation is lost by string masking
#     must still have its continuation line blanked.
$cases += [PSCustomObject]@{
    Name     = 'continued-macro-line-survives'
    Expected = 1
    Source   = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}

void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#define UNUSED_TRACE "a\
b" emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@ + $commonTail
}

$failures = 0
foreach ($case in $cases) {
    if (-not ($case.PSObject.Properties.Name -contains 'MutateExporter')) {
        $case | Add-Member -NotePropertyName MutateExporter -NotePropertyValue $false
    }
}
foreach ($case in $cases) {
    $root = New-Fixture -Name $case.Name -DebugRuntimeSource $case.Source
    if ($case.MutateExporter) {
        $path = Join-Path $root 'src\audioengine\ISREvidenceExporter.cpp'
        $text = Get-Content -Path $path -Raw -Encoding UTF8
        $text = $text.Replace('if (mode == "Release")', 'if (mode == "NeverRelease")')
        Set-Content -Path $path -Value $text -Encoding utf8NoBOM
    }

    $result = Invoke-Verifier -Root $root
    $status = 'PASS'
    if ($result.ExitCode -ne $case.Expected) {
        $status = 'FAIL'
        $failures++
    }

    $expectedText = if ($case.Expected -eq 0) { 'accept' } else { 'reject' }
    Write-Host ("[{0}] {1,-40} expected={2,-6} exit={3}" -f $status, $case.Name, $expectedText, $result.ExitCode)
    if ($status -eq 'FAIL') {
        Write-Host ("        output: {0}" -f ($result.Output -replace "`r?`n", ' | '))
    }
}

Remove-Item -Path $tempRoot -Recurse -Force -ErrorAction SilentlyContinue

if ($failures -gt 0) {
    Write-Host ""
    Write-Host "proof-scope self-test: $failures case(s) failed"
    exit 1
}

Write-Host ""
Write-Host "proof-scope self-test: all $($cases.Count) cases passed"
exit 0
