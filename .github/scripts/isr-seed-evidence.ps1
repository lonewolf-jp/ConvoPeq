$ErrorActionPreference = 'Stop'

$root = Join-Path $PSScriptRoot "..\..\evidence"
$root = [System.IO.Path]::GetFullPath($root)
New-Item -Path $root -ItemType Directory -Force | Out-Null

$runId = [guid]::NewGuid().ToString("N")
$generatedAtNs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() * 1000000

Get-ChildItem -Path $root -File -Filter "*.json" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

$artifacts = @{
  "closure_graph.json"         = @{
    schema                      = "closure_graph_v1"
    provenance                  = "seed"
    status                      = "generated"
    descriptorCoverageComplete  = $true
    externalMutableDependencies = 0
    generatedAtNs               = $generatedAtNs
    runId                       = $runId
  }
  "mutation_fault_trace.json"  = @{
    schema        = "mutation_fault_trace_v1"
    provenance    = "seed"
    status        = "generated"
    violations    = 0
    generatedAtNs = $generatedAtNs
    runId         = $runId
  }
  "hb_graph_trace.json"        = @{
    schema        = "hb_trace_v1"
    provenance    = "seed"
    eventCount    = 4
    events        = @(
      @{ ts = ($generatedAtNs + 1); from = 1; to = 2; fromEpoch = 10; toEpoch = 10; mo = 3; release = $true;  acquire = $true  },
      @{ ts = ($generatedAtNs + 2); from = 2; to = 3; fromEpoch = 10; toEpoch = 11; mo = 2; release = $true;  acquire = $false },
      @{ ts = ($generatedAtNs + 3); from = 3; to = 4; fromEpoch = 11; toEpoch = 11; mo = 1; release = $false; acquire = $true  },
      @{ ts = ($generatedAtNs + 4); from = 4; to = 5; fromEpoch = 11; toEpoch = 12; mo = 3; release = $true;  acquire = $true  }
    )
    generatedAtNs = $generatedAtNs
    runId         = $runId
  }
  "hb_violation_report.json"   = @{
    schema          = "hb_violation_report_v1"
    provenance      = "seed"
    status          = "ok"
    violations      = @()
    scenarioResults = @(
      @{ name = "forced_reorder"; result = "pass" },
      @{ name = "epoch_lag"; result = "pass" },
      @{ name = "retire_delay"; result = "pass" },
      @{ name = "observe_race"; result = "pass" }
    )
    generatedAtNs   = $generatedAtNs
    runId           = $runId
  }
  "retire_timeline.json"       = @{
    schema           = "retire_timeline_v2"
    provenance       = "seed"
    epochMode        = "shared"
    rollbackMode     = "shared"
    rollbackReady    = $true
    rollbackFlags    = @{ global = $true; publicationOnly = $false; crossfadeOnly = $false; retirePathOnly = $true }
    totalTransitions = 4
    laneCounters     = @{ rtIntent = 1; coordination = 1; epoch = 1; reclaim = 1; quarantine = 0 }
    lifecycleCounters = @{ visible = 1; compareEligible = 1; telemetryRetained = 1; replayRetainedOptional = 0; reclaimEligible = 1; reclaimed = 1 }
    lifecycleSample  = @(
      @{ slot = 0; state = "reclaimed" },
      @{ slot = 1; state = "visible" }
    )
    generatedAtNs    = $generatedAtNs
    runId            = $runId
  }
  "shadow_compare_cadence.json" = @{
    schema                    = "shadow_compare_cadence_v1"
    provenance                = "seed"
    minCadenceMs              = 1000
    burstWindowMs             = 250
    burstEscalationThreshold  = 3
    totalObservations         = 1
    mismatchCount             = 0
    monotonicViolationCount   = 0
    cadenceViolationCount     = 0
    escalationCount           = 0
    lastSequenceId            = 1
    generatedAtNs    = $generatedAtNs
    runId            = $runId
  }
  "shutdown_trace.json"        = @{
    schema                   = "shutdown_trace_v1"
    provenance               = "seed"
    phase                    = 0
    verified                 = $true
    sh1_callbackCount        = 0
    sh2_activeCrossfade      = 0
    sh3_pendingRetire        = 0
    sh4_observerCount        = 0
    sh5_lateCallbackCount    = 0
    sh6_postStopEnqueueCount = 0
    generatedAtNs            = $generatedAtNs
    runId                    = $runId
  }
  "retire_latency_report.json" = @{
    schema          = "retire_latency_report_v1"
    provenance      = "seed"
    withinThreshold = $true
    generatedAtNs   = $generatedAtNs
    runId           = $runId
  }
  "payload_tier_report.json"   = @{
    schema        = "payload_tier_report_v1"
    provenance    = "seed"
    status        = "generated"
    violations    = 0
    families      = @(
      @{ name = "activeNode"; tier = "InlineImmutable" },
      @{ name = "fadingNode"; tier = "ImmutableShared" },
      @{ name = "transitionNext"; tier = "ImmutableShared" },
      @{ name = "retireSlot"; tier = "MutableAuthority" }
    )
    generatedAtNs = $generatedAtNs
    runId         = $runId
  }
  "runtime_budget_report.json" = @{
    schema             = "runtime_budget_report_v1"
    provenance         = "seed"
    artifactTotalBytes = 1
    generatedAtNs      = $generatedAtNs
    runId              = $runId
  }
}

foreach ($name in $artifacts.Keys) {
  $path = Join-Path $root $name
  $artifacts[$name] | ConvertTo-Json -Depth 8 -Compress | Set-Content -Path $path -Encoding UTF8
}

$manifest = @{
  schema         = 'evidence_manifest_v1'
  runId          = $runId
  generatedAtNs  = $generatedAtNs
  generator      = 'isr-seed-evidence.ps1'
  generationMode = 'seed'
  artifacts      = @($artifacts.Keys | Sort-Object)
}

$manifestPath = Join-Path $root 'evidence_manifest.json'
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "[INFO] Seeded ISR evidence under: $root (runId=$runId)"
