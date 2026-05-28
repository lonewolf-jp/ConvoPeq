# ConvoPeq ISR Bridge Runtime v7.3 CIチェック実装案（grep/lint）

## 0. 目的

本書は `ConvoPeq_ISR_Bridge_Runtime_AI詳細設計・実装統制規約_v7.3_2026-05-28.md` を、
GitHub Actions + PowerShell スクリプトで機械検証可能な形に落とし込むための具体案である。

対象:

- admission bypass 禁止
- execution semantic 拡張禁止
- suppression authority 分散禁止
- shutdown phase bypass 禁止
- residency authority / producer 増殖検知
- telemetry write authority 逸脱検知

---

## 1. 実装方針（fail-closed）

- 原則: **allowlist 方式**（未定義の新経路は fail）
- 判定階層:
  1. 構文レベル grep/lint（高速）
  2. ファイルスコープ・関数スコープの文脈チェック
  3. ポリシーJSON照合
- 例外は `.github/isr-*.json` に明示し、`owner/issue/rationale/expiry` 必須

---

## 2. 追加するチェック（提案）

### 2.1 CI-ADMISSION-001: `requestRebuild(` 直呼び禁止

対応規約:

- Rule-1A, 1B, 1C, 1D
- OPS-1, OPS-5

判定:

- `requestRebuild(` 呼び出しを全走査
- 以下を除外:
  - 関数定義行（definition site）
  - funnel 実装の allowlist 行
- 除外外で1件でも検出 => fail

PowerShell 実装イメージ:

```powershell
$matches = Select-String -Path "src/**/*.cpp","src/**/*.h" -Pattern "\brequestRebuild\s*\(" -CaseSensitive
# 除外: definition site
$callSites = $matches | Where-Object { $_.Line -notmatch "\bvoid\s+AudioEngine::requestRebuild\s*\(" }
# 除外: allowlist
$violations = $callSites | Where-Object { -not (Test-AllowlistedLocation $_.Path $_.LineNumber 'requestRebuildDirectCall') }
if ($violations.Count -gt 0) { throw "CI-ADMISSION-001 failed" }
```

段階適用（移行期間）:

- Phase-A: warn（可視化のみ）
- Phase-B: fail（CI fail + PR block）

allowlist は `owner/issue/rationale/expiry` 必須。expiry 超過は fail-closed。

---

### 2.2 CI-ADMISSION-002: 新 execution semantic 拡張禁止

対応規約:

- Rule-1E, 1F
- CI-3

禁止トークン例:

- `urgent|critical|alwaysRun|highPriority|realtimeCritical|lowPriorityReplaceable`

判定:

- source + docs + policy を横断し、禁止トークン新規導入を fail

---

### 2.3 CI-ADMISSION-003: suppression authority 分散禁止

対応規約:

- Rule-1M

判定:

- `suppress|reject|drop` 系の rebuild 判定ロジックが funnel 以外に追加されていないか検出
- allowlist 以外の関数で suppression 条件分岐が見つかったら fail

ドメイン分離ルール:

- `RebuildIntentSuppression`（Rule-1M 対象）
- `QueueAdmissionSuppression`（対象外）
- `SnapshotDropSuppression`（対象外）

CI は suppression domain タグに基づいて判定し、Queue/Snapshot を Rule-1M 違反として誤検知しない。

推奨検出キーワード:

- `RebuildTelemetryDecision::Suppressed`
- `Replaceable`
- `retireSaturationActive_`
- `return`（suppression分岐直後）

---

### 2.4 CI-SHUTDOWN-001: phase bypass 禁止

対応規約:

- Rule-3B, 3G

禁止検出（allowlist外）:

- unload/release 経路での direct queue clear
- coordinator reset 直呼び
- force-retire without phase gate

キーワード例:

- `drainDeferredRetireQueues\(`
- `clearPublishedRuntimeSnapshotsNonRt\(`
- `m_epochDomain\.drainAll\(`
- `retireDSP\(`

※ これらは正当箇所もあるため、**関数単位 allowlist**必須。

---

### 2.5 CI-RECLAIM-001: bounded reclaim cadence 必須項目検査

対応規約:

- Rule-4A, 4D, 4E

判定:

- reclaim cadence パラメータ（max iterations / min interval / pressure cap / starvation timeout）がコード上に存在
- 参照のみではなく適用箇所があること（定義のみ禁止）

検出例:

- `max.*reclaim.*iter|reclaim.*max.*iter`
- `min.*reclaim.*interval|reclaim.*min.*interval`
- `pressure.*cap|upper.*bound`
- `starvation.*timeout`

---

### 2.6 CI-RESIDENCY-001: 新 residency queue 追加監視

対応規約:

- Rule-2A〜2F
- CI-4

判定:

- 新しい queue/vector/deque/list のメンバ定義を検出
- `residency authority table` に未登録なら fail（または mandatory review）
- `hard upper bound / warn threshold / force-drain trigger` の定量契約未記載なら fail

検出対象例:

- `std::vector<.*>`
- `std::deque<.*>`
- `LockFree.*Queue`
- `RingBuffer`

除外:

- 既知の allowlist メンバ

---

### 2.7 CI-TELEMETRY-001: telemetry write authority 逸脱検知

対応規約:

- Rule-6B, 6E
- CI-5

判定:

- `fetchAddAtomic|publishAtomic|exchangeAtomic` が telemetry counter に対して行われる箇所を抽出
- owner subsystem 以外での書き込みを fail

実装方法:

1. telemetry counter 名を policy JSON で owner 紐付け
2. 書込行のファイルパスと照合

---

## 3. ポリシーファイル（提案）

`.github/isr-ai-governance-policy.json` を追加し、以下を管理:

- `requestRebuildDirectCall` allowlist
- `requestRebuildDirectCallPhase`（warn/fail）
- `suppressionAuthorityFunctions` allowlist
- `suppressionDomainTags`
- `suppressionReasonAllowlist`（Tier0固定）
- `residencyKnownContainers`
- `residencyBoundednessContracts`
- `telemetryCounterOwners`
- `forbiddenExecutionSemantics`

例（抜粋）:

```json
{
  "schema": "isr_ai_governance_policy_v1",
  "owner": "audio-runtime",
  "issue": "ops-hardening-v73",
  "rationale": "enforce single authority and boundedness",
  "expiry": "2027-12-31",
  "forbiddenExecutionSemantics": [
    "urgent",
    "critical",
    "alwaysRun",
    "highPriority",
    "realtimeCritical",
    "lowPriorityReplaceable"
  ]
}
```

---

## 4. 新規スクリプト提案（.github/scripts）

### 4.1 `isr-verify-v73-admission-funnel.ps1`

責務:

- CI-ADMISSION-001/002/003 を統合検査

出口:

- 違反時 non-zero exit
- `evidence/isr_v73_admission_report.json` 出力

### 4.2 `isr-verify-v73-shutdown-reclaim.ps1`

責務:

- CI-SHUTDOWN-001
- CI-RECLAIM-001

### 4.3 `isr-verify-v73-residency-telemetry.ps1`

責務:

- CI-RESIDENCY-001
- CI-TELEMETRY-001

---

## 5. ワークフロー統合案

既存 `isr-verification.yml` の標準 tier に以下を追加:

1. `isr-verify-v73-admission-funnel.ps1`
2. `isr-verify-v73-shutdown-reclaim.ps1`
3. `isr-verify-v73-residency-telemetry.ps1`

例:

```yaml
- name: Run ISR v7.3 admission funnel checks
  shell: pwsh
  run: .github/scripts/isr-verify-v73-admission-funnel.ps1

- name: Run ISR v7.3 shutdown/reclaim checks
  shell: pwsh
  run: .github/scripts/isr-verify-v73-shutdown-reclaim.ps1

- name: Run ISR v7.3 residency/telemetry checks
  shell: pwsh
  run: .github/scripts/isr-verify-v73-residency-telemetry.ps1
```

---

## 6. チェック項目と規約のトレーサビリティ

| Check ID | Rule対応 | Fail条件 |
| --- | --- | --- |
| CI-ADMISSION-001 | 1A/1B/1C/1D | allowlist外 `requestRebuild(` 直呼び |
| CI-ADMISSION-002 | 1E/1F | 禁止semanticトークン新規導入 |
| CI-ADMISSION-003 | 1M | funnel外 suppression ロジック検出 |
| CI-SHUTDOWN-001 | 3B/3G | phase gate 無視の直接解放経路 |
| CI-RECLAIM-001 | 4A/4D/4E | bounded cadence 要件欠落 |
| CI-RESIDENCY-001 | 2A-2F | 未登録 residency queue 追加 |
| CI-TELEMETRY-001 | 6B/6E | owner外 subsystem からの counter 更新 |

---

## 7. 運用ルール（誤検知抑制）

- allowlist は最小化（行単位 + 理由 + 期限）
- 一時例外は `expiry` 付き、期限超過で fail-closed
- policy 変更時は self-test（配線検証）を必須化
- 「検出だけ」は禁止、CI fail 条件まで接続する

---

## 8. 導入順（最小リスク）

1. CI-ADMISSION-001（最優先）
2. CI-ADMISSION-002 / 003
3. CI-SHUTDOWN-001
4. CI-RECLAIM-001
5. CI-RESIDENCY-001
6. CI-TELEMETRY-001

理由:

```text
まず bypass を封鎖しない限り、他の hardening は効かない。
```

---

## 9. 完了条件（work4段階）

- `work4` 文書として、各 Rule に対する grep/lint チェックIDが1つ以上紐づくこと
- allowlist/definition除外の扱いが明文化されていること
- 既存 workflow へ差し込み可能な step が提示されていること
- fail-closed 条件が明記されていること
