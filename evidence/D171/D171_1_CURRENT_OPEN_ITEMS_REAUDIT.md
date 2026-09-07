# D171-1 — Current Open Items Re-audit（evidence）

- 日付: 2026-09-07
- Type: read-only audit（source 変更 0 / テスト実行 0）
- Authority stamp: `ConvoPeq.md` — `Generated: 2026-09-07 21:38:42`（108,515 行 / 4,905,541 bytes）
- 照合対象: `doc/work88/PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md`（149 行）
- 前提: D169-2 CLOSED（D169-2-7 full regression 40/40 ×3 config PASS）
- 関連: D163（CR-α/CR-β REJECT 判定 2026-09-06）、D159（freeze register D1〜D6）、D169-2-5（MEM_SNAP hazard 初記録）

---

## 0. 目的

9/1 inventory（1-C 未実装 2 件）を最新 source と突き合わせ、**2026-09-07 時点で本当に OPEN な設計項目だけを再抽出**する。CW-8 のように inventory が STALE 化した項目を全候補について排除し、freeze register（D159 D1〜D6）の現行妥当性を再分類する。付随して D169-2-5 で記録された MEM_SNAP dangling-pointer hazard を独立 defect として source-level audit する。

---

## 1. 実測コマンド系譜

```bash
# P1 — inventory 構造抽出
ctx_execute_file(PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md)  # headers + status markers
  → 総合判定 / 1-A(実装済み15) / 1-B(STALE6) / 1-C(未実装2) / 1-D(DEFER) / 2-B(Phase-II) / 3(結論)

# P2 — source 照合（WSL rg、ConvoPeq.md 21:38:42 生成版に対する source-wide grep）
rg -n "PublishedWorldObservation" ConvoPeq.md src/ tests/
rg -n "buildErrorCount_|RetryBackoffPolicy|backoff" src/
rg -n "kMaxRecoveryConsecutiveFailures|RetryScheduler|kMaxWarmupConsecutiveRetries" src/
rg -n "isSemanticSuperset|completedOutOfOrder|pendingRecoveryAdmission_" src/
rg -n "kDefaultWarmupRetryBackoff|retryBackoffDelayMs" src/audioengine/
git show 0aeb22ca --stat          # CR-α / CR-β 実装 commit 特定
rg -n "CR-α|CR-β|buildErrorCount|residual" doc/work88/D163_OPEN_CR_BOUNDARY_PRIORITY_AUDIT_REPORT.md

# P3 — freeze register 照合
rg -n "D1|D2|D3|D4|D5|D13" doc/work88/D159_PROJECT_CLOSURE_RECORD_REPORT.md
rg -n "INV-X2-6" src/audioengine/          # D5 anchor
rg -n "kMaxDeferredRetries" src/audioengine/RuntimePublicationOrchestrator.h   # D3 anchor
rg -n "P2-G2-W1" src/ doc/work88/D159_PROJECT_CLOSURE_RECORD_REPORT.md         # D6 anchor

# P4 — MEM_SNAP hazard
rg -n "MEM_SNAP|getActiveRuntimeDSP|collectTrackedMemoryStatistics" src/
rg -n "activeRuntimeDSPSlot" src/          # 全 writer/reader 実測
rg -n "setActiveRuntimeDSP\(|releaseActiveRuntimeDSP\(" src/
rg -n "destroyDSPCoreNode|destroyRolledBackDSP" src/
rg -n "drainDeferredRetireQueues" src/audioengine/*.cpp
sed -n '260,330p' src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp
sed -n '145,235p' src/audioengine/AudioEngine.Processing.ReleaseResources.cpp
sed -n '1015,1115p' src/audioengine/AudioEngine.Timer.cpp
sed -n '330,370p' src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp
sed -n '2240,2285p' src/audioengine/AudioEngine.h
rg -n "CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS" CMakeLists.txt build.bat
# 交差検証
serena search_for_pattern(getActiveRuntimeDSP\(\);)     # reader set 裏付け（6 file）
semble search "MEM_SNAP sampler dangling pointer timer callback"   # 位置裏付け
cppcheck --enable=warning AudioEngine.Timer.cpp          # MEM_SNAP 領域指摘 0 件（rc=0）
ccc search（タイムアウト — 欠落、serena/semble で代替）
```

---

## 2. P2 照合結果 — inventory 候補ごとの source 実測

### 2-1. CW-8 PublishedWorldObservation — **STALE（実装済み）**

| 種別 | アンカー | 実測 |
|---|---|---|
| 型 | `RuntimeWorldAuthority.h`（ConvoPeq.md L69163-69191 相当） | `class PublishedWorldObservation` + private ctor + nothrow copy |
| factory | 同 L69330-69338 相当 | `observePublishedObservation(const ReadToken&)` — 単一 acquire load から `{world, &world->publication}` 同時確定 |
| テスト | `ISRSemanticValidationTests.cpp` 11 hits | `testCW8_PublishedWorldObservation` + T-CW8-1/2/3/4/6/7 static_asserts（trivially_copyable / non-aggregate / non-default-constructible 等） |
| harness 登録 | ConvoPeq.md L94956-94958 | `if (!testCW8_PublishedWorldObservation()) throw ...` |
| 実装 commit | `git show 0aeb22ca` | ND-01..04（CRBETA0 triage 2026-09-01 も ALREADY COVERED） |

判定: **STALE — 新規 CW-8 implementation track は起票しない**。D163 判定（CR-β = REJECT / ALREADY COVERED）と整合。

### 2-2. BuildError retry backoff / count（CR-α 本体）— **STALE（実装済み）**

| 種別 | アンカー | 実測 |
|---|---|---|
| policy 型 | `BuildErrorPolicy.h:97` | `struct RetryBackoffPolicy` |
| 既定値 | `BuildErrorPolicy.h:104` | `kDefaultWarmupRetryBackoff {10, 80, 2}`（初期 10ms・飽和 80ms・係数 2） |
| 純関数 | `BuildErrorPolicy.h:109-158` | `retryBackoffDelayMs()` / `warmupRetryDecision()` |
| production 接続 | `AudioEngine.RebuildDispatch.cpp:1294` | `schedule(req, decision.delayMs)` — 唯一の call site に policy 接続済み |
| retry budget | 同 :1189-1196 | generation ごとに独立（`warmupRetryBoundGeneration` rebind） |
| テスト | `BuildErrorClassificationTests.cpp:225-260` | T-CRα-1（delay table 0/10/20/40/80/80 飽和）〜 T-CRα-4（disposition→delay mapping） |
| bounded 上限 | `RebuildDispatch.cpp`（CR-α comment「max=3」） | `kMaxWarmupConsecutiveRetries` / Recovery 系は K=4 別ドメイン |
| 実装 commit | CR-α-1..6（2026-09-01） | CRALPHA1..6 報告書一式・CTest 40/40×2・closure 15/15 |

判定: **STALE — inventory 1-C-1 の「delay=0 = backoff 未接続」記述は後続実装で陳腐化**。D163 判定（CR-α = REJECT / Case D）と整合。delay=0 は RetryImmediate（WarmupFailed）の intentional bounded immediate retry。

### 2-3. BuildError telemetry（`buildErrorCount_`）— **UNIMPLEMENTED かつ DEFER**

- `RuntimeBuilder.h:124` コメント（「buildErrorCount_ telemetry …で対応する」）のみ・src 実装 0 hits。
- D163 報告書 §39: 「buildErrorCount_ 集約 counter が唯一の residual（observability-only・trigger 条件付き DEFER — 既決監査 + CRBETA0 B-7 に二重記録済み）」。
- 判定: **DEFER 維持（observability-only・trigger 発生時に再開）**。D171-2 preflight 対象外。

### 2-4. Site 2（RuntimeBuilder build failure）の retry 適用 — **設計通り未適用**

- `RebuildDispatch.cpp:1209-1213`: classify → diagLog のみ。「retry 方針の実適用（backoff 等）は D-5 RetryBackoffPolicy tuning と併せて将来拡張 — 1.8.9 実装手順」コメント実在。
- dash2 §1.8 Phase D の現行仕様（分類をログに記録）どおり。**将来拡張として明示記録済み・非 blocking** → DEFER（設計通り）。

### 2-5. Freeze register（D1〜D6 + 補助 trigger）再実測

| 登録 | 内容 | 現行 source 実測 | trigger 状態 |
|---|---|---|---|
| D1 | R1 MPSC（`pendingRecoveryAdmission_` MPSC 化含む） | `pendingRecoveryAdmission_` は D146 CAS プロトコル現役（`ISRRuntimePublicationCoordinator.h:1111`）。Timer/Processor からの recovery API 呼び出し = 第 2 producer 0 件 | 未発生 |
| D2 | Phase-II Supersession（`isSemanticSuperset` / RC-11 pending supersession full） | `isSemanticSuperset` src 0 hits。equality-only containment（G-4.3-T 回帰）維持 | 未発生 |
| D3 | D102-C3C4 gates | `kMaxDeferredRetries = 2` dormant guard 現役（`RuntimePublicationOrchestrator.h:322` — F6: retention に適用しない） | 未発生 |
| D4 | §1.5 sparse completion | `completedOutOfOrder` src 0 hits（`AudioEngine.h:3768-3773` / `SequenceArithmetic.h:53-55` コメントのみ・gateway+FIFO 維持） | 未発生 |
| D5 | §1.6 X2 wraparound / out-of-order テスト | INV-X2-6 anchor 維持（`SequenceArithmetic.h:55`・`AudioEngine.h:3763`） | 未発生 |
| D6 | P2-G2-W1 static bound | src anchor 0 hits・D159 register 文書登録のみ | 未発生 |
| 補助 | RecoveryEpisodeId（D2 従属）/ pendingRecoveryAdmission_ MPSC 化（R1 同時判断）/ stale コメント清掃（次編集 window） | 前提変化なし | 未発生 |

判定: **freeze register 全 6 件 + 補助 trigger とも妥当・DEFER 維持**。

### 2-6. 1-A 実装済み / 1-B STALE の継続確認

inventory 1-A（15 項目）・1-B（6 項目）は今回の grep（§1.8.10.3 descriptor table・`ISRLifetimeProof.h`・`AdmissionState`・coalesce・SemanticRecoveryTarget 等）でいずれも現行 source と整合。D169/D170 系修復（retire handle 一本路・collapse no-op）により 1-A/1-B 分類が崩れた項目は**なかった**。

---

## 3. P4 — MEM_SNAP dangling-pointer hazard 独立 audit（修復なし・判定のみ）

### 3-1. slot 構造と契約

- `AudioEngine.h:2228`: `std::atomic<DSPCore*> activeRuntimeDSPSlot{nullptr}`。
- 契約（h:2265 comment・ReleaseResources.cpp:151-161 D169-1R RC-D169-1-2 comment）: **placeholder 専用の非所有 topology mirror。rebuild publish は pointer slot を更新しない。ownership authority に昇格させない。capture 値は観測専用**。

### 3-2. writer 全リスト（4 箇所のみ・網羅実測）

| 箇所 | 操作 | 文脈 |
|---|---|---|
| `CtorDtor.cpp:153` | `setActiveRuntimeDSP(nullptr)` | dtor |
| `PrepareToPlay.cpp:287` | `setActiveRuntimeDSP(placeholderRaw)` | placeholder prepare 成功後・publish 前 |
| `PrepareToPlay.cpp:318` | `setActiveRuntimeDSP(nullptr)` | tryAdmit 失敗 → `destroyRolledBackDSP(placeholderRaw)`（:317）の直後 |
| `ReleaseResources.cpp:180` | `setActiveRuntimeDSP(nullptr)` | rebuildMutex 内・capture 後 |

- `releaseActiveRuntimeDSP()`（h:2275）は**呼び出し元 0 件**（dormant accessor）。
- publication / retire / EBR destroy 経路は slot に触れない（Orchestrator は handle 経由）。

### 3-3. reader 全リスト（serena で 6 file 裏付け）

| reader | 使用 | dereference |
|---|---|---|
| `AudioEngine.Timer.cpp:1078`（MEM_SNAP） | 取得 → `collectTrackedMemoryStatistics()` | **あり** — メンバ一式読み取り |
| `AudioEngine.Processing.Latency.cpp:96` | world 未公開時のみ fallback → `dsp->...` | **あり** — ただし world null + slot dangling の窓は構造的に release 内で閉じる（下記 3-5） |
| `AudioEngine.h:3850` | transition validation の current 取得 | dereference の形による（h:3851 以降で world 側と併用） |
| `DSPLifetimeManager.cpp:145` | `getActive()` を `void*` 返却 | 返却のみ（呼び出し側依存） |
| `ReleaseResources.cpp:170/178・223` | capture → `juce::ignoreUnused`（D169-1R で pointer-value retire 廃止済み） | **なし**（観測専用契約どおり） |

### 3-4. hazard 構造（CONFIRMED as hazard class）

1. `prepareToPlay` で placeholder 生成 → :287 で slot 設定 → :301 `commitRuntimePublication`（`needsRegistration(placeholderRaw)`・handle 登録）が成功（Transferred）→ **slot に pointer が残留**。
2. 以後、placeholder は rebuild 経路で置き換え可能。旧 placeholder は `retireDSPHandleForRuntime` → `pendingReclaimHandles_` → `drainDeferredRetireQueues`（**Timer.cpp:1827/1843 = MessageThread、Threading.cpp:237/251/383**）で epoch 安全確認後に EBR → `destroyDSPCoreNode` 物理破壊。
3. **この一連の destroy は slot を null 化しない**（writer 4 箇所に destroy 経路なし）。slot は dangling pointer を保持し続ける。
4. 同状態で timerCallback（juce::Timer = MessageThread）の MEM_SNAP block が `getActiveRuntimeDSP()` → 非null 判定 → `collectTrackedMemoryStatistics()`（`DSPCoreLifecycle.cpp:337`・`ASSERT_NON_RT_THREAD()` は MessageThread で通過）→ **UAF（destroy 後メンバ読み取り）**。メソッドは `const noexcept` でメンバ静的読み取り中心のため、破壊直後の再利用が無ければ silent garbage 値を出力するだけの可能性もあり、timing 依存。
5. `Latency.cpp:96` fallback: world 公開後（通常動作）は world の current を使うため fallback 非到達。world null + slot dangling の窓は、releaseResources が rebuildMutex 内で slot null 化（:180）→ world idle publish（null）を同一名前空間で完結させるため、外部から観測可能な dangling dereference 窓は作られにくい。**主要 exposure は MEM_SNAP（:1078）**。

### 3-5. 緩和事実（重要度判定）

| 事実 | 実測 |
|---|---|
| diagnostic build 専用 | `CMakeLists.txt:129` `option(CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS ... OFF)` = **default OFF**・MEM_SNAP block は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内（Timer.cpp:1022） |
| 実害観測 | 未観測。D169-2-5 の harness crash は初版テストの契約違反（audio thread 走行中 prepareToPlay + capture logger race）が原因で、MEM_SNAP UAF 自体は crash 原因ではない（crash thread が MEM_SNAP 付近と判読されたため混同注意） |
| 静的解析 | cppcheck（warning level）で MEM_SNAP 領域の指摘 0 件 — 構造的 UAF は AST 静的解析の射程外（別 object への atomic store + 遅延 destroy）であり、実証には動的 audit が必要 |
| D170 修復との関係 | D170 修復は handle 経路（pointer-value retire 廃止）であって、raw pointer slot reader の契約には触れていない。slot 契約「非所有 mirror・観測専用」は D169-1R RC-D169-1-2 comment で明文化済み — **MEM_SNAP の dereference はこの観測専用契約を超える使用** |

### 3-6. P4 判定

- **経路の存在: CONFIRMED**（Timer.cpp:1078-1088・現行 source に現存）
- **UAF 構造: CONFIRMED as hazard class**（destroy 経路が slot を null 化しない + reader が観測専用契約を超えて dereference）
- **実害: 未観測・diagnostic build only・timing 依存 flaky AV potential**
- STOP 条件該当: **lifetime / ownership が未証明**（slot 値 dereference には lifetime 保証が存在しない）→ **修復を実装せず D172-1 lifetime/source audit へ回す**。D172-1 の問いは「reader 4 箇所のうちどれが実 dereference か」「world 経由への切替 / destroy 側 mirror 清掃 / slot 廃止のどの修復が authority 契約を破らないか」。
- 修復候補（実装禁止・D172-1 判断材料のみ）: (a) MEM_SNAP reader を world 経由（`publishedWorld->engine.current`）へ切替、(b) destroy 側での mirror slot CAS 清掃（ただし非所有 mirror 契約の拡張になる）、(c) MEM_SNAP block の間接統計（LiveAllocRegistry 系）への置換。

---

## 4. 最終表 — Current verdict

| Candidate | Inventory 9/1 | Current source（2026-09-07 21:38:42 実測） | Freeze | Current verdict |
|---|---|---|---|---|
| CW-8 PublishedWorldObservation | UNIMPLEMENTED（1-C-2） | **実装済み**（RuntimeWorldAuthority.h 型+factory・ISRSemanticValidationTests T-CW8-1..7・commit 0aeb22ca） | 未登録 | **STALE** |
| BuildError retry/backoff（CR-α 本体） | UNIMPLEMENTED（1-C-1） | **実装済み**（BuildErrorPolicy.h:97-158・RebuildDispatch:1294 接続・T-CRα-1..4・CR-α-1..6 CLOSED） | 未登録 | **STALE** |
| BuildError telemetry（buildErrorCount_） | UNIMPLEMENTED（1-C-1 item 15 系） | **未実装**（RuntimeBuilder.h:124 コメントのみ・src 0 hits） | DEFER（D163 §39・CRBETA0 B-7 trigger 条件付き） | **DEFER（observability-only）** |
| Site 2 build failure retry 適用 | UNIMPLEMENTED（1-C-1 内） | 分類 diagLog のみ（:1212-1213 将来拡張コメント・dash2 §1.8 Phase D 通り） | 未登録 | **DEFER（設計通り・将来拡張明示）** |
| MEM_SNAP dangling hazard | inventory 外 / 新規（D169-2-5 記録） | **経路現存**（Timer.cpp:1078-1088・diagnostic build only） | — | **AUDIT REQUIRED → D172-1** |
| D1 R1 MPSC | DEFER（1-D） | trigger 未発生（第 2 producer 0 件・D146 CAS 現役） | D1 | **DEFER 維持** |
| D2 Supersession | DEFER | `isSemanticSuperset` 0 hits・equality-only 維持 | D2 | **DEFER 維持** |
| D3 D102-C3C4 gates | DEFER | `kMaxDeferredRetries=2` dormant 現役 | D3 | **DEFER 維持** |
| D4 sparse completion | DEFER | `completedOutOfOrder` 0 hits | D4 | **DEFER 維持** |
| D5 X2 wraparound テスト | DEFER | INV-X2-6 anchor 維持 | D5 | **DEFER 維持** |
| D6 P2-G2-W1 static bound | DEFER | src anchor なし・文書登録のみ | D6 | **DEFER 維持（doc-level）** |
| 1-A 実装済み 15 項目 | IMPLEMENTED | 継続整合（D169/D170 修復で分類崩れなし） | — | **CLOSED 維持** |
| 1-B STALE 6 項目 | STALE | 継続整合 | — | **CLOSED 維持** |

---

## 5. 結論

1. **9/1 inventory の 1-C（着手可能 OPEN 候補 2 件）は両方とも STALE 化** → 2026-09-07 時点で実装 track に着手可能な OPEN 設計項目は **0 件**。
2. **唯一の新規調査対象は MEM_SNAP dangling hazard**（AUDIT REQUIRED → D172-1）。
3. BuildError Phase-2（retry/backoff 実装）への着手は不適 — 本体は CR-α で実装済み・residual telemetry は DEFER。
4. D159 freeze register（D1〜D6 + 補助 trigger）は全件妥当・DEFER 維持。
5. inventory 更新提案（doc-only）: 1-C-1 → STALE（CR-α CLOSED 記載へ）、1-C-2 → STALE（CW-8 IMPLEMENTED 記載へ）。本 audit では inventory ファイルを改変しない（read-only）。

## 6. 限界

- CTest / 動的検証は未実施（read-only 契約）。MEM_SNAP hazard の実害は D172-1 で動的実証判断が必要。
- `ccc` semantic search はタイムアウトで欠落 — serena / semble で代替裏付け済み。
- `graphify` / `clang-tidy` / `Dr.Memory` は本 audit の調査設計（identity / lifetime 構造確認）では交差検証手段として採用せず、serena + rg + cppcheck + semble で網羅（MEM_SNAP 領域の静的解析は cppcheck 実施済み・指摘 0）。
