# D166 — DS-F2 Device-Reconfigure Admission / Lifecycle Contract Audit

```text
Task:   D166 — DS-F2 Device-Reconfigure Admission / Lifecycle Contract Audit
Date:   2026-09-06
Type:   read-only architectural / source audit（production 変更 0 / test 0 / CMake 0 / build.bat 0 / tool 0 / binary rebuild 0）
目的:   D165 で発見した DS-F2（device reconfigure 後 rebuild-intake 完全停止）を source state machine と
        照合し、root cause を確定し、修復契約を選定する（実装はしない）。
Verdict: **GO — Case A (Defect confirmed)・修復契約 = Option 2（target）/ Option 1（最小增量）を選定**
先行監査: D162-2-I1-D-R0（2026-09-05）が本領域を先行監査済み。D166 は R0 §9-3 が「別監査（G-series 相当）」
        として明示 defer した「publication 復活・admission reopen 設計」の判断そのものである。
```

---

## 1. Scope / baseline

| 項目 | 実測値 |
| --- | --- |
| git HEAD | `9cacee1f4c81f47bf55eb6396627270cd3cb7618` |
| ConvoPeq.md | `Generated: 2026-09-05 12:02:18`（**実測値**。指示記載の 03:04:39 とは食い違い — 実ファイルの値を freeze として採用。なお H5 編集 09-06 00:00 より古いため source authority には使用せず・全て直接 source 読取で検証） |
| binary | build-diag RWDI `sha256 72cce20a…3ad2`（D165 と同一・DIAG=ON） |
| CMake cache | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS:BOOL=ON` |
| S3 / V-D | source 現行確認済（`RuntimePublicationOrchestrator.cpp:632` shutdown-clear / `D162-2G2_VD_RETIRE` in ReleaseResources） |
| runtime 根拠 | D165 実測（I2-DS: REQUESTED 61 / DISPATCHED 0 / gen 5 停滞 / TV=7・I2-R2/4/6 同型・I2-WA 120/120） |

---

## 2. ShutdownRuntime FSM（A1）

### 2.1 AdmissionState 定義（ISRShutdown.h:160-177）

```text
Open = 0, Closing, Closed, Faulted
遷移契約（h:166）: Open→Closing→Closed（✅）
                   Closed→Open（❌ 禁止 — resurrection 防止 INV-LIFE-9）
                   任意→Faulted（underflow/overflow）
```

### 2.2 API surface と全 caller（実測 grep）

| API | 定義 | caller（全 source 検索） |
| --- | --- | --- |
| `closeAdmission()` | ISRShutdown.cpp:429-451 | **唯一**: ReleaseResources.cpp:87（Open→Closing） |
| `joinProducers()` | ISRShutdown.cpp:465-485 | **唯一**: ReleaseResources.cpp:214（Closing→Closed、count==0 が必要・retry loop） |
| `tryAdmit(n)` | ISRShutdown.cpp:501-525 | 4 箇所: RebuildDispatch.cpp:319（**Build 経路**）/ RuntimePublicationOrchestrator.cpp:70（Publication）/ AudioEngine.h:4515（Recovery）/ AudioEngine.h:4613（Publication fire-and-forget） |
| `release(n)` | ISRShutdown.cpp:527-543 | tryAdmit 成功時の対（guard/手動 release） |
| `isAdmissionOpen` / `admissionState` | ISRShutdown.cpp:489-497 | 診断用 read |
| `initiateShutdown()` | ISRShutdown.cpp:50-56 | **caller 0 件（dead API）** |

### 2.3 packedState_ の全書込（Closed→Open 不在の証明）

`packedState_`（ISRShutdown.h:373、`{0}` = Open で ctor 初期化のみ）への書込は **CAS 4 箇所のみ**:

| site | 遷移 |
| --- | --- |
| closeAdmission :447 | Open→Closing |
| joinProducers :480 | Closing→Closed（不可逆・count==0 条件） |
| tryAdmit :517 | Open のまま count++（state 変更なし） |
| release :535 | count-- / underflow 時 Faulted |

**`AdmissionState::Open` への遷移を書くコードは source 全域に存在しない**（grep 実測・Open への言及は
比較/初期化/コメントのみ）。INV-LIFE-9 は ISRShutdown.h:166/:299・ISRLifetimeProof.h:23 で文書化され、
G-H linearization（tryAdmit と closeAdmission の同一 word CAS・AudioEngine.h:4602-4605）と
Q7 NoResurrection proof（D101-30 locked contract）の基盤。**Case A の後半要件（reopen 経路不在）を成立。**

### 2.4 phase_ も同様に一方通行

`phase_{Running}`（h:335）は ctor 初期化のみ。`transitionTo` は forward + terminal-skip のみ許可
（ISRShutdown.cpp:124-152）で backward は `transitionViolations_` を加算して拒否。
`initiateShutdown()` は dead API のため、shutdown の実入口は releaseResources(:74 の transitionTo)。

---

## 3. releaseResources caller matrix（A2）

`AudioEngine::releaseResources()` は `override`（AudioEngine.h:1142）= JUCE callback。全 caller:

| caller | 実装箇所 | semantic |
| --- | --- | --- |
| **JUCE AudioProcessor lifecycle** | `AudioEngineProcessor::releaseResources`（AudioEngineProcessor.cpp:55-67）→ `audioEngine.releaseResources()`（isEnginePrepared() guard 付き — JUCE が device 列挙時に複数回呼ぶための duplicate 抑制） | **reconfigure（device switch / sample-rate 変更 / buffer-size 変更 / device 再列挙）と terminal shutdown の両方が同一入口** |
| Test harness | AudioEngineHarness.cpp:46 `stop()` → `engine_->releaseResources()` | terminal teardown（明示） |
| ~AudioEngine | **呼ばない**（CtorDtor.cpp:219-221 — dtor は releaseResources 未実行の異常系に備え stopRebuildThread を自前実行） | terminal teardown（独立経路） |
| MainWindow / CLI | 直接呼出なし | — |

**確定事項**: `releaseResources()` は callee 側からは terminal shutdown と reconfigure を区別できない。
JUCE 契約上、releaseResources の後に prepareToPlay が来るか来ないかは host の判断で、entry 時点では未知。
ところが ReleaseResources.cpp:87 の `closeAdmission()` は **全ての pass を terminal として扱う**。
→ 「reconfigure でも terminal 扱い」という設計の食い違いが A2 の結論。

### engine 側 lifecycleState との非対称

`EngineLifecycleState`（Unprepared/Preparing/Prepared/Releasing/Destroyed・AudioEngine.h:2646-2653）は
release→prepare で正しく循環する（CAS ガード付き）。prepareToPlay は re-prepare を正常系として処理する
（PrepareToPlay.cpp:68-69 `re-prepare requested without release; proceeding with safe reinitialization`）。
**循環を意図した state（lifecycleState・rebuild thread・generation）と、一方通行の state
（ShutdownRuntime admission/phase）が同一の reconfigure pass を受けて食い違う** — これが不可視状態
（phase 計算機は Running・admission は Closed 永久）の発生機構。

---

## 4. prepareToPlay restart contract（A3）

prepareToPlay が再初期化するもの（実測・PrepareToPlay.cpp）:

| 対象 | 箇所 |
| --- | --- |
| lifecycleState → Preparing（CAS）→ Prepared | :60-69 |
| shutdownPhase（**表示用別 FSM**・大文字 enum）→ Running | :72 |
| rebuild thread 再起動（非 joinable 時） | :76-84 |
| `rebuildRequestGeneration → 0`（staleness rejection 防止） | :98 |
| 出版停滞監視 reset（`resetProgressObservation`） | :101-103 |
| hasPendingTask / publishRetryReady / pendingTask clear | :79-81 |
| IR/DSP/EQ 等 DSP 系 prepare | 以降全般 |

prepareToPlay が**再初期化しない**もの:

| 対象 | 根拠 |
| --- | --- |
| **ShutdownRuntime packedState_（admission）** | `shutdownRuntime_` への参照が PrepareToPlay.cpp に 0 件（grep 実測） |
| **ShutdownRuntime phase_** | 同上・reset API も存在しない（§2.4） |
| runtimePublicationBridge_ の shutdown 要求状態 | pass 1 で `requestShutdown()` 済みのまま |

**session semantics 判定**: prepareToPlay は JUCE 契約上「新しい operational session の開始」であり、
実装もそれに沿って rebuild thread・generation・停滞監視を再初期化している。にもかかわらず
ShutdownRuntime だけが再初期化対象から漏れている — 単なる DSP preparation と割り切るなら
rebuild thread の再起動自体も不要になるはずで、実装の意図は明らかに session restart である。
**よって admission/phase の再初期化欠落は意図的設計ではなく漏れ**、と判定する。

ただし修正は「prepareToPlay が reopen する」形では避ける（後述 §10 — INV-LIFE-9 設計上の理由）。

### 既知問題としての明示コメント（重要）

PrepareToPlay.cpp:292-294（D162-2-I2 の A 案修復箇所の直後）:

```text
※ reconfigure 後も admission Closed のまま publication が復活しない
  （Active=0・bypass 継続）問題は I2 の scope 外（D162-2-I1-D-R0 §5/§7-C/D）。
```

**DS-F2 は本監査で初めて判明したものではなく、D162-2-I1-D-R0（2026-09-05）で source+D-profile 実測と
ともに確定し、A 案（CallerDestroy 履行）のみ I2 で修復、publication 復活/ reopen 設計は
「別監査（G-series 相当）」として明示 defer された既知問題である。** D166 は R0 §9-3 の defer 先監査。
R0 §(5) の構造的背景判定（admission lifetime = engine lifetime と JUCE re-prepare 契約の衝突・
CoordinatorState 側の ShuttingDown→Bootstrapping 復帰との非対称）は本監査の全 grep と整合する。

---

## 5. rebuild admission path（A4 — telemetry loss accounting）

### requestRebuildIntent の telemetry sequence（RebuildDispatch.cpp）

```text
:232  emitRebuildTelemetry(Requested, decision=Accepted)   ← gate 前に必ず記録
:241  isShutdownInProgress() → Suppressed(ShutdownInProgress)   … telemetry あり
:256  pressure → Suppressed(RetirePressureSevere)               … telemetry あり
:271  sameAsPending → Merged                                    … telemetry あり
:287  isShutdownInProgress()（再）→ Suppressed                  … telemetry あり
:301  kind filter → Suppressed(KindFiltered)                    … telemetry あり
:319  if (!shutdownRuntime_.tryAdmit(1)) return;               ← ★ 無出力 return（telemetry 皆無）
```

### 4 admission 経路の failure 表現の比較

| 経路 | tryAdmit site | failure 時の表現 |
| --- | --- | --- |
| **Build（rebuild intent）** | RebuildDispatch.cpp:319 | **完全に無出力**（event なし・counter なし・log なし） |
| Publication（Orchestrator 経由） | RuntimePublicationOrchestrator.cpp:70 | `PublicationAdmission::Decision::RejectedShutdown`（PublicationAdmission.cpp:12・:435 で下流 telemetry/disposition あり） |
| Publication（fire-and-forget） | AudioEngine.h:4613 | `{Failed, CallerDestroy}` を caller に返す（戻り値は可視・log なし） |
| Recovery | AudioEngine.h:4515 | 無出力 return（ただし obligation は durable state に残るため loss ではなく延期） |

**D165 実測（REQUESTED=61 / DISPATCHED=0 / 中間 event 0 件）は :319 の無出力 return と完全一致。**

### 会計上の判定

「Accepted と記録された request が直後の admission failure で無出力消滅する」は:

1. **observability / semantic accounting defect として独立成立**（lifecycle defect とは別個）。
   R0 §(3) が同型の問題（`ignoreUnused` 握り潰し）を別途指摘しており系統は同じ。
2. 修復は小さな 1 event 追加（`Suppressed(AdmissionClosed)` 相当）で可能 — D167 の scope に含めることを推奨。
   ただし lifecycle 修復（§10）を先行させる場合、reconfigure 経路の drop 自体が消えるため、
   telemetry 追加は terminal shutdown 経路の正常 drop 可視化としての価値が残る。

---

## 6. CLI device-switch path（A5 state graph・実測 log との突合）

D165 I2-DS 実測ログ（evidence/D165_DS.log）と source の対応:

```text
             ┌──────────────┐
             │   Running    │（admission Open・phase Running）
             └──────┬───────┘
                    │ --cli-device-type DirectSound
                    │   MainWindow.cpp:457 setCurrentAudioDeviceType
                    ▼
             旧 device close（JUCE）
                    ▼
             releaseResources pass 1（log :9「enter」・SHUTDOWN_BEGIN 前）
                    ▼
             closeAdmission（:87）→ admission Closed（不可逆）
             transitionTo ×8 → phase = ShutdownComplete(10)
             joinProducers → Closed、shutdown trace JSON も pass 1 で上書き emit
                    ▼
             prepareToPlay(DirectSound)
                    ├─ lifecycleState Preparing→Prepared（循環する）
                    ├─ rebuild thread restart（:76-84）
                    ├─ rebuildRequestGeneration reset（:98）
                    └─ shutdownRuntime_ には触れない（admission Closed のまま）
                    ▼
             burst intent ×60 → REQUESTED(accepted)@:232
                    ▼
             tryAdmit(1)@:319 = false（Closed）→ 無出力 return
                    ▼
             generation 5 停滞・DISPATCHED 0・XRUN 0・CONV_STATUS 0 行
                    │
             （420s 静止 → shutdown）
                    ▼
             releaseResources pass 2（log :14701）
             transitionTo ×8 → 7 件 backward reject（TV=7）＋ ShutdownComplete(t==c) 1 件許可
             shutdown closure は完全（既存 2 obj を全 destroy・remaining=0）
```

pass 2 の 8 transitionTo 呼出のうち 7 件（AudioStopped..VerifyDrained）が reject・
ShutdownComplete は t==c で許可 = **transitionViolations=7 の機械的整合**（D164 DS-F1 の確定）。

---

## 7. GUI device-switch path（B — source-only・実機試験なし）

GUI は CLI と**同一の state machine 経路**を通る（分岐なし・CLI 固有の迂回なし）:

| GUI trigger | 実装箇所 | 到達経路 |
| --- | --- | --- |
| Settings ウィンドウでの device 変更 | DeviceSettings（MainWindow.cpp:1427 で生成・settings window content） | `AudioDeviceManager` → device close/open → `AudioEngineProcessor::releaseResources/prepareToPlay` |
| `setAudioDeviceSetup` ×3 | DeviceSettings.cpp:100 / :1098 / :1276 | 同上 |
| `setCurrentAudioDeviceType` | DeviceSettings.cpp:1094 | 同上 |
| **起動時の保存設定復元** | `DeviceSettings::loadSettings`（MainWindow.cpp:1401） | **保存 device setup が初期 default と異なる場合、起動直後に device switch が発生 → session 開始時点で admission Closed** |
| channel auto-recovery | DeviceSettings.cpp:100 | setup 変更時に発火し得る |

**判定: DS-F2 は CLI 固有ではなく、GUI の通常 device switching に構造的に適用される。**
さらに GUI では起動時設定復元（loadSettings）が switch を起こし得るため、
「保存設定 ≠ 既定 device」の環境では **session 開始直後から rebuild intake が停止し得る**
（実機での発生確認は本監査 scope 外 — 次段の targeted validation で確認）。

---

## 8. DS-F2 root-cause verdict

### 判定: **Case A — Defect confirmed（source-level・2 系統の独立証拠で裏付け）**

```text
証明 1（reopen 経路不在）: packedState_ 書込は CAS 4 箇所のみ（§2.3）。
         Closed→Open 遷移コード 0 件。reopen API 0 件。INV-LIFE-9 で明示禁止。
証明 2（terminal 扱いの無条件適用）: closeAdmission の唯一 caller が releaseResources:87 で、
         caller matrix（§3）のとおり releaseResources は reconfigure でも呼ばれる。
証明 3（runtime 実測との一致）: D165 I2-DS（61/0・gen 停滞）・I2-R2/4/6（4/0）が
         §6 の state graph と機械的に一致。WA（switch なし）は 120/120 で正常。
```

**構造的背景（R0 §5 の確認と拡張）**: admission FSM は「engine 1 生涯 1 回限り」の設計
（INV-LIFE-9・G-H linearization・Q7 NoResurrection proof の基盤）に対し、JUCE AudioProcessor 契約は
release/prepare の反復を要求する。intent-loop 側の CoordinatorState には ShuttingDown→Bootstrapping
復帰が存在する（ISRRuntimePublicationCoordinator.cpp:563-574 markShutdownComplete）一方、
admission/phase に復帰概念がない — **責務分離の欠落**が本質。

**Case B（interpretation incorrect）は棄却**: 再 Open 経路は存在せず、rebuild 停止の原因も
tryAdmit Closed 一点に収束（他 gate は全て telemetry を出すため 61/0・中間 0 件の観測と矛盾しない）。

---

## 9. Telemetry accounting verdict

- **独立した observability defect あり**: Build 経路の tryAdmit 失敗のみ telemetry 皆無
  （§5）。Accepted(61) → 無出力消滅は会計として成立しない。
- 修復は `Suppressed(AdmissionClosed)` 相当の 1 event 追加（RebuildDispatch.cpp:319-320）。
  ただし §10 の lifecycle 修復（Option 1/2）を先行させると reconfigure 経路の drop は原理的に
  消滅するため、追加 telemetry の主価値は「terminal shutdown 経路で受注後 close に間に合わなかった
  intent の可視化」になる（正常系として偶発し得るため有用）。

---

## 10. Candidate repair contracts（C — 3 案比較・選定）

### 前提制約

- INV-LIFE-9（Closed→Open 禁止）は D101-30 locked contract・G-H race closure・Q7 proof の基盤であり、
  反転は R0 §7-C が「ISRLifetimeProof 層の再設計級」と既に評価済み。
- D162-2-I2 の A 案（PrepareToPlay.cpp:296-301 `destroyRolledBackDSP`）は CallerDestroy 契約の履行として
  **維持**（修復後も true shutdown 経路で tryAdmit 失敗は正当に発生し得るため defense-in-depth として有効）。

| 案 | 要旨 | INV-LIFE-9 | 変更面積 | 評価 |
| --- | --- | --- | --- | --- |
| **Option 1** | releaseResources を terminal / reconfigure で区別し、**reconfigure pass では closeAdmission（と terminal pipeline）を実行しない** | **完全保全**（Closed→Open を導入しない） | 中（releaseResources 内部の分岐 + terminal 判定信号） | 最小リスク。terminal 判定信号の設計が鍵 |
| **Option 2** | ShutdownRuntime = process/terminal 専用、DeviceRuntime（新 FSM）= prepare/release/reconfigure 責務 | **完全保全** | 大（新 FSM + 接線） | 責務分離として最も正しい。Practical Stable ISR Bridge の「Shutdown は terminal pipeline・Runtime 責務分離」原則に整合 |
| **Option 3** | Closed→Reinitializing→Prepared→Open の中間状態付き restart 遷移 | ** spirit 反転**（Closed からの復活を中間状態で許可） | 大＋proof 層再設計 | 不採用。Q7/G-H/ReclaimPermit identity（AC-5）の前提（shutdown transaction 1 回限り・generation 束縛）を崩す。R0 §7-C と同結論 |

### 採用: **Option 2 を目標架构とし、Option 1 をその最小增量（first increment）として実装する**

選定理由:

1. Option 1 は INV-LIFE-9・G-H・Q7 を一切変更せずに DS-F2/DS-F1 を解消する。reconfigure pass では
   admission が Open のまま維持されるため「Closed→Open」を新設する必要が原理的にない。
2. Option 2 は Option 1 の到達点を FSM として明文化するもの — 段階導入により契約承認と実装リスクを分割できる。
3. Option 3 は proof 前提を崩すため不採用（R0 §7-C の評価を本監査も維持）。

### Option 1 最小增量の設計骨子（D167 への入力・実装はまだしない）

```text
(1) terminal 判定信号: engine に「terminal release」を明示する入口を追加
    （processor/harness が app shutdown flag を渡す、または engine が app quit 状態を参照）。
    JUCE 契約上 releaseResources 単独では terminal 性は未知のため、信号は必須。
(2) reconfigure pass: closeAdmission / transitionTo(terminal 系) / joinProducers / drain /
    shutdown trace emit を skip し、buffer・resource 解放のみ実行（admission は Open 維持）。
(3) terminal pass: 現行動作を全て維持（現行が正しいのはこちらのみ）。
(4) prepareToPlay: 現行のまま（rebuild thread restart・generation reset は reconfigure pass が
    teardown しなくなるため縮小する可能性があるが、D167 で安全側から段階判断）。
```

### D167 へ持ち越すリスク登録（本監査で確認した open 項目）

| リスク | 内容 |
| --- | --- |
| reader registration | 現行 reconfigure pass は reader registration を閉じる（X3 §6.3）。Option 1 では閉じない → 意味論変化を D167 で監査 |
| world/publication 継続 | 現行は reconfigure 後 `[WORLD] Active=0・bypass 継続`（R0 §5 実測・PrepareToPlay:292 コメント）。Option 1 では publication が device switch を跨いで継続する = 新挙動 → targeted test 必須 |
| CoordinatorState | Bootstrapping 復帰 precedent（:563-574）あり。reconfigure で shutdown を通らなくなるため markShutdownComplete の発火条件変化を確認 |
| shutdown trace / FAULT diag | reconfigure pass が terminal trace を出力しなくなる → trace は terminal-only になり清浄化（D164/D165 の TV=7 は消滅するはず — regression 確認項目） |
| JUCE 複数 release | isEnginePrepared guard（Processor:59-65）の意味論が変わる可能性（Releasing 中の呼び方） |

---

## 11. Invariants affected

| Invariant | 影響 |
| --- | --- |
| INV-LIFE-9（no-resurrection） | Option 1/2 で**保全**（本監査の選定条件） |
| G-H linearization（tryAdmit/closeAdmission 同一 word CAS） | 保全（closeAdmission の発火 timing が変わるのみ・1 回限り性は維持） |
| Q1 AdmissionClosed / Q7 NoResurrection proof | 保全 |
| shutdown generation 束縛（AC-5 stale permit） | 保全（transaction 1 回限りは terminal のみで進行） |
| D162-2-I2 A 案（CallerDestroy→destroyRolledBackDSP） | 維持（defense-in-depth） |
| D123/D127-E shutdown trace・15-P-5 FAULT diag | reconfigure pass からの出力消失（清浄化）— regression 項目 |
| DS-F1（TV=7）・DS-F2 | 修復により消滅するはず（D168 で確認） |

---

## 12. Recommended implementation task

```text
D167 = DS-F2 Repair Implementation（契約承認後）
  scope : Option 1 最小增量（terminal 判定信号 + reconfigure pass の terminal pipeline skip）
  非目的: Option 2 の正式 FSM 導入（別 track / 後続）・telemetry event 追加（同時可・小）・
          GUI 実機試験（D168 の targeted validation に含める）
  順序  : D167 minimal implementation → unit/state-machine tests → Debug/Release CTest
          → targeted device-switch validation（CLI + GUI 相当・rebuild dispatch 復活確認・
            TV=0 確認・D165 DS log との差分照合） → D168 regression soak
  検証 条件: reconfigure 後に REBUILD_DISPATCHED > 0 が復活すること（I2-DS 型 workload で）
          + shutdown closure が完全維持（remaining=0・TV=0）+ D164/D165 全 PASS 項目の非退行
```

### GO / NO-GO 判定基準の照合

| NO-GO 条件 | 状態 |
| --- | --- |
| releaseResources の caller semantics 未確定 | **確定**（§3 matrix） |
| prepareToPlay の session semantics 未確定 | **確定**（§4 — operational session restart・admission 漏れは意図的不在） |
| Closed→operational の別経路が存在 | **不存在を証明**（§2.3） |
| CLI と GUI で state machine が異なる | **同一**（§7 — 単一 AudioEngineProcessor 委譲） |
| telemetry の Accepted/Dispatched semantics 未確定 | **確定**（§5） |

→ **GO**（Case A 確定・session boundary 特定・GUI 分類完了・修復契約選定・source 変更 0）。
次は Repair Contract approval → D167 minimal implementation。
