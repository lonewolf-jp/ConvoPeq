# P1-5-IR-P2 — Step 5-AU / P3-5-FPM-Run-1: Full-pipeline Measurement Execution Gate

- **作成**: 2026-09-24 / work113 P1-5-IR Phase 2（P3-5-FPM-Run-1）
- **種別**: **measurement execution gate**。コード変更 0・実測のみ。
  FPM-Impl-1（Step 5-AT・FPM-IMPL-1-PASS）の build 済み vehicle を使用し、
  M0 → M1 → M2 の順で実行・証拠固定する。
- **判定**: **RUN-1-MEASURED — M0/M1/M2 全 run INVALID-TERMINAL として固定**
 （Full-pipeline sequence は T3b terminal contract を満たさないことを測定）。
- **入力**: FPM-IMPL-1-PASS（実装＋build 確認まで・M0/M1/M2 未実行）。
- **範囲**: buzz／limiter／NUC attribution なし・音質評価なし・terminal failure 修正なし・
  shutdown／epoch／reclaim／Recovery semantics 修正なし・production source 変更なし。

---

## 1. State Freeze

```text
HEAD = 1e9e63e34bed7adb9342ebc81259ded9689fc48a
  （docs(work113): record P1-1 polyphase gain characterization・2026-09-22）
  ※ FPM-Impl-1 §1 と同一 HEAD。

ConvoPeq.md = 2026-09-23 23:33:38・Length 5535334・
  SHA256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
  ※ FPM-Impl-1 §1 記載（2026-09-23 23:33:38・5535334）と一致。
  最終 src 編集（AudioEngine.h 2026-09-23 22:06:18）より後の aggregate であり、
  実測開始時点の最新 source aggregate として採用。

FPM vehicle = build\Release\AudioEngineHarness.exe
  （2026-09-24 01:00:47・41216512 bytes・FPM-Impl-1 §10 の成果物と同一。
   src 最終編集より新しく、全 working-tree source を含んで build 済み）
  CLI dispatch: --fpm-m0 FOUND／--fpm-m1 FOUND／--fpm-m2 FOUND（binary 内文字列確認）。

production change (本 gate 内) = 0（編集なし・実測のみ）
test/CMake/build 設定 change (本 gate 内) = 0

M0/M1/M2 = 未実行状態から開始 → 本 gate で M0 → M1 → M2 の順に各 1 回実行。
```

working-tree 差分（HEAD 対比・本 gate 開始前に存在・本 gate では触らず）：

```text
src/audioengine: 8 files（AudioEngine.RebuildDispatch.cpp／AudioEngine.h／
  ISRRuntimePublicationCoordinator.cpp／.h／_ProcessIntent.cpp／
  PublicationExecutor.cpp／.h／RuntimePublicationOrchestrator.cpp）
  ※ R27 等の既存差分（FPM-Impl-1 §3・§11 の監査どおり本 gate の変更ではない）。
  binary（01:00:47）はこれら全 source（最終 22:06:18）より後に build されており、
  実測 vehicle は現 working-tree source を反映している。
src/tests: 3 files（BassBuzzMeasurement.cpp／P1PolyphaseGainCharacterization.cpp
  ＝ R30/R32 vehicle 等の累積＋FPM vehicle 約290行／PublishPipelineIntegrationTests.cpp
  ＝ CLI 分岐 +14行）
CMakeLists.txt／build.bat／CMakePresets.json: 無変更（§10 監査）。
```

---

## 2. M0 Result — Control baseline

```text
Run:     M0（Control baseline・Recovery なし・publication／crossfade／retire 追加操作なし）
Command: .\build\Release\AudioEngineHarness.exe --fpm-m0
Process exit code: 0（sequence 完遂。T3b 成否は class で判定・exit code による捏造なし）

完全 stdout/stderr（tmp_fpm_m0.log・397 bytes・内容結合なし・原文のまま）:
[FPM] start fpm_m0_control
[FPM] m0 terminal phase=10 phaseComplete=1 completed=1 blockingReason=9 violations=0 fullyDrained=0 t3b=0 lateCallbacks=0 postStopEnqueue=0
[FPM] m0 terminal_drain_audit pendPub=0 pendRetire=0 xfade=0 routerPending=2 deferred=0 quarRes=0 activeWorlds=1 published=3 retired=2 activeReaders=0 stuckReaders=0 overflowRes=0
[FPM] m0 summary t3b=0 class=INVALID-TERMINAL

Classification: INVALID-TERMINAL
  （phaseComplete==1 かつ t3b==0。修正・再実行せず測定結果として固定）

phase:            10（ShutdownPhase::ShutdownComplete。enum 順序: Running=0 … TimedOut=8, Failed=9, ShutdownComplete=10）
phaseComplete:    1
completed:        1
blockingReason:   9（ShutdownBlockingReason::Unknown。None=0 … ActiveBuilder=8, Unknown=9）
transitionViolations: 0
fullyDrained:     0
t3b:              0

lateCallbacks:    0
postStopEnqueue:  0

terminal_drain_audit:
  pendPub=0 pendRetire=0 xfade=0 routerPending=2 deferred=0 quarRes=0
  activeWorlds=1 published=3 retired=2 activeReaders=0 stuckReaders=0 overflowRes=0
```

T3b 5 点（M0）：phaseComplete=1 ✓／completed=1 ✓／blockingReason=None ✗（Unknown）／
violations=0 ✓／fullyDrained=1 ✗（0）→ **t3b=0（3/5）**。
`ShutdownComplete＋completed=1＋violations=0` のみでは VALID ではない（指示 §4 境界どおり）。

---

## 3. M1 Result — Recovery 1 episode

```text
Run:     M1（Recovery 1 episode・E2 winner 不使用・Running 中 waitForDrain なし）
Command: .\build\Release\AudioEngineHarness.exe --fpm-m1
  （M0 の結果確定後に実行）
Process exit code: 0（sequence 完遂）

完全 stdout/stderr（tmp_fpm_m1.log・673 bytes・原文のまま）:
[FPM] start fpm_m1_single_recovery
[FPM] m1 episode dSeq=1 dCoord=0 dCmt=0 dTake=0 dBld=0 seqAdvanced=1
[FPM] m1   pre  req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=3 blo=0
[FPM] m1   post req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=4 blo=0
[FPM] m1 episode_ok ok=1
[FPM] m1 terminal phase=10 phaseComplete=1 completed=1 blockingReason=9 violations=0 fullyDrained=0 t3b=0 lateCallbacks=0 postStopEnqueue=0
[FPM] m1 terminal_drain_audit pendPub=0 pendRetire=0 xfade=0 routerPending=2 deferred=0 quarRes=0 activeWorlds=1 published=4 retired=3 activeReaders=0 stuckReaders=0 overflowRes=0
[FPM] m1 summary episodeOk=1 t3b=0 class=INVALID-TERMINAL

Classification: INVALID-TERMINAL（phaseComplete==1 かつ t3b==0・固定）

phase: 10（ShutdownComplete）／phaseComplete: 1／completed: 1／
blockingReason: 9（Unknown）／transitionViolations: 0／fullyDrained: 0／t3b: 0
lateCallbacks: 0／postStopEnqueue: 0

terminal_drain_audit:
  pendPub=0 pendRetire=0 xfade=0 routerPending=2 deferred=0 quarRes=0
  activeWorlds=1 published=4 retired=3 activeReaders=0 stuckReaders=0 overflowRes=0

episodeOk: 1（Recovery sequence 補助判定。shutdown success の代替ではない）
dSeq: 1／dCoord: 0／dCmt: 0／dTake: 0／dBld: 0／seqAdvanced: 1
pre snapshot:  req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=3 blo=0
post snapshot: req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=4 blo=0
  （seq 3→4 前進・他 counter 不変・drop/blo 0）
```

`episodeOk==true` は shutdown success ではない。T3b のみが authority であり M1 の T3b=0。

---

## 4. M2 Result — Full pipeline

```text
Run:     M2（Full pipeline。
  start → startup settle → publish-1 → crossfade settle → Recovery episode →
  publish-2 → settle → stop → T3b capture。
  publish は requestRebuild(convo::RebuildKind::Structural) 経路。
  追加 prepareToPlay／新規 publish 経路／shutdown 操作なし）
Command: .\build\Release\AudioEngineHarness.exe --fpm-m2
  （M0/M1 の結果確定後に実行）
Process exit code: 0（全 sequence 完遂・ABORT 経路 exit 2 には該当せず）

完全 stdout/stderr（tmp_fpm_m2.log・704 bytes・原文のまま）:
[FPM] start fpm_m2_full_pipeline
[FPM] m2 xfade_settle settled=1
[FPM] m2 episode dSeq=1 dCoord=0 dCmt=0 dTake=0 dBld=0 seqAdvanced=1
[FPM] m2   pre  req=2 que=2 dup=0 take=2 bld=2 cmt=2 coord=3 drp=0 seq=4 blo=0
[FPM] m2   post req=2 que=2 dup=0 take=2 bld=2 cmt=2 coord=3 drp=0 seq=5 blo=0
[FPM] m2 episode_ok ok=1
[FPM] m2 terminal phase=10 phaseComplete=1 completed=1 blockingReason=9 violations=0 fullyDrained=0 t3b=0 lateCallbacks=0 postStopEnqueue=0
[FPM] m2 terminal_drain_audit pendPub=0 pendRetire=0 xfade=0 routerPending=2 deferred=0 quarRes=0 activeWorlds=1 published=6 retired=5 activeReaders=0 stuckReaders=0 overflowRes=0
[FPM] m2 summary episodeOk=1 t3b=0 class=INVALID-TERMINAL

Classification: INVALID-TERMINAL（phaseComplete==1 かつ t3b==0・固定）

phase: 10（ShutdownComplete）／phaseComplete: 1／completed: 1／
blockingReason: 9（Unknown）／transitionViolations: 0／fullyDrained: 0／t3b: 0
lateCallbacks: 0／postStopEnqueue: 0

terminal_drain_audit:
  pendPub=0 pendRetire=0 xfade=0 routerPending=2 deferred=0 quarRes=0
  activeWorlds=1 published=6 retired=5 activeReaders=0 stuckReaders=0 overflowRes=0

episodeOk: 1（補助判定のみ）
xfade_settle: settled=1（diagnostic 条件。shutdown success authority ではない）
dSeq: 1／dCoord: 0／dCmt: 0／dTake: 0／dBld: 0／seqAdvanced: 1
pre snapshot:  req=2 que=2 dup=0 take=2 bld=2 cmt=2 coord=3 drp=0 seq=4 blo=0
post snapshot: req=2 que=2 dup=0 take=2 bld=2 cmt=2 coord=3 drp=0 seq=5 blo=0
  （seq 4→5 前進・他 counter 不変・drop/blo 0）
```

期待 sequence（§5 手順）はすべて完遂し T3b capture に到達。terminal は INVALID-TERMINAL。

---

## 5. T3b 5-point Matrix

authority は T3b（5 点 conjunction）のみ。各 run の内訳：

```text
               M0   M1   M2
phaseComplete   1    1    1   ✓
completed       1    1    1   ✓
blocking=None   0    0    0   ✗（全 run Unknown=9）
violations=0    1    1    1   ✓
fullyDrained    0    0    0   ✗（全 run 0）
─────────────────────────────────
t3b             0    0    0   → 3/5・全 run T3b 不成立
class           INVALID-TERMINAL ×3
```

境界確認（指示 §4）：

```text
phase == ShutdownComplete → T3a のみ。T3b ではない。
completed == true → terminal phase 到達のみ。
blockingReason == Unknown → blocking なしではない（None のみが成立条件）。
transitionViolations == 0 → violation なし（成立・ただし単独では VALID にならない）。
isFullyDrained() == false → drain predicate 不成立。
5 条件すべて成立せず → VALID ではない。
collectDrainAudit() の isAllZero 相当による VALID 補正 → 行っていない（禁止どおり）。
```

---

## 6. terminal_drain_audit（3 run 並列）

diagnostic（authority ではない）。全 run で同一パターン：

```text
               M0   M1   M2
pendPub         0    0    0
pendRetire      0    0    0
xfade           0    0    0
routerPending   2    2    2   ← 全 run で残留（非ゼロの唯一の pending 系項目）
deferred        0    0    0
quarRes         0    0    0
activeWorlds    1    1    1
published       3    4    6   ← run 内容に応じ増加（M0:3／M1:4／M2:6）
retired         2    3    5   ← published-1 を維持（M0:2／M1:3／M2:5）
activeReaders   0    0    0
stuckReaders    0    0    0
overflowRes     0    0    0
```

観測事実のみを記録する。帰属（routerPending=2 と blockingReason=Unknown／
fullyDrained=0 の因果）は本 gate の対象外であり、ここでは断定しない。

---

## 7. M1/M2 Supplemental Evidence

```text
M1: episode=1・E2 winner 不使用・Running 中 waitForDrain なし（実装どおり）。
    episodeOk=1・seq 3→4（dSeq=1 のみ前進・dCoord/dCmt/dTake/dBld=0・
    seqAdvanced=1・drop/blo 0）。Recovery sequence は完遂。
M2: publish-1（Structural）→ xfade settle settled=1 →
    Recovery episode（episodeOk=1・seq 4→5・他不変・drop/blo 0）→
    publish-2（Structural）→ settle → stop → T3b capture。
    pre/post の req/que/take/bld/cmt/coord は M1 の 1→1 に対し M2 では 2→2／coord 3
    （publish-1 分の累積を反映・counter 期待値での shutdown 判定は行わない）。
```

`episodeOk==true` はいずれも Recovery sequence の補助判定であり、
shutdown success（T3b）の代替ではない（M1/M2 とも T3b=0）。

---

## 8. Classification

```text
M0: INVALID-TERMINAL（exit 0・ShutdownComplete・t3b=0）
M1: INVALID-TERMINAL（exit 0・ShutdownComplete・t3b=0・episodeOk=1）
M2: INVALID-TERMINAL（exit 0・ShutdownComplete・t3b=0・episodeOk=1）
ABORTED: 0 件（全 run が sequence 完遂・exit 2 なし）
CRASH / PROCESS FAILURE: 0 件（全 run が正常終了・T3b 取得可）
VALID: 0 件
```

M0 の INVALID-TERMINAL 時に修正・再実行は行っていない（指示 §2 どおり結果を固定し M1 へ進行）。
M1・M2 についても同様に固定し、terminal failure の修正は行っていない。

---

## 9. Cross-run Comparison

```text
terminal 5 点: 3 run で完全一致（1,1,Unknown,0,0 → t3b=0）。
  Recovery episode の有無（M0 なし／M1 あり）、publish-1／publish-2・crossfade settle
  の有無（M2 のみ）は terminal signature を変えない。
lateCallbacks／postStopEnqueue: 3 run とも 0。
drain audit: pending 系は routerPending=2 のみが 3 run で同一残留。
  published／retired は run 内容に応じ増加（3/2 → 4/3 → 6/5・差はいずれも 1）。
  activeWorlds=1・activeReaders/stuckReaders=0 は不変。
episode: M1/M2 とも dSeq=1 のみ・seq 1 前進・drop/blo 0・episodeOk=1。
  pipeline 操作量（M2 の publish×2）が episode counter に累積するが terminal には影響なし。
```

---

## 10. Production / Test / CMake Diff Audit

本 gate 内の source 変更：**0**（実測のみ・編集なし）。

```text
production diff（本 gate 内）: 0
  ※ HEAD 対比 working-tree の src/audioengine 8 files 差分は R27 等の既存差分
  （FPM-Impl-1 §11 監査済み・本 gate で触っていない）。
getter diff: 0／counter diff: 0
shutdown semantic diff: 0／epoch-reclaim diff: 0／Recovery semantic diff: 0
FPM vehicle 自体の変更: 0
CMake target 追加: 0／build 設定変更: 0
  （CMakeLists.txt 最終更新 2026-09-22・build.bat 2026-09-12・いずれも実測前から不変）
```

---

## 11. Gate Verdict

```text
M0 → M1 → M2 を順序どおり各 1 回実行・全 run exit 0・証拠固定。
分類: INVALID-TERMINAL ×3（VALID 0・ABORTED 0・CRASH 0）。
T3b 5 点: 3 run とも 3/5（blockingReason=Unknown・fullyDrained=0 が不成立要因）。
→ 実際の Full-pipeline sequence（publish×2＋crossfade settle＋Recovery episode を含む）は
   T3b terminal contract を満たして終了できないことを測定した。
→ P3-5-FPM-Run-1 MEASURED（測定 gate として完遂。VALID 取得ではない）。
```

STOP 条件（指示 §5）の発動有無：

```text
production source 変更の必要 → 本 gate では発生させていない（実測のみ）。
public API／getter／counter 変更の必要 → 同上。
shutdown／epoch／reclaim／Recovery semantics 変更の必要 → 同上。
FPM vehicle 自体の変更の必要 → なし（3 run とも sequence 完遂・T3b 取得可）。
M0/M1/M2 INVALID-TERMINAL → 修正要求と解釈せず測定結果として固定（指示 §5 どおり）。
```

---

## 12. STOP / Next Gate

```text
FPM-Impl-1-PASS
       ↓
P3-5-FPM-Run-1 MEASURED（本 gate・M0/M1/M2 実測・INVALID-TERMINAL×3 固定）
       │
       └── 次 gate は別指示待ち（本 gate では選定しない）。
           ※ terminal failure の修正・shutdown/epoch/reclaim の修正・
           WORK105 の E/D/B2/C 問題への復帰・Recovery E2 winner 評価は
           いずれも本 gate の対象外であり、ここでは着手しない。
```

- source は R27 production＋R23/R30/R32 test vehicle＋FPM M0/M1/M2 vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7 解釈制約・H-B 対象外を維持する。
- 本 gate はここで停止する。
