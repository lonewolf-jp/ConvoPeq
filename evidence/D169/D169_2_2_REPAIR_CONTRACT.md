# D169-2-2 — Repair Contract Approval（RC-D169-2-1〜7 契約固定）

```text
D169-2-2 — Duplicate-Prepare Collapse Repair Contract Approval
Date:        2026-09-07
Type:        read-only / contract freeze（production/test/CMake/build script/tool 変更 0・実装禁止）
Prior audit: evidence/D169/D169_2_1_DUPLICATE_PREPARE_AUDIT.md（Case A — Defect confirmed・availability クラス）
Scope:       D169-2-4（minimal implementation）への拘束力ある契約
Status:      **APPROVED — 契約固定（候補 a「collapse を真の no-op 化」採用）**
```

---

## 0. 契約の根拠（D169-2-1 確定の defect chain）

```text
Prepared（同一 SR/BS で動作中）
  ↓ JUCE device restart（audioDeviceStopped → AboutToStart → setProcessor swap）
prepareToPlay(same SR/BS)
  ↓
enterPrepare() — collapse 経路: phase 不変の LifecycleToken{epochId, Prepared} を返却
  ↓                               (ISRLifecycle.cpp:27-36)
prepareToPlay 本体に collapse 検出点が存在しない → 全 body を実行
  ↓
leavePrepare(token) — currentPhase == Preparing 前提違反
  ↓
std::abort() → 0xC0000409（ISRLifecycle.cpp:52-55）
```

本契約の修復点は **collapse 判定そのものではなく、その結果（collapsed token）を
`prepareToPlay()` が通常 transaction として継続してしまう接続部**である。
修復方針は **候補 a「collapse を真の no-op 化」**（候補 b「collapse 廃止」は不採用）。

採用根拠: 現行 `enterPrepare()` は同一 SR/BS の Prepared に対する collapse semantics を
明示実装済みであり、prepare 本体には RuntimeWorld 既存時の placeholder 作成抑制がある。
「既に Prepared で SR/BS も同一なら、JUCE が再度 prepare を要求しても engine の
prepare transaction は開始しない」という現行の collapse 意図を維持する方が
意味論的にも変更量的にも安全である（JUCE 冪等 re-prepare 契約と整合）。

## RC-D169-2-1 — Collapse は terminal transaction ではなく no-op

同一条件（`phase == Prepared` && `lastPreparedSampleRate == requested sampleRate` &&
`lastPreparedBlockSize == requested blockSize`）では `enterPrepare()` の collapse
semantics を**維持**する。collapsed token は「prepare transaction が開始された token」
ではなく **duplicate request が既存 Prepared state に吸収されたことの結果**として扱う。

2 経路の分離（契約が要求する正規形）:

```text
【通常経路 — ENTERED】                【collapse 経路 — COLLAPSED】
enterPrepare                          enterPrepare
  ↓                                     ↓
Preparing（phase 遷移あり）            Prepared（phase 不変）
  ↓                                     ↓
prepare body 全実行                   NO-OP（prepare body 非実行）
  ↓                                     ↓
leavePrepare                          leavePrepare 非実行
  ↓                                     ↓
Prepared                              Prepared（不変のまま return）
```

## RC-D169-2-2 — collapse 判定は LifecycleRuntime に残す

`enterPrepare()` の duplicate 判定を `prepareToPlay()` 側へ移動しない。
`prepareToPlay()` 側で `if (alreadyPrepared ...)` を再実装して
**第二の lifecycle authority を作らない**（Practical Stable ISR Bridge の
「状態遷移を一箇所に集約」原則）。`prepareToPlay()` が行うのは
**collapsed token の識別と早期 return のみ**である。

## RC-D169-2-3 — collapsed token は leavePrepare に渡さない

collapse 結果では `lifecycleRuntime_.leavePrepare(token)` を**実行しない**
（早期 return により自動的に達成）。これにより `leavePrepare()` が要求する
`currentPhase == Preparing` 前提を変更せずに済む。
**`leavePrepare()` を緩和して Prepared token を受け入れる修復は禁止**。

## RC-D169-2-4 — token identity の新設は禁止

D169-2-1 R3（LifecycleToken は照合されない）は本修復では変更しない。
以下を禁止: prepare transaction ID・generation tag・epoch による token validation・
token validity flag・新しい atomic・新しい mutex・cancellation state・
prepare ownership object。

**判別子の規定**: collapse の識別には既存 `LifecycleToken::expectedPhase` を用いる。
`enterPrepare()` 内で `expectedPhase == LifecyclePhase::Prepared` を返すのは
**collapse 経路のみ**（通常経路は常に `Preparing`）であるため、既存 field の読み取りで
あり新規 state の追加に当たらない。一意性は D169-2-3 preflight で確認する（P1）。

## RC-D169-2-5 — collapse では全 prepare side effect を禁止

collapse 時は **prepare body の入口直後**（`PrepareToPlay.cpp:20` の `enterPrepare()`
帰還直後・`:22` 以降の全 side effect より前）で return する。
「`leavePrepare()` 直前での return」は不十分 — abort は消えても
duplicate prepare の副作用が残るため、修復点は body 入口でなければならない。

collapse 時に実行禁止の side effect（PrepareToPlay.cpp 行番号は D169-2-1 時点）:

```text
rebuildRequestGeneration reset（:96）
pendingTask / hasPendingTask / publishRetryReady reset（:82-84）
idle publish #2（:138-161）
latency buffer reallocation / rollbackPrepareFailure 領域（:186-204）
crossfadeRuntime reset / gain re-init（:133-135）
analyzerFifo / crossfade buffer re-init（:164-172）
placeholder creation（:236-304）
submitRebuildIntent（:311-323）
lifecycleState → Prepared publish（:230）
m_healthMonitor.reset() / resetProgressObservation（:48/:101）
```

診断目的のログ出力（diagLog / DBG — 状態変化なし）は許容する。

## RC-D169-2-6 — 非 collapse prepare semantics は完全維持

以下の遷移は無変更とする:

```text
Uninitialized → Preparing → Prepared
Released      → Preparing → Prepared
Prepared + SR/BS 変更 → Preparing → Prepared
```

SR/BS が異なる場合は collapse せず従来どおり完全 prepare を実行する。
placeholder / RuntimeWorld / rebuild / admission の既存 protocol に変更を加えない。

## RC-D169-2-7 — blocked-return の phase 残留は今回の修復対象外

D169-2-1 §3.3 の latent case（enterPrepare の phase 遷移後、lifecycleState CAS
block で早期 return し phase=Preparing が残留）は本 RC で**修復しない**。
理由: (1) production caller は Message Thread 直列化、(2) D169-2-1 が production
不可達と判定、(3) scope 拡大の防止、(4) prepare FSM の別問題を duplicate-collapse
修復に混ぜない。**D169-2-4 では blocked-return 経路（PrepareToPlay.cpp:53-59）を
変更禁止**。D169-2-3 preflight で early-return 挿入位置が既存 block/rollback
semantics と干渉しないことを確認する（P2）。

---

## 1. 受入条件（RC-1〜RC-10）

| ID | 条件 |
| --- | --- |
| RC-1 | 同一 Prepared + 同一 SR/BS は collapse のまま |
| RC-2 | collapse は真の no-op（prepare body 非実行） |
| RC-3 | collapse token では `leavePrepare()` を呼ばない |
| RC-4 | `leavePrepare()` の `Preparing` 前提は変更しない |
| RC-5 | collapse 時の prepare body side effect = 0（状態変化系。診断ログのみ許容） |
| RC-6 | 非 collapse prepare は既存 protocol 不変 |
| RC-7 | LifecycleRuntime が唯一の collapse authority |
| RC-8 | transaction ID / generation / mutex / atomic 等の追加禁止 |
| RC-9 | blocked-return phase residue は今回 scope 外（同経路の変更禁止） |
| RC-10 | placeholder / RuntimeWorld / rebuild / publish / retire protocol は変更しない |

## 2. defect chain の構造的切断

```text
BEFORE（D169-2-1 確定の defect chain）         AFTER（本契約適用後）
Prepared                                       Prepared
  ↓ same SR/BS                                   ↓ same SR/BS
collapse token returned                        collapse token returned
  ↓                                              ↓
prepare body executes                          [early return] ← 修復点（入口直後）
  ↓                                              ↓
leavePrepare()                                 NO leavePrepare / NO side effect
  ↓                                              ↓
currentPhase != Preparing                      Prepared のまま正常 return
  ↓                                              （JUCE 冪等 re-prepare として吸収）
std::abort() 0xC0000409
```

## 3. D169-2-3 preflight への引き継ぎ項目

| ID | 確認項目 |
| --- | --- |
| P1 | `enterPrepare()` が `expectedPhase == Prepared` の token を返すのは collapse 経路のみ（一意判別子成立）の source 再確認 |
| P2 | early-return 挿入位置（:20 直後）が既存 block/rollback semantics（:30-44 rollbackPrepareFailure・:51-67 CAS block）と干渉しないこと |
| P3 | collapse no-op が JUCE 契約上安全であることの再確認（同一 SR/BS なので latency buffer 要求寸法不変・buffer は前回 prepare のものが生存・world/active DSP 生存） |
| P4 | RC-5 の side effect list が挿入位置以降にすべて存在すること（抜け漏れ確認 — 新規 side effect の混入がないか） |

## 4. D169-2-5〜7 検証フック（要旨）

- D169-2-5（targeted collapse regression）: same-SR/BS prepareToPlay 連続呼出で
  abort 0・generation reset 0・再 publish 0・phase/lifecycleState 不変。
  非 collapse 経路（SR/BS 変更・Released→Preparing）の既存 D167/T-I2-1 テスト無変更 PASS。
- D169-2-6（device restart / stress）: 同一 SR/BS での device restart 相当を繰返し
  abort / dump / TV=0。
- D169-2-7（full regression）: Debug/Release CTest 40/40 + WA/DS long-run（D168 基準）。

## 5. 変更範囲

production 0 / test 0 / CMake 0 / build.bat 0 / tool 0 / binary 0（本契約文書のみ新規作成）。

## 6. 後続

```text
D169-2-3 Preflight Source Audit（P1〜P4）
  ↓
D169-2-4 Minimal Implementation（RC-D169-2-1〜7 に拘束・挿入点は :20 直後）
  ↓
D169-2-5 Targeted collapse regression
  ↓
D169-2-6 Device restart / stress
  ↓
D169-2-7 Full regression → D169-2 close
```
