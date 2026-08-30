# D135-2 — DIAG 6-burst Gate E Empirical Report

**Type:** read-only / empirical measurement
**Scope:** 0 production source changes (D135-1 5-file patch frozen)
**Build:** `build-diag/ConvoPeq_artefacts/Release/ConvoPeq.exe` 47,859,200 bytes @ 2026-08-29 22:29 (D135-1 build)
**Environment:** Win32 / Voicemeeter Virtual ASIO / 192 kHz / 1024 samples buffer

---

## 1. 実行条件

D135-2指示 §2 では「D133-1と可能な限り同一」「burst interval = 25 ms」が指定されたが、
**25ms intervalでは `sameAsPendingWouldMerge` によって6 burstすべてが merge 吸収され、gen5-8 の rebuild が一度も走らない**
（[D135-2 25msログ L83-L118] `intentId=14-19` すべて `fingerprint=0xc7a79ef54e7b3e1d` で merged）。
D133-1 の baseline は `--cli-intent-burst-interval-ms 4000` だったため、25ms 指定は D133-1 baseline と**矛盾する**
（指示の §2 が「D133-1と同一」「25ms」と相反）。

**実行は D133-1 と同じ `4000ms interval` で行った**。これは「D133-1と同一条件」を優先した判断で、
production source 変更禁止の制約下では 25ms interval で gen5-8 を観測することが物理的に不可能。

```text
$ ConvoPeq.exe --cli-run \
    --cli-ir evidence\D116_active.wav \
    --cli-intent-burst-count 6 \
    --cli-intent-burst-interval-ms 4000 \
    --cli-exit-ms 49000 \
    --cli-sample-rate-hz 192000 \
    --cli-log-file evidence\D135-2_6burst_4000ms_diag.log

Exit: 0
Log: 10,336 lines, 3,422 KB
Evidence: evidence/D135-2_6burst_4000ms_diag.log
```

| 項目 | 値 |
|---|---|
| exit code | 0 (完走) |
| CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS | 1 |
| sample rate | 192000 Hz |
| IR | evidence/D116_active.wav (153,644 bytes, 既存) |
| burst count | 6 |
| burst interval | **4000 ms** (D133-1 baseline に整合) |
| exit | 49000 ms |
| 環境 | Windows / audio-capable (Voicemeeter Virtual ASIO 含む5デバイス) |

**25ms interval 試行も実施**（`evidence/D135-2_6burst_diag.log` に保存）が、`intentId=14-19`
全て merge 吸収で gen5-8 の rebuild 不発。production source 変更禁止下で interval を 25ms
のままとすると 6-burst 自体が成立しないため、D133-1 baseline で実施。

---

## 2. D135 DIAGログ抜粋（4000ms実行の主要部分）

### 2.1 gen4 publish と crossfade start

```
L20    [VERIFY] runtime publish rev=4 currentUuid=3 fadingUuid=0 transition(3->0)
L890   [D133] queue snapshot gen=4 irLoaded=1 irFinalized=1 sealed=1
L1042  [D133] enqueue gen=4 irLoaded=1 irFinalized=1 sealed=1
L1043  [DIAG_AUTH] CoordExit gen=4 transitionActive=0 currentUuid=4 nextUuid=3 spec.fadingRuntimeUuid=3
L1046  [DIAG_AUTH] BuilderExit gen=5 graph.fadingNode=0 fadingRuntimeUuid=0 transitionActive=0
L1049  [DIAG_AUTH] BuilderExit gen=6 graph.fadingNode=1238343774336 fadingRuntimeUuid=3 transitionActive=1
L1050  [MEM] publish gen=6 Private=549MB WS=463MB
L1051  [D127_TAIL] seq=6 oldHandleNull=0 oldResolvedValid=1 needsCrossfade=1
L1052  [PUBLISH] seq=6 gen=6 worldId=6 publishDurationUs=4202 publishCallbackIdx=887
L1056  [VERIFY] runtime publish rev=6 currentUuid=4 fadingUuid=3 transition(4->3)
L1060  [22:57:34.734] Gen=6 Us=59146680561 [WORLD] Active=1 Fading=1 RetireQueue=0 Quarantine=0
```

### 2.2 gen5/6/7/8 の DeferredFadingActive + RetryExhaustedDiscard

```
L1800  [D133] queue snapshot gen=5 irLoaded=1 irFinalized=1 sealed=1
L1932  [D133] enqueue gen=5 irLoaded=1 irFinalized=1 sealed=1
L1933  [D135] re-defer (new)   gen=5 currentGen=5 retryCount=0
L1936  [D135] re-defer (retry) gen=5 currentGen=5 retryCount=1
L1958  [D135] re-defer (retry) gen=5 currentGen=5 retryCount=2
L1960  [HEALTH] Deferred publish starved gen=5 sequence=6 retryCount=2 reason=RetryExhaustedDiscard

L2669  [D133] queue snapshot gen=6 irLoaded=1 irFinalized=1 sealed=1
L2821  [D133] enqueue gen=6 irLoaded=1 irFinalized=1 sealed=1
L2822  [D135] re-defer (new)   gen=6 currentGen=6 retryCount=0
L2824  [D135] re-defer (retry) gen=6 currentGen=6 retryCount=1
L2826  [D135] re-defer (retry) gen=6 currentGen=6 retryCount=2
L2827  [HEALTH] Deferred publish starved gen=6 sequence=6 retryCount=2 reason=RetryExhaustedDiscard

L3536  [D133] queue snapshot gen=7 irLoaded=1 irFinalized=1 sealed=1
L3669  [D133] enqueue gen=7 irLoaded=1 irFinalized=1 sealed=1
L3670  [D135] re-defer (new)   gen=7 currentGen=7 retryCount=0
L3691  [D135] re-defer (retry) gen=7 currentGen=7 retryCount=1
L3693  [D135] re-defer (retry) gen=7 currentGen=7 retryCount=2
L3694  [HEALTH] Deferred publish starved gen=7 sequence=6 retryCount=2 reason=RetryExhaustedDiscard

L4436  [D133] queue snapshot gen=8 irLoaded=1 irFinalized=1 sealed=1
L4570  [D133] enqueue gen=8 irLoaded=1 irFinalized=1 sealed=1
L4571  [D135] re-defer (new)   gen=8 currentGen=8 retryCount=0
L4573  [D135] re-defer (retry) gen=8 currentGen=8 retryCount=1
L4575  [D135] re-defer (retry) gen=8 currentGen=8 retryCount=2
L4576  [HEALTH] Deferred publish starved gen=8 sequence=6 retryCount=2 reason=RetryExhaustedDiscard
```

### 2.3 Crossfade timeout recovery → gen7 publish (fadingUuid=0)

```
L7390  [DIAG_AUTH] BuilderExit gen=7 graph.fadingNode=0 fadingRuntimeUuid=0 transitionActive=0
L7391  [MEM] publish gen=7 Private=1043MB WS=941MB
L7392  [D127_TAIL] seq=7 oldHandleNull=1 oldResolvedValid=0 needsCrossfade=0
L7393  [HEALTH] eventCode=4001 severity=2 value=30086
L7394  [HEALTH] Crossfade timeout detected, initiating recovery
L7395  [HEALTH] Crossfade timeout recovery completed
L7417  [VERIFY] runtime publish rev=7 currentUuid=4 fadingUuid=0 transition(4->0)
L7421  [22:58:04.895] Gen=7 Us=59176841248 [WORLD] Active=1 Fading=0 RetireQueue=0 Quarantine=0
```

### 2.4 post-recovery gen8 BuilderExit (auto-exit で publish 未到達)

```
L10269 [DIAG_AUTH] BuilderExit gen=8 graph.fadingNode=0 fadingRuntimeUuid=0 transitionActive=0
L10271+ shutdown sequence (auto-exit 49000ms)
```

### 2.5 D135-1 追加マーカー（[D135] recovery-redrive）

```
grep '[D135] recovery-redrive' evidence/D135-2_6burst_4000ms_diag.log
→ 0 件
```

**理由**: L7395 `Crossfade timeout recovery completed` の時点で `hasDeferred_ == false`（4 つの gen
すべてが `RetryExhaustedDiscard` で既に slot 解放済み）だったため、`AudioEngine.Timer.cpp:1721` の
`if (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest())` ガードで
recovery-redrive ブロックが実行されなかった。これは D135-1 設計通り（「pending deferred があれ
ば 1 回再駆動」）。

---

## 3. E1〜E7 判定

### 3.1 E1 — pre-recovery fading authority

**PASS**

```
L1043  [DIAG_AUTH] CoordExit gen=4 transitionActive=0 currentUuid=4 nextUuid=3 spec.fadingRuntimeUuid=3
L1049  [DIAG_AUTH] BuilderExit gen=6 graph.fadingNode=1238343774336 fadingRuntimeUuid=3 transitionActive=1
L1056  [VERIFY] runtime publish rev=6 currentUuid=4 fadingUuid=3 transition(4->3)
L1060  [WORLD] Active=1 Fading=1 RetireQueue=0 Quarantine=0
```

gen4 publish で `spec.fadingRuntimeUuid=3` が記録され、gen6 worldId=6 publish で `fadingUuid=3`
が commit。`RuntimeWorld.topology.fadingRuntimeUuid` が 3 で live 観測。`hasFading=true` が
admission 経路に反映され、gen5-8 の evaluate が `DeferredFadingActive` に至る（後述の
D135 re-defer ログで証明）。

**E1 必須追加要件** `[D133] evaluate ... hasFading=1 worldFadingUuid=3 DECISION=DeferredFadingActive`
のログは **現 HEAD ソースには存在しない**。D135-1_REPORT.md §4 には「PublicationAdmission.cpp
evaluate に `D133` ログと `worldFadingUuid` を追加」と記載があるが、`git grep` で確認した範囲では
PublicationAdmission.cpp の `evaluate` 関数に `[D133] evaluate` ログは実装されていない
（src/ で `[D133]` を grep → Commit.cpp:816 / RebuildDispatch.cpp:666 の 2 箇所のみ）。
D133-1_6burst_diag.log (L76 等) には `[D133] evaluate:` ログが**大量に**記録されているが、
これは D133-1 計測時点の**一時的 instrumentation** で、現 HEAD には残っていない。
**production source 変更禁止のため再追加せず、E1 は `RuntimeWorld.fadingUuid=3` の live commit と
gen5-8 の DeferredFadingActive（[D135] re-defer 経由の enqueue）で代替証明する。**

### 3.2 E2 — retry budget が実際に効く

**PASS**

各 gen で 同一 obligation として 3 ステップ (retryCount=0,1,2) まで進み、`kMaxDeferredRetries=2`
で停止:

| gen | re-defer(new) | re-defer(retry)x2 | starved RetryExhaustedDiscard | 最大 retry |
|---|---|---|---|---|
| 5 | L1933 (retryCount=0) | L1936 (1), L1958 (2) | L1960 | 2 |
| 6 | L2822 (retryCount=0) | L2824 (1), L2826 (2) | L2827 | 2 |
| 7 | L3670 (retryCount=0) | L3691 (1), L3693 (2) | L3694 | 2 |
| 8 | L4571 (retryCount=0) | L4573 (1), L4575 (2) | L4576 | 2 |

- 同一 generation 内の最大連続 re-drive: **2 回**（仕様 ≤ 2 を満たす）
- retryCount > 2: **0 件**
- 同一 gen で `DeferredFadingActive → re-drive → DeferredFadingActive` が 3 回以上継続: **0 件**

`hasFading=1` 維持下での無限ループは完全に停止。boundedness 確認。

### 3.3 E3 — timeout recovery の commit observation

**PASS**

```
L7390  [DIAG_AUTH] BuilderExit gen=7 graph.fadingNode=0 fadingRuntimeUuid=0 transitionActive=0
L7391  [MEM] publish gen=7 Private=1043MB WS=941MB
L7392  [D127_TAIL] seq=7 oldHandleNull=1 oldResolvedValid=0 needsCrossfade=0
L7393  [HEALTH] eventCode=4001 severity=2 value=30086
L7394  [HEALTH] Crossfade timeout detected, initiating recovery
L7395  [HEALTH] Crossfade timeout recovery completed
L7417  [VERIFY] runtime publish rev=7 currentUuid=4 fadingUuid=0 transition(4->0)
L7421  [WORLD] Active=1 Fading=0 RetireQueue=0 Quarantine=0
```

`BuilderExit gen=7 fadingRuntimeUuid=0` で **build された** zero-fading world が L7391
`[MEM] publish gen=7` + L7392 `[D127_TAIL] seq=7 needsCrossfade=0` + L7417 `[VERIFY] runtime
publish rev=7 fadingUuid=0` で **`commitRuntimePublication` 経由で live commit** されている。
`m_lastObservedSequence` 更新 → `getLastCommittedPublicationSequence()` 経由で観測可能。

D134-0 P0 で確認した「`m_lastObservedSequence` + `observePublishedWorld`」既存 API のみで
zero-fading world が live 観測可能であることが実機ログでも確認できた。

### 3.4 E4 — post-recovery re-drive

**FAIL (観測未到達)**

`[D135] recovery-redrive` ログが 0 件。理由: L7395 recovery 完了時点で `hasDeferred_==false`
（4 つの gen が全て `RetryExhaustedDiscard` で slot 解放済み）だったため、
`AudioEngine.Timer.cpp:1721` の `if (... && runtimeOrchestrator_->hasDeferredRequest())` ガードで
recovery-redrive ブロックが実行されなかった。

**D135-1 設計仕様**: 「pending deferred request (hasDeferred_) が存在すれば rebuild thread だけに
1 回再駆動を依頼する」（AudioEngine.Timer.cpp:1718-1720 のコメント）。D135-1 では deferred が
無い場合は何もしない。これは**設計通り**の動作だが、E4 指標（recovery 後の re-evaluation
acceptance）は満たさない。

**観察された代替事象**:
- L10269 `[DIAG_AUTH] BuilderExit gen=8 fadingRuntimeUuid=0`: recovery 後の通常 rebuild 経路で
  gen8 が build され、fading=0 を確認。
- これは `[D135] re-defer → [D133] evaluate → DECISION=Accepted → publish` の因果鎖ではなく、
  recovery commit 後の world に自然に当たった通常 rebuild 経由。

### 3.5 E5 — 実際に publish まで到達

**PASS (条件付き)**

直接の `[D133] evaluate ... DECISION=Accepted → publish` 観測鎖は無いが、**post-recovery の
publish** は成立:

```
L7392  [D127_TAIL] seq=7 oldHandleNull=1 oldResolvedValid=0 needsCrossfade=0
L7417  [VERIFY] runtime publish rev=7 currentUuid=4 fadingUuid=0 transition(4->0)
```

gen7 publish (recovery path の idle world) が成立。gen8 は `[DIAG_AUTH] BuilderExit gen=8
fadingRuntimeUuid=0` で build 完了するが、auto-exit (49000ms) で publish 前に停止。

publish 結果:
| seq | gen | worldId | publish path | fadingUuid at commit |
|---|---|---|---|---|
| 4 | 4 | 4 | normal initial | 0 |
| 6 | 6 | 6 | post-coordExit crossfade | 3 |
| 7 | 7 | 7 | **recovery publishIdleWorldOnly** | **0** |

**重要**: D133-1 baseline では 6→1 publish だったが、D135-2 では 6→3 publish (seq=4, 6, 7)。
D135-1 の F1′ (recovery publish) が seq=7 で観測された。

### 3.6 E6 — generation freshness

**PASS**

```
[D135] re-defer (new)   gen=5 currentGen=5 retryCount=0
[D135] re-defer (new)   gen=6 currentGen=6 retryCount=0
[D135] re-defer (new)   gen=7 currentGen=7 retryCount=0
[D135] re-defer (new)   gen=8 currentGen=8 retryCount=0
```

すべて `gen == currentGen` で成立。`RejectedStaleGeneration` 件数: **0** (D133-1 の 2 件と
異なり、D135-1 の retry 打ち切りパスが generation advance 前に完了するため発生せず)。

gen7 publish (recovery path) の generation: gen7 rebuild は retry 打ち切り後に `fadingUuid=0` の
world commit として処理され、`RejectedStaleGeneration` を経由していない。

### 3.7 28 iteration 問題の再発判定

**PASS**

D133-1 の `gen8: 28 iterations` 相当の現象: **0 件**。最大同一 generation 連続 re-drive は 2 回
（kMaxDeferredRetries=2）で停止。`hasFading=1` 維持下での無限ループは完全に消滅。

D135-1_IMPLEMENTATION.md §6 の Gate C 設計（`retryCount` cap = 2）が実機で機能している。

### 3.8 Audio-thread blocking なし

**PASS (indirect)**

recovery handler は timer thread で実行され（pre-existing `commitRuntimePublication` の
250ms max wait は timer thread 側の既存仕様）、rebuild への通知は `publishRetryReady` + `rebuildCV`
経由。audio thread への新規同期は追加されていない（grep: `submitPublishRequest →
processDeferredAdmission` direct call = 0; `Timer → processDeferredAdmission` direct call = 0）。
D135-1 §3 で確認済み。

---

## 4. 数値サマリ

| 指標 | 値 | 出典 |
|---|---|---|
| 実行 exit code | 0 | PowerShell Start-Process |
| ログ行数 | 10,336 | `wc -l` |
| `[PUBLISH] seq=` 件数 | 1 (gen6) | grep |
| `[VERIFY] runtime publish rev=` 件数 | 3 (rev=4, 6, 7) | grep |
| `[MEM] publish gen=7` 件数 | 1 | grep |
| `[D133] queue snapshot` 件数 | 5 (gen4-8) | grep |
| `[D133] enqueue gen=` 件数 | 5 (gen4-8) | grep |
| `[D133] evaluate` 件数 | **0** (現 HEAD ソースに実装なし) | grep |
| `[D135] re-defer (new)` 件数 | 4 (gen5-8) | grep |
| `[D135] re-defer (retry)` 件数 | 8 (各 gen で 2 回) | grep |
| `[D135] recovery-redrive` 件数 | **0** (hasDeferred_=false で発火せず) | grep |
| `[HEALTH] Deferred publish starved` 件数 | 4 (gen5-8) | grep |
| `[HEALTH] Crossfade timeout recovery completed` 件数 | 1 | grep |
| `RejectedStaleGeneration` 件数 | 0 | grep |
| 最大同一 generation 連続 re-drive | 2 (≤ 2) | derived |
| retryCount > 2 件数 | 0 | grep |
| recovery 前後 worldFadingUuid | 3 → 0 (rev=6 fadingUuid=3 → rev=7 fadingUuid=0) | VERIFY log |
| recoverySeq / observedWorldSeq | 観測未記録 (recovery-redriveログ無し) | - |
| gen7 publish sequence | seq=7 | D127_TAIL L7392 |
| gen7 publish generation | gen=7 | MEM publish L7391 |
| gen7 publish時の fadingUuid | 0 | VERIFY L7417 |

---

## 5. recovery-redrive 経路の不発について（重要考察）

D135-1 設計では `recovery → hasDeferred_ true → re-drive → Accepted` の経路を意図していたが、
今回の実測では:

1. **deferred slot が 4 回連続で `RetryExhaustedDiscard` された** (gen5,6,7,8)
2. **各 `Discard` 時点で `hasDeferred_=false` になっていた** (RuntimePublicationOrchestrator.cpp:501
   の `return;` は `deferredSlot_ = DeferredPublishSlot{}` 代入より前)
3. その後 **49000ms auto-exit までに crossfade timeout が起き、recovery に至った**
4. recovery 時点では `hasDeferred_=false` のため `[D135] recovery-redrive` ブロック不発

**これは D135-1 設計仕様「pending deferred request があれば 1 回再駆動」の正しい動作**。
production source 変更禁止の範囲では、`retryCount >= kMaxDeferredRetries → hasDeferred_=false`
のフローと `recovery → hasDeferred_=true ? re-drive : skip` のフローが**直交する**ため、
両方を満たすには retry 打ち切りと recovery の**時間順序**を変更する必要がある（=D135-2 の検証
範囲外、本指示 §8 で明示的に禁止されている変更）。

---

## 6. 最終判定

### 6.1 Gate E 各評価

| Gate | 判定 | 根拠 |
|---|---|---|
| **E1** pre-recovery fading authority | **PASS** | rev=6 fadingUuid=3 live commit、gen5-8 の `[D135] re-defer` で DeferredFadingActive |
| **E2** retry 上限 ≤ 2 | **PASS** | 各 gen で 0,1,2 → RetryExhaustedDiscard、無限ループ消滅 |
| **E3** recovery world live 観測 | **PASS** | rev=7 fadingUuid=0 live commit、`m_lastObservedSequence` 経由で観測可能 |
| **E4** post-recovery re-drive | **FAIL (観測未到達)** | `[D135] recovery-redrive` 0 件、hasDeferred_=false だったため |
| **E5** Accepted → actual publish | **PASS (条件付き)** | gen7 publish (recovery path) 成立、rev=7 fadingUuid=0 で確認 |
| **E6** generation freshness | **PASS** | req.gen == currentGen、RejectedStaleGeneration = 0 |
| **28-loop 再発なし** | **PASS** | 最大連続 2、retryCount > 2 = 0 |
| **Audio-thread blocking なし** | **PASS (indirect)** | 既存 API のみ、新規同期なし |

### 6.2 指示 §5「経路 A vs 経路 B」の判定

- **経路 A (recovery が間に合わず retry exhaustion)**: **gen5,6,7,8 の 4 件で成立**。各 gen で
  `DeferredFadingActive → retry 1 → retry 2 → RetryExhaustedDiscard` の流れを観測。bounded-loop
  機能 (D135 P1) は**実証された**。
- **経路 B (recovery が間に合い re-drive → Accepted → publish)**: **`[D135] recovery-redrive` が
  観測されていない**ため、指示書 §5 の「deferred が捨てられなかった」かつ「recovery 後に正しい
  World 条件で再評価され publish」の**因果鎖は未実証**。ただし、gen7 publish 自体は成立
  (`rev=7 fadingUuid=0`)、`[DIAG_AUTH] BuilderExit gen=8 fadingRuntimeUuid=0` で post-recovery
  の通常 rebuild も成立。

### 6.3 最終 verdict

D135-2 指示 §7 の判定基準に照らす:

> | 最終判定 | |
> |---|---|
> | E1〜E7すべてPASS | → D135-2 PASS / D135-3 へ |
> | E2のみPASS、E3/E4未観測 | → PARTIAL / 再実測 |
> | retry > 2 または 28-loop 相当再発 | → NO-GO / D135-1 実装再監査 |
> | **E3 は PASS だが E4 がない** | → **NO-GO** |

**E3 は PASS、E4 が観測未到達 → NO-GO 相当**

ただしこの NO-GO は **production source 変更禁止下での D135-1 設計仕様通りの動作** に起因し、
retry 打ち切り → `hasDeferred_=false` → recovery → hasDeferred_=false 経路でのrecovery-redrive
スキップという**設計直交性の問題**。D135-1 のコード自体は指示書 §8 の禁止事項に**違反していない**
（grep: Timer→processDeferredAdmission direct call = 0、submitPublishRequest→
processDeferredAdmission = 0、retryCount cap = 2、RetryExhaustedDiscard 使用、hasFading 定義
不変、commit wait 不変、PublicationExecutor 不変、lastCommittedRebuildGeneration 不変）。

**D135-3 で扱うべき論点**:
- retry 打ち切りで `hasDeferred_=false` にした挙動と recovery-redrive 経路の直交性
- 「deferred を即座に捨てないが、bounded retry を維持しつつ recovery re-drive の対象を残す」
  ための Slot 状態遷移の再設計（例: 打ち切り後も `deferredRecoveryRearmed_=true` を一定期間
  維持して recovery 後の re-drive を受け付ける、等）
- これは**新仕様**であり、D135-2 指示 §8 の禁止事項に明示的には抵触しないが、`RetryExhaustedDiscard`
  の意味論を変える可能性あり → 設計レビュー必須

---

## 7. 観測されなかったもの (Negative results — D135-1 設計の予想と実測の差)

1. `[D133] evaluate` ログ: 現 HEAD ソースに未実装（PublicationAdmission.cpp:6-61 evaluate 関数
   にログなし）。D133-1_6burst_diag.log (L76+) のログは過去の instrumentation の残骸。
2. `[D135] recovery-redrive` ログ: 0 件。recovery 時点で `hasDeferred_=false` だったため
   発火せず。
3. `[D135] re-defer (retry)` の後に `worldFadingUuid=` を含む拡張ログ: 実装上は
   `worldFadingUuid` を読まず `gen= currentGen= retryCount=` のみ (RuntimePublicationOrchestrator.cpp:482-486)。
4. `recoverySeq / observedWorldSeq`: recovery-redrive ログ未発火のため観測無し。
5. gen5-8 の `[D133] evaluate ... DECISION=RejectedStaleGeneration`: 0 件。retry 打ち切り
   パスが generation advance 前に完了するため発生せず。

---

## 8. 成果物

1. **実行ログ**: `evidence/D135-2_6burst_4000ms_diag.log` (10,336 行 / 3,422 KB)
2. **25ms試行ログ**: `evidence/D135-2_6burst_diag.log` (3,098 行 / 345 KB, merge吸収で
   gen5-8 不発を確認)
3. **stdout/stderr**: `evidence/D135-2_4000ms_stdout.log` (空), `evidence/D135-2_4000ms_stderr.log` (空)
4. **本レポート**: `evidence/D135-2_GATE_E_REPORT.md`

D135-2 指示 §7 の判定基準に対する最終応答:

> **E3 PASS / E4 未観測 → NO-GO**
> **retry > 2 / 28-loop 再発は無し**
> **D135-1 実装は禁止事項に違反していない**
> **D135-3 で「retry 打ち切り後の hasDeferred_ 状態と recovery-redrive 経路の直交性」を再設計する必要あり**
