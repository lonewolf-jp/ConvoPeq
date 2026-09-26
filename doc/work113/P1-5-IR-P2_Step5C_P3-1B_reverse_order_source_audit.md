# P1-5-IR-P2 — Step 5-C / P3-1-B: Reverse-Order Experiment Protocol Freeze — Source Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-1-B source audit）
- **性質**: **完全 read-only**。source / CMake / production / settle / sleep = **0 変更**。
  **コード変更・build・run は実施していない**（指示どおり source audit のみ）。
- **目的**: reverse-order experiment を実施する前に、`runPair()` の現在のケース順序と
  **Structural intent の投入順序**を source-first で固定する。

---

## 0. State Freeze（PASS）

| 項目 | 値 |
| --- | --- |
| HEAD | `1e9e63e3` |
| `kP15FullMatrix` | `true` |
| A3 binary SHA-256[:16] | `b2c39a9a0e3b1fe3` |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `BOOL=OFF` |
| production source / CMake / JUCE / default / calibration diff | **0** |
| settle / sleep diff | **0** |
| measurement semantics diff | **0** |

```text
A3 = REPRODUCED / A4 = PASS / P3-0 = PASS / P3-1-A = PASS / H-A = CLOSED / H-B = OPEN
```

---

## 1. 必須確認項目（1〜10）の source 確定

### 1-1. 確認結果一覧

| # | 確認項目 | 結果 | source |
| --- | --- | --- | --- |
| 1 | `am=-20 → am=-6` の順序を決めている箇所 | **`for (double adb : { -20.0, -6.0 })`**（初期化子リストの記載順） | `P1PolyphaseGainCharacterization.cpp:746`（ループ入れ子は `ir{g0,g3,g6}` :744 → `osF{1,4,8}` :745 → `adb{-20,-6}` :746） |
| 2 | `ensureTestIr()` が各ケースの前に呼ばれるか | **呼ばれる（1 ケースにつき 2 回）** | `runPair:664`（sc0 の runCase 前）・`runPair:682`（sc1 の runCase 前） |
| 3 | `configureChain()` が各ケースで呼ばれるか | **呼ばれる（runCase 内で 1 回）** | `runCase:412` |
| 4 | `setSoftClipEnabled(false)` が各ケースで呼ばれるか | **呼ばれる（sc0 で `false`・sc1 で `true`）**。**変更検出ガードなし** | `configureChain:365` → `Parameters.cpp:499-504`（無条件 `submitRebuildIntent`） |
| 5 | `setOversamplingFactor(1)` が −20/−6 間で再実行されないこと | **関数は毎ケース呼ばれるが no-op**（`!= newFactor` ガードにより intent 不投入・UI convolver 再 prepare なし） | `configureChain:367` → `Parameters.cpp:547` `if (consumeAtomic(manualOversamplingFactor) != newFactor)` |
| 6 | `waitBacklogZero()` | `waitBacklogZero(e, 30000)`（戻り値は WARN のみ・継続） | `runCase:414`／定義 `:73-84` |
| 7 | `waitWorldPublished()` | `waitWorldPublished(e, seqBefore, 30000)`（`seqBefore` は `configureChain` の**前**に取得） | `runCase:411,417`／定義 `:86-103` |
| 8 | `sleepPump(800)` | `waitWorldPublished` の**後**・`configureCapture`/capture の**前** | `runCase:418` |
| 9 | `waitWorldPublished()` の戻り値が無視されること | **無視される**（`if` を伴わない文として呼ばれる） | `runCase:417`（`waitWorldPublished(e, seqBefore, 30000);` のみ） |
| 10 | capture / DFT / `gainDb` の計算が順序変更以外では同一であること | **同一**（同一 `runCase` 経路・同一 `configureCapture`・同一 `dftMag`・同一 `gainDb = dftDb − ampDb`。差は `p.ampDb` のみ） | `runCase:420-461` |

### 1-2. ケース順序の実測確認（A3 ログとの一致）

A3 実測ログの `p15ir` 行順序（先頭 6 行）:

```text
g0_os1_am-20  →  g0_os1_am-6  →  g0_os4_am-20  →  g0_os4_am-6  →  g0_os8_am-20  →  g0_os8_am-6
```

- source（:744-746 の入れ子）が生成する順序と**完全一致**。
- したがって「−20 → −6」は **`adb` 初期化子リストの記載順のみ**で決まっており、
  他に順序を決める要因（ソート・マップ・乱数等）は存在しない。

### 1-3. `p15eq os=8 → g0_os1` 遷移の位置

```text
(H) p15eq ループ最終ケース  = os=8, amp=0   →  configureChain が setOversamplingFactor(8)
(I) p15ir ループ最初のケース = g0, os=1, am=-20  →  configureChain が setOversamplingFactor(1)  ★ OS 変更
```

- A3/historical 両ログで直前ケースは `p15eq id=b9_am0 os=8`（L132）、直後は `p15ir id=g0_os1_am-20 os=1`（L235）。
- **`setOversamplingFactor(1)` が「変化あり」として実行されるのはこの 1 回のみ**（am=-20 の sc0）。
  以後 `g0_os1_am-6` および sc1 はガードにより no-op。

---

## 2. `runPair()` が投入する Structural intent の順序（source-first・最重要）

### 2-1. `configureChain()` の呼び出し順と、各 setter の intent 投入（確定表）

`configureChain`（:352-388）の実行順に、各 setter が intent を投入するかを確定した:

| # | 呼び出し（`configureChain` 内の順） | setter の変更検出ガード | am=-20(sc0) | am=-6(sc0) |
| --- | --- | --- | --- | --- |
| 0 | `beginBulkParameterRestore()` :357 | — （`m_isRestoringState = true`） | — | — |
| 1 | `setEqBypassRequested(false)` :359 | **ガードなし**（`Parameters.cpp:161` 無条件） | **intent** | **intent** |
| 2 | `setConvolverBypassRequested(false)` :360 | **ガードなし**（`:172` 無条件） | **intent** | **intent** |
| 3 | `setAutoGainStagingEnabled(false)` :361 | ガードあり（`AudioEngine.h:1419` `current == enabled → return`） | no-op | no-op |
| 4 | `setInputHeadroomDb(0)` :362 | ガードあり（`:236` `abs(diff) > 1e-5f`） | no-op | no-op |
| 5 | `setOutputMakeupDb(0)` :363 | ガードあり（`:255`） | no-op | no-op |
| 6 | `setConvolverInputTrimDb(0)` :364 | ガードあり（`:283`） | no-op | no-op |
| 7 | **`setSoftClipEnabled(softClip)`** :365 | **ガードなし**（`:503` 無条件） | **intent** | **intent** |
| 8 | `setSaturationAmount(1.0f)` :366 | ガードあり（`:525` `abs(diff) > 1e-6f`） | no-op | no-op |
| 9 | `setOversamplingFactor(1)` :367 | ガードあり（`:547` `!= newFactor`） | **intent**（8→1 変化） | no-op |
| 10 | `setOversamplingType(IIR)` :368 | ガードあり（`:646` 配下） | no-op | no-op |
| 11 | `setDitherBitDepth(32)` :369 | ガードあり | no-op | no-op |
| 12 | EQ 20 バンド × `setEQBandEnabled(false)` / `setEQBandGain(0)` :371-375 | ガードあり（値不変） | no-op | no-op |
| 13 | `setEQTotalGain(0)` / `setEQAGCEnabled(false)` / `setEQNonlinearSaturation(0)` :384-386 | ガードあり（値不変） | no-op | no-op |
| 14 | **`endBulkParameterRestore(true)`** :387 | **ガードなし**（`:212-220`、`requestRebuildNow && sr > 0` で無条件） | **intent** | **intent** |

- **重要**: `setEqBypassRequested` / `setConvolverBypassRequested` / `setSoftClipEnabled` は
  **値が同じでも必ず intent を投入する**（変更検出ガードが無い）。
  したがって **am=-20 と am=-6 で intent の本数・順序は同一**（値の変化に依存しない）。

### 2-2. intent の merge 挙動（dispatch 抑制条件）

`submitRebuildIntent`（`RebuildDispatch.cpp:151-`）:

| 判定 | 条件 | 効果 |
| --- | --- | --- |
| `sameAsPendingWouldMerge` | `rebuildOutstanding && pending.valid && kind/class/policy/fingerprintVersion/structuralHash/fingerprint/deferCategory が一致`（:201-209） | 同一 signature の pending と同一視 |
| `shouldApplyLatestWinsMerge` | 上記 **かつ** `collapsePolicy == Replaceable` **かつ** `elapsed <= latestWinsWindowMs`（`:217`。window = `getRebuildDebounceMs()`） | — |
| **Merged の効果** | `sameAsPendingWouldMerge && shouldApplyLatestWinsMerge` → `REBUILD_MERGED` を emit して **`return`（dispatch しない）**（`:271-284`） | **rebuild は実行されない** |

- 本ケースの intent 1・2・7 は **同一 signature**（`kind=Structural` / `class=Snapshot` / `policy=Replaceable` /
  fingerprint 同一）であるため、**同一 debounce 窓内では 1 本に merge される**（＝ dispatch は 1 回）。
- intent 14（`endBulkParameterRestore`）は **`class=Structural`** で signature が異なるため、
  intent 1/2/7 とは merge されない（別 dispatch）。

```text
1 runCase あたりの実効 structural rebuild 投入 ≈ 2 本
  (a) Snapshot class: eqBypass / convBypass / softClip が merge されて 1 本
  (b) Structural class: endBulkParameterRestore(true) が 1 本
  ※ am=-20 のみ (c) setOversamplingFactor(1) が Snapshot class でもう 1 本（値変化あり）
```

### 2-3. 新 `DSPCore` 生成との関係（P3-1-A の事実と接続）

- `RuntimeBuilder.cpp:469` は **build ごとに `aligned_make_unique<DSPCore>()`** を生成する。
- したがって **rebuild が完了すれば新しい DSPCore（＝新しい `SimplePeakLimiter`、`envelope = 1.0`）が
  処理に使われる**。
- `runCase` は capture 前に `waitBacklogZero(30000)` → `waitWorldPublished(seqBefore, 30000)` →
  `sleepPump(800)` を置くが、**`waitWorldPublished` の戻り値は無視**される（:417）。
  よって **rebuild が wait 窓内に完了したか否かは source だけでは確定できない**。
- **帰結（本 audit の要点）**: 「limiter `envelope` がケース間で持続する」という H6 の機構は、
  **ケース境界で rebuild が完了する場合には成立しない**（新 core で `envelope` は default 1.0 に戻る）。
  一方、**同一ケース内**（warm 4 blocks + capture 12 blocks）では `envelope` は持続する。

```text
H6 の「ケース間持続」: rebuild 完了タイミング依存 → source だけでは UNKNOWN（P3-1-A と同じ結論）
H6 の「同一ケース内持続」: source 上成立（warm blocks で gain reduction が起きれば capture 中も残留）
```

---

## 3. P3-1-B の実験条件（固定・未実施）

### 3-1. 2 sequence

```text
Pattern F（historical・既知）:
  p15eq os=8  →  g0/os1/am=-20 sc0  →  g0/os1/am=-6 sc0
  既知値: -20 = -14.5035 / -6 = -13.0276 / Δ = +1.4759 dB

Pattern R（reverse・新規）:
  p15eq os=8  →  g0/os1/am=-6  sc0  →  g0/os1/am=-20 sc0
```

- **IR reload topology を維持する**（各 `runCase` の前の `ensureTestIr()` を保持）。
- 変更は `adb` 初期化子リストの順序（`:746`）のみに限定する設計とする。
  ただし **本 audit では変更していない**。

### 3-2. 記録項目（P3-1-A の制約を反映）

| 項目 | 取得 |
| --- | --- |
| case id / order position / `ampDb` | ✅ |
| `gainDb_sc0` / `limitingEngaged_sc0` / `hardClamp_sc0` / `clipEngagement` / `clipEngMax` | ✅ |
| `os` / `n` | ✅ |
| IR geometry（`[IR_TAIL_GEOM]` / `[L0_WRITE]`） | ✅ |
| `F_scale`（`[IR_CHAIN] F_scale`） | ✅ |
| publication sequence / world generation / limiter envelope | ❌ **今回は取得しない**（P3-1-A のとおり。logger/accessor 追加は禁止） |

### 3-3. 判定規則（固定・順位付けなし）

| Pattern | 観測 | 分類 |
| --- | --- | --- |
| **1** | 順序反転後も `-20 ≈ -14.5035` / `-6 ≈ -13.0276` が維持 | `absolute-level dependent` |
| **2** | 値が直前ケースに追従（例: R の `-20` が直前 `-6` の状態に対応） | `state/history dependent` |
| **3** | 両順序で同一値 / `Δ ≈ 0` | `order-dependent hypothesis = not reproduced` |

- いずれも **原因確定ではない**。
- 3 値に明確に分類できない場合は `P3-1-B = INCONCLUSIVE` とし、**勝手に repeat しない**。
- 実行は **各 order 1 run**（Forward 1 / Reverse 1）。3 回反復は不要。

---

## 4. 停止条件との照合（実装可否）

| # | 停止条件 | 該当 | 備考 |
| --- | --- | --- | --- |
| 1 | targeted vehicle に production source が必要 | **非該当** | 変更は test TU の `adb` 順序のみ |
| 2 | `runCase` の settle 変更が必要 | **非該当** | 順序以外は不変 |
| 3 | `sleepPump` 変更が必要 | **非該当** | 不変 |
| 4 | `waitWorldPublished` 変更が必要 | **非該当** | 不変 |
| 5 | `gainDb` / `dftMag` 変更が必要 | **非該当** | 不変 |
| 6 | 新しい CLI flag が必要 | **非該当** | 不要 |
| 7 | limiter reset の production API が必要 | **非該当** | reset しない（介入禁止） |
| 8 | targeted vehicle が historical topology を維持できない | **非該当** | `ensureTestIr` / `configureChain` / settle / 2 パス構造を維持したまま順序のみ変更可能 |
| 9 | OS8→OS1 transition を省略する必要 | **非該当** | p15eq os=8 を直前ケースとして保持可能 |
| 10 | IR reload topology を省略する必要 | **非該当** | `ensureTestIr` を保持 |

- **停止条件 1〜10 はいずれも非該当** → reverse-order experiment は
  **test-only / topology 維持 / 新 CLI flag なし / settle 変更なし**で実装可能。
- P3-1-A で指摘した観測 3 項目の不足は**今回は埋めない**（意図的な制限）。

---

## 5. P3-1-B source audit の結論

```text
P3-1-B source audit = 完了（read-only・変更 0）
reverse-order experiment = 実装可能（停止条件 1〜10 非該当）
```

**確定事項**:

1. `am=-20 → am=-6` の順序は **`P1PolyphaseGainCharacterization.cpp:746` の初期化子リストのみ**で決まる。
   他に順序を決める要因は存在しない。
2. `ensureTestIr` は**各ケースの前**（sc0 前・sc1 前の計 2 回）、`configureChain` は**各ケースで 1 回**呼ばれる。
3. `setSoftClipEnabled(false)` は各ケースで呼ばれ、**変更検出ガードが無く必ず intent を投入する**。
4. `setOversamplingFactor(1)` は**毎ケース呼ばれるが no-op**（`!= newFactor` ガード）。
   実変化は `p15eq os=8 → g0/os1` の 1 回のみ。
5. `waitWorldPublished` の**戻り値は無視**される（timeout でも capture へ進む経路が実在）。
6. **intent 投入順序（確定）**: `setEqBypassRequested` → `setConvolverBypassRequested` →
   `setSoftClipEnabled` →（値変化があれば `setOversamplingFactor` / 他）→ `endBulkParameterRestore(true)`。
   前 3 者は同一 signature で **debounce 窓内に merge され 1 本に collapse**（dispatch 1 回）。
   `endBulkParameterRestore` は class が異なり別 dispatch。
   **am=-20 と am=-6 で intent の本数・順序は同一**（値変化に依存しない）。
7. **`RuntimeBuilder` は build ごとに新 `DSPCore` を生成**するため、rebuild が完了すれば
   limiter は新規（`envelope = 1.0`）になる。**H6 の「ケース間持続」は rebuild 完了タイミング依存で
   source だけでは UNKNOWN**。一方 **同一ケース内の持続は source 上成立**。

**判定（次段階への申し送り）**:

- reverse-order experiment 自体は実施可能。ただし **intent/rebuild 構造上、ケース境界で
  新 DSPCore が入りうる**ため、reverse-order の結果だけで H6 の「ケース間持続」を肯定/否定することは
  できない（**結果の解釈にこの制約を明記する必要がある**）。
- 本 audit の結果を提示した時点で停止する（指示どおり）。実験実装の可否は判断を仰ぐ。

## 6. 禁止事項の遵守

```text
production source 0 / DSPCore 0 / SimplePeakLimiter 0 / RuntimeBuilder 0 / ConvolverProcessor 0
AudioEngine 0 / CMake 0 / JUCE 0 / runCase settle 0 / waitBacklogZero 0 / waitWorldPublished 0
sleepPump 0 / dftMag 0 / gainDb formula 0 / IR compensation 0 / gain compensation 0
normalization 0 / new CLI flag 0 / kP15FullMatrix 変更 0 / CONVOPEQ_CORRECT_POLYPHASE_GAIN 0
limiter envelope ログ追加 0
```

- **コード変更・build・run は 0**（source audit のみ）。
