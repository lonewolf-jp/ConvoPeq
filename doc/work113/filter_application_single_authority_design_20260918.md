# WORK113-10 — HC/LC Single-Application Contract & Implementation Design

- **作成日**: 2026-09-18
- **種別**: **Read-only / implementation design**（production 変更 0 / test-only 変更 0 / commit 0 / 実装 0）
- **前工程**: `doc/work113/filter_application_contract_reconciliation_20260918.md`（113-9）
- **保留**: 113-7 block sweep / WORK114（resample 比 7.6241）

---

## A. Source baseline

| 項目 | 値 |
|---|---|
| `ConvoPeq.md` | 5,276,742 bytes / 2026-09-18 22:33:38 |
| HEAD | `0654e7b5` |
| production changes | **0** |
| test-only changes | **0** |

113-9 のソース位置を最新ソースで再確認：

| 対象 | 位置（再確認） |
|---|---|
| `AudioEngine::setConvHCFilterMode` | `AudioEngine.Parameters.cpp:665-679`（`publishAtomic(convHCFilterMode)` + `uiConvolverProcessor.setNUCFilterModes`） |
| `AudioEngine::setConvLCFilterMode` | 同 `:681-696`（同じ fan-out） |
| `ConvolverProcessor::setNUCFilterModes` | `ConvolverProcessor.StateAndUI.cpp:864-887` → `postCoalescedChangeNotification()`（`:885`） |
| `FilterSpec` | `MKLNonUniformConvolver.h:123-133`（`hcMode`/`lcMode`/`tail*`/`sampleRate`） |
| `applySpectrumFilter` | `MKLNonUniformConvolver.cpp:361-468`（**全 layer** ループ `li=0..m_numActiveLayers-1`） |
| `OutputFilter::process` | `OutputFilter.cpp:199-...`（`convIsLast` 分岐 `:214`） |
| `convIsLast` | `AudioEngine.Processing.DSPCoreDouble.cpp:458-461` / `DSPCoreFloat.cpp:358-361` |
| `DSPCoreLifecycle` | `AudioEngine.Processing.DSPCoreLifecycle.cpp:390`（`outputFilter.reset()`） |
| `postCoalescedChangeNotification` | 定義 `ConvolverProcessor.Rebuild.cpp:21`（coalesced な rebuild 経路） |

**113-10 で新たに確認した追加事実（重要）**

1. **direct head は HC/LC 未適用**：`m_directIRRev[i] = impulse[m_directTapCount-1-i] * scale`（`MKLNonUniformConvolver.cpp:756`）。
   direct head の畳み込み（`:1419-1474`）は `impulse` の先頭タップを直接使い、`applySpectrumFilter` は
   `irFreqReal/Imag` しか変更しないため、**direct head 経路だけ HC/LC が掛からない**（`enableDirectHead=true` 時）。
   `kMaxDirectTaps = 32`（`:728`）。
2. `applySpectrumFilter` は **L0/L1/L2 すべて**に適用される（同関数の `li` ループ）。
3. `tailMode == 0` のとき、**L1/L2 の `irFreq` に別のゲイン（Air Absorption の HF tilt）**が追加適用される
   （`:1167-1203`）。これは HC/LC とは別の機能だが、同じ `irFreq` を触る（順序依存の確認対象）。
4. `FilterSpec.hcMode/lcMode` は `getStructuralHash()` に含まれる（`StateAndUI.cpp:913-914`）→
   **mode 変更は NUC 側の rebuild を要求する**。

---

## B. Current contract（as-is）

```text
UI: Convolver HC (Sharp/Natural/Soft) / LC (Natural/Soft)   [ConvolverControlPanel]
        ↓
AudioEngine::setConvHC/LCFilterMode
        ├─(1) atomic convHCFilterMode/convLCFilterMode  → OutputFilter::process(convIsLast=true)  … 時間領域 biquad
        └─(2) uiConvolverProcessor.setNUCFilterModes     → nucHCMode/nucLCMode → FilterSpec
                                                          → applySpectrumFilter  … IR 周波数領域 gain
        (+ direct head 経路: どちらのフィルタも掛からない)
```

**適用回数（現状）**：routing により **0 / 1 / 2** が発生し、direct head を含めると **3 経路**が存在。

| 現状の問題 | 内容 |
|---|---|
| 非等価 | NUC 版（ゼロ位相・振幅テーパ・LC=DC のみ）と OutputFilter 版（IIR biquad・LC=18 Hz HPF）が別演算 |
| 回数不定 | `EQThenConvolver` は 2 回、`ConvolverThenEQ` は 1 回（NUC のみ） |
| direct head | 0 回（未適用） |
| Disabled 不在 | HC/LC を無効化する正規手段がない |
| transition 差 | NUC=rebuild+crossfade / OutputFilter=係数差替え・状態継続 |

---

## C. Required target contract

```text
UI HC/LC intent
      ↓
[Single Canonical Filter Authority]
      ↓
exactly one application per convolution path
      ↓
audio output
```

### INV-FILTER-001
Convolver HC/LC のユーザー指定は **1 つの logical filter intent** として扱う。

### INV-FILTER-002
1 つの convolution processing path に対し、HC/LC の semantic filtering は **exactly once**。

### INV-FILTER-003
HC/LC の **mathematical transfer function は routing order によって変化してはならない**。

### INV-FILTER-004
`ConvolverThenEQ` と `EQThenConvolver` の違いは **EQ と Convolver の順序だけ**であり、
HC/LC が「存在する／消える」差を生じさせてはならない。

---

## D. Exactly-once invariant

`count(HC/LC semantic application)` を routing・bypass・layer・direct head の全組合せで定義し、
**valid convolution path（convActive=1）では常に 1**、conv bypass 時は 0（convolver が介在しないため）とする。

`count` は以下の独立経路の和：
```text
count = [NUC applySpectrumFilter] + [OutputFilter convIsLast branch] + [direct head] + [future dedicated stage]
```
現状：NUC=1（convActive 時）, OutputFilter=1 iff convIsLast, direct head=0（常に）, future=0。
→ INV-FILTER-002 は現状で **違反**（EQThenConvolver で 2）。

---

## E. Routing matrix（4 案比較）

`convActive/eqActive` と order、bypass を含む 7 行。値は HC/LC の適用回数（direct head=off 前提）。

| Routing | 現状 | A（IR 前処理） | B（partition 再設計） | C（OutputFilter canonical） | D（専用 ConvolverFilter） |
|---|---|---|---|---|---|
| Conv only | 1（NUC） | 1（IR） | 1（partition） | **1**（conv 出力） | **1**（専用段） |
| Conv → EQ | 1（NUC） | 1 | 1 | **1** | **1** |
| EQ → Conv | **2**（NUC+OF①） | 1 | 1 | **1** | **1** |
| Conv bypass | 0 | 0 | 0 | 0（※） | 0 |
| EQ bypass | 1 | 1 | 1 | 1 | 1 |
| Both bypass | 0（process 未呼出） | 0 | 0 | 0 | 0 |
| direct head 有効 | **0**（未適用） | 1（IR に含まれる） | 1 | direct も出力段に含まれ **1** | **1** |

（※）C 案で conv bypass 時に HC/LC を 0 にするか 1 にするかは**契約判断が必要**（Bypass 節 K 参照）。

理想：`HC/LC count = 1` が convActive=1 の全 valid path で成立。

---

## F. FilterSpec state ownership

```text
FilterSpec { sampleRate, hcMode, lcMode, tailMode, tailEnabled, tailStartSeconds, tailStrength, tailL1L2Multiplier }
   ↓  SetImpulse(filterSpec) → applySpectrumFilter のみが hcMode/lcMode を消費
   ↓
structural hash（nucHCMode/nucLCMode を含む）→ rebuild 判定
   ↓
snapshot（ConvolverProcessor）/ clone / crossfade / serialization（"nucHCMode"/"nucLCMode"）
```

- **持続状態は 2 系統**：AudioEngine `convHCFilterMode`/`convLCFilterMode` と
  ConvolverProcessor `nucHCMode`/`nucLCMode`（`ConvolverStateRoundTripTests` が round-trip を固定）。
- A/C/D 案では `FilterSpec` から HC/LC を除く（または未使用化する）必要が生じ得るが、
  **削除すると structural hash / snapshot / rebuild 判定 / serialization が変わる**。
  特に `nucHCMode/nucLCMode` を hash から外すと **mode 変更で rebuild が起きなくなる**（現状の crossfade 経路が消える）。
  → **FilterSpec state migration は G8 の未確定項目**。

---

## G. A/B/C/D design comparison（winner は決めない）

| 軸 | A: IR 前処理 | B: partition 再設計 | C: OutputFilter canonical | D: 専用 ConvolverFilter |
|---|---|---|---|---|
| 1. intended TF | ○（IR に 1 回） | ○（条件付き） | ○（出力に 1 回） | ○（出力に 1 回） |
| 2. OLS validity | ○（partition 前なので安全） | **条件**: `L_p ≤ P ⇔ L_f ≤ 1`、拡張 `N ≥ 2(P+L_f−1)` | ○（OLS 外） | ○（OLS 外） |
| 3. latency | △（フィルタ群遅延が IR 先頭・L1/L2 へ） | ○ | ○（biquad 数 sample） | ○（biquad 数 sample） |
| 4. CPU | △（load 時 IR 前処理・numPartsIR 増） | ○ | ○（RT に biquad 追加） | ○（同） |
| 5. memory | △（フィルタ後 IR 長に比例） | ×（FFT 拡張） | ○ | ○ |
| 6. API compatibility | ○ | ○ | ○（OutputFilter 既存） | △（新規段の追加） |
| 7. double application risk | 低（NUC から撤去前提） | 低 | 要明示（EQ 最終段の扱い） | 低（段を明示） |
| 8. mode transition | rebuild（既存経路） | rebuild | 係数差替え（reset/ramp 要設計） | 係数差替え（同） |
| 9. stereo | ch 毎 IR | ch 毎 | 既存 ch 独立 | 新規段で ch 独立 |
| 10. tail (L1/L2) | IR 全体なので自動で 1 回 | 全 layer で 1 回 | **Get 合算後に 1 回** | **合算後に 1 回** |
| 11. EQ ordering | 影響なし | 影響なし | **`convIsLast` と EQ HPF/LP の分離が必要** | 影響なし（段が固定） |
| 12. FilterSpec semantic | 「IR 前処理」へ再定義 | 現状維持＋条件 | HC/LC を撤去し 1 本化 | 撤去し専用段へ |
| 13. direct head | ○（IR に入るので自動） | ×（direct は未適用のまま） | ○（出力段なので自動） | ○（同） |
| 14. Disabled | 契約 gap | 契約 gap | 契約 gap | 契約 gap |

**A の追加論点**：IR 全体を時間領域で HC/LC すると **IR が長くなる** → `numPartsIR` 増 → L0 被覆と
tail 開始位置（`tailStartSec`）の意味が変わり得る（SR-02 の l0Len 上限式と相互作用）。
**C の追加論点**：単に「常時①」にすると **②（EQ 用 HPF20+LP）が失われる**か、
EQ 最終段で ① と ② が二重適用になる。**EQ の HPF/LP と conv の HC/LC を別段に分離する**必要がある。
**D の追加論点**：新規段は AudioEngine 側で convolver 直後に置く（order 非依存）。
`OutputFilter` は EQ 用（HPF20 + LP）に限定し、crossfade 権威は既存 coalesced 経路のまま。

---

## H. L0 / L1 / L2 / tail analysis

| 案 | L0 | L1 | L2 | tail 合算 | 回数 |
|---|---|---|---|---|---|
| 現状（NUC あり） | irFreq に HC/LC | irFreq に HC/LC | irFreq に HC/LC | `Get` で合算 | 合算後 1 回と等価（同一 gain のため）。ただし tailMode=0 の Air tilt が別途 L1/L2 に掛かる |
| A | IR 全体 → 自動 | 同 | 同 | IR に含まれる | **1** |
| B | partition 毎 | 同 | 同 | 合算 | 各 layer 1 だが線形性で合算 1 と等価（条件付き） |
| C | conv 出力段 | 同 | 同 | `Get` 合算後 | **1** |
| D | 専用段 | 同 | 同 | 専用段は `Get` 後 | **1** |

- C/D は **`Get` の合算（L0 ring + L1/L2 tail）に対して 1 回**掛かるため、tail を含めて exactly-once ✓。
- `tailMode==0` の Air Absorption tilt（L1/L2 のみ）は **HC/LC とは独立**として扱う（同一契約に混ぜない）。

---

## I. EQ ordering

```text
契約（案）:
  - HC/LC は「convolution output に対する 1 段」として routing order に依存せず適用する
  - EQ の HPF/LP（固定 20 Hz / 19-24 kHz）は EQ 段のフィルタとして独立に扱う
  - したがって order の違いは EQ と Conv の順序のみを変え、HC/LC の存在/特性を変えない
```
現状の `convIsLast` は「①=LC/HC」「②=EQ 用 HPF/LP」を排他選択しており、
**HC/LC の存在が order に依存**する（INV-FILTER-003/004 違反）。C/D はこの排他を解体する必要がある。

---

## J. Mode transition

| 項目 | 現状 | 契約候補 |
|---|---|---|
| NUC 側 | `setNUCFilterModes` → `postCoalescedChangeNotification` → rebuild + crossfade | A/B 案はこの経路を維持 |
| OutputFilter 側 | snapshot を毎 block 取得 → 係数差替え・**state 継続**（reset なし） | C/D 案は transition policy を明文化 |
| click 可能性 | OutputFilter 側で係数不連続により発生し得る | 要求: mode change → **one canonical filter state update** + **defined transition policy** |
| reset 要否 | 現状は prepare のみ | 「reset+ramp」or「crossfade」を 1 箇所で決定 |
| crossfade 決定主体 | 既存 coalesced 経路 | RT では判断しない（既存原則） → 既存 Authority を維持 |

**未確定**：C/D で「係数差替え＋状態継続」を許容するか、「reset して ramp/crossfade」するか。
既存原則「Crossfade 判定箇所は 1 箇所」「RT は判断しない」に整合させる必要がある。

---

## K. Bypass / Disabled semantics

- `HCMode = {Sharp, Natural, Soft}`、`LCMode = {Natural, Soft}` → **Disabled 値は存在しない**。
- `FilterSpec == nullptr` は `applySpectrumFilter` を止めるが、`tailMode=1`/`tailEnabled=true` を強制する
  （113-7C）→ 「HC/LC だけ無効」を意味しない。
- `convBypassed` は OutputFilter の分岐を切り替えるだけで、フィルタを除去しない。

→ **「HC/LC を無効化する」概念は現在の contract に存在しない**。
本 WORK では **Disabled を新設しない**。**`Not specified` として contract gap に残す**（G7）。

また C 案で conv bypass 時に HC/LC を適用するか（count 0 か 1 か）も **契約判断待ち**（E 表の※）。

---

## L. Crossfade / reset authority

| 対象 | 現状 Authority | 契約候補 |
|---|---|---|
| NUC rebuild/crossfade | `ConvolverProcessor` の coalesced 通知（単一） | 維持（A/B） |
| OutputFilter state reset | `DSPCoreLifecycle`（prepare 系）のみ | C/D では mode 変更時の transition も同じ 1 箇所に集約 |
| RT | 判断しない（実行のみ） | 維持 |

→ 「crossfade 判定は 1 箇所」「RT は判断しない」を満たすには、
**mode 変更時の係数差替えタイミングと ramp/crossfade を NonRT 側の 1 Authority に集約**する設計が必要。

---

## M. State / preset compatibility

| 保存対象 | 保管 | 影響 |
|---|---|---|
| `convHCFilterMode` / `convLCFilterMode` / `eqLPFFilterMode` | AudioEngine state（StateIO:219-221） | 既存 preset が保持 |
| `nucHCMode` / `nucLCMode` | ConvolverProcessor state（StateAndUI:261-262, 389-395） | round-trip テストで固定 |
| `getStructuralHash` | `nucHCMode/nucLCMode` を含む | mode 変更→rebuild の根拠 |

→ A/C/D で `nucHCMode/nucLCMode` を廃止・別機能化する場合、
**既存 preset の読み書き互換**と **rebuild 判定の代替**を同時に設計する必要（G8/G10）。

---

## N. Regression test design（実装後に必要）

**Transfer function**（入力正弦 or IR、`H(f)` を 0.5 dB 以内で比較）
```
20 Hz / 50 Hz / 100 Hz / 1 kHz / 10 kHz / 18 kHz / 22 kHz / Nyquist
```
**Routing**：Conv only / Conv→EQ / EQ→Conv / Conv bypass / EQ bypass
**Mode**：HC Sharp/Natural/Soft × LC Natural/Soft
**IR**：delta / short IR / real IR / long IR
**Partition**：numPartsIR = 1 / 3 / 32
**Channels**：mono / stereo
**Regression detector（今回の signature）**
```
f_in ∈ {40, 50, 60, 80, 100} Hz について
   |X(n·f_blk ± f_in)| を測定し、閾値以下（基本波の -80 dB など）であること
   f_blk は実測 fs / P から算出（固定 187.5 Hz に依存したテストにしない）
```
**Exactly-once 検証**：単一正弦入力で `H(f_in)` の振幅が
「意図した 1 回適用の理論値」と一致すること（例: `0.25 × H_total(50 Hz) = 0.2479`）。
**direct head**：`enableDirectHead=true` でも HC/LC が 1 回掛かること（現状は 0 回）。
**transition**：mode 変更時の click/段差（時間領域の不連続量）を測定。

---

## O. Implementation gates

| Gate | 条件 | 現状 |
|---|---|---|
| G1 | HC/LC semantic contract が明文化 | **○**（本 WORK §C, INV-001..004） |
| G2 | canonical authority が 1 つ | **PARTIAL**（C または D が候補、未選択） |
| G3 | 全 routing で exactly-once | **PARTIAL**（設計上は達成可能。C の bypass 方針未決） |
| G4 | L0/L1/L2/tail を含めて exactly-once | **○**（C/D/A で成立する設計を示した） |
| G5 | EQ ordering が明文化 | **PARTIAL**（§I の案は提示。既存 ② との統合が未決） |
| G6 | mode transition policy が明文化 | **PARTIAL**（reset/ramp vs crossfade 未決） |
| G7 | Disabled/bypass semantics が明文化 | **PARTIAL**（Disabled 不在を gap として記録。方針未決） |
| G8 | FilterSpec state migration が安全 | **UNKNOWN**（`nucHCMode/nucLCMode` の扱い未決） |
| G9 | crossfade/reset authority が既存原則と整合 | **PARTIAL** |
| G10 | 既存 preset/state compatibility が確認 | **PARTIAL**（2 系統の保存が存在。移行方針未決） |
| G11 | regression test matrix が定義 | **○**（§N。実装は未） |
| G12 | RT safety invariant に影響なし | **○**（C/D は biquad 追加のみ。NonRT 側で係数決定） |

**G1〜G12 のいずれかが PARTIAL/UNKNOWN のため production implementation は禁止。**

---

## P. Final verdict

```text
DESIGN_PARTIALLY_READY
```

**Proven（測定済み）**
- `R2 == NUC`（corr 0.9999999999985）、`R1 != R2`、`FilterSpec ON → sidebands`。
- `FilterSpec.hcMode/lcMode` は ConvolverProcessor の独立状態であり、AudioEngine setter が両経路へ fan-out する。
- direct head は HC/LC 未適用（第 3 経路）。
- stereo 起因説は棄却（状態は独立・対称）。

**Strongly supported**
- 「partition spectral filtering + 現行 OLS」が sideband operator を生成する（NUC 実装の異常ではない）。

**Not yet contract-proven**
- `OutputFilter` が唯一の canonical 実装であること（Level C 契約文が見つからない）。
- `applySpectrumFilter` を撤去して安全であること（**convIsLast=false で HC/LC が消失する**ため、単独撤去は不可）。

**設計として成立する形（提案の骨子・実装はしない）**
- 目標契約 INV-FILTER-001..004 を満たす最短路は、**HC/LC を「convolver 出力に対する 1 段」に固定**し
  （C または D）、NUC から `FilterSpec` の HC/LC を撤去、EQ の HPF/LP とは別段として扱うこと。
- `convIsLast` を「HC/LC を掛けるか」の判定に使わない（`applyConvHC_LC` と `convIsLast` を分離）。
- A 案（IR 前処理）は、IR 長変化が `numPartsIR`/tail semantics へ波及するため追加検討が必要。

**次工程**：`113-11 Implementation Plan`（G2/G3/G5/G6/G7/G8/G10 の決定を含む）。
それまで **production source は変更しない**。
