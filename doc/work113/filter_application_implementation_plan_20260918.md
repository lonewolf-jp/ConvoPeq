# WORK113-11 — HC/LC Single-Application Implementation Plan

- production changes: 0
- test-only changes: 0
- implementation: 0
- commit: 0

- **作成日**: 2026-09-18
- **種別**: Read-only / implementation plan（契約・変更境界・移行計画の確定のみ）
- **基準ソース**: `ConvoPeq.md` 5,276,742 B / 2026-09-18 22:33:38（HEAD `0654e7b5`）
- **前工程**: `doc/work113/filter_application_single_authority_design_20260918.md`（113-10）
- **保留**: 113-7 block sweep / WORK114（resample 比 7.6241）

---

## 0. Source-based 確認（本 WORK で追加確認した事実）

**DSPCore の実行順序（`AudioEngine.Processing.DSPCoreDouble.cpp:309-469` / Float 同型）**

```text
processInputDouble
oversampling.processUp（OS>1）
DC blocker（oversampled）
── routing ──
if (order == ConvolverThenEQ) {
    if (!convBypassed) convolverRt().process(block);      // ← conv 出力 = L0+L1+L2+tail+direct head
    eqRt().process(block);
} else { // EQThenConvolver
    eqRt().process(block);
    if (!convBypassed) { convolverInputTrim; convolverRt().process(block); }
}
── output filter ──
if (convActive || eqActive) {
    convIsLast = convActive && (!eqActive || order == EQThenConvolver);
    outputFilter.process(block, convIsLast, convHCMode, convLCMode, eqLPFMode);
}
scaleBlockFallback(outputMakeupGain)
```

**direct head は `Get()` 内で合算**（`MKLNonUniformConvolver.cpp:1905-1913` → `addFallback(toOut, m_directOutBuf)`）
→ **conv 出力に含まれる**ため、conv 出力に掛ける後段フィルタは direct head を自動的に包含する。

**テスト前提**
- `ConvolverStateRoundTripTests.cpp` は `nucHCMode/nucLCMode` の **round-trip のみ**を検証（mode change → rebuild の assertion は無い）。
- `PublishPipelineIntegrationTests.cpp` は rebuild 機構一般（SR 変更・reconfigure・admission）を検証（`nucHCMode` 起因の rebuild を要求しない）。

---

## 1. G2 — Canonical Authority の決定（source-based）

B の除外（再確認）：

```text
L_p = partition にフィルタ適用後の kernel 長 ≤ P + L_f − 1
2P-OLS valid-half が厳密 ⟺ L_p ≤ P ⟺ L_f ≤ 1
FFT 拡張: N ≥ 2(P + L_f − 1)   かつ  減衰しない DC 起因 floor は有限 N で不可（7F-0 実測）
→ B は implementation candidate から除外（NUC 内部再設計・メモリ・latency を要求し、根治保証もない）
```

| 評価軸 | A（IR 前処理） | C（OutputFilter as-is） | **D′（conv 出力段）** |
|---|---|---|---|
| required source changes | IR 前処理の追加（loader/trim 後）＋ NUC から HC/LC 撤去 | `convIsLast` の解体、② との統合 | **① の呼出位置を conv 直後へ移動（両分岐）＋ NUC の HC/LC 無効化** |
| runtime topology | IR→HC/LC→partition→NUC→EQ | conv→EQ→①/② 排他 | conv→**①**→EQ→**②**（②は EQ 段） |
| state ownership | IR に焼き込み（rebuild 依存） | OutputFilter ① | OutputFilter ①（conv サイトで駆動） |
| transition ownership | rebuild 経路（既存） | 係数差替え | 係数差替え（NonRT 公開） |
| preset compatibility | 影響あり（IR 前処理が変わる） | 既存キー維持 | **既存キー維持（変更不要）** |
| RT safety | load 時処理なので RT 追加なし | 既存同等 | **既存同等（biquad のみ）** |
| tail semantics | IR 長変化 → numPartsIR/tailStart 再導出が必要 | Get 合算に 1 回 | **Get 合算に 1 回（変更なし）** |
| direct-head semantics | 自動包含 | 自動包含 | **自動包含（Get 内合算）** |
| EQ ordering | 独立 | `convIsLast` と結合（要解体） | **conv 直後に固定 → order 非依存** |
| 契約充足 | 充足可能だが IR 長/ tail の再導出が未確定 | 充足には `convIsLast` 解体が必須（= D′化） | **充足（未知点最小）** |

**決定：canonical authority = 「conv 出力に対する single HC/LC stage」**。
実装形態は **既存 `OutputFilter` の ① 分岐（LC→HC0→HC1）を、convolver 呼出直後に適用**する（新クラス不要）。
`convIsLast` は **HC/LC の適用判定に使用しない**（禁止事項の遵守）。

> A は契約を満たし得るが、IR 長変化が `numPartsIR` / `tailStartSec` / SR-02 の l0Len 上限式へ波及し、
> 再導出が未確定なため **本 plan では採らない**（除外ではなく保留）。

---

## 2. G3 — Exactly-once の証明

構成：`count = [conv 出力段 HC/LC]`（NUC 側 HC/LC は撤去。direct head は conv 出力に含まれる）

| conv | EQ | order | direct head | count | 根拠 |
|---:|---:|---|---:|---:|---|
| 1 | 0 | — | 0 | **1** | conv 呼出後の ① が 1 回 |
| 1 | 1 | Conv→EQ | 0 | **1** | conv→①→EQ、① は 1 回 |
| 1 | 1 | EQ→Conv | 0 | **1** | EQ→conv→①、① は 1 回 |
| 1 | 0 | — | 1 | **1** | direct head は `Get` 合算 → ① が包含 |
| 1 | 1 | Conv→EQ | 1 | **1** | 同上 |
| 1 | 1 | EQ→Conv | 1 | **1** | 同上 |
| 0 | 1 | — | any | **0（契約定義）** | conv 出力が存在しないため HC/LC semantic も存在しない |
| 0 | 0 | — | any | **0** | 同上（full bypass） |

- ① は `if (!state.convBypassed)` の内部でのみ呼ぶ → conv=0 で count=0 が構造的に保証。
- L0/L1/L2/tail/direct-head は **`convolverRt().process(processBlock)` の出力に全て含まれる**ため、
  「conv 出力に 1 回」が全 contribution に対する exactly-once と同値。

---

## 3. G5 — EQ ordering の完全分離

**semantic stage の分離**

```text
Convolver HC/LC  = OutputFilter ①（LC → HC0 → HC1）   … conv 出力直後
EQ HPF/LP        = OutputFilter ②（HPF20 → LP0 → LP1）… EQ 出力（routing 位置に依らず EQ 段の一部）
```

| order | 直列順 |
|---|---|
| ConvolverThenEQ | conv → **①(HC/LC)** → EQ → **②(HPF20/LP)** → makeup |
| EQThenConvolver | EQ → conv → **①(HC/LC)** → **②(HPF20/LP)** → makeup |

- ①② は LTI であり、間の convolver も LTI のため、② の位置（EQ 後）は routing に依らず等価。
- **`convIsLast` は HC/LC の存在判定に使わない**。② の適用は `eqActive` で判定（EQ が無効なら EQ 段の HPF/LP も無効）。
- INV-FILTER-003/004（transfer function は order に依存しない / HC/LC の有無が order に依存しない）が成立。

**副次的に判明した対称な欠陥**：現行は `convIsLast=true`（EQThenConvolver）で **② が適用されない**
（EQ の固定 HPF20/LP が消失）。本 plan は ② を `eqActive` で判定するため、これも同時に解消する。

---

## 4. G6 / G9 — Mode transition / Crossfade authority

決定（policy）：

```text
UI → intent（convHCFilterMode / convLCFilterMode atomic）
   → NonRT: 新モードの biquad 係数を計算し、publish（既存 snapshot/StateIO 経路 = 単一 Authority）
   → RT: 係数を読むだけ（判断しない）
transition = coefficient replacement（state 保持・reset なし）
```

| 決定項目 | 決定 | 理由 |
|---|---|---|
| NUC rebuild | **行わない**（IR は mode 非依存になる） | mode はもはや IR に焼き込まれない |
| coefficient replacement | **採用** | ② 分岐で既に採用されている既存挙動と同一 |
| reset | **しない**（prepare 系のみ） | 二重 Authority を作らない |
| ramp | 係数差替えの過渡が基準を超えた場合の**唯一の緩和手段**として既存 ramp を使用 | 「crossfade 判定は 1 箇所」維持 |
| crossfade | 新設しない | 既存 coalesced/rebuild Authority を変更しない |

→ 「Crossfade 判定箇所は 1 箇所」「RT は判断しない」を満たす。
残る検証項目：「係数差替えの過渡量が受入基準（§13）以内か」を Phase 5 で実測。

---

## 5. G7 — Bypass / Disabled の明文化

```text
Case A: convActive = 1  → HC/LC は必ず存在し、exactly once
Case B: convActive = 0  → HC/LC count = 0
        （conv 出力が存在しないため HC/LC semantic も存在しない。② は EQ 段として別に適用）
Case C: Disabled 値     → 本 WORK では導入しない（enum 変更禁止）。
        `Not specified` として contract gap に残す（将来 WORK で扱う）。
```

- `FilterSpec == nullptr` は「HC/LC 無効」を意味しない（`tailMode=1`/`tailEnabled=true` を強制）→ **意味を変更しない**。
- `convBypassed` は HC/LC の count を 0 にする（Case B）— これを正規の bypass 意味論として固定。

---

## 6. G8 / G10 — FilterSpec / State / Preset migration

決定：**State 1 + State 2 の併用**（残すが HC/LC の authority から外す。serialization は維持）

| State | 現在 | target | migration |
|---|---|---|---|
| `convHCFilterMode` | AudioEngine atomic（② と同一値を供給） | **維持（唯一の intent 源）** | なし |
| `convLCFilterMode` | 同上 | **維持** | なし |
| `nucHCMode` / `nucLCMode` | FilterSpec 経由で IR に焼き込み。structural hash に含む | **保持（state/serialization はそのまま）／structural hash から除外／FilterSpec へ配線しない** | 既存 preset はそのまま読み書き可（inert 化） |
| `FilterSpec.hcMode/lcMode` | `applySpectrumFilter` が使用 | **メンバは残す（構造体互換）。使用しない** | 変更なし（未使用化） |
| `applySpectrumFilter` | HC/LC を適用 | **HC/LC を適用しない**（関数は Phase 2 で無効化。削除は本 plan の範囲外） | — |
| structural hash | `nucHCMode/nucLCMode` を含む | **除外**（mode change で IR rebuild しない） | rebuild 判定の代替は不要（IR は mode 非依存） |
| preset serialization | `convHC/LCFilterMode` + `nucHCMode/LCMode` | **両方維持** | old/new preset とも round-trip 不変 |

- **「mode change → rebuild」を担保していたもの**は `structural hash ← nucHCMode/nucLCMode`。
  HC/LC を IR から外した後は **rebuild は不要**であり、担保先は
  **`convHCFilterMode/convLCFilterMode` → NonRT 係数計算 → publish（② と同じ経路）** に移る。
- 既存テスト（`ConvolverStateRoundTripTests`: `nucHCMode/nucLCMode` round-trip、out-of-range clamp）は
  **メンバと serialization を残すため影響なし**。

---

## 7. direct head の正式契約（独立 Gate）

```text
現状: FFT/L0/L1/L2 → applySpectrumFilter（HC/LC）
      direct head   → raw impulse（HC/LC 0 回）
target: floor 出力 = L0 + L1 + L2 + tail + direct head  →  single HC/LC stage（exactly once）
```

- source 根拠：direct head は `MKLNonUniformConvolver::Get` 内で `addFallback(..., m_directOutBuf)` により
  **conv 出力へ合算**される（`:1905-1913`）。したがって conv 出力後の HC/LC stage が **無変更で包含**する。
- **direct-head 実装は変更しない**（禁止事項）。`enableDirectHead` の ON/OFF で
  HC/LC transfer function が同一であることを regression で検証（§13）。

---

## 8. L0/L1/L2/tail の再確認

| | 現状 | target（本 plan） |
|---|---|---|
| HC/LC 適用位置 | 各 layer の `irFreq`（L0/L1/L2） | conv 出力（`Get` 合算後）1 回 |
| 回数 | layer ごと 1（線形性で合算 1 と等価） | **1（合算後）** |
| `tailStartSec` / `numPartsIR` | HC/LC と独立 | **独立（変更なし）** |
| `tailMode==0` の Air tilt | L1/L2 の `irFreq` に別ゲイン | **変更なし（HC/LC とは別機能として維持）** |
| direct head | 0 回（欠陥） | 1 回（包含） |

A 案を採らないため、IR 長 → `numPartsIR` → L0/L1/L2 → `tailStart` の再計算は**不要**。

---

## 9. FilterSpec の扱い（決定）

```text
State 1（採用）: FilterSpec.hcMode/lcMode は残すが HC/LC semantic authority から外す
State 2（採用）: legacy/ABI/構造体互換のため保持（serialization は FilterSpec 自体を持たない）
State 3（不採用）: serialization からの削除は行わない（preset 互換を壊すため）
```

追跡：

| 項目 | old preset | new preset | round-trip | mode change | structural hash | rebuild |
|---|---|---|---|---|---|---|
| `convHC/LCFilterMode` | 読込→② と同一 intent | 同一 | 不変 | 係数差替え | 非依存 | 不要 |
| `nucHCMode/LCMode` | 読込（inert） | 同一 | 不変 | 影響なし | **除外** | **発生しない** |
| `FilterSpec.hcMode/lcMode` | 未使用 | 未使用 | — | — | — | — |

---

## 10. Implementation boundary（source symbol 単位）

| ファイル | 変更 | 理由 |
|---|---|---|
| `AudioEngine.Processing.DSPCoreDouble.cpp` | **変更**: `convolverRt().process()` 直後に `outputFilter.process(block, /*convIsLast=*/true, convHCMode, convLCMode, eqLPFMode)` を追加（両分岐）。出力位置の呼出は `/*convIsLast=*/false` 固定（② のみ、`eqActive` ゲート） | HC/LC を conv 出力直後に 1 回へ固定。`convIsLast` を HC/LC 判定から除去 |
| `AudioEngine.Processing.DSPCoreFloat.cpp` | **同上**（Float 経路。構造は同一） | 同上 |
| `OutputFilter.h/.cpp` | **変更しない** | ①/② の実装と per-channel 状態はそのまま利用（呼出位置のみ変更） |
| `AudioEngine.Parameters.cpp` | **変更**: `setConvHCFilterMode`/`setConvLCFilterMode` の fan-out のうち (2) `setNUCFilterModes` を停止（(1) の publish のみ残す） | HC/LC を IR に伝えない |
| `MKLNonUniformConvolver.cpp`（`applySpectrumFilter`） | **HC/LC 適用を停止**（Phase 2）。関数削除はしない | 二重適用の解消。B 案を採らないため IR 側は raw |
| `ConvolverProcessor.StateAndUI.cpp`（`setNUCFilterModes`） | **変更しない**（API と serialization は維持）。呼出側のみ停止 | state/preset 互換の維持 |
| `ConvolverProcessor.StateAndUI.cpp`（`getStructuralHash`） | **変更**: `nucHCMode/nucLCMode` の hashCombine を除去 | mode change での不要 rebuild を排除 |
| `ConvolverProcessor.h`（`FilterSpec`） | **変更しない**（メンバ残置） | 構造体互換 |
| `AudioEngine.StateIO.cpp` | **変更しない** | `convHC/LCFilterMode` が正本。`nucHCMode` は互換のため保存継続 |
| `AudioEngine.h` | **変更しない**（`outputFilter` をそのまま使用） | 新クラス不要 |
| `tests/AudioEngineHarness/*` | **変更（後続 Phase 7）**: regression matrix 追加 | §13 |
| `CMakeLists.txt` | **変更しない** | 新規 TU なし |

---

## 11. RT safety audit

| 原則 | 本 plan |
|---|---|
| no lock / no malloc / no delete | 変更なし（① は既存の固定バッファ biquad。呼出位置のみ移動） |
| no decision in RT | 係数は NonRT が決定・publish。RT は読むだけ（`convIsLast` の新規判断は追加しない） |
| Publish 経路は 1 本 | 既存 snapshot/StateIO 経路のみ（新経路を追加しない） |
| Crossfade 判定は 1 箇所 | 既存 coalesced/rebuild Authority を維持（crossfade 新設なし） |
| Retire 判定は 1 箇所 | 変更なし |
| atomic は wrapper 経由のみ | 遵守（`publishAtomic`/`consumeAtomic`） |

→ G12 充足。

---

## 12. 実装順序（設計のみ）

```text
Phase 0  Contract freeze（本 plan の INV-FILTER-001..004 と G1–G12 を凍結）
Phase 1  State ownership migration
         （convHC/LC = intent 正本の維持、nucHC/LC の hash 除外、fan-out 停止）
Phase 2  Filter application topology
         （conv 出力直後に ① を適用、applySpectrumFilter の HC/LC を無効化）
Phase 3  EQ ordering / convIsLast separation
         （② を eqActive ゲートで固定、convIsLast を HC/LC 判定から除去）
Phase 4  Direct-head inclusion（検証のみ。実装変更なし）
Phase 5  Transition / crossfade（係数差替えの過渡実測、必要時のみ既存 ramp を使用）
Phase 6  Preset compatibility（old/new preset round-trip 検証）
Phase 7  Regression matrix（§13 の受入条件を実装）
Phase 8  Release / Debug / RT validation（assert、診断ログ、RT safety 再確認）
```

---

## 13. Regression acceptance criteria

**測定定義**
- `f_blk = fs / P` を**実測値から算出**（固定 187.5 Hz を仕様に埋め込まない）。
- 入力正弦 `f_in ∈ {40, 50, 60, 80, 100} Hz`。

**A. Exactly-once（振幅）**
```
|A_meas(f_in) - 0.25 × H_total(f_in)| ≤ 0.5 dB
  H_total = |H_LC| × |H_HC|（OutputFilter 係数から独立に再計算）
```

**B. Distortion（block-rate sideband）**
```
|X(n·f_blk ± f_in)| ≤ A_meas(f_in) - 80 dB    (n = 1, 2)
```

**C. Direct head**
```
direct OFF と direct ON で |H(f)| が 0.5 dB 以内で一致（同一 HC/LC が 1 回）
```

**D. Routing（5 条件で A/B が成立）**
```
Conv only / Conv→EQ / EQ→Conv / Conv bypass / EQ bypass
  - conv bypass では HC/LC 指標を評価しない（count=0 の定義）
```

**E. Mode**
```
HC = Sharp / Natural / Soft,  LC = Natural / Soft
mode 変更後の A（exactly-once）が成立し、過渡量が既定閾値内
```

**F. その他**：IR = delta / short / real / long × numPartsIR = 1 / 3 / 32 × ch = mono / stereo。

---

## 14. 最終 Gate 判定

| Gate | 内容 | 判定 |
|---|---|---|
| G1 | HC/LC semantic contract が明文化 | **CONFIRMED** |
| G2 | canonical authority が 1 つ（conv 出力段＝OutputFilter ① @ conv 直後） | **CONFIRMED** |
| G3 | 全 routing で exactly-once（§2 の表で構造的に保証） | **CONFIRMED** |
| G4 | L0/L1/L2/tail を含めて exactly-once（`Get` 合算後 1 回） | **CONFIRMED** |
| G5 | EQ ordering が明文化（①/② を semantic 分離、`convIsLast` を判定に使わない） | **CONFIRMED** |
| G6 | mode transition policy が明文化（係数差替え・state 保持・NonRT 判断） | **CONFIRMED** |
| G7 | Bypass/Disabled semantics が明文化（conv=0 ⇒ count 0、Disabled は gap として明示） | **CONFIRMED** |
| G8 | FilterSpec state migration が安全（State 1+2、hash 除外、既存テスト影響なし） | **CONFIRMED** |
| G9 | crossfade/reset authority が既存原則と整合（crossfade 新設なし・RT 判断なし） | **CONFIRMED** |
| G10 | 既存 preset/state compatibility（両キー維持 → 移行不要） | **CONFIRMED** |
| G11 | regression test matrix が定義（§13 の受入条件） | **CONFIRMED** |
| G12 | RT safety invariant に影響なし（§11） | **CONFIRMED** |

**判定：`DESIGN_READY`**

- ただし本 plan は **設計の確定のみ**。実装は Phase 0 以降の工程として別途開始する。
- 実装開始時に検証すべき残項目（判定を覆さない verification item）：
  1. 係数差替え時の過渡量（Phase 5 実測）
  2. `applySpectrumFilter` 無効化後に IR が raw になることの確認（`[L0_WRITE]`/`[7C]` 系トレースで irFreqEnergy がフィルタ前値に一致）
  3. `nucHCMode/nucLCMode` を hash から外した後、既存の PublishPipeline 系テスト（D167 等）が
     想定外の rebuild 回数変化を起こさないことの確認
