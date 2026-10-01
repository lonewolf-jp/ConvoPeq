# STG-11-D9 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D9_REPAIR-CONTRACT-AUDIT_20261001.md`
- Work item: **STG-11-D9-1** — NaN tail state の無条件 store による NaN DSP 出力／形状破壊
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20260930-153609).md` と同一）
- Commit / push: **禁止**

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

---

## 1. Defect statement

`ConvolverProcessor::setTailStrength` / `setTailStartSec`
（`ConvolverProcessor.Runtime.cpp:1181,1161`）は `juce::jlimit` の結果を
**無条件で** `pendingOverride.tailStrength` / `tailStartSec` に格納する。
`jlimit` は NaN を素通しするため、破損 session の NaN がそのまま格納される。
格納値は `BuildSnapshot` → `LoadPipeline.cpp:683-684` の `FilterSpec` →
`MKLNonUniformConvolver.cpp:669-723` に流れ、同層の `jlimit` も NaN を素通しする。
結果：

- `tailStrength=NaN` → `userTailStrength` → `strength01` → `layer1Gain`/`layer2Gain` →
  `m_tailLayerGain[1..2]` = NaN → RT の `delayLineReadAdd(..., NaN)`（`:1933` 付近）→
  **NaN audio 出力**。
- `tailStartSec=NaN` → `l0LenByTailStart = (int)llround(NaN*sr)` = INT_MIN →
  layer geometry 歪曲、および `startNorm = jlimit(0.65,1.55,NaN)` = NaN →
  `dampingBase` = NaN → damping 係数が NaN → **NaN IR spectra**。

## 2. Exact source location

| 要素 | file:line |
| --- | --- |
| 無条件 store（strength） | `ConvolverProcessor.Runtime.cpp:1188`（`pendingOverride.tailStrength = clamped`） |
| 無条件 store（start） | 同 `:1168`（`pendingOverride.tailStartSec = clamped`） |
| session 入口 | `ConvolverProcessor.StateAndUI.cpp:378`（tailStrength）、`:377`（tailStartSec） |
| spec 転送 | `ConvolverProcessor.LoadPipeline.cpp:683-684` |
| NaN 素通し jlimit 群 | `MKLNonUniformConvolver.cpp:669-670,676,691,693,695-696,700-701,707-708` |
| 被害適用（gain） | 同 `:721-723` → `:1933` の `delayLineReadAdd` |
| 被害適用（形状） | 同 `:787`（`l0LenByTailStart`）、`:1175`（`startNorm`） |

## 3. Reproduction

```text
1. ValueTree(Convolver) に tailStrength = NaN(float) を設定
2. uiConvolverProcessor.setState(corrupted) を呼ぶ
3. getTailStrength() が NaN を返す（期待: 以前の有限値）
4. 同一条件で tailStartSec について同様
```

全段 deterministic。MKL 層の jlimit-NaN 素通しは C++ 比較意味論により確定
（テスト不要の言語規則。テストは格納拒否を検証する）。

## 4. Expected behavior

非有限の tail 値は適用されず、以前の有限値が残る
（D8 の IR length ガード、`setMix` の abs-gate と同一の意味）。

## 5. Actual behavior

NaN が格納され、下流の gain／形状計算が NaN 化する。
`m_tailLayerGain` の NaN は RT 出力を NaN 化する（ユーザー可聴の破壊）。

## 6. Root cause

D8 と同一パターン：jlimit を有限性保証と誤認した無条件 store。
`setTargetIRLength`（D8 修正済み）と同型の残存。

## 7. Affected thread/context

- 格納：Message Thread（setState / UI）。
- 被害適用：LoaderThread / rebuild（NonRT）の tail 計算、および RT の
  `delayLineReadAdd`（NaN gain 適用）。
- 被害：ユーザーの convolver 出力（NaN 化）。

## 8. Data-integrity / DSP impact

破損 session の load が valid な tail 設定を破壊し、NaN 出力を生む。
クラッシュではないが、ユーザー可視の DSP 破壊である。

## 9. Invariant violation

- User-data integrity（破損入力に対する old-valid 維持。D8 §12 と同一契約）。
- 同一 TU 内の `setMix` 等との不整合。

## 10. Minimal repair

両 setter の store 前に有限性ガードを追加する
（既存 `convo::numeric_policy::isFinite`）：

```cpp
// setTailStrength / setTailStartSec の冒頭:
if (!convo::numeric_policy::isFinite(static_cast<double>(strength/sec)))
    return;  // 非有限は適用せず以前値を維持
```

RT / Publish / Retire / Epoch / ownership の変更なし。新規 authority なし。

## 11. Regression test

新規 harness サブテスト（新規 CTest target なし）：

- T1：NaN `tailStrength` / `tailStartSec` の setState → 有限値維持。
- T2：有効 round-trip（1.0/0.2 設定→保存→復元→一致）。
- T3：+Inf / -Inf 拒否。
- Negative control：ガードを一時除去すると T1 が FAIL。

## 12. Negative control

§11 T1 がその役割を果たす。旧 code では NaN が格納され FAIL、修正後は PASS。

## 13. D1〜D8 との境界

- D1〜D5（retire/epoch/MMCSS）、D6 NO-GO、D7（enum）、D8（IR length）に触れない。
- D8 の再発見ではない（D8 は IR length。tail 系は D8 Gate §4 で範囲外記録したもの）。
- mixedF 系は LoaderThread の `validateBuffer`（`:830-871`）が NaN 出力を拒否するため
  対象外（中和を確認済み）。tailL1L2Multiplier は int-domain＋clamp のため対象外。

## 14. Risk assessment

- Low。Message Thread 上の早期 return 追加のみ。正常値の動作は不変（T2 で実証）。
- 唯一の挙動変更：非有限 tail 値の load 結果が「NaN 格納」から「維持」に変わる。
