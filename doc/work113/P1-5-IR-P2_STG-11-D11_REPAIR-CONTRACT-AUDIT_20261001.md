# STG-11-D11 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D11_REPAIR-CONTRACT-AUDIT_20261001.md`
- Work item: **STG-11-D11-1** — NaN totalGain の無条件 store による NaN RT 出力
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20260930-231641).md` と同一。
  D10 マーカー 9 件・FRESH を作業開始時に確認）
- Commit / push: **禁止**

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

---

## 1. Defect statement

`EQProcessor::setTotalGain`（`EQProcessor.Parameters.cpp:103`）は
`juce::jlimit` の結果を無条件で `storeTotalGainDb`＋`EQState.totalGainDb` に格納する。
`jlimit` は NaN を素通しするため、破損 session の NaN `totalGain` がそのまま
格納される。格納された NaN は `prepareToPlay` の
`smoothTotalGain.setCurrentAndTargetValue(decibelsToGain(NaN))`
（`EQProcessor.Core.cpp:703`）でランプの current＋target を NaN 化し、
RT の gain ramp が NaN を出力する（NaN audio）。

なお `Processing.cpp:963` の abs-gate は `gainTarget(NaN) - targetGain(NaN)` の
比較が偽になるため更新を抑止するが、これは既に NaN 化した後の話であり、
prepare 時の直接設定を防げない。

## 2. Exact source location

| 要素 | file:line |
| --- | --- |
| 無条件 store | `EQProcessor.Parameters.cpp:103-108`（jlimit 後 `storeTotalGainDb`＋state 書き込み） |
| session 入口 | `EQProcessor.Core.cpp:581`（`setTotalGain(v.getProperty("totalGain"))`。var→float は NaN 保持） |
| ランプ NaN 化 | `EQProcessor.Core.cpp:701-703`（prepare 時の `setCurrentAndTargetValue`） |
| RT 被害適用 | `EQProcessor.Processing.cpp:961-973`（`startGain`/`increment`/gain ramp が NaN） |
| `LinearRamp` 意味 | `DspNumericPolicy.h:362`（`current = target = v`。NaN をそのまま保持） |

## 3. Reproduction

```text
1. ValueTree(EQ) に totalGain = NaN(float) を設定
2. eq.setState(corrupted) を呼ぶ（harness engine の getEQProcessor）
3. getTotalGain() が NaN を返す（期待: 以前の有限値）
4. prepareToPlay 後の smoothTotalGain が NaN（RT 出力が NaN）
```

全段 deterministic。RT 出力の NaN 化はコード trace により確定
（`setCurrentAndTargetValue` は無条件代入、`applyGainRamp` は乗算のみ）。

## 4. Expected behavior

非有限の `totalGain` は適用されず、以前の有限値が残る
（D8/D9 の有限性ガードと同一の意味）。

## 5. Actual behavior

NaN が格納され、prepare 後の RT gain ramp が NaN 化する。
`getState()` による再保存で NaN が永続化し得る（persistence corruption の連鎖）。

## 6. Root cause

D8/D9 と同一パターン：jlimit を有限性保証と誤認した無条件 store。
`setMix` 等の abs-gate 付き setter と異なり、`setTotalGain` の store 経路には
有限性条件が存在しない。

## 7. Affected thread/context

- 格納：Message Thread（setState / UI / prepare）。
- 被害：Audio Thread（RT gain ramp の NaN 出力）。

## 8. Data-integrity / DSP impact

破損 session の load が valid な gain 設定を破壊し、NaN 出力を生む。
再保存で NaN が永続化する連鎖あり。

## 9. Invariant violation

- User-data integrity（破損入力に対する old-valid 維持）。
- RT NaN-free（`processBand` の per-sample ガードは band 係数用であり、
  gain ramp 経路には適用されない）。

## 10. Minimal repair

`setTotalGain` の冒頭に有限性ガードを追加する
（既存 `convo::numeric_policy::isFinite`）：

```cpp
// ★ STG-11-D11: 非有限値の格納を拒否。jlimit は NaN を素通しするため、
//   破損 session の NaN が格納されると prepare 時のランプが NaN 化し
//   RT 出力が NaN 化する。以前値を維持する。
if (!convo::numeric_policy::isFinite(static_cast<double>(gainDb)))
    return;
```

RT / Publish / Retire / Epoch / ownership の変更なし。新規 authority なし。

## 11. Regression test

新規 harness サブテスト（新規 CTest target なし）：

- T1：NaN `totalGain` の setState → 有限値維持。
- T2：有効 round-trip（+6dB 設定→保存→復元→一致）。
- T3：+Inf / -Inf 拒否。
- Negative control：ガードを一時除去すると T1 が FAIL。

## 12. Negative control

§11 T1 がその役割を果たす。旧 code では NaN が格納され FAIL、修正後は PASS。

## 13. D1〜D10 との境界

- D1〜D5（retire/epoch/MMCSS）、D6 NO-GO、D7（別 TU の enum）、D8/D9（別 setter）に触れない。
- D10 の EQ enum 監査とは別 repair unit（enum 値域 vs float 有限性）。混ぜない。
- `nonlinearSaturation` の NaN は `saturation > 0.0` が偽になるため適用スキップ
  （中和を確認済み。対象外）。

## 14. Risk assessment

- Low。Message Thread 上の早期 return 追加のみ。正常値の動作は不変（T2 で実証）。
- 唯一の挙動変更：非有限 `totalGain` の load 結果が「NaN 格納」から「維持」に変わる。
