# STG-11-D10 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D10_REPAIR-CONTRACT-AUDIT_20261001.md`
- Work item: **STG-11-D10-1** — EQ band type/channel の未検証 enum cast による無言の band 無効化
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20260930-231641).md` と同一。
  D9 マーカー 9 件・FRESH を作業開始時に確認）
- Commit / push: **禁止**

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

---

## 1. Defect statement

`EQProcessor::setBandType` / `setBandChannelMode`
（`EQProcessor.Parameters.cpp:159,183`）は session の int 値を範囲検証なしに
enum へ cast し、`EQState` に格納する。範囲外値（破損・改変・将来 version の
session）はそのまま runtime 状態になる。下流では：

- `bandTypes` OOB → `calcSVFCoeffs` の switch がフォールスルーし `return {}`
  （ゼロ係数）→ 当該 band が無音化。
- `bandChannelModes` OOB → `Processing.cpp` の等価 chain のいずれにも一致せず
  当該 band が無処理（無音化）。

RT の NaN ガード（`processBand` の per-sample clamp＋state reset）により
NaN 伝播・クラッシュは起きないが、破損 session の load が valid な band 設定を
無言で無効化する（誤った DSP 動作。D10 §4 の認定条件に該当）。

## 2. Exact source location

| 要素 | file:line |
| --- | --- |
| 未検証 cast（type） | `EQProcessor.Parameters.cpp:166`（`newState->bandTypes[band] = type`） |
| 未検証 cast（channel） | 同 `:190`（`newState->bandChannelModes[band] = mode`） |
| session 入口 | `EQProcessor.Core.cpp:594`（`setBandType(i, (EQBandType)(int)...)`）、`:595`（channel） |
| 下流（type） | `EQProcessor.Coefficients.cpp:40`（switch フォールスルー → `return {}`） |
| 下流（channel） | `EQProcessor.Processing.cpp:680-745,776-796`（等価 chain 不一致 → 無処理） |
| 封じ込め（RT NaN ガード） | `EQProcessor.Processing.cpp:128-189`（per-sample clamp＋reset） |

## 3. Reproduction

```text
1. 有効な EQ state を取得（getState）
2. Band 0 の type に範囲外値（例: 99）を設定
3. uiEqEditor または engine 経由で setState(corrupted) を呼ぶ
4. getBandType(0) が (EQBandType)99 を返す（期待: 変更前値）
```

全段 deterministic。band index 自体は範囲検査済み（`0 <= i < NUM_BANDS`）のため、
OOB は type/channel 値に限定される。

## 4. Expected behavior

範囲外の type/channel は適用されず、現在の値が維持される
（D7 の filter mode ガードと同一の意味）。

## 5. Actual behavior

範囲外値が格納され、当該 band が無音化する（ゼロ係数／無処理）。
`getState()` で再保存すると破損値が永続化される。

## 6. Root cause

D7 と同一パターンの残存：session enum の未検証 cast。
EQ band の float（freq/gain/q）は `calcSVFCoeffs` 系の `isfinite` ガードで中和され、
int-domain の index は検査済みだが、type/channel enum だけが無検証だった。

## 7. Affected thread/context

- 格納：Message Thread（setState / UI）。
- 被害：Audio Thread の band 処理（無音化）。NaN 伝播・クラッシュなし
  （RT ガードが封じ込めることを確認済み）。

## 8. Data-integrity / DSP impact

破損 session の load が valid な band 設定を無言で無効化する。
`getState()` による再保存で破損が永続化し得る。

## 9. Invariant violation

- User-data integrity（破損入力に対する old-valid 維持）。
- D7 確定契約との不整合（同型 2 箇所の残存）。

## 10. Minimal repair

両 setter に範囲ガードを追加する（D7 パターン）：

```cpp
// setBandType:
if (type != EQBandType::LowShelf && type != EQBandType::Peaking
    && type != EQBandType::HighShelf && type != EQBandType::LowPass
    && type != EQBandType::HighPass)
    return;
// setBandChannelMode: Stereo/Left/Right/Mid/Side の同様のガード
```

RT / Publish / Retire / Epoch / ownership の変更なし。新規 authority なし。
`setFilterStructure`（正規化済み）は触れない。

## 11. Regression test

新規 harness サブテスト（新規 CTest target なし）：

- T1：範囲外 type/channel の setState → 変更前値維持。
- T2：有効 round-trip（全 5 type×全 5 channel の設定→保存→復元→一致）。
- T3：境界値（min/max 適用、min-1/max+1 拒否）。
- Negative control：ガードを一時除去すると T1 が FAIL。

## 12. Negative control

§11 T1 がその役割を果たす。旧 code では範囲外値が格納され FAIL、修正後は PASS。

## 13. D1〜D9 との境界

- D1〜D5（retire/epoch/MMCSS）、D6 NO-GO、D7（別 TU の enum）、D8/D9（float 有限性）に触れない。
- D7 の再報告ではない（D7 は AudioEngine.StateIO の 6 件。EQ band は未監査域）。
- EQ float（freq/gain/q）は SVF 系の isfinite ガードで中和済みのため対象外。

## 14. Risk assessment

- Low。Message Thread 上の早期 return 追加のみ。正常値の動作は不変（T2 で実証）。
- 唯一の挙動変更：範囲外 type/channel の load 結果が「不正値適用」から「維持」に変わる。
