# STG-11-D7 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D7_REPAIR-CONTRACT-AUDIT_20260930.md`
- Work item: **STG-11-D7-1** — session 復元時の未検証 enum cast による RT 配列 OOB
- Date: 2026-09-30
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20260930-132503).md` と同一）
- Commit / push: **禁止**

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

---

## 1. Defect statement

`AudioEngine::requestLoadState`（`AudioEngine.StateIO.cpp`）は session の
`convHCFilterMode` / `convLCFilterMode` / `eqLPFFilterMode`
（＋ `processingOrder` / `analyzerSource`）を範囲検証なしに enum へ cast し、
atomic に publish する。範囲外値（破損・他者改変・将来 version の preset）は
そのまま runtime 状態になる。filter mode 3 種は RT の `OutputFilter::process` で
**配列 index として直接使用**され、bounds check がないため、範囲外値で
OOB read（誤係数による無言の DSP 破壊、最悪 AV crash）が発生する。

## 2. Exact source location

| 要素 | file:line |
| --- | --- |
| 未検証 cast 4 件 | `AudioEngine.StateIO.cpp:165`（analyzerSource）、`:169`（convHC）、`:171`（convLC）、`:173`（eqLPF） |
| 未検証 cast（order） | `AudioEngine.StateIO.cpp:40`（processingOrder） |
| setter（正規化なし・直接 publish） | `AudioEngine.Parameters.cpp:698-730`、`AudioEngine.h:1394` |
| RT index 使用（bounds なし） | `AudioEngine.OutputFilter.cpp:226-228`（`hcCoeff[hcIdx]` / `lcCoeff[lcIdx]` / `lpCoeff[...]`。配列は `[3][2]` / `[2]` / `[3][2]`） |
| 未使用の validator | `AudioEngine.Parameters.cpp:74` `validatePresetStateTreeForDebug`（caller 0 件） |

## 3. Reproduction sequence

```text
1. 有効な preset ValueTree を取得（getCurrentState）
2. convHCFilterMode property に範囲外値（例: 5）を設定
3. requestLoadState(corrupted) を呼ぶ
4. getConvHCFilterMode() が HCMode(5) を返す（期待: Natural(1) のまま）
5. OutputFilter::process(..., hcMode=(HCMode)5, ...) が hcCoeff[5] を読む → OOB
```

全段 deterministic（OS 注入不要・thread 不要・実機不要）。

## 4. Expected behavior

範囲外 enum 値は適用されず、現在の runtime 値（デフォルト）が維持される。
これは `noiseShaperType` / `oversamplingType` に対する work92 B-3 の確定動作と同一である。

## 5. Actual behavior

範囲外値がそのまま atomic に publish され、RT の配列 index になる。
`hcCoeff[5]` は `hcCoeff[3][2]`（24 BiquadCoeff）の範囲外であり、
隣接メンバ（`lcCoeff` / `hpfCoeff` / `lpCoeff` / state）の誤読（無言の誤フィルタ）
または AV 違反になる。

## 6. Root cause

B-3（big 2-10）と同一の欠陥パターンの残存。B-3 では `noiseShaperType` /
`oversamplingType` に範囲ガードが追加されたが（`StateIO.cpp:99-105,136-142`）、
同ファイルの `analyzerSource` / `convHCFilterMode` / `convLCFilterMode` /
`eqLPFFilterMode` / `processingOrder` は未ガードのまま残った。
範囲 validator（`validatePresetStateTreeForDebug`）は存在するが caller 0 件であり、
`requestLoadState` から呼ばれていない。

## 7. Affected thread/context

- 復元：Message Thread（`requestLoadState`。`setStateInformation` は非 MT 時に
  `callAsync` で MT へ marshaling する）。
- 被害：Audio Thread（RT）。`OutputFilter::process` は audio callback から呼ばれる。
- 保存側（`getCurrentState`）は current runtime 値のみ書くため、破損値の発生源は
  外部 session（破損・改変・将来 version）のみ。正常 session では発火しない。

## 8. Ownership/lifetime impact

なし（整数 enum の値域問題。ownership 輸送・lifetime 変更を伴わない）。

## 9. Invariant violation

- Practical Stable ISR「RT は判断しない」：RT が不正 index を判断できず OOB する。
- User-data integrity：破損 session の load が valid runtime を破壊する
 （failure 時に old valid state が残らない）。
- B-3 確定契約との不整合：同一ファイル内の同型 6 箇所のうち 2 箇所のみガード済み。

## 10. Minimal repair

B-3 と同一パターンの範囲ガードを 6 箇所に追加する（`StateIO.cpp` のみ。
新規 authority / queue / worker / member なし）：

```cpp
if (state.hasProperty("processingOrder"))
{
    const int raw = static_cast<int>(state.getProperty("processingOrder"));
    if (raw >= static_cast<int>(ProcessingOrder::ConvolverThenEQ)
        && raw <= static_cast<int>(ProcessingOrder::EQThenConvolver))
        ...publish...;
}
// analyzerSource / convHC / convLC / eqLPF も同様（既存 setter を呼ぶ）
```

`processingOrder` は 2 箇所の atomic publish を guard 内に移動するだけで意味不変。
`analyzerSource` は UI 比較のみのため実害は軽微だが、同型欠陥として同時修正する
（1 defect / 1 repair unit = 未検証 session enum cast）。

RT 側（`OutputFilter::process`）の clamp は行わない。
復元ガードにより到達不能になるためであり、RT 差分を最小化するためである。

## 11. Regression test

新規 TU（harness サブテスト。新規 CTest target なし）：

- T1：破損 ValueTree（6 property に範囲外値）→ `requestLoadState` → 全 mode が
  デフォルトのまま（`Natural` / `ConvolverThenEQ` / `Output`）。
- T2：有効 ValueTree の round-trip（既存動作の非回帰。`getCurrentState` → 改変なし →
  `requestLoadState` → 値が一致）。
- T3：境界値（min/max は適用、min-1/max+1 は拒否）。
- Negative control：一時的にガードを外すと T1 が FAIL すること。

## 12. Negative control

§11 T1 がその役割を果たす。旧 code（guard なし）では範囲外値が適用されて FAIL、
修正後は PASS。両方向を実証する。

## 13. D1〜D6 との関係

- D1〜D5 の production 変更（retire/epoch/MMCSS 観測）に触れない。
- D6 の NO-GO（ownership/RT 系に concrete なし）と矛盾しない（別 defect class）。
- B-3（work92）の確定契約を拡張するものであり、再設計ではない。

## 14. Invariant impact

変更は Message Thread 上の分岐追加のみ。RT / Publish / Retire / Epoch /
ownership に影響なし。性能影響なし（復元時のみの整数比較）。

## 15. Risk assessment

- Low。既存 B-3 パターンの逐語的拡張。正常 session の動作は不変
 （範囲内値は従来どおり適用。T2 で実証）。
- 唯一の挙動変更：範囲外値を含む session の load 結果が「不正値適用」から
  「デフォルト維持」に変わる。これが期待動作（§4）である。
