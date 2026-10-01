# STG-11-D8 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D8_REPAIR-CONTRACT-AUDIT_20260930.md`
- Work item: **STG-11-D8-1** — NaN IR length の無条件 store による IR 破壊
- Date: 2026-09-30
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20260930-144045).md` と同一）
- Commit / push: **禁止**

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

---

## 1. Defect statement

`ConvolverProcessor::setTargetIRLength` および `applyAutoDetectedIRLength`
（`ConvolverProcessor.Runtime.cpp:929,941`）は `juce::jlimit` の結果を
**無条件で** `pendingOverride.targetIRLengthSec` /
`autoDetectedIRLengthSec` に格納する。`jlimit` は NaN を素通しするため
（比較が偽 → 入力をそのまま返す）、破損 session の NaN `irLength` が
そのまま格納される。格納された NaN は `computeTargetIRLength`
（`ConvolverProcessor.StateAndUI.cpp:993`）で
`(int)(sampleRate * NaN)` → `INT_MIN` → `min(..., kMaxIRCap)` → `max(..., 1)` となり
**target 1 sample** に確定する。LoaderThread は IR を 1 サンプルに trim する。
ユーザーの IR が無言で破壊される（user data loss）。

## 2. Exact source location

| 要素 | file:line |
| --- | --- |
| 無条件 store（target） | `ConvolverProcessor.Runtime.cpp:936-937` |
| 無条件 store（auto） | `ConvolverProcessor.Runtime.cpp:954,964` |
| NaN 素通しの jlimit | 同 `:931,:945`（`juce::jlimit` は NaN 比較偽 → 入力返却） |
| session 入口 | `ConvolverProcessor.StateAndUI.cpp:353`（`setTargetIRLength(v.getProperty)`。var→float は NaN 保持） |
| 被害関数 | `ConvolverProcessor.StateAndUI.cpp:993-1020`（`computeTargetIRLength`） |
| 被害適用 | `ConvolverProcessor.LoaderThread.cpp:731`（`targetLength` に使用） |

## 3. Reproduction

```text
1. ValueTree(Convolver) に irLength = NaN(float) を設定
2. uiConvolverProcessor.setState(corrupted) を呼ぶ
3. getTargetIRLength() が NaN を返す（期待: 以前の有限値）
4. computeTargetIRLength(48000.0, N) が 1 を返す（期待: 有限の正数）
```

全段 deterministic（thread・OS・実機不要）。

## 4. Expected behavior

非有限の `irLength` は適用されず、以前の有限値が残る。
（`setMix` / `setSmoothingTime` / D8 gain 系と同一の「abs-diff gate により
NaN が格納されない」動作と一致する）

## 5. Actual behavior

NaN が格納され、`computeTargetIRLength` が 1 を返す。
`static_cast<int>(NaN)` は処理系定義（x86 で INT_MIN）だが、
どの結果でも `min`/`max` の後は 1 または巨大値のいずれかであり、
いずれも正しい IR 長ではない。

## 6. Root cause

jlimit を範囲安全と誤認した store パターン。
`setMix` 等は store 自体を abs-diff で gate しているため NaN が格納されないが、
`setTargetIRLength` / `applyAutoDetectedIRLength` は store が無条件であり、
notification の abs-gate だけでは格納を防げない。

## 7. Affected thread/context

- 格納：Message Thread（setState / UI）。
- 被害適用：LoaderThread / rebuild（NonRT）の trim 計算。RT DSP 自体は
  trim 済み IR を読むだけであり、RT 安全性の問題ではない。
- 被害：ユーザーの IR 音（無言の 1-sample 化）。

## 8. Data-integrity impact

破損 session の load が valid な IR 設定を破壊する（failure 時に old valid が残らない）。
正常 session の動作は不変。

## 9. DSP impact

trim 後 IR（1 sample）による無音化／原音乖離。数値発散・クラッシュではない。

## 10. Invariant violation

- User-data integrity（破損入力に対する old-valid 維持）。
- D8 §12 の failure semantics（invalid → old valid remains）違反。
- 同一 TU 内の `setMix` 等との不整合（NaN 扱いの非対称）。

## 11. Minimal repair

両 setter の store 前に有限性ガードを追加する
（既存 `convo::numeric_policy::isFinite` を使用。fp:fast 安全）：

```cpp
// setTargetIRLength:
if (!convo::numeric_policy::isFinite(clampedTime))
    return;  // 非有限は適用せず以前値を維持
// applyAutoDetectedIRLength: 同様（auto / target の両 store 前）
```

RT / Publish / Retire / Epoch / ownership の変更なし。新規 authority なし。
`computeTargetIRLength` 自体は変更しない（entry で保証されるため）。

## 12. Regression test

新規 harness サブテスト（新規 CTest target なし）：

- T1：NaN `irLength` の setState → `getTargetIRLength()` が有限のまま。
  `computeTargetIRLength(48000, N)` が 1 ではなく正数範囲内。
- T2：有効値 round-trip（1.5s 設定→保存→復元→一致）。
- T3：Inf（+Inf/-Inf）も拒否されること。
- Negative control：ガードを一時除去すると T1 が FAIL（NaN 格納＋target=1）。

## 13. Negative control

§12 T1 がその役割を果たす。旧 code では NaN が格納され `computeTargetIRLength==1`
で FAIL、修正後は PASS。両方向を実証する。

## 14. D1〜D7 との関係

- D1〜D5（retire/epoch/MMCSS）に触れない。D6 NO-GO と矛盾しない（別 class）。
- D7-1（enum cast）と同型の session 入力値問題だが、別 repair unit
  （enum 値域 vs float 有限性）。混ぜない。

## 15. Risk assessment

- Low。Message Thread 上の早期 return 追加のみ。正常値の動作は不変（T2 で実証）。
- 唯一の挙動変更：非有限 `irLength` の load 結果が「NaN 格納」から「維持」に変わる。
  これが期待動作（§4）である。
