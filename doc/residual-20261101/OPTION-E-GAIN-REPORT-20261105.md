# 案 E 利得規約・音声品質調査報告（2026-11-05）

- 基準: `main` = `f90f56be`、`ConvoPeq.md` `ac2d5f76…`（FRESH）
- 制約: 案 E の採用・既定 ON は未承認。既定 OFF を変更していない。本報告は測定と解析のみ。

## 1. 現行の利得規約

```text
prepareStage: center = 0.5 / 非 center = 0.5 / sum = 1.0（係数和の契約）
interpolateStage:
    convValue   *= 2.0                       // 常時
    centerValue *= 2.0                       // フラグ ON のみ
decimateStage: centerCoeff * centerSample（×2 なし）
```

- 分岐係数和は conv 1.0、center 0.5 で不対称。
- 往復 DC = 0.75^N（N は段数）。補償は production に存在しない。

## 2. 実測（現行 OFF ビルド、icx 2026.2.0）

`PolyphaseGainFidelityTests` を実行し、54 項目すべて PASS。

| 項目 | 実測 |
|---|---|
| DC（IIR3/LP3 × r=2/4/8） | 0.750000 / 0.562500 / 0.421875（= 0.75^N） |
| DC（candidate） | 1.000000（全構成） |
| impulse S1(31/90) | Σ base 0.750000 / cand 1.000000、peak offset 15 |
| P0-F latency ×4 | argmax ∈ [floor(D), floor(D)+1]、base == cand（bitwise） |
| P0-I SoftClip OS DC（A=0.5） | base 0.375000 / cand 0.500000、clamp 0、tanh clamp 18200 |
| P0-G float/double | maxAbs 2.58e-08 / 2.96e-08、maxRel 1.84e-05 |
| D2（画像） | −223.31 / −217.69 / −184.26 dB（linear 経路） |

## 3. ON ビルドでの確認

`build-icx-on`（`-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`）で同じテストを実行し、OFF との差を記録する。

| 項目 | OFF | ON |
|---|---|---|
| DC（IIR3/LP3 × r=2/4/8） | 0.75 / 0.5625 / 0.421875 | **1.0**（全 6 構成） |
| テスト合計 | PASS=54 FAIL=0 | **PASS=19 FAIL=35**（OFF 期待値に対する差＝正の対照） |
| P0-I SoftClip OS DC（A=0.5） | 0.375 | **0.5** |
| P0-G float/double | 2.58e-08 / 2.96e-08 | 3.31e-08 / 2.96e-08 |
| P0-F latency（S1 31/90） | PASS（D=15） | S1 は PASS。LP3 は bitwise 契約（base==cand）が ON で不一致のため FAIL |
| hard clamp / tanh clamp | 0 / 18200 | 0 / 0（この battery 条件） |

ON ビルドは Phase 0 の「OFF を基準にした期待値」を多数含むため FAIL するのが設計どおりで、P1-1 の記録（ON 19/54）と一致した。

## 4. 影響範囲

| 対象 | 影響 |
|---|---|
| 既定出力レベル | OS 倍率に依存して +2.4988 / +4.9975 / +7.4963 dB（8x が最大） |
| SoftClip drive | 0.0018〜0.0687 → 0.0182〜0.1473（5〜10 倍） |
| limiter 到達 | 0 → 119/4096（os≥2・0 dBFS） |
| hard clamp | 0（変化なし） |
| latency | 不変（4 経路） |
| float/double | 一致（2.96e-08） |
| preset 保存値 | 不変（makeup/headroom/trim/OS は独立属性） |
| preset の出力レベル | 変わり得る（未測定） |
| EQ boost / IR boost 併用 | 未測定 |

## 5. 採用判断に必要な残データ

1. 聴感評価（N2）
2. 既存プリセットの replay 比較（N4）
3. EQ boost／IR boost 併用時の挙動（N3）
4. 設計意図の確認（N1。0.75^N が仕様かどうか）

## 6. 結論

- 数値（0.75^N、+2.5/+5.0/+7.5 dB、SoftClip DC 0.375→0.5）は現行コードの実測で再確認できた。
- ただし「バグか設計意図か」は未確定であり、採用は音声仕様の変更にあたる。
- 推奨は **保留（既定 OFF 維持）**。採用する場合は、上記 1〜4 を埋めたうえで Owner 承認を得る。
