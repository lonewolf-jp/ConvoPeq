# ConvoPeq 統合バグリスト兼バグ改修計画書 — 実装準備視点版

- **日付**: 2026-09-13
- **版**: v1.3（網羅性・改修設計・実装準備視点）
- **前版**: v1.2（反証・改修副作用視点）
- **本版の視点**: ①網羅性の欠落 ②改修設計の代替案 ③実装準備度（テスト・段階リリース・ロールバック）

---

## 0. v1.3 で新しく判明した事項

| 区分 | 内容 |
|------|------|
| **テスト欠落** | mix=0.5 dry/wet null test・`estimatePeakLatencySamples` ユニットテストが**存在しない** |
| **代替案** | SR-01 はメモリ倍増ではなく **IR 上限の動的化**でも解決可能 |
| **格上げ** | M-04 は「素通し」ではなく **前回 callback の stale データ残存** |
| **実装順序** | H-02 → SR-03 の順が必須。逆にすると無駄なバッファ拡張になる |
| **網羅性限界** | 本監査はコンボルバー経路が主対象。EQ / Oversampler / NoiseShaper のレート依存は未監査 |

---

## 1. 網羅性の欠落

### 1.1 テストカバレッジ（実装準備のボトルネック）

| 改修 ID | 必要テスト | 現状 |
|---------|------------|------|
| H-01 | mix=0.5 dry/wet 交差相関 null test | **なし** |
| H-02 | `estimatePeakLatencySamples` ユニットテスト | **なし** |
| SR-01 | 768 kHz 3s IR ロード / スナップショット復元 | **なし** |
| SR-02 | 768 kHz / bs=512 で L0=85ms | work57 にあるが**期待値が変わる** |
| SR-03 | 遅延 > 2s でのクロスフェード整合 | **なし** |

既存テスト:
- `MT-NUPC-Measurement.cpp` — NUC 構造 / M1/M2/M3（H-01/H-02 に非依存）
- `GainStagingContractTests.cpp` — ゲイン契約
- `EQProcessorMaxGainTests.cpp` — EQ

**結論**: Phase 1 着手前に **H-01 / H-02 の回帰テストを先に書く**のが安全。

### 1.2 監査範囲外（本リストに含めないが将来課題）

| 領域 | 状態 |
|------|------|
| EQ プロセッサのレート依存 | 未監査（`EQProcessor.Coefficients.cpp` に sr>384k 上限削除の修正履歴あり） |
| Oversampler のレート依存 | OversamplingPolicy で整合確認済みだが詳細未監査 |
| Noise Shaper | 768 kHz 係数あり（`FixedNoiseShaper.h`）。整合は未監査 |
| SoftClip 局所 OS | `Latency.cpp` に「要実測確認」コメントあり |

---

## 2. 改修設計の代替案

### 2.1 SR-01 — メモリ倍増 vs 動的上限

| 方案 | 内容 | メリット | デメリット |
|------|------|----------|------------|
| **A（現案）** | `MAX_IR_LATENCY` を 2^22 へ。`DELAY_BUFFER_SIZE` も 2^23 へ | 3s を全 SR で保証 | 768 kHz で **+134 MB** |
| **B（推奨）** | `IR_LENGTH_MAX_SEC` を SR 依存に動的化。`min(3.0, MAX_IR_LATENCY/sr)` | メモリ増なし。現行定数を維持 | 768 kHz では 2.73 s が上限（UI に明示） |
| **C（折衷）** | `MAX_IR_LATENCY` は維持。UI/警告のみ整備 | 最小変更 | 3s 保証を諦める |

**推奨**: **方案 B**。  
「3 s を全 SR で保証する」要件がなければ、メモリ 134 MB を払う必要はない。  
`getMaximumAllowedIRLengthSecForSampleRate` は既に存在し、**applySnapshot と UI に適用するだけ**。

### 2.2 H-01 — dry 遅延の切り分け

| 方案 | 内容 | リスク |
|------|------|--------|
| **A（現案）** | 内部 dry 遅延から `algorithmLatency` を外す | mix 聴感変化 |
| **B** | `m_latency` 自体を 0 にし host PDC も変える | **host PDC が壊れる**（非推奨） |
| **C** | dry 遅延を `irPeak` のみにし、host PDC は `algorithmLatency+irPeak` のまま | A と同一。明確化 |

**推奨**: **方案 C**（A の明確化版）。host PDC を触らない。

### 2.3 SR-02 — l0MaxLen の計算式

| 方案 | 内容 |
|------|------|
| **A（現案）** | `l0MaxLen` を秒ベースで計算し partSize の倍数に切り上げ |
| **B** | `kL0MaxParts` を SR 依存に拡大（例: `max(32, ceil(0.085*sr/l0Part))`） |
| **C** | tailStart のユーザー設定時に l0MaxLen でクランプした値を UI に表示 | 

**推奨**: **方案 A + C**（計算式修正 + ユーザー可視化）。

---

## 3. 重大度の再調整

### 3.1 M-04 の格上げ（Medium → Medium-High）

```593:594:src/convolver/ConvolverProcessor.Runtime.cpp
if (activeSmoothingCapacity < numSamples)
    return;   // ← block を一切書き換えない
```

**v1.2 までは**「素通し」と記載。  
**実際には** `process()` の呼び出し元が既に dry 遅延バッファへ入力を書き込み済みで、**出力 block には前回 callback の値が残る**（stale data）。

| 状況 | 影響 |
|------|------|
| prepare 済みで smoothing バッファ未確保 | 前回出力のリピート = **周期的ノイズ** |
| mix 変更の最中に capacity 不足 | クロスフェードが途切れる |

**判定**: Medium-High に格上げ。改修案は dry 転送フォールバックで確定。

### 3.2 SR-03 の格下げ検討（High → Medium-High）

H-02 修正後は `irPeakLatency` が真のピーク（通常 < 1000 サンプル）になり、**2 s 整合バッファを超える確率はほぼ消える**。

| 条件 | SR-03 発生確率 |
|------|----------------|
| H-02 未修正 + 長 IR | 中（重心で ~1 s、768 kHz で 768k サンプル < 2 s だが境界） |
| H-02 修正後 | **ほぼゼロ** |

**判定**: **High のまま維持**（防御的 clamp は必要）だが、**実装順序は H-02 の後**。

---

## 4. 実装準備度マトリクス

| ID | テスト準備 | 段階リリース | ロールバック | 実装難易度 |
|----|------------|--------------|--------------|------------|
| H-02 | **要新規**（ユニットテスト） | 不要（単純差替） | 1 行 revert | 低 |
| H-01 | **要新規**（null test） | フィーチャーフラグ推奨 | フラグで切替 | 中 |
| SR-01(B) | 既存で足りる | 不要 | 1 行 revert | **低**（UI/applySnapshot のみ） |
| SR-02 | **work57 期待値更新必須** | 不要 | revert + テスト戻し | 中 |
| SR-03 | 要新規（境界テスト） | 不要 | 1 行 revert | 低 |
| M-04 | 要新規 | 不要 | 1 行 revert | 低 |
| M-02 | 要新規 | 不要 | 1 行 revert | 低 |

### 4.1 段階リリースの推奨（H-01）

H-01 は mix 聴感が変わる唯一の改修。  
```cpp
// 例: コンパイル時フラグまたは実行時設定
static constexpr bool kDryDelayUsesAlgorithmLatency = false;  // true = 旧挙動
```
旧挙動にフォールバックできるようにして、聴感比較テストを行う。

---

## 5. 実装順序の確定（v1.3）

```
Step 0: テスト先行
  ├─ H-02 ユニットテスト（estimatePeakLatencySamples）
  └─ H-01 null test（mix=0.5 交差相関）

Step 1: H-02（真のピーク）
  └─ 副作用: SR-03 の実害確率がほぼ消える

Step 2: SR-03（防御的 clamp）
  └─ H-02 後なので clamp 値は現実的

Step 3: H-01（内部 dry 遅延、フラグ付き）
  └─ null test で検証 → フラグを false に固定

Step 4: SR-01 方案 B（動的 IR 上限）
  ├─ applySnapshot を hardMax クランプに
  ├─ UI スライダー上限を動的化
  └─ computeTargetIRLength のキャップをログ

Step 5: SR-02（l0MaxLen 秒ベース）
  └─ work57 テスト期待値を同時に更新

Step 6: M-04 / M-02 / M-01（RT 衛生）
```

**重要な順序制約**:
- **H-02 → SR-03**（逆にすると無駄なバッファ拡張）
- **SR-02 → work57 テスト更新**（同時に）
- **SR-01 は DELAY_BUFFER 拡張不要**（方案 B なら）

---

## 6. 更新後の優先順序サマリ

| Phase | ID | 方案 | テスト | メモリ |
|-------|-----|------|--------|--------|
| 0 | テスト先行 | — | H-01/H-02 回帰テスト | — |
| 1 | H-02 | 真のピーク | ユニットテスト新規 | なし |
| 1 | SR-03 | 防御的 clamp | 境界テスト | なし（H-02 後） |
| 1 | H-01 | 内部 dry のみ（フラグ） | null test | なし |
| 2 | SR-01 | **方案 B（動的上限）** | 既存で足りる | **なし** |
| 2 | SR-02 | l0MaxLen 秒ベース | work57 更新 | L0 SoA 倍増 |
| 2 | SR-04 | UI 動的化 | — | なし |
| 3 | M-04 | dry フォールバック | 要新規 | なし |
| 3 | M-02 | FDL クリア | 要新規 | なし |
| 3 | 他 | — | — | — |

---

## 7. 結論（v1.3）

1. **テスト欠落が最大の実装障壁**。H-01/H-02 の回帰テストを先に書くべき。
2. **SR-01 は方案 B（動的上限）でメモリ増ゼロ**にできる。2^22 拡張は不要な場合がある。
3. **M-04 は stale データ残存**であり、素通しより深刻。Medium-High へ。
4. **実装順序: H-02 → SR-03 → H-01 → SR-01(B) → SR-02**。
5. **H-01 のみフィーチャーフラグ**で段階リリースすべき。
6. 監査範囲はコンボルバー経路が主。EQ/Oversampler/NoiseShaper のレート依存は別途。

以上。
