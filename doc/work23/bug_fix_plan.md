# バグ改修計画書 (v6 — 4回目レビュー反映版)

**作成日**: 2026-06-08 (初版/v2/v3/v4/v5), 2026-06-08 (v6)
**対象**: doc/work23/bug_report_validation.md で確認された実バグ6件 + 推奨改善3件
**レビュー**: Practical Stable ISR Bridge Runtime 設計思想に基づく評価を反映

---

## 目次 {#toc}

1. [総合判定表](#sec1)
2. [Phase 1: 即時実施 — BUG-01 OutputFilter](#sec2)
3. [Phase 2: 即時実施 — BUG-02 ringWrite](#sec3)
4. [Phase 3: トレース検証完了 — BUG-03 DeferredDeletionQueue](#sec4)
5. [Phase 4: 修正推奨 — BUG-04 softClipBlockAVX2](#sec5)
6. [Phase 5: 即時実施 — BUG-05 kIdleEpoch](#sec6)
7. [Phase 6: 推奨改善 — R1/R2](#sec7)
8. [付録: 現物コード確認結果](#sec8)

---

## 1. 総合判定表 {#sec1}

### 1.1 実施判定

| ID | 項目 | 判定 | 理由 |
| --- | --- | --- | --- |
| BUG-01 | OutputFilter HC/LC未適用 | ✅ **即時実施** | 実バグ確実。ISR Runtime影響なし。 |
| BUG-02 | ringWrite二重更新 | ✅ **即時実施** | 実バグ確実。1行削除で完了。 |
| BUG-03 | reclaim 1エントリ制限 | ✅ **高確度で実バグ（トレース検証済み）** | doc/work23/bug03_trace_analysis.md にて6シナリオ + 並行性リスク検証。本修正で新たなABA導入の兆候なし。 |
| BUG-04 | softClip prevScalar prevScalar | ✅ **修正推奨（AVX2/スカラー不一整合）** | AVX2版とスカラー版で prevScalar の意味が不一致。インターサンプルピーク保護として理論的に問題。ISR Runtime影響なし。 |
| BUG-05 | kIdleEpochコメント | ✅ **即時実施** | コメント修正のみ。 |
| BUG-06 | ORDER=16→15 | ❌ **実施非推奨** | 16個中16番目が常に0.0であるのみ。配列サイズ変更は危険。効果なし。 |
| R1 | memory order統一 | 🔵 **優先度低** | 理論的正当性向上のみ。releaseでも実用上問題なし。 |
| R2 | AudioSegmentBuffer改善 | 🔵 **長期課題** | 非atomic共有はUBだがUI用途。SPSC移行は将来。 |
| R3 | snapshotRcuEpoch→markRetireEpoch | ❌ **実施非推奨** | Epoch Inflation 発生。ISR Runtime設計思想と逆方向。 |

### 1.2 ISR Runtime 適合性マトリクス

| 変更 | lock禁止 | alloc禁止 | retire不変 | publication不変 | Authority不変 |
| --- | :---: | :---: | :---: | :---: | :---: |
| BUG-01 | ✅ | ✅ | ✅ | ✅ | ✅ |
| BUG-02 | ✅ | ✅ | ✅ | ✅ | ✅ |
| BUG-03 | ✅ | ✅ | ⚠️ 変更あり | ✅ | ✅ |
| BUG-04 | ✅ | ✅ | ✅ | ✅ | ✅ |
| BUG-05 | ✅ | ✅ | ✅ | ✅ | ✅ |
| R1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| R2 | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 2. Phase 1: 即時実施 — BUG-01 OutputFilter {#sec2}

### ステータス

**実施**: 可 ✅ | **リスク**: 極小 | **ISR Runtime影響**: なし

### 現象

`OutputFilter::process()` が `convIsLast=true` 時に一切呼ばれず、HC/LC フィルターが完全に無効。

現行コード:

```cpp
const bool convIsLast = convActive &&
    (!eqActive || state.order == ProcessingOrder::EQThenConvolver);
if (!convIsLast)
{
    outputFilter.process(processBlock, false,
                         state.convHCMode, state.convLCMode, state.eqLPFMode);
}
```

### 修正

```cpp
outputFilter.process(processBlock, convIsLast,
                     state.convHCMode, state.convLCMode, state.eqLPFMode);
```

### 変更ファイル

- `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` (line 445-450)
- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` (line 246 相当)

### 事前確認事項

`OutputFilter::process()` 内部実装 (`src/OutputFilter.cpp` line 198-) は `if (convIsLast)` で完全分岐されており、モード① (HC+LC) / モード② (HP+LP) の切り替えが正しく実装されている。確認済み。

### ISR Runtime評価

- lock: なし ✅
- allocation: なし ✅
- retire: なし ✅
- publication: なし ✅
- Authority: 変更なし ✅

---

## 3. Phase 2: 即時実施 — BUG-02 ringWrite {#sec3}

### ステータス

**実施**: 可 ✅ | **リスク**: 極小 | **ISR Runtime影響**: なし

### 現象

overflow ブランチで `m_ringWrite` が [A] と [D] の2箇所で更新される。[D] の追加更新により次回書き込み位置が未読データを指す。

現行コード:

```cpp
m_ringWrite = (m_ringWrite + n) & m_ringMask;          // [A]

if (nextAvail > m_ringSize)
{
    const int overflow = nextAvail - m_ringSize;
    m_ringRead = (m_ringRead + overflow) & m_ringMask;
    m_ringAvail = m_ringSize;
    m_ringWrite = (m_ringWrite + overflow) & m_ringMask; // [D] 二重更新!
}
```

### 修正

```cpp
    m_ringWrite = (m_ringWrite + n) & m_ringMask;          // [A]

    if (nextAvail > m_ringSize)
    {
        const int overflow = nextAvail - m_ringSize;
        m_ringRead = (m_ringRead + overflow) & m_ringMask;
        m_ringAvail = m_ringSize;
        // [D] 削除: [A] で m_ringWrite は既に正しく更新済み。
        // m_ringWrite = (m_ringWrite + overflow) & m_ringMask;
    }
```

### 変更ファイル

- `src/MKLNonUniformConvolver.cpp` (line 1154): 1行削除 (またはコメント化)

### ISR Runtime評価

- lock: なし ✅
- allocation: なし ✅
- retire: なし ✅
- publication: なし ✅
- Authority: 変更なし ✅

---

## 4. Phase 3: トレース検証完了 — BUG-03 DeferredDeletionQueue {#sec4}

### ステータス

**実施**: ✅ 高確度で実バグ（トレース検証済み） | **リスク**: 低 (Vyukov MPMC, 修正 `++deqPos` は `drainAllUnsafe()` の `pos++` と同一パターン)

### 現物コード確認結果

`compareExchangeAtomic` の実装は `std::atomic_compare_exchange_strong_explicit`:

- **成功時**: `expected` (deqPos) は **更新されない** (標準C++仕様)
- 失敗時: `expected` に現在値が書き込まれる

CAS成功後のコード:

```cpp
if (convo::compareExchangeAtomic(dequeuePos,
                                 deqPos,                    // expected (更新されない)
                                 static_cast<uint32_t>(deqPos + 1),  // desired
                                 std::memory_order_release,
                                 std::memory_order_acquire))
{
    // ... 削除 ...
    convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release);

    scanPos = deqPos;  // deqPos は CAS前の旧値 (deqPos) のまま!
    scanned = 0;
}
```

### トレース検証（6シナリオ + 並行性リスク評価）

詳細は `doc/work23/bug03_trace_analysis.md` 参照。

#### バグ確認トレース

```text
CAS成功前: dequeuePos=5, deqPos(局所変数)=5
CAS成功:  dequeuePos→6, deqPos(局所変数)=5 (変わらず!)
scanPos = deqPos = 5  (解放済みの位置を再検査)
seq_atom[5] = 5 + kQueueSize  (解放済み)
diff = (5+4096) - (5+1) = 4095 ≠ 0 → break!
→ 1回の reclaim() で最大1エントリしか解放できない
```

#### 検証シナリオ結果

| シナリオ | 結果 |
| --- | --- |
| 5件連続 enqueue → reclaim | ✅ 全件回収成功 (修正前は1件のみ) |
| EBR条件で一部削除不可 | ✅ 削除可能な3件のみ正しく回収 |
| Wrap-around (slot循環) | ✅ 正しく動作 |
| キュー空 | ✅ 即 break、何もせず終了 |
| kMaxScan 制限 | ✅ 正常動作 |
| 2スレッド同時 CAS競合 | ✅ 競合解決、CAS失敗側は dequeuePos 再読取 |

#### 並行性リスク評価

| リスク | 判定 | 根拠 |
| --- | --- | --- |
| ABA | ✅ 新たなABA導入の兆候なし | CAS expected は単調増加カウンタ。本修正によって新たなABA問題が導入される兆候は確認されなかった。 |
| Slot skip | ❌ 発生しない | `scanPos == deqPos` チェックが FIFO を保証。 |
| Starvation | ✅ リスク低減 | 複数エントリ解放により滞留解消。enqueue/reclaimは異なるatomic操作のため相互ブロックなし。 |
| FIFO順序 | ✅ 維持 | 先頭エントリのみ CAS 可能。 |

#### 修正の安全性根拠

`drainAllUnsafe()` が既に同一パターンを使用:

```cpp
if (convo::compareExchangeAtomic(dequeuePos, pos, pos + 1, ...)) {
    // ... 削除 ...
    pos++;  // ← これと同じパターン!
}
```

CAS成功後、`dequeuePos` は確実に `deqPos + 1` であるため `++deqPos` は正しい。CASの不可分性により複数スレッド競合時も問題なし。

### 修正

```cpp
    ++deqPos;           // dequeuePos の新値 (deqPos+1) に追従
    scanPos = deqPos;
    scanned = 0;
```

### ISR Runtime評価

- lock: なし ✅
- allocation: なし ✅
- retire: **不変** ✅ (retire順序・頻度は変わらない。変わるのは reclaim 効率)
  - 現在: 1回のreclaimで1エントリ → 解放が遅延、キュー飽和リスク
  - 修正後: 1回のreclaimで複数エントリ (burst drain) → **DeferredDeletionQueue飽和耐性向上**
  - retire順序はFIFO維持のため不変
- reclaim: **改善** ✅ (burst drain により backlog 蓄積抑制、飽和耐性向上)
- publication: なし ✅
- Authority: 変更なし ✅

---

## 5. Phase 4: 修正推奨 — BUG-04 softClipBlockAVX2 {#sec5}

### ステータス

**実施**: ✅ 修正推奨（AVX2/スカラー不一整合） | **リスク**: 極小 | **ISR Runtime影響**: なし

### 問題の本質

本バグは単なる「prevScalar 読取位置の問題」ではなく、**AVX2最適化版とスカラー版でアルゴリズムの意味が一致していない** という不整合。

#### インターサンプルピーク保護としての理論的考察

- スカラー実装は「前回入力＋今回入力」で中点推定（入力系列の連続性を保護）
- AVX2実装は「前回出力(クリップ後)＋今回入力」で中点推定
- 開発者が意図的に出力波形連続性を見たかった可能性は理論上残るが、スカラー実装と一致していないことが決定的

#### AVX2版（現在）

```cpp
prevScalar = data[i + 3];  // クリップ後の出力値を次ブロックへ渡す
```

#### スカラー版

```cpp
prevSample = x;  // クリップ前の入力値を次ブロックへ渡す
```

AVX2版では「前回出力＋今回入力」で中点を計算する一方、スカラー版では「前回入力＋今回入力」で中点を計算する。CPUによって音が変わる可能性がある。

#### スカラー実装と整合しない点

- スカラー実装はクリップ前サンプルを履歴として保持しており、AVX2版とのアルゴリズム整合性が失われている
- プロ用オーディオアプリとしてはCPU依存で挙動が変わらないことが重要であり、AVX2版をスカラー版に一致させるべき
- 修正によりAVX2版とスカラー版の履歴保持が一致する

### 追加確認項目（重要）

修正後に AVX2版とスカラー版の一致試験を実施すること:

- 正弦波 (各周波数)
- 矩形波 (エッジ部分)
- インパルス
- フルスケール近傍の信号

上記を入力し、最大誤差・RMS誤差を比較。差異が浮動小数点誤差範囲内であることを確認。

### 現物コード確認結果

`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`:

```cpp
result = _mm256_blendv_pd(x, result, needClip);
    _mm256_storeu_pd(data + i, result);

prevScalar = data[i + 3];  // ← data[i+3] は store 後のクリップ済み値!
```

`_mm256_storeu_pd(data + i, result)` の時点で `data[i]～data[i+3]` はクリップ済み出力値で上書きされている。その後に `data[i+3]` を読むため、prevScalar には元の入力値ではなくクリップ後の値が入る。

これは次イテレーションのインターサンプルピーク検出 (`midVec = (prevScalar + x) * 0.5`) で使用され、ピーク推定が過小評価される。

### 修正

```cpp
result = _mm256_blendv_pd(x, result, needClip);
const double nextPrev = data[i + 3];  // store前に元の入力値を退避
    _mm256_storeu_pd(data + i, result);

prevScalar = nextPrev;
```

### 影響度

- 典型的な音楽信号では -0.5dB 未満の影響
- 「定期的なグリッチ」は過大評価
- ソフトクリップ品質の微細な改善

### ISR Runtime評価

- lock: なし ✅
- allocation: なし ✅
- retire: なし ✅
- publication: なし ✅
- Authority: 変更なし ✅

---

## 6. Phase 5: 即時実施 — BUG-05 kIdleEpoch {#sec6}

### ステータス

**実施**: 可 ✅ | **リスク**: 極小 (コメントのみ)

### 現物コード

```cpp
// Reader が exitReader() を呼び出し、自身のエポックを kIdleEpoch (UINT64_MAX) に
//                                                     ^^^^^^^^^^ ここが誤り
static constexpr uint64_t kIdleEpoch = 0;  // 実際は 0
```

### 修正

コメント `(UINT64_MAX)` → `(0)` に修正。

---

## 7. Phase 6: 推奨改善 — R1/R2 {#sec7}

### R1: exchangeCurrentState memory order統一

**判定**: 🔵 優先度低（ただし将来の保守性を考慮すると `acq_rel` が望ましい）

`setBandGain`, `setBandQ`, `setBandEnabled`, `setTotalGain` で `std::memory_order_release` を使用。理論的には `acq_rel` が正確だが、戻り値 `prev` は `retireEQStateDeferred()` に渡すのみでオブジェクト内容を読み取らないため、`release` でも実用上問題なし。

**注意**: 将来 `prev->xxx` を読むコードが追加された場合、`acq_rel` が必要になる。現在は成立しているが、保守性を考慮すると `acq_rel` への統一が望ましい。

実施する場合の変更:

```cpp
// Before:
auto prev = exchangeCurrentState(newState, std::memory_order_release);
// After:
auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel);
```

### R2: AudioSegmentBuffer データレース対応

**判定**: 🔵 長期課題 (SPSCキュー移行)

非atomic配列に対するRT/UIスレッド間の同時read/writeはC++メモリモデル上UBだが、UI波形表示用途では実用上の問題は稀。既存の `LockFreeRingBuffer` への移行を将来検討。

---

## 8. 付録: 現物コード確認結果 {#sec8}

### 8.1 BUG-06 Fixed15Tap ORDER 調査結果

**結論**: ORDER=16は正しい設計。変更非推奨。

`COEFF_PRESETS` 全10プリセットの係数確認:

```cpp
// 各プリセットは16係数。16番目 (index 15) は常に 0.0
{ 2.157553, -2.356649, ..., 0.009877, 0.0 },  // 44.1kHz
{ 2.172009, -2.313034, ..., 0.001068, 0.0 },  // 48kHz
... 全10プリセットで index 15 = 0.0
```

ORDER=16 の理由:

- 係数配列は16要素、うち15個が有効係数、16番目は常に0.0
- DSP のエラーフィードバックフィルタでは次数=15 (タップ数=15) だが、`ORDER` を配列サイズとして使っている
- リングバッファのモジュロ演算 `(idx - 1 + ORDER) % ORDER` は配列サイズが正しければ問題なし
- ORDER を15に変更すると、配列サイズ変更 + 全 COEFF_PRESETS 修正 + 境界整合性再確認が必要
- リスクに対してベネフィットが小さすぎる

### 8.2 R3 Authority Violation + Epoch Inflation リスク分析

`snapshotRcuEpoch()` (= `currentEpoch()` = 読取専用) → `markRetireEpoch()` (= `publishEpoch()` = `fetchAdd(globalEpoch,1)`) に変更した場合:

```text
変更前: Convolver交換5回で epoch 読取5回 (globalEpoch不変)
変更後: Convolver交換5回で epoch advance 5回 (globalEpoch+5)
```

#### 主理由: Authority Violation（責務境界破壊）

ConvoPeq の ISR Runtime 設計では、epoch の推進権限は ISRRetireRouter 側に一元管理されている。ConvolverProcessor は observer（読取専用）であり epoch owner ではない。

snapshotRcuEpoch()（= currentEpoch()、読取専用）→ markRetireEpoch()（= publishEpoch()、epoch advance）への変更はこの責務境界を破壊する:

1. ConvolverProcessor が epoch を勝手に進めることになる
2. ISRRetireRouter の epoch 管理と競合する可能性
3. ISR Runtime の Authority 境界違反 — Practical Stable ISR Bridge Runtime で最も避けるべき問題

#### 副理由: Epoch Inflation

1. **Epoch Inflation**: 不必要な epoch 消費により epoch 値の範囲が早期に枯渇
1. **reclaim頻度増加**: epoch advance ごとに reclaim が走り、EBR に不要な負荷
1. **他コンポーネントとの相互作用**: AudioEngine 側の publishEpoch タイミングと競合
1. ConvolverProcessor の epoch 読取は「観測」であって「推進」ではない。ISR Runtime では推進は ISRRetireRouter が一元管理する設計。

**現在の `snapshotRcuEpoch()` 維持が安定動作に寄与する。**

---

*作成日: 2026-06-08 (v2)*
*レビュアー: Practical Stable ISR Bridge Runtime 設計思想に基づく評価*
