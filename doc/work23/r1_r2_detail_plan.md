# R1/R2 詳細改修計画書 (v2 — レビュー反映版)

**作成日**: 2026-06-08 (初版), 2026-06-08 (v2)
**対象**: doc/work23/bug_fix_plan.md v6 の将来課題 R1, R2
**優先度**: 低（現状運用に支障なし、コード品質・保守性向上目的）

---

## 目次

1. [R1: exchangeCurrentState memory order統一](#r1)
2. [R2: AudioSegmentBuffer データレース対策](#r2)
3. [優先度評価](#priority)

---

## R1: exchangeCurrentState memory order統一 {#r1}

### 現状

`src/eqprocessor/EQProcessor.Parameters.cpp` において、`exchangeCurrentState()` の呼び出しで memory order が不統一:

| 関数 | 現在の order | 備考 |
| --- | --- | --- |
| `setBandFrequency` | `acq_rel` ✅ | 正しい |
| `setBandGain` | `release` ⚠️ | 不統一 |
| `setBandQ` | `release` ⚠️ | 不統一 |
| `setBandEnabled` | `release` ⚠️ | 不統一 |
| `setTotalGain` | `release` ⚠️ | 不統一 |

### 問題の詳細

`std::memory_order_release` を `exchange` に指定した場合:

- **store部**: release semantics → 後続の `loadCurrentState(acquire)` と正しく同期 ✅
- **load部**: relaxed semantics → 戻り値 `prev` の読み取りに acquire がかからない ⚠️

現在は `prev` を `retireEQStateDeferred(prev)` に渡す**だけで内容を読み取らない**ため、実用上問題なし。しかし将来 `prev->xxx` を読むコードが追加された場合にバグの原因となる。

### 修正内容

4箇所の `exchangeCurrentState(newState, std::memory_order_release)` を `exchangeCurrentState(newState, std::memory_order_acq_rel)` に変更:

```cpp
// Before (4箇所):
auto prev = exchangeCurrentState(newState, std::memory_order_release);

// After:
auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel);
```

### 変更ファイル

- `src/eqprocessor/EQProcessor.Parameters.cpp` (4行)

### 影響評価

| 項目 | 評価 |
| --- | --- |
| パフォーマンス | 🔹 無視できる (x86ではacq_relもreleaseも同一命令) |
| 機能 | 🔹 変更なし (現状と同じ動作) |
| ISR Runtime | ✅ 影響なし (lock/alloc/retire/publication/Authority すべて不変) |
| リスク | ★☆☆ 極小 (acq_relはreleaseより強い制約) |

### 判定

**実施可（推奨）** — ただし「バグ修正」ではなく「保守性改善」。単独タスク化せず、次回EQProcessor改修時についでに実施することで十分。

---

## R2: AudioSegmentBuffer データレース対策 {#r2}

### 現物コード調査結果

#### 実際のデータフロー

現物コードを全ツール(grep/Serena/ccc/CodeGraph/semble)で解析した結果、AudioSegmentBuffer の実際の使用状況は以下の通り:

```
Audio Thread (process)
    │
    │ pushAdaptiveCaptureBlocks()
    ▼
captureQueue (LockFreeRingBuffer<AudioBlock, 4096>)  ★ SPSC
    │
    │ drainCaptureQueue() [pop]
    ▼
workerThreadMain (単一 std::jthread)                 ★ Single Consumer
    │
    ├── segmentBuffer.pushBlock()       ← 単一スレッドからのみ
    ├── segmentBuffer.buildTrainingSegments() → copyLatest() ← 同スレッド
    └── segmentBuffer.clear()           ← 同スレッド
```

**重要な判明事項**:

1. **pushBlock, copyLatest, clear はすべて同一スレッド (workerThreadMain) から呼ばれている**
   - 呼び出し元の完全追跡: `workerThreadMain` (line 718) が唯一の caller
   - `drainCaptureQueue()` (line 1109) → `pushBlock()` (line 1143)
   - `buildTrainingSegments()` (line 1155) → `copyLatest()` (line 1165)
   - 両関数は同一ループ内で逐次実行（同時実行なし）

2. **Audio → Worker 間は既に SPSC (captureQueue) で分離済み**
   - `captureQueue` は `LockFreeRingBuffer<AudioBlock, 4096>` (既存のロックフリーSPSC)
   - Audio Thread からの `pushAdaptiveCaptureBlocks()` → エンキュー
   - Worker Thread からの `drainCaptureQueue()` → デキュー

3. **AudioSegmentBuffer に対する concurrent access は存在しない**
   - C++メモリモデル上の Data Race は **発生していない**
   - ただし AudioSegmentBuffer 自体は atomic ガードを持っており、設計としては過剰だが安全

#### 結論: 現状で実害なし

| 観点 | 評価 |
| --- | --- |
| Data Race | ❌ 発生していない (単一スレッドアクセス) |
| Audio SegmentBuffer | クラス設計はやや過剰だが機能的に問題なし |
| captureQueue | 既存の SPSC が Audio/Worker 間を分離済み |
| 改修優先度 | **最低** (現状で問題なし) |

#### 推奨: 現状維持

AudioSegmentBuffer の改修は以下の理由から**現時点では不要**:

1. 実質的な Data Race は存在しない（単一スレッドアクセス）
2. 既存の captureQueue (LockFreeRingBuffer) が Audio/Worker 間を適切に分離
3. 現状の AudioSegmentBuffer は Worker スレッド内の単なるリングバッファとして機能
4. 改修によるリスク（コピー量増加、設計変更の複雑さ）に対してベネフィットが小さい

**Practical Stable ISR Bridge Runtime の観点**: 「動いているものを壊さない」が最優先。AudioSegmentBuffer の atomic ガードは冗長だが安全サイドであり、削除する積極的な理由もない。**現状維持を推奨**。

### 将来の注意点

- 現在の `segmentBuffer` は `workerThreadMain` 専有
- `copyLatest()` は `NoiseShaperLearner` の private メソッドであり、外部から呼ばれる経路はない
- もし将来設計変更により UI スレッドから `copyLatest()` を呼ぶ必要が生じた場合、その時点で SPSC キュー導入を再検討すること
- `AudioSegmentBuffer` の atomic ガードは現状不要だが安全サイドであり、削除によるリスクとベネフィットを比較して判断すること

---

## 優先度評価 {#priority}

| 項目 | 優先度 | 判定 | 理由 |
| --- | --- | --- | --- |
| R1 | 🔵 **低** | ✅ 実施可（次回EQProcessor改修時） | 保守性改善。release→acq_rel統一。動作変更なし、リスク極小。 |
| R2 | 🔵 **最低** | ❌ **改修不要・現状維持推奨** | 実在しないData Raceを解決しない。既存captureQueueで適切にスレッド分離済み。 |

### 実施判断基準

- R1: **次回のEQProcessor改修時にまとめて実施** で十分（単独タスク化不要）
- R2: **実施しない**（問題が存在しない箇所へ新たな複雑性を導入するリスクを避ける）

---

*作成日: 2026-06-08 (v3)*
