# ConvoPeq 未修正バグ改修設計書（v9 — 完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v8 レビュー → v9 反映：Bug H Strong Exception Guarantee 完全実現）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（98〜99点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v7 | 2026-07-21 | 段階的にレビュー反映 |
| v8 | 2026-07-21 | v7 レビュー反映：Bug H Strong Exception Guarantee 再設計、BugA static_assert 追加、BugC jassert 可読性改善、bug3-8 got>=0 チェック追加 |
| v9 | 2026-07-21 | **v8 レビュー反映：Bug H で状態変更を全成功後に一括コミット（Strong Exception Guarantee 完全実現）** |

---

## 評価サマリ（v9 — 最終）

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用 |
| Bug C | ★★★★★ | P0 | 採用（drop 方針） |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用** |
| Finding 9 | ★★★★★ | P1 | 採用 |
| Finding 10 | ★★★☆☆ | P2 | 改善項目 |
| Bug F | ★★☆☆☆ | P3 | 保留 |
| Bug G | ★★★☆☆ | P2 | 整理程度 |
| Bug H | ★★★★★ | P0 | **採用**（Strong Exception Guarantee 完全実現） |

---

## Bug H — StereoConvolver::init 例外安全性（v9: 完全な Strong Exception Guarantee）

### v8 の問題

v8 では `irData[0].reset()` / `irData[1].reset()` が初期化処理の**前**に実行されていた。また `storedSampleRate` 等も成功前に更新されていた。これは Strong Exception Guarantee を満たさない。

### v9 の修正案: 全状態変更を成功後に一括コミット

**すべての初期化処理をローカル変数で実行し、全成功後にのみメンバーを更新する。**

```cpp
bool init(double* irL, double* irR, int length, double sr, int peakDelay, int knownBlockSize, int preferredCallSize, double scale = 1.0,
      bool enableDirectHead = false,
      const convo::FilterSpec* filterSpec = nullptr,
      ConvolverProcessor* ownerProcessor = nullptr)
{
    // ============================================================
    // Phase 1: すべてローカル変数で初期化を実行（メンバー未更新）
    // ============================================================

    // ★ irData の一時オブジェクト（成功時のみ irData に move）
    convo::aligned_unique_ptr<double[]> newIrL(irL);
    convo::aligned_unique_ptr<double[]> newIrR(irR);

    // ★ NUC の一時オブジェクト（成功時のみ nucConvolvers に move）
    auto newNuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto newNuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

    // ★ SetImpulse は一時オブジェクトに対して実行（メンバー未更新）
    if (!newNuc0->SetImpulse(newIrL.get(), length, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        DBG("Convolver: init failed - SetImpulse ch0 failed");
        return false;  // 全ての状態が元のまま（Strong Exception Guarantee）
    }
    if (!newNuc1->SetImpulse(newIrR.get(), length, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        DBG("Convolver: init failed - SetImpulse ch1 failed");
        return false;  // 全ての状態が元のまま（Strong Exception Guarantee）
    }

    // ============================================================
    // Phase 2: 全成功 — ここで初めてメンバーを更新（一括コミット）
    // ============================================================

    // 既存のリソースを解放
    destroyNUCConvolver(nucConvolvers[0]);
    destroyNUCConvolver(nucConvolvers[1]);
    irData[0].reset();
    irData[1].reset();

    // ★ すべての状態を一括更新（commit）
    irData[0] = std::move(newIrL);
    irData[1] = std::move(newIrR);
    nucConvolvers[0] = newNuc0.release();
    nucConvolvers[1] = newNuc1.release();

    irDataLength = length;
    this->irLatency = peakDelay;
    latency = nucConvolvers[0]->getLatency();
    callQuantumSamples = juce::jmax(1, preferredCallSize);
    storedSampleRate = sr;
    storedKnownBlockSize = knownBlockSize;
    storedScale = scale;
    storedDirectHeadEnabled = enableDirectHead;
    if (filterSpec != nullptr) {
        storedFilterSpec = *filterSpec;
        hasStoredFilterSpec = true;
    } else {
        hasStoredFilterSpec = false;
    }

    DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
    return true;
}
```

### Strong Exception Guarantee の保証（v9）

| 状態 | SetImpulse 失敗時 | 全成功時 |
|------|-------------------|----------|
| `irData[0/1]` | **元のまま** | move（更新） |
| `nucConvolvers[0/1]` | **元のまま** | move（更新） |
| `irDataLength` | **元のまま** | length に更新 |
| `storedSampleRate` 等 | **元のまま** | 新値に更新 |
| `latency` | **元のまま** | 新値に更新 |

**全失敗パスでメンバー状態が変更されることはない。** これが完全な Strong Exception Guarantee。

### API 契約（v9 明確化）

- `init()` は成功時にのみ全メンバーを更新する
- 失敗時は**すべてのメンバー**が元の状態を維持する
- 呼び出し側は失敗時に元の状態が保たれていると確実に期待してよい

### テスト方法

1. **Exception injection test（全パターン）**:
   - `SetImpulse` 1回目で例外 → 全メンバーが元のままであることを確認
   - `SetImpulse` 1回目成功・2回目で例外 → 全メンバーが元のままであることを確認
2. 通常動作で問題がないことを確認
3. メモリリークがないことを確認（Valgrind / AddressSanitizer）

---

## Bug A/B/D — NaN/Inf 伝播防止

変更なし。v8 の内容をそのまま採用。

```cpp
static_assert(std::numeric_limits<double>::is_iec559, "IEEE754 binary64 前提");
static_assert(std::numeric_limits<float>::is_iec559, "IEEE754 binary32 前提");
```

---

## Bug C — pushBlock 境界チェック

変更なし。v8 の内容をそのまま採用。

```cpp
if (numSamples > kCapacity)
{
    jassert(numSamples <= kCapacity);
    return;
}
```

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。

---

## Finding #9 — emitRetireIntentRT 命名

変更なし。

---

## Finding #10 — MKL バッファ std::vector

変更なし。「MKL SIMD 処理へ直接渡す配列」のみ対象。

---

## Bug G — 冗長な負値チェック

変更なし。

---

## bug3群 — 第3回報告バグ

### bug3-1 / bug3-2

Bug H の Strong Exception Guarantee 完全実現で**同時に解決**。

### bug3-3 — numSamples <= 0 チェック

```cpp
if (channel < 0 || channel >= 2 || !nucConvolvers[channel] || numSamples <= 0)
{
    if (numSamples > 0)
        std::memset(out, 0, numSamples * sizeof(double));
    return;
}
```

### bug3-4 / bug3-5 — init 失敗時の状態リセット

**v9 の方針変更**: Strong Exception Guarantee のため、**初期化途中でメンバーを更新しない**。全成功後に一括コミット。

```cpp
// ★ v9: storedSampleRate 等は Phase 2（成功時）で一括更新
//   Phase 1（初期化処理）では一切更新しない
```

### bug3-6 — delayWritePos API 契約

```cpp
// ★ API 契約: reset() は Audio Thread 停止後にのみ呼び出すこと。
int delayWritePos = 0;
```

### bug3-8 — got > numSamples の防御チェック

```cpp
const int got = nucConvolvers[channel]->Get(out, numSamples);
jassert(got >= 0 && got <= numSamples);
if (got < numSamples)
    std::memset(out + got, 0, (numSamples - got) * sizeof(double));
```

---

## 実装優先順位

| 優先 | 項目 | 理由 |
|------|------|------|
| P0-1 | Bug C | メモリ破壊防止 |
| P0-2 | Bug 2 | stale data 防止 |
| P0-3 | Bug A/B/D | 音声破綻防止 |
| P0-4 | Bug H / bug3-1/2 | irData リーク（Strong Exception Guarantee） |
| P0-5 | bug3-3 | numSamples 負値でバッファオーバーフロー |
| P1 | Finding 9 | Debug での誤呼び出し検知 |
| P2 | Finding 10 | 設計ポリシー統一 |
| P2 | Bug G | コード整理 |
| P3 | bug3-6/8 | 各種防御的改善 |
| 保留 | Bug 3/E, Bug F | 将来対応 |

### 総工数推定

- P0: 4 日
- P2: 3 日
- P3: 1 日
- **合計: 8 日**

---

## テスト計画

### 単体テスト

1. **NaN/Inf テスト**: NaN/Inf を注入した入力でフィルタをテスト
2. **境界テスト**: `pushBlock` で `numSamples > kCapacity` の場合をテスト
3. **null テスト**: `processBypassWithLatencyCompensation` で delayBuffer null の場合をテスト
4. **Exception injection test（全パターン）**:
   - `SetImpulse` 1回目で例外 → 全メンバーが元のままであることを確認
   - `SetImpulse` 1回目成功・2回目で例外 → 全メンバーが元のままであることを確認

### 統合テスト

1. 通常動作テスト、パフォーマンステスト、メモリテスト

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
