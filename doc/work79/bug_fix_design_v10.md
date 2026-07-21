# ConvoPeq 未修正バグ改修設計書（v10 — 完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v9 レビュー → v10 反映：Bug H 保証範囲の正確な限定、テストケース追加）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（96〜97点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v8 | 2026-07-21 | 段階的にレビュー反映 |
| v9 | 2026-07-21 | Bug H Strong Exception Guarantee 再設計（2段階アプローチ） |
| v10 | 2026-07-21 | **v9 レビュー反映：Bug H 保証範囲の正確な限定、API 契約の明確化、テストケース追加** |

---

## 評価サマリ（v10 — 最終）

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
| Bug H | ★★★★★ | P0 | **採用**（保証範囲を正確に限定） |

---

## Bug H — StereoConvolver::init 例外安全性（v10: 保証範囲の正確な限定）

### Strong Exception Guarantee の正確な定義（v10 明確化）

**保証対象**: `StereoConvolver` オブジェクトの内部状態（`irData`、`nucConvolvers`、`storedSampleRate` 等）

**非保証対象**:
- 呼び出し側の所有権状態（`LoaderThread` が `irL.release()` で渡した後の状態）
- `commit` フェーズ自体が将来例外を投げる実装に変更された場合

**前提**: `commit` フェーズ（Phase 2）は現在 `noexcept` 操作のみで構成されている。

### 修正案（v10）

```cpp
bool init(double* irL, double* irR, int length, double sr, int peakDelay, int knownBlockSize, int preferredCallSize, double scale = 1.0,
      bool enableDirectHead = false,
      const convo::FilterSpec* filterSpec = nullptr,
      ConvolverProcessor* ownerProcessor = nullptr)
{
    // ============================================================
    // Phase 1: すべてローカル変数で初期化を実行（メンバー未更新）
    // ★ Strong Exception Guarantee: このフェーズ中の例外で
    //   StereoConvolver の内部状態は変更されない
    // ============================================================

    convo::aligned_unique_ptr<double[]> newIrL(irL);
    convo::aligned_unique_ptr<double[]> newIrR(irR);

    auto newNuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto newNuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

    if (!newNuc0->SetImpulse(newIrL.get(), length, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        DBG("Convolver: init failed - SetImpulse ch0 failed");
        return false;
    }
    if (!newNuc1->SetImpulse(newIrR.get(), length, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        DBG("Convolver: init failed - SetImpulse ch1 failed");
        return false;
    }

    // ============================================================
    // Phase 2: 全成功 — メンバーを一括更新（commit）
    // ★ 前提: このフェーズは現在 noexcept 操作のみで構成されている
    //   将来例外を投げる可能性がある操作を追加する場合は、
    //   この前提を見直すこと
    // ★ 注意: 呼び出し側は irL.release() で既に所有権を放棄済み。
    //   保証されるのは StereoConvolver の内部状態であり、
    //   呼び出し側への所有権返却は保証しない
    // ============================================================

    destroyNUCConvolver(nucConvolvers[0]);
    destroyNUCConvolver(nucConvolvers[1]);
    irData[0].reset();
    irData[1].reset();

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

### API 契約（v10 明確化）

| 項目 | 契約 |
|------|------|
| **保証対象** | `StereoConvolver` の内部状態（`irData`、`nucConvolvers`、`storedSampleRate` 等） |
| **非保証対象** | 呼び出し側の所有権状態（`irL.release()` 済み） |
| **前提** | Phase 2（commit）は `noexcept` 操作のみで構成されている |
| **成功時** | 全メンバーを更新 |
| **失敗時** | 全メンバーが元の状態を維持（Phase 1 中の例外時） |

### 呼び出し側との整合（LoaderThread）

```cpp
// LoaderThread::run() 内
auto l = convo::makeAlignedArray<double>(...);
auto r = convo::makeAlignedArray<double>(...);
// ...
if (!newConv->init(l.release(), r.release(), ...))
{
    // ★ l/r は release() 済みなので、ここで l/r を解放してはならない
    // ★ init 失敗時、irData は StereoConvolver 内で自動解放される
    // ★ 呼び出し側は newConv のみを管理すればよい
}
```

### テスト方法（v10 追加）

1. **Exception injection test**:
   - `SetImpulse` 1回目で例外 → StereoConvolver 内部状態が元のままであることを確認
   - `SetImpulse` 1回目成功・2回目で例外 → StereoConvolver 内部状態が元のままであることを確認

2. **API 契約テスト（v10 追加）**:
   - `SetImpulse == false`（例外ではなく失敗）→ 内部状態が元のままであることを確認
   - `aligned_make_unique` が失敗（`std::bad_alloc`）→ 内部状態が元のままであることを確認
   - `filterSpec == nullptr` → 正常動作することを確認
   - `length == 0` → 正常動作することを確認
   - `preferredCallSize == 0` → 正常動作することを確認
   - `irL == nullptr` → 正常動作することを確認（`aligned_unique_ptr` が null を保持）
   - `irR == nullptr` → 正常動作することを確認

---

## Bug A/B/D — NaN/Inf 伝播防止

変更なし。v8 の内容をそのまま採用。

---

## Bug C — pushBlock 境界チェック

変更なし。v8 の内容をそのまま採用。

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。

---

## Finding #9 — emitRetireIntentRT 命名

変更なし。

---

## Finding #10 — MKL バッファ std::vector

変更なし。

---

## Bug G — 冗長な負値チェック

変更なし。

---

## bug3群 — 第3回報告バグ

### bug3-1 / bug3-2

Bug H の Strong Exception Guarantee で**同時に解決**。

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

v10 では Strong Exception Guarantee のため、**初期化途中でメンバーを更新しない**。全成功後に一括コミット。

### bug3-6 — delayWritePos API 契約

```cpp
// ★ API 契約: reset() は Audio Thread 停止後にのみ呼び出すこと。
int delayWritePos = 0;
```

### bug3-8 — got >= 0 && got <= numSamples の防御チェック

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
   - `SetImpulse` 1回目で例外 → 内部状態が元のままであることを確認
   - `SetImpulse` 1回目成功・2回目で例外 → 内部状態が元のままであることを確認
5. **API 契約テスト（v10 追加）**:
   - `SetImpulse == false` → 内部状態が元のままであることを確認
   - `aligned_make_unique` 失敗 → 内部状態が元のままであることを確認
   - `filterSpec == nullptr` / `length == 0` / `preferredCallSize == 0` / `irL == nullptr` / `irR == nullptr` → 正常動作確認

### 統合テスト

1. 通常動作テスト、パフォーマンステスト、メモリテスト

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
