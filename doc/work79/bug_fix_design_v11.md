# ConvoPeq 未修正バグ改修設計書（v11 — 最終完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v10 レビュー → v11 反映：Phase 2 noexcept 根拠明記、nullptr テスト修正）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（97〜98点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v9 | 2026-07-21 | 段階的にレビュー反映 |
| v10 | 2026-07-21 | Bug H 保証範囲の正確な限定、テストケース9パターン拡充 |
| v11 | 2026-07-21 | **v10 レビュー反映：Phase 2 noexcept 根拠明記、nullptr テストの期待結果修正** |

---

## 評価サマリ（v11 — 最終）

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
| Bug H | ★★★★★ | P0 | **採用**（noexcept 根拠明記済み） |

---

## Bug H — StereoConvolver::init 例外安全性（v11: noexcept 根拠明記）

### Phase 2 の noexcept 根拠（v11 追加）

Phase 2（commit フェーズ）が `noexcept` 操作のみで構成されている根拠：

| 操作 | noexcept 根拠 |
|------|--------------|
| `destroyNUCConvolver(ptr)` | `ptr->~MKLNonUniformConvolver()` + `mkl_free()`。デストラクタは `noexcept`（C++11 以降の規約）。`mkl_free` は C 関数で例外を投げない |
| `irData[i].reset()` | `aligned_unique_ptr::reset()` は `aligned_free()` を呼ぶだけ。`aligned_free` は `mkl_free` のラッパーで `noexcept` |
| `std::move(irData[i])` | `unique_ptr` の move コンストラクタは `noexcept` |
| `nucConvolvers[i] = ptr.release()` | `unique_ptr::release()` は生ポインタを返すだけ。`noexcept` |
| メンバー代入（`irDataLength = length` 等) | 組み込み型・POD 型の代入。`noexcept` |

**結論**: Phase 2 の全操作は `noexcept` である。したがって、Phase 2 で例外が発生することは現時点ではない。

### 修正案（v11）

```cpp
bool init(double* irL, double* irR, int length, double sr, int peakDelay, int knownBlockSize, int preferredCallSize, double scale = 1.0,
      bool enableDirectHead = false,
      const convo::FilterSpec* filterSpec = nullptr,
      ConvolverProcessor* ownerProcessor = nullptr)
{
    // ============================================================
    // Phase 1: すべてローカル変数で初期化を実行（メンバー未更新）
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
    //
    // ★ noexcept 根拠:
    //   - destroyNUCConvolver: ptr->~T() (noexcept) + mkl_free (C関数, noexcept)
    //   - aligned_unique_ptr::reset: aligned_free (mkl_free wrapper, noexcept)
    //   - std::move(unique_ptr): noexcept
    //   - unique_ptr::release: noexcept
    //   - メンバー代入 (int/double/bool): noexcept
    //   → Phase 2 全操作は noexcept。例外は発生しない。
    //
    // ★ 保証対象: StereoConvolver の内部状態のみ。
    //   呼び出し側の所有権状態は保証しない（irL.release() 済み）。
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

### テスト方法（v11 修正）

| テストケース | 期待結果 | 根拠 |
|-------------|----------|------|
| `SetImpulse` 1回目で例外 | 内部状態が元のまま | Strong Exception Guarantee |
| `SetImpulse` 1回目成功・2回目で例外 | 内部状態が元のまま | Strong Exception Guarantee |
| `SetImpulse == false` | 内部状態が元のまま | Strong Exception Guarantee |
| `aligned_make_unique` 失敗 | 内部状態が元のまま | Strong Exception Guarantee |
| `filterSpec == nullptr` | 正常動作（既存のデフォルト処理） | API が受け入れる |
| `length == 0` | 正常動作（空 IR） | API が受け入れる |
| `preferredCallSize == 0` | 正常動作（`jmax(1, 0)` で 1 になる） | API が受け入れる |
| `irL == nullptr` | **`false` を返す**（API 契約違反） | `SetImpulse` が nullptr を受け付けないため |
| `irR == nullptr` | **`false` を返す**（API 契約違反） | 同上 |

---

## Bug A/B/D — NaN/Inf 伝播防止

変更なし。

---

## Bug C — pushBlock 境界チェック

変更なし。

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

bug3-1/2 は Bug H で解決。bug3-3/6/8 は変更なし。

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
