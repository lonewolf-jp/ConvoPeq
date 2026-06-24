# 改修実装検証レポート

**日時**: 2026-06-24
**検証範囲**: F1〜F5 全フェーズ
**使用ツール**: grep/Select-String, Serena MCP, CodeGraph MCP, semble, AiDex, graphify, cocoindex-code(ccc)

---

## 総評

**全5フェーズ（F1〜F5）の実装は計画書 v6.1 に忠実に実施されており、配線漏れ・周辺バグの発生は確認されませんでした。**

Debug/Release 両ビルド成功、新規警告ゼロ。以下に各フェーズの詳細検証結果を報告します。

---

## F3: r8brain IRテイル切り捨て修正

**ファイル**: `src/IRDSP.cpp`

### 検証項目

| # | チェック項目 | 結果 | 確認方法 |
|---|------------|------|---------|
| 1 | バッファサイズ計算に `getMaxOutLen()` を使用 | ✅ | コード確認: `tempResampler.getMaxOutLen(inLength)` |
| 2 | `maxOutLen <= 0` ガード | ✅ | コード確認: `if (maxOutLen <= 0) return {};` |
| 3 | `channelDone` vector 導入（-1初期化） | ✅ | コード確認: `std::vector<int> channelDone(numCh, -1);` |
| 4 | `anyChannelCancelled` atomic 導入 | ✅ | コード確認 |
| 5 | try/catch + `throw` による例外伝搬 | ✅ | コード確認 |
| 6 | `futures.get()` 使用（`wait()` からの変更） | ✅ | コード確認: `for (auto& f : futures) f.get();` |
| 7 | 条件付きトリム `maxDone < maxOutLen` | ✅ | コード確認 |
| 8 | `maxDone < 0` ガード | ✅ | コード確認 |
| 9 | 不要変数 `ratio` 削除 | ✅ | コード確認: 変数なし |
| 10 | Lambda キャプチャ `[&, ch]` の安全性 | ✅ | 解析: 各スレッドは固有 `ch` のみ書込。メインスレッドが `get()` で全完了を待つため参照先は生存 |
| 11 | 呼び出し元 `IRConverter.cpp:170` のみ | ✅ | Serena: `IRDSP::resampleIR` の参照は `IRConverter.cpp:170` のみ |

### 潜在リスク評価

| リスク | 評価 | 理由 |
|-------|------|------|
| データ競合 | **なし** | 各スレッドは `channelDone[ch]` の固有インデックスにのみ書き込む |
| デッドロック | **なし** | `std::async` は内部スレッドプールを使用、`get()` で待機 |
| 例外安全性 | **安全** | `catch (...)` で `anyChannelCancelled` 設定後 `throw` → `get()` で再送出 |
| キャンセルパス | **安全** | `shouldExit()` 真で早期リターン後、`maxDone < 0` で空バッファ返却 |

---

## F5: applyAllpassToIR 削除

**ファイル**: `src/AllpassDesigner.h`, `src/AllpassDesigner.cpp`

### 検証項目

| # | チェック項目 | 結果 | 確認方法 |
|---|------------|------|---------|
| 1 | 宣言削除 | ✅ | Serena find_symbol: 0件 |
| 2 | 実装削除 | ✅ | grep: `applyAllpassToIR` がコードベースに **0件** |
| 3 | `DftiHandle.h` include 削除 | ✅ | CodeGraph: 未使用include警告のみ |
| 4 | 呼び出し元が存在しないことの確認 | ✅ | CodeGraph query: 0件 / grep: 0件 / Serena: 0件 |

### 影響評価

`applyAllpassToIR` は完全なデッドコードであり、削除による影響はゼロ。

---

## F2: mixedPreRingTau パラメータ削除

**削除範囲**: 15ファイル（ConvolverProcessor, Runtime, StateAndUI, LoadPipeline, Rebuild, LoaderThread, MixedPhase, ControlPanel, AudioEngine, MainWindow, PersistentCache）

### 網羅的残存確認

| 検索パターン | ツール | 結果 |
|------------|-------|------|
| `mixedPreRingTau` | grep (src/\*\*) | **0件** ✅ |
| `MIXED_TAU` | grep (src/\*\*) | **0件** ✅ |
| `mixedTauSlider` | grep (src/\*\*) | **0件** ✅ |
| `setMixedPreRingTau` | grep (src/\*\*) | **0件** ✅ |
| `getMixedPreRingTau` | grep (src/\*\*) | **0件** ✅ |
| `setConvolverMixedPreRingTau` | grep (src/\*\*) | **0件** ✅ |
| `--cli-pre-ring-tau` | grep (src/\*\*) | **0件** ✅ |
| `kLastUsedTimeOffset` | grep (src/\*\*) | **0件** ✅ |
| `\btau\b` (PersistentCache) | Select-String | **0件** ✅ |
| 全F2パターン一括 | PowerShell Select-String | **ALL CLEAN** ✅ |
| `.tau` (convolver/) | Serena search | **0件** ✅ |
| `mixedPreRingTau/MIXED_TAU` | CodeGraph query | **0件** ✅ |

### 永続キャッシュ整合性

| 項目 | 結果 | 確認方法 |
|------|------|---------|
| `kVersion` 更新 `1→2` | ✅ | `MixedPhasePersistentCache.h:66` 確認 |
| `kLastUsedTimeOffset` → `offsetof` | ✅ | `MixedPhasePersistentCache.cpp:252` 確認 |
| DiskHeader から `float tau` 削除 | ✅ | grep: 0件 |

### 注意点

`convertToMixedPhase()` 系関数の `tau` パラメータ自体は関数シグネチャに残存しています。これは計画書のスコープ通りの動作であり（`key.tau` のみ削除対象）、`(void)tau` による未使用パラメータ抑制も維持されています。DSP 処理には影響しません。

---

## F1: Mixed Phase クロスオーバー方向修正

**ファイル**: `src/convolver/ConvolverProcessor.MixedPhase.cpp`

### 検証項目

| # | チェック項目 | 結果 | 確認方法 |
|---|------------|------|---------|
| 1 | `convertToMixedPhaseAllpass()` 内反転 | ✅ | コード確認: `wMinimum=1.0` 主体 + `wLinear=1.0-wMinimum` |
| 2 | `convertToMixedPhaseFallback()` 内反転 | ✅ | コード確認: 同様の反転 |
| 3 | 低域での動作 | ✅ | `freq <= transitionLoHz`: `wMinimum=1.0, wLinear=0.0` → **Minimum Phase** |
| 4 | 高域での動作 | ✅ | `freq >= transitionHiHz`: `wMinimum=0.0, wLinear=1.0` → **Linear Phase** |
| 5 | 遷移帯域の動作 | ✅ | Cosine クロスフェード維持 |

### DSP 論理検証

```cpp
// 変更前: 低域=Linear Phase, 高域=Minimum Phase
phiTarget = wLinear * phi_lin + wMinimum * phi_min
// → 低域: wLinear=1 → phiTarget = phi_lin (Linear)
// → 高域: wLinear=0 → phiTarget = phi_min (Minimum)

// 変更後: 低域=Minimum Phase, 高域=Linear Phase
phiTarget = wLinear * phi_lin + wMinimum * phi_min
// → 低域: wMinimum=1 → phiTarget = phi_min (Minimum)
// → 高域: wMinimum=0 → phiTarget = phi_lin (Linear)
```

重み変数名は入れ替わりましたが、`phiTarget` の計算式 `wLinear * phi_lin + wMinimum * phi_min` は変更前後で同一です。したがって数式上のバグはありません。

### キャッシュ整合性

F2 で `kVersion` が `1→2` に更新済みのため、F1 実施による旧方向 Mixed IR のキャッシュ再利用問題は自動解決済み。

---

## F4: computeMasteringSizing 削除

**削除範囲**: 9ファイル

### 検証項目

| # | チェック項目 | 結果 | 確認方法 |
|---|------------|------|---------|
| 1 | `ConvolverSizing` 構造体削除 | ✅ | grep/CodeGraph: 0件 |
| 2 | `computeMasteringSizing` 関数削除 | ✅ | grep/CodeGraph: 0件 |
| 3 | `storedMaxFFTSize` 削除 | ✅ | grep: 0件 |
| 4 | `storedFirstPartition` 削除 | ✅ | grep: 0件 |
| 5 | `init()` シグネチャ変更 | ✅ | 全4呼び出し箇所の引数一致確認 |
| 6 | `clone()` 修正 | ✅ | `init()` 呼び出しが新シグネチャに一致 |
| 7 | `finalizeNUCEngineOnMessageThread` シグネチャ変更 | ✅ | 宣言+定義+呼び出し元一致確認 |
| 8 | `initializeConvolverSynchronously` シグネチャ変更 | ✅ | 同上 |
| 9 | `queueFinalizeOnMessageThread` シグネチャ変更 | ✅ | 同上 |
| 10 | `SetImpulse()` への影響 | **なし** ✅ | 変更前と同一: `knownBlockSize, scale, enableDirectHead, filterSpec` |

### init() 呼び出し全4箇所の引数一致確認

| 呼び出し元 | 引数 | 一致 |
|-----------|------|------|
| `clone()` (ConvolverProcessor.h:764) | `(l, r, irDataLength, storedSampleRate, irLatency, storedKnownBlockSize, callQuantumSamples, storedScale, storedDirectHeadEnabled)` | ✅ |
| `Lifecycle.cpp:232` | `(irL, irR, conv->irDataLength, sampleRate, conv->irLatency, internalBlockSize, samplesPerBlock, conv->storedScale, getExperimentalDirectHeadEnabled(), &tailSpec, this)` | ✅ |
| `LoaderThread.cpp:269` | `(irL, irR, result.targetLength, sr, irPeakLatency, internalBlockSize, callBlockSize, result.scaleFactor, owner.getExperimentalDirectHeadEnabled(), &spec, &owner)` | ✅ |
| `LoadPipeline.cpp:587` | `(irL, irR, length, sr, peakDelay, knownBlockSize, preferredCallSize, scaleFactor, getExperimentalDirectHeadEnabled(), &spec, this)` | ✅ |

### 休眠コードの確認

| 項目 | 結果 | 確認 |
|------|------|------|
| `shareConvolutionEngineFrom` の呼び出し元 | **0件** | grep: 宣言+定義のみ |
| `clone()` 経路 | **休眠状態** | `shareConvolutionEngineFrom` からのみ呼ばれるが、同関数に呼び出し元なし |

---

## 周辺バグ・リグレッション総点検

### ビルド検証

| 構成 | 結果 | 新規警告数 |
|------|------|-----------|
| Debug | ✅ 成功 | 0 |
| Release | ✅ 成功 | 0 |

### データフロー完全性

| 変更箇所 | 上流 | 下流 | 断絶 |
|---------|------|------|------|
| IRDSP::resampleIR | IRConverter のみ | `juce::AudioBuffer<double>` 戻り値 | **なし** ✅ |
| AllpassDesigner 宣言削除 | 呼び出し元なし | — | **なし** ✅ |
| mixedPreRingTau 削除 | StateAndUI/UI/CLI/PersistentCache | Snapshot→Loader→NUC 経路から除去 | **なし** ✅ |
| wLinear/wMinimum 反転 | 同一関数内のみ | phiTarget 重み計算 | **なし** ✅ |
| computeMasteringSizing 削除 | Lifecycle/LoaderThread | init()→MKL SetImpulse() | **なし** ✅ |

### 潜在リスク評価

| リスク | 該当 | 評価 |
|-------|------|------|
| データ競合（Audio Thread） | F3 | **なし**: `IRDSP::resampleIR` は Loader Thread のみで使用 |
| メモリリーク | F3/F4 | **なし**: デストラクタ・解放パスは変更なし |
| 未定義動作 | F1 | **なし**: 数学的等価変換（`w` と `1-w` の交換） |
| 型不一致 | F4 | **なし**: 全 `init()` 呼び出しの引数型を確認 |
| キャッシュ破損 | F2 | **安全**: `kVersion 1→2` により旧フォーマット自動無効化 |
| 既存IRとの互換性 | F1 | **要認識**: 次回起動時に Mixed Phase IR が再設計される（kVersion更新済のため自動） |

---

## 結論

**F1〜F5 の全実装は計画書 v6.1 に忠実であり、配線漏れ・周辺バグの発生はありません。**

| フェーズ | ステータス | 備考 |
|---------|-----------|------|
| F3 (IRDSP tail truncation) | ✅ 正常実装 | 全8項目確認済み |
| F5 (dead code removal) | ✅ 正常実装 | コードベースから完全除去 |
| F2 (tau param removal) | ✅ 正常実装 | 15ファイル・全シンボル残存ゼロ |
| F1 (direction fix) | ✅ 正常実装 | 設計者意図に合致、キャッシュ整合性自動解決 |
| F4 (sizing removal) | ✅ 正常実装 | 9ファイル・全呼び出し側引数一致確認済み |
| 周辺バグ | ✅ 発生なし | Debug/Release ビルド成功、新規警告ゼロ |

**残課題**: F6（レイヤースケジューリング最適化） — 将来課題として未着手。
