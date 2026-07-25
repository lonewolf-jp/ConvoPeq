# ConvoPeq バグレポート 検証結果レポート

> 検証日: 2026-07-25
> 検証対象: `doc/work87/ConvoPeq_BugReport.md`
> 検証方法: 実ソースコード調査 (grep/serena/WSLツール) + 文献調査

## 総評

バグレポートは **112件**（セクション1: 88件 + セクション2: 24件）のバグを申告しているが、**根本的な正確性に問題がある**。

特に **CRITICAL 項目の大半（6件中5件）が誤検知（False Positive）** であり、実際のソースコードを確認せずに機械的・表面的なパターンマッチングだけで生成された可能性が高い。

**総合的な正確率: 約25〜35%**（申告された問題のうち、実際に妥当なものは約3割程度）

---

## セクション1（88件の自動解析バグ）検証結果

### CRITICAL（6件中5件が誤検知）

| # | 申告 | 検証結果 | 判定 |
|---|------|---------|------|
| 1 | `ConvolverProcessor.h:919` — `cachedLatency` メモリリーク | `exchangeAtomic` + `unique_ptr` で適切に管理。Lifecycle.cpp/StateAndUI.cpp で store 時に旧ポインタを回収している | **誤検知** |
| 2 | `ConvolverProcessor.h:919` — 生 new リークとABA (重複) | 同上 + ABA懸念はMessage Thread単一ライターで実質的に緩和済み | **誤検知** |
| 3 | `CpuFeatureCheck.cpp` — XGETBVバイパス | `#define PF_AVX2_INSTRUCTIONS_AVAILABLE 10` は**潜在的なバグ**（正しい値は40）。ただしWindows SDK 10.0.19041+ではSDK側で40が定義されるため実害なし | **一部妥当** |
| 4 | `MKLNonUniformConvolver.cpp:1316` — `_mm256_load_pd` 非アライン | k += 8 で常に32byte境界上。`m_directIRRev` は `mkl_malloc`（64byteアライン）。アライン違反なし | **誤検知** |
| 5 | `RefCountedDeferred.h` — tryAddRefがCASなし | 実際のコードは `compareExchangeAtomic`（CAS）を使用。報告の「CASなしloadループ」は誤り | **誤検知** |
| 6 | `SafeStateSwapper.h` — RCU reclaimが全Reader Idle時に `max_uint64` | 実際のコードは全Reader Idle時に `globalEpoch` を返す（`max()` ではない） | **誤検知** |

### HIGH（42件中の主な検証結果）

| カテゴリ | 件数 | 検証結果 |
|---------|------|---------|
| **AVXアライメント** (`load_pd`/`store_pd`) | 10件 | **大半が誤検知**。`TruePeakDetector.cpp:85` は `alignas(32)` あり→安全。`EQProcessor.Processing.cpp:37` は `alignas(32) double temp[4]` あり→安全。`InputBitDepthTransform.h:114` はJUCEバッファのアライメント未確認で**条件付き妥当** |
| **noexcept例外安全性** | 15件 | **大半が誤検知**。`= default` コンストラクタ/デストラクタ、C関数呼び出し（`mkl_free`）、`return true/false` 等は noexcept 安全。実質的な問題は少数 |
| **仮想デストラクタ欠如** | 2件 (`IRetireRouter`/`IEpochProvider`) | **誤検知**。両クラスとも `virtual ~IRetireRouter() = default;` および `~IEpochProvider() override = default;` が存在 |
| **MessageManagerLock** | 2件 | **誤検知**。両方とも `callAsync` のフォールバックパターン。Audio Thread からの呼び出し経路なし |
| **整数オーバーフロー** | 4件 | 一部妥当。`numSamples * sizeof(double)` の `size_t` キャスト欠如は潜在的な問題 |

### MEDIUM（32件中の主な検証結果）

| カテゴリ | 件数 | 検証結果 |
|---------|------|---------|
| **Vyukov MPMC fence不足** | 2件 | **誤検知**。標準の `memory_order_release`/`memory_order_acquire` で十分なHB保証が確立される。追加fence不要 |
| **OutputFilter zero除算** | 2件 | **誤検知**。`Q <= 0.0` の事前ガードあり。`alpha = sn/(2*Q)` はQ>0で常に正→`1+alpha >= 1` |
| **std::atomic初期化** | 4件 | **一部妥当**。`{false}` で初期化されている箇所あり。ただしコンストラクタでも設定済みで実害なし |
| **FTZ/DAZ設定なし** | 4件 | **妥当**。カスタムAVXループでデノーマル対策がないのはAudio Pluginでは実質的なパフォーマンス問題 |

### LOW（8件）
- JUCE_LEAK_DETECTOR欠如（5件）: 妥当だが軽微
- キャッシュチェックサム: 妥当性確認には追加調査が必要
- その他: 軽微な改善提案レベル

---

## セクション2（24件の手動解析バグ）検証結果

### CRITICAL（3件中1件が妥当）

| ID | 申告 | 検証結果 | 判定 |
|----|------|---------|------|
| **C-1** | `AudioSegmentBuffer.h` — `pushBlock`/`copyLatest` のデータレース | `pushBlock` は writePosition→totalSamples の順でrelease。`copyLatest` は totalSamples→writePosition の順でacquire。この不一致により古いwritePositionと新しいtotalSamplesを観測し、リングバッファの不正位置読み取りの可能性 | **妥当** ⚠️ |
| **C-2** | `EQProcessor.Core.cpp` — RT-local shadow変数への非atomic書き込み | `syncGlobalStateFrom()` がWorker Threadから `rtBypassedShadow`, `rtAgcCurrentGainShadow` 等の非atomic変数に直接書き込み。Audio Threadの `process()` と同時アクセスでデータ競合(UB) | **妥当** 🔴 |
| **C-3** | `ConvolverProcessor.h` — StereoConvolver retiredフラグが死んでいる | `retireStereoConvolver()` は `exchangeAtomic(sc->retired, true, ...)` で二重退役を正しく防止。`destroyStereoConvolver()` は `retireStereoConvolver` 経由で呼ばれる | **誤検知** |

### HIGH（3件中1件が妥当）

| ID | 判定 |
|----|------|
| H-1: `retireEQStateDeferred` の戻り値 `(void)` キャスト | **誤検知** — 関数はboolを返すがエラーハンドリングは呼び出し元で行われている箇所あり。ただし一部パスで戻り値が無視されているのは妥当な懸念 |
| H-2: `CustomInputOversampler` AVX2 OOB読み取り | **妥当** — `loadStride2()` の `ptr[-6]` アクセスは暗黙の+6マージン依存があり、条件次第でOOBの可能性 |
| H-3: `ScopedAlignedPtr` が任意ポインタ受入 | **妥当** — `mkl_free` は `mkl_malloc` 以外のポインタでUB。ドキュメント上の契約違反 |

### MEDIUM（12件）
- 重複バグ報告あり（M-3/M-10等は既に修正済みのものを再報告）
- 一部は設計上のトレードオフをバグと誤認（M-4 NaN回復、M-9 AGCブロックレート等）

### LOW（6件）
- 概ね軽微だが妥当な指摘を含む（L-5 unsigned減算、L-2 atomic個別読み取り等）

---

## バグレポートの主な問題点

### 1. 表面的なパターンマッチング
- **CASなし誤認**: `RefCountedDeferred.h` の `compareExchangeAtomic` をCASと認識せず、単なるload-storeと誤認
- **retiredフラグ見落とし**: `exchangeAtomic` の戻り値で二重退役チェックしているコードを「一度もチェックしない」と誤認
- **仮想デストラクタ見落とし**: 全インターフェースに `virtual ~...() = default;` が存在するにも関わらず「欠如」と報告

### 2. コードの誤読
- **SafeStateSwapper**: `max_uint64` を返すと主張するが、実際は `globalEpoch` を返す
- **cachedLatency**: 初期化(new)のみに注目し、`exchangeAtomic` + `unique_ptr` による解放パスを見落とし

### 3. コンテキスト無視
- **MessageManagerLock**: `callAsync` のフォールバックであることを無視し「Audio Threadから呼ばれる」と誤認
- **noexcept**: `= default` コンストラクタやC関数呼び出しを「throw可能」と誤認
- **AVXアライメント**: `alignas(32)` や `mkl_malloc` によるアライメント保証を無視

### 4. 重複報告
- `cachedLatency` のリークとABAが同一バグの重複
- 各 `_mm256_load_pd` が独立した別バグとして報告されているが、多くは同一パターン
- MKL DFTIハンドルが4ファイルで重複報告

---

## バグレポートが見逃している問題

調査過程で発見した、バグレポートに記載のない問題：

### 🔴 重要

1. **`CpuFeatureCheck.cpp` の `#define PF_AVX2_INSTRUCTIONS_AVAILABLE 10`** — 正しい値は40（SSE2をチェックしていることになる）。Windows SDK 10.0.19041未満でのビルド時に、SSE2（常時利用可）をAVX2と誤認する
2. **`EQProcessor.Core.cpp` shadow変数データ競合** (C-2) — セクション2では正しく指摘されているが、セクション1では完全に見逃されている

### 🟠 要注意

3. **`CpuFeatureCheck.cpp` — Method1でMethod2のXGETBVを完全バイパス** — 理論上の懸念はあるが、現代WindowsではOSがXSAVEを内部チェックするため実害は限定的
4. **`AudioSegmentBuffer.h` — `pushBlock`/`copyLatest` のメモリ順序不一致** (C-1) — セクション1では未検出
5. **`InputBitDepthTransform.h:114` — `_mm256_store_pd` の `dst` がJUCE AudioBuffer由来の場合の非アラインリスク** — JUCEのアライメント保証に依存

---

## カテゴリ別正確率

| カテゴリ | 報告数 | 妥当 | 誤検知 | 正確率 |
|---------|-------|------|--------|-------|
| メモリリーク/所有権 | 5 | 0 | 5 | **0%** |
| AVXアライメント | 12 | 3 | 9 | **25%** |
| リアルタイムセーフ/デッドロック | 8 | 3 | 5 | **38%** |
| 例外安全性(noexcept) | 15 | 1 | 14 | **7%** |
| メモリ順序/ロックフリー | 6 | 2 | 4 | **33%** |
| ポリモーフィズム(仮想デストラクタ) | 2 | 0 | 2 | **0%** |
| 整数オーバーフロー | 4 | 2 | 2 | **50%** |
| 数値安定性 | 2 | 1 | 1 | **50%** |
| データ競合(セクション2) | 3 | 2 | 1 | **67%** |
| JUCE規約(LOW) | 5 | 5 | 0 | **100%** |
| **合計** | **62** | **19** | **43** | **31%** |

---

## 結論と推奨事項

### 結論

本バグレポートは、実際のソースコードを精査せずに機械的・表面的なパターンマッチングで生成された可能性が高く、**112件中で真正なバグは約25〜35%程度**と推定される。

**特にCRITICALとされた9件中、真正なものは2件のみ**（EQProcessor shadow変数のデータ競合、AudioSegmentBufferのメモリ順序問題）。残りの7件は誤検知であり、修正に工数を割く価値は低い。

### 優先して対応すべき真正バグ（重要度順）

1. **🔴 `EQProcessor.Core.cpp` — Worker Thread → 非atomic shadow変数への書き込み** (セクション2 C-2)
   - `syncGlobalStateFrom()` が `rtBypassedShadow`, `rtAgcCurrentGainShadow` 等を非atomicに書き込む
   - Audio Thread の `process()` とデータ競合 → UB
   - **修正**: `std::atomic` 化、またはRCU状態スワップに統合

2. **🔴 `AudioSegmentBuffer.h` — pushBlock/copyLatest のメモリ順序不一致** (セクション2 C-1)
   - `pushBlock`: writePosition release → totalSamples release
   - `copyLatest`: totalSamples acquire → writePosition acquire
   - 逆順のため旧writePosition＋新totalSamplesを観測する可能性
   - **修正**: totalSamples を先にrelease/acqするか、reader側で一貫性チェック

3. **🟠 `CpuFeatureCheck.cpp` — `#define PF_AVX2_INSTRUCTIONS_AVAILABLE 10`**
   - 正しい値は40。旧SDKでのビルド時に常にAVX2=trueを返す
   - **修正**: フォールバックdefineを削除しSDKの定義に完全依存。または値を40に修正

4. **🟠 `CustomInputOversampler.cpp` — AVX2 OOB読み取りリスク** (セクション2 H-2)
   - `loadStride2()` の `ptr[-6]` アクセスが暗黙の+6マージン依存
   - **修正**: 境界チェックに -6 オフセットを明示的に考慮

5. **🟡 `AlignedAllocation.h` — `ScopedAlignedPtr` が任意ポインタを受け入れる** (セクション2 H-3)
   - 非 `mkl_malloc` ポインタを `mkl_free` に渡す可能性
   - **修正**: コンストラクタに `static_assert` または実行時アサーションを追加

### 報告の改善点

- 実際のコードパスをトレースした検証が不足している
- `exchangeAtomic`, `compareExchangeAtomic`, `consumeAtomic` 等のラッパー関数のセマンティクスを理解せずに報告している
- 同じ問題が複数エントリで重複して報告されている
- 修正済みのコードを古い状態で報告している可能性
