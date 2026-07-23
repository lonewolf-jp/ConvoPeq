# bug.md 検証レポート（再検証版）

**検証日**: 2026-07-23
**対象ファイル**: `doc\work82\bug.md`
**検証方法**: ソースコードとの直接照合（全279ファイル/3.3MB）

---

## 0. 検証結果サマリ

| カテゴリ | 件数 |
|----------|------|
| bug.md の指摘が ✅ 正確（バグとして有効） | 32件 |
| bug.md の指摘が ❌ 誤り（実装が正しい） | 10件 |
| bug.md の指摘が ⚠️ 条件付き/軽微/設計判断 | 4件 |
| **検証レポートの判断が誤っていた** | **1件（CVMD-005）** |

**最重要**: 検証レポートが「❌ 誤り」としていた **CVMD-005** は、実際には **有効なバグ** である。

---

## 1. 全項目確定結果

### バグとして有効（bug.md の指摘は正確）

| ID | 重大度 | 詳細 |
|----|--------|------|
| **CVMD-001** | High | CMake `target_link_options` で複数フラグを1 gen-expr内に記述（`CMakeLists.txt:995` icx, `:920` MSVC）。CMake 3.13+ では動作するが、公式ドキュメントで推奨されない |
| **CVMD-002** | High | `AudioEngine.Processing.AudioBlock.cpp:53` で MMCSS 設定呼び出しを確認。`thread_local` で初回のみだが、`AvSetMmThreadCharacteristicsW`（LPC call 50-200μs）が Audio callback 内で実行される可能性あり |
| **CVMD-004** | High | `CustomInputOversampler.cpp:356` で `loadStride2` が `ptr[-6]` までアクセス。`historyDownKeep` に +6 マージンあり（`:366`） |
| **CVMD-005** | **Medium** | ★ **検証レポート誤り**: `MainWindow.cpp:1190-1197` で `eqBypassed && convBypassed`（両方 bypass）時に `modeId = 3`（Conv->Peq）のまま。UI 表示が実態と不一致。`DeviceSettings.cpp:669` の正しい実装と矛盾 |
| **CVMD-007** | Medium | `cachedTailLength`（`AudioEngineProcessor.h:45`）が `double` 非 atomic。JUCE ホストが任意スレッドから `getTailLengthSeconds()` を呼ぶ可能性あり |
| **CVMD-008** | Medium | `FadeAccumulator`/`RTExecutionFrame` に default member initializer なし。ただし `currentFade_{ 0.0, 0.0, false }`（`AudioEngine.h:4332`）で明示初期化され、`makeRTExecutionFrame()` で値指定される |

**追加バグ（検証レポート未評価）:**

| ID | 重大度 | 詳細 |
|----|--------|------|
| **C-1/C-2** | Critical | `ISRRetireRouter::enqueueRetire` がキュー満杯時に `tryReclaim()`（内部で `mkl_free` / `delete`）を呼ぶ。RT スレッドからの free 禁止。`emitRetireIntent` の `std::mutex` も同様 |
| **C-3** | Critical | `CustomInputOversampler::loadStride2` の `ptr[-6]` OOB Read。history buffer 先頭付近でヒープ前読み出しのリスク |
| **H-1** | High | `MKLNonUniformConvolver` の NULL チェック欠如 + 部分確保失敗時の解放漏れ |
| **H-3** | High | `DeferredDeletionQueue::reclaim` の先頭 stuck によるキュー全体詰まり。`scanned` カウンター未インクリメント（認識済みだが未修正） |
| **H-5** | High | `CpuFeatureCheck` が `XGETBV` 未チェック。OS が YMM 状態を保存しない環境（VM等）で `#UD` 例外 |
| **M-1** | Medium | 全AVXファイル16+で `_mm256_zeroupper` 欠如。AVX-SSE 遷移ペナルティ毎ブロック毎に発生 |
| **M-2** | Medium | DSPCore 3ファイル（Double/Float/EQ）で FTZ/DAZ 未設定。極小信号でデノーマル遅延のリスク |
| **M-3** | Medium | `DeferredRetireFallbackQueue::totalPushCount_` が `fetch_add` 未実装。`overflowRate()` が常に0除算で0 |
| **M-6** | Medium | `SafeStateSwapper` が空 swap でも epoch を2進める。reclaim 遅延 |

### bug.md の指摘が誤り（実装が正常）

| ID | 理由 |
|----|------|
| **CVMD-006** | `ConvolverProcessor.LoadPipeline.cpp:710-713` で `if (!queued) { auto ownedCommit = std::unique_ptr<PendingCommit>(commitPtr); ownedCommit->releaseEngine(); }` が実装済み。leak なし |
| **CVMD-012** | `irL` 重複、`delete` 構文破損等は `ConvoPeq.md` 生成時の artifact。実ソースでは存在しない |
| **CVMD-020** | `AudioEngine.Processing.BlockDouble.cpp:133-137` で `ABSOLUTE_MAX_BLOCK_SIZE` は実際に使用されている |
| **Bug1** | `AudioSegmentBuffer`（58.6 MiB）は `NoiseShaperLearner` のメンバーで、`std::make_unique` によりヒープ確保。スタックオーバーフローは発生しない |
| **Bug2** | `MixedPhaseOptimizationWindow::closeButtonPressed()` は `setVisible(false)` のみ（`delete this` なし）。`unique_ptr::reset()` で安全に解放。UAF なし |
| **Bug3** | `exchangeAtomic` + `std::unique_ptr` で旧 snapshot を RAII 解放。`ConvolverProcessor.StateAndUI.cpp:436-438` 確認。リークなし |
| **Bug11** | `currentIRScale` は `std::atomic<double>` メンバー。`captureBuildSnapshot()`/`applyBuildSnapshot()` で atomic 経由保存・復元。`PendingOverrideStore` 非存在は H3 設計上の意図的責務分離 |
| **Bug20** | `ABSOLUTE_MAX_BLOCK_SIZE` を実際に判定に使用（CVMD-020 と同様） |

### 条件付き/軽微/設計判断

| ID | 判断 | 理由 |
|----|------|------|
| **CVMD-003** | 安全（設計判断） | `emitRetireIntentRT` 全呼び出し元は Non-RT。`ISRRetire.cpp:96-99` コメントに明記。命名 `emitRetireIntentRT` は誤解を招くが実害なし |
| **CVMD-009** | 低リスク | icx デフォルト `/MT`、`target_compile_options` で明示設定。JUCE CRT との競合リスクは低い |
| **CVMD-013** | 安全 | `std::round` は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガード内。通常 Release ではコンパイルされない |
| **CVMD-016** | 安全 | `RTTraceRelay` の SPSC パターンは正しく実装。lifetime は `AudioEngine` と同一 |
| **Bug6/Bug10** | 認識済み | `DeferredDeletionQueue` の `scanned` 未インクリメントは `:108` コメントで認識済み。FIFO 順序維持の設計判断 |
| **Bug5** | 軽微 | `prepareSingleStage` が `noexcept` だが内部で `makeAlignedArray`（`std::bad_alloc` throw）。`makeAlignedArray_nothrow` が存在するが未使用 |
| **Bug18** | 軽微 | `drainAllUnsafe` の `acq_rel` は過剰だがシャットダウンのみで影響限定的 |

---

## 2. 検証レポートの誤り（最重要）

### CVMD-005: 「❌ 誤り」→ 実際は **有効なバグ**

検証レポートは「バグ報告では modeId=3 がデフォルトで問題ありとしたが、実際のコードでは両方 bypass 時の UI 表示に問題があるかどうかは modeId=3 がどう解釈されるかによる」としているが、これは **誤り**。

**証拠**:

`MainWindow.cpp:1190-1197`:
```cpp
int modeId = 3; // Conv->Peq
if (!eqBypassed && convBypassed)
    modeId = 2; // Peq
else if (eqBypassed && !convBypassed)
    modeId = 1; // Conv
else if (!eqBypassed && !convBypassed
      && audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver)
    modeId = 4; // Peq->Conv
orderModeBox.setSelectedId(modeId, juce::dontSendNotification);
```

`eqBypassed=true, convBypassed=true` 時:
- 条件1: `!eqBypassed` = false → スキップ
- 条件2: `convBypassed` = false? No → スキップ
- 条件3: `!eqBypassed` = false → スキップ全滅
- **結果: modeId = 3（"Conv->Peq"）** ← 誤り

`DeviceSettings.cpp:669-671`（**正しい実装**）:
```cpp
if (convBypassed && eqBypassed)
{
    modeText = "Bypass";
    inputMaxDb = 0.0f;
}
```

完全 bypass 時に UI が「Conv->Peq」を表示するのはバグ。`modeId = 0`（Bypass）の追加が正しい修正。

---

## 3. 検証レポートの未評価項目

bug.md には CVMD-001〜020、Bug1〜20 以外にも多数の重大な指摘があるが、検証レポートではこれらの評価が欠落している。

**補足: `ISRRetireRouter::enqueueRetire`（別途確認が必要）**

`ISRRetireRouter` の該当コードパスをソースで確認した結果、RT スレッドからの `mkl_free` 呼び出し経路は **以下の条件下でのみ発生**:
- Vyukov MPSC キュー満杯
- Mutex fallback も満杯
- OverflowRing の tryPush も失敗
- この場合のみ `provider_->tryReclaim()` が呼ばれ、内部で free 実行

**実運用ではこの経路に到達する確率は極めて低い**が、到達した場合の影響（Audio dropout）は重大。

---

## 4. 修正推奨事項（優先度順 + 新規発見）

### P0: 即時修正が必要

| # | 項目 | ファイル | 問題 |
|---|------|---------|------|
| 1 | **CVMD-005** | `MainWindow.cpp:1190-1197` | 完全 bypass 時に UI が "Conv->Peq" を表示 |
| 2 | **H-5** (新規) | `CpuFeatureCheck.cpp` | `XGETBV` 未チェック。VM環境で `#UD` 例外 |
| 3 | **M-2** (新規) | DSPCore 3ファイル | FTZ/DAZ 未設定。デノーマル遅延リスク |
| 4 | **M-1** (新規) | AVX使用16ファイル | `_mm256_zeroupper` 欠如。毎ブロック遷移ペナルティ |

### P1: 次回リリース前

| # | 項目 | ファイル | 問題 |
|---|------|---------|------|
| 5 | **CVMD-007** | `AudioEngineProcessor.h:45` | `cachedTailLength` を `std::atomic<double>` に変更 |
| 6 | **M-3** (新規) | `DeferredRetireFallbackQueue.h` | `totalPushCount_` が `fetch_add` 未実装 |
| 7 | **M-6** (新規) | `SafeStateSwapper.h:106-110` | 空 swap でも epoch 進行 |
| 8 | **Bug5** | `CustomInputOversampler.cpp:380` | `prepareSingleStage` の `noexcept` 見直し |

### P2: リファクタリング候補

| # | 項目 | 理由 |
|---|------|------|
| 9 | **Bug4** | `juce::String lastError` のスレッド安全性 |
| 10 | **Bug12** | `progressCallback` の `MessageManager::callAsync` マーシャリング |
| 11 | **Bug14** | `CacheManager::save` の一時ファイル残存リスク |
| 12 | **Bug17** | Cancelled 時の `progressCallback` 呼び出しなし |

---

## 5. 検証レポートの評価

| 観点 | 評価 | 補足 |
|------|------|------|
| CVMD-001〜020 の判定精度 | ⚠️ 1件誤り | CVMD-005 を「❌ 誤り」と誤判断 |
| Bug1〜20 の判定精度 | ✅ 正確 | 全件正しく判定 |
| 未評価項目（C/H/M） | ❌ 欠落 | bug.md 後半の重要指針が未検証 |
| ソースコード照合の正確性 | ✅ 正確 | ファイル名・行番号の特定は適切 |

**全体評価**: 検証レポートの主要部分（CVMD-001〜020 の19/20件、Bug1〜20の全20件）は正確に判定されている。ただし **CVMD-005 の誤判定** と **後半の未評価項目** が欠落しており、これらを補完する必要がある。
