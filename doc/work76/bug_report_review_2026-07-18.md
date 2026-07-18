# ConvoPeq バグ報告書 検証レポート

**検証日**: 2026-07-18
**対象**: `doc/work76/ConvoPeq_bug_report_2026-07-17.md`
**検証方法**: 実ソースコード精査 + git履歴確認（commit `8956ca7` の変更を含む）
**使用ツール**: WSL grep/rtk, 直接ファイル読み取り, AiDex インデックス

---

## 総評

本バグ報告書は**非常に高品質で正確**である。報告された8件のバグはすべて実在が確認され、誤検知として除外した10候補も正当な理由をもって正確に除外されている。

報告書作成（2026-07-17）後、即座に修正作業が行われ、commit `8956ca7`（2026-07-18 00:09）により**8件中6件が修正済み**である。残り2件（Bug#5, Bug#8）は未修正だが、Bug#5は実害が限定的、Bug#8は修正が強く推奨される。

---

## バグ別検証結果

### Bug#1 [Critical] `Layer::delayLineBuf` 解放漏れ

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ✅ **修正済み** (commit `8956ca7`) |
| 根拠の正確性 | ✅ 完全に正確。診断ビルドの `freeAll()` で `freeTracked` が欠落していること、`LayerAllocSizes` にフィールドがないこと、`mkl_malloc` が `DIAG_MKL_MALLOC` でないことを正確に指摘。 |
| 修正の確認 | `LayerAllocSizes` に `size_t delayLineBuf = 0;` 追加済み。`SetImpulse()` で `DIAG_MKL_MALLOC` + `allocSizes.delayLineBuf = delayLineBytes` に修正済み。`freeAll()` の診断/非診断両ブランチで解放処理が追加済み。 |
| 影響評価 | ✅ 適切。「診断ビルドでIR再ロードごとに確実にリーク」の指摘は正確。 |

### Bug#2 [Critical] Retireキュー枯渇時のポインタロスト

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ✅ **4箇所中3箇所修正済み + Site 3は呼び出し元でカバー** |
| 根拠の正確性 | ✅ 4箇所の特定、各呼び出し元のコード引用、overflowCount監視の説明、すべて正確。 |
| Site 1 (`DSPLifetimeManager::retire`) | ✅ 修正済み。`router_->enqueueWithRetry()` に委譲。 |
| Site 2 (`AudioEngine` 系) | ✅ 修正済み。`enqueueDeferredDeleteNonRtWithResult()` が `enqueueWithRetry` を使用。`retireDSP()` はデッドコードとして削除。 |
| Site 3 (`RuntimePublicationCoordinator::enqueueRetire`) | ⚠️ 単体ではリトライなしだが、唯一の実質的呼び出し元 (`EQProcessor.Core.cpp`) が `enqueueWithRetry` へのフォールバックを持つ。 |
| Site 4 (`RefCountedDeferred<T>::release`) | ✅ 修正済み。`router.retire()` 経由で `enqueueWithRetry` に委譲。 |
| `enqueueWithRetry` 実装 | `ISRRetireRouter` に集約。kMaxRetry=2 + tryReclaim の有界ループ + QueuePressure 通知。500ms クールダウン付き即時 tryReclaim も併用。 |
| 影響評価 | ✅ 適切。ただし overflowCount の実測監視が未実施であり、理論上のリスクを超えるかは未確認。 |

### Bug#3 [High] `fastTanh` 閾値と係数の不整合

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ✅ **部分的に修正済み**（スカラー/SIMD不整合は解消、数学的課題は別チケット） |
| 数値解析の正確性 | ✅ 完全に正確。`x=4.5` での関数値と早期returnの不連続、スカラー/SIMD間の挙動差の分析は数学的に正しい。 |
| 修正内容 | 共有ユーティリティ `convo::dsp::fastTanh<DefaultFastTanhPolicy>` に委譲。`DefaultFastTanhPolicy` は同じ27/9係数＋閾値4.5を維持。スカラー版とSIMD版で同一の `compute()` を使用するため、経路間不整合は解消。 |
| 未解決課題 | コメントに「Padé近似の変更（5次/6次）は別チケットで実施」と明記。`\|output\|>1.0` 問題は残存。 |
| 高次近似 | `SoftClipPadéPolicy`（5次/6次, 10395係数, 閾値4.5, x=4.5で≈0.99927）が別途定義済み。 |
| 影響評価 | ✅ 適切。開発者のサウンドデザイン意図を尊重したバランスの取れた判断。 |

### Bug#4 [High] `ProgressiveUpgradeThread` のFTZ/DAZ未設定

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ✅ **修正済み** (commit `8956ca7`) |
| 根拠の正確性 | ✅ 他スレッドとの比較、`IRDSP::resampleIR` のデノーマル発生リスクの指摘、すべて正確。 |
| 修正の確認 | `ProgressiveUpgradeThread.cpp` に `<xmmintrin.h>`/`<pmmintrin.h>` インクルード追加、`run()` 先頭で `_MM_SET_FLUSH_ZERO_MODE` + `_MM_SET_DENORMALS_ZERO_MODE` 設定済み。 |
| 影響評価 | ✅ 適切。「2エンジン共存時間の延長がメモリ削減効果を相殺」の指摘は洞察が深い。 |

### Bug#5 [Medium] `DeferredDeletionQueue::reclaim()` のデッドコード

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ❌ **未修正** |
| 制御フロー解析の正確性 | ✅ 完全に正確。`scanPos` が常に `deqPos` と等しいため、`else` 節に到達するのは `!canDelete` の場合のみ → `break` で即終了。`kMaxScan` による先読みロジックは確かに到達不能。 |
| MPMC逆転リスクの指摘 | ✅ 妥当。複数プロデューサが同時にenqueueする場合、epoch逆転の理論的可能性は存在する。ただし実運用での発生頻度は不確定。 |
| 影響評価 | ✅ 適切。実害は限定的（先頭エントリが古くなれば通常通り回収される）。ただし `kMaxScan=1024` の設計意図と実装の乖離はメンテナンス上のリスク。 |
| 推奨追記 | 単純化（デッドコード削除）とepoch逆転対策（先読み実装）のトレードオフを実測に基づき判断すべき。 |

### Bug#6 [Low/Medium] `std::abs` と `absNoLibm` の不整合

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ✅ **修正済み** (commit `8956ca7`) |
| 根拠の正確性 | ✅ 同一ファイル内の規約不整合、呼び出し箇所の特定、すべて正確。 |
| 修正の確認 | `delayLineReadAdd()` 内の3箇所の `std::abs` がすべて `absNoLibm` に置換済み。 |
| 影響評価 | ✅ 適切な「Low/Medium」評価。実害は環境依存だが、規約違反である点を正確に指摘。 |

### Bug#7 [Low] ブレース漏れによる `allocSizes.tailOutputBuf` の誤設定

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ✅ **修正済み** (commit `8956ca7`) |
| 根拠の正確性 | ✅ コードの正確な引用、「実害なし」の検証まで含めて完全に正確。 |
| 修正の確認 | `allocSizes.tailOutputBuf = l.partSize * sizeof(double);` が `if (!l.isImmediate) { ... }` ブロック内に移動済み。 |
| 影響評価 | ✅ 「実害なし」の判断も正確。ポインタnullチェックにより安全網が機能することを確認。 |

### Bug#8 [Critical] `~NoiseShaperLearner()` のUse-After-Free

| 項目 | 結果 |
| --- | --- |
| 報告の正確性 | ✅ **正確** |
| 現在の状態 | ❌ **未修正** |
| メンバ宣言順の分析 | ✅ 正確。宣言順: `workerThread` → ... → `evaluationWorkers` → ... → `candidatePopulation/Fitness/sharedMappedPopulation`。C++の逆順破棄により、CMA-ESバッファがスレッドより先に破棄される。 |
| デストラクタの欠陥 | ✅ `~NoiseShaperLearner()` は `workerThread.request_stop()` のみで `join()` なし。`evaluationWorkers[]` の join もなし。 |
| `stopLearning()` の欠陥 | ✅ 補助スレッドは `stopEvaluationWorkers()` で正しくjoinするが、主スレッドは `request_stop()` のみ。`startLearning()` 自身がこの不足を認識し、事前に明示的 join を行っている。 |
| 他クラスとの比較 | ✅ `AudioEngine::~AudioEngine()`, `WorkerThread::~WorkerThread()`, `DeferredFreeThread::~DeferredFreeThread()` はいずれも正しい join パターンを実装している。 |
| 影響評価 | ✅ 「タイミング依存でたまに落ちる」系の不具合として現れる可能性がある、という評価は正確。 |
| **推奨**: | デストラクタに `workerThread.join()` + `evaluationWorkers[].join()` ループを追加すべき。またはスレッドメンバをクラスの末尾に移動する構造的対策も有効。 |

---

## 誤検知除外候補の検証結果

報告書で「誤検知として除外」とされた10候補について、以下を確認した：

| # | 候補 | 報告書の判断 | 検証結果 |
| --- | --- | --- | --- |
| 1 | AoS/SoA二重保持 | 既知課題・修正済み | ✅ 確認済み |
| 2 | NaNサニタイズFloat経路欠如 | 両経路とも保護あり | ✅ 確認済み。`processOutput()`（`DSPCoreIO.cpp`）にAVX2 NaN/Infスクラブ存在。 |
| 3 | `pushAdaptiveCaptureBlocks` 重複 | 匿名名前空間の別定義 | ✅ 確認済み。内部リンケージのため問題なし。 |
| 4 | `prevSample`/`softClipPrevSample` 未使用 | ADAA用に意図的保持 | ✅ 確認済み。コメントに明記。 |
| 5 | SoftClip Float/Double実装差異 | 最終NaNスクラブ通過 | ✅ 確認済み。 |
| 6 | クロスフェード線形補間 | 相関信号に線形補間は正しい | ✅ 確認済み。Wet/Dryミックス側で等電力補間が別途使用。 |
| 7 | Biquad係数符号 | RBJ Cookbookと一致 | ✅ 確認済み。 |
| 8 | `vdRngUniform` RT呼び出し | リングバッファ枯渇時にfallback | ✅ 確認済み。 |
| 9 | 状態クランプ上限値差異 | NaN保護の多層防御 | ✅ 確認済み。 |
| 10 | 他クラスの同種破棄順序問題 | すべて正しいjoin実装 | ✅ 確認済み。3クラスすべて確認。 |

---

## 修正状況サマリ

| # | 重大度 | 状態 | 修正commit |
| --- | --- | --- | --- |
| 1 | Critical | ✅ 修正済み | `8956ca7` (2026-07-18) |
| 2 | Critical | ⚠️ 大半修正済み（Site 3のみ呼び出し元でカバー） | `8956ca7` |
| 3 | High | ✅ 部分修正（不整合解消、数学的課題は別チケット） | `8956ca7` |
| 4 | High | ✅ 修正済み | `8956ca7` |
| 5 | Medium | ❌ **未修正** | - |
| 6 | Low/Medium | ✅ 修正済み | `8956ca7` |
| 7 | Low | ✅ 修正済み | `8956ca7` |
| 8 | Critical | ❌ **未修正** | - |

**関連ファイル**: `src/dsp/math/FastTanhApprox.h`（新規作成）, `src/core/ScopedMXCSR.h`（新規作成）, `src/core/IRetireRouter.h`（新規作成）

---

## 総合評価

### 報告書の品質: ★★★★★（優秀）

- **正確性**: 全8バグが実在確認。数値解析（Bug#3のPadé近似）、制御フロー解析（Bug#5の到達不能コード）、メモリモデル解析（Bug#8のC++破棄順序）のいずれも正確。
- **網羅性**: 修正パッチの提示、影響範囲の定量評価、関連コードの横断調査まで含めた包括的な調査。
- **客観性**: 誤検知10件を独立して記録し、推測ベースの指摘を排除している。
- **実用性**: 全バグに修正パッチ（diff形式）が付属し、即座に適用可能。

### 未修正バグの優先度

1. **Bug#8 (Critical)**: `~NoiseShaperLearner()` のUse-After-Free — **最優先で修正推奨**。タイミング依存のクラッシュ原因。修正はデストラクタへの `join()` 追加のみで完了する。
2. **Bug#5 (Medium)**: `reclaim()` のデッドコード — 優先度低。epoch逆転の実測が取れてから本格対応でよいが、少なくともコメントと定数を実態に合わせる軽微な整理は推奨。

### 報告書の限界

報告書自身が認めている通り、261ファイル中26ファイルの調査で残りは未精査。RCU中核の `SafeStateSwapper.h`、状態マシン `ConvolverProcessor.StateAndUI.cpp`、CMA-ES数値アルゴリズムなどに潜在的な問題がないとは言い切れない。
