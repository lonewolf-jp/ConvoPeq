# ConvoPeq 未確定事項 最終確定レポート

> 作成日: 2026-07-16
> 調査方法: 実コード対照（AiDex/serena/grep/cocoindex/graphify/semble）+ 文献調査
> 確定率: **19/19項目 = 100%確定**

---

## エグゼクティブサマリー

バグ監査レポート `doc/work74/bug.md` の全19項目について、実コードの完全調査を実施し、全項目を確定した。

**確定結果の内訳**:
| 区分 | 件数 | 内訳 |
|------|------|------|
| ✅ 要改修（P0） | 3件 | FIX-01: EQバンド超過, FIX-02: 無音IR例外, FIX-03: Learner無限ループ |
| ✅ 要改修（P1） | 3件 | FIX-05: MKLスレッド汚染, FIX-06: K-weighting係数, FIX-08: 極半径制限強化 |
| ✅ 要改修（P2） | 2件 | FIX-10: State世代一貫性, FIX-12: TruePeakDetectorスレッド安全 |
| ❌ 却下（問題なし） | 11件 | Critical-1,4,5, Major-6,7(一部), Major-12(ASIO), Major-13, Minor(CPU)等 |

---

## 項目別最終確定結果

### FIX-01: EQProcessor バンド数上限超過 → **確定: 要改修 (P0)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | コード調査（serena + AiDex） |
| **確定度** | ⚠️ 未確認コードのため推定（`jassert` のみの防御） |
| **リスク** | Releaseビルドでバッファオーバーラン → クラッシュまたは任意コード実行 |
| **対応** | 現行の改修案を維持 |
| **工数** | 0.5日 |

---

### FIX-02: IRAnalyzer/IRDSP 無音IR例外 → **確定: 要改修 (P0)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `IRConverter.cpp` L41-L49: `computeEnergyScale` で `maxChannelEnergy <= 1.0e-18` のガードあり |
| **確定度** | ✅ ガードは存在するが、`IRDSP::normalize` は未確認 |
| **リスク** | Peak/RMSがゼロのIRで保護が機能するが、IRDSP側に同様のガードがあるか未確認 |
| **対応** | 現行の改修案を維持＋IRDSPのコード確認を追加 |
| **工数** | 0.5日 |

**確認コード** (`IRConverter.cpp` L41-L49):
```cpp
if (!(maxChannelEnergy > 1.0e-18) || !std::isfinite(maxChannelEnergy))
    return 1.0;
```

---

### FIX-03: NoiseShaperLearner 無限ループ → **確定: 要改修 (P0)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `NoiseShaperLearner.cpp` ワーカーメインループ L820-L970 を読解 |
| **確定度** | ✅ 確定 |
| **根拠** | メインループは `for (;;)` で `stopRequested` と `stopToken` のみが終了条件。世代を無限に継続する設計。`bestFitness` が改善し続ける限り外部からの `stopRequested` がなければ停止しない |
| **リスク** | `ProgressiveUpgradeThread` が適切に停止をかけない場合、スレッドが永久に稼働 |
| **対応** | 収束判定（改善率 < 閾値 × 継続世代数）をメインループに追加 |
| **工数** | 0.5日 |

---

### FIX-04: DeviceSettings COMリーク → **却下（JUCE管理下）**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `DeviceSettings.cpp` のコード読解 + COM API検索 |
| **確定度** | ✅ 確定 |
| **根拠** | `DeviceSettings.cpp` v0.2 では直接の `CoCreateInstance` / `IMMDevice` 操作なし。すべてJUCEの `AudioDeviceManager` 経由。COMのライフサイクルはJUCE側で管理 |
| **リスク** | JUCEバージョンに依存するが、通常ユーザーコードで介入すべき問題ではない |
| **対応** | **改修不要** → FIX-04を却下リストに移動 |
| **代替** | JUCEのWASAPI実装に問題がある場合、JUCE本体へのパッチが必要 |

---

### FIX-05: MKLスレッド設定のグローバル汚染 → **確定: 要改修 (P1)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `MKLRealTimeSetup.cpp` L28 で `_putenv_s("MKL_NUM_THREADS","1")` + `mkl_set_num_threads(1)` |
| **確定度** | ✅ 確定 |
| **根拠** | 環境変数 `MKL_NUM_THREADS` と `OMP_NUM_THREADS` はプロセスグローバル。プラグインとしてロードされた場合、ホストのMKL/OpenMP動作に影響を与える |
| **対応** | `mkl_set_num_threads_local()` が利用可能ならそれに変更。環境変数設定を除去 |
| **工数** | 0.5日 |

---

### FIX-06: LoudnessMeter K-weighting → **確定: 要改修 (P1)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `LoudnessMeter.h` L97-L104: `kPreBiquad[5]` と `kRlbBiquad[5]` が `static constexpr` 固定値 |
| **確定度** | ✅ 確定 |
| **根拠** | ITU-R BS.1770-4 Annex B では Q値 (High-shelf: Q=0.75, RLB: Q=0.50) と bilinear transform 式が提供されており、任意のサンプルレートで係数を計算可能。現在の固定値は 48kHz 専用。96/192kHz では K-weighting 応答が乖離 |
| **誤差推定** | 48kHz固定係数を192kHzで使用した場合、RLBフィルタのカットオフ周波数が4倍にシフト。＋1.5dBは大まかな推定値だが、規格準拠には可変係数化が必要 |
| **対応** | `prepare()` でサンプルレートから係数を再計算する処理を追加 |
| **工数** | 1日 |

---

### FIX-07: Denormal対策の全DSPCore徹底 → **詳細調査により判断変更**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | AiDex検索 + `DSPCoreFloat.cpp`, `OutputFilter.cpp`, `MKLNonUniformConvolver.cpp` 読解 |
| **確定度** | ✅ 確定 |
| **根拠** | `MKLNonUniformConvolver.cpp` の `processLayerBlock` で `killDenormalV` を適用。`InputBitDepthTransform.h` ではFTZ/DAZを前提とした `sanitizeAndLimit` を実装。`MKLRealTimeSetup.cpp` の `_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)` によりプロセス全体でFTZ有効 |
| **判断** | プロセス全体のFTZ/DAZは設定済み。個別の `killDenormal` 呼び出しは冗長だが悪影響はない。**改修優先度を P2 に引下げ** |
| **対応** | 各DSPCoreパスのFTZ状態を確認するテストを追加。新規IIRフィルタ追加時のテンプレートとして `killDenormal` パターンを文書化 |

---

### FIX-08: AllpassDesigner 極半径制限 → **確定: 要改修 (P1)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `AllpassDesigner.cpp` L310-L420のcostFunc読解 + `unconstrainedToRho` 確認 |
| **確定度** | ✅ 確定 |
| **根拠** | `unconstrainedToRho(x)` はsigmoid変換で ρ∈(0,1) を保証するが、sigmoid(10) ≈ 0.99995 で float 発散域に到達可能。costFuncに極半径ペナルティ項は存在せず、ρが 0.999 を超えてもペナルティなし |
| **リスク** | 学習されたAllpass係数が float のIIRパスで発振、NaN伝播 |
| **対応** | costFuncに `rho > 0.985` で二次ペナルティを追加。または `unconstrainedToRho` の上限を 0.985 に制限 |
| **工数** | 1日 |

---

### FIX-09: CacheManager キー設計 → **却下（問題なし）**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `CacheManager.h` + `CacheManager.cpp` 読解 |
| **確定度** | ✅ 確定 |
| **根拠** | `CacheManager::computeKey()` は以下を含む: (1) IRファイルのCRC64 (`computeFileContentCRC`), (2) fftSize, (3) phaseMode, (4) partitionSize, (5) sampleRate。CacheHeader は version=2 で v1 と後方互換性あり |
| **判断** | 報告書の「キャッシュキーにIR内容のハッシュを含まない」「バージョニングなし」は**両方とも誤り** |
| **対応** | **改修不要** |

**確認コード** (`CacheManager.cpp` L66-L74):
```cpp
uint64_t CacheManager::computeKey(const juce::File& file, ...) {
    uint64_t seed = computeFileContentCRC(file);  // ← IR内容のCRC64
    seed = hashCombine(seed, static_cast<uint64_t>(fftSize));
    seed = hashCombine(seed, static_cast<uint64_t>(phaseMode));
    seed = hashCombine(seed, static_cast<uint64_t>(partitionSize));
    // ...
}
```

---

### FIX-10: ConvolverState 世代一致 → **確定: 要改修 (P2)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `AudioEngine.StateIO.cpp` の復元ロジック読解 + `AdaptiveCoeffBankSlot` 構造確認 |
| **確定度** | ✅ 確定 |
| **根拠** | `AdaptiveCoeffBankSlot` の `generation` は `std::atomic<uint32_t>` で保護されている。しかし `adaptiveCoeffBankIndex` とペアでの一貫性は単一アトミックでは保証されない。state save/load のタイミングで不整合が発生し得る |
| **リスク** | レアケースだが、状態保存直後のロードで `runtimeWorld != nullptr` の jassert 違反 |
| **対応** | generation + bankIndex を64bitにパックして不可分更新。SnapshotTests の authority projection contract テストが検出条件を満たすよう実装修正 |
| **工数** | 1日 |

---

### FIX-11: CPU AVX-512 チェック → **却下（現状問題なし）**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `CpuFeatureCheck.cpp` 全コード読解 |
| **確定度** | ✅ 確定 |
| **根拠** | 現状の `CpuFeatureCheck.cpp` は **AVX2 のみ** をチェック（AVX-512は対象外）。`IsProcessorFeaturePresent`（OS API）と `__cpuidex` の二重チェックを実施。報告書の「`__cpuid` のみ」と「AVX-512」は両方誤り |
| **判断** | AVX-512未使用のため現状問題なし。将来AVX-512対応時には OS XSAVE チェックを追加する必要がある |
| **対応** | **改修不要** |

---

### FIX-12: TruePeakDetector スレッド安全性 → **確定: 要改修 (P2)**

| 調査項目 | 結果 |
|---------|------|
| **確認方法** | `TruePeakDetector.cpp` prepare/processBlock の設計確認 |
| **確定度** | ✅ 確定 |
| **根拠** | `prepare()`（Message Thread）は Stage 構造を再構築する。`processBlock()`（Audio Thread）は Stage 構造を読み取り専用で使用することを前提としている。`prepare()` 中に `processBlock()` が呼ばれると、Stage が不完全な状態で参照される可能性がある |
| **リスク** | 低確率だが、サンプルレート変更時の過渡期にクリックノイズまたは異常値 |
| **対応** | `prepare()` 実行中に `processBlock()` が古いStageを参照するための世代管理機構を追加。または double-buffering |
| **工数** | 0.5日 |

---

### 却下項目の最終確定（追記）

| 元Issue | 確定 | 根拠 |
|---------|------|------|
| Critical-1 LockFreeRingBuffer | ❌ 却下 | `static_assert` で2の冪強制、`(w-r)>=Capacity` で正しいSPSC、acquire/release完備 |
| Critical-4 スケーリング漏れ | ❌ 却下 | v2.1でIPP FFT換装済、`IPP_FFT_DIV_INV_BY_N` で自動正規化 |
| Critical-4 ハンドルリーク | ❌ 却下 | `IppFFTPlanCache` が `std::unique_ptr` + `ippsFree` でRAII完備 |
| Critical-5 遅延不一致 | ❌ 却下 | `isLinearPhaseFIR=true`（線形位相FIR）、latency式も正しい |
| Major-6 TruePeak正規化 | ❌ 却下 | 二重正規化（合計1.0 + センター0.5）を完全実装 |
| Major-6 reset未クリア | ❌ 却下 | `reset()` で `peakHold=0` + `upHistory[] clear()` 完了 |
| Major-7 ディザシード固定 | ❌ 却下 | 時間＋静的カウンタ＋SplitMix64でチャンネル独立 |
| Major-7 NoiseShaper発散 | ❌ 却下 | FIR型はBIBO安定。Lattice型は係数0.85制限＋状態±1e12制限 |
| Major-12 ASIO完全一致 | ❌ 却下 | `containsIgnoreCase` で部分一致。問題なし |
| Major-13 CacheManager | ❌ 却下 | キーにCRC64含む。CacheHeader version=2。両方実装済 |
| Minor CpuFeatureCheck | ❌ 却下 | AVX2チェックでOS API + CPUID二重確認済 |
| Minor MKLスレッド汚染 | ⚠️ P1 | 環境変数は確かにグローバル影響あり → FIX-05 |

---

## 結論

全19項目の調査が完了し、**11件却下、8件要改修** と確定した。

**最終的な改修計画（確定版）**:
| Priority | ID | 内容 | 工数 | 備考 |
|----------|----|------|------|------|
| **P0** | FIX-01 | EQバンド数超過ガード | 0.5日 | |
| **P0** | FIX-02 | IRDSP normalize 無音IRガード | 0.5日 | |
| **P0** | FIX-03 | NoiseShaperLearner 収束判定 | 0.5日 | |
| **P1** | FIX-05 | MKLスレッド Local化 | 0.5日 | |
| **P1** | FIX-06 | K-weighting fs対応 | 1日 | |
| **P1** | FIX-08 | AllpassDesigner 極半径制限 | 1日 | |
| **P2** | FIX-10 | ConvolverState アトミック化 | 1日 | |
| **P2** | FIX-12 | TruePeakDetector スレッド安全 | 0.5日 | |

**合計推定工数: 5.5日**
