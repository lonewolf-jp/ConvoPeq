# C3: LatencyService / WarmupService 分離設計調査

作成日: 2026-06-06
調査者: GitHub Copilot (AI Assistant)
位置付け: 概念設計段階 — 現状では未実装のため調査のみ

---

## 1. 現状レイテンシ管理の分散状況

レイテンシ関連のコードは以下の4箇所に分散している。専用サービス/クラスは存在しない。

### 1.1 レイテンシ計算 (AudioEngine直メンバ)

| 場所 | 責務 | 呼び出し元 |
|------|------|-----------|
| `AudioEngine::getCurrentLatencyBreakdown()` | 総レイテンシ算出 (OS+Convolver) | UI / 診断 |
| `AudioEngine::estimateOversamplingLatencySamples()` | OSフィルター遅延推定 | 内部 |
| `AudioEngine::estimateRuntimeLatencyBaseRateSamples()` | publish時の推定レイテンシ | `RuntimeBuilder` |
| `AudioEngine::setFixedLatencySamples()` | PDC固定遅延設定 | UIパラメーター変更 |
| `AudioEngine::getProcessingSampleRate()` | 処理サンプルレート取得 | 内部 |

### 1.2 レイテンシ補償 (Audio Thread 内)

| 場所 | 責務 |
|------|------|
| `DSPCore::fixedLatencyBufferL/R` | PDC用遅延バッファ (AudioEngine.h:488-533) |
| `DSPCore::configureFixedLatencySamples()` | 遅延量設定 (Audio Engine スレッド外) |
| `DSPCore::applyFixedLatencyDelay()` | Audio Thread 内遅延適用 |

### 1.3 レイテンシセマンティック (RuntimeWorld)

| 場所 | 責務 |
|------|------|
| `RuntimeState::latency` (`LatencySemantic`) | publishされたレイテンシ状態 |
| `LatencySemantic::latencyDelayOld/New/Delta` | クロスフェード前後の遅延差管理 |

### 1.4 ConvolverProcessor レイテンシ

| 場所 | 責務 |
|------|------|
| `ConvolverProcessor::getLatencyBreakdown()` | コンボルバー個別レイテンシ算出 |
| `ConvolverProcessor::LatencyBreakdown` | algorithm/IR-peak/total breakdown |
| `ConvolverProcessor::refreshLatency()` | コンボルバーレイテンシ更新 (RT) |

---

## 2. 現状 Warmup 管理の分散状況

Warmup も専用サービス/クラスは存在しない。

### 2.1 ビルド時Warmup

| 場所 | 責務 |
|------|------|
| `RuntimeBuilder::validateWarmup()` | 新DSPの初期化完了確認 |
| `shouldRetryWarmupFailure()` | Warmup失敗時のリトライ判断 |
| `AudioEngine.RebuildDispatch.cpp:800-814` | Warmup失敗→リトライ or 破棄の全体フロー |

### 2.2 Convolver内部Warmup状態

| 場所 | 責務 |
|------|------|
| `MKLNonUniformConvolver::warmupCompleted` | コンボルバー内部warmup完了フラグ (atomic) |
| 同 `debugWarmupGuardCount()` | デバッグ用カウンタ |

---

## 3. 想定される分離設計案

### 3.1 LatencyService 設計案

```cpp
// LatencyService: レイテンシ計算・PDC管理をAudioEngineから分離
class LatencyService {
public:
    struct Breakdown {
        int oversamplingBaseRateSamples = 0;
        int convolverAlgorithmBaseRateSamples = 0;
        int convolverIRPeakBaseRateSamples = 0;
        int convolverTotalBaseRateSamples = 0;
        int totalBaseRateSamples = 0;
    };

    explicit LatencyService(AudioEngine& engine) noexcept;

    [[nodiscard]] Breakdown getCurrentBreakdown() const noexcept;
    [[nodiscard]] int getCurrentLatencySamples() const noexcept;
    [[nodiscard]] int estimateRuntimeLatencyBaseRateSamples(
        const AudioEngine::DSPCore* dsp, bool forCrossfade) const noexcept;

    // PDC (固定遅延補償)
    void configureFixedLatency(int samples, int maxBlockSize) noexcept;
    void applyFixedLatencyDelay(double* dataL, double* dataR, int numSamples) noexcept;

private:
    AudioEngine& engine_;
    // PDC バッファは DSPCore から移設
    convo::ScopedAlignedPtr<double> fixedLatencyBufferL_;
    convo::ScopedAlignedPtr<double> fixedLatencyBufferR_;
    int fixedLatencyBufferSize_ = 0;
    int fixedLatencySamples_ = 0;
    int fixedLatencyWritePos_ = 0;
};
```

**課題**: PDCバッファ (`fixedLatencyBufferL/R`) は現在 `DSPCore` のメンバ。Audio Thread から直接アクセスされるため、分離には Audio Thread パスも変更が必要。

### 3.2 WarmupService 設計案

```cpp
// WarmupService: rebuild 後の Warmup 検証を RuntimeBuilder から分離
class WarmupService {
public:
    explicit WarmupService(AudioEngine& engine) noexcept;

    // 新DSPのWarmup検証
    [[nodiscard]] convo::BuildError validateWarmup(
        const AudioEngine::DSPCore& dsp) noexcept;

    // リトライ判断
    [[nodiscard]] bool shouldRetry(const AudioEngine::DSPCore& dsp) const noexcept;

private:
    AudioEngine& engine_;
    // Warmup 状態の追跡 (必要に応じて)
    std::atomic<int> warmupRetryCount_{0};
};
```

**課題**: `validateWarmup()` は現在 `RuntimeBuilder` のメソッド。分離すると `RuntimeBuilder` から `RuntimePublicationOrchestrator` への依存が増える可能性がある。

---

## 4. 分離の優先度評価

| サービス | 現状の分散度 | 分離の価値 | 実装コスト | リスク | 優先度 |
|----------|------------|-----------|-----------|-------|--------|
| **LatencyService** | 高い (4箇所分散) | 中 (凝集度向上) | 高 (Audio Thread パス変更) | 中 | **Low** |
| **WarmupService** | 低 (2箇所) | 低 (小さな改善) | 低 | 低 | **Very Low** |

### 判断: Phase2 送りが妥当

- LatencyService は PDC バッファが Audio Thread 直アクセスであり、分離にはスレッド安全性の再検証が必要
- WarmupService は `RuntimeBuilder` の1メソッドであり、分離効果が薄い
- 現状のレイテンシ管理には Authority 違反は存在しない (全て Execution/Diagnostic 用途)
- 本計画の目的 (Authority 純化) に対する寄与は限定的

### 代替案 (より実践的)

現状のコードはレイテンシ計算を `AudioEngine` の inline メソッドとして実装している。これらを `.cpp` ファイルに切り出すだけでも保守性は向上するが、Authority との関連性は低い。
