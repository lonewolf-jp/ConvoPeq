# ISR JUCE Lifecycle Isolation Runtime 仕様

## 位置づけ

本書は **ISR 10層 architecture における Layer 0** の正本仕様である。

ConvoPeq は JUCE host driven architecture であり、
オーディオエンジンの全起動・停止・callback シーケンスは JUCE により制御される。
本仕様は **JUCE lifecycle の非決定性が ISR invariant を破壊しないよう隔離する**
ための runtime 定義を提供する。

位置づけ: `plan5.md` → `ISR_10Layer_Implementation_Specification.md` 層0 → 本書（詳細）

---

## 問題の本質

### JUCE lifecycle の非決定性

```text
JUCE が保証する callback 順序:
  prepareToPlay → audioCallback* → releaseResources

しかし ISR が要求する invariant:
  publish graph は seal 後 immutable
  DSP lifetime は明示的に管理される
  RT thread は reclaim authority を持たない

Gap:
  JUCE は "shutdown 後に audioCallback が来ない" を runtime enforce しない
  JUCE は "prepareToPlay overlap が起きない" を runtime enforce しない
  JUCE は "releaseResources 中の RT publish" を detect しない
```

### 解決方針

```text
LifecycleIsolationRuntime:
  JUCE callback の入口に gate を置き、
  ISR-safe な LifecyclePhase ステートマシンに変換する。
  phase violation を runtime abort で閉塞する。
```

---

## 定義

### LifecyclePhase

```cpp
enum class LifecyclePhase
{
    Uninitialized,  // 初期状態
    Preparing,      // prepareToPlay 実行中
    Prepared,       // audioCallback 受付可能
    AudioRunning,   // audioCallback 実行中（re-entrant 禁止）
    Releasing,      // releaseResources 実行中
    Released,       // リソース解放完了
    Shutdown        // 終端（再 prepare 禁止）
};
```

### 遷移規則

```text
Uninitialized → Preparing   (enterPrepare 呼び出し)
Preparing     → Prepared    (leavePrepare 呼び出し)
Prepared      → AudioRunning (enterAudioCallback)
AudioRunning  → Prepared    (leaveAudioCallback)
Prepared      → Releasing   (enterRelease)
Releasing     → Released    (leaveRelease)
Released      → Shutdown    (shutdown 明示呼び出し)

Uninitialized → Releasing   ← 禁止（LIF-2）
Preparing     → AudioRunning ← 禁止（LIF-4）
AudioRunning  → Releasing   ← 禁止（LIF-2）
Releasing     → AudioRunning ← 禁止（LIF-2）
Shutdown      → Preparing   ← 禁止（once-only lifecycle）
```

### LifecycleToken

```cpp
struct LifecycleToken
{
    uint64_t epochId;
    LifecyclePhase expectedPhase;
};
```

---

## LifecycleIsolationRuntime

```cpp
class LifecycleIsolationRuntime
{
public:
    // NonRT: prepareToPlay 入口
    LifecycleToken enterPrepare(int sampleRate, int blockSize);
    void leavePrepare(LifecycleToken token);

    // RT: audioCallback 入口
    LifecycleToken enterAudioCallback();
    void leaveAudioCallback(LifecycleToken token);

    // NonRT: releaseResources 入口
    LifecycleToken enterRelease();
    void leaveRelease(LifecycleToken token);

    // 終端
    void shutdown();

    // 現在 phase（RT safe: atomic read）
    LifecyclePhase current() const noexcept;

    // phase が AudioRunning であることを assert（RT callable）
    void assertAudioRunning() const noexcept;

    // artifact emit（shutdown time または CI trigger）
    void emitPhaseTrace(const std::filesystem::path& outputPath) const;

private:
    std::atomic<LifecyclePhase> phase_{ LifecyclePhase::Uninitialized };
    std::atomic<uint64_t> epochCounter_{ 0 };

    // NonRT guard（prepareToPlay / releaseResources overlap 防止）
    std::mutex nonRtGuard_;
};
```

---

## LifecycleBarrierRuntime

phase transition に HB edge を付与する補完 runtime。

```cpp
class LifecycleBarrierRuntime
{
public:
    // prepareToPlay 完了後、Prepared HB edge を emit
    void publishPreparedBarrier();

    // releaseResources 開始前、AudioStopped HB edge を emit
    void publishReleasingBarrier();

    // shutdown 完了後、Shutdown HB edge を emit
    void publishShutdownBarrier();
};
```

---

## CallbackExecutionEpoch

RT callback 内で使用される stack-local epoch 識別子。

```cpp
struct CallbackExecutionEpoch
{
    uint64_t lifecycleEpoch;  // LifecycleToken.epochId に対応
    uint64_t sampleCursor;    // 現在 callback の先頭 sample 位置
};
```

---

## Invariants

| 識別子 | 内容 | 違反時アクション |
| --- | --- | --- |
| LIF-1 | prepareToPlay は serialized（overlap 禁止） | Abort |
| LIF-2 | releaseResources は AudioRunning 中に呼べない | Abort |
| LIF-3 | Releasing phase 中の runtime publish 禁止 | Abort |
| LIF-4 | crossfade start は Prepared phase 以降のみ | Abort |
| LIF-5 | callback 中に runtimeVersion が変化しない | Abort |
| LIF-6 | callback 中に DSP generation が変化しない | Abort |
| LR-1 | LifecycleToken epoch は単調増加 | Abort |
| LR-2 | leaveXxx で token phase mismatch 禁止 | Abort |
| LR-3 | Shutdown phase 後の re-prepare 禁止 | Abort |

---

## 必須 artifacts

### lifecycle_phase_trace.json

```json
{
  "schema": "lifecycle_phase_trace_v1",
  "transitions": [
    {
      "from": "Uninitialized",
      "to": "Preparing",
      "epochId": 1,
      "timestamp_ns": 0
    },
    {
      "from": "Preparing",
      "to": "Prepared",
      "epochId": 1,
      "timestamp_ns": 12345
    }
  ],
  "invariant_violations": []
}
```

---

## ConvoPeq 統合パターン

### prepareToPlay

```cpp
void AudioEngine::prepareToPlay(int sampleRate, int samplesPerBlock)
{
    auto token = lifecycleRuntime_.enterPrepare(sampleRate, samplesPerBlock);
    // ... 既存初期化処理 ...
    lifecycleRuntime_.leavePrepare(token);
}
```

### getNextAudioBlock（audioCallback）

```cpp
void AudioEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& info)
{
    auto token = lifecycleRuntime_.enterAudioCallback();
    // ... 既存 RT 処理 ...
    lifecycleRuntime_.leaveAudioCallback(token);
}
```

### releaseResources

```cpp
void AudioEngine::releaseResources()
{
    auto token = lifecycleRuntime_.enterRelease();
    // ... 既存リソース解放処理 ...
    lifecycleRuntime_.leaveRelease(token);
}
```

---

## Closed criteria

- [ ] LifecycleIsolationRuntime が prepareToPlay / releaseResources / audioCallback の入口に配置済み
- [ ] LIF-1 ～ LIF-6 の違反が runtime で abort される
- [ ] LR-1 ～ LR-3 の違反が runtime で abort される
- [ ] lifecycle_phase_trace.json が emit される
- [ ] JUCE callback から直接 publish graph を変更するパスが存在しない
- [ ] CallbackExecutionEpoch が RT callback 内で stack-local に保持される
- [ ] LifecycleBarrierRuntime が phase transition に HB edge を付与している

---

## 関連文書

- `plan5.md`: REV2 未閉塞4系統 A 系統参照
- `ISR_10Layer_Implementation_Specification.md`: Layer 0 概要
- `ISR_HB_Graph_Specification.md`: HB edge 詳細
- `ISR_RT_Execution_Frame.md`: RT thread 局所状態管理
