# ISR RT Execution Frame Separation 仕様

## 位置づけ

本書は **ISR 10層 architecture における RT execution contamination 防止**の正本仕様である。

RT thread への authority leakage を封止するために、
RT callback 内の全状態を **RTExecutionFrame** としてスタックローカルに完全局所化する。

位置づけ: `plan5.md` → REV2 未閉塞系統 C → 本書（詳細）

### REV3.2運用優先注記

- 本書の RT firewall / frame 分解は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- RT 診断は Release 常時full verify を要求せず、Debug/CI を優先する。

---

## 問題の本質

### RT execution contamination

```text
現状のリスク:
  RT callback から MessageManager / Logger / mutex に到達するパスが残存する可能性
  RT callback から runtime publish（atomic store publish graph）が呼ばれる可能性
  RT callback 内で heap allocation が発生する可能性
  RT callback がスレッド間共有 state を直接参照する可能性

要求:
  RT callback 内の全状態をスタックローカルな RTExecutionFrame に封じ込める
  RTCapabilityFirewall を callback 入口に置き、authority leakage を事前検出する
  RT-1 / RT-2 / RT-3 invariant により enforcement する
```

---

## RTExecutionFrame

RT callback ごとに stack-local で生成・破棄される実行コンテキスト。
**heap allocation 禁止**。

```cpp
struct RTExecutionFrame
{
    // 現在の DSP 参照（DSPHandleRuntime 経由で解決済み）
    DSPHandle activeDSP;
    DSPHandle fadingDSP;   // crossfade 中のみ有効

    // crossfade 状態
    FadeAccumulator fade;

    // scratch メモリ（preallocated pool から取得済み pointer）
    ScratchArena* scratch;

    // 現在 callback の sample cursor
    uint64_t sampleCursor;

    // lifecycle epoch（LifecycleIsolationRuntime から取得）
    CallbackExecutionEpoch epoch;

    // RT trace relay（non-RT side への relay buffer）
    RTTraceRelay* traceRelay;  // nullptr は tracing 無効
};
```

### 制約

```text
RT-1: RTExecutionFrame は stack-local のみ（heap 確保禁止）
RT-2: RTExecutionFrame.activeDSP / fadingDSP の変更禁止（read-only view）
RT-3: RTExecutionFrame を RT callback 外で保持禁止
```

---

## FadeAccumulator

```cpp
struct FadeAccumulator
{
    double gainFrom;   // crossfade 元の現在ゲイン [0.0, 1.0]
    double gainTo;     // crossfade 先の現在ゲイン [0.0, 1.0]
    bool   active;     // crossfade 実行中フラグ
};
```

---

## RTCapabilityFirewall

RT callback の入口に配置し、callback 内での authority leakage を事前検出する。

```cpp
class RTCapabilityFirewall
{
public:
    // RT callback 入口: 現在の capability を check
    // 違反（MessageManager アクセス可能状態等）を検出したら Abort
    FirewallToken enter();

    // RT callback 出口: 副作用の漏洩を検出
    void leave(FirewallToken token);

    // audit: RT callback 内から publishAtomic が呼ばれていないか検査
    // （Debug/CI build のみ有効）
    void auditPublishAttempt(const char* callSite);
};
```

---

## RTAllocatorFirewall

RT callback 内での heap allocation を検出・Abort する。

```cpp
class RTAllocatorFirewall
{
public:
    // Debug/CI build: operator new / malloc override で呼ばれる
    static void onAllocAttempt(size_t size, const char* callSite);

    // RT callback 中であることを thread-local flag で示す
    static void markRTContext(bool entering);

    static bool isRTContext() noexcept;
};
```

---

## RTTraceRelay

RT callback 内で収集した軽量 trace を非RT 側に relay する lock-free relay buffer。

```cpp
struct RTTraceEvent
{
    uint64_t sampleCursor;
    uint32_t eventCode;
    uint32_t data;
};

class RTTraceRelay
{
public:
    // RT: trace event を enqueue（lock-free、固定サイズリングバッファ）
    void enqueue(RTTraceEvent event) noexcept;

    // NonRT: relay buffer を drain して HBTraceRuntime へ転送
    void drain();
};
```

---

## RTExecutionFrame 生成パターン

### audioCallback 入口

```cpp
void AudioEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& info)
{
    auto lifecycleToken = lifecycleRuntime_.enterAudioCallback();
    auto firewallToken  = rtFirewall_.enter();
    RTAllocatorFirewall::markRTContext(true);

    RTExecutionFrame frame
    {
        .activeDSP    = resolveActiveDSP(),   // DSPHandleRuntime 経由
        .fadingDSP    = resolveFadingDSP(),   // DSPHandleRuntime 経由
        .fade         = currentFade_,         // 前 callback からの継続値
        .scratch      = scratchPool_.get(),   // preallocated
        .sampleCursor = sampleCursor_,
        .epoch        = { lifecycleToken.epochId, sampleCursor_ },
        .traceRelay   = &traceRelay_
    };

    processAudio(info, frame);  // frame を引数で伝搬

    RTAllocatorFirewall::markRTContext(false);
    rtFirewall_.leave(firewallToken);
    lifecycleRuntime_.leaveAudioCallback(lifecycleToken);

    sampleCursor_ += static_cast<uint64_t>(info.numSamples);
}
```

---

## Invariants

| 識別子 | 内容 | 違反時アクション |
| --- | --- | --- |
| RT-1 | RTExecutionFrame は stack-local のみ（heap 確保禁止） | Abort |
| RT-2 | RTExecutionFrame 内の DSP handle は read-only | Abort |
| RT-3 | RTExecutionFrame を callback 外で保持禁止 | Abort |
| RT-4 | RT callback 内からの publishAtomic 呼び出し禁止 | Abort |
| RT-5 | RT callback 内からの MessageManager アクセス禁止 | Abort |
| RT-6 | RT callback 内からの mutex lock 禁止 | Abort |
| RT-7 | RT callback 内からの heap allocation 禁止 | Abort |
| GI-2 | RT thread owns no reclaim authority | Abort |

---

## ScratchArena

RT callback 内で使用する非ヒープ一時バッファ。`prepareToPlay` で preallocate 済み。

```cpp
class ScratchArena
{
public:
    // NonRT: prepareToPlay 時に確保
    void allocate(size_t byteSize);   // mkl_malloc / aligned_malloc(64)

    // NonRT: releaseResources 時に解放
    void free();

    // RT: scratch pointer 取得（allocation なし）
    void* data() noexcept;

    size_t size() const noexcept;
};
```

---

## 必須 artifacts

### rt_execution_trace.json（RTTraceRelay 経由で収集）

```json
{
  "schema": "rt_execution_trace_v1",
  "callbacks": [
    {
      "sampleCursor": 0,
      "lifecycleEpoch": 1,
      "activeDSPSlot": 1,
      "activeDSPGeneration": 2,
      "crossfadeActive": false
    }
  ],
  "violations": []
}
```

---

## Closed criteria

- [ ] RTExecutionFrame が全 RT callback で stack-local にのみ生成されている
- [ ] RTCapabilityFirewall が callback 入口に配置されている
- [ ] RTAllocatorFirewall が Debug/CI build で heap allocation を検出する
- [ ] RT-1 ～ RT-7 の violation が runtime abort または CI 検出で閉塞される
- [ ] RT callback からの publishAtomic が全て排除されている
- [ ] ScratchArena が prepareToPlay で preallocate されている
- [ ] RTTraceRelay が lock-free enqueue を提供している

---

## 関連文書

- `plan5.md`: REV2 未閉塞4系統 C 系統参照
- `ISR_JUCE_Lifecycle_Isolation.md`: CallbackExecutionEpoch 定義
- `ISR_DSPHandle_Runtime.md`: DSPHandle resolve
- `ISR_10Layer_Implementation_Specification.md`: 修正版実装順序（ステップ 1）
- `ISR_HB_Graph_Specification.md`: RTTraceRelay → HBTraceRuntime 連携
