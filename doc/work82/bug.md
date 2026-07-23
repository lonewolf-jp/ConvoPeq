承知しました。以下は **本スレッドで提供された `ConvoPeq.md` の記載のみ**を対象にした再検証結果です。  
過去スレッドの指摘・過去データは考慮していません。

また、`Practical Stable ISR Bridge Runtime.md` は本スレッド提供内容が空のため、追加制約としては使用していません。  
代わりに `coding_rule_jp.txt` の制約、特に Audio thread 内禁止事項を基準にしています。

---

# ConvoPeq.md 限定検証レポート

## 0. 前提

- `ConvoPeq.md` はソース全体ではなく、断片集合です。
- 関数全体が見えない箇所は「確定」ではなく「要確認」として扱います。
- 本ファイル内には同一断片の重複が含まれています。重複箇所は同一内容として扱いました。
- 一部、Markdown 結合時に壊れたと思われる構文があります。これらは「実ソースにそのまま存在するなら問題」「抽出artifactなら無害」を区別して記載します。

---

# 1. 優先度サマリ

| ID | 重大度 | 確度 | 概要 |
|---|---:|---|---|
| CVMD-001 | High | 条件付き確定 | CMake `target_link_options` で複数フラグが 1 generator expression 内にまとまっている |
| CVMD-002 | High | 確定 | MMCSS 設定/解除が Audio thread から呼ばれる設計に見える。coding_rule 違反の可能性 |
| CVMD-003 | High | 条件付き | `RetireRuntime` fallback で `std::mutex` を使用。RT 呼び出しなら規約違反 |
| CVMD-004 | High | 要確認 | `CustomInputOversampler` AVX2 path の history 負方向アクセス境界が断片では確認できない |
| CVMD-005 | Medium | 確定 | bypass 状態の modeId 判定で「EQ/Conv 両方 bypass」が考慮されていない |
| CVMD-006 | Medium | 可能性高 | `MessageManager::callAsync` 失敗時に raw `commitPtr` が leak する可能性 |
| CVMD-007 | Medium | 確定 | `cachedTailLength` が非 atomic。`getTailLengthSeconds()` からの読み取りが thread-safe でない可能性 |
| CVMD-008 | Medium | 確定 | `FadeAccumulator` / `RTExecutionFrame` に default member initializer がない |
| CVMD-009 | Medium | 条件付き | icx Release で `/MT` を指定しているが、`MSVC_RUNTIME_LIBRARY` property との整合が不明 |
| CVMD-010 | Low | 確定 | icx で `CONVOPEQ_PGO_USE` が要求されても警告が出ない |
| CVMD-011 | Low | 確定 | 不明プラットフォームで AVX2 check が `return true` になる |
| CVMD-012 | Low / Info | 確定 | 重複・破損した構文断片がある。実ソース確認が必要 |
| CVMD-013 | Medium | 条件付き | Audio callback 内診断で `std::round` を使用している可能性。Release で有効なら libm 呼び出し |
| CVMD-014 | Low | 確定 | Audio callback 冒頭で telemetry 有効/無効に関わらず `getCurrentTimeUs()` を呼んでいる |
| CVMD-015 | Low / Medium | 確定 | icx Release flags に `/fp:fast` がある。数値再現性リスク |
| CVMD-016 | Low | 要確認 | `RTTraceRelay` が SPSC 前提か、lifetime 保証があるか断片では確認できない |
| CVMD-017 | Low | 確定 | `vdTanh` 用に stack 上 `alignas(64) double tanhBuffer[totalCoeffs]` を確保している |
| CVMD-018 | Low | 確定 | `const_cast` で `const DSPCore*` を非 const 化している。意図的だが const 安全性リスク |
| CVMD-019 | Info | 確定 | 診断コードの空 `if` 文が誤解を招く |
| CVMD-020 | Info | 確定 | `ABSOLUTE_MAX_BLOCK_SIZE` が定義されているが、実際の判定に使われていない可能性 |

---

# 2. 詳細

---

## CVMD-001: CMake `target_link_options` で複数フラグが 1 generator expression 内にある

### 重大度
High

### 確度
条件付き確定  
※ `ConvoPeq.md` の記載が実ファイルそのままなら問題。

### 該当箇所

```cmake
target_link_options(ConvoPeq PRIVATE
$<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<CONFIG:Release>>:/DEBUG /Qipo>
)
```

```cmake
target_link_options(ConvoPeq PRIVATE
$<$<CXX_COMPILER_ID:MSVC>:/LTCG /GENPROFILE:PGD=${CONVOPEQ_PGO_PGD}>
)
```

```cmake
target_link_options(ConvoPeq PRIVATE
$<$<CXX_COMPILER_ID:MSVC>:/LTCG /USEPROFILE:PGD=${CONVOPEQ_PGO_PGD}>
)
```

### 問題点

CMake の generator expression 内で、空白区切りの複数フラグを 1 項目として渡しています。  
この記述がそのまま実ファイルにある場合、CMake の引数解析または generator expression 展開の段階で意図しない分割・不正なフラグになる可能性があります。

### 影響

- `/DEBUG` や `/Qipo` が正しく渡らない
- PGO instrument/use が silently に無効化される
- リンクオプション欠落
- Release ビルドの最適化/PDB 出力が意図と異なる

### 修正提案

フラグごとに generator expression を分割してください。

```cmake
target_link_options(ConvoPeq PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<CONFIG:Release>>:/DEBUG>
    $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<CONFIG:Release>>:/Qipo>
)
```

```cmake
target_link_options(ConvoPeq PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/LTCG>
    $<$<CXX_COMPILER_ID:MSVC>:/GENPROFILE:PGD=${CONVOPEQ_PGO_PGD}>
)
```

```cmake
target_link_options(ConvoPeq PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/LTCG>
    $<$<CXX_COMPILER_ID:MSVC>:/USEPROFILE:PGD=${CONVOPEQ_PGO_PGD}>
)
```

---

## CVMD-002: MMCSS 設定/解除が Audio thread から呼ばれる設計に見える

### 重大度
High

### 確度
確定  
※ coding_rule 違反の可能性が高い。

### 該当箇所

```cpp
// RT impact: first call only (~50-200μs for LPC call to MMCSS service). Subsequent calls: O(1) TLS read.
[[nodiscard]] bool AudioEngine::tryApplyMmcssForSelfManagedThread() noexcept
{
    if (t_mmcssTried)
        return (t_mmcssHandle != nullptr);

    t_mmcssTried = true;

    // ★ CPU affinity + NativeRT: always set on first callback, regardless of policy.
    // applyMmcssPriority() handles SetThreadAffinityMask (all backends)
    // and SetPriorityClass/SetThreadPriority (NativeRT mode only).
    applyMmcssPriority();
```

さらに、Audio callback 末尾付近に以下のコメントがあります。

```cpp
// ★ [work70 v9.11] MMCSS shutdown: handled above via mmcssShutdownRequested flag.
// If policy is SelfManaged*, revertMmcssOnAudioThread() was called above.
```

### 問題点

`coding_rule_jp.txt` には、Audio thread 内で禁止する処理として以下が明記されています。

> MMCSS設定

また、 genel な RT 禁止事項として「待つ可能性が 1% でもある処理」も禁止されています。

コメントを読む限り、`tryApplyMmcssForSelfManagedThread()` は初回 Audio callback 内で呼ばれ、MMCSS service への LPC call が発生し得ます。  
また、`revertMmcssOnAudioThread()` も Audio thread 内で呼ばれている可能性があります。

### 影響

- 初回 callback で 50–200μs の遅延
- MMCSS API 呼び出しによる blocking 可能性
- coding_rule 違反
- ASIO/WASAPI での xrun 原因

### 修正提案

MMCSS 設定/解除を Audio thread 外へ移動してください。

理想：

- `prepareToPlay()`
- device start 前
- 専用 thread 初期化時
- host/device callback 開始前の非 RT 初期化

ただし、ASIO driver thread 制約などで「同一 thread でなければならない」場合、coding_rule と設計が衝突します。  
その場合は、規約側で例外を明文化するか、別設計にする必要があります。

---

## CVMD-003: `RetireRuntime` fallback で `std::mutex` を使用している

### 重大度
High  
※ RT 呼び出しの場合。

### 確度
条件付き

### 該当箇所

```cpp
convo::publishAtomic(slots_[idx].sequence, ticket + 1, std::memory_order_release);

std::lock_guard<std::mutex> lock(fallbackMutex_);
if (fallbackCount_ < FALLBACK_QUEUE_CAPACITY) {
    const size_t tail = (fallbackHead_ + fallbackCount_) % FALLBACK_QUEUE_CAPACITY;
    fallbackQueue_[tail] = localIntent;
    ++fallbackCount_;
    convo::publishAtomic(fallbackQueuePeak_, fallbackCount_.load(std::memory_order_relaxed), std::memory_order_release);
    (void)convo::fetchAddAtomic(overflowCount_,
```

### 問題点

`std::lock_guard<std::mutex>` は Audio thread 内で禁止されています。  
このコードが `RetireRuntime::emitRetireIntent()` またはそれに準じる RT producer 経路にある場合、規約違反です。

### 影響

- priority inversion
- Audio dropout
- mutex 取得待ちによる xrun

### 確認すべき点

実コードで以下を確認してください。

```bash
grep -RIn "fallbackMutex_" src
grep -RIn "emitRetireIntent" src
```

呼び出し元が Audio callback / RT processing なら High です。  
Non-RT thread 専用なら問題は小さくなります。

### 修正提案

RT producer から呼ぶなら、fallback も lock-free にすべきです。

- RT 側は spin せず、slot 取得失敗時は overflow ring へ tryPush
- mutex を使う fallback 移動は Non-RT timer/thread が実施

---

## CVMD-004: `CustomInputOversampler` AVX2 path の history 負方向アクセス境界が要確認

### 重大度
High  
※ invariant が不足している場合 Critical。

### 確度
要確認

### 該当箇所

```cpp
acc += dotProductDecimateAvx2(
    history + (base - stage.convParity),
    coeffs,
    stage.convCount);
```

```cpp
__m256d vS = loadStride2(
    history + (base - stage.convParity) - (r << 1));
```

### 問題点

`history + (base - stage.convParity) - (r << 1)` は history buffer を負方向に参照します。  
`loadStride2()` が内部でさらに負オフセットを読む場合、`base` が十分大きくないと buffer 先頭前を読み込みます。

提供断片だけでは、以下が確認できません。

- `base` の最小値
- `stage.convParity` の取り得る値
- `stage.convCount` と `r` の最大値
- `loadStride2()` が読む最小オフセット
- history buffer に padding があるか
- 境界チェックが関数冒頭にあるか

### 影響

- buffer 先頭前読み込み
- access violation
- heap corruption
- 特定 FFT size / oversampling stage でのみ発生するクラッシュ

### 確認すべき点

```bash
grep -RIn "loadStride2" src
grep -RIn "dotProductDecimateAvx2" src
grep -RIn "historyDownKeep" src
grep -RIn "convParity" src
```

### 修正提案

最小アクセス index を計算し、以下を保証してください。

```cpp
jassert(minAccessIndex >= 0);
```

または、history buffer 先頭前に十分な padding を確保し、その invariant をコメントと debug assert で明示してください。

---

## CVMD-005: bypass 状態の modeId 判定で「両方 bypass」が考慮されていない

### 重大度
Medium

### 確度
確定

### 該当箇所

```cpp
// Active は Audio Thread 反映タイミング依存のため、直後に旧値へ戻ることがある。
const bool eqBypassed = audioEngine.isEqBypassRequested();
const bool convBypassed = audioEngine.isConvolverBypassRequested();

int modeId = 3; // Conv->Peq

if (!eqBypassed && convBypassed)
    modeId = 2; // Peq
else if (eqBypassed && !convBypassed)
    modeId = 1; // Conv
else if (!eqBypassed && !convBypassed
         && audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver)
    modeId = 4; // Peq->Conv
```

### 問題点

`eqBypassed && convBypassed` の場合、どの分岐にも入らず、`modeId` は初期値の `3 // Conv->Peq` のままになります。

### 影響

- 完全 bypass 中なのに UI が `Conv->Peq` を表示する可能性
- gain staging 表示や処理モード説明が実態と一致しない
- ユーザーが信号経路を誤認する

### 修正提案

両方 bypass のケースを明示してください。

```cpp
int modeId = 0; // Bypass

if (eqBypassed && convBypassed)
{
    modeId = 0; // Bypass
}
else if (!eqBypassed && convBypassed)
{
    modeId = 2; // Peq
}
else if (eqBypassed && !convBypassed)
{
    modeId = 1; // Conv
}
else if (!eqBypassed && !convBypassed
         && audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver)
{
    modeId = 4; // Peq->Conv
}
else
{
    modeId = 3; // Conv->Peq
}
```

---

## CVMD-006: `MessageManager::callAsync` 失敗時に raw `commitPtr` が leak する可能性

### 重大度
Medium

### 確度
可能性高

### 該当箇所

```cpp
auto weakThis = juce::WeakReference<ConvolverProcessor>(this);
const bool queued = juce::MessageManager::callAsync(
    [weakThis, commitPtr]()
    {
        auto ownedCommit = std::unique_ptr<PendingCommit>(commitPtr);
        if (auto* self = weakThis.get())
        {
            try
            {
                self->executePendingCommit(std::move(ownedCommit));
            }
            catch (...)
```

### 問題点

`commitPtr` が raw pointer で lambda に capture されています。  
lambda が実行された場合のみ `unique_ptr` が所有権を取ります。

もし `MessageManager::callAsync()` が失敗し、lambda が一度も実行されなかった場合、`commitPtr` は解放されない可能性があります。

提供断片には、`if (!queued)` 時の `delete commitPtr;` が見当たりません。

### 影響

- `PendingCommit` メモリリーク
- shutdown 時や message queue 停止時に発生しやすい
- IR commit 情報の消失

### 修正提案

```cpp
const bool queued = juce::MessageManager::callAsync(...);

if (!queued)
{
    delete commitPtr;
}
```

または、所有権を `std::shared_ptr<PendingCommit>` にしてください。

---

## CVMD-007: `cachedTailLength` が非 atomic

### 重大度
Medium

### 確度
確定

### 該当箇所

```cpp
// D5: IR長のみの概算値。oversampling やフィルターによるテール延長は未反映。
// C5: cachedTailLength を返す（Runtime Publish 時に更新。ValueTree 依存を断つ）
// 現在の ConvoPeq 実装の Runtime Publish シーケンスでは同一スレッドで実行されるため double（非 atomic）で十分
double AudioEngineProcessor::getTailLengthSeconds() const
{
    return cachedTailLength;
}
```

### 問題点

コメントでは「現在の Runtime Publish シーケンスでは同一スレッド」とされています。  
しかし、JUCE host が `getTailLengthSeconds()` をどの thread から呼ぶかは完全には制御できません。

- message thread
- audio thread
- host query thread
- plugin wrapper 内部 thread

から呼ばれる可能性があります。

`cachedTailLength` が Runtime Publish 時に書き込まれ、別 thread から読まれるなら、C++ 的には data race です。

### 影響

- コンパイラ最適化で古い値が読まれる可能性
- host 側の tail 計算が不安定になる可能性
- 稀に不正な tail length を返す可能性

### 修正提案

```cpp
std::atomic<double> cachedTailLength { 0.0 };
```

書き込み：

```cpp
cachedTailLength.store(newTailLength, std::memory_order_release);
```

読み込み：

```cpp
return cachedTailLength.load(std::memory_order_acquire);
```

---

## CVMD-008: `FadeAccumulator` / `RTExecutionFrame` に default member initializer がない

### 重大度
Medium

### 確度
確定

### 該当箇所

```cpp
struct FadeAccumulator
{
    double gainFrom; // crossfade 元の現在ゲイン [0.0, 1.0]
    double gainTo;   // crossfade 先の現在ゲイン [0.0, 1.0]
    bool active;     // crossfade 実行中フラグ
};
```

```cpp
uint64_t sampleCursor;
// callback 単位の一貫 view 識別子
uint64_t callbackEpoch;
// lifecycle epoch（LifecycleIsolationRuntime から取得）
uint64_t lifecycleEpoch;
// RT trace relay buffer へのポインタ（nullptr は tracing 無効）
class RTTraceRelay* traceRelay;
};
```

### 問題点

これらの struct に default member initializer がありません。  
もし以下のように生成された場合、未初期化メンバが残ります。

```cpp
RTExecutionFrame frame;
```

### 影響

- crossfade 状態の誤認識
- 未初期化 `traceRelay` による nullptr 判定失敗
- 未初期化 epoch による診断誤り
- デバッグ困難な不安定性

### 修正提案

```cpp
struct FadeAccumulator
{
    double gainFrom = 0.0;
    double gainTo = 0.0;
    bool active = false;
};
```

```cpp
struct RTExecutionFrame
{
    uint64_t sampleCursor = 0;
    uint64_t callbackEpoch = 0;
    uint64_t lifecycleEpoch = 0;
    class RTTraceRelay* traceRelay = nullptr;
};
```

または、生成箇所で必ず value-initialize してください。

```cpp
RTExecutionFrame frame{};
```

---

## CVMD-009: icx Release の `/MT` 指定と CRT property の整合が不明

### 重大度
Medium

### 確度
条件付き

### 該当箇所

```cmake
set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(CMAKE_CXX_COMPILER_ID
```

```cmake
# icx Release: CMake規定の-MDを/MTで上書き（静的CRT+静的MKLリンク）
target_compile_options(ConvoPeq PRIVATE
    $<$<CONFIG:Release>:/MT>
)
```

### 問題点

MSVC 分岐では `MSVC_RUNTIME_LIBRARY` property を設定していますが、icx 分岐で同等の runtime library 設定が行われているか、提供断片では確認できません。

コメントは「CMake 規定の -MD を /MT で上書き」となっていますが、以下の確認が必要です。

- JUCE module が `/MD` で compile されていないか
- Debug/Release で CRT が混在しないか
- MKL/IPP のリンク CRT と一致しているか
- link command line に `/DEFAULTLIB:LIBCMT` と `/DEFAULTLIB:MSVCRT` が混在しないか

### 影響

- heap corruption
- invalid free
- CRT を跨ぐ new/delete 不一致
- Release だけクラッシュ

### 確認提案

verbose build で link option を確認してください。

```bash
cmake --build build --verbose
```

---

## CVMD-010: icx で `CONVOPEQ_PGO_USE` が要求されても警告が出ない

### 重大度
Low

### 確度
確定

### 該当箇所

```cmake
# B3: icx で PGO が要求された場合は警告
if(CONVOPEQ_PGO_INSTRUMENT AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(WARNING "[PGO] PGO is currently supported only for MSVC. "
        "icx users: /Qprof-gen and /Qprof-use are available but not yet integrated.")
endif()
```

### 問題点

警告は `CONVOPEQ_PGO_INSTRUMENT` のみで出ています。  
`CONVOPEQ_PGO_USE` が icx で指定されても警告されず、通常ビルドになる可能性があります。

### 影響

- icx で PGO 最適化ビルドを期待しても silently に通常ビルドになる
- CI が非 PGO バイナリを生成し続ける

### 修正提案

```cmake
if((CONVOPEQ_PGO_INSTRUMENT OR CONVOPEQ_PGO_USE)
   AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(WARNING "[PGO] PGO is currently supported only for MSVC. "
        "icx users: /Qprof-gen and /Qprof-use are available but not yet integrated.")
endif()
```

---

## CVMD-011: 不明プラットフォームで AVX2 check が `return true` になる

### 重大度
Low

### 確度
確定

### 該当箇所

```cpp
#else
// 未知のプラットフォーム: 安全側に倒してチェック通過
return true;
#endif
```

### 問題点

コメントでは「安全側」とされていますが、AVX2 必須アプリにおいては、未知プラットフォームで true にするのは安全とは限りません。

### 影響

- 非対応 CPU で実行を続けてクラッシュする可能性
- エラーメッセージなしで異常終了する可能性

### 修正提案

Windows x64 専用であるなら、unknown を error 扱いにするか、最低限サポート compiler のみを許可してください。

```cpp
#else
return false;
#endif
```

---

## CVMD-012: Markdown 結合由来と思われる重複・破損構文

### 重大度
Low / Info  
※ 実ソースに存在するなら build error または logic bug。

### 確度
確定  
※ ただし `ConvoPeq.md` 抽出 artifact の可能性が高い。

### 該当箇所 1: 重複 parameter

```cpp
bool ConvolverProcessor::LoaderThread::queueFinalizeOnMessageThread(LoadResult& result,
    convo::ScopedAlignedPtr<double> irL,convo::ScopedAlignedPtr<double> irL,
    convo::ScopedAlignedPtr<double> irR,
```

`irL` が重複しています。実ソースにそのままあれば compile error です。

---

### 該当箇所 2: 壊れた delete 宣言

```cpp
ConvolverState& operator=(const ConvolverState&) =ConvolverState&) = delete;
ConvolverState& operator=(const ConvolverState&) = delete;
```

実ソースにそのままあれば syntax error です。

---

### 該当箇所 3: 重複 if

```cpp
if (outL != nullptr)if (outL != nullptr)
    outL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
```

動作上はほぼ無害ですが、重複しています。

---

### 該当箇所 4: 重複 for loop

```cpp
for (int g = 0; g < 3; ++g)
{for (int g = 0; g < 3; ++g)
{
```

実ソースにこのままあると、inner loop が outer の `g` を shadow し、意図しない 9 回評価になる可能性があります。

---

### 判断

これらは `ConvoPeq.md` 生成時の結合ミスである可能性が高いです。  
ただし、実ソースに存在すると重大なので、実ファイル確認を推奨します。

```bash
grep -RIn "irL,convo::ScopedAlignedPtr<double> irL" src
grep -RIn "=ConvolverState&) = delete" src
grep -RIn "if (outL != nullptr)if (outL != nullptr)" src
grep -RIn "for (int g = 0; g < 3; ++g)" src
```

---

## CVMD-013: Audio callback 内診断で `std::round` を使用している可能性

### 重大度
Medium  
※ Release で診断有効かつ Audio thread 内で compile される場合。

### 確度
条件付き

### 該当箇所

```cpp
entry.budgetPermille = static_cast<uint16_t>(
    std::min(999.0, std::round(pct * 10.0)));
```

### 問題点

`std::round` は libm 呼び出しになる可能性があります。  
`coding_rule_jp.txt` では Audio thread 内で libm 呼び出しとなる関数を禁止しています。

このコードが `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内であり、Release で無効化されるなら問題ありません。  
しかし、診断有効の Release ビルドを作る可能性があるなら注意が必要です。

### 修正提案

整数演算で代替するか、診断を Release から完全に排除してください。

例：

```cpp
const uint64_t permille = static_cast<uint64_t>(pct * 10.0 + 0.5);
entry.budgetPermille = static_cast<uint16_t>(std::min<uint64_t>(999, permille));
```

ただし、浮動小数点丸め自体を避けたい場合は、固定小数点化が望ましいです。

---

## CVMD-014: Audio callback 冒頭で telemetry 有効/無効に関わらず `getCurrentTimeUs()` を呼んでいる

### 重大度
Low

### 確度
確定

### 該当箇所

```cpp
// ★ [work66-P2-4] 共通開始時刻（関数先頭で1回のみ取得。XRUN/t0_start と共有）
const auto cbStartUs = convo::getCurrentTimeUs();

struct CallbackTelemetryScope final
{
    AudioEngine& engine;
    int samples;
    bool enabled;
    uint64_t startUs;
```

### 問題点

telemetry が無効でも、callback 冒頭で時刻取得が行われています。  
`getCurrentTimeUs()` が `QueryPerformanceCounter` 系の場合、通常は高速ですが、RT path では不要な system call cost になる可能性があります。

### 影響

- 微小な RT overhead
- 低 latency 設定での margin 減少

### 修正提案

telemetry または診断が有効なときだけ時刻取得する。

```cpp
const bool telemetryEnabled = owner.isCliProcessingTelemetryEnabled();
const uint64_t cbStartUs = telemetryEnabled ? convo::getCurrentTimeUs() : 0;
```

---

## CVMD-015: icx Release flags の `/fp:fast`

### 重大度
Low / Medium

### 確度
確定

### 該当箇所

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8")
set(CMAKE_C_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8")
```

### 問題点

`/fp:fast` は IEEE 厳密性を緩和します。  
Audio DSP では、フィルター安定性、微小信号、denormal、丸め誤差に影響する可能性があります。

### 影響

- 数値再現性の低下
- compiler/CPU 依存の微妙な音質差
- 微小信号の消失
- 安定性 margin の変動

### 修正提案

- DSP core だけ `/fp:precise` を検討
- または denormal / FTZ / DAZ 方針を明文化
- 重要係数計算は fast FP の影響を外す

---

## CVMD-016: `RTTraceRelay` の SPSC 前提と lifetime が要確認

### 重大度
Low

### 確度
要確認

### 該当箇所

```cpp
RTTraceRelay::RTTraceRelay()
{
    convo::publishAtomic(buffer_, new RTTraceEvent[RELAY_BUFFER_SIZE], std::memory_order_release);
}

RTTraceRelay::~RTTraceRelay()
{
    auto* buf = convo::consumeAtomic(buffer_, std::memory_order_acquire);
    if (buf) {
        delete[] buf;
    }
}

void RTTraceRelay::enqueue(const RTTraceEvent& event) noexcept
{
    auto* buf = convo::consumeAtomic(buffer_, std::memory_order_acquire);
    if (!buf) return;

    uint64_t writeIdx = convo::consumeAtomic(writeIndex_, std::memory_order_relaxed);
    uint64_t nextIdx = (writeIdx + 1) %
```

### 問題点

断片だけでは以下が確認できません。

- producer が単一 RT thread か
- consumer が単一 Non-RT thread か
- `writeIndex_` / `readIndex_` の publish/consume order が完全か
- destructor 実行時に RT enqueue が停止している保証があるか

### 影響

- 複数 producer がある場合、index race
- destructor と enqueue の競合で use-after-free

### 確認提案

```bash
grep -RIn "RTTraceRelay" src
grep -RIn "enqueue" src
grep -RIn "drain" src
```

---

## CVMD-017: `vdTanh` 用に stack 上配列を確保している

### 重大度
Low

### 確度
確定

### 該当箇所

```cpp
alignas(64) double tanhBuffer[totalCoeffs] = {};
```

### 問題点

`totalCoeffs` が大きい場合、stack overflow の可能性があります。  
`totalCoeffs` が compile-time constant であり、十分小さいなら問題ありません。

### 確認提案

```bash
grep -RIn "totalCoeffs" src
grep -RIn "kPopulation" src
grep -RIn "kDim" src
```

### 修正提案

大きい場合は heap または aligned buffer を使用してください。

```cpp
auto tanhBuffer = convo::makeAlignedArray<double>(totalCoeffs);
```

---

## CVMD-018: `const_cast` で `const DSPCore*` を非 const 化している

### 重大度
Low

### 確度
確定

### 該当箇所

```cpp
auto engineState = engine.makeEngineRuntimeState(
    const_cast<AudioEngine::DSPCore*>(current),
    const_cast<AudioEngine::DSPCore*>(next),
    policy, fadeTimeSec, active,
    spec.currentRuntimeWorld);
```

### 問題点

意図的だと思われますが、`const` 性を外しています。  
もし `makeEngineRuntimeState()` 内で対象を実際に書き換えるなら、const 安全性に問題があります。

### 影響

- immutable であるべき object の変更
- 将来の保守で UB を埋め込む可能性

### 修正提案

- `makeEngineRuntimeState()` が本当に non-const 参照を必要とするか再確認
- 必要ないなら `const DSPCore*` を受け取る overload を作る
- 必要なら、const でない object を渡す設計にする

---

## CVMD-019: 診断コードの空 `if` 文が誤解を招く

### 重大度
Info

### 確度
確定

### 該当箇所

```cpp
if ((cbSeq & CONVOPEQ_DIAG_SAMPLE_MASK) != 0)
    ; // skip print
else
{
    ...
}
```

### 問題点

空の `if` 文は、将来の修正で壊れやすいです。  
意図は「mask に引っかかったら skip」ですが、可読性が低いです。

### 修正提案

条件を反転してください。

```cpp
if ((cbSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
{
    ...
}
```

---

## CVMD-020: `ABSOLUTE_MAX_BLOCK_SIZE` が定義されているが、実際の判定に使われていない可能性

### 重大度
Info

### 確度
確定

### 該当箇所

```cpp
// 事前サニティチェック: 絶対的な上限 (1<<20 ≒ 100万サンプル) で明らかな破損データを弾く。
// DSPCore::prepare() でホスト指定のブロックサイズが maxSamplesPerBlock に反映されるため、
// ここでは固定の SAFE_MAX_BLOCK_SIZE (65536) を使わず、取得済み DSPCore の値で最終判定する。
constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20; // 破損データ検出用上限

if (numSamples <= 0||
```

その後：

```cpp
if (numSamples > dsp->maxSamplesPerBlock)
{
    bufferToFill.clearActiveBufferRegion();
    return;
}
```

### 問題点

コメントでは `ABSOLUTE_MAX_BLOCK_SIZE` で弾くように見えますが、実際の判定は `dsp->maxSamplesPerBlock` です。  
`ABSOLUTE_MAX_BLOCK_SIZE` が dead code になっています。

### 影響

- 保守性の低下
- コメントと実装の不一致

### 修正提案

どちらかに統一してください。

案 1：絶対上限も使う

```cpp
if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
{
    bufferToFill.clearActiveBufferRegion();
    return;
}

if (numSamples > dsp->maxSamplesPerBlock)
{
    bufferToFill.clearActiveBufferRegion();
    return;
}
```

案 2：dead constant を削除し、コメントを修正

---

# 3. 提供断片から確認できた良好な点

以下は、`coding_rule_jp.txt` の観点で好ましい設計です。

## 3.1 libm 回避 helper

```cpp
inline bool isFinite(double x) noexcept
{
    const auto bits = std::bit_cast<uint64_t>(x);
    return ((bits >> 52) & 0x7FFu) != 0x7FFu;
}

inline double absNoLibm(double x) noexcept
{
    auto bits = std::bit_cast<uint64_t>(x);
    bits &= 0x7FFFFFFFFFFFFFFFULL;
    return std::bit_cast<double>(bits);
}
```

RT 内で libm を避ける意図が明確で良いです。

---

## 3.2 lock-free diag buffer の drop counter

```cpp
if (diagBuffer.push(event))
{
    rtAuxMutable_.diagTickPushed.value.fetch_add(1, std::memory_order_relaxed);
    rtAuxMutable_.diagTotalPushed.fetch_add(1, std::memory_order_relaxed);
}
else
{
    rtAuxMutable_.diagTickDropped.value.fetch_add(1,
```

RT 内で無理にログを書かず、drop を count する設計は適切です。

---

## 3.3 `MKLNonUniformConvolver::getDiagnostics()` の message thread assert

```cpp
jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
```

診断取得を Non-RT に限定する意図が明確です。

---

## 3.4 `StereoConvolver` 初期化の Strong Exception Guarantee

```cpp
// Phase 1: すべてローカル変数で初期化を実行（メンバー未更新）
// Phase 2: 全成功後にのみメンバーを一括更新（commit）
```

MKL/aligned allocation を伴う初期化として適切な設計です。

---

# 4. 実ソースで優先確認すべき grep

以下を実行すると、本レポートの要確認項目を潰せます。

```bash
# CMake link options
grep -RIn "target_link_options" CMakeLists.txt

# MMCSS
grep -RIn "tryApplyMmcssForSelfManagedThread" src
grep -RIn "revertMmcssOnAudioThread" src
grep -RIn "applyMmcssPriority" src

# Retire mutex
grep -RIn "fallbackMutex_" src
grep -RIn "emitRetireIntent" src

# Oversampler boundary
grep -RIn "loadStride2" src
grep -RIn "dotProductDecimateAvx2" src
grep -RIn "convParity" src
grep -RIn "historyDownKeep" src

# callAsync leak
grep -RIn "MessageManager::callAsync" src
grep -RIn "if (!queued)" src

# cachedTailLength
grep -RIn "cachedTailLength" src

# RTExecutionFrame initialization
grep -RIn "RTExecutionFrame" src
grep -RIn "FadeAccumulator" src

# icx CRT
grep -RIn "MSVC_RUNTIME_LIBRARY" CMakeLists.txt
grep -RIn "/MT" CMakeLists.txt

# PGO
grep -RIn "CONVOPEQ_PGO_USE" CMakeLists.txt
grep -RIn "CONVOPEQ_PGO_INSTRUMENT" CMakeLists.txt

# Markdown artifact
grep -RIn "irL,convo::ScopedAlignedPtr<double> irL" src
grep -RIn "=ConvolverState&) = delete" src
grep -RIn "if (outL != nullptr)if (outL != nullptr)" src
```

---

# 5. 結論

本スレッド提供の `ConvoPeq.md` を対象にした場合、最優先で確認・修正すべきは以下です。

1. **CVMD-001**  
   CMake `target_link_options` の複数フラグまとめ書き。

2. **CVMD-002**  
   MMCSS 設定/解除が Audio thread から呼ばれている可能性。  
   `coding_rule_jp.txt` と明確に衝突する。

3. **CVMD-003**  
   `RetireRuntime` fallback の `std::mutex`。  
   RT 呼び出し元があるなら規約違反。

4. **CVMD-004**  
   `CustomInputOversampler` AVX2 path の history 負方向アクセス境界。  
   境界 invariant が確認できないため、Crash 候補として扱うべき。

5. **CVMD-005**  
   両方 bypass 時の modeId 誤り。

6. **CVMD-006**  
   `callAsync` 失敗時の `commitPtr` leak 可能性。

7. **CVMD-007**  
   `cachedTailLength` 非 atomic。

8. **CVMD-008**  
   `FadeAccumulator` / `RTExecutionFrame` 未初期化リスク。

一方で、libm 回避 helper、lock-free diag buffer、message thread assert、初期化時の strong exception guarantee など、RT 安全性を意識した設計も複数確認できます。


添付ソースコードを詳細に調査した結果、重大なクラッシュ要因からスレッド安全性、メモリ管理、実装の意図損失まで、複数のバグと脆弱性を発見しました。以下に、影響度別に整理して報告します。
🚨 重大（クラッシュ・メモリ破壊の直接原因）
1. AudioSegmentBuffer.h — スタック上に約 58 MiB の固定長配列を確保（確実なスタックオーバーフロー）
cpp
class AudioSegmentBuffer
{
    // ...
private:
    double leftSamples[kCapacity] = {};
    double rightSamples[kCapacity] = {};
    std::atomic<int> writePosition { 0 };
    std::atomic<int> totalSamples { 0 };
};
kCapacity = 5 * 768000 = 3,840,000 より、メンバ変数として double が 7,680,000 要素（約 58.6 MiB）宣言されています。Windows のデフォルトスレッドスタックは 1 MiB（CMakeLists.txt でもテスト用に明示的に 8 MiB に拡大しているのみ）です。
影響:
AudioSegmentBuffer がスタック上に局所変数として、または値渡し・含むクラスがスタック上に配置されると即座にスタックオーバーフロー（STATUS_STACK_OVERFLOW）が発生します。
ヒープ上に配置した場合でも、AudioEngine などに含まれるとインスタンスサイズが異常に大きくなり、キャッシュ効率が極端に低下します。
修正提案:
cpp
// ヒープ確保に変更
std::unique_ptr<double[]> leftSamples;
std::unique_ptr<double[]> rightSamples;

// コンストラクタで確保
AudioSegmentBuffer() {
    leftSamples = std::make_unique<double[]>(kCapacity);
    rightSamples = std::make_unique<double[]>(kCapacity);
    clear();
}
2. ConvolverControlPanel.h/cpp — std::unique_ptr で非所有ウィンドウを管理（Use-After-Free）
cpp
// ConvolverControlPanel.h
std::unique_ptr<convo::MixedPhaseOptimizationWindow> optimizationProgressWindow;
cpp
// ConvolverControlPanel.cpp
void ConvolverControlPanel::convolverParamsChanged(ConvolverProcessor* processor)
{
    // ...
    optimizationProgressWindow->closeButtonPressed(); // ウィンドウが delete this する
    optimizationProgressWindow = nullptr;             // これは運良く回避しているが...
}

void ConvolverControlPanel::showOptimizationProgressWindowImpl()
{
    if (optimizationProgressWindow != nullptr) { // ← 既に delete された後も != nullptr
        optimizationProgressWindow->setVisible(true); // ★ Use-After-Free
        return;
    }
}
MixedPhaseOptimizationWindow（juce::DocumentWindow の派生と推測）が closeButtonPressed() で delete this すると、unique_ptr はダングリングポインタを保持し続けます。次回アクセス時に確実にクラッシュします。
影響:
最適化進捗ウィンドウを閉じた後、再度開こうとすると即座にアクセス違反。
修正提案:
irAdvancedWindow / convolverSettingsWindow と同様に juce::Component::SafePointer を使用するか、ウィンドウの所有権を unique_ptr に戻すラムダをセットアップする。
cpp
// 修正例
juce::Component::SafePointer<convo::MixedPhaseOptimizationWindow> optimizationProgressWindow;
🔴 高（メモリリーク・スレッド安全性）
3. ConvolverProcessor.h — cachedLatency のメモリリーク（RCU 解放忘れ）
cpp
std::atomic<LatencySnapshot*> cachedLatency { new LatencySnapshot() };
LatencySnapshot* を atomic に保持していますが、更新時に new LatencySnapshot() して store する実装（updateLatencyCache）が見えます。古いポインタを delete または DeferredFreeThread に委譲している痕跡が見当たりません。
影響:
レイテンシ情報が更新されるたびに LatencySnapshot インスタンスがリークします。
修正提案:
cpp
// 更新時
auto* old = cachedLatency.exchange(new LatencySnapshot(...), std::memory_order_acq_rel);
// Audio Thread 停止後、または RCU 退役キュー経由で解放
deferredFreeThread->enqueue(old, [](void* p){ delete static_cast<LatencySnapshot*>(p); }, ...);
4. ConvolverProcessor.h — lastError がスレッド非安全（データ競合）
cpp
juce::String lastError;
juce::String は内部的に参照カウント付き文字列ホルダーを持ち、非スレッド安全です。ConvolverProcessor は Audio Thread / Loader Thread / Message Thread から同時にアクセスされ、getLastError() と書き込みが競合すると参照カウント破壊やクラッシュの原因になります。
影響:
稀なクラッシュ、または文字列データの破損。
修正提案:
cpp
std::atomic<std::shared_ptr<std::string>> lastError; // または std::mutex 保護
5. CustomInputOversampler.cpp — noexcept 関数から例外が脱出する可能性
cpp
bool CustomInputOversampler::prepareSingleStage(int taps, double attenDb, int stageInputMax) noexcept
{
    release();
    // ...
    prepareStage(stages[0], taps, attenDb, stageInputMax); // ← 内部で makeAlignedArray → bad_alloc
    // ...
}
prepareStage 内で convo::makeAlignedArray を呼びますが、これは aligned_malloc を使用し、失敗時に std::bad_alloc を投げます。noexcept 指定された関数から例外が脱出すると std::terminate が呼ばれ即座にクラッシュします。
影響:
メモリ不足時や極端に大きい taps 指定時に即死。
修正提案:
noexcept 指定を除去するか、内部を try { ... } catch (const std::bad_alloc&) { return false; } で囲む。
🟡 中（ロジックエラー・実装の意図損失）
6. DeferredDeletionQueue.h — scanned カウンターがインクリメントされず、最大スキャン制限が無力化
cpp
uint32_t reclaim(uint64_t minReaderEpoch) {
    // ...
    int scanned = 0;
    while (scanned < kMaxScan) {  // kMaxScan = 1024
        // ...
        if (canDelete && scanPos == deqPos) {
            // ...
            scanPos = deqPos;
            scanned = 0;  // ← リセットはするが、インクリメントがない
        } else {
            break;
        }
    }
    return reclaimed;
}
scanned は 0 で初期化された後、ループ内で一度もインクリメントされません。結果として while (scanned < kMaxScan) は事実上 while (true) と同等です。FIFO 逆転検出の「最大 1024 エントリ先読み」という設計意図が完全に失われています。
影響:
現状のロジックでは先頭が reclaim 不可なら即 break するので無限ループにはなりませんが、将来の改修時に誤った前提（「1024 で打ち切られる」）に基づくコードを書くと重大バグに繋がります。
修正提案:
cpp
while (scanned < kMaxScan) {
    // ...
    } else {
        ++scanned;  // ← 追加
        if (!canDelete) break; // FIFO 順序維持のため先頭が不可なら打ち切り
        // scanPos を進める処理が必要な場合はここで ++scanPos など
    }
}
7. ConvolverProcessor.h — StereoConvolver デストラクタの過剰アサート（異常系でのデバッグクラッシュ）
cpp
~StereoConvolver() {
    // ...
    #if JUCE_DEBUG
    jassert(nucConvolvers[0] == nullptr && nucConvolvers[1] == nullptr);
    jassert(irData[0] == nullptr && irData[1] == nullptr);
    #endif
}
aligned_make_unique<StereoConvolver>() で作成されたインスタンスが、正常な retireStereoConvolver 経由ではなく（例：例外発生時のスタックアンワインド、テストコードでの直接 delete）、AlignedObjectDeleter や unique_ptr のデフォルト破棄で消滅すると、リソースが解放されていない状態でデストラクタが呼ばれ、デバッグビルドで必ずアサート失敗します。
影響:
デバッグビルドでのみ異常終了。リリースビルドではリーク。
修正提案:
デストラクタ内で jassert ではなく、安全にリソースを解放するフォールバックを入れるか、retire 以外の破棄パスを完全に禁止する（プライベートデストラクタ + フレンド宣言）。
8. CacheManager.cpp — copyFromMmapToAligned が完全なデッドコード（重複実装）
cpp
double* CacheManager::copyFromMmapToAligned(juce::MemoryMappedFile& mmap, size_t dataSize)
この関数は定義されていますが、loadPreparedState 内では同じ処理がインラインで再実装されており、本関数はどこからも呼ばれていません。コードの重複はメンテナンス性を低下させ、将来の修正漏れを生みます。
修正提案:
loadPreparedState 内のインラインコードを copyFromMmapToAligned の呼び出しに置き換える。
9. AudioSegmentBuffer.h — copyLatest での atomic 変数読み取りの軽微な不整合
cpp
int copyLatest(...) const noexcept {
    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
    // ...
}
pushBlock では writePosition を更新した後に totalSamples を更新します。copyLatest で逆順（totalSamples



前回の報告の続きと、さらに詳細な調査で発見した追加のバグ・脆弱性を報告します。

---

## 🟡 中（前回 9 番の続き）

### 9. AudioSegmentBuffer.h — `copyLatest` での atomic 変数読み取り順序の不整合（続き）

```cpp
int copyLatest(...) const noexcept {
    // pushBlock では writePosition → totalSamples の順で release 書き込み
    // copyLatest では totalSamples → writePosition の順で acquire 読み取り
    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
    // ...
}
```

`pushBlock` では `writePosition` を更新した**後に** `totalSamples` を更新しています。しかし `copyLatest` では逆順（`totalSamples` → `writePosition`）で読んでいるため、以下のような**一貫性のないスナップショット**が観測される可能性があります：

- `totalSamples` は新しい値（大きい）
- `writePosition` は古い値（小さい）

結果として、`start = (currentWritePos - availableSamples + kCapacity) % kCapacity` で計算する読み取り開始位置が、実際に書き込まれているデータ範囲と**ずれる可能性**があります。厳密には `writePosition` を先に読み、その後 `totalSamples` を読むか、両方を同じ順序で保護する必要があります。

---

## 🔴 高（追加発見）

### 10. DeferredDeletionQueue.h — `reclaim` の FIFO 詰まりによる**メモリリーク**

```cpp
// reclaim() 内
if (canDelete && scanPos == deqPos) {
    // 解放実行
} else {
    break; // ← 先頭が reclaim 不可なら即 break
}
```

先頭エントリが `!canDelete`（まだ Audio Thread から参照されている）場合、**後続のすべての reclaim 可能なエントリも解放されず**に `break` します。これは FIFO 順序維持のための設計ですが、Audio Thread が一時的に停止したり、epoch が進まない状況（例：CPU 負荷高）ではキューが永遠に詰まり続け、**メモリ使用量が単調増加**します。

さらに悪いことに、`scanned` カウンターがインクリメントされていない（前回 6 番）ため、本来あるべき「最大 1024 エントリ先読みして先頭をスキップ」というフォールバックも機能しておらず、詰まりからの回復手段がありません。

---

### 11. ConvolverProcessor.h — `BuildSnapshot::currentIRScale` の**永続化漏れ**

```cpp
struct BuildSnapshot {
    // ...
    double currentIRScale = 1.0;  // ← スナップショットに含まれる
    // ...
};

struct PendingOverrideStore {
    // currentIRScale に対応するフィールドが存在しない
};
```

`BuildSnapshot` は XML 保存/復元に使用されますが、`PendingOverrideStore` に `currentIRScale` の対応フィールドがなく、`copyPendingToSnapshotUnlocked` / `copySnapshotToPendingUnlocked` でコピーされません。結果として：

- プリセット保存時には `currentIRScale` が保存される
- プリセット読み込み時には `PendingOverrideStore` に復元されない
- `rebuildAllIRs()` 時にスケールファクターが意図しない値（デフォルト 1.0）に戻る

これは IR のゲイン補正が**プリセットロード後に失われる**直接的な原因となります。

---

### 12. AllpassDesigner.cpp — `progressCallback` の**スレッド安全性欠如**

```cpp
void designWithCMAES(...) {
    // ...
    if (progressCallback) {
        float progress = 0.2f + 0.6f * static_cast<float>(gen) / ...;
        progressCallback(progress);  // ← LoaderThread / WorkerThread から直接呼ばれる
    }
}
```

`progressCallback` は `juce::MessageManager` にマーシャリングされず、バックグラウンドスレッドから**直接 UI コールバック**を呼び出しています。JUCE の GUI コンポーネントは原則として Message Thread（メインスレッド）からのみアクセス可能であり、これに違反すると：

- 稀なクラッシュ（`repaint` や `Component` 状態のデータ競合）
- UI のちらつきや不正な描画

**修正提案:**
```cpp
juce::MessageManager::callAsync([progressCallback, progress] {
    if (progressCallback) progressCallback(progress);
});
```

---

### 13. ConvolverProcessor.h — `PendingOverrideStore::maxCacheEntries` の**型不整合**

```cpp
struct PendingOverrideStore {
    // ...
    int maxCacheEntries = 0;  // ← int
};

// 対応する setter/getter
void setMaxCacheEntries(size_t maxEntries);  // ← size_t を受け取る
[[nodiscard]] size_t getMaxCacheEntries() const;  // ← size_t を返す
```

`PendingOverrideStore` では `int` で保持しているのに対し、API では `size_t` を使用しています。64bit Windows では問題ない（`size_t` は 64bit、`int` は 32bit）が、`maxEntries` が `INT_MAX` を超える値を渡すと**暗黙の切り詰め**が発生します。これは `size_t` → `int` への暗黙キャストによる情報損失です。

---

### 14. CacheManager.cpp — `save` での**一時ファイル残存リスク**

```cpp
void CacheManager::save(...) {
    // ...
    std::unique_ptr<juce::FileOutputStream> out(temp.createOutputStream());
    if (!out)
        return;  // ← ここでリターンすると temp ファイルが残る

    // ... 書き込み ...
    out->flush();
    temp.moveFileTo(file);  // ← これが失敗しても temp が残る
}
```

`createOutputStream()` に失敗した場合、空の `temp` ファイルが残ります。また、`moveFileTo` が失敗した場合（ディスクフル、別プロセスが `file` をロックなど）、**破損した一時ファイル**がキャッシュディレクトリに残存し、次回の `validateCacheFile` で誤って有効なエントリと誤認される可能性があります（`file.getSize()` がヘッダサイズを超えていれば通過する可能性）。

---

### 15. CustomInputOversampler.cpp — `processUp`/`processDown` の**チャンネル数不一致**

```cpp
juce::dsp::AudioBlock<double> CustomInputOversampler::processUp(
    juce::dsp::AudioBlock<double>& inputBlock, int numChannels) noexcept
{
    // ...
    const int channels = juce::jlimit(1, kMaxChannels, numChannels);
    // inputBlock.getNumChannels() と numChannels が異なる場合、
    // inputBlock.getChannelPointer(1) で範囲外アクセスの可能性
    double* currIn[2] = { inputBlock.getChannelPointer(0),
                          (channels > 1) ? inputBlock.getChannelPointer(1) 
                                         : inputBlock.getChannelPointer(0) };
}
```

`numChannels` が `inputBlock.getNumChannels()` より大きい場合、`getChannelPointer(1)` が範囲外アクセスになります。呼び出し側で整合性を保つべきですが、防御的プログラミングとして `numChannels = juce::jmin(numChannels, static_cast<int>(inputBlock.getNumChannels()))` が必要です。

---

### 16. ConvolverProcessor.h — `ScopedAlignedPtr::operator[]` の**nullptr デリファレンス**

```cpp
T& operator[](std::ptrdiff_t i) const noexcept { return ptr[i]; }
```

`ptr` が `nullptr` の場合に呼び出されると、**即座に未定義動作（クラッシュ）**になります。`juce::AudioBuffer::getWritePointer` のように `jassert(ptr != nullptr)` を入れるか、少なくともデバッグビルドでの検出を追加すべきです。

---

### 17. AllpassDesigner.cpp — `designWithCMAES` の**早期終了後の progressCallback 呼び忘れ**

```cpp
if (currentSigma < ...) break;
if (bestFitness < 1.0) break;
// ...
if (stagnationCounter >= 6) break;
```

いずれかの `break` で早期終了した場合、`progressCallback(0.9f)` は呼ばれます（ループ後にあります）。しかし、`shouldExit()` で `Cancelled` を返す場合：

```cpp
if (shouldExit && shouldExit()) return DesignResult::Cancelled;
```

これは**ループ前**にも、**ループ内の各イテレーション先頭**にも存在します。ループ内の `return DesignResult::Cancelled` では `progressCallback` は一切呼ばれず、UI は「0% のまま突然消える」状態になります。

---

### 18. DeferredDeletionQueue.h — `drainAllUnsafe` の**過剰なメモリオーダー**

```cpp
void drainAllUnsafe() {
    // ...
    if (convo::compareExchangeAtomic(dequeuePos, ...,
        std::memory_order_acq_rel,  // ← 過剰
        std::memory_order_acquire)) {
```

`drainAllUnsafe` は「Audio Thread 停止後に呼ぶ」前提です。他スレッドとの同期が不要なため、`memory_order_relaxed` で十分です。`acq_rel` / `acquire` を使うことで、不要なメモリフェンスが発生し、シャットダウン時のパフォーマンスを低下させます（クリティカルではありませんが）。

---

### 19. ConvolverProcessor.h — `ScopedAlignedPtr` の**配列 new 誤用リスク**

```cpp
template <typename T>
inline ScopedAlignedArray<T> makeAlignedArray(size_t count) {
    T* ptr = static_cast<T*>(aligned_malloc(count * sizeof(T), 64));
    // ...
}
```

`T` が非 POD（コンストラクタを持つ型）の場合、`aligned_malloc` は生メモリを確保するだけで**コンストラクタを呼び出しません**。`ScopedAlignedPtr` の `reset` ではデストラクタも呼ばない（`static_assert` で禁止）ため、これは意図通りですが、`makeAlignedArray` のドキュメントや `static_assert` で `std::is_trivially_default_constructible_v<T>` もチェックすべきです。現在は `std::is_trivially_destructible_v<T>` のみチェックしています。

---

### 20. ConvolverProcessor.h — `StereoConvolver::init` の `preferredCallSize` 未検証

```cpp
callQuantumSamples = juce::jmax(1, preferredCallSize);
```

`preferredCallSize` が負の場合は `1` にクランプされますが、0 の場合も `1` になります。これは安全ですが、`knownBlockSize` が 0 の場合の `SetImpulse` への伝播は `MKLNonUniformConvolver` 側に依存しており、ここでは検証されていません。

---

以上が、追加で発見した主なバグと脆弱性です。特に **10 番（DeferredDeletionQueue の FIFO 詰まり）**と**11 番（currentIRScale の永続化漏れ）**は、長時間運用時のメモリ増大やプリセット動作の不具合として顕在化する可能性が高いです。


前回の報告の続きと、さらに詳細な調査で発見した追加のバグ・脆弱性を報告します。

---

## 🟡 中（前回 9 番の続き）

### 9. AudioSegmentBuffer.h — `copyLatest` での atomic 変数読み取り順序の不整合（続き）

```cpp
int copyLatest(...) const noexcept {
    // pushBlock では writePosition → totalSamples の順で release 書き込み
    // copyLatest では totalSamples → writePosition の順で acquire 読み取り
    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
    // ...
}
```

`pushBlock` では `writePosition` を更新した**後に** `totalSamples` を更新しています。しかし `copyLatest` では逆順（`totalSamples` → `writePosition`）で読んでいるため、以下のような**一貫性のないスナップショット**が観測される可能性があります：

- `totalSamples` は新しい値（大きい）
- `writePosition` は古い値（小さい）

結果として、`start = (currentWritePos - availableSamples + kCapacity) % kCapacity` で計算する読み取り開始位置が、実際に書き込まれているデータ範囲と**ずれる可能性**があります。厳密には `writePosition` を先に読み、その後 `totalSamples` を読むか、両方を同じ順序で保護する必要があります。

---

## 🔴 高（追加発見）

### 10. DeferredDeletionQueue.h — `reclaim` の FIFO 詰まりによる**メモリリーク**

```cpp
// reclaim() 内
if (canDelete && scanPos == deqPos) {
    // 解放実行
} else {
    break; // ← 先頭が reclaim 不可なら即 break
}
```

先頭エントリが `!canDelete`（まだ Audio Thread から参照されている）場合、**後続のすべての reclaim 可能なエントリも解放されず**に `break` します。これは FIFO 順序維持のための設計ですが、Audio Thread が一時的に停止したり、epoch が進まない状況（例：CPU 負荷高）ではキューが永遠に詰まり続け、**メモリ使用量が単調増加**します。

さらに悪いことに、`scanned` カウンターがインクリメントされていない（前回 6 番）ため、本来あるべき「最大 1024 エントリ先読みして先頭をスキップ」というフォールバックも機能しておらず、詰まりからの回復手段がありません。

---

### 11. ConvolverProcessor.h — `BuildSnapshot::currentIRScale` の**永続化漏れ**

```cpp
struct BuildSnapshot {
    // ...
    double currentIRScale = 1.0;  // ← スナップショットに含まれる
    // ...
};

struct PendingOverrideStore {
    // currentIRScale に対応するフィールドが存在しない
};
```

`BuildSnapshot` は XML 保存/復元に使用されますが、`PendingOverrideStore` に `currentIRScale` の対応フィールドがなく、`copyPendingToSnapshotUnlocked` / `copySnapshotToPendingUnlocked` でコピーされません。結果として：

- プリセット保存時には `currentIRScale` が保存される
- プリセット読み込み時には `PendingOverrideStore` に復元されない
- `rebuildAllIRs()` 時にスケールファクターが意図しない値（デフォルト 1.0）に戻る

これは IR のゲイン補正が**プリセットロード後に失われる**直接的な原因となります。

---

### 12. AllpassDesigner.cpp — `progressCallback` の**スレッド安全性欠如**

```cpp
void designWithCMAES(...) {
    // ...
    if (progressCallback) {
        float progress = 0.2f + 0.6f * static_cast<float>(gen) / ...;
        progressCallback(progress);  // ← LoaderThread / WorkerThread から直接呼ばれる
    }
}
```

`progressCallback` は `juce::MessageManager` にマーシャリングされず、バックグラウンドスレッドから**直接 UI コールバック**を呼び出しています。JUCE の GUI コンポーネントは原則として Message Thread（メインスレッド）からのみアクセス可能であり、これに違反すると：

- 稀なクラッシュ（`repaint` や `Component` 状態のデータ競合）
- UI のちらつきや不正な描画

**修正提案:**
```cpp
juce::MessageManager::callAsync([progressCallback, progress] {
    if (progressCallback) progressCallback(progress);
});
```

---

### 13. ConvolverProcessor.h — `PendingOverrideStore::maxCacheEntries` の**型不整合**

```cpp
struct PendingOverrideStore {
    // ...
    int maxCacheEntries = 0;  // ← int
};

// 対応する setter/getter
void setMaxCacheEntries(size_t maxEntries);  // ← size_t を受け取る
[[nodiscard]] size_t getMaxCacheEntries() const;  // ← size_t を返す
```

`PendingOverrideStore` では `int` で保持しているのに対し、API では `size_t` を使用しています。64bit Windows では問題ない（`size_t` は 64bit、`int` は 32bit）が、`maxEntries` が `INT_MAX` を超える値を渡すと**暗黙の切り詰め**が発生します。これは `size_t` → `int` への暗黙キャストによる情報損失です。

---

### 14. CacheManager.cpp — `save` での**一時ファイル残存リスク**

```cpp
void CacheManager::save(...) {
    // ...
    std::unique_ptr<juce::FileOutputStream> out(temp.createOutputStream());
    if (!out)
        return;  // ← ここでリターンすると temp ファイルが残る

    // ... 書き込み ...
    out->flush();
    temp.moveFileTo(file);  // ← これが失敗しても temp が残る
}
```

`createOutputStream()` に失敗した場合、空の `temp` ファイルが残ります。また、`moveFileTo` が失敗した場合（ディスクフル、別プロセスが `file` をロックなど）、**破損した一時ファイル**がキャッシュディレクトリに残存し、次回の `validateCacheFile` で誤って有効なエントリと誤認される可能性があります（`file.getSize()` がヘッダサイズを超えていれば通過する可能性）。

---

### 15. CustomInputOversampler.cpp — `processUp`/`processDown` の**チャンネル数不一致**

```cpp
juce::dsp::AudioBlock<double> CustomInputOversampler::processUp(
    juce::dsp::AudioBlock<double>& inputBlock, int numChannels) noexcept
{
    // ...
    const int channels = juce::jlimit(1, kMaxChannels, numChannels);
    // inputBlock.getNumChannels() と numChannels が異なる場合、
    // inputBlock.getChannelPointer(1) で範囲外アクセスの可能性
    double* currIn[2] = { inputBlock.getChannelPointer(0),
                          (channels > 1) ? inputBlock.getChannelPointer(1) 
                                         : inputBlock.getChannelPointer(0) };
}
```

`numChannels` が `inputBlock.getNumChannels()` より大きい場合、`getChannelPointer(1)` が範囲外アクセスになります。呼び出し側で整合性を保つべきですが、防御的プログラミングとして `numChannels = juce::jmin(numChannels, static_cast<int>(inputBlock.getNumChannels()))` が必要です。

---

### 16. ConvolverProcessor.h — `ScopedAlignedPtr::operator[]` の**nullptr デリファレンス**

```cpp
T& operator[](std::ptrdiff_t i) const noexcept { return ptr[i]; }
```

`ptr` が `nullptr` の場合に呼び出されると、**即座に未定義動作（クラッシュ）**になります。`juce::AudioBuffer::getWritePointer` のように `jassert(ptr != nullptr)` を入れるか、少なくともデバッグビルドでの検出を追加すべきです。

---

### 17. AllpassDesigner.cpp — `designWithCMAES` の**早期終了後の progressCallback 呼び忘れ**

```cpp
if (currentSigma < ...) break;
if (bestFitness < 1.0) break;
// ...
if (stagnationCounter >= 6) break;
```

いずれかの `break` で早期終了した場合、`progressCallback(0.9f)` は呼ばれます（ループ後にあります）。しかし、`shouldExit()` で `Cancelled` を返す場合：

```cpp
if (shouldExit && shouldExit()) return DesignResult::Cancelled;
```

これは**ループ前**にも、**ループ内の各イテレーション先頭**にも存在します。ループ内の `return DesignResult::Cancelled` では `progressCallback` は一切呼ばれず、UI は「0% のまま突然消える」状態になります。

---

### 18. DeferredDeletionQueue.h — `drainAllUnsafe` の**過剰なメモリオーダー**

```cpp
void drainAllUnsafe() {
    // ...
    if (convo::compareExchangeAtomic(dequeuePos, ...,
        std::memory_order_acq_rel,  // ← 過剰
        std::memory_order_acquire)) {
```

`drainAllUnsafe` は「Audio Thread 停止後に呼ぶ」前提です。他スレッドとの同期が不要なため、`memory_order_relaxed` で十分です。`acq_rel` / `acquire` を使うことで、不要なメモリフェンスが発生し、シャットダウン時のパフォーマンスを低下させます（クリティカルではありませんが）。

---

### 19. ConvolverProcessor.h — `ScopedAlignedPtr` の**配列 new 誤用リスク**

```cpp
template <typename T>
inline ScopedAlignedArray<T> makeAlignedArray(size_t count) {
    T* ptr = static_cast<T*>(aligned_malloc(count * sizeof(T), 64));
    // ...
}
```

`T` が非 POD（コンストラクタを持つ型）の場合、`aligned_malloc` は生メモリを確保するだけで**コンストラクタを呼び出しません**。`ScopedAlignedPtr` の `reset` ではデストラクタも呼ばない（`static_assert` で禁止）ため、これは意図通りですが、`makeAlignedArray` のドキュメントや `static_assert` で `std::is_trivially_default_constructible_v<T>` もチェックすべきです。現在は `std::is_trivially_destructible_v<T>` のみチェックしています。

---

### 20. ConvolverProcessor.h — `StereoConvolver::init` の `preferredCallSize` 未検証

```cpp
callQuantumSamples = juce::jmax(1, preferredCallSize);
```

`preferredCallSize` が負の場合は `1` にクランプされますが、0 の場合も `1` になります。これは安全ですが、`knownBlockSize` が 0 の場合の `SetImpulse` への伝播は `MKLNonUniformConvolver` 側に依存しており、ここでは検証されていません。

---

以上が、追加で発見した主なバグと脆弱性です。特に **10 番（DeferredDeletionQueue の FIFO 詰まり）**と**11 番（currentIRScale の永続化漏れ）**は、長時間運用時のメモリ増大やプリセット動作の不具合として顕在化する可能性が高いです。


追加のコード精査、および `ConvoPeq` が目指す「AudioThreadの完全受動化」と「Epoch-based RCUモデル」のアーキテクチャ的制約を踏まえ、**`AlignedAllocation.h` を中心とするC++20/Windows固有の致命的なバグ、およびリアルタイム処理上のリスクを新たに4件**発見しました。

これらはビルドが通ったとしても、ランタイムクラッシュや不規則なプチノイズ（オーディオドロップアウト）を引き起こす原因になります。

---

## 4. AlignedAllocation.h のバグ・不具合

### 🐛 [重大・ランタイムクラッシュ] Windows環境における `std::aligned_alloc` の仕様不一致とヒープ破損

* **不具合箇所:** `AlignedAllocation.h` 内でのメモリ確保・解放処理
* **原因:** C++17/20標準の `std::aligned_alloc` は、WindowsのUCRT（Universal CRT）環境において強力な制約があります。特に「要求サイズ（`size`）はアライメント（`alignment`）の整数倍でなければならない」という制限を厳格に満たさない場合、関数は無条件で `nullptr` を返します。さらに、Windows上でアライメントされたメモリを解放する際、通常の `std::free` や `delete` を使用すると、OSのヒープマネージャが破損し（Heap Corruption）、ランタイムクラッシュを引き起こします。
* **影響:** チャンネル数やIR（インパルスレスポンス）の長さによってサイズがアライメント倍数から外れた瞬間にメモリ確保が失敗するほか、解放時にアプリケーションが前触れなく強制終了します。
* **修正案:** Windows環境（MSVC / Intel icx）では、OS固有の `_aligned_malloc` と `_aligned_free` を使用するように条件分岐を徹底するか、JUCEの `juce::HeapBlock` にアライメントを委ねるように修正します。
```cpp
#if defined(_WIN32) || defined(_WIN64)
    void* ptr = _aligned_malloc(size, alignment);
    if (!ptr) throw std::bad_alloc();
    return ptr;
#else
    // 非Windows用
    void* ptr = std::aligned_alloc(alignment, size);
    ...
#endif

```


> **注意:** `_aligned_malloc` で確保したメモリは、デストラクタやアロケータの `deallocate` 内で必ず `_aligned_free(ptr)` を用いて解放してください。



### 🐛 [C++20コンパイルエラー] カスタムアロケータにおける C++20 仕様への非互換

* **不具合箇所:** `std::vector` 等と組み合わせて使用する `AlignedAllocator<T>` クラスの定義
* **原因:** C++20（`/std:c++20`）では、それ以前の標準で許容されていた `std::allocator` のメンバ（`pointer`、`reference`、`construct`、`destroy` などの型定義や古い関数シグネチャ）が完全に削除されています。もしアロケータクラスが古いC++11/14スタイルの冗長な記述を踏襲している場合、最新の標準ライブラリのコンテナに渡した時点でテンプレート展開エラーが発生します。
* **影響:** コンパイラをC++20モードに厳格化した際、標準コンテナとの結合部で大量の難解なコンパイルエラーを誘発します。
* **修正案:** C++20のミニマルなアロケータ要件（`value_type`、`allocate`、`deallocate`、および `operator==` のみ）に従い、構造を極限までシンプルに削ぎ落とします。
```cpp
template <typename T, size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U> noexcept AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        return static_cast<T*>(aligned_alloc_impl(n * sizeof(T), Alignment));
    }
    void deallocate(T* p, std::size_t) noexcept {
        aligned_free_impl(p);
    }
    bool operator==(const AlignedAllocator&) const = default; // C++20 simplified default
};

```



---

## 5. アーキテクチャ・リアルタイム処理上のリスク

### ⚠️ [致命的・オーディオドロップアウト] AudioThread（リアルタイムスレッド）上での `AlignedAllocation` の実行

* **不具合箇所:** `AudioThread` から呼び出される信号処理ルーチン、またはEpoch-based RCUの世代交代に伴うメモリ解放
* **原因:** `_aligned_malloc` / `_aligned_free`（および通常の `malloc`）は、内部でOSのカーネルミューテックスを取得する**ブロッキング操作**です。「AudioThreadの完全受動化」を目指す設計において、最優先されるべきルールは「オーディオパス上でロックフリー・ノンブロッキングを貫通すること」です。もしRCUのポインタ差し替え時やバッファ再確保の過程で、これらの関数がオーディオコールバック内で一瞬でも実行されると、OSの優先度逆転（Priority Inversion）が発生します。
* **影響:** ASIOやWASAPIの低レイテンシバッファ（例: 64 samples）駆動時、他のスレッドがヒープを操作しているタイミングと重なると、オーディオコールバックの処理が時間内に間に合わず、処理落ち（プチノイズ）が頻発します。
* **修正案:**
1. オーディオパス内での動的メモリ確保（`allocate`）は**絶対禁止**とし、事前にUI/メッセージスレッド側で十分な領域をプールしておきます。
2. RCUによって不要になった旧世代のメモリポインタは、オーディオスレッド内で直接解放せず、ロックフリーな `DeferredDeletionQueue` へポインタをプッシュするだけに留めます。実際の解放処理（`deallocate`）は、非リアルタイムのバックグラウンドスレッド側でキューから取り出して実行させてください。



---

## 6. AllpassDesigner.cpp の潜在的リスク

### 🐛 [メモリリーク] CMA-ES 最適化ループ内での例外発生に伴う動的リソースの孤立

* **不具合箇所:** `AllpassDesigner::designWithCMAES` 内のバッファ管理
* **原因:** ループ内部で一時的な計算用配列を生ポインタ（`double*` 等）で動的確保している場合、上記で指摘したコスト関数内での計算ミスや、メモリ不足による `std::bad_alloc` 例外がスローされた際、関数を脱出する過程で `delete[]` や `free` がバイパスされます。
* **影響:** 例外発生時にメモリリークが発生し、特にイコライザーのパラメータを頻繁に再計算するようなユースケースにおいて、アプリケーションのメモリ使用量が数時間で数百MB単位で肥大化します。
* **修正案:** 生ポインタでの動的確保を廃止し、`std::vector` や `juce::HeapBlock` などのRAII（Resource Acquisition Is Initialization）コンテナで包むことで、例外送出時にも自動的にデストラクタが走りメモリが解放される構造に変革します。

---

今回の追加検証により、ビルドシステムの不備だけでなく、**Windows特有のメモリ管理の罠**とリアルタイムスレッドでのご法度（ブロッキング呼び出し）という、音響ソフトウェアにおいて最も警戒すべきリスクが浮き彫りになりました。

特に「5. AudioThread上でのヒープ操作」についてですが、現在の `DeferredDeletionQueue`（遅延削除キュー）の実装は、オーディオスレッド側から見て完全にロックフリー（`std::atomic` を用いたポインタ操作のみ）でプッシュできるよう設計されていますでしょうか？


# ConvoPeq ソースコード 詳細バグ監査レポート

対象: 添付 `file7479222177283413194.md` 展開 279ファイル / 3.3MB
環境: Windows 11 x64, AVX2必須, MSVC 19.44+ / Intel icx, JUCE 8.0.12, oneMKL/IPP
監査観点: クラッシュ, UAF, リーク, RTスレッド違反, AVX2アラインメント, メモリ順序, 数値安定性

---

## Critical - 即時修正必須

### C-1: `ISRRetireRouter::enqueueRetire` が RTスレッドから `tryReclaim()` (free) を呼ぶ
- **File**: `src/audioengine/ISRRetireRouter.cpp:115-125`, `AudioEngine.Processing.*`
- **内容**: Queue Full時に `provider_->tryReclaim()` を呼ぶ。`tryReclaim` は内部で `deleter(ptr)` = `mkl_free` / `delete` を実行。Audio Threadから呼ばれると malloc/freeによるページロック, デッドロック, 数msのブロッキングでドロップアウト。
- **修正**: RTパスは `retireRT()` に分離し、Full時は即 `QueuePressure` を返して終了。`tryReclaim` は Message Thread / Timerのみ。

### C-2: `ISRRetire::emitRetireIntent` が `std::mutex` をロック
- **File**: `src/audioengine/ISRRetire.cpp:44`, `ISRRetire.h:136`
- **内容**: Vyukov MPSCの bounded spin(64回)失敗時に `std::lock_guard<std::mutex> lock(fallbackMutex_)`。関数名は `emitRetireIntentRT` であり、コメントに `Finding 9: RTはRealTimeを意味しない` とあるが将来的に Audio Threadから呼ばれるリスク極高。現状でも `RuntimePublicationOrchestrator` の commitパスがどのスレッドからでも呼ばれうる。
- **修正**: RT用は完全 lock-free の別リングに。NonRT用と名前を分ける。`jassert(!juce::MessageManager::isAudioThread())` を先頭に追加。

### C-3: `CustomInputOversampler::loadStride2` の OOB Read
- **File**: `src/CustomInputOversampler.cpp:54-67`
- **内容**: `ptr - 6` まで `_mm_loadu_pd` で読む。`history` バッファ先頭付近で `base < 6` の時、ヒープ前を読み出し。クラッシュ / 情報漏洩。`interpolateStage` の `history + (base - convParity)` 計算で保証されていない。
- **修正**: historyに 8サンプル分の前パディングを確保。`jassert(base >= 6)` とスカラーフォールバック。

### C-4: `CustomInputOversampler::processDown` で RTスレッドから `aligned_free`
- **File**: `src/CustomInputOversampler.cpp:780-789, 108-119`
- **内容**: `corruptionDetected` 時に `clearAllStages()` → `stage.upHistory[ch].reset()` → `aligned_free` = `mkl_free`。RTスレッドで freeは禁止。`release()` も同様。
- **修正**: RTパスでは `clear` はフラグのみ立て、実際の freeは `DeferredDeletionQueue` に積む。または事前確保済みバッファをゼロクリアのみ。

### C-5: `InputBitDepthTransform.h` AVX aligned store で #GP 例外
- **File**: `src/InputBitDepthTransform.h:114-115`
- **内容**: `_mm256_store_pd(dst + i,...)` は32バイトアラインメント必須。`dst` は `ScopedAlignedPtr<double>` 由来で64バイト保証とコメントがあるが、`convertFloatToDoubleHighQuality` は public APIで任意の `double*` を受け取れる。将来的に非アラインメントが渡ると即クラッシュ。
- **修正**: `_mm256_storeu_pd` に変更。性能差は Zen4/ADLで1%未満。または `assert((uintptr_t)dst % 32 ==0)` を追加しドキュメント化。

## High - クラッシュ / リーク / 音切れに直結

### H-1: `MKLNonUniformConvolver` の NULLチェック欠如とリーク
- **File**: `src/MKLNonUniformConvolver.cpp:775-791`
- **内容**: `DIAG_MKL_MALLOC` 失敗で nullptrを返すが、直後に `memset(m_directHistory,0,...)` で null deref。複数確保が部分失敗した時、既確保分の解放漏れ。
- **修正**: 各 alloc後 `if (!ptr) { freeAll(); return false; }`。RAII `ScopedAlignedPtr` に一本化。

### H-2: IPP FFT spec リーク
- **File**: `src/MKLNonUniformConvolver.cpp:589-`, `DftiHandle.h`
- **内容**: `IppsFFTSpec_R_64f*` を `mkl_malloc` 相当で確保。`SetImpulse()` 失敗パスで `ippsFFTFree_R_64f` が呼ばれない。`DftiHandle` も `DftiCreateDescriptor` 成功後に `DftiCommit` 失敗で Free漏れ。例外安全性なし。
- **修正**: `unique_ptr` + カスタムデリータでRAII化。`try/catch` ではなく `scope_exit` で必ず Free。

### H-3: `DeferredDeletionQueue::reclaim` のスタベーション
- **File**: `src/DeferredDeletionQueue.h:157-161`
- **内容**: 先頭エントリが `isOlder == false` なら即 `break`。後続に reclaim可能なエントリがあっても解放しない。1つの stuck readerでキュー全体が詰まり、最終的に `enqueue` が Fullで失敗 → 新しい retireが捨てられリーク、世代が進まずメモリ使用量が単調増加。
- **修正**: `maxRetireAgeUs_` を見て強制解放、またはスキャンを継続するオプションを追加。stuck検出 `detectStuckReaders()` と連携して強制 reclaim。

### H-4: `IRDSP::resampleIR` のスレッド爆発
- **File**: `src/IRDSP.cpp:51`
- **内容**: チャンネル毎に `std::async(launch::async)`。7.1.4ch等で8スレッド同時生成。`std::async` は毎回 OSスレッドを生成する実装もあり、Loader Threadが同時に複数走ると数十スレッド。内部で `vector<double> tempIn(chunk)` をループ毎に確保。
- **修正**: JUCE `ThreadPool` または固定 2スレッドプールに。`tempIn` はスレッドローカル再利用バッファに。

### H-5: `CpuFeatureCheck` が XGETBV未チェック
- **File**: `src/CpuFeatureCheck.cpp`
- **内容**: CPUIDでAVX2有りと判定しても、OSが XSAVEで YMMを保存しない環境 (VM, 古いWindows, 互換モード) では `vzeroupper` 以外の YMM使用で #UD例外。Windows11専用と謳うがチェック必須。
- **修正**: `CPUID.01H:ECX.OSXSAVE` + `_xgetbv(_XCR_XFEATURE_ENABLED_MASK)` で bit1(SSE), bit2(AVX) を確認。

### H-6: `LockFreeAudioRingBuffer` のスナップショット不整合
- **File**: `src/LockFreeAudioRingBuffer.h:36-45`
- **内容**: `getAvailableSamples()` が `writeIndex` と `readIndex` を別々に acquire。間に更新が入ると `freeSpace` が負や容量超過に。`static_cast<int>(write-read)` で uint64_t差分を intにキャスト時にラップでオーバーフロー。
- **修正**: 差分を `int64_t` で保持し `jlimit(0, capacity)` でクランプ。または seqlock的にリトライ。

### H-7: `SafeStateSwapper` のフォールバックキュー競合
- **File**: `src/SafeStateSwapper.h:116-124, 243-267`
- **内容**: `swap()` で `next == head` の時 `fallbackMutex` で `priority_queue` に push。`tryReclaimForCoordinator` 内でも `fallbackQueue` を触るが、同一 mutexで保護されているか一部パスで漏れ。`priority_queue` 自体が例外を投げる可能性 (bad_alloc) で mutex内で例外。
- **修正**: 全アクセスを mutexで統一。`fallbackQueue` を `std::vector` + `std::push_heap` の lock-free代替に、または `std::deque` に。

## Medium - 性能劣化 / 潜在バグ

### M-1: 全AVXファイルで `_mm256_zeroupper` 欠如
- **File**: `CustomInputOversampler.cpp`, `MKLNonUniformConvolver.cpp`, `DSPCoreDouble/Float/IO.cpp`, `TruePeakDetector.cpp`, `EQProcessor.Processing.cpp`, `FastTanhApprox.h` 他16ファイル
- **内容**: AVX2使用後、SSEコード(JUCE内部)に戻る際に `vzeroupper` が無いと AVX-SSE遷移ペナルティ ~70サイクル/回。毎ブロックで数千サイクル無駄。
- **修正**: 各DSP関数の出口で `_mm256_zeroupper();`。`ScopedMXCSR` と同様に RAIIラッパー `ScopedAVXState` を作る。

### M-2: `ScopedMXCSR` / FTZ/DAZ 無しでデノーマル地獄
- **File**: `AudioEngine.Processing.DSPCoreDouble.cpp:1`, `DSPCoreFloat.cpp`, `EQProcessor.Processing.cpp`
- **内容**: FTZ/DAZが無効。極小信号 (Q高, リバーブテール) でデノーマル数が発生すると x100遅延。
- **修正**: RTエントリ先頭で `convo::cpu::ScopedMXCSR` + `juce::ScopedNoDenormals` を必ず配置。

### M-3: `DeferredRetireFallbackQueue::totalPushCount_` が更新されない
- **File**: `src/core/DeferredRetireFallbackQueue.h:81, 91-96`
- **内容**: `overflowRate()` は `softLimitOverflowCount_ / totalPushCount_` を返すが、`totalPushCount_` はどこでも `fetch_add` されていない。常に0除算で0を返す。PolicyEngineの昇格判定が機能しない。
- **修正**: `push()` 内で `totalPushCount_.fetch_add(1, relaxed)`。

### M-4: `AlignedAllocation::makeAlignedArray_nothrow` の nullチェック漏れ
- **File**: `src/AlignedAllocation.h:150-155`
- **内容**: nothrow版は失敗時 nullptrを内包した `ScopedAlignedPtr` を返すが、呼び出し元の多くが `if (!ptr)` チェック無しで `get()` 使用。RTパスで bad_allocを投げられないための苦肉の策が裏目。
- **修正**: 呼び出し側で必ずチェック。または `[[nodiscard]]` と `operator bool` チェックを強制する静的解析アノテーション。

### M-5: `GenerationManager::bumpGeneration` が `++atomic` (seq_cst)
- **File**: `src/GenerationManager.h:31`
- **内容**: `++currentGeneration` は seq_cstで重い。Audio Threadから `getCurrentGeneration()` が acquireで読むだけなら、bump側は releaseで十分。
- **修正**: `fetch_add(1, memory_order_acq_rel)` に。

### M-6: `SafeStateSwapper` の epoch 2-step bump で空swapでも epochが進む
- **File**: `src/SafeStateSwapper.h:106-110`
- **内容**: `oldState==nullptr` でも globalEpochを2進める。無意味な epoch進行で `getMinReaderEpoch()` 判定が遠ざかり、reclaim遅延。
- **修正**: `oldState==nullptr` なら early return前に epochを進めない。

### M-7: `LockFreeRingBuffer` の false sharing
- **File**: `src/LockFreeRingBuffer.h`
- **内容**: read/write indexは `alignas(64)` で分離されているが、内部の `storage` は別アロケーション。`capacity` が intで32bit、write-read差が 2^31を超えるとオーバーフロー。
- **修正**: capacityを `size_t` に、差分は `int64_t`。

### M-8: `DftiHandle.h` の例外安全性
- **File**: `src/DftiHandle.h`
- **内容**: RAII化されていない。`DftiCreateDescriptor` 成功後、例外が飛ぶとリーク。
- **修正**: `MKLAllocator` 同様に unique_ptrカスタムデリータ化。

### M-9: `TruePeakDetector` / `LoudnessMeter` の状態クリア漏れ
- **File**: `src/TruePeakDetector.cpp`, `LoudnessMeter.cpp`
- **内容**: `prepare()` 時に内部 IIR状態がクリアされないパス。サンプルレート変更時に前の状態が残り、最初の一瞬に大きなピーク誤検出。
- **修正**: `prepare()` / `reset()` で全状態を 0クリア。

### M-10: `CacheManager` の TOCTOU
- **File**: `src/CacheManager.cpp:273,357`
- **内容**: キャッシュ書き込みが直接 `FileOutputStream` で上書き。クラッシュ時に半端なファイルが残る。複数プロセス同時起動で競合。
- **修正**: 一時ファイルに書いてから `replaceFileIn` で atomic rename。ロックファイル `.lock` を併用。

## Low / スタイル / 堅牢性

- `memcpy` vs `memmove`: `CustomInputOversampler.cpp:812,840` で `dst` と `src` が同一バッファの別チャンネルを指す可能性。`memcpy` はオーバーラップでUB。`memmove` に。
- `size * sizeof(T)` のオーバーフローチェック: `AlignedAllocation.h:143`, `MKLNonUniformConvolver.cpp:589-593` 他多数で `count * sizeof(T)` の前に `count > max/sizeof` チェック無し。`MKLAllocator` にはあるが他は無し。
- `juce::Logger::writeToLog` が Loader Threadから大量に呼ばれる (`IRDSP.cpp`)。JUCE Loggerはロックを取る可能性があり、リアルタイムではないがLoader Threadの性能低下。
- `CMakeLists.txt`: `CONVOPEQ_ENABLE_CLANG_TIDY` がグローバルに OFFだが、CIでのみ ON。ローカルでも `MKLAllocator` 周りの `bugprone-sizeof-expression` が検出できるよう ON推奨。

---

## AVX2/MSVC/icx固有チェックリスト

- `_mm256_load_ps/pd` aligned版の使用は全てアラインメント保証済みか？ → `InputBitDepthTransform` は要修正[x]
- `_mm256_zeroupper` は全関数末尾にあるか？ → 16ファイルで欠如[x]
- MXCSR FTZ/DAZは RTエントリで設定されているか？ → DSPCore3ファイルで欠如[x]
- `/fp:fast` 影響下での `precise` pragmaは？ → `MKLNonUniformConvolver.cpp` で `#pragma float_control(precise,on)` あり Good[x]
- `icx` での `const_cast` による `mkl_free` は安全か？ → `AlignedObjectDeleter` の `const_cast` はコメント通り必要だが、constオブジェクトのデストラクタ呼び出しはUBではないか要検証[x]

## 修正優先度ロードマップ

1. **P0 (今すぐ)**: C-1, C-2, C-3, H-5
2. **P1 (次回リリース前)**: H-1, H-2, H-3, C-4, M-1, M-2
3. **P2 (リファクタ)**: M-3~M-10, Low群, AVXチェックリストのRAIIラッパー化

必要なら、各バグについて該当コード断片を抜粋したパッチ diff を生成できます。


