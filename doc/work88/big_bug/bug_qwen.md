以下、前回レポートの指摘事項を、今回提供された `ConvoPeq.md` の該当箇所と突き合わせて再検証した結果です。

> 重要な前提  
> - 本再検証は、提供された **結合 Markdown の抜粋** をもとにしています。全翻訳単位の完全なソースに対して `ast-tree` / `clang-tidy` / コンパイルを実行した結果ではありません。  
> - したがって、以下の判定は「抜粋上で確認できた」「抜粋上では否定された」「全ソースがないと確定できない」という形で行います。  
> - `Practical Stable ISR Bridge Runtime.md` は提供内容上、実質空のため、ISR Bridge Runtime の仕様適合性は検証できていません。

---

# 1. 再検証方針

前回レポートでは、以下の観点でバグ候補を挙げていました。

1. AVX2 / CPU 機能チェック
2. リアルタイムスレッドの禁止操作
3. atomic / memory ordering
4. lifetime / dangling reference
5. CMake / ASan / icx / MSVC 差異
6. DSP 数値安全性
7. UI 状態同期
8. Evidence / JSON 出力
9. MKL / FFT / convolver 境界検査

今回提供された抜粋により、一部は **確度が上昇** し、一部は **撤回または下方修正** すべきと判断しました。

---

# 2. 再検証ステータス定義

| ステータス | 意味 |
|---|---|
| ✅ Confirmed | 抜粋上で問題の存在、または強い疑いを確認 |
| ⚠️ Partial | 一部は確認できたが、全ソースがないと確定不可 |
| ❌ Rejected | 抜粋上では問題なし、または前回指摘が誤り |
| ❓ NeedsFullCode | 該当箇所が今回抜粋に含まれず判定不能 |

---

# 3. 前回 Critical / High 指摘の再検証

## C1. AVX2 ランタイムチェックが不十分かもしれない

### 前回判定
Critical / High

### 再検証結果
⚠️ Partial / 重大度：**High に修正**

### 確認できた内容

今回抜粋に、CPUID を使った AVX2 チェックが含まれていました。

```cpp
// Step 0: CPUID(0) で最大 leaf を確認
int leaf0[4] = { 0 };
#if defined(_MSC_VER)|| defined(__INTEL_COMPILER)
__cpuid(leaf0, 0);
#elif defined(__GNUC__)|| defined(__clang__)
__get_cpuid(0, &leaf0[0], &leaf0[1], &leaf0[2], &leaf0[3]);
#endif
if (static_cast<unsigned>(leaf0[0]) < 7u)
    return false; // leaf 7 未対応 → AVX2 不可

// Step 1: leaf 1 で OSXSAVE + AVX + FMA を確認
int leaf1[4] = { 0 };
```

これにより、**少なくとも CPUID によるランタイムチェックは存在する** と判断できます。

したがって、前回指摘の「ランタイムチェックがない可能性」は撤回します。

### ただし残るリスク

CMake 側で AVX2 を強く有効化しています。

```cmake
# /QxCORE-AVX2: Haswell以降のAVX2+FMA必須、Intel専用コード生成
```

また、MSVC 側でも `/arch:AVX2` を使っている可能性が高いです。

この場合、`checkAVX2SupportAndWarn()` より前に、CRT 初期化、静的コンストラクタ、JUCE 初期化、または他の翻訳単位で AVX2 命令が実行されると、AVX2 非対応 CPU ではチェック前にクラッシュします。

### 再検証結論

- CPUID チェック自体は存在する。
- だが、**チェック前に AVX2 命令が実行されない保証** が必要。
- 重大度は Critical ではなく **High**。

### 推奨修正

- `CpuFeatureCheck.cpp` と起動エントリは AVX2 非依存でコンパイルする。
- DSP 本体のみ `/arch:AVX2` または `/QxCORE-AVX2` を付ける。
- 可能なら、CPU 判定用ランチャーを分離する。

---

## C2. `MessageBoxW` 文字列破損 / 構文エラー疑い

### 前回判定
Critical

### 再検証結果
❓ NeedsFullCode / 重大度：**Low に下方修正**

### 理由

前回指摘した以下の壊れた断片は、今回提供された抜粋には含まれていません。

```cpp
} //。",
L"ConvoPeq - CPU 非対応",
MB_OK| MB_ICONERROR);
```

そのため、前回レポートのこの項目は、**Markdown 結合時の破損だった可能性**が高いです。

### 再検証結論

- 今回抜粋だけでは実コードの構文エラーとは断定できない。
- 実際の `CpuFeatureCheck.cpp` を直接確認すべき。
- もし実ファイルに存在すれば Critical。
- 存在しなければ撤回。

### 確認コマンド例

```bat
rg -n "ConvoPeq - CPU 非対応" src
rg -n "MB_OK\| MB_ICONERROR" src
rg -n "//。" src
```

---

## C3. `m_pendingIRChange` を publication 成功前にクリアしている疑い

### 前回判定
Critical / High

### 再検証結果
❓ NeedsFullCode / 重大度：**High 維持**

### 理由

前回抜粋には以下がありました。

```cpp
const bool promoteToStructural =
    convo::exchangeAtomic(m_pendingIRChange, false, std::memory_order_acq_rel);
```

今回抜粋では、この周辺コードが直接含まれていません。

したがって、撤回はできません。

### リスク

もし publication / validation / closure check が失敗した後に pending flag が既に false になっていると、IR 変更要求が消失します。

### 推奨設計

pending flag は **publication commit 成功後** にクリアすべきです。

```cpp
bool pending = convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire);

if (!pending)
    return;

auto result = publishRuntime(...);

if (result.succeeded())
{
    convo::publishAtomic(m_pendingIRChange, false, std::memory_order_release);
}
else
{
    // pending を維持して再試行
}
```

### 再検証結論

- 今回抜粋では判定不能。
- だが設計リスクは高いため、**High 維持**。

---

## C4. Intel icx の AddressSanitizer 設定が不完全

### 前回判定
Critical / High

### 再検証結果
⚠️ Partial / 重大度：**Medium〜High**

### 確認できた内容

今回抜粋に以下があります。

```cmake
# 注意: /MTd は /fsanitize=address と非互換のため、Debugタスクから ASan を除去すること
set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
"MultiThreaded$<$<CONFIG:Debug>:Debug>")
```

また、IntelLLVM 側では以下が見えます。

```cmake
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
target_compile_options(ConvoPeq PRIVATE
-Wall -Wextra
-Wno-unused-parameter
/EHsc
/utf-8
-Wno-unknown-argument
)
```

さらに Release では以下があります。

```cmake
# icx Release: CMake規定の-MDを/MTで上書き（静的CRT+静的MKLリンク）
target_compile_options(ConvoPeq PRIVATE
$<$<CONFIG:Release>:/MT>
)
```

### 問題点

コメントに「/MTd は /fsanitize=address と非互換」とあるため、ASan を Debug から除去している可能性があります。

もし ASan を有効化する構成が残っているなら、以下を確認する必要があります。

- icx で `-fsanitize=address` が link option にも入っているか
- CRT が ASan と整合しているか
- Debug 限定になっているか
- Release に混入しないか

### 再検証結論

- 前回指摘の「icx ASan 設定が不完全」は、完全には否定できない。
- ただし、今回抜粋のコメントから、**ASan を Debug から除去している可能性**もある。
- 実 CMake 全体を確認する必要がある。

---

## C5. `ENABLE_ASAN` が Debug 限定になっていない

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium**

### 理由

今回抜粋には `ENABLE_ASAN` の全体定義が含まれていません。

ただし、以下のコメントから、ASan と CRT の非互換性を認識していることは確認できます。

```cmake
# 注意: /MTd は /fsanitize=address と非互換のため、Debugタスクから ASan を除去すること
```

### 再検証結論

- 実 CMake で `ENABLE_ASAN` が `$<$<CONFIG:Debug>:...>` でガードされているか確認が必要。
- もし全構成で有効なら Medium〜High。
- 今回抜粋だけでは確定不可。

---

## C6. `AudioEngineProcessor` / UI コンポーネントの参照寿命

### 前回判定
Critical / High

### 再検証結果
⚠️ Partial / 重大度：**High 維持**

### 確認できた内容

今回抜粋で、複数のクラスが参照を保持していることを確認しました。

```cpp
class MixedPhaseOptimizationComponent
{
    ConvolverProcessor& processor;
};
```

```cpp
explicit IRAdvancedSettingsComponent(AudioEngine& audioEngine)
    : engine(audioEngine)
```

```cpp
explicit AudioCallbackRuntimeScope(AudioEngine& owner) noexcept
    : engine(owner)
```

### リスク

参照保持自体は必ずしもバグではありません。  
しかし、以下のような破棄順序になると dangling reference になります。

1. `AudioEngine` 破棄
2. その後に UI コンポーネントや processor が生存
3. timer / callback / paint で参照使用

### 再検証結論

- 参照保持を複数確認。
- lifetime 設計は要監査。
- **High 維持**。

### 推奨

- `AudioEngine` を owner とし、参照先より先に破棄しない。
- UI は `juce::Component::SafePointer` を使っている箇所もあるが、全箇所を精査すべき。
- Timer / AsyncUpdater / ChangeListener の remove / cancel を破棄時に行う。

---

## C7. `ConvolverProcessor::ref()` が Release で null deref する可能性

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**High 維持**

### 理由

今回抜粋に `ref()` 本体は含まれていません。

ただし、`ConvolverProcessor& processor;` のような参照保持があるため、初期化順序ミスによる null / dangling 风险は残ります。

### 再検証結論

- 抜粋では判定不能。
- だが、Release で `jassert` が消えるなら防御コードを入れるべき。
- **High 維持**。

---

## C8. `setConvHCFilterMode()` が atomic 公開と再構築の順序で競合する可能性

### 前回判定
High

### 再検証結果
⚠️ Partial / 重大度：**Medium〜High**

### 確認できた内容

UI から直接以下を呼んでいます。

```cpp
engine.setConvLCFilterMode(convo::LCMode::Soft);
```

また、state 復元では複数の atomic を個別に公開しています。

```cpp
convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
```

```cpp
convo::publishAtomic(eqBypassRequested, bypassed, std::memory_order_release);
convo::publishAtomic(m_currentEqBypass, bypassed, std::memory_order_release);
```

### リスク

複数の状態を別々の atomic で公開すると、audio thread 側で以下のような中間状態が見える可能性があります。

- order は新しいが bypass は古い
- trim dB は新しいが gain は古い
- eq bypass は新しいが convolver bypass は古い

### 再検証結論

- 単一 atomic だけでは状態の一貫性が保証されない箇所がある。
- **Medium〜High**。
- 可能なら snapshot / generation 単位で公開すべき。

---

## C9. `juce::Logger::setCurrentLogger(fileLogger.release())` の所有権

### 前回判定
High / Medium

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium 維持**

### 理由

今回抜粋に当該箇所は含まれていません。

### 再検証結論

- JUCE 8.0.12 の `Logger::setCurrentLogger` の所有権仕様を確認すべき。
- もし所有権を取らないなら、`release()` はリーク。
- **Medium 維持**。

---

## C10. `MTNUPCMeasurement` が MKL を使うのに MKL リンク不足の疑い

### 前回判定
High

### 再検証結果
⚠️ Partial / 重大度：**Medium**

### 確認できた内容

CMake 抜粋で、MKL をリンクしているテストが一部だけ確認できました。

```cmake
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_link_libraries(RuntimePublicationCoordinatorTests PRIVATE MKL::MKL)
```

一方、`MTNUPCMeasurement` のリンク設定は今回抜粋に含まれていません。

### 再検証結論

- `MKLNonUniformConvolver.cpp` を使うターゲットが MKL をリンクしているか要確認。
- icx では `/Qmkl:sequential` 相当が必要かもしれない。
- **Medium**。

---

# 4. 前回 High / Medium 指摘の再検証

## H1. `diagLog()` がリアルタイムスレッドで呼ばれると音声ドロップする

### 前回判定
High

### 再検証結果
✅ Partial / 重大度：**Medium**

### 確認できた内容

MMCSS 層で、診断ログが audio thread 側で呼ばれる可能性を確認しました。

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
// ★ [diagnostic] MMCSS already managed by JUCE/driver — expected, not an error
diagLog("[MMCSS-" + juce::String(policyTag) + "] already registered by JUCE/driver (err="
    + juce::String(static_cast<int>(err)) + ") task="
    + juce::String(primaryTask));
#endif
```

コメントには以下があります。

```cpp
// RT-safety:
// - thread_local ensures no lock contention across driver-owned threads (ASIO)
// - Registration attempted ONCE (t_mmcssTried) → minimal RT impact
// - Logging is guarded by #if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
```

### 評価

- 通常 Release では診断ログが無効化されている可能性が高い。
- だが `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 有効時、audio thread 初回 MMCSS 登録時に `juce::String` 生成 + `Logger` 出力が発生する。
- 初回のみでも 50〜200μs 以上の jitter / dropout 要因になる。

### 再検証結論

- リアルタイムパスでの文字列生成・ログ出力は、診断有効時に限り残存。
- 重大度は **Medium**。
- Debug / CI で問題化しやすい。

### 推奨修正

RT thread ではログ文字列を作らず、lock-free diag buffer に numeric event だけpushすべきです。

実際、別箇所では以下のように numeric-only DiagEvent を push しています。

```cpp
DiagEvent event{};
event.category = DiagCategory::CallbackSequence;
...
if (diagBuffer.push(event))
{
    rtAuxMutable_.diagTickPushed.value.fetch_add(1, std::memory_order_relaxed);
}
```

この設計に統一するのが望ましいです。

---

## H2. `onHealthEvent()` が文字列生成している

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium 維持**

### 理由

今回抜粋に `onHealthEvent()` 本体は含まれていません。

### 再検証結論

- health event がどのスレッドで発火するか要確認。
- audio thread なら High。
- 非 RT thread なら Low。
- 現時点では **Medium 維持**。

---

## H3. `currentSpec` が非 atomic で複数スレッドから触られる可能性

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium**

### 理由

今回抜粋に `currentSpec` は含まれていません。

### 再検証結論

- `prepare()` が message thread / audio thread 停止中に呼ばれる契約なら問題なし。
- そうでなければ data race。
- JUCE の `prepareToPlay` は通常メッセージスレッド側なので、設計上は許容可能。
- ただし契約の明文化と assert が必要。

---

## H4. `lastError` と `lastErrorMutex_` の利用が不透明

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium**

### 理由

今回抜粋に該当コードなし。

### 再検証結論

- `lastError` が UI 専用なら問題なし。
- audio thread から触るならリアルタイム規約違反。
- **Medium 維持**。

---

## H5. `ListenerList` の thread-safety 疑い

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium**

### 理由

今回抜粋に `ListenerList` 本体は含まれていません。

ただし、UI 側で `SafePointer` や `callAsync` を使っている箇所は確認できました。

```cpp
juce::Component::SafePointer<ConvolverControlPanel> safeThis(this);
juce::MessageManager::callAsync([safeThis]
{
    if (safeThis == nullptr)
        return;
```

これは良い設計です。

### 再検証結論

- 全 Listener が message thread 限定なら問題なし。
- audio thread から callback しているなら High。
- 現時点では **Medium**。

---

## H6. `dryBufferStorage` / `delayBuffer` の所有権と再確保が危険

### 前回判定
High

### 再検証結果
⚠️ Partial / 重大度：**Medium**

### 確認できた内容

dry bypass buffer について、容量チェック付きコピーを確認しました。

```cpp
if (dryBypassBufferDoubleL && dryBypassBufferDoubleR && dryBypassCapacityDouble >= numSamples)
{
    juce::FloatVectorOperations::copy(dryBypassBufferDoubleL.get(), alignedL.get(), numSamples);
    juce::FloatVectorOperations::copy(dryBypassBufferDoubleR.get(), alignedR.get(), numSamples);
}
```

### 評価

- 容量チェックがあるのは良い。
- だが、容量不足時に何もしない場合、後段で古い dry buffer や未初期化 buffer を使うなら問題。

### 再検証結論

- 二重解放の直接証拠はなし。
- 容量不足時の fallback が不明。
- **Medium**。

---

## H7. `loadImpulseResponse()` / IR resample の検証不足

### 前回判定
High

### 再検証結果
✅ Partial / 重大度：**High 維持**

### 確認できた内容

IR resample には一定の防御があります。

```cpp
int maxOutLen = tempResampler.getMaxOutLen(inLength);
if (maxOutLen <= 0)
{
    juce::Logger::writeToLog("[DIAG_IR] resampleIR: maxOutLen<=0 ("
        + juce::String(maxOutLen) + ") inLen=" + juce::String(inLength)
        + " srIn=" + juce::String(inputSR, 1) + " srOut=" + juce::String(targetSR, 1));
    return {};
}
```

```cpp
std::vector<int> channelDone(numCh, -1); // -1初期化: 例外・未完了を識別
```

```cpp
const int maxDone = *std::max_element(channelDone.begin(), channelDone.end());
if (maxDone < 0)
{
    juce::Logger::writeToLog("[DIAG_IR] resampleIR: all channels failed (maxDone="
        + juce::String(maxDone) + " numCh=" + juce::String(numCh) + ")");
    return {};
}
```

### 評価

良い点：

- `maxOutLen <= 0` チェックあり。
- channel 失敗を `-1` で識別。
- 全チャンネル失敗を検出。

残る問題：

- `numCh` の上限 / 下限チェックが抜粋に見えない。
- `inLength` の上限チェックが見えない。
- `maxOutLen` が巨大な場合の OOM 対策が見えない。
- resample 中に別 IR 要求が来てもキャンセルできるか不明。
- `done == maxOutLen` 時の loop 停止が不明。

### 再検証結論

- 前回より防御があることは確認できた。
- だが IR ロードは依然として High risk。
- **High 維持**。

---

## H8. EQ/SVF 係数計算で NaN/Inf が生じる可能性

### 前回判定
High

### 再検証結果
⚠️ Partial / 重大度：**Medium**

### 確認できた内容

EQ 側に clamp / validate 関数があることを確認しました。

```cpp
static void validateAndClampParameters(float& freq, float& gainDb, float& q, double sr) noexcept;
```

また、EQ 範囲定義もあります。

```cpp
static constexpr float Q_MIN = 0.01f;
static constexpr float Q_MAX = 20.0f;
static constexpr float MIN_BAND_GAIN = -15.0f;
static constexpr float MAX_BAND_GAIN = 15.0f;
static constexpr float MIN_TOTAL_GAIN = -24.0f;
static constexpr float MAX_TOTAL_GAIN = 24.0f;
```

### 評価

- パラメータ clamp は存在する。
- したがって、前回よりリスクは低い。
- だが、clamp 後の係数計算で NaN/Inf が出ないか、最終 finite check が必要。

### 再検証結論

- **Medium** に下方修正。
- 係数生成後の `std::isfinite` 検証があるか要確認。

---

## H9. AGC / gain calculation のゼロ除算疑い

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium**

### 理由

今回抜粋に AGC 本体なし。

### 再検証結論

- `calculateAGCGain()` で epsilon が入っているか要確認。
- **Medium 維持**。

---

## H10. NoiseShaper / PsychoacousticDither の学習係数が不安定になる可能性

### 前回判定
High

### 再検証結果
⚠️ Partial / 重大度：**Medium**

### 確認できた内容

`LatticeNoiseShaper` に状態 clamp があることを確認しました。

```cpp
const __m256d limit = _mm256_set1_pd(kStateLimit);
const __m256d negLimit = _mm256_set1_pd(-kStateLimit);

__m256d v0 = _mm256_loadu_pd(state);
v0 = _mm256_min_pd(v0, limit);
v0 = _mm256_max_pd(v0, negLimit);
_mm256_storeu_pd(state, v0);
```

また、格子フィルタの再帰式について、過去のバグ修正コメントがあります。

```cpp
// [P7] 修正: 前段の prev_backward ではなく、自段の nextBackward を保存
// 旧コード: state[i] = std::clamp(prev_backward, ...);
// 格子フィルタの正しい再帰式では、各段で計算した後方反射波 g_{i+1}(n)
// を次サンプル用に保存する必要がある。
state[i] = std::clamp(nextBackward, -kLatticeStateLimit, kLatticeStateLimit);
```

### 評価

- 状態 clamp あり。
- 過去バグ修正痕跡あり。
- 完全に放置されているわけではない。

### 残リスク

- 学習係数そのものの安定性判定があるか不明。
- 学習結果適用前に peak gain / stability check があるか不明。

### 再検証結論

- **Medium** に下方修正。
- 学習係数の安定性検証は要確認。

---

## H11. UI の processing order / bypass 状態が過渡的に不整合になる可能性

### 前回判定
High / Medium

### 再検証結果
✅ Partial / 重大度：**Medium**

### 確認できた内容

state 復元で複数の atomic を個別公開しています。

```cpp
convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
```

```cpp
convo::publishAtomic(eqBypassRequested, bypassed, std::memory_order_release);
convo::publishAtomic(m_currentEqBypass, bypassed, std::memory_order_release);
```

```cpp
convo::publishAtomic(convolverInputTrimDb, clampedDb, std::memory_order_release);
convo::publishAtomic(convolverInputTrimGain, juce::Decibels::decibelsToGain((double)clampedDb), std::memory_order_release);
convo::publishAtomic(m_currentConvInputTrimDb, clampedDb, std::memory_order_release);
```

### 評価

- 個別 atomic 公開は、コーディング規約の「小さなデータは atomic」には合う。
- だが、複数値が一致している必要がある場合、snapshot 化が望ましい。

### 再検証結論

- 過渡的不整合の可能性がある。
- **Medium**。

---

## H12. `orderModeBox` の状態復元が曖昧

### 前回判定
Medium

### 再検証結果
❌ Rejected / 重大度：**Low / 撤回**

### 確認できた内容

今回抜粋で、両方 bypass の場合を明示的に処理していました。

```cpp
const bool eqBypassed = audioEngine.isEqBypassRequested();
const bool convBypassed = audioEngine.isConvolverBypassRequested();

int modeId;

if (eqBypassed && convBypassed)
    modeId = 5; // Bypass
else if (!eqBypassed && convBypassed)
    modeId = 2; // Peq
else if (eqBypassed && !convBypassed)
    modeId = 1; // Conv
else if (!eqBypassed && !convBypassed
    && audioEngine.getProcessingOrder() ==
```

### 評価

前回レポートでは「両方 bypass のとき Conv->Peq になる可能性」を指摘しましたが、今回抜粋では `modeId = 5` になっています。

### 再検証結論

- 前回指摘は誤りの可能性が高い。
- **撤回**。

---

## H13. Evidence 出力先ディレクトリが作成されていない可能性

### 前回判定
High / Medium

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium 維持**

### 確認できた内容

Evidence 出力は確認できました。

```cpp
writeTextFile(root / "evidence_manifest.json", manifest);
```

```cpp
writeTextFile(root / "recovery_trace.json", oss.str());
```

だが、`create_directories(root)` が抜粋に見えません。

### 再検証結論

- `writeTextFile()` 内部で directory 作成している可能性あり。
- していなければ evidence 消失。
- **Medium 維持**。

---

## H14. JSON 手組み立てが壊れやすい

### 前回判定
High / Medium

### 再検証結果
✅ Partial / 重大度：**Medium**

### 確認できた内容

manifest を手動生成しています。

```cpp
if (!first) {
    manifest += ",\n";
}
manifest += " \"";
manifest += name;
manifest += "\"";
first = false;
```

### 問題点

- `name` が固定文字列なら問題ない。
- だが、ファイル名や runId などをエスケープせず入れると JSON が壊れる。

### 再検証結論

- 手動 JSON は依然としてリスクあり。
- **Medium**。

---

## H15. `escapeJson()` が制御文字/Unicode を十分エスケープしているか要確認

### 前回判定
Medium

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium 維持**

### 理由

`escapeJson()` 本体が抜粋にない。

### 再検証結論

- RFC 8259 準拠か要確認。
- **Medium 維持**。

---

## H16. `RuntimePublicationOrchestrator` で `diagLog` が Release で未定義になる可能性

### 前回判定
Medium

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium 維持**

### 理由

今回抜粋では該当箇所なし。

### 再検証結論

- 全ファイルで共通 no-op `diagLog` を使うべき。
- **Medium 維持**。

---

## H17. `#pragma warning(push)` に対応する `pop` がない疑い

### 前回判定
Medium / Low

### 再検証結果
⚠️ Partial / 重大度：**Low**

### 確認できた内容

以下を確認。

```cpp
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
```

```cpp
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
```

抜粋内には `pop` が見えません。

### 再検証結論

- ファイル末尾に pop がある可能性あり。
- 抜粋だけでは確定不可。
- **Low**。

---

## H18. Ninja + icx 用の RC 回避条件が誤っている可能性

### 前回判定
High / Medium

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium 維持**

### 理由

今回抜粋に当該条件が含まれていません。

### 再検証結論

- 実 CMake で `NOT MSVC` 条件になっているか要確認。
- IntelLLVM で `MSVC` が真になるなら条件ミス。
- **Medium 維持**。

---

## H19. `CMAKE_CXX_FLAGS_RELEASE` を全局的に上書きしている

### 前回判定
High

### 再検証結果
❓ NeedsFullCode / 重大度：**Medium**

### 理由

今回抜粋では、グローバル上書きではなくターゲット固有 `target_compile_options` が見えます。

```cmake
target_compile_options(ConvoPeq PRIVATE
    $<$<CONFIG:Release>:/MT>
)
```

### 再検証結論

- 前回指摘の全局上書きが実際にあるか要確認。
- もしターゲット固有だけなら問題なし。
- **Medium** に下方修正。

---

## H20. `/fp:fast` はプロオーディオではリスクがある

### 前回判定
High

### 再検証結果
✅ Partial / 重大度：**Medium**

### 確認できた内容

icx Release で `/fp:fast` を使用しています。

```cmake
# /fp:fast: 高速浮動小数点（icx デフォルト）。明示的なデノーマル対策により
# 微小信号の消失リスクを排除しつつ、最高の演算性能を実現。
```

一方、DSP コアでは以下があります。

```cpp
// D2: /fp:fast の影響を回避するため、DSP コアファイルで float_control を precise に指定
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#endif
```

### 評価

- DSP コア側で `/fp:fast` 影響を緩和しようとしている。
- だが、MSVC 用の `float_control` が見えるため、icx で同等効果があるか要確認。
- また、DSP コア以外で数値計算がある場合、`/fp:fast` の影響が残る。

### 再検証結論

- リスクは残るが、緩和策あり。
- **Medium**。

---

## H21. `/QxCORE-AVX2` は AMD CPU で性能低下または互換性リスク

### 前回判定
Medium

### 再検証結果
✅ Confirmed / 重大度：**Medium**

### 確認できた内容

```cmake
# /QxCORE-AVX2: Haswell以降のAVX2+FMA必須、Intel専用コード生成
```

### 評価

- 仕様として Intel 専用コード生成を意図している。
- AMD64 対応を謳うなら、性能/互換性リスクがある。

### 再検証結論

- **Medium**。
- 製品要件として「Intel CPU 推奨」を明記するか、portable AVX2 ビルドを用意すべき。

---

# 5. 今回抜粋で新たに確認されたバグ候補

以下は、前回レポートには含めていなかったが、今回抜粋で確認された問題です。

---

## N1. Crossfade 箇所に明らかな構文エラー疑い

### 箇所

```cpp
const double dryScaledL =const double dryScaledL = alignedOldL * dryScale;
```

### 問題

これは C++ として構文エラーです。

もし実ソースに存在するなら、コンパイルが通りません。

### 重大度

**Critical**  
ただし、Markdown 結合時の破損の可能性もあります。

### 確認コマンド

```bat
rg -n "dryScaledL =const" src
rg -n "const double dryScaledL =const double dryScaledL" src
```

### 修正

正しくはおそらく以下です。

```cpp
const double dryScaledL = alignedOldL * dryScale;
```

---

## N2. r8brain resample loop で `done == maxOutLen` 時に停止しない可能性

### 箇所

```cpp
double* r8bOutput = nullptr;
const int generated = resampler->process(nullptr, 0, r8bOutput);
if (generated <= 0) break;

const int toCopy = std::min(generated, maxOutLen - done);
std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
done += toCopy;
```

### 問題

`done == maxOutLen` のとき、

```cpp
maxOutLen - done == 0
```

となり、`toCopy == 0` になります。

このとき `done` は増えません。  
もし `generated > 0` が続くなら、同じ状態を繰り返す可能性があります。

### 重大度

**High**

### 推奨修正

```cpp
if (done >= maxOutLen)
    break;

const int remaining = maxOutLen - done;
if (remaining <= 0)
    break;

double* r8bOutput = nullptr;
const int generated = resampler->process(nullptr, 0, r8bOutput);
if (generated <= 0)
    break;

const int toCopy = std::min(generated, remaining);
std::memcpy(outPtr + done, r8bOutput, static_cast<size_t>(toCopy) * sizeof(double));
done += toCopy;

if (done >= maxOutLen)
    break;
```

また、`generated > remaining` で切り捨てるなら、ログを出すべきです。

---

## N3. IR resample がキャンセル要求を内部で確認していない可能性

### 箇所

```cpp
const int requestId = convo::fetchAddAtomic(irPreviewRequestId, 1, std::memory_order_acq_rel) + 1;
```

```cpp
for (int ch = 0; ch < numCh; ++ch) {
    futures.emplace_back(std::async(std::launch::async, [&, ch]() {
        const convo::cpu::ScopedMXCSR mxcsr;
        try {
            auto resampler = std::make_unique<r8b::CDSPResampler>(
```

### 問題

`requestId` を発行していますが、抜粋を見る限り、ワーカー内で現在の requestId と比較して中断していません。

### 影響

- ユーザーが別 IR を選択しても、古い resample が最後まで走る。
- UI 応答性低下。
- CPU 負荷増加。

### 重大度

**Medium**

### 推奨修正

```cpp
const int myRequestId = requestId;

futures.emplace_back(std::async(std::launch::async, [&, ch, myRequestId]() {
    for (...)
    {
        if (convo::consumeAtomic(irPreviewRequestId, std::memory_order_acquire) != myRequestId)
        {
            channelDone[ch] = -1;
            return;
        }
        ...
    }
}));
```

---

## N4. 時間差計算で uint64 underflow の可能性

### 箇所

```cpp
const uint64_t observeLatencyUs = observeUs - matchedPublishEndUs;
```

条件は以下のみ。

```cpp
if (matchedPublishEndUs > 0)
```

### 問題

`matchedPublishEndUs > observeUs` の場合、uint64 が wrap します。

### 重大度

**Medium**

### 推奨修正

```cpp
const uint64_t observeLatencyUs =
    (observeUs >= matchedPublishEndUs)
        ? (observeUs - matchedPublishEndUs)
        : 0;
```

同様に以下も要修正。

```cpp
const auto callbackUs = static_cast<uint32_t>(nowUs - cbStartUs);
const auto intervalUs = static_cast<uint32_t>(cbStartUs - cbPrevEndUs);
```

推奨：

```cpp
const uint64_t callbackUs64 =
    (nowUs >= cbStartUs) ? (nowUs - cbStartUs) : 0;

const uint32_t callbackUs =
    static_cast<uint32_t>(std::min<uint64_t>(callbackUs64, UINT32_MAX));
```

---

## N5. `NoiseShaperType` などの enum cast が検証なし

### 箇所

```cpp
if (state.hasProperty("noiseShaperType"))
    setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"));
```

### 問題

state ファイルが壊れていると、範囲外の enum 値を cast して渡す可能性があります。

### 重大度

**Medium**

### 推奨修正

```cpp
if (state.hasProperty("noiseShaperType"))
{
    const int value = static_cast<int>(state.getProperty("noiseShaperType"));

    if (value >= static_cast<int>(NoiseShaperType::Psychoacoustic)
        && value <= static_cast<int>(NoiseShaperType::Fixed15Tap))
    {
        setNoiseShaperType(static_cast<NoiseShaperType>(value));
    }
}
```

同様に、`oversamplingType`, `convHCFilterMode`, `convLCFilterMode` も検証すべきです。

---

## N6. CMA-ES クラスの `mean` / `covariance` の型が矛盾している可能性

### 箇所 1

```cpp
double* mean = nullptr;
double* covariance = nullptr;
double sigma = 0.12;
```

### 箇所 2

```cpp
std::copy(inMean, inMean + dim, mean.begin());
```

### 問題

もし同じクラスのメンバが `double* mean` なら、`mean.begin()` はコンパイルエラーです。

### 可能性

- Markdown 結合で別クラスの断片が混ざった。
- 実際には `std::vector<double> mean;` である。
- 実コードに型不一致バグがある。

### 重大度

**High**  
ただし結合ミス可能性あり。

### 確認コマンド

```bat
rg -n "double\* mean" src
rg -n "mean.begin\(\)" src
rg -n "class CmaEsOptimizerDynamic" src -A 80
```

### 推奨

- raw pointer 所有なら Rule of Five を徹底。
- 可能なら `std::vector<double>` にする。

---

## N7. Analyzer / oversampling の内部バッファ上限チェックが不完全な可能性

### 箇所

```cpp
if (numSamples > maxSamplesPerBlock)
{
    buffer.clear();
    return;
}

if (oversamplingFactor > 1)
{
    const int expectedUpSize = numSamples * static_cast<int>(oversamplingFactor);
    if (expectedUpSize > maxInternalBlockSize)
    {
```

### 問題

抜粋が切れていますが、`expectedUpSize > maxInternalBlockSize` のときに安全に return / clear しているか要確認です。

もしそのまま `oversampling.processUp()` に進むと、内部バッファを超過する可能性があります。

### 重大度

**High**

### 推奨

```cpp
if (expectedUpSize > maxInternalBlockSize)
{
    buffer.clear();
    return;
}
```

---

## N8. dry bypass buffer 容量不足時の fallback が不明

### 箇所

```cpp
if (dryBypassBufferDoubleL && dryBypassBufferDoubleR && dryBypassCapacityDouble >= numSamples)
{
    juce::FloatVectorOperations::copy(dryBypassBufferDoubleL.get(), alignedL.get(), numSamples);
    juce::FloatVectorOperations::copy(dryBypassBufferDoubleR.get(), alignedR.get(), numSamples);
}
```

### 問題

容量不足時にコピーしません。  
その後の dry/wet mix で dry bypass buffer を使うなら、古いデータや未初期化データを使う可能性があります。

### 重大度

**Medium**

### 推奨

容量不足時は以下のように安全側にする。

```cpp
else
{
    // dry path を無効化するか、clear して安全な fallback にする
}
```

---

## N9. `audioCallbackActiveCount` が uint32 でオーバーフローする可能性

### 箇所

```cpp
(void)convo::fetchAddAtomic(engine.rtLocalState_.audioCallbackActiveCount, uint32_t{1}, std::memory_order_acq_rel);
```

### 問題

active count が uint32 の場合、長時間稼働で wrap する可能性があります。

例：1000 callback/sec でも約49日で wrap。

### 重大度

**Low / Medium**

### 推奨

- uint64_t にする。
- または active count と total callback count を分離する。

---

## N10. MMCSS 初回登録が audio thread で 50〜200μs の jitter を起こす可能性

### 箇所

```cpp
// RT impact: first call only (~50-200μs).
static_cast<void>(tryApplyMmcssForSelfManagedThread());
```

### 問題

初回のみとはいえ、audio callback 内で WinAPI を呼ぶと dropout 要因になります。

### 重大度

**Low / Medium**

### 推奨

- 可能なら device start 前の非 RT thread で登録する。
- ドライバ管理の ASIO では難しい場合もあるため、初回のみなら許容する設計判断もあり。
- ただし診断ログは RT で出力しない。

---

# 6. 今回抜粋で確認できた良い設計・バグ緩和策

再検証の結果、前回レポートより評価を上げるべき点もあります。

---

## G1. CPUID チェックが実装されている

```cpp
__cpuid(leaf0, 0);
```

```cpp
if (static_cast<unsigned>(leaf0[0]) < 7u)
    return false;
```

AVX2 必須アプリとして正しい方向性です。

---

## G2. UI の両方 Bypass 状態が扱われている

```cpp
if (eqBypassed && convBypassed)
    modeId = 5; // Bypass
```

前回指摘の UI mode 曖昧さは、少なくともこの抜粋では問題ありません。

---

## G3. DSP コアで `/fp:fast` 影響を避けようとしている

```cpp
#pragma float_control(precise, on)
```

MSVC では有効です。  
icx での同等性だけ要確認です。

---

## G4. Denormal / NaN 対策が入っている

```cpp
const __m256d vThresh = _mm256_set1_pd(kDenormThreshold);
```

```cpp
if (fastAbs(acc) < kDenormThreshold)
```

```cpp
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
```

リアルタイム DSP として重要です。

---

## G5. Convolver AVX2 境界チェックに +6 マージンを入れている

```cpp
// loadStride2 が ptr[-6] までアクセスするため、historyDownKeep に +6 マージンを追加
// ref: doc/work46/bug.md (Bug #1)
```

```cpp
static constexpr int kLoadStride2Offset = 6;
const int avxMinConvIdx = globalMinConvIdx - kLoadStride2Offset;
const bool convTapOk = (avxMinConvIdx >= 0) && (globalMaxConvIdx < capacity);
```

境界検査を意識した実装になっています。

---

## G6. IR resample ワーカーで MXCSR を保存/復元している

```cpp
// ★ Bug#4: std::async ワーカー — ThreadPool 実装依存のため RAII で保存＋復元
const convo::cpu::ScopedMXCSR mxcsr;
```

良い設計です。

---

## G7. RCU / EBR / lifecycle scope を意識している

```cpp
explicit AudioCallbackRuntimeScope(AudioEngine& owner) noexcept
    : engine(owner)
    , lifecycleToken(owner.lifecycleRuntime_.enterAudioCallback())
    , firewallToken(owner.rtCapabilityFirewall_.enter())
{
    convo::isr::RTAllocatorFirewall::markRTContext(true);
```

RT 保護設計が入っています。

---

# 7. 再検証後の優先修正リスト

再検証後、優先すべきは以下です。

---

## 優先度 1：実コードの構文エラー確認

### 対象

```cpp
const double dryScaledL =const double dryScaledL = alignedOldL * dryScale;
```

### 対応

```bat
rg -n "dryScaledL =const" src
```

もし存在すれば即修正。

---

## 優先度 2：r8brain resample loop の停止条件

### 対象

```cpp
const int toCopy = std::min(generated, maxOutLen - done);
```

### 対応

`done >= maxOutLen` で必ず break する。

---

## 優先度 3：AVX2 起動前チェック

### 対象

- `CpuFeatureCheck.cpp`
- CMake の `/arch:AVX2` / `/QxCORE-AVX2`

### 対応

- 起動エントリと CPU 判定は AVX2 なしでコンパイル。
- DSP 本体のみ AVX2 有効化。

---

## 優先度 4：RT 診断ログの排除

### 対象

```cpp
diagLog("[MMCSS-" + juce::String(policyTag) + ...
```

### 対応

- RT では numeric event のみ ring buffer へ。
- 文字列化とログ出力は非 RT thread へ。

---

## 優先度 5：IR 変更 pending flag の消失防止

### 対象

```cpp
exchangeAtomic(m_pendingIRChange, false, ...)
```

### 対応

publication 成功後にクリアする。

---

## 優先度 6：複数 atomic の snapshot 化

### 対象

```cpp
convo::publishAtomic(convolverInputTrimDb, ...);
convo::publishAtomic(convolverInputTrimGain, ...);
convo::publishAtomic(m_currentConvInputTrimDb, ...);
```

### 対応

- audio thread が一貫して読む必要がある値は、構造体 snapshot + generation で公開。
- または「どの atomic を正とするか」を明確化。

---

## 優先度 7：enum / state restore の range validation

### 対象

```cpp
setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"));
```

### 対応

全 state property に range check を入れる。

---

## 優先度 8：時間差 underflow 対策

### 対象

```cpp
observeUs - matchedPublishEndUs
nowUs - cbStartUs
cbStartUs - cbPrevEndUs
```

### 対応

saturating subtraction にする。

---

## 優先度 9：CMA-ES メンバ型矛盾の確認

### 対象

```cpp
double* mean = nullptr;
std::copy(inMean, inMean + dim, mean.begin());
```

### 対応

実ファイルで型を確認。  
可能なら `std::vector<double>` 化する。

---

## 優先度 10：MKL thread / allocation 監査

### 対象

- `MKLNonUniformConvolver.cpp`
- `DSPCore*.cpp`
- `ConvolverProcessor.*`

### 確認項目

- audio thread 内で MKL allocation がないか
- MKL thread が 1 に固定されているか
- DFTI descriptor を process 内で commit/free していないか
- 64-byte alignment が保証されているか

---

# 8. 再検証結論

前回レポートと比較すると、以下のように修正すべきです。

## 撤回 / 下方修正すべき指摘

| 旧指摘 | 再検証結果 |
|---|---|
| MessageBox 文字列破損 | Markdown 破損の可能性が高い。実コード確認が必要。 |
| AVX2 ランタイムチェックがない | CPUID チェックは存在する。 |
| UI mode が両方 bypass で曖昧 | `modeId = 5` が確認できたため撤回。 |
| NoiseShaper 完全未防御 | state clamp と過去修正痕跡あり。重大度低下。 |
| EQ 係数完全未検証 | `validateAndClampParameters` あり。重大度低下。 |

## 依然として高リスクな指摘

| 指摘 | 再検証後重大度 |
|---|---|
| AVX2 起動前チェック順序 | High |
| `m_pendingIRChange` 早期クリア | High |
| AudioEngine / UI / Processor 寿命 | High |
| IR resample 停止条件 | High |
| RT 診断ログ | Medium |
| 複数 atomic の不整合 | Medium |
| CMake ASan 設定 | Medium |
| MKL RT safety | High audit |
| CMA-ES 型矛盾 | High / 要確認 |
| Crossfade 構文エラー疑い | Critical / 要確認 |

---

# 9. 最終評価

提供された抜粋を見る限り、ConvoPeq は以下のようなリアルタイム安全性機構をかなり意識しています。

- atomic publish / consume
- RCU / EBR 風 lifecycle
- RTAllocatorFirewall
- MXCSR 保存/復元
- denormal / NaN 対策
- AVX2 境界検査
- SafePointer による UI 非同期保護
- state validation

一方で、以下のような「複数状態の一貫性」「起動前 AVX2」「IR resample 停止条件」「RT 診断ログ」「lifetime」にリスクが残っています。

したがって、次回以降の検証では、以下の実ファイル全体を確認すべきです。

1. `src/CpuFeatureCheck.cpp`
2. `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
3. `src/audioengine/AudioEngine.Retire.cpp`
4. `src/audioengine/RuntimePublicationOrchestrator.cpp`
5. `src/audioengine/PublicationExecutor.cpp`
6. `src/audioengine/CrossfadeAuthority.cpp`
7. `src/ConvolverProcessor.cpp`
8. `src/IRResampler` 系
9. `CMakeLists.txt` 全体
10. MKL を使う DSP 実装全体

必要であれば、次はこれらを **GitHub Issue 形式** または **修正パッチ案付き** で、さらに詳細なバグリストに整理できます。