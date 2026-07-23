以下は、提供された `ConvoPeq.md` の抽出コード断片に基づく詳細な静的調査結果です。  
ただし、提示資料はソース全体ではなく **断片化・省略された Markdown 抽出** であり、行番号・完全な制御フロー・全実装が揃っているわけではありません。したがって以下は「確定バグ」だけでなく、**実コードで確認すべき高リスクの不具合候補**を含みます。

特に、ConvoPeq のようなプロ向けオーディオソフトウェアでは、Ross Bencina の "Real-time audio programming 101" にある通り、オーディオスレッド内で以下を行うことが重大バグになり得ます。

- メモリアロケーション
- ミューテックス / ロック
- ファイル I/O
- ログ出力
- 例外送出
- 巨大な計算 / システムコール

この観点から、優先度順に指摘します。

---

# 1. 高優先度：クラッシュ / 未定義動作 / オーディオ破綻の可能性

## H1. AVX2 ループで `s + i - 1` により前方越境読みの可能性

### 該当箇所

```cpp
const double* s = srcBuf + iRead;
#if defined(__AVX2__)
const __m256d vw0 = _mm256_set1_pd(w0);
...
for (; i <= samplesToRead - 4; i += 4)
{
    __m256d p0 = _mm256_loadu_pd(s + i - 1);
    __m256d p1 = _mm256_loadu_pd(s + i);
    __m256d p2 = _mm256_loadu_pd(s + i + 1);
```

### 問題

`i == 0` のとき、

```cpp
_mm256_loadu_pd(s + i - 1)
```

は `s - 1` を読みます。  
もし `srcBuf + iRead` の前に有効なサンプルが保証されていなければ、**配列前方の越境読み込み**です。

### 影響

- デバッグビルドでクラッシュ
- リリースビルドで不定値混入
- ノイズ / クラックル
- ASan で global-buffer-overflow / heap-buffer-overflow
- 特定のブロックサイズでのみ発生する再現性の低い不具合

### 重要度

**High / Critical 寄り**

### 確認方法

- `srcBuf` の前に最低 1 サンプル、できれば AVX2 用に 4 サンプル以上のガードがあるか確認
- `iRead` が常に 1 以上か確認
- `samplesToRead == 0, 1, 2, 3, 4` の境界テストを追加

### 修正方針

ベクトル化ループの最初だけスカラ処理するか、入力バッファ前に履歴サンプルを確保すべきです。

例：

```cpp
if (samplesToRead <= 0)
    return;

// 先頭はスカラで処理
processOneSample(s, 0);

for (int i = 1; i <= samplesToRead - 4; i += 4)
{
    __m256d p0 = _mm256_loadu_pd(s + i - 1);
    ...
}
```

または、バッファを次のように設計します。

```cpp
// 実際には buffer[4] から有効データが始まる
// buffer[0..3] は履歴 / ガード
```

---

## H2. `transferIRStateFrom()` で `acquireIRState()` 後に対応する `releaseIRState()` が見当たらない

### 該当箇所

```cpp
void transferIRStateFrom(const ConvolverProcessor& source) noexcept
{
    const IRState* srcState = source.acquireIRState();

    if (srcState && srcState->ir && srcState->ir->getNumSamples() > 0 && srcState->sampleRate > 0.0)
    {
        const int channels = srcState->ir->getNumChannels();
        const int
```

### 問題

Epoch-based RCU または acquire/release 型状態保護において、`acquireIRState()` した場合は必ず対応する `releaseIRState()` が必要です。  
抽出コードには release が見当たりません。

### 影響

もし release 漏れがある場合：

- 古い `IRState` が永久に解放されない
- Epoch が進まず retire キューが肥大化
- メモリリーク
- 状態更新が停滞
- 長時間稼働でメモリ増大
- 最悪、リソース枯渇

### 重要度

**High**

### 修正方針

RAII で release を保証すべきです。

```cpp
void transferIRStateFrom(const ConvolverProcessor& source)
{
    const IRState* srcState = source.acquireIRState();
    if (!srcState)
        return;

    struct Releaser
    {
        const ConvolverProcessor& owner;
        const IRState* state;

        ~Releaser()
        {
            if (state != nullptr)
                owner.releaseIRState(state);
        }
    } releaser { source, srcState };

    if (srcState->ir == nullptr)
        return;

    if (srcState->ir->getNumSamples() <= 0)
        return;

    if (srcState->sampleRate <= 0.0)
        return;

    // copy buffer...
}
```

---

## H3. `std::atomic<IRState*> currentIRState` の寿命管理が不十分なら Use-After-Free の可能性

### 該当箇所

```cpp
std::atomic<IRState*> currentIRState { nullptr };
...
[[nodiscard]] const IRState* acquireIRState() const noexcept;
void releaseIRState(const IRState* state) const noexcept;
void updateIRState(const juce::AudioBuffer<double>& newIR, double newSR, ...);
```

### 問題

`IRState*` を raw pointer で atomic 公開している場合、`updateIRState()` 側で旧状態を即座に `delete` すると、オーディオスレッドがまだ参照中の場合に **use-after-free** になります。

抽出コードから、旧 `IRState` をどのように retire しているかが確認できません。

### 想定される危険な実装

```cpp
auto* old = currentIRState.exchange(newState);
delete old; // 危険
```

### 影響

- オーディオスレッドでクラッシュ
- IR 切替時のみ発生するランダムクラッシュ
- ASan で heap-use-after-free
- 本番環境で再現困難な障害

### 重要度

**Critical / High**

### 修正方針

旧状態は必ず epoch / RCU / hazard pointer / deferred deletion queue を通して解放すべきです。

```cpp
auto* old = currentIRState.exchange(newState, std::memory_order_acq_rel);

if (old != nullptr)
{
    retireQueue.retire(old, [](IRState* p)
    {
        delete p->ir;
        delete p;
    });
}
```

---

## H4. `m_ready` を `true` にする前に、更新開始時点で `false` にしているか不明

### 該当箇所

```cpp
vdMul(l.complexSize, re, gainReal.get(), re);
vdMul(l.complexSize, im, gainReal.get(), im);
...
convo::publishAtomic(m_ready, true, std::memory_order_release);
```

### 問題

`m_ready` がすでに `true` の状態で再構築が始まり、更新途中でオーディオスレッドが処理すると、**部分的に更新された IR / MKL バッファ**を読む可能性があります。

抽出コードには、更新開始時に

```cpp
convo::publishAtomic(m_ready, false, std::memory_order_release);
```

している様子が見えません。

### 影響

- IR 再構築中にノイズ
- 一時的な大音量
- 未初期化 / 中間状態の FFT バッファ使用
- クラッシュ

### 重要度

**High**

### 修正方針

更新開始時に ready を落とし、全構築完了後に上げます。

```cpp
convo::publishAtomic(m_ready, false, std::memory_order_release);

// build all layers...

convo::publishAtomic(m_ready, true, std::memory_order_release);
```

ただし、オーディオスレッドが `m_ready == false` のときに何を出力するかも定義が必要です。  
通常はバイパス、無音、または旧状態継続のいずれかにします。

---

## H5. `diagLog()` がオーディオスレッドから呼ばれるとリアルタイム性が破壊される

### 該当箇所

```cpp
void diagLog(const juce::String& message)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    DBG(message);
    juce::Logger::writeToLog(message);
#else
    juce::ignoreUnused(message);
#endif
}
```

### 問題

`DBG()` や `juce::Logger::writeToLog()` は一般に以下を起こし得ます。

- メモリアロケーション
- 文字列構築
- ファイル書き込み
- ロック
- デバッガ出力

これらがオーディオスレッドで呼ばれると、リアルタイムオーディオとして致命的です。

### 影響

- ドロップアウト
- 巨大なレイテンシスパイク
- ASIO ドライバ停止
- DAW 全体の音声停止
- 再現困難なプチノイズ

### 重要度

**High**

### 修正方針

オーディオスレッドからは、ロックフリーの診断イベントバッファにだけ書き込み、実際のログ出力は別スレッドで行うべきです。

```cpp
// Audio thread
diagBuffer.push(event);

// Message thread / logger thread
while (diagBuffer.pop(event))
    Logger::writeToLog(format(event));
```

---

## H6. MKL / IPP FFT plan 生成・破棄がオーディオスレッドで行われる可能性

### 該当箇所

```cpp
convo::ScopedDftiDescriptor fftHandle;
int fftHandleSize = 0;
...
#include <ipp.h>
```

また、`MKLNonUniformConvolver` 系に MKL バッファ確保があります。

### 問題

MKL / IPP の plan 作成、バッファ確保、`SetImpulse()` などがオーディオスレッドから呼ばれると、リアルタイム安全性を大きく損ないます。

抽出コードからは、これらがすべてメッセージスレッド / 事前準備スレッドに限定されているかが不明です。

### 影響

- 音声処理中のストール
- MKL 内部スレッド生成による優先度逆転
- 巨大なメモリ確保によるページフォルト
- 最悪、オーディオスレッド timeout

### 重要度

**High**

### 確認項目

- `prepare()` 以外で MKL allocation がないか
- `processBlock()` 経路で `DftiCreateDescriptor` 等がないか
- IR 再構築がオーディオスレッドで走らないか
- MKL が sequential モードか
- MKL スレッド数が 1 か

### 修正方針

- FFT plan は `prepare()` またはバックグラウンド構築時のみ生成
- オーディオスレッドでは既存 plan のみ使用
- MKL は `MKL_Set_Num_Threads(1)` 相当で固定
- 可能なら `mkl::sequential` を明示

---

## H7. `alignedL` / `alignedR` バッファがオーディオスレッド内で再確保される可能性

### 該当箇所

```cpp
convo::ScopedAlignedPtr<double> alignedL;
convo::ScopedAlignedPtr<double> alignedR;
int alignedCapacity = 0;
int maxSamplesPerBlock = 0;
```

### 問題

もし `processBlock()` 内で

```cpp
if (alignedCapacity < required)
    alignedL.reallocate(required);
```

のような処理がある場合、オーディオスレッド内でアロケーションが発生します。

### 影響

- リアルタイム性破壊
- 特定のブロックサイズ変更時のみ発生するスパイク
- ホストがブロックサイズを変えた瞬間に音切れ

### 重要度

**High**

### 修正方針

- `prepareToPlay()` で最大ブロックサイズ分を事前確保
- `processBlock()` では再確保しない
- 容量不足時は安全にエラー処理、または事前準備済み容量のみ処理

---

## H8. `delayBuffer` / `delayWritePos` の `reset()` 契約が破られるとデータレース

### 該当箇所

```cpp
convo::ScopedAlignedPtr<double> delayBuffer[2];
int delayBufferCapacity = 0;

// ★ bug3-6: API 契約: reset() は Audio Thread 停止後にのみ呼び出すこと。
// Audio Thread 実行中に reset() を呼び出すとデータレースが発生する。
int delayWritePos = 0;
```

### 問題

コメント自体が、`reset()` をオーディオスレッド実行中に呼ぶとデータレースになると認めています。  
つまり、呼び出し規約に依存する設計であり、**規約違反が起きると即バグ**です。

JUCE / ホスト実装によっては、`prepareToPlay()` や `reset()` がオーディオスレッド近傍で呼ばれる可能性があります。

### 影響

- 遅延バッファの破損
- ノイズ
- 未初期化メモリ読み込み
- クラッシュ

### 重要度

**High**

### 修正方針

- `reset()` 内部でオーディオスレッド停止を保証する
- またはオーディオスレッド内で安全に初期化できるよう atomic / sample-accurate な切り替え機構を持つ
- デバッグビルドでスレッド ID を検証する

```cpp
jassert(isAudioThreadStopped());
```

---

## H9. `ConvolverProcessor::prepare()` で `processor == nullptr` のとき Release で null 参照

### 該当箇所

```cpp
ConvolverProcessor& ref() noexcept
{
    jassert(processor != nullptr);
    return *processor;
}

const ConvolverProcessor& ref() const noexcept
{
    jassert(processor != nullptr);
    return *processor;
}

void prepare(AudioEngine* ownerEngine, double processingRate, int processingBlockSize) noexcept
{
    auto& proc = ref();
```

### 問題

`jassert` はリリースビルドで消える可能性が高いです。  
その場合、`processor == nullptr` なら `*processor` で即クラッシュします。

### 影響

- 初期化順序ミスでクラッシュ
- プラグイン再起動時クラッシュ
- デバイス切替時クラッシュ

### 重要度

**High**

### 修正方針

```cpp
ConvolverProcessor* get() noexcept
{
    return processor;
}

void prepare(...) noexcept
{
    if (processor == nullptr)
    {
        jassertfalse;
        return;
    }

    auto& proc = *processor;
    ...
}
```

---

## H10. `loadImpulseResponse()` の非同期読み込みでローカル `AudioFormatManager` / `AudioFormatReader` 寿命が危険

### 該当箇所

```cpp
bool loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime = false);
...
juce::AudioFormatManager formatManager;
formatManager.registerBasicFormats();

std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
...
const int64 fileLength = reader->lengthInSamples;
```

### 問題

もしこの関数が「読み込み開始」を非同期で行い、`reader` や `formatManager` をローカルに保持したまま返す場合、関数終了後にこれらが破棄されます。

非同期タスクが `reader` を使い続けるなら dangling の可能性があります。

### 影響

- IR 読み込み中クラッシュ
- 壊れた IR データ
- 特定フォーマットでのみ発生
- ファイルサイズが大きいとき発生率増

### 重要度

**High**

### 修正方針

- `AudioFormatManager` を静的 / メンバとして生存させる
- または `reader` を非同期タスク側へ所有権ごと移動する
- 非同期タスク完了まで `formatManager` を生存させる

---

## H11. `setConvHCFilterMode()` が状態を変更しているだけで、再構築が走るか不明

### 該当箇所

```cpp
void AudioEngine::setConvHCFilterMode(convo::HCMode mode) noexcept
{
    convo::publishAtomic(convHCFilterMode, mode, std::memory_order_release);

    // [Mem-Fix] NUC SoA (irFreqReal/irFreqImag) を再適用するため、uiConvolverProcessor を再構築する。
    // DSPCore::convolver は次回 requestRebuild 時に syncStateFrom +
```

### 問題

コメントには再構築が必要とあります。  
しかし抽出コードからは、この直後に実際に `requestRebuild()` しているか確認できません。

もし atomic 値を更新しているだけなら、フィルターモード変更が DSP に反映されません。

### 影響

- UI でハイカットモードを変えても音に反映されない
- 次回何らかの再構築が起きたときだけ反映される
- 状態と音が不一致

### 重要度

**High**

### 修正方針

```cpp
void AudioEngine::setConvHCFilterMode(convo::HCMode mode) noexcept
{
    convo::publishAtomic(convHCFilterMode, mode, std::memory_order_release);
    requestRebuild(RebuildReason::ConvFilterModeChanged);
}
```

---

## H12. `transferIRStateFrom()` は `noexcept` だが、AudioBuffer コピーで `bad_alloc` の可能性

### 該当箇所

```cpp
void transferIRStateFrom(const ConvolverProcessor& source) noexcept
{
    const IRState* srcState = source.acquireIRState();
    ...
```

### 問題

`IRState` の `AudioBuffer` をコピーするなら、メモリ確保が発生します。  
メモリ確保は `std::bad_alloc` を投げる可能性があります。

`noexcept` 関数内で例外が外へ出ると `std::terminate()` になります。

### 影響

- メモリ不足時、アプリケーションが即終了
- エラー回復不能

### 重要度

**Medium / High**

### 修正方針

- `noexcept` を外す
- またはコピー前にサイズ検証し、確保失敗を `try/catch` で握ってエラー状態にする
- リアルタイムパスでないなら、例外を許容する設計のほうが安全な場合もある

---

# 2. 並行性 / アトミック操作の問題

## C1. スナップショット取得が `memory_order_relaxed` ばかりで世代不整合が起きる可能性

### 該当箇所

```cpp
spec.processing.outputMakeupGain =
    static_cast<float>(convo::consumeAtomic(engine.outputMakeupGain, std::memory_order_relaxed));

spec.processing.convolverInputTrimGain =
    static_cast<float>(convo::consumeAtomic(engine.convolverInputTrimGain, std::memory_order_relaxed));

spec.processing.autoGainStagingEnabled =
    convo::consumeAtomic(engine.autoGainStagingEnabled, std::memory_order_relaxed);
```

### 問題

複数の関連パラメータを relaxed で個別に読むと、読み出し中に他スレッドが更新した場合、**新旧混在スナップショット**になります。

例：

- `inputHeadroomGain` は旧値
- `outputMakeupGain` は新値
- `generation` は別値

### 影響

- リビルド判定ミス
- 過剰な再構築
- 再構築漏れ
- UI と音声状態の不一致
- ゲインステージングの瞬間的不整合

### 重要度

**Medium / High**

### 修正方針

- acquire fence を使う
- generation counter で二重読み取り検証する
- またはメッセージスレッドでロック付きスナップショットを作る

例：

```cpp
for (;;)
{
    auto g1 = convo::consumeAtomic(engine.snapshotGeneration, std::memory_order_acquire);

    // read values...

    auto g2 = convo::consumeAtomic(engine.snapshotGeneration, std::memory_order_acquire);

    if (g1 == g2)
        break;
}
```

---

## C2. `RebuildTask` / `lastQueuedTaskSignature` の比較がパディング込みなら偽陽性 / 偽陰性

### 該当箇所

```cpp
struct RebuildTask
{
    ...
    convo::BuildInput buildInput {};
    ConvolverProcessor::BuildSnapshot convolverBuildSnapshot {};
    convo::RuntimeBuildSnapshot runtimeBuildSnapshot {};
    convo::BuildAnalysis buildAnalysis {};
    convo::OversamplingResult oversamplingResult {};
    convo::BuildDiagnostics buildDiagnostics {};
    int generation = 0;
};

RebuildTask pendingTask;
RebuildTask lastQueuedTaskSignature;
```

### 問題

もし `RebuildTask` や内部構造体にパディングがあり、それを `memcmp` 等で比較している場合、未初期化パディングのせいで以下が起きます。

- 実際は同一タスクなのに違うと判定され、再構築が止まらない
- 実際は違うのに同じと判定され、再構築が漏れる

### 影響

- CPU 負荷増大
- 再構築スパム
- 状態更新漏れ
- オーディオ途切れ

### 重要度

**Medium**

### 修正方針

- `memcmp` を使わない
- フィールド単位で `operator==` を書く
- 構造体を `memset` しない
- パディングを排除するか、比較対象から外す
- 可能ならハッシュ対象を明示する

---

## C3. `EQEditProcessor::scheduleDebounce()` と `timerCallback()` の競合

### 該当箇所

```cpp
void EQEditProcessor::scheduleDebounce()
{
    convo::publishAtomic(pendingSnapshot, true, std::memory_order_release);

    if (!isTimerRunning())
        startTimer(kDebounceMs);
}

void EQEditProcessor::timerCallback()
{
    stopTimer();

    if (convo::exchangeAtomic(pendingSnapshot, false,
```

### 問題

もし `scheduleDebounce()` がメッセージスレッド以外から呼ばれる場合、JUCE Timer の制約に違反します。  
また、同一スレッドだとしても、以下の順序で更新が失われる可能性があります。

1. `timerCallback()` が `stopTimer()`
2. 別処理が `pendingSnapshot = true`
3. `timerCallback()` が `exchangeAtomic(pendingSnapshot, false)`

この場合、2 の要求が消失します。

### 影響

- EQ 編集が反映されない
- デバウンスが更新を飲み込む
- UI 操作取りこぼし

### 重要度

**Medium**

### 修正方針

- `scheduleDebounce()` はメッセージスレッド専用にする
- `pendingSnapshot` のクリアと処理を原子化する
- タイマー再起動を `exchange` 後に行う

---

## C4. `rcuProvider` が `std::reference_wrapper<AudioEngine>` で、寿命が保証されない

### 該当箇所

```cpp
std::optional<std::reference_wrapper<AudioEngine>> rcuProvider;

[[nodiscard]] AudioEngine* getRcuProvider() noexcept
{
    return rcuProvider ? &rcuProvider->get() : nullptr;
}
```

### 問題

`reference_wrapper` は所有しません。  
`AudioEngine` が先に破棄されると、`ConvolverProcessor` 側に dangling reference が残ります。

### 影響

- 破棄順序ミスでクラッシュ
- デバイス切替 / ウィンドウ破棄時クラッシュ

### 重要度

**Medium / High**

### 修正方針

- 所有関係を見直す
- `AudioEngine` が `ConvolverProcessor` より必ず長生きすることを保証する
- または weak 相当の安全な参照機構にする

---

## C5. `AudioEngineProcessor::getTailLengthSeconds()` が ValueTree を読むときスレッド安全性が不明

### 該当箇所

```cpp
double AudioEngineProcessor::getTailLengthSeconds() const
{
    const auto convState = audioEngine.getConvolverStateTree();

    if (!convState.isValid())
        return 0.0;

    const double irLengthSec =
        static_cast<double>(convState.getProperty("irLength", 0.0));
```

### 問題

`juce::ValueTree` は基本的にメッセージスレッド想定の API です。  
もし `getConvolverStateTree()` が内部状態をロックなしで返す場合、他スレッドでの更新と競合します。

### 影響

- 不正な tail length
- クラッシュ
- ホストが誤ったテール処理をする

### 重要度

**Medium**

### 修正方針

- `irLength` を atomic な値として保持する
- または ValueTree へのアクセスをメッセージスレッドに限定する
- `getTailLengthSeconds()` 用にロックフリーな最新値を持つ

---

# 3. DSP / 数値精度の問題

## D1. `doubleArrayToString()` が 16 桁では double を完全往復できない可能性

### 該当箇所

```cpp
juce::String doubleArrayToString(const double* arr, int size)
{
    juce::StringArray strArr;

    for (int i = 0; i < size; ++i)
        strArr.add(juce::String(arr[i], 16));

    return
```

### 問題

IEEE 754 double を十進文字列から完全に復元するには、一般に **17 桁**必要です。  
16 桁では、値によって往復誤差が出ます。

### 影響

- ノイズシェイパー係数が保存 / 復元で微妙に変わる
- EQ 係数ハッシュが不安定になる
- 学習結果が再読み込み後に変わる
- 再現性が失われる

### 重要度

**Medium**

### 修正方針

```cpp
strArr.add(juce::String(arr[i], 17));
```

または、C++17 以降なら `std::to_chars` を使うのがより安全です。

```cpp
char buf[32];
auto res = std::to_chars(buf, buf + sizeof(buf), arr[i], std::chars_format::general, 17);
```

---

## D2. `/fp:fast` が全 Release に適用されており、IEEE 厳密性が崩れる

### 該当箇所

```cmake
set(CMAKE_CXX_FLAGS_RELEASE
    "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /fp:fast /Gw /Gy /Zi /utf-8")
```

icx：

```cmake
set(CMAKE_CXX_FLAGS_RELEASE
    "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8")
```

### 問題

`/fp:fast` は浮動小数点の厳密な IEEE 挙動を崩す可能性があります。  
オーディオ DSP では、フィルター安定性、EQ 係数、閾値比較、NaN/Inf 判定に影響することがあります。

### 影響

- Debug と Release で音が違う
- コンパイラバージョンで挙動が変わる
- フィルターが不安定になる
- 微小信号が消える
- 比較演算が意図と異なる

### 重要度

**Medium**

### 修正方針

- DSP コアだけ `/fp:precise` または `/fp:strict` を検討
- どうしても `/fp:fast` を使うなら、重要箇所を個別に検証
- denormal / NaN / Inf 対策を明示的に行う

---

## D3. ノイズシェイパーのスケールが 24bit 固定に見える

### 該当箇所

```cpp
= 1.0 / 8388608.0; // 2^23（24 bit signed PCM デフォルト）
double invScale = 8388608.0; // 2^23（24bit signed PCM デフォルト）
```

### 問題

ビット深度が可変なら、スケールも可変であるべきです。  
16bit, 24bit, 32bit で同じスケールを使うと、ディザー / ノイズシェイプの量子化が正しくありません。

### 影響

- 16bit 出力でノイズ特性が悪化
- 32bit 出力で効果がほぼ消える
- 意図しない量子化歪み

### 重要度

**Medium**

### 修正方針

```cpp
double getScaleForBitDepth(int bitDepth)
{
    switch (bitDepth)
    {
        case 16: return 1.0 / 32768.0;
        case 24: return 1.0 / 8388608.0;
        case 32: return 1.0 / 2147483648.0;
        default: return 1.0;
    }
}
```

---

## D4. EQ / フィルター係数計算で NaN / Inf / 不正範囲の防御が見えない

### 関連箇所

- `EQProcessor.Coefficients.cpp`
- `OutputFilter.cpp`
- `AllpassDesigner.cpp`
- `BandHelper.cpp`

### 問題

EQ パラメータで以下が入った場合の防御が必要です。

- `frequency <= 0`
- `Q <= 0`
- `gain == ±inf`
- `sampleRate <= 0`
- `NaN`

抽出コードからは、これらの検証が十分か不明です。

### 影響

- フィルター係数が NaN になる
- 音声が永久に無音 / 発振
- クラッシュ
- メーターが壊れる

### 重要度

**Medium / High**

### 修正方針

- 全パラメータ入口で clamp
- 係数生成後に `std::isfinite` 検証
- 不正値なら安全なデフォルトへフォールバック

---

## D5. `getTailLengthSeconds()` が IR 長だけで、テール強度 / oversampling を反映していない可能性

### 該当箇所

```cpp
const double irLengthSec =
    static_cast<double>(convState.getProperty("irLength", 0.0));
```

### 問題

実際のテール長が以下で変わる場合、ホストに過小な tail length を伝えます。

- tail strength
- oversampling
- filter
- saturation
- dither

### 影響

- テールが途中で切れる
- 残響尾が不自然に途切れる
- オフラインレンダリングで末尾欠け

### 重要度

**Low / Medium**

### 修正方針

実際の DSP チェーンに基づく有効テール長を計算すべきです。

---

# 4. UI / 状態管理の問題

## U1. 両方バイパス時の表示が誤っている可能性

### 該当箇所

```cpp
if (convBypassed && !eqBypassed)
{
    modeText = "PEQ only";
}
else if (!convBypassed && !eqBypassed && order == AudioEngine::ProcessingOrder::EQThenConvolver)
{
    modeText = "PEQ -> Conv";
}
else if (eqBypassed && !convBypassed)
{
    modeText = "Conv only";
}
else
{
    modeText = "Conv -> PEQ";
}
```

### 問題

`convBypassed && eqBypassed` の場合、最後の `else` に入り、

```cpp
modeText = "Conv -> PEQ";
```

になります。

実際には両方バイパスなのに、UI は Conv -> PEQ と表示します。

### 影響

- ユーザーが状態を誤認
- バイパス中なのにエフェクト有効に見える

### 重要度

**Medium**

### 修正例

```cpp
if (convBypassed && eqBypassed)
{
    modeText = "Bypass";
}
else if (convBypassed && !eqBypassed)
{
    modeText = "PEQ only";
}
else if (!convBypassed && eqBypassed)
{
    modeText = "Conv only";
}
else if (order == AudioEngine::ProcessingOrder::EQThenConvolver)
{
    modeText = "PEQ -> Conv";
}
else
{
    modeText = "Conv -> PEQ";
}
```

---

## U2. `MessageBoxA` に UTF-8 日本語文字列を渡しており文字化けする可能性

### 該当箇所

```cpp
::MessageBoxA(nullptr,
    "ConvoPeq には AVX2 および FMA 命令に対応した CPU が必要です。\n"
    ...
    "ConvoPeq - CPU 非対応",
    MB_OK| MB_ICONERROR);
```

### 問題

プロジェクトは `/utf-8` や `_UNICODE` / `UNICODE` を使っています。  
それなのに `MessageBoxA` を使うと、 narrow string literal が UTF-8 の場合に ANSI code page として解釈され、日本語が文字化けする可能性があります。

### 影響

- エラーメッセージが読めない
- 製品品質低下

### 重要度

**Low / Medium**

### 修正例

```cpp
::MessageBoxW(nullptr,
    L"ConvoPeq には AVX2 および FMA 命令に対応した CPU が必要です。\n"
    L"Intel Haswell (2013) 以降、または AMD Excavator (2015) 以降の\n"
    L"CPU が必要です。\n\n"
    L"この CPU ではアプリケーションがクラッシュする可能性があるため、\n"
    L"実行を中断します。",
    L"ConvoPeq - CPU 非対応",
    MB_OK | MB_ICONERROR);
```

---

## U3. `DeviceSettings::getSettingsFile()` がディレクトリ作成失敗を無視

### 該当箇所

```cpp
auto appDataDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
    .getChildFile("ConvoPeq");

if (!appDataDir.exists())
    appDataDir.createDirectory();

return appDataDir.getChildFile("device_settings.xml");
```

### 問題

`createDirectory()` の戻り値を確認していません。  
権限不足、パス長制限、ディスクエラー時に、存在しないディレクトリを返します。

### 影響

- 設定保存失敗
- 設定が毎回初期化
- エラーがユーザーに見えない

### 重要度

**Low**

### 修正方針

```cpp
if (!appDataDir.exists())
{
    auto result = appDataDir.createDirectory();

    if (!result.wasOk())
    {
        // fallback path or error log
    }
}
```

---

## U4. `doubleArrayToString()` が `nullptr` / 負サイズに対して防御されていない

### 該当箇所

```cpp
juce::String doubleArrayToString(const double* arr, int size)
{
    juce::StringArray strArr;

    for (int i = 0; i < size; ++i)
        strArr.add(juce::String(arr[i], 16));
```

### 問題

- `arr == nullptr && size > 0`
- `size < 0`

の場合に危険です。

### 影響

- クラッシュ
- 巨大ループ

### 重要度

**Low**

### 修正例

```cpp
if (arr == nullptr || size <= 0)
    return {};

juce::StringArray strArr;
strArr.ensureStorageAllocated(size);
...
```

---

# 5. ビルド設定 / CMake の問題

## B1. `/MT` を手動で足しており、`MSVC_RUNTIME_LIBRARY` property と競合する可能性

### 該当箇所

```cmake
set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>")
```

icx：

```cmake
target_compile_options(ConvoPeq PRIVATE
    $<$<CONFIG:Release>:/MT>
)
```

### 問題

CMake の `MSVC_RUNTIME_LIBRARY` property と、手動の `/MT` / `/MD` 指定が混在すると、以下が起き得ます。

- 二重指定
- リンク警告
- CRT 不整合
- MKL / JUCE との ABI 不整合

### 影響

- ビルド失敗
- リンク警告
- 実行時 CRT ヒープ破損
- static/dynamic CRT 混在

### 重要度

**Medium**

### 修正方針

- `CMP0091` を NEW にする
- runtime library は property に統一する
- 手動 `/MT` は削除する

---

## B2. icx Release だけ `/MT`、Debug で runtime library が不整合の可能性

### 該当箇所

```cmake
target_compile_options(ConvoPeq PRIVATE
    $<$<CONFIG:Release>:/MT>
)
```

### 問題

Release は static CRT、Debug は dynamic CRT になる可能性があります。  
MKL も static link する場合、Debug/Release で ABI やヒープが混在すると問題になることがあります。

### 影響

- Debug だけクラッシュ
- Release だけクラッシュ
- ヒープ破損
- MKL 内部状態の不整合

### 重要度

**Medium**

---

## B3. PGO オプションが Intel icx で暗黙に無効化される可能性

### 該当箇所

```cmake
if(CONVOPEQ_PGO_INSTRUMENT)
    target_compile_options(ConvoPeq PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/GL>
    )
    target_link_options(ConvoPeq PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/LTCG /GENPROFILE:PGD=${CONVOPEQ_PGO_PGD}>
    )
```

### 問題

PGO フラグが MSVC 限定です。  
ユーザーが `icx` ビルドで PGO を期待しても、何も起きません。

### 影響

- PGO されていない Release が生成される
- ユーザーが性能改善されたと誤認

### 重要度

**Low / Medium**

### 修正方針

```cmake
if(CONVOPEQ_PGO_INSTRUMENT AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(WARNING "PGO is currently supported only for MSVC.")
endif()
```

---

## B4. `add_dependencies(ConvoPeq ...Tests)` によりアプリ本体がテストビルドに依存している

### 該当箇所

```cmake
add_dependencies(ConvoPeq GainStagingContractTests EQProcessorMaxGainTests)
```

### 問題

アプリ本体をビルドするたびにテストもビルドされます。  
テスト側のコンパイルエラーがアプリ本体のビルドを阻害します。

### 影響

- 開発体験悪化
- CI で本体だけ欲しいときに失敗
- テスト用依存関係が本体に漏れる

### 重要度

**Low**

### 修正方針

通常は逆です。

```cmake
add_dependencies(AllTests ConvoPeq)
```

または `ctest` / 専用ターゲットに分離します。

---

## B5. テストの `/STACK:8388608` と `/GS-` が MSVC 限定

### 該当箇所

```cmake
if(MSVC)
    target_compile_options(BuildInputSemanticContractTests PRIVATE /GS-)
    target_link_options(BuildInputSemanticContractTests PRIVATE "/STACK:8388608")
endif()
```

### 問題

icx で同じテストをビルドすると、スタックサイズが足りずスタックオーバーフローする可能性があります。  
また `/GS-` はセキュリティチェックを無効化します。

### 影響

- icx テストでクラッシュ
- スタック破損検知不能

### 重要度

**Medium**

### 修正方針

- スタックを大量に使うテスト自体を見直す
- 再帰 / 巨大ローカル配列をヒープへ移動
- どうしても必要なら icx 側にも同等のスタック指定を入れる

---

## B6. 警告抑制 `/wd4100` `/wd4189` が広すぎて潜在バグを隠す可能性

### 該当箇所

```cmake
/wd4100 # C4100: 参照されないパラメーター
/wd4189 # C4189: ローカル変数が初期化されましたが、参照されていません
```

### 問題

JUCE 内部警告抑制としては理解できますが、プロジェクト全体に適用すると、実際のバグの兆候を見逃します。

### 影響

- 未使用変数に隠れたロジックミス
- 初期化だけされて使われていない重要変数の見逃し

### 重要度

**Low**

### 修正方針

- 警告抑制は外部ライブラリ限定にする
- 自前コードでは有効にする
- `SYSTEM` include を活用する

---

## B7. `target_include_directories(... SYSTEM PRIVATE ...)` が自前ヘッダまで SYSTEM 化している可能性

### 該当箇所

```cmake
# Mark JUCE includes as SYSTEM to suppress clang-tidy warnings.
target_include_directories(ConvoPeq SYSTEM PRIVATE
# This makes all headers under
```

### 問題

もし自前 `src/` まで SYSTEM 指定していると、コンパイラ / clang-tidy の警告が自前コードでも抑制されます。

### 影響

- 自前コードの警告が見えない
- バグ発見が遅れる

### 重要度

**Medium**

### 修正方針

- JUCE のみ SYSTEM
- r8brain のみ SYSTEM
- 自前コードは通常 include

---

# 6. ISR / Runtime Publication / Evidence 系の問題

## R1. `shutdown_trace.json` が最初から `verified:true` / counters zero で出力されている可能性

### 該当箇所

```cpp
"{\"artifact\":\"shutdown_trace.json\",\"schema\":\"shutdown_trace_v1\",\"status\":\"generated\",\"phase\":0,\"verified\":true,\"sh1_callbackCount\":0,...}"
```

### 問題

もしこれが実際のシャットダウン計測前に生成されるテンプレートなら、**検証済みという偽のエビデンス**になります。

### 影響

- 障害調査で誤った判断
- 実際にはシャットダウンが壊れているのに正常に見える

### 重要度

**Medium**

### 修正方針

- 実測値を入れる
- 未計測なら `"verified": false`
- `"status": "template"` にする

---

## R2. `retire_latency_report.json` が既定で `withinThreshold:true` になっている可能性

### 該当箇所

```cpp
"{\"artifact\":\"retire_latency_report.json\",\"schema\":\"retire_latency_report_v1\",\"status\":\"generated\",\"withinThreshold\":true}"
```

### 問題

実測なしに `withinThreshold:true` なら、 retire latency が閾値超過していても隠れます。

### 影響

- リソース解放遅延の見逃し
- メモリ肥大化の兆候を見逃す

### 重要度

**Medium**

---

## R3. JSON 手組み立てでエスケープ不足の可能性

### 該当箇所

```cpp
manifest += " \"runtimeRunId\": \"" + runId + "\",\n";
manifest += " \"runId\": \"" + runId + "\",\n";
manifest += " \"buildMode\": \"" + buildMode + "\",\n";
manifest += " \"proofLevel\": \"" + proofLevel + "\",\n";
```

### 問題

`runId`, `buildMode`, `proofLevel` に以下が含まれると JSON が壊れます。

- `"`
- `\`
- 改行
- 制御文字

### 影響

- evidence manifest が解析不能
- 外部ツールで読めない
- 証跡チェーン破損

### 重要度

**Medium**

### 修正方針

JSON エスケープ関数を使うか、JSON ライブラリを使うべきです。

---

## R4. `RuntimePublicationOrchestrator` や ISR 系メンバが多数あるが、初期化失敗時のフェイルセーフが見えない

### 該当メンバ例

```cpp
convo::isr::ClosureGraphWalker closureGraphWalker_;
convo::isr::DebugRuntime debugRuntime_;
convo::isr::RetireRuntime retireRuntime_;
convo::isr::RetireRuntimeEx retireRuntimeEx_;
convo::isr::ShutdownRuntime shutdownRuntime_;
convo::isr::EvidenceExporter evidenceExporter_;
convo::isr::WorldLifecycleAudit worldLifecycleAudit_;
convo::isr::BudgetManager budgetManager_;
convo::isr::FailureHandler failureHandler_;
convo::isr::IntrospectionConsole introspectionConsole_;
```

### 問題

これらの初期化がメモリ確保を伴う場合、初期化失敗時にどうなるかが重要です。  
`noexcept` 初期化で `bad_alloc` が起きると terminate します。

### 影響

- 起動時クラッシュ
- メモリ不足時に回復不能

### 重要度

**Medium**

---

# 7. その他のバグ候補 / 設計リスク

## O1. `build.bat` で遅延展開が使われているが、`setlocal enabledelayedexpansion` が見えない

### 該当箇所

```bat
if exist "!EXE_PATH!" (
```

### 問題

`!EXE_PATH!` を使うには `setlocal enabledelayedexpansion` が必要です。  
抽出コードに見当たらない場合、バッチが正しく動きません。

### 影響

- ビルド成功判定が常に失敗
- 変数が空になる

### 重要度

**Medium**

### 修正例

```bat
setlocal enabledelayedexpansion
```

---

## O2. `CMAKE_C_FLAGS_DEBUG` に `/EHsc` が入っている

### 該当箇所

```cmake
set(CMAKE_C_FLAGS_DEBUG "/D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /EHsc")
```

### 問題

`/EHsc` は C++ 例外用オプションです。C コンパイルに不要、または警告の原因になります。

### 影響

- ビルド警告
- コンパイラメッセージ増加

### 重要度

**Low**

---

## O3. `/Zm400` は最新 MSVC で効果が薄い / 非推奨気味

### 該当箇所

```cmake
/Zm400
```

### 問題

最新 MSVC では `/Zm` はほぼ不要、または無視されることがあります。

### 影響

- 設定の誤解
- ビルド時間増加の可能性

### 重要度

**Low**

---

## O4. `/MP1` コメントが「マルチプロセッサコンパイル有効化」だが、実質 1 並列

### 該当箇所

```cmake
/MP1 # マルチプロセッサコンパイル有効化 (1コアに制限してメモリ使用量を最小化)
```

### 問題

`/MP1` は並列数を 1 にします。  
コメントが誤解を招きます。

### 影響

- ビルドが遅い
- 意図が伝わらない

### 重要度

**Low**

---

## O5. CPU 検査が AVX2 実行より後になる可能性

### 該当箇所

```cpp
::MessageBoxA(nullptr, ... "AVX2 および FMA 命令に対応した CPU が必要です。" ...);
return false;
```

### 問題

AVX2 必須バイナリの場合、CPU 非対応環境では **チェックに到達する前**に AVX2 命令でクラッシュする可能性があります。

特に、CRT 初期化、静的コンストラクタ、JUCE 初期化で AVX2 命令が生成されると防げません。

### 影響

- 非対応 CPU で MessageBox が出ずクラッシュ
- エラーメッセージが表示されない

### 重要度

**Medium**

### 修正方針

- AVX2 を使わない小さな launcher を用意する
- launcher で CPUID を確認してから本体起動
- または全コードを AVX2 なしでコンパイルし、DSP 部分だけ動的に AVX2 関数ポインタへ切り替える

---

## O6. `JUCE_USE_SSE_INTRINSICS=1` と AVX2 必須の整合性

### 該当箇所

```cmake
JUCE_USE_SSE_INTRINSICS=1
JUCE_USE_SIMD=1
```

### 問題

JUCE 内部 SIMD と自前 AVX2 の前提が混在すると、最適化の前提が崩れることがあります。

### 影響

- 想定しないコード生成
- 性能低下
- 数値差異

### 重要度

**Low**

---

## O7. `Logger::writeToLog()` が UI コールバックで同期的に呼ばれている

### 該当箇所

```cpp
DBG("[DIAG] MainWindow::changeListenerCallback leave (audioEngine)");
juce::Logger::writeToLog("[DIAG]
```

### 問題

UI スレッドでも、ファイル書き込みが同期的だと操作感が悪化します。

### 影響

- UI カクつき
- ディスク遅延に引っ張られる

### 重要度

**Low**

---

## O8. `bitDepthComboBox.setSelectedId(999, ...)` の 999 が実際に存在するか不明

### 該当箇所

```cpp
bitDepthComboBox.setSelectedId(999, juce::dontSendNotification);
audioEngine.setDitherBitDepth(0);
```

### 問題

ComboBox に ID 999 の項目がなければ、未選択または不正表示になります。

### 影響

- UI 表示と内部状態の不一致

### 重要度

**Low**

---

## O9. `orderModeBox.setSelectedId(modeId, ...)` が項目構築前だと無効

### 該当箇所

```cpp
orderModeBox.setSelectedId(modeId, juce::dontSendNotification);
```

### 問題

ComboBox に item が追加される前に `setSelectedId` すると、選択が反映されません。

### 影響

- 起動時 UI が正しいモードを表示しない

### 重要度

**Low / Medium**

---

## O10. `ConvolverProcessor::updateIRState(const std::unique_ptr<...>&)` が const 参照で所有権を受け取れない

### 該当箇所

```cpp
void updateIRState(const std::unique_ptr<juce::AudioBuffer<double>>& newIR, double newSR, ...);
```

### 問題

`unique_ptr` の const 参照では move できません。  
もし所有権移譲を意図しているなら API が誤っています。

### 影響

- 不要なコピー
- 意図しない所有権
- コピーによるメモリ増加

### 重要度

**Low / Medium**

### 修正例

```cpp
void updateIRState(std::unique_ptr<juce::AudioBuffer<double>> newIR, double newSR, ...);
```

---

# 8. 特に優先して直すべき項目

優先度が高い順は以下です。

1. **H1 AVX2 の `s + i - 1` 越境読み込み**
2. **H3 `IRState*` の寿命管理 / use-after-free**
3. **H2 `acquireIRState()` 対応 release 漏れ**
4. **H4 `m_ready` の false/true 切替順序**
5. **H5/H6 オーディオスレッド内ログ / MKL 確保**
6. **H7 aligned buffer の RT 再確保**
7. **H11 filter mode 変更後の rebuild 漏れ**
8. **C1 relaxed atomic によるスナップショット不整合**
9. **C2 RebuildTask 比較のパディング問題**
10. **U1 両方バイパス時の UI 表示誤り**

---

# 9. 推奨される検証テスト

## 9.1 AVX2 / SIMD 境界テスト

最低でも以下を入力してください。

```cpp
samplesToRead = 0
samplesToRead = 1
samplesToRead = 2
samplesToRead = 3
samplesToRead = 4
samplesToRead = 5
samplesToRead = 7
samplesToRead = 8
samplesToRead = 4095
samplesToRead = 4096
```

確認項目：

- 前方読み込みが合法か
- 後方読み込みが合法か
- スカラ remainder と SIMD 本体の結果一致
- フィルター状態が正しく更新されるか

---

## 9.2 IRState RCU ストレステスト

手順：

1. メッセージスレッドから高頻度に `updateIRState()`
2. オーディオスレッド模倣スレッドから高頻度に `acquireIRState()` / `releaseIRState()`
3. 数分間実行
4. メモリが増え続けないか確認
5. ASan / Intel Inspector で UAF 確認

---

## 9.3 RebuildTask signature テスト

確認：

- 同一パラメータで再構築が再発しない
- 1 パラメータ変更で必ず再構築される
- パディングで差分が出ない
- 構造体初期化漏れがない

---

## 9.4 double 文字列往復テスト

```cpp
for (double v : testValues)
{
    auto s = doubleToString(v);
    auto r = stringToDouble(s);
    EXPECT_EQ(v, r);
}
```

特に以下：

- 極小値
- 極大値
- denormal 付近
- 負のゼロ
- NaN / Inf は保存しない設計ならそれを検証

---

## 9.5 リアルタイムパスのアロケーション検査

Windows では以下が有効です。

- WPR / WPA で ETW 取得
- VirtualAlloc / HeapAlloc イベントを確認
- オーディオスレッド内の `ntdll.dll!RtlAllocateHeap` を確認
- MKL / IPP のスレッド生成を確認
- `juce::Logger::writeToLog` がオーディオスレッドに出ていないか確認

---

# 10. 総合所見

提供された断片から見る限り、ConvoPeq は非常に大規模で、ISR / RCU / MKL / AVX2 / PGO / evidence exporter まで含む高度な設計です。  
しかし、それゆえに以下の領域に高リスクバグが集中しています。

1. **SIMD 境界処理**
2. **RCU / Epoch / IRState 寿命管理**
3. **オーディオスレッド内のアロケーション / ログ / MKL 呼び出し**
4. **atomic のメモリオーダーとスナップショット整合性**
5. **再構築トリガの漏れ**
6. **UI 状態と DSP 状態の不一致**
7. **CMake の CRT / PGO / テスト依存関係**

中でも、`s + i - 1` の AVX2 読み込みと、`IRState*` の寿命管理は、実際にクラッシュや未定義動作を引き起こし得るため、最優先で実コードを確認すべきです。


はい。追加で提示された `ConvoPeq.md`、`coding_rule_jp.txt` を踏まえると、前回指摘分以外にも **高リスク〜中リスクのバグ候補** がまだ複数あります。  
なお `Practical Stable ISR Bridge Runtime.md` は提供内容が空に見えるため、このファイル由来の追加制約は反映できていません。

以下は「確定」ではなく、**実コードで確認すべき強い疑い**を含みます。

---

# 追加バグ候補：優先度高

---

## A1. IR キャッシュ読み込みで `size_t` アンダーフローによる越境読み込みの可能性

### 該当箇所

```cpp
const uint8_t* tdStart = dataStart + header.dataSize;
const size_t expectedTdBytes = static_cast<size_t>(header.timeDomainSizeBytes);

if (static_cast<size_t>(mmap.getSize() - headerSize - header.dataSize) >= expectedTdBytes)
{
    auto tdBuffer = std::make_unique<juce::AudioBuffer<double>>(
        static_cast<int>(header.timeDomainChannels),
        static_cast<int>(header.timeDomainNumSamples));

    const double* tdSrc = reinterpret_cast<const double*>(tdStart);
```

### 問題

`mmap.getSize()` はファイルサイズです。  
もし以下の場合、

```cpp
mmap.getSize() < headerSize + header.dataSize
```

引き算が負になり、それを `size_t` にキャストすると **巨大な正の値** になります。

その結果、

```cpp
static_cast<size_t>(negative_value) >= expectedTdBytes
```

が true になり、本来通ってはいけない分岐に入る可能性があります。

### 影響

- メモリマップ領域外読み込み
- クラッシュ
- 壊れた IR キャッシュの読み込み
- セキュリティ上の問題

### 重要度

**High / Critical 寄り**

### 修正例

```cpp
const int64_t fileSize = mmap.getSize();

if (fileSize < static_cast<int64_t>(headerSize))
    return false;

const int64_t remainingAfterHeader =
    fileSize - static_cast<int64_t>(headerSize);

if (header.dataSize > static_cast<uint64_t>(remainingAfterHeader))
    return false;

const int64_t remainingAfterFreqData =
    remainingAfterHeader - static_cast<int64_t>(header.dataSize);

if (remainingAfterFreqData < 0)
    return false;

if (static_cast<uint64_t>(remainingAfterFreqData) < expectedTdBytes)
    return false;
```

---

## A2. IR キャッシュヘッダの `int` キャストが未検証

### 該当箇所

```cpp
static_cast<int>(header.timeDomainChannels)
static_cast<int>(header.timeDomainNumSamples)
```

### 問題

`header.timeDomainChannels` や `header.timeDomainNumSamples` が巨大な場合、`int` へキャストすると負値や桁落ちになります。

例えば：

```cpp
uint32_t timeDomainNumSamples = 0xFFFFFFFFu;
static_cast<int>(timeDomainNumSamples); // -1 になる可能性
```

その値で `AudioBuffer<double>` を構築すると、異常なメモリ確保、負サイズ、クラッシュにつながります。

### 影響

- 巨大メモリ確保
- `std::bad_alloc`
- クラッシュ
- 不正キャッシュによる起動不能

### 重要度

**High**

### 修正方針

```cpp
if (header.timeDomainChannels == 0 || header.timeDomainChannels > 2)
    return false;

if (header.timeDomainNumSamples == 0)
    return false;

if (header.timeDomainChannels > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    return false;

if (header.timeDomainNumSamples > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    return false;

const uint64_t expected =
    static_cast<uint64_t>(header.timeDomainChannels)
    * static_cast<uint64_t>(header.timeDomainNumSamples)
    * sizeof(double);

if (expected != header.timeDomainSizeBytes)
    return false;
```

---

## A3. `isBadSampleV()` の閾値がスカラ版 `isBadSample()` と一致していない

### 該当箇所

スカラ版：

```cpp
constexpr uint64_t limit = 0x4340000000000000ULL;
return (bits & 0x7FFFFFFFFFFFFFFFULL) > limit;
```

AVX2 版：

```cpp
const __m256d vInfMask =
    _mm256_cmp_pd(vAbs, _mm256_set1_pd(1e20), _CMP_GT_OQ);
```

### 問題

スカラ版の `0x4340000000000000` は およそ `2^53 = 9007199254740992` です。  
一方、AVX2 版は `1e20` を使っています。

つまり、AVX2 版は以下の範囲の異常値を検出しません。

```text
約 9.0e15 < |x| <= 1e20
```

### 影響

- SIMD パスだけ異常サンプルを見逃す
- Debug と Release、またはスカラ路径と SIMD 路径で挙動不一致
- 破損検出が遅れる
- 過大値が後段に伝播する

### 重要度

**High / Medium**

### 修正例

```cpp
constexpr double kBadSampleLimit = 9007199254740992.0; // 2^53

const __m256d vInfMask =
    _mm256_cmp_pd(vAbs, _mm256_set1_pd(kBadSampleLimit), _CMP_GT_OQ);
```

---

## A4. TruePeakDetector の AVX2 / 履歴参照でバッファ越境の疑い

### 該当箇所

```cpp
const double centerSample = history[base - stage.centerTap];
...
acc += dotProductDecimateAvx2(
    history + (base
```

および：

```cpp
output[n * 2 + 0] = 0.0;
output[n * 2 + 1] = 0.0;
```

### 問題

以下の前提が崩れると越境アクセスになります。

1. `history` が `base` から十分後方まで有効
2. `base - stage.centerTap >= 0`
3. `base - stage.convParity - (r << 1) >= 0`
4. `output` が最低 `numSamples * 2` 要素持つ

抽出コードだけでは、これらが常に保証されているか確認できません。

### 影響

- ヒープ破損
- クラッシュ
- 特定サンプル数でのみ発生するノイズ
- ASan で buffer-overflow

### 重要度

**High**

### 修正方針

リリースビルドでも安全に抜けるガードを入れるべきです。

```cpp
if (base < stage.centerTap)
{
    output[n] = 0.0;
    markCorruptionDetected();
    continue;
}
```

また 2x 出力バッファも検証します。

```cpp
jassert(output != nullptr);
jassert(numSamples >= 0);
jassert(outputCapacity >= static_cast<size_t>(numSamples) * 2);
```

---

## A5. `stateSnapshot` ポインタがオーディオスレッドで参照されている場合、retire 済みオブジェクト参照の疑い

### 該当箇所

```cpp
const double saturation = (stateSnapshot != nullptr)
    ? ...
```

### 問題

もし `stateSnapshot` が非 RT スレッドで更新・破棄されるオブジェクトを指しており、オーディオスレッドがその生ポインタを参照しているなら、retire 後に use-after-free になる可能性があります。

コーディング規約では、大きなデータや状態は SPSC FIFO、二重バッファ、スナップショット、アトミックなポインタ取得で扱うことになっています。

### 影響

- 状態更新時のみクラッシュ
- saturation 値が壊れる
- 不正な DSP 状態
- 再現困難なランダムクラッシュ

### 重要度

**High**

### 修正方針

- `stateSnapshot` は RCU / epoch / deferred deletion で保護する
- オーディオスレッドでは acquire したスナップショットのみ参照する
- 非 RT スレッドは即座に delete せず、grace period 後に delete する

---

## A6. `diagLog()` / 静的変数付き診断コードがオーディオスレッドで呼ばれる可能性

### 該当箇所

```cpp
if ((s_blockTickCount % 100) == 0)
{
    ...
    diagLog(diagPrefix(gen) + " [BLOCK_TIMING] tick=" + ...);
}
```

`diagLog()` は前回指摘の通り、内部で以下を実行します。

```cpp
DBG(message);
juce::Logger::writeToLog(message);
```

### 問題

このブロックタイミング診断がオーディオスレッドで実行される場合、以下の規約違反になります。

- ファイル I/O 禁止
- コンソール出力禁止
- ロックの可能性
- メモリアロケーションの可能性
- リアルタイム性破壊

さらに `s_blockTickCount` などが static local 場合、複数インスタンスや複数スレッドで競合します。

### 影響

- XRUN
- 音声途切れ
- 計測自体が原因のスパイク
- デバッグビルドで問題が隠れ、Release で出る可能性

### 重要度

**High**

### 修正方針

オーディオスレッドでは、ロックフリーリングバッファにイベントを書き込むだけにして、実際のログ出力は非 RT スレッドで行うべきです。

---

## A7. `releaseResources()` 内の `flushPendingEpochAdvance()` が削除処理を伴うと RT 規約違反の疑い

### 該当箇所

```cpp
flushPendingEpochAdvance();
juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: end");
```

### 問題

`flushPendingEpochAdvance()` が保留中の epoch 進め、つまり旧 DSP / 旧バッファの破棄を伴う場合、それがオーディオスレッドから呼ばれるとコーディング規約違反です。

規約では、使用済みメモリの free を RT スレッドで行うのは厳禁とされています。

### 影響

- `mkl_free`
- `_aligned_free`
- デストラクタ実行
- メモリ返却

がオーディオスレッドで走り、音声停止やスパイクの原因になります。

### 重要度

**High / Medium**

### 修正方針

- `releaseResources()` がメッセージスレッド専用であることを保証する
- または epoch advance による破棄は DeferredDeletionQueue / 返却用 FIFO 経由で非 RT スレッドに移動する

---

## A8. `LoaderThread` の例外ハンドリングでエラー状態が完了状態として伝播しない可能性

### 該当箇所

```cpp
try
{
    while (true)
    {
        const bool terminal = stepOnce();
        if (terminal) break;
    }
}
catch (const std::bad_alloc&)
{
    stepResult.errorMessage = "IR too large (Out of Memory)";
    juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
}
catch (const std::exception& e)
{
    stepResult.errorMessage = "Error loading IR: " + juce::String(e.what());
    juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
}
catch (...)
```

### 問題

例外時に `errorMessage` を設定していますが、抽出コードからは以下が不明です。

- `stepResult.success = false` を設定しているか
- `stepResult.finished = true` を設定しているか
- 待機側に通知しているか
- ローダー状態を terminal にしているか

もしこれらが欠けていると、UI 側が永遠にロード中になる可能性があります。

### 影響

- IR 読み込み失敗後に UI が固まる
- 進捗が完了しない
- 再読み込み不能
- ハング

### 重要度

**Medium / High**

### 修正方針

```cpp
catch (...)
{
    stepResult.success = false;
    stepResult.finished = true;
    stepResult.errorMessage = "Unknown error while loading IR";
    juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
}
```

---

# 追加バグ候補：中優先度

---

## B1. `FailureRecorder` で `FailureReason::Count` 境界チェックがない可能性

### 該当箇所

```cpp
auto& bucket = buckets_[static_cast<size_t>(reason)];
```

### 問題

`reason` が `FailureReason::Count` 以上、または不正値の場合、配列外アクセスになります。

### 影響

- クラッシュ
- メモリ破損
- 診断データ破損

### 重要度

**Medium**

### 修正例

```cpp
if (reason >= FailureReason::Count)
    reason = FailureReason::Count - 1; // または Unknown 扱い
```

または専用 Unknown バケットを用意します。

---

## B2. MMCSS エラーコード `1552` を成功扱いにしている危険性

### 該当箇所

```cpp
// 1552(ERROR_NO_MORE_ITEMS): MSDN未定義だが、...
// このケースは ASIO ドライバが自前で MMCSS 登録済みの環境で発生する。
if (err == ...)
```

### 問題

コメント自体が「MSDN未定義」に近い扱いをしています。  
未文書化・環境依存のエラーコードを成功として扱うと、本来の失敗を隠す可能性があります。

### 影響

- MMCSS 未登録なのに登録済みと誤認
- 音声スレッド優先度が上がらない
- XRUN が増える
- 環境依存の不具合

### 重要度

**Medium**

### 修正方針

- 可能なら実際に MMCSS に登録されているかを別 API で確認する
- 少なくとも「成功」ではなく「警告付き継続」にする
- 診断ログに残す
- 特定ドライバ環境でのみ許可する

---

## B3. `GetLastError()` が stale な可能性

### 該当箇所

```cpp
const DWORD err = ::GetLastError();
```

### 問題

`AvSetMmThreadCharacteristics()` 呼び出し前に `SetLastError(0)` していない場合、直前のエラーコードを拾う可能性があります。

### 影響

- 失敗しているのに成功扱い
- 成功しているのに失敗扱い

### 重要度

**Low / Medium**

### 修正例

```cpp
::SetLastError(0);
HANDLE h = ::AvSetMmThreadCharacteristicsW(task, &taskIndex);
const DWORD err = ::GetLastError();
```

---

## B4. `build.bat` の `-D` 引数処理がユーザー指定値を壊す可能性

### 該当箇所

```bat
if "!arg:~0,2!"=="-D" (
    REM cmd.exe strips =VALUE, so append =ON.
    set "CMAKE_EXTRA_FLAGS=!CMAKE_EXTRA_FLAGS! !arg!=ON"
    echo [INFO] Extra CMake define: !arg!=ON
)
```

### 問題

ユーザーが以下のように指定した場合、

```bat
build.bat -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF
```

意図せず次のようになる可能性があります。

```bat
-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF=ON
```

またはコメントの意図次第では：

```bat
-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON
```

になり、`OFF` が無視されます。

### 影響

- CMake オプションが意図せず ON になる
- 診断ログが有効化される
- Release ビルドが重くなる
- ビルド設定の再現性が壊れる

### 重要度

**Medium**

### 修正方針

`-DKEY=VALUE` をそのまま渡すようにすべきです。

```bat
if "!arg:~0,2!"=="-D" (
    set "CMAKE_EXTRA_FLAGS=!CMAKE_EXTRA_FLAGS! %%~A"
)
```

---

## B5. `build.bat` が Visual Studio 2022 ではなく VS 18 路径を探している可能性

### 該当箇所

```bat
if exist "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat" ...
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" ...
```

### 問題

コーディング規約では Visual Studio 2022 / MSVC v143 が指定されています。  
もし `17` ではなく `18` のみを探している場合、VS2022 環境で `vcvarsall.bat` を見つけられない可能性があります。

### 影響

- ビルド環境初期化失敗
- Windows SDK が見つからない
- Ninja / MSVC ビルド失敗

### 重要度

**Medium**

### 修正方針

VS2022 の `17` も検索対象に含めるべきです。

```bat
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" ...
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" ...
```

または `vswhere.exe` を使う方が安全です。

---

## B6. CMake の ASan 用設定が静的 CRT 方針と矛盾している可能性

### 該当箇所

```cmake
set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
    target_link_options(ConvoPeq PRIVATE -fsanitize=address)
endif()
```

### 問題

通常ビルドでは静的 CRT が使われている一方、ASan 有効時は DLL CRT になる可能性があります。  
また IntelLLVM に対する `-fsanitize=address` が Windows icx-cl 環境で完全にサポートされているかも注意が必要です。

### 影響

- CRT 不整合
- リンクエラー
- 実行時ヒープ破損
- ASan 誤検出または検出不能

### 重要度

**Medium**

### 修正方針

- ASan 用ビルドは専用構成として分離する
- 静的 CRT と ASan の互換性を明確にする
- IntelLLVM ASan はサポート環境を限定する
- 通常ビルドに ASan 設定が混入しないようにする

---

## B7. IPP が見つからない場合でも `ippInit()` が無条件に呼ばれる可能性

### 該当箇所

CMake：

```cmake
find_package(IPP QUIET CONFIG COMPONENTS ippcore ipps)

if(IPP_FOUND)
    message(STATUS "Intel IPP found.")
    target_link_libraries(ConvoPeq PRIVATE IPP::ippcore IPP::ipps)
else()
    message(STATUS "Intel IPP not found - skipping (CI build or no IPP SDK).")
endif()
```

MainApplication：

```cpp
const IppStatus ippSt = ippInit();
```

### 問題

ソース側が `IPP_FOUND` 相当のマクロでガードされていない場合、IPP なし CI ビルドでコンパイルエラーまたはリンクエラーになります。

### 影響

- CI ビルド失敗
- IPP がない環境でビルド不能
- `ippInit` 未解決シンボル

### 重要度

**Medium**

### 修正方針

CMake で定義を追加します。

```cmake
if(IPP_FOUND)
    target_compile_definitions(ConvoPeq PRIVATE CONVOPEQ_HAS_IPP=1)
endif()
```

ソース側：

```cpp
#if CONVOPEQ_HAS_IPP
    const IppStatus ippSt = ippInit();
    ...
#endif
```

---

## B8. JSON 出力で文字列エスケープが不足している可能性

### 該当箇所

```cpp
file << "\"scenario_failed:" << scenario.name << "\"";
...
file << "{\"name\": \"" << scenario.name << "\", \"result\": \"" << ...
```

### 問題

`scenario.name` や `result` に以下が含まれると JSON が壊れます。

- `"`
- `\`
- 改行
- 制御文字

### 影響

- テスト結果 JSON が解析不能
- CI 連携失敗
- evidence 破損

### 重要度

**Medium**

### 修正方針

最低限のエスケープ関数を用意します。

```cpp
juce::String escapeJsonString(const juce::String& s)
{
    juce::String out;

    for (auto c : s)
    {
        switch (c)
        {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:   out << c;      break;
        }
    }

    return out;
}
```

---

## B9. `aligned_free()` が DIAG 追跡を迂回する

### 該当箇所

```cpp
inline void aligned_free(void* ptr) noexcept
{
    if (ptr != nullptr)
    {
        // allocation tracking は DIAG_MKL_MALLOC 側で行うため、解放トラッキングは省略する
        mkl_free(ptr);
    }
}
```

### 問題

`DIAG_MKL_MALLOC` でメモリ使用量を追跡しているのに、解放側が `mkl_free` 直接呼び出しの場合、診断上の `allocatedBytes` が単調増加します。

コメントにも「単調増加傾向を示す」とあります。

### 影響

- メモリリーク誤検出
- 診断情報の信頼性低下
- 実際のリークと区別不能

### 重要度

**Medium**

### 修正方針

解放サイズを管理するか、アロケーションヘッダにサイズを持たせて `DIAG_MKL_FREE` を呼ぶべきです。

---

## B10. `freeTracked()` に渡す `size` が実際の確保サイズと一致しない可能性

### 該当箇所

```cpp
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        if (size > 0)
```

### 問題

呼び出し側が確保時と異なる `size` を渡すと、メモリ追跡が壊れます。

### 影響

- 診断メモリ統計の不整合
- リーク誤検出
- 二重解放検出の精度低下

### 重要度

**Medium**

### 修正方針

- 確保時にサイズを記録する
- ポインタからサイズを引けるようにする
- `freeTracked(p)` のようにサイズ不要 API にする

---

## B11. TruePeakDetector のハーフバンド係数正規化で符号反転 / 巨大化の疑い

### 該当箇所

```cpp
double sum = 0.0;
for (int i = 0; i < stage.taps; ++i)
    sum += rawCoeffs[i];

if (std::abs(sum) > 1.0e-20)
{
    const double inv = 1.0 / sum;
    for (int i = 0; i < stage.taps; ++i)
        rawCoeffs[i] *= inv;
}

rawCoeffs[stage.centerTap] = 0.5;

double nonCenterSum = 0.0;
for (int i = 0; i < stage.taps; ++i)
    if (i != stage.centerTap)
        nonCenterSum += rawCoeffs[i];

if (std::abs(nonCenterSum) > 1.0e-20)
{
    const double scale = 0.5 / nonCenterSum;
    for (int i = 0; i < stage.taps; ++i)
        if (i != stage.centerTap)
            rawCoeffs[i] *= scale;
}
```

### 問題

`nonCenterSum` が極小、または負の場合、`scale` が巨大化したり負になったりします。

特に：

```cpp
scale = 0.5 / nonCenterSum;
```

は、`nonCenterSum` が `-1e-12` のときに `-5e11` になります。

### 影響

- フィルター係数が発散
- 異常なピーク
- 過大出力
- TruePeak 検出不能

### 重要度

**Medium**

### 修正方針

```cpp
if (nonCenterSum > 1.0e-6)
{
    const double scale = 0.5 / nonCenterSum;
    ...
}
else
{
    // 安全な既定係数へフォールバック
}
```

---

## B12. `AllpassDesigner` で `sampleRate <= 0`、配列サイズ不一致の防御が不足している可能性

### 該当箇所

```cpp
const double theta = 2.0 * juce::MathConstants<double>::pi * freqHz / sampleRate;
```

および：

```cpp
for (int i = 0; i < config.numSections; ++i)
```

```cpp
weightedSquaredError += weight[i] * diff * diff;
```

### 問題

以下の場合に危険です。

- `sampleRate <= 0`
- `freq_hz.size() != target_group_delay_samples.size()`
- `weight.size() != freq_hz.size()`
- `config.numSections <= 0`

### 影響

- 0 除算
- 配列越境
- NaN 発生
- CMA-ES 最適化の破綻

### 重要度

**Medium**

### 修正方針

```cpp
if (sampleRate <= 0.0)
    return DesignResult::makeFailure("Invalid sample rate");

if (freq_hz.size() != target_group_delay_samples.size())
    return DesignResult::makeFailure("Size mismatch");

if (weight.size() != freq_hz.size())
    return DesignResult::makeFailure("Weight size mismatch");
```

---

## B13. `stableSigmoid01()` が `std::exp()` を使うため、RT パスで呼ばれると規約違反

### 該当箇所

```cpp
inline double stableSigmoid01(double x) noexcept
{
    x = std::clamp(x, -50.0, 50.0);

    if (x >= 0.0)
    {
        const double expNegX = std::exp(-x);
        return 1.0 / (1.0 + expNegX);
    }

    const double expX = std::exp(x);
    return expX / (1.0 + expX);
}
```

### 問題

`std::exp()` は libm 呼び出しになる可能性があります。  
コーディング規約では、オーディオスレッド内で libm 呼び出しとなる関数を避けることになっています。

### 影響

- オーディオスレッドで呼ばれると XRUN
- 処理時間の非決定性

### 重要度

**Medium**  
※呼び出し箇所が非 RT なら問題ありません。

### 確認項目

- `AllpassDesigner::designWithCMAES()` がオーディオスレッドから呼ばれていないか
- UI タイマーから呼ばれていないか
- リビルド専用スレッドか

---

## B14. `UltraHighRateDCBlocker` の状態初期化が不完全な疑い

### 該当箇所

```cpp
m_state[0] = isFiniteAndBelowThresholdMask(state0, 1.0e15) ? state0 : 0.0;
m_state[1] = isFiniteAndBelowThresholdMask(state1, 1.0e15) ? state1 : 0.0;
```

### 問題

抽出コードでは、`process()` 冒頭で `state0` / `state1` が `m_state[0]` / `m_state[1]` から正しく読み込まれているか確認できません。

もしローカル変数が未初期化なら、未定義動作です。

### 影響

- 初回ブロックでノイズ
- 未初期化メモリ使用
- 発散

### 重要度

**Medium**

### 確認項目

```cpp
double state0 = m_state[0];
double state1 = m_state[1];
```

が `process()` 冒頭にあるか確認してください。

---

## B15. FFT Overlap-Save 処理で `partSize` がバッファ容量を超えると越境

### 該当箇所

```cpp
juce::FloatVectorOperations::copy(l.fftTimeBuf, l.prevInputBuf, l.partSize);
juce::FloatVectorOperations::copy(l.fftTimeBuf + l.partSize, l.inputAccBuf, l.partSize);
juce::FloatVectorOperations::copy(l.prevInputBuf, l.inputAccBuf, l.partSize);
```

### 問題

`l.fftTimeBuf` が最低 `2 * partSize` 要素、`prevInputBuf` と `inputAccBuf` が最低 `partSize` 要素持っている必要があります。

容量検証が不足していると越境します。

### 影響

- ヒープ破損
- クラッシュ
- FFT 入力破損

### 重要度

**Medium**

### 修正方針

```cpp
jassert(l.fftTimeBuf != nullptr);
jassert(l.prevInputBuf != nullptr);
jassert(l.inputAccBuf != nullptr);
jassert(l.fftTimeBufCapacity >= 2 * l.partSize);
jassert(l.prevInputBufCapacity >= l.partSize);
jassert(l.inputAccBufCapacity >= l.partSize);
```

---

## B16. `errorEnvelope` が NaN の場合、閾値比較が常に false になる可能性

### 該当箇所

```cpp
if (errorEnvelope > kErrorStateThreshold)
    convo::publishAtomic(needsReset, true, std::memory_order_release);

errorEnvelope = 0.0;
```

### 問題

`errorEnvelope` が NaN になると、

```cpp
NaN > threshold
```

は false です。

つまりエラー状態でリセットされない可能性があります。

### 影響

- 異常状態が回復不能
- NaN が伝播
- 無音または破綻

### 重要度

**Medium**

### 修正例

```cpp
if (!std::isfinite(errorEnvelope) || errorEnvelope > kErrorStateThreshold)
    convo::publishAtomic(needsReset, true, std::memory_order_release);
```

---

## B17. `xxh64Digest()` が端数バイトを処理していない可能性

### 該当箇所

```cpp
while (...)
{
    h ^= static_cast<uint64_t>(*p) * kPrime5;
    h = rotl64(h, 11) * kPrime1;
    ++p;
}
return xxh64Avalanche(h);
```

### 問題

抽出コードだけでは、8 バイト未満の端数バイトを処理しているか不明です。  
もし端数処理が欠けていると、ファイル末尾の違いがハッシュに反映されません。

### 影響

- 異なる IR ファイルを同一と誤認
- キャッシュ汚染
- 古い IR が使われる

### 重要度

**Medium**

### 確認項目

- `size % 8` の処理があるか
- 1 バイトずつ処理する fallback があるか

---

# 追加バグ候補：低〜中優先度

---

## C1. `pushDiagnosticErrors()` の `errorWritePos` 更新が fetch_add ではない

### 該当箇所

```cpp
const uint32_t nextPos =
    (convo::consumeAtomic(errorWritePos, std::memory_order_acquire)
     + static_cast<uint32_t>(written)) & (kDiagnosticsCapacity - 1u);

convo::publishAtomic(errorWritePos, nextPos, std::memory_order_release);
```

### 問題

単一オーディオスレッド専用なら問題ありませんが、複数プロデューサから呼ばれると lost update になります。

### 影響

- 診断 FIFO 位置不整合
- 診断データ欠損

### 重要度

**Low**

### 修正方針

単一プロデューサであることを static assert / コメントで保証するか、`fetch_add` を使います。

---

## C2. `kDiagnosticsCapacity` が 2 の累乗である保証が見えない

### 該当箇所

```cpp
& (kDiagnosticsCapacity - 1u)
```

### 問題

このマスクは `kDiagnosticsCapacity` が 2 の累乗でないと正しくありません。

### 影響

- リングバッファインデックス破損

### 重要度

**Low**

### 修正例

```cpp
static_assert((kDiagnosticsCapacity & (kDiagnosticsCapacity - 1u)) == 0,
              "kDiagnosticsCapacity must be power of two");
```

---

## C3. `rotl64()` のシフト量が未検証なら未定義動作

### 該当箇所

```cpp
inline uint64_t rotl64(uint64_t value, int count) noexcept
```

### 問題

`count` が 0 未満、または 63 超の場合、未定義動作です。

### 影響

- ハッシュ計算破損
- 最適化による奇妙な挙動

### 重要度

**Low**

### 修正例

```cpp
count &= 63;
```

---

## C4. `getState(int)` の境界チェックがない可能性

### 該当箇所

```cpp
double getState(int
```

### 問題

`index < 0` や `index >= 2` の場合に越境読み込みの可能性があります。

### 影響

- デバッグ用 API でのクラッシュ

### 重要度

**Low**

### 修正例

```cpp
if (index < 0 || index >= 2)
    return 0.0;
```

---

## C5. CLI の dither bit depth に範囲チェックがない可能性

### 該当箇所

```cpp
int postLoadBitDepth = 0;
if (tryParseIntOption(postLoadDitherValue, postLoadBitDepth))
{
    ...
}
```

### 問題

`postLoadBitDepth` に負値や巨大値が入っても、そのまま使われる可能性があります。

### 影響

- 不正なディザー設定
- 無音
- 量子化異常

### 重要度

**Low / Medium**

### 修正例

```cpp
if (postLoadBitDepth != 0 && postLoadBitDepth != 16 && postLoadBitDepth != 24 && postLoadBitDepth != 32)
{
    juce::Logger::writeToLog("[CLI] Invalid dither bit depth ignored: " + juce::String(postLoadBitDepth));
    return;
}
```

---

## C6. `--cli-rebuild` の 500ms 固定遅延が IR 読み込み完了に間に合わない可能性

### 該当箇所

```cpp
// Must fire after deferred IR load (200ms) to include IR in rebuild
const int rebuildDelayMs = 500;
```

### 問題

IR 読み込みが 500ms 以内に完了しない環境では、リビルドが IR 読み込み前に発火します。

### 影響

- IR がリビルドに含まれない
- 音が更新されない
- CLI テストが不安定

### 重要度

**Medium**

### 修正方針

固定時間ではなく、IR ロード完了イベントを待つべきです。

---

## C7. `latencySnapshotChanged` 比較用の last 変数が未初期化の可能性

### 該当箇所

```cpp
!= lastLatencySamples
|| latencyMsX10 != lastLatencyMsX10
|| latencySrValid != lastLatencySrValid;
```

### 問題

`lastLatencySamples` などが初期化されていない場合、初回更新判定が不定になります。

### 影響

- 初回 UI 更新漏れ
- 不要な更新

### 重要度

**Low**

### 修正例

```cpp
int lastLatencySamples = 0;
int lastLatencyMsX10 = 0;
bool lastLatencySrValid = false;
```

---

## C8. `processBlock()` で `MAX_CHANNELS` を超えるチャンネルが未処理のまま残る可能性

### 該当箇所

```cpp
const int numChannels = std::min((int)block.getNumChannels(), MAX_CHANNELS);
```

### 問題

ブロックが `MAX_CHANNELS` より多いチャンネルを持つ場合、超過チャンネルは処理されません。

### 影響

- 多チャンネル環境で無処理チャンネルが残る
- 無音または旧データ

### 重要度

**Low**  
※ステレオ専用なら仕様として許容できます。

### 修正方針

- ステレオ専用であることを明示する
- または超過チャンネルをクリアする

---

## C9. CMake テストにベンチマークが含まれており CI が不安定になる可能性

### 該当箇所

```cmake
add_test(NAME EQBoundExcessBenchmark COMMAND EQBoundExcessBenchmark --quick)
add_test(NAME MTNUPCMeasurement COMMAND MTNUPCMeasurement)
```

### 問題

ベンチマークや測定を CTest に含めると、実行時間やマシン性能によって不安定になります。

### 影響

- CI 時間増大
- 環境依存の失敗
- テスト結果の揺れ

### 重要度

**Low**

### 修正方針

- ベンチマークは `ctest -L benchmark` のようにラベル分離する
- 既定のテストから外す

---

## C10. `HeadlessAudioPathVerification` が既存プロセスを kill する危険性

### 該当箇所

```cmake
add_test(NAME HeadlessAudioPathVerification
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File ... -KillExisting -RequireAudioCallbacks)
```

### 問題

`-KillExisting` により、開発中の ConvoPeq プロセスを強制終了する可能性があります。

### 影響

- 開発者の作業中データ損失
- デバッグセッション破壊

### 重要度

**Low / Medium**

### 修正方針

- CI 専用フラグにする
- 開発環境では既定で無効にする
- プロセス名とパスを検証してから kill する

---

# 特に優先して確認すべき追加項目

前回指摘分と合わせて、特に優先度が高いのは以下です。

1. **A1：IR キャッシュの `size_t` アンダーフロー**
2. **A2：キャッシュヘッダの `int` キャスト未検証**
3. **A3：`isBadSampleV()` と `isBadSample()` の閾値不一致**
4. **A4：TruePeakDetector の履歴 / 出力バッファ越境**
5. **A5：`stateSnapshot` の use-after-free 疑い**
6. **A6：オーディオスレッド内 diagLog / 静的変数**
7. **A7：`flushPendingEpochAdvance()` の RT 破棄疑い**
8. **A8：LoaderThread 例外時の完了状態未設定疑い**
9. **B4：`build.bat -D` 引数破壊**
10. **B7：IPP 未検出時の `ippInit()` 無条件呼び出し疑い**

---

# 推奨される追加テスト

## 1. 破損 IR キャッシュテスト

以下を含むファズファイルを生成してください。

- `mmap.getSize() < headerSize`
- `dataSize` が巨大
- `timeDomainChannels = 0`
- `timeDomainChannels = 0xFFFFFFFF`
- `timeDomainNumSamples = 0xFFFFFFFF`
- `timeDomainSizeBytes` が不整合
- 端数バイトだけ短いファイル

期待結果：

- クラッシュしない
- false を返す
- フォールバック読み込みに移行する

---

## 2. `isBadSample` / `isBadSampleV` 一致テスト

```cpp
for (double v : testValues)
{
    EXPECT_EQ(isBadSample(v), isBadSampleV(_mm256_set1_pd(v)));
}
```

テスト値：

```text
0.0
-0.0
1.0
-1.0
1e15
9.0e15
9.1e15
1e16
1e20
1e21
NaN
+Inf
-Inf
DBL_MAX
```

---

## 3. TruePeakDetector 境界テスト

- `numSamples = 0`
- `numSamples = 1`
- `numSamples = 2`
- `numSamples = 3`
- `numSamples = 4`
- `numSamples = maxInternalBlockSize`
- `numSamples = maxInternalBlockSize + 1`

確認：

- `output` が `2 * numSamples` 分あるか
- `history` が負インデックスを参照しないか
- ASan でクリーンか

---

## 4. MMCSS 登録失敗テスト

- すでに別 MMCSS タスクに所属しているスレッド
- 権限なし
- タスク名が存在しない
- ASIO ドライバが MMCSS 登録済み

確認：

- 誤って成功扱いにならないか
- 優先度が実際に上がっているか
- ログが過剰になりすぎていないか

---

## 5. `build.bat -D` テスト

```bat
build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF
```

確認：

- CMakeCache に `OFF` が入るか
- `ON` に書き換わらないか

---

# 総合評価

追加資料から見えてきた主なリスクは次の 4 系統です。

1. **ファイル / キャッシュパースの境界検査不足**
2. **SIMD 版とスカラ版の挙動不一致**
3. **オーディオスレッド内で診断・削除・例外・ログが混入する可能性**
4. **ビルドスクリプト / CMake 設定の矛盾**

中でも、IR キャッシュ読み込みの `size_t` アンダーフローと、ヘッダ値の未検証 `int` キャストは、クラッシュやセキュリティ問題に直結し得るため、最優先で修正すべきです。


ConvoPeq ソースコードの徹底的な調査に基づき、以下のバグ・問題点を報告します。

---

## 1. メモリ管理・リソース解放の問題

### 1-1. `MKLNonUniformConvolver::releaseAllLayers()` における二重解放リスク (MKLNonUniformConvolver.cpp:1950)
- **問題**: `releaseAllLayers()` 内で `m_layers[i].freeAll()` を呼び出した後、NUCレベルバッファ (`m_ringBuf`, `m_directIRRev` 等) を解放している。`freeAll()` は `Layer::allocSizes` を使用して `freeTracked()` を呼び出すが、`freeTracked()` はポインタを `nullptr` にセットする。その後、NUCレベルバッファの解放でも同じポインタを解放しようとする可能性はない。しかし、`m_directIRRev` 等は `freeAll()` では解放されないため、重複はない。ただし、`freeAll()` 内で `m_directIRRev` を解放していないので、`releaseAllLayers()` が２回呼ばれた場合、`m_directIRRev` は二重解放される可能性がある（最初の呼び出しで `nullptr` になり、２回目で `freeTracked()` は `ptr==nullptr` をチェックするが、`freeTracked()` は `if (p)` でガードしているので問題ない）。

### 1-2. `ConvolverProcessor::applyComputedIR()` の `updateConvolverState()` 呼び出しと `IRState` の寿命 (ConvolverProcessor.StateAndUI.cpp:1210)
- **問題**: `applyComputedIR()` 内で `updateIRState()` を呼び出し、その後 `updateConvolverState()` を呼び出す。`updateIRState()` は新しい `IRState` を作成し、古い `IRState` を `exchangeAtomic` で交換する。しかし、`updateConvolverState()` は `ConvolverState` を交換するが、`ConvolverState` は `IRState` への参照を持たない。そのため、`IRState` の寿命は `ConvolverState` とは独立して管理される。もし `IRState` が古い `ConvolverState` より先に解放されると、`ConvolverState` が参照している `IRState` が無効になる可能性がある。しかし、`ConvolverState` は `IRState` を参照していないので問題ない。

### 1-3. `CacheManager::loadPreparedState()` の `mkl_malloc` 失敗時の処理 (CacheManager.cpp:116)
- **問題**: `DIAG_MKL_MALLOC` が失敗した場合、`nullptr` を返すが、呼び出し元（`CacheManager::loadPreparedState`）は `if (!copied) return nullptr;` でチェックしている。しかし、その後に `std::memcpy` を使用しており、`copied` が `nullptr` の場合は未定義動作となる。`if (!copied)` で早期リターンしているので問題ない。

### 1-4. `AlignedAllocation.h` の `aligned_malloc_nothrow` と `makeAlignedArray_nothrow` (AlignedAllocation.h:41)
- **問題**: `makeAlignedArray_nothrow` は `aligned_malloc_nothrow` を呼び出し、失敗時に `nullptr` を内包した `ScopedAlignedArray` を返す。しかし、呼び出し元が `nullptr` をチェックせずに使用する可能性がある。コードベース全体で `makeAlignedArray_nothrow` の使用箇所を確認する必要がある。現時点では `CmaEsOptimizerDynamic.cpp` などでの使用は確認されていないが、将来的な使用に注意。

### 1-5. `ProgressiveUpgradeThread::upgradeStep()` の `prepared` ポインタ管理 (ProgressiveUpgradeThread.cpp:72)
- **問題**: `prepared = converter.convertToHighRes(...)` の後、`if (!prepared) return false;` でチェックしている。その後、`cacheManager.save()` を呼び出し、最後に `preparedRaw = prepared.release()` して `callAsync` に渡している。もし `cacheManager.save()` が例外を投げた場合、`prepared` は解放されずにリークする。`cacheManager.save()` は例外を投げないが、安全のため `std::unique_ptr` で管理するべき。

---

## 2. スレッド安全性・同期の問題

### 2-1. `AudioEngine::processBlockDouble()` の `dspCrossfadeDoubleBuffer` のサイズ確認 (AudioEngine.Processing.BlockDouble.cpp:145)
- **問題**: `dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples` のチェックがあるが、`dspCrossfadeDoubleBuffer` は `prepareToPlay` で確保される。しかし、`prepareToPlay` と `releaseResources` の間に `maxSamplesPerBlock` が変更される可能性はないため、問題ない。

### 2-2. `ConvolverProcessor::process()` の `mixSmootherResetPendingGen` と `smoothingTimeChangePendingGen` の扱い (ConvolverProcessor.Runtime.cpp:190)
- **問題**: `mixSmootherResetPendingGen` と `smoothingTimeChangePendingGen` は `NonRT` から `fetch_add` され、`AudioThread` で `consume` される。しかし、`process()` 内でこれらをチェックした後、`applyImmediateValueRT` や `reset` を呼び出している。もし `NonRT` が複数回更新した場合、`RT` が一度の処理で複数の変更を検出できるか？`curGen` と `m_` の比較で、`curGen != m_` の場合にのみ処理するため、もし `NonRT` が２回更新した場合、`RT` は１回目の更新と２回目の更新をまとめて処理する（１回の `process` 呼び出しで `curGen` が２つ進んでいるため、`if` ブロック内で最新の値のみを適用する）。これは意図された動作かもしれないが、もし間に別の更新が挟まった場合、中間の値がスキップされる。

### 2-3. `AudioEngine::tryApplyMmcssForSelfManagedThread()` の `thread_local` 変数 (AudioEngine.Mmcss.cpp:65)
- **問題**: `t_mmcssHandle` と `t_mmcssTried` は `thread_local` で宣言されている。`tryApplyMmcssForSelfManagedThread()` は `AudioThread` のコールバック内で呼ばれる。`thread_local` はスレッドごとに独立しているため、`AudioThread` が複数存在する場合（例：ASIOドライバが複数のコールバックスレッドを生成する場合）、各スレッドで独立にMMCSS登録が試行される。これは問題ないが、`t_mmcssTried` が `false` のまま再試行されることはない。

### 2-4. `ConvolverProcessor::updateConvolverState()` の `writerActive` CAS (ConvolverProcessor.StateAndUI.cpp:1840)
- **問題**: `writerActive` を `false` から `true` にCASで排他制御している。もしCASに失敗した場合、`newState` を破棄している。しかし、`newState` は `std::unique_ptr` で渡されている場合もあれば、生ポインタで渡されている場合もある。`updateConvolverState(ConvolverState* newState)` のオーバーロードでは、CAS失敗時に `std::unique_ptr<ConvolverState> discard{newState};` で破棄している。これは安全。

### 2-5. `DeferredDeletionQueue::reclaim()` の先頭ブロッキング (DeferredDeletionQueue.h:123)
- **問題**: `reclaim()` は先頭エントリが `canDelete` でない場合、即座に `break` してループを抜ける。そのため、先頭が削除不可能なエントリでブロックされると、後続の削除可能なエントリが永遠に処理されない。これは設計上のトレードオフ（FIFO順序の維持）だが、キューが長期間ブロックされる可能性がある。`kMaxScan` ループ上限はあるが、先頭が削除不可能な場合、１回の `reclaim` 呼び出しで何も処理されずに終了する。そのため、`reclaim` を呼ぶスレッドが定期的に実行されていれば問題ないが、もし呼び出し頻度が低いと遅延が発生する。

---

## 3. 数値安定性・未定義動作

### 3-1. `EQProcessor::processBand()` の NaN/Inf チェック (EQProcessor.Processing.cpp:65)
- **問題**: `isFiniteAndAbsInRangeMask` を使用してNaN/Infをチェックしているが、この関数は `_mm_cmpeq_pd` を使用してNaNを検出している。NaNは自分自身と等しくないため、`_mm_cmpeq_pd` は `false` を返す。これにより、NaNは `validMask` で除外され、`output` は `0.0` に設定される。これは正しい。ただし、`state` 変数も同様にチェックしている。

### 3-2. `LatticeNoiseShaper::processSample()` の `advanceState()` (LatticeNoiseShaper.h:229)
- **問題**: `advanceState()` 内で `state[i] = std::clamp(nextBackward, -kLatticeStateLimit, kLatticeStateLimit);` としている。`kLatticeStateLimit` は `2.0` に設定されている。しかし、格子フィルタの反射係数が `0.85` に制限されているため、`nextBackward` が発散する可能性は低いが、もしクリッピングが発生すると非線形歪みが生じる。これは許容範囲かもしれない。

### 3-3. `FixedNoiseShaper::quantize()` の `replaceNonFiniteWithZero` (FixedNoiseShaper.h:231)
- **問題**: `quantize()` 内で `v = replaceNonFiniteWithZero(v);` を呼び出し、その後クランプとディザを適用している。`replaceNonFiniteWithZero` はNaN/Infを `0.0` に変換する。これは安全だが、もし入力が非常に大きな値の場合、`std::clamp` でクランプされる前にディザが加算される。ディザは `scale` 程度の小さな値なので、大きな値には影響しない。

### 3-4. `PsychoacousticDither::processStereoBlock()` の `killDenormal` (PsychoacousticDither.h:360)
- **問題**: `killDenormal` は `FTZ/DAZ` が有効な場合は何もしない。リリースビルドでは `#if !defined(JUCE_DEBUG)` で最適化され、デバッグビルドでのみチェックする。これはパフォーマンス向上のためだが、デバッグビルドでデノーマルが発生した場合に正しく処理される。

### 3-5. `MKLNonUniformConvolver::processLayerBlock()` のデノーマル除去 (MKLNonUniformConvolver.cpp:1580)
- **問題**: `killDenormalV` を使用して `accumBuf` のデノーマルを除去しているが、リリースビルドでは何もしない。`FTZ/DAZ` が有効であることを前提としている。

---

## 4. エラーハンドリング・ロジックの問題

### 4-1. `ConvolverProcessor::loadImpulseResponse()` の `isRebuild` 判定 (ConvolverProcessor.LoadPipeline.cpp:39)
- **問題**: `isRebuild = (irFile == juce::File());` としている。`juce::File` のデフォルトコンストラクタは空のファイルを表す。しかし、`irFile` が存在しないファイルを指定した場合も `isRebuild` が `false` になるため、`loadImpulseResponse` が実行される。その後、`irFile.existsAsFile()` でチェックされ、`false` を返す。これは問題ないが、意図しない動作の可能性がある。

### 4-2. `AudioEngine::requestRebuild()` の `forceMustExecute` パラメータ (AudioEngine.RebuildDispatch.cpp:303)
- **問題**: `forceMustExecute` が `true` の場合、重複抑制を無視して rebuild を強制する。しかし、`submitRebuildIntent` の `collapsePolicy` に `MustExecute` を設定し、`requestRebuild` 内で `allowDuplicateSuppression` を `false` にしている。しかし、`submitRebuildIntent` 内で `sameAsPendingWouldMerge` の判定に `collapsePolicy` が使われていないため、`MustExecute` でも merge が発生する可能性がある。実際には `requestRebuild` 内で `blockedAsDuplicate` 判定に `forceMustExecute` を使っているので、`MustExecute` の場合は pending task があっても置き換えられる。これは正しい。

### 4-3. `ConvolverProcessor::getState()` の `irPath` 保存 (ConvolverProcessor.StateAndUI.cpp:1250)
- **問題**: `irPath` を保存する際、`currentIrFile.getFullPathName()` を使用している。このパスは絶対パスであり、ユーザーがファイルを移動した場合、次回起動時にファイルが見つからなくなる。しかし、エラーメッセージで「Click to locate...」と表示されるので、ユーザーが再選択できるようになっている。

### 4-4. `AudioEngine::setConvolverStateTree()` の状態検証 (AudioEngine.Parameters.cpp:720)
- **問題**: `validateConvolverStateTreeForDebug` でデバッグ時のみ検証しているが、リリースビルドでは検証されない。そのため、不正な状態が設定される可能性がある。しかし、`setState` 内で値がクランプされるため、致命的な問題にはならない。

### 4-5. `NoiseShaperLearner::workerThreadMain()` のループ終了条件 (NoiseShaperLearner.cpp:480)
- **問題**: ループ内で `if (convo::consumeAtomic(stopRequested, std::memory_order_acquire) || stopToken.stop_requested())` をチェックしているが、`stopRequested` は `stopLearning()` で `true` に設定される。しかし、`stopLearning()` は `shutdownWorkerThread()` を呼び出し、`workerThread.join()` を待つ。そのため、`workerThreadMain` が終了する前に `stopRequested` が `true` になる。問題はない。

---

## 5. パフォーマンス・最適化の問題

### 5-1. `OutputFilter::process()` の分岐と `_mm_set_pd` (OutputFilter.cpp:255)
- **問題**: ループ内で `_mm_set_pd(dataR[i], dataL[i])` を毎回呼び出している。`_mm_set_pd` は即値で構築するため比較的高速だが、アライメントされていないロードよりも遅い可能性がある。しかし、`dataL` と `dataR` は別々の配列のため、代わりに `_mm_loadu_pd` で２つのレジスタにロードしてから `_mm_unpacklo_pd` で結合する方が効率的かもしれない。現状でも十分な性能だと思われる。

### 5-2. `EQProcessor::process()` の `processBandStereo` 呼び出し (EQProcessor.Processing.cpp:580)
- **問題**: `processBandStereo` は SSE2/FMA を使用してステレオ同時処理を行うが、ループ内で `_mm_prefetch` を使用している。`n + 8` のプリフェッチは `dataL` と `dataR` の両方に対して行われているが、`dataL` と `dataR` は異なるメモリ領域のため、プリフェッチの効果は限定的かもしれない。

### 5-3. `MKLNonUniformConvolver::processLayerBlock()` の `_mm_prefetch` (MKLNonUniformConvolver.cpp:1540)
- **問題**: `_mm_prefetch` を `T1` で使用しているが、`T1` はL2キャッシュへのプリフェッチを指示する。`T0`（L1）の方が効果的な場合もあるが、`T1` はストリーミングアクセスに向いている。

---

## 6. コンパイラ・プラットフォーム依存の問題

### 6-1. `build.bat` の `icx` コンパイラ検出 (build.bat:125)
- **問題**: `icx` モードで `-j2` を指定しているが、`cmake --build` の `--` の後に `-j 2` を渡している。これは正しいが、`Ninja Multi-Config` ジェネレータを使用しているため、`-j` オプションは Ninja に渡される。問題ない。

### 6-2. `CMakeLists.txt` の `IPP` 検出 (CMakeLists.txt:460)
- **問題**: `find_package(IPP QUIET CONFIG COMPONENTS ippcore ipps)` で IPP を検出しているが、`IPP_LINK` や `IPP_THREADING` は設定しているが、`IPP` のバージョンによってはコンポーネント名が異なる可能性がある。`ippcore` と `ipps` は標準的なコンポーネント名なので問題ない。

### 6-3. `MKLNonUniformConvolver.cpp` の `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ブロック内での `#pragma comment(lib, "psapi.lib")` (MKLNonUniformConvolver.cpp:37)
- **問題**: `psapi.lib` は `GetProcessMemoryInfo` を使用するためにリンクされている。`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` が有効な場合のみリンクされる。これは正しい。

---

## 7. コード品質・保守性の問題

### 7-1. `ConvolverProcessor.h` の `StereoConvolver` 構造体のメンバ変数 `storedDirectHeadEnabled` の初期化 (ConvolverProcessor.h:260)
- **問題**: `storedDirectHeadEnabled` は `init()` で設定されるが、デフォルト値が `false` であることが保証されていない。構造体のメンバは `StereoConvolver()` コンストラクタで初期化されていないため、未初期化の可能性がある。`storedSampleRate` なども同様。ただし、`init()` が呼ばれる前にこれらの変数が参照されることはないため、問題はない。

### 7-2. `AudioEngine.h` の `RTLocalState` 構造体の `publishTimingHistory` (AudioEngine.h:2470)
- **問題**: `publishTimingHistory` は固定サイズの配列で、`publishTimingWriteCount` でインデックスを管理している。`writeCount` は `fetch_add` でインクリメントされるが、`publishTimingHistory` の要素は `NonRT` スレッドからのみ書き込まれ、`RT` スレッドから読み取られる。`RT` は `consume` で読むが、`writeCount` が更新される前にデータが書き込まれているとは限らない。しかし、`RT` は `writeCount` を `acquire` で読み、その後 `publishTimingHistory` を読むため、`release` ストアで可視性が保証されている。問題ない。

### 7-3. `DeferredDeletionQueue.h` の `reclaim` 内の `scanned` 変数 (DeferredDeletionQueue.h:123)
- **問題**: `scanned` はインクリメントされているが、実際には `break` で抜けるため、`scanned` が `kMaxScan` に達することはない。`kMaxScan` は将来の先読みスキャン実装のために残されている。

---

## 8. その他の潜在的な問題

### 8-1. `ConvolverProcessor::loadImpulseResponse()` の `loaderTrashBin` (ConvolverProcessor.LoadPipeline.cpp:55)
- **問題**: `activeLoader` を `loaderTrashBin` に移動した後、`cleanup()` で終了したスレッドを削除している。もし `cleanup()` が呼ばれる前に `activeLoader` が複数回作成されると、`loaderTrashBin` が増え続ける可能性がある。しかし、`loaderTrashBin` のサイズが大きくなりすぎないように `if (loaderTrashBin.size() > 2)` で制限している。

### 8-2. `AudioEngine::rebuildThreadLoop()` の `DSPGuard` と `dspGuard.ptr` (AudioEngine.RebuildDispatch.cpp:520)
- **問題**: `DSPGuard` は `AudioEngine* owner` と `DSPCore* ptr` を保持し、デストラクタで `owner->destroyDSPCoreNode(ptr)` を呼び出す。`ptr` は `buildResult.runtime` で設定され、その後 `dspGuard.ptr = nullptr;` で解放を防いでいる。しかし、`destroyDSPCoreNode` は `aligned_free` を呼び出すため、正しく解放される。問題ない。

### 8-3. `ConvolverProcessor::applyBuildSnapshot()` の `pendingOverrideLock` (ConvolverProcessor.StateAndUI.cpp:1280)
- **問題**: `applyBuildSnapshot` 内で `pendingOverrideLock` をロックして `copySnapshotToPendingUnlocked` を呼び出している。その後 `irFileLock` もロックしている。ロックの順序は `pendingOverrideLock` → `irFileLock` で、他の箇所でも同じ順序でロックされているか確認する必要がある。`captureBuildSnapshot` では `pendingOverrideLock` をロックした後、`irFileLock` をロックしている。デッドロックのリスクは低いが、一貫性は保たれている。

---

## 9. 修正提案

- **1-5**: `ProgressiveUpgradeThread::upgradeStep()` で `cacheManager.save()` の前に `prepared` を `std::unique_ptr` でラップする。
- **2-5**: `DeferredDeletionQueue::reclaim()` で先頭ブロッキングを回避するため、後続のエントリもスキャンして削除可能なものがあれば処理するよう改善する（ただし、FIFO順序が崩れる可能性があるため、設計意図を確認する必要がある）。
- **7-1**: `StereoConvolver` のメンバ変数をコンストラクタで初期化する。

---

以上がバグ・問題点のリストです。全体としてコードは高品質で、多くの注意が払われていることがわかります。特にリアルタイム処理とスレッドセーフに重点が置かれており、ほとんどの問題はエッジケースに限定されています。


はい、前回の分析に加えて、さらに重要なバグを発見しました。

特に **「MKLNonUniformConvolver のバッファオーバーラン」** は非常に深刻で、ランダムなクラッシュやメモリ破壊を引き起こす可能性が高いです。

---

### 1. 【重大】MKLNonUniformConvolver における FDL バッファのインデックスオーバーラン (ヒープ破壊)

- **該当ファイル**: `src/MKLNonUniformConvolver.cpp`
- **該当箇所**: `processLayerBlock()` 関数内のループ (L1500-1520付近) および `Add()` 関数内の非即時レイヤー分散処理部。
- **問題の詳細**:
  `fdlReal` / `fdlImag` 配列の確保サイズは `(numParts * 2) * complexSize` です (mirrorスロットを含む)。
  しかし、読み取りインデックス `index = linStart + p` は以下の計算で導出されます。
  ```cpp
  const int linStart = l.fdlIndex - l.numPartsIR + 1 + l.numParts;
  for (int p = 0; p < l.numPartsIR; ++p) {
      const int index = linStart + p;
      // index を用いて fdlReal + index * complexSize にアクセス
  }
  ```
  `l.fdlIndex` の最大値は `numParts - 1` です。`l.numPartsIR` が `l.numParts` に近い場合、`index` の最大値は `(numParts-1) - 1 + numParts + (numParts-1) = 3*numParts - 3` となります。
  これは確保済みの有効インデックス範囲 `0 〜 (2*numParts - 1)` を大幅に超えており、**後続のヒープメモリ領域を破壊**します。結果として、`ConvolverProcessor` の他のメンバ変数や MKL 内部構造が破損され、ランダムなタイミングでアクセス違反 (0xC0000005) が発生します。

- **修正提案**:
  `linStart` の計算式を再評価し、インデックスが常に `2 * numParts` 未満に収まるようにする必要があります。具体的には、`linStart` に `% (2 * l.numParts)` を適用するか、理論的に値が収まるようにオフセットを補正する修正が求められます。

---

### 2. 【中程度】`shrinkToFit` 関数の例外安全性の欠如

- **該当ファイル**: `src/convolver/ConvolverProcessor.Internal.h`
- **該当箇所**: `shrinkToFit()` 関数
- **問題の詳細**:
  ```cpp
  inline void shrinkToFit(juce::AudioBuffer<double>& buffer) {
      juce::AudioBuffer<double> newBuffer(...);
      buffer = std::move(newBuffer);
  }
  ```
  `newBuffer` のメモリ確保に失敗した場合（`std::bad_alloc`）、`buffer` は元のデータを保持したまま関数を抜けますが、例外がスローされるため呼び出し元が適切にハンドルしない限りプロセスが異常終了します。オーディオ処理の LoaderThread 内で使用されており、メモリ逼迫時にクラッシュの原因となり得ます。

- **修正提案**:
  `try-catch` で囲み、確保失敗時は元のバッファをそのまま維持する（縮小を諦める）実装に変更します。

---

### 3. 【中程度】`ConvolverProcessor` の `IRLoadPreview` における推奨長警告の条件漏れ

- **該当ファイル**: `src/convolver/ConvolverProcessor.StateAndUI.cpp`
- **該当箇所**: `analyzeImpulseResponseFile()` 関数
- **問題の詳細**:
  ユーザーが手動で `IR Length` をオーバーライド（Manual Override）している場合でも、`analyzeImpulseResponseFile()` で `exceedsRecommended` フラグが `true` になり、長いIRロード時の警告ダイアログが表示されます。手動オーバーライド時はユーザーが意図的に長さを設定しているため、警告を抑制するか、適用する設定値（`targetIRLengthSec`）を基準に警告を出すべきです。

- **修正提案**:
  `exceedsRecommended` の判定に、現在の `pendingOverride.irLengthManualOverride` の状態を考慮する条件を追加します。

---

### 4. 【軽微】`ConvolverProcessor` の `forceCleanup` と `loaderTrashBin` の競合リスク

- **該当ファイル**: `src/convolver/ConvolverProcessor.Lifecycle.cpp` / `.StateAndUI.cpp`
- **問題の詳細**:
  `forceCleanup()` は `loaderTrashBin` をスワップして全 `LoaderThread` の `stopThread(500)` を呼びますが、`cleanup()` は Timer 駆動で `loaderTrashBin` から終了したスレッドを削除します。もし `forceCleanup()` 実行中に Timer の `cleanup()` が別スレッド（Message Thread）で走ると、`loaderTrashBin` へのアクセス競合が発生する可能性があります（`std::deque` の操作に対して同期が取られていない）。

- **修正提案**:
  `loaderTrashBin` へのアクセスに対して、`irFileLock` など既存のミューテックスを流用するか、専用のミューテックスを追加して保護します。

---

### 5. 【軽微】`SpectrumAnalyzerComponent` のピークホールド減衰におけるフレームレート依存

- **該当ファイル**: `src/SpectrumAnalyzerComponent.cpp`
- **該当箇所**: `timerCallback()` 内のピークホールド減衰処理
- **問題の詳細**:
  `peakBuffer` の減衰に `peakDecayFactor = std::exp(-dt / 0.4)` を使用して時間依存にしていますが、`smoothedBuffer` への減衰適用がないため、アンダーラン時（データ欠落時）に `UNDERRUN_DECAY_DB` で固定値減衰させています。この固定値減衰は Timer の呼び出し間隔（通常 ~16ms）に依存するため、タイマーのジッタや遅延の影響で減衰速度が変わります。

- **修正提案**:
  `UNDERRUN_DECAY_DB` を時間ベースの減衰（例：`dt * DECAY_DB_PER_SEC`）に変更し、フレームレート非依存にします。

---

### まとめ

最も優先度が高いのは **「バグ #1 (MKL バッファオーバーラン)」** です。このバグは再現性こそ低いものの、発症すると深刻なメモリ破壊を引き起こし、デバッグが極めて困難になります。他のバグはエッジケースやパフォーマンスの領域ですが、安定性向上のためには併せて修正することを推奨します。


提示された「ConvoPeq」のビルドスクリプトおよびソースコード（`CMakeLists.txt`, `build.bat`, `src\AlignedAllocation.h`, `src\AllpassDesigner.cpp`）を詳細に調査しました。

プロ用オーディオソフトウェアとして要求されるリアルタイム性、堅牢性、および `msvc`/`intel icx` によるクロスコンパイル環境の観点から、**修正が必要な致命的なバグ、ビルドエラーの要因、および潜在的な不具合・冗長性**を機能別に分類して報告します。

---

## 1. ビルドスクリプト（build.bat）の致命的な不具合

### 🔴 引数パースにおける `=` の区切り文字による値の強制書き換えバグ

* **問題の箇所**: `build.bat` 内の引数パース用 `for` ループ
```bat
for %%A in (%*) do (
    set "arg=%%~A"
    if "!arg:~0,2!"=="-D" (
        REM cmd.exe strips =VALUE, so append =ON.
        set "CMAKE_EXTRA_FLAGS=!CMAKE_EXTRA_FLAGS! !arg!=ON"
        echo [INFO] Extra CMake define: !arg!=ON
    )

```


* **原因と影響**: `cmd.exe` の仕様上、バッチファイルの引数において `=` （イコール）はスペースと同様に**トークンの区切り文字**として扱われます。そのため、ユーザーが `build.bat -DCONVOPEQ_DIAG_SAMPLE_MASK=ON` や `-DNUM=1` のように指定して実行すると、ループの1回目では `%%A` に `-DCONVOPEQ_DIAG_SAMPLE_MASK` が入り、2回目で `ON` または `1` が入ります。
結果として、条件分岐によって強制的に `!arg!=ON` （値がすべて `=ON`）に変形されて `CMAKE_EXTRA_FLAGS` に蓄積され、任意の数値をフラグに渡すことが不可能になります。
* **修正案**: `%*` をそのままトークン分割するのではなく、`%1` を `shift` しながら処理するか、引数文字列全体から置換を行うロジックへ変更する必要があります。

### 🟡 ビルド完了時における画面出力の重複（UIの記述ミス）

* **問題の箇所**: `build.bat` 末尾の成功ログ出力
* **原因と影響**: スクリプトの最末尾において、まったく同じ `Build configuration`、`Build directory`、`Executable location` の表示ブロックが**2回連続で重複して記述**されています。動作上の害はありませんが、コンソール出力の視認性を損なっています。

---

## 2. ビルド構成（CMakeLists.txt）の不備とリンクエラー要因

### 🔴 テストターゲット群におけるプラットフォーム定義と例外処理の適用漏れ

* **問題の箇所**: `WIN32` 固有の定義を適用する `foreach` ループ、および `/EHsc` オプションの追加処理
* **原因と影響**: `DeferredDeletionQueueReclaimTests` と `PriorityIntegrationTests` の2つのターゲットが、`_UNICODE` や `NOMINMAX` を適用するターゲットリストから漏れています。さらに、例外処理を有効化する `target_compile_options(... PRIVATE /EHsc)` のリストからも漏れています。
`DeferredDeletionQueueReclaimTests` は `AlignedAllocation.h` をインクルードしており、内部で `throw std::bad_alloc()` などの例外パスが存在するため、MSVC/icx でビルドした際に `C4530` 警告（または未定義動作）が発生します。
* **修正案**: `foreach(tgt IN ITEMS ...)` および `/EHsc` を適用するターゲット群に、上記2つのテストターゲットを明示的に追加してください。

### 🔴 icxビルド時におけるテストターゲットのLTCG解除漏れによるリンク失敗

* **問題の箇所**: `if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")` ブロック内の `INTERPROCEDURAL_OPTIMIZATION OFF` 設定
* **原因と影響**: スクリプト内のコメントにもある通り、`icx` (clang-cl) 環境下でテスト用の実行ファイルをリンクする際、LTCG が有効だと無効なオブジェクト形式となり `LNK1107` エラーが発生します。しかし、ここでも `DeferredDeletionQueueReclaimTests` と `PriorityIntegrationTests` が解除対象から漏れています。このため、`icx` でのビルド時にこれら2つのテストのリンクが確実に失敗します。

### 🔴 コンソールアプリ（MTNUPCMeasurement）におけるIntelランタイムパスの不足

* **問題の箇所**: `MTNUPCMeasurement` に対するリンクライブラリ設定
* **原因と影響**: `ConvoPeq` アプリ本体のターゲットに対しては、`icx` リンク時の `LNK1104 (libircmt.lib)` 回避のために `target_link_directories(ConvoPeq PRIVATE "${_INTEL_COMPILER_ROOT}")` を通していますが、`MTNUPCMeasurement` ターゲットにはこのパスが通っていません。そのため、`icx` を使用してテストを含めたフルビルドを行うと、このコンソールアプリのリンクフェーズで静的ランタイムを見つけられずビルドエラーになります。

### 🟡 非推奨プロパティ「COMPILE_FLAGS」の使用と最適化衝突の懸念

* **問題の箇所**: 末尾の `icx/LLVM` 向け OOM 回避策
```cmake
set_source_files_properties(src/audioengine/AudioEngine.Snapshot.cpp PROPERTIES COMPILE_FLAGS "/O1")

```


* **原因と影響**: `COMPILE_FLAGS` は古い CMake のプロパティであり、現在は複数のオプションを安全に扱える `COMPILE_OPTIONS` の使用が推奨されています。また、文字列で直接 `/O1` を指定すると、上位で指定されている Release 最適化フラグ `/O3`（`CMAKE_CXX_FLAGS_RELEASE`）と衝突し、コンパイラによっては警告が発生するか、意図した並列度制限や最適化レベルの制御が正しく伝わらない可能性があります。

---

## 3. メモリ管理（src\AlignedAllocation.h）の設計矛盾と冗長性

### 🔴 ScopedAlignedPtr のコメントと static_assert の致命的矛盾

* **問題の箇所**: `ScopedAlignedPtr` の実装および `reset` メソッド
```cpp
// Note: This class calls the destructor of a single object (~T()).
//       It is suitable for managing a single non-POD object...
...
static_assert(std::is_trivially_destructible_v<T>,
              "ScopedAlignedPtr only supports trivially destructible types (POD arrays)");

```


* **原因と影響**: クラスのヘッダーコメントでは「非PODの単一オブジェクトの管理に適しており、デストラクタ `~T()` を呼び出す」と明記されていますが、実際の `reset` 内にはデストラクタの明示的呼び出しがありません。そればかりか、`static_assert` によって `std::is_trivially_destructible_v<T>`（デストラクタがトリビアルな型、つまりPOD等）であることが強制されています。
このため、仕様書の記述を信じて非トリビアルなカスタムクラスをこのポインタで管理しようとすると、**コンパイルエラー**になります。仮に `static_assert` を外したとしても、今度はデストラクタが呼ばれずにリソースリークを引き起こすため、設計と実装が完全に乖離しています。
* **修正案**: 単一の非PODオブジェクトの管理には、すでに正しくデストラクタが実装されている `aligned_unique_ptr` を使用するようにコードベースを一貫させるか、`ScopedAlignedPtr` からトリビアル制約を分離して正しくデストラクトする専用の特殊化を用意すべきです。

### 🟡 例外送出後のヌルポインタチェック（デッドコード）

* **問題の箇所**: `makeAlignedArray` および `makeAlignedCopy`
```cpp
T* ptr = static_cast<T*>(aligned_malloc(count * sizeof(T), 64));
if (!ptr) throw std::bad_alloc(); // デッドコード

```


* **原因と影響**: `convo::aligned_malloc` は、内部でメモリ確保に失敗した（`ptr == nullptr`）場合に**自ら `throw std::bad_alloc()` を送出する契約**になっています。そのため、呼び出し側にある `if (!ptr)` というチェックは絶対に実行されないデッドコードです。実害はありませんが、コードの冗長性を招いています。

---

## 4. DSP・最適化（src\AllpassDesigner.cpp）における問題

### 🔴 大量の未使用アルゴリズム（xxHash関連の完全なデッドコード）

* **問題の箇所**: 匿名名前空間内およびファイル上部に定義された以下の関数群
* `rotl64` / `readLE64` / `readLE32`
* `xxh64Round` / `xxh64MergeRound` / `xxh64Avalanche` / `xxh64Digest`


* **原因と影響**: 非常に精巧な `xxHash64` のバイナリハッシュ解析ロジックが実装されていますが、`AllpassDesigner.cpp` 内の主要アルゴリズムである `designWithCMAES` や群遅延計算のどこからも**一切呼び出されていません**。コンパイル後のバイナリサイズ肥大化や、コードレビュー時の混乱の原因になります。キャッシュ管理等でハッシュが必要な場合は、`CacheManager` などの適切なコンポーネントへ移譲するか、不要であれば削除すべきです。

### ⚠️ 最適化ループ終了後における結果の書き戻し処理（提示コード内の懸念）

* **問題の箇所**: `AllpassDesigner::designWithCMAES` 内の最末尾（提示コードで切れている直後）
* **原因と影響**: スクロール末尾の収束判定の手前まで調査した段階で、最適化された無制約空間のパラメータ `bestParams` (サイズ $D$) を物理パラメータである $\rho, \theta$ に逆変換し、引数で渡された `std::vector<SecondOrderAllpass>& sections` へ反映・詰める処理が未完成、あるいは見当たりません。
もし最末尾の省略されたブロックにおいて `sections` への書き戻し（および物理パラメータへの確定クランプ）が行われていない場合、呼び出し元には最適化される前の初期状態のフィルタしか返らないという致命的な計算バグになります。結果の反映が確実に行われているか、後続コードの再確認が必要です。


提示されたコードベースおよび「ConvoPeq」のアーキテクチャ（JUCE 8 / Intel oneAPI MKL / Windows 11 x64環境）において、前回の指摘のほかに**シニアオーディオデベロッパーの視点からレビューすべき、潜在的かつ致命的なバグ・設計上の脆弱性**をさらに3点検出しました。

これらはコンパイルを通過しても、ランタイムでのクラッシュ、オーディオの音飛び（ドロップアウト）、またはフィルター設計の破綻を引き起こす原因になります。

---

## 5. SIMD/MKLアライメントの不整合とヒープ破損リスク

### 🔴 `std::vector` の標準アロケータ使用によるMKL/SIMDアライメント違反

* **問題の箇所**: `AllpassDesigner.cpp` 内における `std::vector<double>` および内部行列計算
* **原因と影響**: `AlignedAllocation.h` で64バイトアライメント用のカスタムアロケータ（`convo::aligned_allocator`など）の基盤が用意されているにもかかわらず、`AllpassDesigner.cpp` 内のCMA-ESの実装や群遅延計算で標準の `std::vector<double>` がそのまま使用されている箇所があります。
Intel MKLの関数（特に動的ベクタライズやBLAS/LAPACK系）にこれらのポインタを直接渡した場合、あるいはコンパイラ（`intel icx`）がAVX-512/AVX2によるアライメント済みロード・ストア（`_mm512_load_pd` 等）を自動生成した場合、標準アロケータが返したメモリが64バイト境界に整列していないと、**ランタイムで `Access Violation (Segmentation Fault)` を起こしてアプリが即座にクラッシュ**します。
* **修正案**: MKLやSIMD処理に絡むすべての `std::vector` は、以下のようにカスタムアロケータを明示的に指定してください。
```cpp
std::vector<double, convo::aligned_allocator<double, 64>> buffer;

```



### 🔴 Windows環境における `_aligned_free` の不一致リスク

* **問題の箇所**: `AlignedAllocation.h` 内の解放処理の抽象化
* **原因と影響**: Windows (MSVC/icx) の仕様上、`_aligned_malloc` で確保したメモリは**必ず `_aligned_free` で解放しなければなりません**。標準の `free` や `delete` を使用して解放を試みると、Windowsのヒープマネージャが内部管理情報を正しく認識できず、`Heap Corruption Detected` エラーを引き起こしてクラッシュします。`ScopedAlignedPtr` やカスタムアロケータのデリート側で、Windows固有の `_aligned_free` へのバインドが厳密に保証されているか再確認が必要です。

---

## 6. DSP最適化（CMA-ES）における数値的・アルゴリズム的脆弱性

### 🔴 フィルタ安定性（極の単位円内拘束）の境界条件処理の欠落

* **問題の箇所**: `AllpassDesigner::designWithCMAES` 内のパラメータ変換処理
* **原因と影響**: オールパスフィルタの2次セクション（SOS）が安定であるためには、極（poles）が複素平面上の単位円の内部（半径 $\rho < 1.0$）に存在しなければなりません。しかし、CMA-ESは「無制約空間」を探索するアルゴリズムであるため、探索ベクトルが一時的に $\rho \ge 1.0$ に相当する値を生成することが頻繁にあります。
もし物理パラメータへの逆変換時に適切なクランプ（例: $\rho = \tanh(x)$ による強制収束）や、コスト関数への圧倒的なペナルティ値の加算を行っていない場合、**システムが不安定化（発散）して出力が $\pm\infty$ や `NaN` になり、最悪の場合スピーカーや聴覚を損なう大音量のノイズが発生**します。

### ⚠️ 共分散行列の正定値性崩壊による `NaN` の伝播

* **問題の箇所**: CMA-ESの共分散行列（Covariance Matrix $C$）の更新と固有値分解
* **原因と影響**: 最適化のループが進行し、解が特定の局所解に極端に収束し始めると、数値丸め誤差によって共分散行列の正定値性（すべての固有値が正である状態）が失われ、半正定値または負の固有値を持つようになります。この状態でMKLによる固有値分解や平方根計算を行うと、結果に `NaN` が混入し、それ以降の探索が完全に破綻します。
* **修正案**: 各世代の更新ごとに、行列の対角成分に微小な値（マシンイプシロン程度のバリヤ：例 $C = C + \epsilon I$）を強制的に加算する、数値的プロテクションコードを追加してください。

---

## 7. リアルタイムスレッド安全性（オーディオドライバ競合）

### 🔴 最適化ループ内での動的メモリ確保（Real-time Safety 違反）

* **問題の箇所**: `AllpassDesigner::designWithCMAES` 内でループごとに生成されるテンポラリ配列
* **原因と影響**: もしこの `AllpassDesigner` が、オーディオインターフェースのバッファ交換を行うリアルタイムスレッド（ASIOやWASAPIのコールバック内部）から同期的に呼び出される設計になっている場合、致命的なドロップアウトが発生します。CMA-ESの内部で `std::vector` のリサイズやメモリ確保が繰り返されると、OSのカーネルミューテックスのロックを奪い合うことになり、リアルタイム性が完全に破壊されます。
* **修正案**: 最適化処理は必ずJUCEの `ThreadPool` や `BackgroundJob` などのワーカースレッドへ完全に移譲し、オーディオスレッド側は前述の **Epoch-based RCUモデル** を介して、確定したフィルタ係数のポインタのみをスワップする設計を徹底してください。

---

今回検出されたアライメントやリアルタイム性の問題は、プロ仕様のオーディオエンジンにおいて非常に重要な境界線となります。これらの計算やスレッドの分離について、現在の実装でワーカースレッドへの完全な分離は達成されていますでしょうか？


以下が全文マークダウンです。そのままコピペできます。

```markdown
# ConvoPeq ソースコード バグ監査レポート

- 対象: `ConvoPeq` 全279ファイル / 81,346行 (Windows11 x64 / AVX2必須 / msvc-icx / MKL+JUCE)
- 日付: 2026-07-23
- 観点: クラッシュ / UAF / 競合 / MKLハンドルリーク / AVX2アライメント / RT安全性 / 数値安定性

## Critical - 即クラッシュ / UAF / メモリ破損

### 1. `src/SafeStateSwapper.h` - RCU epoch順序逆転によるUAF
**概要:** `swap()`が `globalEpoch` を2回bumpしてから `activeState` をexchange。

**現状:**
```cpp
auto e1 = fetchAdd(globalEpoch,1);
fetchAdd(globalEpoch,1);
auto* old = exchange(activeState,newState);
```

リーダーが2つのbumpの間に `enterReader()` すると epoch=2 で old を掴んだまま、reclaim側は `minReaderEpoch=2` と判定して old を解放 → UAF。

**修正:**
```cpp
auto* old = exchange(activeState,newState, std::memory_order_acq_rel);
auto e1 = fetchAdd(globalEpoch,1, std::memory_order_acq_rel);
fetchAdd(globalEpoch,1, std::memory_order_acq_rel);
```

### 2. `src/SafeStateSwapper.h` - 同一ポインタ再スワップでDouble Free
`newState == activeState` の時に `old==new` をそのまま `retiredBuffer` に入れる。activeなポインタを解放キューに入れるため次回 `tryReclaim()` で使用中解放。

**修正:**
```cpp
if (old==nullptr || old==newState) return;
```

### 3. `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` - LifecycleTokenリーク
`enterPrepare()` 後に `aligned_alloc` 失敗や `catch(...)` で `rollbackPrepareFailure()` から早期return。`leavePrepare()` が呼ばれず `phase_` が `Preparing` のまま固まり、次回コールバックで `validateTransition` → `std::abort()` がオーディオスレッドで実行。

**修正:** RAII ScopeGuardで必ず `leavePrepare()` を呼ぶ。

### 4. `src/audioengine/AudioEngine.h` `makeRuntimeReadHandle()` - Hazardポインタ寿命バグ
```cpp
auto readToken = coordinator.acquireReadToken();
auto* world = consumeWorldHandle(store, readToken);
return RuntimeReadHandle{observed, world}; // readTokenはここで破棄
```
ローカルの `readToken` がreturn前に破棄され raw `world` は保護無し。別スレッドが publish→retire するとコールバック実行中に `aligned_free(world)` → UAF。

**修正:** `RuntimeReadHandle` が Token を所有するように変更。

### 5. `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` - ActiveCount非対称
`AudioCallbackRuntimeScope` 構築前に `bufferToFill.clearActiveBufferRegion()` で早期returnするパスがあり、`audioCallbackActiveCount` の inc/dec が非対称になる。長期稼働で `activeCount` が負に飽和。

### 6. `src/ConvolverProcessor.h` - ConvolverState publishのメモリオーダー不足
reader側は `consume/acquire` だが writer側の一部が `relaxed` で publish。icxの積極的最適化で state の古い値を見る。

## High - リーク、競合、MKLハンドル枯渇

### 7. `src/AlignedAllocation.h` - `aligned_free()` が診断追跡をバイパス
`DIAG_MKL_MALLOC` で確保、`mkl_free` 直呼びで解放。`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` 時、allocatedBytesが単調増加しリーク検出が機能しない。

**修正:** `DIAG_MKL_FREE(ptr, size)` ラッパーを用意するか、解放時にもサイズ追跡。

### 8. `src/AlignedAllocation.h` - `makeAlignedArray_nothrow` サイズオーバーフロー未チェック
`count * sizeof(T)` が `size_t` でオーバーフローすると小さな確保でヒープ破壊。`MKLAllocator::allocate` はチェック済みだが nothrow版は未チェック。

**修正:**
```cpp
if (count > SIZE_MAX / sizeof(T)) return {nullptr};
```

### 9. `src/DftiHandle.h` / `MklFftEvaluator.h` - DFTI記述子リーク
`DftiCreateDescriptor` → `DftiSetValue` → `DftiCommitDescriptor` 間で例外が飛ぶと `DftiFreeDescriptor` が呼ばれない。

**修正:** `DftiHandle` を `unique_ptr` + custom deleter で完全RAII化。

### 10. `src/MKLNonUniformConvolver.cpp` - 非均一分割 off-by-one
Tailブロック長が `fftSize` 境界ちょうどで `partitionSize=0` になりゼロ除算または `mkl_malloc(0)` → nullptr → 後続AVXループでnull deref。IR長が2の累乗+1で再現。

### 11. `src/MKLRealTimeSetup.cpp` - `mkl_set_num_threads_local` 復帰漏れ
オーディオスレッドで `mkl_set_num_threads_local(1)` をセットしたまま例外/早期returnで元に戻さず、別スレッドのMKLバッチがシングルスレッド劣化。

### 12. `src/DeferredDeletionQueue.h` / `DeferredFreeThread.h` - 固定長キュー溢れでfree消失
`tryPush` が false を返すが呼び出し側が無視してポインタ破棄 → リーク。長時間稼働でメモリ単調増加。

### 13. `src/LockFreeRingBuffer.h` - ABAとmemory_order
`head/tail` が `uint32_t` でラップ、ABA検出無し。`fetch_add(relaxed)` と `load(acquire)` 混在。

### 14. `src/audioengine/ISRRetireRouter.cpp` - Quarantine二重enqueue
`ISRDSPQuarantine::push` と `ISRRetireRouter::routeForRetire` が同一オブジェクトを2レーンに入れる競合。`compare_exchange_weak` の spurious failure 考慮漏れで double-free。

## Medium - DSP数値・AVX2・ロジック

### 15. AVX2カーネル全体 - `_mm256_load_ps` vs `_mm256_loadu_ps` 混在
`MKLAllocator<64>` は64Bアラインだが `AudioSegmentBuffer` や JUCE `AudioBuffer` は16Bアライン止まり。`DSPCoreFloat.cpp` で `_mm256_load_ps` 使用 → #GP 例外でクラッシュ。

**修正:** 常に `loadu/storeu`、アライン版は `assert(is_aligned<32>)` 後にのみ。

### 16. `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` - Tailはみ出し書き
`numSamples % 8 !=0` の時、残りを `storeu` で8個書くため終端を4サンプルはみ出し書き。タイトな `AudioBlock` でヒープ破壊。

**修正:** マスクストア `_mm256_maskstore_ps` またはスカラーフォールバック。

### 17. `src/Fixed15TapNoiseShaper.h` / `LatticeNoiseShaper.h` - 係数半更新
学習スレッドが `coeffs[15]` 書き、オーディオスレッドがAVXで8個ずつ読む。半更新状態で発散 → Inf出力。

**修正:** ダブルバッファ + アトミックポインタスワップ。

### 18. `src/UltraHighRateDCBlocker.h` - 状態初期化漏れ
`prepareToPlay` で `dcState` 未クリア。192kHz→48kHzでDC残留、初回ブロックで denormal 大量発生 → CPUスパイク。

### 19. `src/OutputFilter.cpp` / `InputBitDepthTransform.h` - FTZ/DAZ不整合
オーディオスレッドでは `ScopedNoDenormals` でFTZ有効、学習スレッドでは無効。denormalを含む学習結果が渡ると100倍遅延。

### 20. `src/TruePeakDetector.cpp` - オーバーサンプル状態リセット漏れ
`reset()` が遅延ラインをクリアせず、シーク後に前曲ピーク保持 → 誤TruePeak、リミッタ誤動作。

### 21. `src/CmaEsOptimizer.h` - NaN伝播で無限ループ
`sigma <= 1e-12` でコレスキー分解失敗 → `sqrt(negative)` → NaN。`isFinite` チェックが `costFunc` 後にあるため `bestMean` が NaN 汚染。

### 22. `src/AllpassDesigner.cpp` - 群遅延発散
`eps=1e-12*(1+rho2)` が `rho=0.98` で小さすぎ、極が単位円に近いと `tau > 1e6` で重み発散。

### 23. `src/CacheManager.cpp` - TOCTOUとキャッシュ汚染
`computeIRHash` 後に mtime再チェックするが `xxh64Digest` と2回目チェックの間に書き換わると古いハッシュで新ファイルをキャッシュ。`clearCache()` がロック無しで filesystem を触る競合。

### 24. `src/CpuFeatureCheck.cpp` - OSXSAVE未チェック
`__cpuid(1)` のAVX2ビットのみで `XGETBV` 未チェック。VMで `_mm256` が #UD。

**修正:** `OSXSAVE && (XCR0[2:1]==0b11)` を確認。

### 25. `src/audioengine/AudioEngine.Mmcss.cpp` - MMCSSハンドルリーク
`tryApply...` が成功時に `t_mmcssHandle` を上書き、既存ハンドルを `AvRevert` せずリーク。ASIOスレッド使い回しでハンドル無限増。

### 26. `src/audioengine/AudioEngine.Timer.cpp` - MessageThread違反
タイマー内で `ValueTree` を直接触り、オーディオスレッドの `getConvolverState()` とデータ競合。JUCE `ValueTree` はスレッドセーフではない。

### 27. `src/ConvolverControlPanel.cpp` - `irPreviewRequestId` ABA
`fetchAdd+1` でID生成、`load(acquire)` 比較で staleness判定だが int ラップで古いリクエストが新しいと誤認。

## Low - 潜在的UB / ビルド

### 28. `build.bat` / `CMakeLists.txt` - `/arch:AVX2` 付与漏れ
ユーティリティライブラリに `/arch:AVX2` が付いておらず SSE/AVX混在で呼び出し規約不整合。

### 29. `src/RefCountedDeferred.h` - refCountチェック
`fetch_sub` 戻り値が `==1` 判定のみ。オーバーフローで早期解放の可能性。

### 30. `src/StateKey.h` - `operator==` 比較不足
`generation` と `fftSize` のみ比較、`sampleRate` 違いが同キー扱いで誤キャッシュヒット。

### 31. `src/PsychoacousticDither.cpp` - RT違反
`std::mt19937` をオーディオスレッドで毎ブロック生成、`random_device` が例外をオーディオスレッドで投げる。

### 32. `src/DiagnosticsConfig.h` - マクロ二重評価
`DIAG_MKL_MALLOC(x++, 64)` でカウンタが2回進む。`do { auto _s=(size); ... } while(0)` 化が必要。

---

## まず直すべきTop5

1. **SafeStateSwapperのepoch順序** - 1行入替でUAF解消
2. **RuntimeReadHandleのHazard Token所有化** - オーディオスレッドUAF根源
3. **PrepareToPlayのLifecycleTokenリーク** - `abort()` 回避
4. **AVXのload/storeをloadu/storeu統一とtailマスク** - クラッシュ/ヒープ破壊
5. **DftiHandleのRAII化とMKLスレッドローカルScopeGuard** - ハンドル枯渇/性能劣化
```
