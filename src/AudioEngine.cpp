//============================================================================
// AudioEngine.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// AudioEngineの実装
//============================================================================
#include "AudioEngine.h"
#include <cmath>
#include <complex>
#include <algorithm>
#include <mutex>
#include <avrt.h>

#if JUCE_INTEL
 #include <xmmintrin.h>
 #include <pmmintrin.h>
 #include <immintrin.h>
#endif

#if JUCE_DSP_USE_INTEL_MKL
#include <mkl.h>
#endif

// コンストラクタ
//--------------------------------------------------------------
AudioEngine::AudioEngine()
{
    // デフォルトサンプルレート (0 = 未初期化/デバイスなし)
    currentSampleRate.store(0.0);

    // バッファ初期化
    audioFifoBuffer.setSize (2, FIFO_SIZE);
    currentDSP.store(nullptr);
}



void AudioEngine::initialize()
{
    // 初期DSP構築 (デフォルト設定)
    // 安全対策: バッファサイズを余裕を持って確保 (SAFE_MAX_BLOCK_SIZE)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ
    requestRebuild(48000.0, SAFE_MAX_BLOCK_SIZE);
    maxSamplesPerBlock.store(SAFE_MAX_BLOCK_SIZE);
    currentSampleRate.store(48000.0);

    uiConvolverProcessor.addChangeListener(this);
    uiEqProcessor.addChangeListener(this);
    uiConvolverProcessor.addListener(this);
    uiEqProcessor.addListener(this);

    // タイマー開始 (50ms間隔)
    // - DSP再構築リクエストのポーリング (Audio Threadからの依頼を処理)
    // - ガベージコレクション
    startTimer(50);
}



AudioEngine::~AudioEngine()
{
    uiConvolverProcessor.removeChangeListener(this);
    uiEqProcessor.removeChangeListener(this);
    uiConvolverProcessor.removeListener(this);
    uiEqProcessor.removeListener(this);

    stopTimer();

    // 進行中のコールバックが完了するのを待つため、DSPを無効化
    // これにより、changeListenerCallback内でdsp->へのアクセスを防ぐ
    currentDSP.store(nullptr, std::memory_order_release);
    activeDSP.reset();
}

//--------------------------------------------------------------
// FIFOからデータ読み出し (UI Thread)
//--------------------------------------------------------------
void AudioEngine::readFromFifo(float* dest, int numSamples)
{
    // 単一のリーダーを保証: 複数のUIコンポーネント（アナライザーなど）からの同時読み出しを防ぐ。
	//
    // Note: 書き込み側 (DSPCore::pushToFifo / Audio Thread) はこのロックを使用しないため、ロックフリーです。
    //       したがって、ここでロックを取得してもオーディオスレッドをブロックする恐れはありません (Deadlock Free)。
    const juce::ScopedLock sl(fifoReadLock);

    int start1, size1, start2, size2;
    audioFifo.prepareToRead(numSamples, start1, size1, start2, size2);

    // 実際に読み取れるサンプル数を計算 (FIFO内の有効データ量に依存)
    // prepareToRead は numSamples 分の領域を返さない場合がある (FIFO不足時)
    const int actualRead = size1 + size2;
    const bool hasRightChannel = (audioFifoBuffer.getNumChannels() > 1);

    // AVX2 L+R 平均化ヘルパー
    auto mixToMono = [](const float* srcL, const float* srcR, float* dst, int n) noexcept
    {
#if defined(__AVX2__)
        const __m256 half = _mm256_set1_ps(0.5f);
        int i = 0;
        const int vEnd = n / 8 * 8;
        for (; i < vEnd; i += 8)
        {
            __m256 vL  = _mm256_loadu_ps(srcL + i);
            __m256 vR  = _mm256_loadu_ps(srcR + i);
            __m256 avg = _mm256_mul_ps(_mm256_add_ps(vL, vR), half);
            _mm256_storeu_ps(dst + i, avg);
        }
        for (; i < n; ++i) dst[i] = (srcL[i] + srcR[i]) * 0.5f;
#else
        for (int i = 0; i < n; ++i) dst[i] = (srcL[i] + srcR[i]) * 0.5f;
#endif
    };

    if (size1 > 0)
    {
        const float* srcL = audioFifoBuffer.getReadPointer(0, start1);
        const float* srcR = hasRightChannel ? audioFifoBuffer.getReadPointer(1, start1) : srcL;
        mixToMono(srcL, srcR, dest, size1);
    }

    if (size2 > 0)
    {
        const float* srcL = audioFifoBuffer.getReadPointer(0, start2);
        const float* srcR = hasRightChannel ? audioFifoBuffer.getReadPointer(1, start2) : srcL;
        mixToMono(srcL, srcR, dest + size1, size2);
    }

    // 実際に読み取った分だけFIFOを進める
    if (actualRead > 0)
        audioFifo.finishedRead(actualRead);

    // 足りない分はゼロ埋め (グリッチ防止)
    if (actualRead < numSamples)
        juce::FloatVectorOperations::clear(dest + actualRead, numSamples - actualRead);
}

//--------------------------------------------------------------
// EQ応答曲線計算
// 現在のEQ設定に基づき、周波数ごとのトータルゲイン応答（マグニチュード）を計算する
//--------------------------------------------------------------
void AudioEngine::calcEQResponseCurve(float* outMagnitudesL,
                                     float* outMagnitudesR,
                                     const std::complex<double>* zArray,
                                     int numPoints,
                                     double sampleRate)
{
    const double sr = sampleRate;
    if (sr <= 0.0)
    {
        for (int i = 0; i < numPoints; ++i)
        {
            if (outMagnitudesL) outMagnitudesL[i] = 1.0f;
            if (outMagnitudesR) outMagnitudesR[i] = 1.0f;
        }
        return;
    }

    // ── 最適化: 有効なバンドの係数をスタック上で事前に計算 ──
    // UIスレッドでの計算負荷を下げるため、無効なバンドやゲイン0のバンドは除外する
    struct ActiveBand {
        EQCoeffsBiquad coeffs;
        EQChannelMode mode;
    };
    ActiveBand activeBands[EQProcessor::NUM_BANDS];
    int numActiveBands = 0;

    // 状態スナップショットを取得して一貫性を確保
    auto eqState = uiEqProcessor.getEQState();

    if (eqState == nullptr)
    {
        if (outMagnitudesL) std::fill_n(outMagnitudesL, numPoints, 1.0f);
        if (outMagnitudesR) std::fill_n(outMagnitudesR, numPoints, 1.0f);
        return;
    }

    for (int band = 0; band < EQProcessor::NUM_BANDS; ++band)
    {
        const auto& params = eqState->bands[band];
        if (!params.enabled) continue;

        EQBandType type = eqState->bandTypes[band];

        // LowPass/HighPass以外でゲインがほぼ0の場合はスキップ
        if (type != EQBandType::LowPass && type != EQBandType::HighPass &&
            std::abs(params.gain) < EQ_GAIN_EPSILON)
            continue;

        activeBands[numActiveBands++] = {
            EQProcessor::calcBiquadCoeffs(type, params.frequency, params.gain, params.q, sr),
            eqState->bandChannelModes[band]
        };
    }

    float totalGainLinear = 1.0f;
    if (!uiEqProcessor.getAGCEnabled())
    {
        totalGainLinear = juce::Decibels::decibelsToGain(eqState->totalGainDb);
    }

    // ── 最適化: 有効なバンドがない、かつトータルゲインが0dBの場合は計算をスキップ ──
    if (numActiveBands == 0 && std::abs(totalGainLinear - 1.0f) < EQ_UNITY_GAIN_EPSILON)
    {
        if (outMagnitudesL) std::fill_n(outMagnitudesL, numPoints, 1.0f);
        if (outMagnitudesR) std::fill_n(outMagnitudesR, numPoints, 1.0f);
        return;
    }

    const float totalGainSq = totalGainLinear * totalGainLinear;

    for (int i = 0; i < numPoints; ++i)
    {
        // 各バンドの応答を計算
        // 二乗マグニチュードを積算して最後にsqrtすることで、ループ内のsqrtを回避
        float totalMagSqL = totalGainSq;
        float totalMagSqR = totalGainSq;

        // 事前計算された z (e^jw) を使用
        // これにより、ここで sin/cos を計算する必要がなくなる
        const std::complex<double> z = zArray[i];

        for (int b = 0; b < numActiveBands; ++b)
        {
            const auto& band = activeBands[b];
            float magSq = EQProcessor::getMagnitudeSquared(band.coeffs, z);

            // 数値安定性のため、NaN/Infの伝播を防止
            if (!std::isfinite(magSq))
                magSq = 1.0f;

            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Left)
                totalMagSqL *= magSq;
            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Right)
                totalMagSqR *= magSq;
        }

        // 最終的なNaNチェック
        if (outMagnitudesL)
        {
            float val = std::sqrt(totalMagSqL);
            outMagnitudesL[i] = std::isfinite(val) ? val : 1.0f;
        }
        if (outMagnitudesR)
        {
            float val = std::sqrt(totalMagSqR);
            outMagnitudesR[i] = std::isfinite(val) ? val : 1.0f;
        }
    }
}

//--------------------------------------------------------------
// getProcessingSampleRate
//--------------------------------------------------------------
double AudioEngine::getProcessingSampleRate() const
{
    const double sr = currentSampleRate.load();
    if (sr <= 0.0) return 0.0;

    int factor = manualOversamplingFactor.load();
    int actualFactor = 1;

    if (factor > 0)
    {
        if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
            actualFactor = factor;
    }
    else
    {
        // Auto
        if (sr <= 96000.0)       actualFactor = 8;
        else if (sr <= 192000.0) actualFactor = 4;
        else if (sr <= 384000.0) actualFactor = 2;
        else                     actualFactor = 1;
    }

    // 制限: サンプルレートに応じた最大倍率を適用
    int maxFactor = 1;
    if (sr <= 96000.0)       maxFactor = 8;
    else if (sr <= 192000.0) maxFactor = 4;
    else if (sr <= 384000.0) maxFactor = 2;

    actualFactor = std::min(actualFactor, maxFactor);

    return sr * static_cast<double>(actualFactor);
}

//--------------------------------------------------------------
// prepareToPlay
//--------------------------------------------------------------
void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    // パラメータ検証 (Parameter Validation)
    // 不正なパラメータから保護
    double safeSampleRate = sampleRate;
    if (safeSampleRate <= 0.0 || safeSampleRate > SAFE_MAX_SAMPLE_RATE || !std::isfinite(safeSampleRate))
    {
        jassertfalse; // デバッグビルドで警告
        safeSampleRate = 48000.0; // デフォルト値
    }

    if (samplesPerBlockExpected <= 0)
    {
        jassertfalse;
        samplesPerBlockExpected = 512; // フォールバックして続行
    }

    // ASIO同期: デバイスの実際のブロックサイズを使用する
    // SAFE_MAX_BLOCK_SIZEで固定すると、ASIOのブロックサイズ変更に追従できず破綻するため。
    const int bufferSize = samplesPerBlockExpected;

    // サンプルレート変更検知
    const bool rateChanged = (std::abs(currentSampleRate.load() - safeSampleRate) > 1e-6);
    // ブロックサイズ変更検知 (FFTConvolverのパーティションサイズ最適化のため)
    const bool blockSizeChanged = (maxSamplesPerBlock.load() != bufferSize);

    maxSamplesPerBlock.store(bufferSize);
    currentSampleRate.store(safeSampleRate);

    // DSP再構築リクエスト (Audio Thread Safe)
    // MessageManagerへのアクセスやメモリ確保を避けるため、フラグを立ててTimerで処理する
    rebuildRequested.store(true, std::memory_order_release);

    audioFifo.reset();

    // レベルメーターのリセット
    inputLevelDb.store(LEVEL_METER_MIN_DB);
    outputLevelDb.store(LEVEL_METER_MIN_DB);

    // ===== bypass 状態の初期化 =====
    // 再生中のリアルタイムな更新は getNextAudioBlock() で行われる
    eqBypassActive.store (eqBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);
    convBypassActive.store (convBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);

}

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    // UIコンポーネント(uiEqProcessor等)へのアクセスやMKLメモリ確保を行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // 新しいDSPコアを作成
    DSPCore::Ptr newDSP = std::make_shared<DSPCore>();

    // UIプロセッサから状態をコピー
    newDSP->eq.syncStateFrom(uiEqProcessor); // 最適化: ValueTreeを経由せず直接同期
    newDSP->convolver.syncStateFrom(uiConvolverProcessor);

    // 準備
    newDSP->prepare(sampleRate, samplesPerBlock, ditherBitDepth.load(), manualOversamplingFactor.load(), oversamplingType.load());

    // 最適化: 現在のDSPから有効なオーバーサンプリング済みIRを再利用する
    // これにより、EQ変更などのたびにIRリビルド（音切れ）が発生するのを防ぐ
    bool irReused = false;
    DSPCore::Ptr current = activeDSP;

    if (current)
    {
        // 既存のDSPからAGCの状態を引き継ぎ、ゲインの急変を防ぐ
        newDSP->eq.syncGlobalStateFrom(current->eq);
    }

    if (current && std::abs(current->sampleRate - sampleRate) < 1e-6 &&
        current->oversamplingFactor == newDSP->oversamplingFactor && newDSP->oversamplingFactor > 1)
    {
        // IRの生成条件（ファイル、位相設定、長さ）が一致しているか確認
        if (newDSP->convolver.getIRName() == current->convolver.getIRName() &&
            newDSP->convolver.getUseMinPhase() == current->convolver.getUseMinPhase() &&
            std::abs(newDSP->convolver.getTargetIRLength() - current->convolver.getTargetIRLength()) < 0.001f)
        {
            // 既存のConvolutionオブジェクト（計算済みIRデータ）を再利用
            newDSP->convolver.syncStateFrom(current->convolver);

            // UIで変更された可能性のあるパラメータ（Mix, Bypass等）を再適用
            newDSP->convolver.syncParametersFrom(uiConvolverProcessor);
            irReused = true;
        }
    }

    // オーバーサンプリング有効時、UIからコピーされたIR(1xレート)はDSP(Nxレート)にとって不適切です。
    // そのため、正しいサンプルレートでIRを再構築(リサンプリング)する必要があります。
    // これを行わないと、IRのピッチが変わり、レイテンシー補正も誤った値になります。
    //
    // 重要: ここで newDSP (ローカル変数) に対してリビルドを行うため、
    // Audio Thread で稼働中の currentDSP とは完全に独立しており、競合は発生しない。
    if (!irReused && newDSP->oversamplingFactor > 1 && newDSP->convolver.isIRLoaded())
    {
        // 同期的にリビルドを実行 (再生中の音切れやピッチズレを防ぐため、準備完了まで待機)
        newDSP->convolver.rebuildAllIRsSynchronous();
    }

    // Fix: Refresh Convolver latency
    // IRのロードやリビルド、あるいは状態コピーによってレイテンシーが確定した後、
    // 再度prepareToPlayを呼び出してlatencySmootherの初期値を正しいレイテンシーに合わせる。
    // これにより、DSP切り替え時にレイテンシーが0からランプすることによるグリッチ（ピッチ揺れ）を防ぐ。
    const double processingRate = sampleRate * static_cast<double>(newDSP->oversamplingFactor);
    const int processingBlockSize = samplesPerBlock * static_cast<int>(newDSP->oversamplingFactor);
    newDSP->convolver.prepareToPlay(processingRate, processingBlockSize);

    // ==================================================================
    // 【Issue 5 修正】新DSPにFade-in Rampを開始（42ms ≈ 2048サンプル @48kHz）
    // これで新出力が0から滑らかに立ち上がる
    // ==================================================================
    newDSP->fadeInSamplesLeft.store(DSPCore::FADE_IN_SAMPLES, std::memory_order_relaxed);

    commitNewDSP(newDSP);
}

void AudioEngine::commitNewDSP(DSPCore::Ptr newDSP)
{
    // 1. Update the atomic raw pointer for the Audio Thread (Wait-free)
    currentDSP.store(newDSP.get(), std::memory_order_release);

    // 2. Move the previous active DSP to the trash bin
    if (activeDSP)
    {
        const juce::ScopedLock sl(trashBinLock);
        trashBinPending.push_back(activeDSP);
    }

    // 3. Take ownership of the new DSP
    activeDSP = newDSP;
}

void AudioEngine::timerCallback()
{
    // ── DSP再構築リクエストの処理 ──
    // Audio Thread (prepareToPlay) からのリクエストを Message Thread で処理する
    // 【Parameter安全設計】
    // Audio Thread内でのメモリ確保や重い初期化処理を回避するため、
    // フラグ(rebuildRequested)を介してMessage Threadで安全に再構築を実行する。
    if (rebuildRequested.exchange(false, std::memory_order_acquire))
    {
        const double sr = currentSampleRate.load();
        const int bs = maxSamplesPerBlock.load();
        if (sr > 0.0 && bs > 0)
        {
            uiConvolverProcessor.prepareToPlay(sr, bs);
            uiEqProcessor.prepareToPlay(sr, bs);
            uiConvolverProcessor.setBypass(convBypassRequested.load(std::memory_order_relaxed));
            requestRebuild(sr, bs);
            sendChangeMessage();
        }
    }

    std::vector<DSPCore::Ptr> toDelete;

    {
        const juce::ScopedLock sl(trashBinLock);
        const uint32 now = juce::Time::getMillisecondCounter();

        // 1. Move pending items to main trash bin with timestamp
        for (const auto& p : trashBinPending)
            trashBin.push_back({p, now});
        trashBinPending.clear();

        // 2. Identify items to delete (older than 2000ms)
        // This ensures that any Audio Thread processing cycle (typically <100ms)
        // that might have started using the pointer has finished.
        auto it = std::remove_if(trashBin.begin(), trashBin.end(),
                                 [now](const auto& entry) {
                                     // Handle wrap-around of uint32 roughly
                                     return (now >= entry.second) ? (now - entry.second > 2000)
                                                                  : (now + (std::numeric_limits<uint32>::max() - entry.second) > 2000);
                                 });

        for (auto i = it; i != trashBin.end(); ++i)
            toDelete.push_back(i->first);

        trashBin.erase(it, trashBin.end());
    }

    // Lock解放後にデストラクタを実行 (stopThread等の重い処理をロック外で行う)
    toDelete.clear();

    // 3. 内部プロセッサのクリーンアップを実行
    // 現在アクティブなDSPの内部ゴミ箱も掃除する
    // Note: activeDSP is safe to access here (Message Thread)
    if (activeDSP)
    {
        activeDSP->eq.cleanup();
        activeDSP->convolver.cleanup();
    }

    // UI用プロセッサのクリーンアップ
    uiEqProcessor.cleanup();
    uiConvolverProcessor.cleanup();
}

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    // UIプロセッサからの構造変更（プリセットロード、IRロードなど）を検知
    if (source == &uiEqProcessor || source == &uiConvolverProcessor)
    {
        const double sr = currentSampleRate.load();
        if (sr <= 0.0) return;

        // DSPグラフを安全に再構築
        requestRebuild(sr, maxSamplesPerBlock.load());

        // UIに更新を通知 (MainWindowが受け取る)
        sendChangeMessage();
    }
}

void AudioEngine::eqBandChanged(EQProcessor* processor, int bandIndex)
{
    if (processor == &uiEqProcessor)
    {
        if (activeDSP)
            activeDSP->eq.syncBandNodeFrom(uiEqProcessor, bandIndex);
    }
}

void AudioEngine::eqGlobalChanged(EQProcessor* processor)
{
    if (processor == &uiEqProcessor)
    {
        if (activeDSP) {
            // syncGlobalStateFrom は AGC の実行状態も上書きしてしまうため、
            // UIからの変更通知では、UIが管理するパラメータのみを個別に設定する。
            // これにより、アクティブなDSPのAGC状態がリセットされるのを防ぐ。
            activeDSP->eq.setTotalGain(uiEqProcessor.getTotalGain());
            activeDSP->eq.setAGCEnabled(uiEqProcessor.getAGCEnabled());
        }
    }
}

void AudioEngine::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor == &uiConvolverProcessor)
    {
        if (activeDSP)
            activeDSP->convolver.syncParametersFrom(uiConvolverProcessor);
    }
}

//--------------------------------------------------------------
// releaseResources
// デバイス停止時に呼ばれる（Audio Thread停止後）
// JUCE v8.0.12 完全対応版（MMCSSはJUCEが自動管理）
//--------------------------------------------------------------
void AudioEngine::releaseResources()
{
    // サンプルレートをリセット (描画停止用)
    currentSampleRate.store(0.0);

    // レベルをリセット
    inputLevelDb.store(-120.0f);
    outputLevelDb.store(-120.0f);

    // ==================================================================
    // 【Issue 2 完全解消】手動MMCSS revertを削除
    // 理由:
    //   1. JUCE 8.0.12 の setMMCSSModeEnabled() が内部で管理
    //   2. mmcssHandle はローカル変数だったため未定義エラー発生
    //   3. 手動revertは不要・リークリスクあり → JUCEに任せる
    // ==================================================================
}

//--------------------------------------------------------------
// getNextAudioBlock - オーディオ処理コールバック (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    1. メモリ割り当て禁止 (No memory allocation): new, malloc, vector::resize, AudioBuffer::setSize 等はNG。
//    2. ロック禁止 (No locks): Mutex, CriticalSection 等によるブロックはNG。
//    3. システムコール禁止 (No system calls): ファイルI/O, コンソール出力(printf) 等はNG。
//    4. 待機禁止 (No waiting): sleep や 重い計算によるストールを避ける。IRの再ロードもNG。
//    5. 禁止API: AudioBlock::allocate, AudioBlock::copyFrom (確保伴うもの), FFT::performFrequencyOnlyForwardTransform (事前確保なしはNG)
//    6. std::vector使用時は、必ず AudioBuffer / 生ポインタを wrap する形で使用すること。
//--------------------------------------------------------------
void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    // (A) Denormal対策 (重要: CPU負荷スパイク防止)
    juce::ScopedNoDenormals noDenormals;

#if JUCE_INTEL
    // MKL/AVX最適化のためにFTZ/DAZフラグを明示的に設定
    // ScopedNoDenormalsでも設定されるが、MKLの要件として明示しておく
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#if JUCE_DSP_USE_INTEL_MKL
    // VML (Vector Math Library) のDenormal扱いをゼロに設定
    // vdHypot, vdLn 等のパフォーマンス低下を防ぐ
    // この設定はスレッドローカルなので、オーディオスレッドで毎回設定する必要がある
    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#endif

    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    // サンプル数の妥当性チェック
    // maxSamplesPerBlock.load() (Atomic) の代わりに定数 SAFE_MAX_BLOCK_SIZE を使用する。
    // これにより、Message Threadでの更新との競合を回避し、DSPCoreのバッファ確保サイズ(SAFE_MAX_BLOCK_SIZE)に基づく安全なチェックを行う。
    if (numSamples <= 0 || numSamples > SAFE_MAX_BLOCK_SIZE)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // startSampleの妥当性チェック
    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }


    // DSPコアの取得 (Atomic Load - Raw Pointer)
    // shared_ptrの参照カウント操作(atomic RMW)を回避し、完全なWait-freeを実現
    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);

    if (dsp != nullptr)
    {
        // 安全対策: サンプルレート不整合チェック
        // DSPのサンプルレートとエンジンの現在のサンプルレートが一致しない場合、
        // レート変更処理中とみなし、グリッチを防ぐために無音を出力する。
        const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
        if (std::abs(dsp->sampleRate - engineSampleRate) > 1e-6)
        {
            // 不整合時はレベルメーターもリセットして誤表示を防ぐ
            inputLevelDb.store(LEVEL_METER_MIN_DB);
            outputLevelDb.store(LEVEL_METER_MIN_DB);
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // パラメータのロード
        // 【Parameter安全設計】
        // Audio ThreadではAtomic変数の読み取りのみを行い、ロックやメモリ確保を伴う処理は行わない。
        // 構造変更が必要な場合は、別途フラグやUIスレッド経由で再構築を行う。
        const bool eqBypassed = eqBypassRequested.load(std::memory_order_acquire);
        const bool convBypassed = convBypassRequested.load(std::memory_order_acquire);
        const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
        const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
        const bool softClip = softClipEnabled.load(std::memory_order_relaxed);
        const float satAmt = saturationAmount.load(std::memory_order_relaxed);

        // UI表示用の状態更新
        if (eqBypassActive.load(std::memory_order_relaxed) != eqBypassed)
            eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        if (convBypassActive.load(std::memory_order_relaxed) != convBypassed)
            convBypassActive.store(convBypassed, std::memory_order_relaxed);

        // 処理委譲
        dsp->process(bufferToFill, audioFifo, audioFifoBuffer, inputLevelDb, outputLevelDb, {eqBypassed, convBypassed, order, analyzerSource, softClip, satAmt}); // スマートポインタでDSPを呼び出し
    }
    else
    {
        bufferToFill.clearActiveBufferRegion();
    }
}

//--------------------------------------------------------------
// DSPCore Implementation
//--------------------------------------------------------------
AudioEngine::DSPCore::DSPCore() = default;

void AudioEngine::DSPCore::prepare(double newSampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType)
{
    this->sampleRate = newSampleRate;

    int targetFactor = 1;
    if (manualOversamplingFactor > 0)
    {
        // 手動設定
        if (manualOversamplingFactor == 8)      targetFactor = 8;
        else if (manualOversamplingFactor == 4) targetFactor = 4;
        else if (manualOversamplingFactor == 2) targetFactor = 2;
        else                                    targetFactor = 1;
    }
    else
    {
        // 自動設定 (デフォルト)
        if (newSampleRate <= 96000.0)       targetFactor = 8;
        else if (newSampleRate <= 192000.0) targetFactor = 4;
        else if (newSampleRate <= 384000.0) targetFactor = 2;
        else                                targetFactor = 1;
    }

    // 制限: サンプルレートに応じた最大倍率を適用
    int maxFactor = 1;
    if (newSampleRate <= 96000.0)       maxFactor = 8;
    else if (newSampleRate <= 192000.0) maxFactor = 4;
    else if (newSampleRate <= 384000.0) maxFactor = 2;

    targetFactor = std::min(targetFactor, maxFactor);

    size_t factorLog2 = 0;
    if (targetFactor >= 8)      factorLog2 = 3;
    else if (targetFactor >= 4) factorLog2 = 2;
    else if (targetFactor >= 2) factorLog2 = 1;
    else                        factorLog2 = 0;

    oversamplingFactor = (size_t)1 << factorLog2;

    // ==================================================================
    // 【Issue 3 完全修正】内部最大バッファサイズの計算（推奨A）
    // 固定で SAFE_MAX_BLOCK_SIZE × 8 を確保
    // 理由:
    //   ・OS=8x時のupBlockサイズを完全にカバー
    //   ・RCU再構築（IRロード・プリセット切替・OS変更）ごとにresizeしない
    //   ・MKLAllocator + 64byteアライメントの最適化が最大限活きる
    //   ・将来16x OS対応もこの定数1箇所変更だけで済む
    // ==================================================================
    constexpr int MAX_OS_FACTOR = 8;
    const int inputMaxBlock     = SAFE_MAX_BLOCK_SIZE;
    const int internalMaxBlock  = inputMaxBlock * MAX_OS_FACTOR;

    maxSamplesPerBlock   = inputMaxBlock;
    maxInternalBlockSize = internalMaxBlock;

    // === バッファ確保（ここが核心）===
    // 初回 or サイズ不足時のみresize（以後ほぼ触らない）
    if (alignedL.size() < static_cast<size_t>(internalMaxBlock))
    {
        alignedL.resize(internalMaxBlock);
        alignedR.resize(internalMaxBlock);

        // 明示的ゼロクリア（Denormal/NaN防止）
        juce::FloatVectorOperations::clear(alignedL.data(), internalMaxBlock);
        juce::FloatVectorOperations::clear(alignedR.data(), internalMaxBlock);
    }

    if (factorLog2 > 0)
    {
        auto filterType = (oversamplingType == OversamplingType::LinearPhase)
                          ? juce::dsp::Oversampling<double>::filterHalfBandFIREquiripple
                          : juce::dsp::Oversampling<double>::filterHalfBandPolyphaseIIR;
        oversampling = std::make_unique<juce::dsp::Oversampling<double>>(2, factorLog2, filterType);
        oversampling->initProcessing(SAFE_MAX_BLOCK_SIZE);
    }
    else
    {
        oversampling.reset();
    }

    const double processingRate = newSampleRate * static_cast<double>(oversamplingFactor);
    const int processingBlockSize = samplesPerBlock * static_cast<int>(oversamplingFactor);

    // プロセッサの準備
    // Convolverには実際のブロックサイズを渡す (パーティションサイズ決定やLoaderThreadで使用)
    convolver.prepareToPlay(processingRate, processingBlockSize);

    // EQも内部最大サイズで準備（より安全）
    eq.prepareToPlay(processingRate, internalMaxBlock);

    // 出力段(processOutput)で実行されるため、オーバーサンプリング前のレートとサイズを使用する
    dcBlockerL.prepare(newSampleRate, SAFE_MAX_BLOCK_SIZE);
    dcBlockerR.prepare(newSampleRate, SAFE_MAX_BLOCK_SIZE);

    // 入力段用DCBlockerの準備
    inputDCBlockerL.prepare(newSampleRate, SAFE_MAX_BLOCK_SIZE);
    inputDCBlockerR.prepare(newSampleRate, SAFE_MAX_BLOCK_SIZE);

    // オーバーサンプリング後のDC除去用 (1Hzカットオフ)
    osDCBlockerL.init(processingRate, 1.0);
    osDCBlockerR.init(processingRate, 1.0);

    // ディザの準備 (出力段で行うため元のサンプルレート)
    dither.prepare(newSampleRate, bitDepth);
    this->ditherBitDepth = bitDepth; // DSPCoreのメンバーに保存

    // 【Issue 5】Fade-inカウンタをリセット
    fadeInSamplesLeft.store(0, std::memory_order_relaxed);
}

void AudioEngine::DSPCore::reset()
{
    convolver.reset();
    eq.reset();
    dcBlockerL.reset();
    dcBlockerR.reset();
    inputDCBlockerL.reset();
    inputDCBlockerR.reset();
    osDCBlockerL.reset();
    osDCBlockerR.reset();
    dither.reset();
    if (oversampling)
        oversampling->reset();
    if (!alignedL.empty())
        // size() はすでに internalMaxBlockSize になっているので安全
        juce::FloatVectorOperations::clear(alignedL.data(), static_cast<int>(alignedL.size()));
    if (!alignedR.empty())
        // size() はすでに internalMaxBlockSize になっているので安全
        juce::FloatVectorOperations::clear(alignedR.data(), static_cast<int>(alignedR.size()));
}

void AudioEngine::DSPCore::process(const juce::AudioSourceChannelInfo& bufferToFill,
                                  juce::AbstractFifo& audioFifo,
                                  juce::AudioBuffer<float>& audioFifoBuffer,
                                  std::atomic<float>& inputLevelDb,
                                  std::atomic<float>& outputLevelDb,
                                  const ProcessingState& state) // ProcessingState構造体でパラメータを受け取る
{
    const int numSamples = bufferToFill.numSamples;

    // バッファサイズ超過ガード (Buffer Overrun Protection)
    if (numSamples > maxSamplesPerBlock)
    {
        // この状況は通常発生しないが、万が一ホストが予期せぬサイズのバッファを渡してきた場合の安全策
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // ==================================================================
    // 【Issue 3 追加防御】オーバーラン即検出（リリースでも有効）
    // オーバーサンプリング有効時にupBlockサイズが内部バッファを超えないことを保証
    // ==================================================================
    if (oversampling)
    {
        const int expectedUpSize = numSamples * static_cast<int>(oversamplingFactor);

        // Fix: Releaseビルドでも確実にチェックし、バッファ破壊を防ぐ
        if (expectedUpSize > maxInternalBlockSize)
        {
            jassertfalse; // Debug時は停止
            bufferToFill.clearActiveBufferRegion(); // 無音を出力
            return;
        }
    }

    processInput(bufferToFill, numSamples);

    //----------------------------------------------------------
    // AudioBlockの構築 (AlignedBufferを使用)
    //----------------------------------------------------------
    double* channels[2] = { alignedL.data(), alignedR.data() };
    juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);

    //----------------------------------------------------------
    // 入力レベル計算
    //----------------------------------------------------------
    const float inputDb = measureLevel(processBlock);
    inputLevelDb.store(inputDb, std::memory_order_relaxed);

    // ── Analyzer Input Tap (Pre-DSP) ──
    if (state.analyzerSource == AnalyzerSource::Input)
    {
        pushToFifo(processBlock, audioFifo, audioFifoBuffer);
    }

    //----------------------------------------------------------
    // オーバーサンプリング処理ブロック
    //----------------------------------------------------------
    // バッファ全体ではなく、有効なサンプル数のみをラップする (重要)
    juce::dsp::AudioBlock<double> originalBlock = processBlock; // 元サイズを保存

    // アップサンプリング
    if (oversampling)
    {
        processBlock = oversampling->processSamplesUp(originalBlock);

        // [追加] 実際のアップサンプリング後サイズが内部バッファを超えていないか最終確認
        if (processBlock.getNumSamples() > static_cast<size_t>(maxInternalBlockSize))
        {
            jassertfalse;
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // オーバーサンプリング直後に高精度DC除去を適用
        // これにより、後段のDSP処理（Convolver/EQ）にクリーンな信号を渡す
        const int numOSSamples = (int)processBlock.getNumSamples();
        if (processBlock.getNumChannels() > 0)
            osDCBlockerL.process(processBlock.getChannelPointer(0), numOSSamples);
        if (processBlock.getNumChannels() > 1)
            osDCBlockerR.process(processBlock.getChannelPointer(1), numOSSamples);
    }

    int numProcSamples = (int)processBlock.getNumSamples();
    int numProcChannels = (int)processBlock.getNumChannels(); // 通常は2

    //----------------------------------------------------------
    // DSP処理チェーン (Dynamic Processing Order)
    //----------------------------------------------------------
    // プロセッサには AudioBlock を直接渡す (AudioBuffer作成によるmalloc回避)
    if (state.order == ProcessingOrder::ConvolverThenEQ) // stateから読み出し
    {
        // 1. Convolver
        if (!state.convBypassed) // stateから読み出し
            convolver.process(processBlock);
        // 2. EQ
        if (!state.eqBypassed) // stateから読み出し
            eq.process(processBlock);
    }
    else
    {
        // 1. EQ
        if (!state.eqBypassed) // stateから読み出し
            eq.process(processBlock);
        // 2. Convolver
        if (!state.convBypassed) // stateから読み出し
            convolver.process(processBlock);
    }

    //----------------------------------------------------------
    // ソフトクリッピング (Soft Clipping)
    // 配置: ダウンサンプリング前に行うことで、倍音成分の折り返しノイズ(エイリアシング)を低減する。
    //----------------------------------------------------------
    if (state.softClipEnabled) // stateから読み出し
    {
        const double sat = static_cast<double>(state.saturationAmount); // stateから読み出し
        const double CLIP_THRESHOLD = 0.95 - 0.45 * sat;
        const double CLIP_KNEE      = 0.05 + 0.35 * sat;
        const double CLIP_ASYMMETRY = 0.10 * sat;
        const double CLIP_START = CLIP_THRESHOLD - CLIP_KNEE;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            for (int i = 0; i < numProcSamples; ++i)
            {
                if (std::abs(data[i]) > CLIP_START)
                    data[i] = musicalSoftClip(data[i], CLIP_THRESHOLD, CLIP_KNEE, CLIP_ASYMMETRY);
            }
        }
    }

    //----------------------------------------------------------

    // ダウンサンプリング (結果は processBuffer に書き戻される)
    if (oversampling)
    {
        oversampling->processSamplesDown(originalBlock);
        processBlock = originalBlock;
    }

    //----------------------------------------------------------
    // 出力レベル計算 (DC除去後のクリーンな信号で計測)
    //----------------------------------------------------------
    // オーバーサンプリング有効時は、ダウンサンプリング後の信号(originalBlock)を使用する
    const float outputDb = measureLevel(originalBlock);
    outputLevelDb.store(outputDb, std::memory_order_relaxed);

    // ── Analyzer Output Tap (Post-DSP) ──
    if (state.analyzerSource == AnalyzerSource::Output)
    {
        pushToFifo(originalBlock, audioFifo, audioFifoBuffer);
    }

    processOutput(bufferToFill, numSamples);

    // === 【Issue 5 追加】新DSP切り替え時のFade-in Ramp（最終出力に適用）===
    {
        int fadeLeft = fadeInSamplesLeft.load(std::memory_order_relaxed);
        if (fadeLeft > 0)
        {
            const int rampThisBlock = std::min(numSamples, fadeLeft);
            const float gainStep = 1.0f / static_cast<float>(FADE_IN_SAMPLES);
            auto* buffer = bufferToFill.buffer;
            const int startSample = bufferToFill.startSample;
            const int numChannels = buffer->getNumChannels();

            // Optimize: Channel-first loop for cache locality (Planar buffer friendly)
            for (int ch = 0; ch < numChannels; ++ch)
            {
                float* data = buffer->getWritePointer(ch, startSample);
                for (int i = 0; i < rampThisBlock; ++i)
                {
                    const float gain = static_cast<float>(FADE_IN_SAMPLES - fadeLeft + i) * gainStep;
                    data[i] *= gain;
                }
            }
            fadeInSamplesLeft.store(fadeLeft - rampThisBlock, std::memory_order_relaxed);
        }
    }
}

float AudioEngine::DSPCore::measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept
{
    double maxLevel = 0.0;
    const int numChannels = (int)block.getNumChannels();
    const int numSamples = (int)block.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        // getMagnitudeは内部でSIMD化されたfindMinAndMaxを使用するため高速
        auto range = juce::FloatVectorOperations::findMinAndMax(block.getChannelPointer(ch), numSamples);
        const double level = std::max(std::abs(range.getStart()), std::abs(range.getEnd()));
        if (level > maxLevel) maxLevel = level;
    }

    return (maxLevel > static_cast<double>(LEVEL_METER_MIN_MAG)) ? static_cast<float>(juce::Decibels::gainToDecibels(maxLevel)) : LEVEL_METER_MIN_DB;
}

void AudioEngine::DSPCore::pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                                      juce::AbstractFifo& audioFifo,
                                      juce::AudioBuffer<float>& audioFifoBuffer) const noexcept
{
    const int numSamples = (int)block.getNumSamples();

    // 安全性確認: FIFOバッファのサイズが初期化時から変更されていないことを保証
    jassert (audioFifoBuffer.getNumSamples() == audioFifo.getTotalSize());

    // FIFO空き容量チェック (Overflow Protection)
    // 部分書き込み対応: 空き容量分だけ書き込む (完全ドロップによる時間軸ジャンプを軽減)
    int start1, size1, start2, size2;
    audioFifo.prepareToWrite(numSamples, start1, size1, start2, size2);

    if (size1 + size2 <= 0)
        return;

    const double* l = block.getChannelPointer(0);
    const double* r = (block.getNumChannels() > 1) ? block.getChannelPointer(1) : nullptr;
    const bool hasRightChannel = (audioFifoBuffer.getNumChannels() > 1);

    // AVX2 double→float 変換ヘルパー (4 doubles → 4 floats)
    auto convertBlock = [&](const double* srcL, const double* srcR,
                             float* dstL, float* dstR, int n) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = n / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(srcL + i);
            __m128  fL = _mm256_cvtpd_ps(vL);
            _mm_storeu_ps(dstL + i, fL);
            if (dstR && srcR)
            {
                __m256d vR = _mm256_loadu_pd(srcR + i);
                _mm_storeu_ps(dstR + i, _mm256_cvtpd_ps(vR));
            }
            else if (dstR)
            {
                _mm_storeu_ps(dstR + i, fL); // モノ → ステレオ
            }
        }
        for (; i < n; ++i)
        {
            dstL[i] = static_cast<float>(srcL[i]);
            if (dstR) dstR[i] = srcR ? static_cast<float>(srcR[i]) : dstL[i];
        }
#else
        for (int i = 0; i < n; ++i)
        {
            dstL[i] = static_cast<float>(srcL[i]);
            if (dstR) dstR[i] = srcR ? static_cast<float>(srcR[i]) : dstL[i];
        }
#endif
    };

    if (size1 > 0)
    {
        convertBlock(l, r,
                     audioFifoBuffer.getWritePointer(0, start1),
                     hasRightChannel ? audioFifoBuffer.getWritePointer(1, start1) : nullptr,
                     size1);
        l += size1;
        if (r != nullptr) r += size1;
    }

    if (size2 > 0)
    {
        convertBlock(l, r,
                     audioFifoBuffer.getWritePointer(0, start2),
                     hasRightChannel ? audioFifoBuffer.getWritePointer(1, start2) : nullptr,
                     size2);
    }

    audioFifo.finishedWrite(size1 + size2);
}

void AudioEngine::DSPCore::processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    const int effectiveInputChannels = std::min(buffer->getNumChannels(), 2);

    //----------------------------------------------------------
    // 入力データを processBuffer (double) にコピー
    //----------------------------------------------------------
    // ループ分割による分岐排除と最適化
    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        // float (I/O) -> double (DSP) 変換
        const float* src = buffer->getReadPointer(ch, startSample);
        double* dst = (ch == 0) ? alignedL.data() : alignedR.data();
        auto& blocker = (ch == 0) ? inputDCBlockerL : inputDCBlockerR;

        for (int i = 0; i < numSamples; ++i)
        {
            float v = src[i];
            // 安全対策: NaN/Infチェック (不正な入力値によるノイズ発生を防止)
            if (!std::isfinite(v)) v = 0.0f;
            // 入力信号のサニタイズ: 極端な値をクランプしてフィルタ発散や矩形波ノイズを防ぐ
            // ドライバによっては未初期化バッファ等で巨大な値を返すことがあるため
            double val = static_cast<double>(juce::jlimit(-2.0f, 2.0f, v));
            dst[i] = blocker.process(val);
        }
    }

    // 入力がないチャンネル、または余剰チャンネルはクリア
    // ただし、Mono->Stereo展開を行う場合はCh 1のクリアをスキップする (直後に上書きされるため)
    // ロジック整理:
    // 1. 入力が1chで出力が2ch以上の場合 -> Ch 1 (R) はコピーされるのでクリア不要。Ch 2以降をクリア。
    // 2. それ以外 -> 入力チャンネル数以降をすべてクリア。
    // AlignedBufferは常に2ch分 (L/R) 用意されていると仮定
    const bool expandMono = (effectiveInputChannels == 1);
    const int clearStartCh = expandMono ? 2 : effectiveInputChannels;

    for (int ch = clearStartCh; ch < 2; ++ch)
    {
        double* dst = (ch == 0) ? alignedL.data() : alignedR.data();
        juce::FloatVectorOperations::clear(dst, numSamples);
    }

    // ── Mono -> Stereo 展開 ──
    // 入力が1chのみで、処理バッファが2ch以上ある場合、LchをRchにコピーする
    // これにより、モノラルマイク入力時などでもステレオ処理として扱えるようにし、
    // 後段のステレオエフェクト（Convolver等）での片側無音を防ぐ。
    if (expandMono)
    {
        const double* src = alignedL.data();
        double* dst = alignedR.data();
        // 高速なメモリコピー (double配列)
        std::memcpy(dst, src, numSamples * sizeof(double));
    }
}

// Padé近似による高速tanh (std::exp回避)
// 精度: |x| < 3.0 で誤差 1e-4 以下
static inline double fastTanh(double x) noexcept
{
    if (x >= 3.0) return 1.0;
    if (x <= -3.0) return -1.0;
    const double x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}

// 音楽的なソフトクリッピング関数
// 閾値を超えた信号を滑らかにクリップし、真空管アンプのような温かみのある歪みを加える。
// @param x 入力信号
// @param threshold クリッピングが開始される閾値
// @param knee 閾値周辺のカーブの滑らかさ（ニー）
// @param asymmetry 非対称性の量。正の値で正の波形が、負の値で負の波形がより強くクリップされ、偶数次倍音を生成する。
double AudioEngine::DSPCore::musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept
{
    const double abs_x = std::abs(x);
    const double clip_start = threshold - knee;

    // 安全対策: kneeが極端に小さい場合のゼロ除算防止
    if (knee < 1.0e-9) return (x > threshold) ? threshold : ((x < -threshold) ? -threshold : x);

    // 閾値以下はリニア
    if (abs_x < clip_start)
        return x;

    const double sign = (x > 0.0) ? 1.0 : -1.0;

    // ソフトニー領域 (ブレンド率計算)
    double knee_shape = 1.0;
    if (abs_x < threshold + knee)
    {
        // 3次多項式でスムーズなニー
        const double t = (abs_x - clip_start) / (2.0 * knee);
        knee_shape = t * t * (3.0 - 2.0 * t); // Smoothstep
    }

    const double linear = abs_x;
    // tanhによるソフトクリッピングカーブ
    const double clipped = threshold + knee * fastTanh((abs_x - threshold) / knee);

    // 非対称性の追加（真空管風）
    const double asymmetric_factor = 1.0 + asymmetry * sign * knee_shape;
    return sign * (linear * (1.0 - knee_shape) + clipped * knee_shape) * asymmetric_factor;
}

void AudioEngine::DSPCore::processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;

    // ビット深度に基づくディザリング判定
    // ユーザー設定に従い、32-bit (float/int) でもディザリングを適用する。
    const bool applyDither = (ditherBitDepth > 0);

    //----------------------------------------------------------
    // 統合処理ループ: DC除去 -> ディザリング -> 出力
    // メモリ読み書きの回数を減らし、キャッシュ効率を向上させる
    //----------------------------------------------------------
    for (int ch = 0; ch < buffer->getNumChannels(); ++ch)
    {
        if (ch < 2)
        {
            double* data = (ch == 0) ? alignedL.data() : alignedR.data();
            auto& blocker = (ch == 0) ? dcBlockerL : dcBlockerR;
            float* dst = buffer->getWritePointer(ch, startSample);

            // ── フェーズ1: DCブロッカー (IIR 逐次依存, スカラー) ──
            if (applyDither)
            {
                for (int i = 0; i < numSamples; ++i)
                {
                    double processed = blocker.process(data[i]);
                    data[i] = dither.process(processed, ch); // ディザ後も data[] に保存
                }
            }
            else
            {
                for (int i = 0; i < numSamples; ++i)
                    data[i] = blocker.process(data[i]);
            }

            // ── フェーズ2: double→float 変換 + NaN除去 + クランプ (AVX2 一括) ──
#if defined(__AVX2__)
            {
                int i = 0;
                const int vEnd = numSamples / 16 * 16; // Unroll 4x
                const __m256d vMax  = _mm256_set1_pd(1.0);
                const __m256d vMin  = _mm256_set1_pd(-1.0);
                const __m256d vZero = _mm256_setzero_pd();

                for (; i < vEnd; i += 16)
                {
                    _mm_prefetch(reinterpret_cast<const char*>(data + i + 64), _MM_HINT_T0);

                    // 1
                    __m256d v0 = _mm256_loadu_pd(data + i);
                    __m256d mask0 = _mm256_cmp_pd(v0, v0, _CMP_ORD_Q);
                    v0 = _mm256_blendv_pd(vZero, v0, mask0);
                    v0 = _mm256_min_pd(_mm256_max_pd(v0, vMin), vMax);
                    _mm_storeu_ps(dst + i, _mm256_cvtpd_ps(v0));

                    // 2
                    __m256d v1 = _mm256_loadu_pd(data + i + 4);
                    __m256d mask1 = _mm256_cmp_pd(v1, v1, _CMP_ORD_Q);
                    v1 = _mm256_blendv_pd(vZero, v1, mask1);
                    v1 = _mm256_min_pd(_mm256_max_pd(v1, vMin), vMax);
                    _mm_storeu_ps(dst + i + 4, _mm256_cvtpd_ps(v1));

                    // 3
                    __m256d v2 = _mm256_loadu_pd(data + i + 8);
                    __m256d mask2 = _mm256_cmp_pd(v2, v2, _CMP_ORD_Q);
                    v2 = _mm256_blendv_pd(vZero, v2, mask2);
                    v2 = _mm256_min_pd(_mm256_max_pd(v2, vMin), vMax);
                    _mm_storeu_ps(dst + i + 8, _mm256_cvtpd_ps(v2));

                    // 4
                    __m256d v3 = _mm256_loadu_pd(data + i + 12);
                    __m256d mask3 = _mm256_cmp_pd(v3, v3, _CMP_ORD_Q);
                    v3 = _mm256_blendv_pd(vZero, v3, mask3);
                    v3 = _mm256_min_pd(_mm256_max_pd(v3, vMin), vMax);
                    _mm_storeu_ps(dst + i + 12, _mm256_cvtpd_ps(v3));
                }
                // Remaining
                for (; i < (numSamples / 4 * 4); i += 4)
                {
                    __m256d v = _mm256_loadu_pd(data + i);
                    __m256d mask = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
                    v = _mm256_blendv_pd(vZero, v, mask);
                    v = _mm256_min_pd(_mm256_max_pd(v, vMin), vMax);
                    _mm_storeu_ps(dst + i, _mm256_cvtpd_ps(v));
                }
                for (; i < numSamples; ++i)
                {
                    double val = data[i];
                    if (!std::isfinite(val)) val = 0.0;
                    dst[i] = static_cast<float>(juce::jlimit(-1.0, 1.0, val));
                }
            }
#else
            for (int i = 0; i < numSamples; ++i)
            {
                double val = data[i];
                if (!std::isfinite(val)) val = 0.0;
                dst[i] = static_cast<float>(juce::jlimit(-1.0, 1.0, val));
            }
#endif
        }
        else
        {
            // 3ch以降は使用しないためクリア (ゴミデータ出力防止)
            buffer->clear(ch, startSample, numSamples);
        }
    }
}

void AudioEngine::setEqBypassRequested (bool shouldBypass) noexcept
{
    eqBypassRequested.store (shouldBypass, std::memory_order_release);
}

void AudioEngine::setConvolverBypassRequested (bool shouldBypass) noexcept
{
    convBypassRequested.store (shouldBypass, std::memory_order_release);
}

void AudioEngine::setConvolverUseMinPhase(bool useMinPhase)
{
    uiConvolverProcessor.setUseMinPhase(useMinPhase);
}

bool AudioEngine::getConvolverUseMinPhase() const
{
    return uiConvolverProcessor.getUseMinPhase();
}

void AudioEngine::requestEqPreset (int presetIndex) noexcept
{
    uiEqProcessor.loadPreset (presetIndex);
    sendChangeMessage();
}

void AudioEngine::requestEqPresetFromText(const juce::File& file) noexcept
{
    if (uiEqProcessor.loadFromTextFile(file))
        sendChangeMessage();
}

void AudioEngine::requestConvolverPreset (const juce::File& irFile) noexcept
{
    uiConvolverProcessor.loadImpulseResponse (irFile);
}

void AudioEngine::requestLoadState (const juce::ValueTree& state)
{
    // グローバル設定の読み込み
    if (state.hasProperty("processingOrder"))
        setProcessingOrder((ProcessingOrder)(int)state.getProperty("processingOrder"));

    if (state.hasProperty("softClipEnabled"))
        setSoftClipEnabled(state.getProperty("softClipEnabled"));

    if (state.hasProperty("saturationAmount"))
        setSaturationAmount(state.getProperty("saturationAmount"));

    if (state.hasProperty("analyzerSource"))
        setAnalyzerSource((AnalyzerSource)(int)state.getProperty("analyzerSource"));

    if (state.hasProperty("eqBypassed"))
    {
        bool bypassed = state.getProperty("eqBypassed");
        setEqBypassRequested(bypassed);
        uiEqProcessor.setBypass(bypassed);
    }

    if (state.hasProperty("convBypassed"))
    {
        bool bypassed = state.getProperty("convBypassed");
        setConvolverBypassRequested(bypassed);
        // ConvolverProcessor::setState でも設定される可能性があるが、
        // 整合性を保つためにここでも設定する
        uiConvolverProcessor.setBypass(bypassed);
    }

    // EQ
    auto eqState = state.getChildWithName ("EQ");
    if (eqState.isValid())
        uiEqProcessor.setState (eqState);

    // Convolver
    auto convState = state.getChildWithName ("Convolver");
    if (convState.isValid())
        uiConvolverProcessor.setState (convState);

    // UI更新通知
    sendChangeMessage();
}

juce::ValueTree AudioEngine::getCurrentState() const
{
    juce::ValueTree state ("Preset");

    // グローバル設定の保存
    state.setProperty("processingOrder", (int)currentProcessingOrder.load(), nullptr);
    state.setProperty("softClipEnabled", softClipEnabled.load(), nullptr);
    state.setProperty("saturationAmount", saturationAmount.load(), nullptr);
    state.setProperty("analyzerSource", (int)currentAnalyzerSource.load(), nullptr);
    state.setProperty("eqBypassed", eqBypassRequested.load(), nullptr);
    state.setProperty("convBypassed", convBypassRequested.load(), nullptr);

    state.addChild (uiEqProcessor.getState(), -1, nullptr);
    state.addChild (uiConvolverProcessor.getState(), -1, nullptr);
    return state;
}

void AudioEngine::setDitherBitDepth(int bitDepth)
{
    if (ditherBitDepth.load() != bitDepth)
    {
        ditherBitDepth.store(bitDepth);
        const double sr = currentSampleRate.load();
        if (sr > 0.0)
            requestRebuild(sr, maxSamplesPerBlock.load());
    }
}

int AudioEngine::getDitherBitDepth() const
{
    return ditherBitDepth.load();
}

void AudioEngine::setSoftClipEnabled(bool enabled)
{
    softClipEnabled.store(enabled, std::memory_order_relaxed);
}

bool AudioEngine::isSoftClipEnabled() const
{
    return softClipEnabled.load(std::memory_order_relaxed);
}

void AudioEngine::setSaturationAmount(float amount)
{
    saturationAmount.store(juce::jlimit(0.0f, 1.0f, amount), std::memory_order_relaxed);
}

float AudioEngine::getSaturationAmount() const
{
    return saturationAmount.load(std::memory_order_relaxed);
}

void AudioEngine::setOversamplingFactor(int factor)
{
    // 0=Auto, 1, 2, 4, 8
    int newFactor = 0;
    if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
    {
        newFactor = factor;
    }

    if (manualOversamplingFactor.load() != newFactor)
    {
        manualOversamplingFactor.store(newFactor);
        const double sr = currentSampleRate.load();
        if (sr > 0.0)
            requestRebuild(sr, maxSamplesPerBlock.load());
    }
}

int AudioEngine::getOversamplingFactor() const
{
    return manualOversamplingFactor.load();
}

void AudioEngine::setOversamplingType(OversamplingType type)
{
    oversamplingType.store(type);
    const double sr = currentSampleRate.load();
    if (sr > 0.0)
        requestRebuild(sr, maxSamplesPerBlock.load());
}

AudioEngine::OversamplingType AudioEngine::getOversamplingType() const
{
    return oversamplingType.load();
}
