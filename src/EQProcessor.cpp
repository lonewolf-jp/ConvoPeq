//============================================================================
// EQProcessor.cpp  ── v0.2 (JUCE 8.0.12 対応)
//
// 変更履歴:
//   2026-05-03: retireEQState() / reclaimRetiredEQStates() を実装。
//               EQState の解放を退役キュー方式に変更し、メモリリークを修正。
//               この遅延解放は JUCE メッセージスレッドの直列化に依存した実用策であり、
//               形式的な RCU やロックフリー回収機構ではない。
//
// 20 バンドパラメトリックイコライザー処理実装
// 参照：Vadim Zavalishin "The Art of VA Filter Design" (TPT SVF)
//       https://www.w3.org/2011/audio/audio-eq-cookbook.html  (Biquad Coeffs)
//============================================================================
#include "EQProcessor.h"
#include "DspNumericPolicy.h"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <complex>
#include <numeric>
#include <cstring>
#include <regex>

#if defined(__AVX2__) || defined(__FMA__)
 #include <immintrin.h>
#endif

// ============================================================================
// retireEQState: 古い EQState を退役キューに登録する。
// 即時 delete は行わない。
// 理由：SpectrumAnalyzerComponent などが getBandParams() 経由で
//       currentStateRaw をロードしている可能性があり、解放が早すぎると
//       Use-After-Free が発生しうるため。
// 退役キューは reclaimRetiredEQStates() で定期的に処理される。
// ============================================================================
void EQProcessor::retireEQState(EQState* state) noexcept
{
    if (state == nullptr) return;

    std::lock_guard<std::mutex> lock(retiredEQStateMutex);
    if (retiredEQStates.size() < kMaxRetiredEQStates)
    {
        retiredEQStates.push_back(state);
        // 診断用：最大深度を記録
        if (retiredEQStates.size() > maxRetiredDepthObserved)
            maxRetiredDepthObserved = retiredEQStates.size();
    }
    else
    {
        // キュー溢れ時：安全性を最優先し、新しい状態は破棄せずにリークさせる。
        // これにより、UAF のリスクを負うことなく、過負荷時も動作を継続できる。
        ++retiredEQStateDropCount;
        // ログスパム防止：初回と 100 回ごとに出力
        if (retiredEQStateDropCount == 1 || (retiredEQStateDropCount % 100) == 0)
            juce::Logger::writeToLog("EQProcessor: retired EQState queue overflow, dropped "
                                     + juce::String(retiredEQStateDropCount) + " states. Old states leaked.");
        // デバッグビルドではアサートで検出
        jassertfalse;
    }
}

// ============================================================================
// reclaimRetiredEQStates: 退役キュー内の全 EQState を解放する。
// メッセージスレッド（AudioEngine::timerCallback）から定期的に呼び出す。
// ロック保持時間を短くするため、キューをローカルにスワップしてから解放する。
// ============================================================================
void EQProcessor::reclaimRetiredEQStates() noexcept
{
    JUCE_ASSERT_MESSAGE_THREAD;  // メッセージスレッドでの呼び出しを前提とする

    std::deque<EQState*> localQueue;
    {
        std::lock_guard<std::mutex> lock(retiredEQStateMutex);
        if (!retiredEQStates.empty())
            localQueue.swap(retiredEQStates);
    }
    // ロック外で解放（deque のデストラクタに任せる）
    for (auto* state : localQueue)
        delete state;
}

static inline double calculateRMS(const double* data, int numSamples) noexcept
{
    if (data == nullptr || numSamples <= 0)
        return 0.0;

    double sumSq = 0.0;
#if defined(__AVX2__)
    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    __m256d vSumSq = _mm256_setzero_pd();
    for (; i < vEnd; i += 4)
    {
        __m256d vData = _mm256_loadu_pd(data + i);
        vSumSq = _mm256_fmadd_pd(vData, vData, vSumSq);
    }
    alignas(32) double temp[4];
    _mm256_store_pd(temp, vSumSq);
    sumSq = temp[0] + temp[1] + temp[2] + temp[3];
    for (; i < numSamples; ++i)
        sumSq += data[i] * data[i];
#else
    for (int i = 0; i < numSamples; ++i)
        sumSq += data[i] * data[i];
#endif

    // RMS = sqrt(sumSq / n)
    // Audio Thread内でのlibm呼び出しを避けるため、SSE2命令で平方根を計算する
    __m128d n = _mm_set_sd(static_cast<double>(numSamples));
    __m128d vSumSqSd = _mm_set_sd(sumSq);
    __m128d vRms = _mm_sqrt_sd(_mm_setzero_pd(), _mm_div_sd(vSumSqSd, n));
    double rms;
    _mm_store_sd(&rms, vRms);
    return rms;
}

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
EQProcessor::EQProcessor()
{
    // 初期係数ノードの作成
    resetToDefaults();
}

//--------------------------------------------------------------
EQProcessor::~EQProcessor()
{
    juce::Logger::writeToLog("[DIAG EQProcessor] ~EQProcessor: enter");
    if (auto* oldState = currentStateRaw.exchange(nullptr, std::memory_order_acq_rel))
        retireEQState(oldState);

    reclaimRetiredEQStates();   // デストラクタで最終的な解放
    releaseResources();
    juce::Logger::writeToLog("[DIAG EQProcessor] ~EQProcessor: exit");
}

void EQProcessor::releaseResources()
{
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before scratchBuffer.reset");
