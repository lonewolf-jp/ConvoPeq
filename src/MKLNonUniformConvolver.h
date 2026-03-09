//============================================================================
// MKLNonUniformConvolver.h  ── v1.0 (JUCE 8.0.12 / Intel oneMKL 対応)
//
// Intel MKL を使用した Non-Uniform Partitioned Convolution (NUP) エンジン。
// WDL_ConvolutionEngine_Div の低遅延特性を維持しつつ、
// MKL DFTI + AVX2 FMA でスループットを最大化する。
//
// ■ パーティション構成 (3 Layer):
//   Layer 0 (即時): fftSize=512,  partSize=256   ← 最低遅延ブロック
//   Layer 1 (遅延): fftSize=4096, partSize=2048  ← IR 中盤
//   Layer 2 (遅延): fftSize 可変, partSize 可変  ← IR 末尾
//
// ■ スレッド安全設計:
//   SetImpulse()  : Message Thread (prepareToPlay 相当) からのみ呼び出す
//   Add() / Get() : Audio Thread から呼び出す（メモリ確保なし）
//   Reset()       : Message Thread または releaseResources() から呼び出す
//
// ■ 規約遵守:
//   - 64bit double 処理 (MKL_Complex16)
//   - mkl_malloc(64) によるメモリ管理 (new / std::vector 禁止)
//   - Audio Thread 内でのメモリ確保・DftiCommitDescriptor 禁止
//   - FTZ/DAZ は呼び出し元 (ConvolverProcessor) で設定済みを前提とする
//============================================================================
#pragma once

#include <mkl.h>
#include <mkl_dfti.h>
#include <atomic>
#include <JuceHeader.h>  // juce::nextPowerOfTwo, JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR

namespace convo
{

//==============================================================================
// MKLNonUniformConvolver
//
// ステレオ 2ch を個別インスタンスで処理する (呼び出し元が ch 単位で保持)。
// SetImpulse() に渡す impulse は 1ch 分 (モノラル) の double 配列。
//==============================================================================
class MKLNonUniformConvolver
{
public:
    MKLNonUniformConvolver();
    ~MKLNonUniformConvolver();

    //----------------------------------------------------------
    // SetImpulse  ─ Message Thread のみ
    //
    // @param impulse       IR データ (64-byte アライン推奨, 所有権は呼び出し元)
    // @param irLen         IR サンプル数 (> 0)
    // @param blockSize     Audio Thread の呼び出しブロックサイズ
    // @return true=成功, false=パラメータ不正またはMKL初期化失敗
    //----------------------------------------------------------
    bool SetImpulse(const double* impulse, int irLen, int blockSize, double scale = 1.0);

    //----------------------------------------------------------
    // Add  ─ Audio Thread のみ
    // 入力サンプルを畳み込みエンジンへ投入する。
    // @param input   入力バッファ (numSamples サンプル)
    // @param numSamples  ブロックサイズ (SetImpulse 時の blockSize と一致推奨)
    //----------------------------------------------------------
    void Add(const double* input, int numSamples);

    //----------------------------------------------------------
    // Get  ─ Audio Thread のみ
    // 畳み込み結果を output へ書き出す。
    // 出力は Add() 1 ブロック遅延 (Layer0 の partitionSize サンプル) を伴う。
    // @return 実際に書き出したサンプル数 (通常 = numSamples)
    //----------------------------------------------------------
    int Get(double* output, int numSamples);

    //----------------------------------------------------------
    // Reset  ─ Message Thread / releaseResources() から
    // すべての内部バッファをゼロクリアし、位置変数を初期化する。
    //----------------------------------------------------------
    void Reset();

    //----------------------------------------------------------
    // isReady  ─ いつでも呼び出し可
    //----------------------------------------------------------
    bool isReady() const noexcept { return m_ready.load(std::memory_order_acquire); }

    //----------------------------------------------------------
    // getLatency  ─ 出力の先頭レイテンシー (サンプル数)
    // = Layer0 の partitionSize
    //----------------------------------------------------------
    int getLatency() const noexcept { return m_latency; }

private:
    //----------------------------------------------------------
    // Layer  ─ 1 つのパーティション層
    //----------------------------------------------------------
    struct Layer
    {
        // ── 設定値 ──
        int fftSize       = 0;   // FFT サイズ (2 * partSize)
        int partSize      = 0;   // パーティションサイズ
        int numParts      = 0;   // IR パーティション数 (power-of-two)
        int numPartsIR    = 0;   // 実 IR パーティション数 (ゼロパディング前)
        int fdlMask       = 0;   // = numParts - 1 (巡回インデックス用)
        int complexSize   = 0;   // = fftSize / 2 + 1
        int partStride    = 0;   // double 換算 complexSize*2 を 8-double アライン
        bool isImmediate  = false; // true = L0 (Add() 内で即時処理)

        // ── MKL ──
        DFTI_DESCRIPTOR_HANDLE fftHandle = nullptr;

        // ── IR 周波数領域 (Message Thread で確保・プリコンピュート) ──
        // レイアウト: [numParts][partStride] (double 配列として管理、MKL_Complex16 として解釈)
        double* irFreqDomain  = nullptr;  // mkl_malloc(numParts * partStride * sizeof(double), 64)

        // ── 入力 FDL (Frequency Domain Delay Line, Audio Thread で更新) ──
        // レイアウト: [numParts][partStride]
        double* fdlBuf        = nullptr;  // mkl_malloc(...)

        // ── 重複保存バッファ (Overlap-Add 用, Audio Thread) ──
        double* overlapBuf    = nullptr;  // mkl_malloc(partSize * sizeof(double), 64)

        // ── 作業バッファ (Audio Thread, FFT 入力/出力/複素積算) ──
        double* fftTimeBuf    = nullptr;  // mkl_malloc(fftSize * sizeof(double), 64)  前半=prev, 後半=cur
        double* fftOutBuf     = nullptr;  // mkl_malloc(fftSize * sizeof(double), 64)  IFFT 出力
        double* prevInputBuf  = nullptr;  // mkl_malloc(partSize * sizeof(double), 64) 前ブロック (Overlap-Save)
        double* accumBuf      = nullptr;  // mkl_malloc(partStride * sizeof(double), 64) 複素積算バッファ

        // ── 状態変数 (Audio Thread) ──
        int  fdlIndex   = 0;   // FDL の現在書き込みインデックス
        int  inputPos   = 0;   // 入力蓄積バッファ内の書き込み位置
        double* inputAccBuf = nullptr; // mkl_malloc(partSize * sizeof(double), 64) 入力蓄積

        // ── 遅延処理用 (非 Immediate Layer) ──
        int  partsPerCallback = 0; // 1 コールバックあたり処理するパーティション数
        int  nextPart         = 0; // 次に処理すべきパーティション番号

        void freeAll() noexcept;
    };

    //----------------------------------------------------------
    // 内部ヘルパー
    //----------------------------------------------------------

    // Layer を 1 ブロック分処理 (Forward FFT 済みの currentFDL スロットを使って畳み込み → IFFT → OLA)
    void processLayerBlock(Layer& l) noexcept;

    // リングバッファへの書き込み (2 分割コピー)
    void ringWrite(const double* src, int n) noexcept;

    // リングバッファからの読み出し (2 分割コピー)
    int  ringRead(double* dst, int n) noexcept;

    static void layerFreeAll(Layer& l) noexcept;
    void releaseAllLayers() noexcept;

    //----------------------------------------------------------
    // メンバ変数
    //----------------------------------------------------------
    static constexpr int kNumLayers = 3;

    Layer m_layers[kNumLayers];
    int   m_numActiveLayers = 0;
    int   m_latency         = 0;   // = Layer0.partSize

    // 出力リングバッファ (Audio Thread で Add/Get が同一スレッドなので lock 不要)
    double* m_ringBuf     = nullptr;
    int     m_ringSize    = 0;
    int     m_ringMask    = 0;   // = m_ringSize - 1 (power-of-two 前提)
    int     m_ringWrite   = 0;
    int     m_ringRead    = 0;
    int     m_ringAvail   = 0;   // 利用可能サンプル数

    std::atomic<bool> m_ready { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MKLNonUniformConvolver)
};

} // namespace convo
