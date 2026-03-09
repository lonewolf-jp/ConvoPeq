//============================================================================
// MKLNonUniformConvolver.h  ── v2.0 (JUCE 8.0.12 / Intel oneMKL 対応)
//
// Intel MKL を使用した Non-Uniform Partitioned Convolution (NUP) エンジン。
// WDL_ConvolutionEngine_Div の低遅延特性を維持しつつ、
// MKL DFTI + AVX2 FMA でスループットを最大化する。
//
// ■ パーティション構成 (3 Layer):
//   Layer 0 (即時): partSize = nextPow2(max(blockSize,64))
//                   最大 kL0MaxParts 個のパーティション (=IRの先頭約85ms@48kHz)
//                   毎コールバックで全パーティションを即時処理 → 低レイテンシー
//   Layer 1 (遅延): partSize = L0.partSize * 8
//                   最大 kL1MaxParts 個のパーティション (=約1365ms@48kHz)
//                   partsPerCallback ずつ分散処理 → CPUスパイク抑制
//   Layer 2 (遅延): partSize = L1.partSize * 8
//                   残りの IR テール
//                   partsPerCallback ずつ分散処理
//
// ■ 出力アーキテクチャ:
//   L0  → 出力リングバッファ (ringBuf) に即時書き込み
//   L1/L2 → tailOutputBuf にIFFT完了時コピー
//   Get() = ringRead(L0) + vdAdd(L1.tailOutputBuf) + vdAdd(L2.tailOutputBuf)
//
//   L1/L2は1パーティション(partSize サンプル)分の遅延を持つ。
//   リバーブのテール領域では知覚不可のため実用上問題なし。
//   この設計により共有リングバッファのOLA時間整合問題を根本解消する。
//
// ■ CPU最適化効果 (例: IR=3s@48kHz=144000samples, blockSize=128):
//   旧設計 (L0単独): 144000/128 = 1125 パーティション/コールバック → スパイク
//   新設計 (3層NUC): L0=32 + L1=8(分散) + L2=1(分散) = 41相当/コールバック → フラット
//
// ■ バグ修正:
//   [Bug2 fix] baseFdlIdxSaved: 分散累積サイクル全体で同一のFDL時間軸を参照。
//              トリガ時に一度だけ保存し、IFFT完了まで再計算しない。
//   [Bug1 fix] 出力先分離: L0はリング独占使用、L1/L2はtailOutputBuf経由で合算。
//              共有リングへの不整合書き込みを排除。
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
    // @param scale         IRの振幅スケール (ヘッドルーム確保用, デフォルト=1.0)
    // @return true=成功, false=パラメータ不正またはMKL初期化失敗
    //----------------------------------------------------------
    bool SetImpulse(const double* impulse, int irLen, int blockSize, double scale = 1.0);

    //----------------------------------------------------------
    // Add  ─ Audio Thread のみ
    // 入力サンプルを畳み込みエンジンへ投入する。
    // @param input      入力バッファ (numSamples サンプル)。nullptr=無音として扱う
    // @param numSamples ブロックサイズ (SetImpulse 時の blockSize と一致推奨)
    //----------------------------------------------------------
    void Add(const double* input, int numSamples);

    //----------------------------------------------------------
    // Get  ─ Audio Thread のみ
    // 畳み込み結果を output へ書き出す。
    // L0 の出力 (リング) と L1/L2 の遅延出力 (tailOutputBuf) を合算する。
    // 出力は Layer0 の partitionSize サンプルのレイテンシーを伴う。
    // @param output     出力バッファ (numSamples サンプル)。nullptr=破棄
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
        int numParts      = 0;   // FDL スロット数 (power-of-two)
        int numPartsIR    = 0;   // 実 IR パーティション数 (ゼロパディング前)
        int fdlMask       = 0;   // = numParts - 1 (巡回インデックス用)
        int complexSize   = 0;   // = fftSize / 2 + 1
        int partStride    = 0;   // double 換算 complexSize*2 を 8-double アライン
        bool isImmediate  = false; // true = L0 (Add() 内で即時処理, リングを使用)

        // ── MKL ──
        DFTI_DESCRIPTOR_HANDLE fftHandle = nullptr;

        // ── IR 周波数領域 (Message Thread で確保・プリコンピュート) ──
        // レイアウト: [numParts][partStride] (double 配列として管理、MKL_Complex16 として解釈)
        double* irFreqDomain  = nullptr;  // mkl_malloc(numParts * partStride * sizeof(double), 64)

        // ── 入力 FDL (Frequency Domain Delay Line, Audio Thread で更新) ──
        // レイアウト: [numParts][partStride]
        double* fdlBuf        = nullptr;  // mkl_malloc(...)

        // ── 重複保存バッファ (Overlap-Save 用, L0 専用) ──
        double* overlapBuf    = nullptr;  // mkl_malloc(partSize * sizeof(double), 64)

        // ── 作業バッファ (Audio Thread, FFT 入力/出力/複素積算) ──
        double* fftTimeBuf    = nullptr;  // mkl_malloc(fftSize * sizeof(double), 64)  前半=prev, 後半=cur
        double* fftOutBuf     = nullptr;  // mkl_malloc(fftSize * sizeof(double), 64)  IFFT 出力
        double* prevInputBuf  = nullptr;  // mkl_malloc(partSize * sizeof(double), 64) 前ブロック (Overlap-Save)
        double* accumBuf      = nullptr;  // mkl_malloc(partStride * sizeof(double), 64) 複素積算バッファ

        // ── 状態変数 (Audio Thread) ──
        int     fdlIndex    = 0;          // FDL の現在書き込みインデックス
        int     inputPos    = 0;          // 入力蓄積バッファ内の書き込み位置
        double* inputAccBuf = nullptr;    // mkl_malloc(partSize * sizeof(double), 64) 入力蓄積

        // ── L1/L2: テール出力バッファ (遅延出力を一時保持) ──
        // IFFT 完了時に fftOutBuf の有効部分 (後半 partSize サンプル) をここにコピーし、
        // Get() で L0 出力 (リング) に vdAdd で合算する。
        // L0 (isImmediate=true) では使用しない (nullptr のまま)。
        double* tailOutputBuf = nullptr;  // mkl_malloc(partSize * sizeof(double), 64)
        int     tailOutputPos = 0;        // Get() での読み出しカーソル [0, partSize]

        // ── 遅延処理用 (L1/L2 非 Immediate Layer) ──
        int  partsPerCallback  = 0;  // 1 コールバックあたり処理するパーティション数
        int  nextPart          = 0;  // 次に処理すべきパーティション番号

        // [Bug2 fix] 分散累積サイクル開始時に保存した FDL スナップショット。
        // nextPart==0 の時点 (トリガ) で一度だけ計算・保存し、
        // 同一サイクルの全パーティションで再計算を禁止する。
        // 再計算すると fdlIndex が進むため、各パーティションが異なる FDL 時間位置を
        // 参照してしまい、出力に時間的な不整合が生じる。
        int  baseFdlIdxSaved   = 0;

        // 分散計算進行中フラグ (トリガ → true, IFFT 完了 → false)
        bool distributing      = false;

        void freeAll() noexcept;
    };

    //----------------------------------------------------------
    // 内部ヘルパー
    //----------------------------------------------------------

    // L0 を 1 パーティション分処理 (Overlap-Save → Forward FFT → FDL × IR → IFFT → ringWrite)
    void processLayerBlock(Layer& l) noexcept;

    // L0 専用: リングバッファへの順次書き込み (memcpy + m_ringWrite/m_ringAvail 更新)
    void ringWrite(const double* src, int n) noexcept;

    // L0 専用: リングバッファからの読み出し (memcpy + Zero-Flush + m_ringRead/m_ringAvail 更新)
    int  ringRead(double* dst, int n) noexcept;

    void releaseAllLayers() noexcept;

    //----------------------------------------------------------
    // メンバ変数
    //----------------------------------------------------------
    static constexpr int kNumLayers = 3;

    // L0: 最大パーティション数 (即時処理対象の上限, CPUキャッシュに収まるサイズ)
    // 32 * 128 = 4096 samples ≈ 85ms@48kHz → L1/L2キャッシュ収まりサイズ
    static constexpr int kL0MaxParts = 32;

    // L1: 最大パーティション数
    // 64 * 1024 = 65536 samples ≈ 1365ms@48kHz
    static constexpr int kL1MaxParts = 64;

    Layer m_layers[kNumLayers];
    int   m_numActiveLayers = 0;
    int   m_latency         = 0;   // = Layer0.partSize

    // 出力リングバッファ (L0 専用。Add/Get が同一 Audio Thread なので lock 不要)
    double* m_ringBuf     = nullptr;
    int     m_ringSize    = 0;
    int     m_ringMask    = 0;   // = m_ringSize - 1 (power-of-two 前提)
    int     m_ringWrite   = 0;
    int     m_ringRead    = 0;
    int     m_ringAvail   = 0;   // 利用可能サンプル数 (L0 出力のみカウント)

    std::atomic<bool> m_ready { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MKLNonUniformConvolver)
};

} // namespace convo
