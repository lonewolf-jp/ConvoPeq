//============================================================================
// MKLNonUniformConvolver.h  ── v2.1 (JUCE 8.0.12 / Intel oneMKL + IPP 対応)
//
// Intel IPP FFT を使用した Non-Uniform Partitioned Convolution (NUP) エンジン。
// v2.0 から FFT バックエンドを MKL DFTI → Intel IPP に換装。
//
// ■ v2.1 変更点 (v2.0 → v2.1):
//   [FFT換装] オーディオスレッド内 FFT を MKL DFTI → Intel IPP に変更。
//     - DFTI_DESCRIPTOR_HANDLE → IppsFFTSpec_R_64f* (ipps.h)
//     - DftiComputeForward/Backward → ippsFFTFwd_RToCCS_64f / ippsFFTInv_CCSToR_64f
//     - ワークバッファをSetImpulse()で事前確保 → Audio Thread 内メモリ確保ゼロ保証
//     - IPP_FFT_DIV_INV_BY_N フラグ: IFFT 時に 1/N 正規化 (MKL DFTI_BACKWARD_SCALE相当)
//     - IPP の CCS 出力形式: [re0,im0,re1,im1,...] は MKL DFTI_COMPLEX_COMPLEX と同一
//       → 既存の複素乗算 AVX2 コードは無変更で動作する
//     - MKL VML (vdMul) / MKL BLAS (cblas_dscal) は引き続き Message Thread で使用
//
// ■ 3層レイヤー構造 (v2.0 から変更なし):
//   Layer 0 (即時): partSize = nextPowerOfTwo(max(blockSize,64))
//                   最大 kL0MaxParts 個のパーティション (=IRの先頭約85ms@48kHz)
//                   毎コールバックで全パーティションを即時処理 → 低レイテンシー
//   Layer 1 (遅延): partSize = L0.partSize * 8
//                   最大 kL1MaxParts 個のパーティション (=約1365ms@48kHz)
//                   partsPerCallback ずつ分散処理 → CPUスパイク抑制
//   Layer 2 (遅延): partSize = L1.partSize * 8
//                   残りの IR テール
//                   partsPerCallback ずつ分散処理
//
// ■ 出力アーキテクチャ (v2.0 から変更なし):
//   L0  → 出力リングバッファ (ringBuf) に即時書き込み
//   L1/L2 → tailOutputBuf にIFFT完了時コピー
//   Get() = ringRead(L0) + vdAdd(L1.tailOutputBuf) + vdAdd(L2.tailOutputBuf)
//
// ■ スレッド安全設計:
//   SetImpulse()  : Message Thread (prepareToPlay 相当) からのみ呼び出す
//   Add() / Get() : Audio Thread から呼び出す（メモリ確保なし）
//   Reset()       : Message Thread または releaseResources() から呼び出す
//
// ■ 規約遵守:
//   - 64bit double 処理
//   - mkl_malloc(64) によるオーディオデータメモリ管理 (new / std::vector 禁止)
//   - ippsMalloc_8u による IPP FFT spec/work バッファ管理
//   - Audio Thread 内でのメモリ確保・FFT 再初期化禁止
//   - FTZ/DAZ は呼び出し元 (ConvolverProcessor) で設定済みを前提とする
//============================================================================
#pragma once

#include <mkl.h>        // mkl_malloc, mkl_free, VML, CBLAS (オーディオデータ用)
#include <ipp.h>       // IppsFFTSpec_R_64f (MKL DFTI の代替)
#include <atomic>
#include <memory>
#include <JuceHeader.h>  // juce::nextPowerOfTwo, JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR
#include "OutputFilter.h" // convo::HCMode, convo::LCMode

namespace convo
{

struct IppFFTPlan;

//==============================================================================
// FilterSpec  ─ SetImpulse() に渡す出力周波数フィルター仕様
//
// NUC は SetImpulse() 内で irFreqDomain に周波数ゲインを乗算して「焼き込む」。
// Audio Thread の追加コストはゼロ。モード変更時は SetImpulse() を再実行する
// (rebuildAllIRs() トリガー)。
//
// hcMode = HCMode を参照 (OutputFilter.h):
//   Sharp   : Butterworth 4次相当の急峻ロールオフ
//   Natural : コサインクロスフェード (デフォルト)
//   Soft    : ガウス型緩やかロールオフ
//   (値なし=Disabled相当として nullptr で SetImpulse() に渡す)
//
// lcMode = LCMode を参照 (OutputFilter.h):
//   Natural : コサインロールオン fc≈18Hz (デフォルト)
//   Soft    : コサインロールオン fc≈15Hz
//==============================================================================
struct FilterSpec
{
    double sampleRate = 48000.0; ///< 処理サンプルレート (Hz)
    HCMode hcMode     = HCMode::Natural; ///< ハイカットモード
    LCMode lcMode     = LCMode::Natural; ///< ローカットモード
    int tailMode      = 0; ///< 0=Air Absorption(全レイヤー), 1=Layer Tail(L1/L2のみ)
    float tailRolloffStartHz    = 3500.0f; ///< テール減衰開始周波数
    float tailRolloffStrength   = 0.3f;    ///< テール減衰強度
    float partitionTailStrength = 1.0f;    ///< tailMode=1 時の L1/L2 追加係数
};

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
    // @param filterSpec    出力周波数フィルター仕様。nullptr の場合フィルターなし。
    //                      irFreqDomain に周波数ゲインを焼き込む (Audio Thread コストゼロ)。
    // @return true=成功, false=パラメータ不正またはIPP初期化失敗
    //----------------------------------------------------------
    bool SetImpulse(const double* impulse, int irLen, int blockSize,
                    double scale = 1.0,
                    bool enableDirectHead = false,
                    const FilterSpec* filterSpec = nullptr);

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
    // areFftDescriptorsCommitted  ─ いつでも呼び出し可
    // Audio Thread で使用する IPP FFT スペックが SetImpulse() で
    // 初期化済みかを検証する。
    //----------------------------------------------------------
    bool areFftDescriptorsCommitted() const noexcept;

    //----------------------------------------------------------
    // getLatency  ─ 出力の先頭レイテンシー (サンプル数)
    // = Layer0 の partitionSize
    //----------------------------------------------------------
    int getLatency() const noexcept { return m_latency; }

    //----------------------------------------------------------
    // setOverflowCallback  ─ Audio Thread セーフなオーバーフロー通知
    // ringWrite() でオーバーフローが発生した際に一度呼ばれる。
    // 実装はロックフリー・メモリ確保禁止。生関数ポインタのみ (std::function 禁止)。
    //----------------------------------------------------------
    using OverflowCallback = void(*)(void* userData);
    void setOverflowCallback(OverflowCallback cb, void* userData) noexcept
    {
        overflowCallback = cb;
        overflowUserData = userData;
    }

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

        // ── IPP FFT ──
        // [v2.1] MKL DFTI_DESCRIPTOR_HANDLE から Intel IPP へ換装。
        //
        // fftSpec    : ippsFFTInit_R_64f が管理する共有プラン内スペック。
        //              Audio Thread からは Read-Only で参照 (スレッドセーフ)。
        // fftPlanOwner: サイズ単位で共有される FFT plan の所有権。
        // fftWorkBuf : 各 FFT 計算呼び出しが使用するスクラッチバッファ。
        //              ippsFFTGetSize_R_64f が sizeWork > 0 を返した場合のみ確保。
        //              sizeWork == 0 の場合 nullptr のまま (IPP が外部バッファ不要)。
        //              ★ Audio Thread での確保を防ぐため SetImpulse() で事前確保済み。
        //                 nullptr かつ sizeWork==0 の場合のみ IPP に nullptr を渡してよい。
        std::shared_ptr<const IppFFTPlan> fftPlanOwner; ///< 共有 FFT plan (サイズ単位キャッシュ)
        IppsFFTSpec_R_64f* fftSpec    = nullptr; ///< IPP FFT スペック (共有 plan 内を指す)
        Ipp8u*             fftWorkBuf = nullptr; ///< FFT スクラッチ (sizeWork==0なら nullptr)
        bool               descriptorCommitted = false; ///< IPP 初期化成功フラグ

        // ── IR 周波数領域 (Message Thread で確保・プリコンピュート) ──
        // レイアウト: [numParts][partStride] (double 配列として管理)
        // IPP CCS 形式: [re0,im0,re1,im1,...] は MKL DFTI_COMPLEX_COMPLEX と同一レイアウト。
        // → 既存の AVX2 複素乗算コードは無変更で動作する。
        double* irFreqDomain  = nullptr;  // mkl_malloc(numParts * partStride * sizeof(double), 64)
        // split-complex 検証用 SoA ストレージ（実部/虚部分離）
        double* irFreqReal    = nullptr;  // mkl_malloc(numParts * complexSize * sizeof(double), 64)
        double* irFreqImag    = nullptr;  // mkl_malloc(numParts * complexSize * sizeof(double), 64)

        // ── 入力 FDL (Frequency Domain Delay Line, Audio Thread で更新) ──
        // レイアウト: [numParts][partStride]
        double* fdlBuf        = nullptr;  // mkl_malloc(...)
        double* fdlReal       = nullptr;  // mkl_malloc((numParts*2) * complexSize * sizeof(double), 64)
        double* fdlImag       = nullptr;  // mkl_malloc((numParts*2) * complexSize * sizeof(double), 64)

        // ── 作業バッファ (Audio Thread, FFT 入力/出力/複素積算) ──
        double* fftTimeBuf    = nullptr;  // mkl_malloc(fftSize * sizeof(double), 64)  前半=prev, 後半=cur
        double* fftOutBuf     = nullptr;  // mkl_malloc(fftSize * sizeof(double), 64)  IFFT 出力
        double* prevInputBuf  = nullptr;  // mkl_malloc(partSize * sizeof(double), 64) 前ブロック (Overlap-Save)
        double* accumBuf      = nullptr;  // mkl_malloc(partStride * sizeof(double), 64) 複素積算バッファ
        double* accumReal     = nullptr;  // mkl_malloc(complexSize * sizeof(double), 64)
        double* accumImag     = nullptr;  // mkl_malloc(complexSize * sizeof(double), 64)

        // ── 状態変数 (Audio Thread) ──
        int     fdlIndex    = 0;          // FDL の現在書き込みインデックス
        int     inputPos    = 0;          // 入力蓄積バッファ内の書き込み位置
        double* inputAccBuf = nullptr;    // mkl_malloc(partSize * sizeof(double), 64) 入力蓄積

        // ── L1/L2: テール出力バッファ (遅延出力を一時保持) ──
        double* tailOutputBuf = nullptr;  // mkl_malloc(partSize * sizeof(double), 64)
        int     tailOutputPos = 0;        // Get() での読み出しカーソル [0, partSize]

        // B7: FFT ウォームアップ済みフラグ（レイヤーごと、Non-Audio Thread でセット）
        std::atomic<bool> warmupCompleted { false };

        // ── 遅延処理用 (L1/L2 非 Immediate Layer) ──
        int  partsPerCallback  = 0;  // 1 コールバックあたり処理するパーティション数
        int  nextPart          = 0;  // 次に処理すべきパーティション番号

        // [Bug2 fix] 分散累積サイクル開始時に保存した FDL スナップショット。
        int  baseFdlIdxSaved   = 0;

        // 分散計算進行中フラグ (トリガ → true, IFFT 完了 → false)
        bool distributing      = false;

        void freeAll() noexcept;
    };

    //----------------------------------------------------------
    // 内部ヘルパー
    //----------------------------------------------------------
    void processLayerBlock(Layer& l) noexcept;
    void ringWrite(const double* src, int n) noexcept;
    int  ringRead(double* dst, int n) noexcept;
    void processDirectBlock(const double* input, int numSamples) noexcept;
    void releaseAllLayers() noexcept;
    void applySpectrumFilter(const FilterSpec& spec) noexcept;

    //----------------------------------------------------------
    // メンバ変数
    //----------------------------------------------------------
    static constexpr int kNumLayers = 3;
    static constexpr int kL0MaxParts = 32;
    static constexpr int kL1MaxParts = 64;

    Layer m_layers[kNumLayers];
    int   m_numActiveLayers = 0;
    int   m_latency         = 0;

    double* m_ringBuf     = nullptr;
    int     m_ringSize    = 0;
    int     m_ringMask    = 0;
    int     m_ringWrite   = 0;
    int     m_ringRead    = 0;
    int     m_ringAvail   = 0;

    std::atomic<int> m_ringOverflowCount { 0 };
    OverflowCallback overflowCallback = nullptr;
    void*            overflowUserData = nullptr;

    int     m_directTapCount = 0;
    int     m_directHistLen  = 0;
    int     m_directMaxBlock = 0;
    int     m_directPendingSamples = 0;
    bool    m_directEnabled  = false;
    double* m_directIRRev    = nullptr;
    double* m_directHistory  = nullptr;
    double* m_directWindow   = nullptr;
    double* m_directOutBuf   = nullptr;

    std::atomic<bool> m_ready { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MKLNonUniformConvolver)
};

} // namespace convo
