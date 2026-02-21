//============================================================================
// MainApplication.cpp ── v0.2 (JUCE 8.0.12対応)
//
// アプリケーション起動・終了実装
// JUCE_CREATE_APPLICATION マクロが main() を自動生成する
//
// ■ プログラム一般共通事項:
//   - Framework: JUCE V8.0.12
//   - ASIO: シングルクライアント対応 (BRAVO-HD, ASIO4ALL等) を考慮し、ブラックリスト機能を実装
//   - スタンドアローンアプリ (VST3等ではない) として設計
//   - 構造化例外処理 (SEH) 不使用
//============================================================================
#include "MainApplication.h"
#include "MainWindow.h"

#if JUCE_INTEL
 #include <xmmintrin.h>
 #include <pmmintrin.h>
#endif

#if JUCE_DSP_USE_INTEL_MKL
#include <mkl.h>
#endif

void MainApplication::initialise(const juce::String& /*commandLine*/)
{
#if JUCE_INTEL
#pragma warning(push)
#pragma warning(disable: 6815)
#endif


#if JUCE_DSP_USE_INTEL_MKL
    // リアルタイムオーディオ用 Intel MKL 設定
    // リアルタイムオーディオ処理において、MKL内部のスレッド管理による
    // 予測不可能なレイテンシ（ジッター）の発生は致命的です。
    // そのため、MKLの並列処理を完全に無効化し、シングルスレッドで動作させることが
    // 安定性確保のために不可欠です。
    mkl_set_num_threads(1); // 1スレッドに固定
    mkl_set_dynamic(0);     // 動的なスレッド数調整を無効化
#endif

#if JUCE_INTEL
#pragma warning(pop)
#endif

    // メインウィンドウを生成する
    mainWindow = std::make_unique<MainWindow>(getApplicationName());
}

void MainApplication::shutdown()
{
    // unique_ptr のデストラクタで MainWindow が閉じられる
    // MainWindow デストラクタ内で:
    //   1) オーディオコールバック停止
    //   2) UI コンポーネント破棄 (AudioEngineへの参照を解放)
    //   3) AudioEngine 破棄
    // の順で安全にシャットダウンされる
    mainWindow.reset();
}

// ── JUCEエントリポイント生成マクロ ──
// このマクロが int main(...) を自動生成する
START_JUCE_APPLICATION(MainApplication)
