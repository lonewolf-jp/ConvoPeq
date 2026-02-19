//============================================================================
// MainApplication.cpp  ── v0.1 (JUCE 8.0.12対応)
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

#if JUCE_DSP_USE_INTEL_MKL
 #include <mkl.h>
#endif

void MainApplication::initialise(const juce::String& /*commandLine*/)
{
#if JUCE_DSP_USE_INTEL_MKL
    // Intel MKL Configuration for Real-time Audio
    mkl_set_num_threads(1);                    // 1スレッド固定（必須）
    // mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL); // リンクエラー回避のため無効化 (CMakeでsequential指定済み)
    mkl_set_dynamic(0);                        // 動的調整無効
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
