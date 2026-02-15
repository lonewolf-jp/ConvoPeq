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

void MainApplication::initialise(const juce::String& /*commandLine*/)
{
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
