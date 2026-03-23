//============================================================================
// MainApplication.cpp ── v0.5.3 (JUCE 8.0.12対応)
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

#include <xmmintrin.h>
#include <pmmintrin.h>

#include <mkl.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <mmsystem.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#include <processthreadsapi.h> // For disabling efficiency mode

// Windows 11 Efficiency Mode (EcoQoS) API definitions for older SDKs
#ifndef PROCESS_POWER_THROTTLING_CURRENT_VERSION
 #define PROCESS_POWER_THROTTLING_CURRENT_VERSION 1
 #define PROCESS_POWER_THROTTLING_EXECUTION_SPEED 0x1
 typedef struct _PROCESS_POWER_THROTTLING_STATE {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
 } PROCESS_POWER_THROTTLING_STATE, *PPROCESS_POWER_THROTTLING_STATE;
 // ProcessPowerThrottling = 4
 #define ProcessPowerThrottling (static_cast<PROCESS_INFORMATION_CLASS>(4))
#endif
#endif

void MainApplication::initialise(const juce::String& /*commandLine*/)
{
    {
        const auto exeDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
        const auto logFile = exeDir.getChildFile("ConvoPeq.log");
        fileLogger = std::make_unique<juce::FileLogger>(logFile, "ConvoPeq Log", 0);
        juce::Logger::setCurrentLogger(fileLogger.get());
        juce::Logger::writeToLog("Logger initialized: " + logFile.getFullPathName());
    }

#ifdef _WIN32
    // システム全体のタイマー精度を 1ms に上げる
    // 48kHz以下の環境で高負荷時のオーディオドロップアウトを防ぐ
    // 高解像度タイマーは、消費電力増加やシステム全体のパフォーマンスに影響を与える可能性があるため、注意が必要です。
    // 参照: FlexASIO/src/audio/InternalWasapiAudioClient.cpp
    //       https://github.com/dechamps/FlexASIO/blob/15314925a9434aeb95d5419c033d376240568e20/src/audio/InternalWasapiAudioClient.cpp#L329

    timeBeginPeriod(1);

    // --- Windows 11 効率モードの無効化 ---
    // アプリが最小化されたり、フォーカスを失ったりした際に
    // OSが自動的にプロセスを「効率モード」に設定し、CPUリソースを制限することがあります。
    // これがオーディオの音切れ（ドロップアウト）の主な原因となるため、プロセス単位で無効化します。
    // このAPIはWindows 11 (build 22000) 以降で有効です (Windows 8以降で関数自体は存在)。
    // Windows 7等との互換性を保つため、動的にロードして呼び出します。
    typedef BOOL (WINAPI *PSETPROCESSINFORMATION)(HANDLE, PROCESS_INFORMATION_CLASS, LPVOID, DWORD);
    auto pSetProcessInformation = (PSETPROCESSINFORMATION) GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "SetProcessInformation");

    if (pSetProcessInformation != nullptr)
    {
        PROCESS_POWER_THROTTLING_STATE PowerThrottling;
        juce::zerostruct(PowerThrottling);
        PowerThrottling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        PowerThrottling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        PowerThrottling.StateMask = 0; // スロットリングを無効化

        pSetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &PowerThrottling, sizeof(PowerThrottling));
    }

    // --- プロセス優先度の設定 (High Priority) ---
    // 最小化時でもOSから優先的にCPUリソースを割り当てられるようにする。
    // REALTIME_PRIORITY_CLASS は入力ブロックのリスクがあるため、HIGH_PRIORITY_CLASS を使用。
    // MMCSS (スレッド優先度) と併用することで、オーディオ処理の安定性を最大化する。
    if (SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
        juce::Logger::writeToLog("Process priority set to HIGH_PRIORITY_CLASS.");
    else
        juce::Logger::writeToLog("Failed to set process priority to HIGH_PRIORITY_CLASS.");
#endif


    // リアルタイムオーディオ用 Intel MKL 設定
    // リアルタイムオーディオ処理において、MKL内部のスレッド管理による
    // 予測不可能なレイテンシ（ジッター）の発生は致命的です。
    // そのため、MKLの並列処理を完全に無効化し、シングルスレッドで動作させることが
    // 安定性確保のために不可欠です。
    mkl_set_num_threads(1); // 1スレッドに固定
    mkl_set_dynamic(0);     // 動的なスレッド数調整を無効化
    // Audio Thread内でのVMLモード変更を避けるため、起動時に1回だけ設定する。
    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    // メインスレッドでも Denormal 対策を有効化
    // UIスレッドで実行されるEQ応答曲線計算 (AVX2) 等のパフォーマンスを向上させる
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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

    juce::Logger::writeToLog("MainApplication shutting down.");
    juce::Logger::setCurrentLogger(nullptr);
    fileLogger.reset();

#ifdef _WIN32
    timeEndPeriod(1);
#endif
}

// ── JUCEエントリポイント生成マクロ ──
// このマクロが int main(...) を自動生成する
START_JUCE_APPLICATION(MainApplication)
