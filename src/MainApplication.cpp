//============================================================================
// MainApplication.cpp ── v0.5.3 → v2.1 (Intel IPP 初期化追加)
//
// アプリケーション起動・終了実装
// JUCE_CREATE_APPLICATION マクロが main() を自動生成する
//
// ■ プログラム一般共通事項:
//   - Framework: JUCE V8.0.12
//   - ASIO: シングルクライアント対応 (BRAVO-HD, ASIO4ALL等) を考慮し、ブラックリスト機能を実装
//   - スタンドアローンアプリ (VST3等ではない) として設計
//   - 構造化例外処理 (SEH) 不使用
//
// ■ v2.1 変更点:
//   - #include <ipp.h> を追加
//   - initialise() 内に ippInit() を追加
//     MKL 設定ブロックの直後に配置し、Audio Thread 到達前に
//     IPP CPU ディスパッチテーブルの確定を保証する。
//============================================================================
#include "MainApplication.h"
#include "MainWindow.h"
#include "MKLRealTimeSetup.h"

#include <xmmintrin.h>
#include <pmmintrin.h>

#include <mkl.h>
#include <ipp.h>    // [v2.1] ippInit() の宣言 (IPP CPU ディスパッチ初期化)

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


    // ────────────────────────────────────────────────────────────
    // Intel MKL リアルタイム設定（B7 対応）
    //   シングルスレッド固定、動的調整無効、環境変数による強制
    // ────────────────────────────────────────────────────────────
    MKLRealTime::setup();

    // [v2.1] Intel IPP 初期化
    // CPU ディスパッチテーブル (SSE2/AVX2/AVX-512 等) を確定させ、
    // ippsFFTFwd_RToCCS_64f / ippsFFTInv_CCSToR_64f の初回呼び出しによる
    // 遅延初期化が Audio Thread コールバックに乗ることを防ぐ。
    //
    // ■ 呼び出しタイミングの根拠:
    //   MKL 設定ブロック直後に配置することで「Audio Thread に触れる前の
    //   すべてのライブラリ初期化をここに集約する」という既存のコード規約に従う。
    //
    // ■ 安全性:
    //   - スレッドセーフ（複数回呼び出し可、2回目以降は即時リターン）
    //   - ippStsNoErr 以外は実機上ほぼ発生しないが、ログで診断可能にする
    //   - SetImpulse() 内の ippsFFTGetSize_R_64f でも暗黙初期化されるが、
    //     ここで先に完了させることで SetImpulse() の初回コストも削減される
    {
        const IppStatus ippSt = ippInit();
        if (ippSt != ippStsNoErr)
            juce::Logger::writeToLog("[MainApplication] ippInit() returned status="
                                     + juce::String(static_cast<int>(ippSt)));
        else
            juce::Logger::writeToLog("[MainApplication] ippInit() succeeded.");
    }

    // メインスレッドでも Denormal 対策を有効化
    // UIスレッドで実行されるEQ応答曲線計算 (AVX2) 等のパフォーマンスを向上させる
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // VMLモード設定（Audio Thread 用、setup() では設定しない）
    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    // メインウィンドウを生成する
    mainWindow = std::make_unique<MainWindow>(getApplicationName());

    if (auto* engine = mainWindow->getAudioEngine())
        engine->getAffinityManager().applyMessageThreadPolicy();
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
