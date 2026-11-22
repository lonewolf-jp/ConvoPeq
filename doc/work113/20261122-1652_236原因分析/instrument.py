"""Insert measurement-only QPC marks into DeviceSettings.cpp / MainWindow.cpp (issue #236).

Not committed. Run `instrument.py apply` then build, or `apply revert` to restore
the two files from git.
"""
import io
import os
import re
import subprocess
import sys

W = r"C:\VSC_Project\fix236"
GIT = r"C:\Program Files\Git\cmd\git.exe"
DS = os.path.join(W, r"src\DeviceSettings.cpp")
MW = os.path.join(W, r"src\MainWindow.cpp")
HDR = os.path.join(W, r"src\M236Trace.h")
AP = os.path.join(W, r"src\audioengine\AudioEngineProcessor.cpp")
MA = os.path.join(W, r"src\MainApplication.cpp")
PP = os.path.join(W, r"src\audioengine\AudioEngine.Processing.PrepareToPlay.cpp")

HEADER = r'''// Measurement-only QPC marks for issue #236. Not committed.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <windows.h>

namespace m236
{
    struct Mark
    {
        const char* tag;
        long long qpc;
        long long cpu;
    };

    inline long long threadCpu100ns() noexcept
    {
        FILETIME creation, exit, kernel, user;
        if (GetThreadTimes (GetCurrentThread(), &creation, &exit, &kernel, &user) == 0)
            return 0;
        const long long k = (static_cast<long long> (kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime;
        const long long u = (static_cast<long long> (user.dwHighDateTime) << 32) | user.dwLowDateTime;
        return k + u;
    }

    inline Mark* buffer() noexcept
    {
        static Mark marks[256];
        return marks;
    }

    inline int& count() noexcept
    {
        static int n = 0;
        return n;
    }

    inline void mark (const char* tag) noexcept
    {
        LARGE_INTEGER c;
        QueryPerformanceCounter (&c);
        int& n = count();
        if (n < 256)
        {
            buffer()[n].tag = tag;
            buffer()[n].qpc = c.QuadPart;
            buffer()[n].cpu = threadCpu100ns();
            ++n;
        }
    }

    inline void dump() noexcept
    {
        const char* path = std::getenv ("CONVOPEQ_M236_TRACE");
        if (path == nullptr)
            return;

        FILE* f = nullptr;
        if (fopen_s (&f, path, "w") != 0 || f == nullptr)
            return;

        for (int i = 0; i < count(); ++i)
            std::fprintf (f, "%s %lld %lld\n", buffer()[i].tag, buffer()[i].qpc, buffer()[i].cpu);

        std::fclose (f);
    }

    struct AutoDump
    {
        ~AutoDump() { dump(); }
    };
}
'''

DS_INCLUDE = """#include "M236Trace.h"
"""


def read(p):
    return io.open(p, encoding="utf-8").read()


def write(p, t):
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)


def sub1(t, old, new, label):
    n = t.count(old)
    if n != 1:
        raise SystemExit("anchor %s matched %d times" % (label, n))
    return t.replace(old, new, 1)


def apply():
    ds = read(DS)
    mw = read(MW)
    if "M236Trace.h" in ds:
        print("already instrumented")
        return

    write(HDR, HEADER)

    # include + one AutoDump instance
    ds = sub1(ds, '#include "DeviceSettings.h"', '#include "DeviceSettings.h"\n' + DS_INCLUDE, "ds include")
    ds = sub1(ds, "namespace\n{", "namespace\n{\n    const m236::AutoDump m236AutoDumpInstance;\n", "ds anon ns")

    # loadSettings body
    ds = sub1(ds,
              "void DeviceSettings::loadSettings (juce::AudioDeviceManager& deviceManager, AudioEngine& engine)\n{\n    loadNoiseShaperState(engine);\n",
              "void DeviceSettings::loadSettings (juce::AudioDeviceManager& deviceManager, AudioEngine& engine)\n{\n"
              "    m236::mark(\"noise_begin\");\n    loadNoiseShaperState(engine);\n    m236::mark(\"noise_end\");\n\n"
              "    m236::mark(\"bulk_begin\");\n", "load start")

    ds = sub1(ds,
              "    deviceManager.closeAudioDevice();\n",
              "    m236::mark(\"close_begin\");\n    deviceManager.closeAudioDevice();\n    m236::mark(\"close_end\");\n",
              "close")

    ds = sub1(ds,
              "    auto file = getSettingsFile();\n\n    if (file.existsAsFile())\n    {\n        if (auto xml = juce::XmlDocument::parse (file))\n        {\n",
              "    if (std::getenv(\"CONVOPEQ_M236_PREWARM\") != nullptr)\n"
              "    {\n"
              "        m236::mark(\"prewarm_begin\");\n"
              "        deviceManager.initialise (2, 2, nullptr, true);\n"
              "        m236::mark(\"prewarm_end\");\n"
              "    }\n\n"
              "    m236::mark(\"file_begin\");\n"
              "    auto file = getSettingsFile();\n"
              "    const bool settingsFileExists = file.existsAsFile();\n"
              "    m236::mark(\"file_end\");\n\n"
              "    if (settingsFileExists)\n    {\n"
              "        m236::mark(\"parse_begin\");\n"
              "        auto xml = juce::XmlDocument::parse (file);\n"
              "        m236::mark(\"parse_end\");\n"
              "        if (xml != nullptr)\n        {\n",
              "file/parse")

    ds = sub1(ds,
              "            juce::String error = deviceManager.initialise (2, 2, xml.get(), false);\n",
              "            m236::mark(\"init_begin\");\n"
              "            juce::String error = deviceManager.initialise (2, 2, xml.get(), false);\n"
              "            m236::mark(\"init_end\");\n",
              "init")

    ds = sub1(ds,
              "                auto relaxedSetup = makeRelaxedSetupFromXml(*xml);\n"
              "                recoveredWithRelaxedSetup = deviceManager.setAudioDeviceSetup(relaxedSetup, false);\n",
              "                auto relaxedSetup = makeRelaxedSetupFromXml(*xml);\n"
              "                m236::mark(\"relaxed_begin\");\n"
              "                recoveredWithRelaxedSetup = deviceManager.setAudioDeviceSetup(relaxedSetup, false);\n"
              "                m236::mark(\"relaxed_end\");\n",
              "relaxed")

    ds = sub1(ds,
              "                    const juce::String fallbackError = initialiseDefaultDevice();\n",
              "                    m236::mark(\"fallback_begin\");\n"
              "                    const juce::String fallbackError = initialiseDefaultDevice();\n"
              "                    m236::mark(\"fallback_end\");\n",
              "fallback")

    ds = sub1(ds,
              "            ensureUsableChannelSelection(deviceManager);\n\n"
              "            // ★ [work70 v9.11] MMCSS policy 判定用にデバイス種類名をエンジンに通知\n",
              "            m236::mark(\"channel_begin\");\n"
              "            ensureUsableChannelSelection(deviceManager);\n"
              "            m236::mark(\"channel_end\");\n\n"
              "            m236::mark(\"restore_begin\");\n\n"
              "            // ★ [work70 v9.11] MMCSS policy 判定用にデバイス種類名をエンジンに通知\n",
              "channel+restore")

    ds = sub1(ds,
              "            {\n                bool hasBankedAdaptiveCoefficients = false;\n",
              "            m236::mark(\"adaptive_begin\");\n"
              "            {\n                bool hasBankedAdaptiveCoefficients = false;\n",
              "adaptive begin")

    ds = sub1(ds,
              "            // Fixed 4-tap 比較ログ設定\n",
              "            m236::mark(\"adaptive_end\");\n\n"
              "            // Fixed 4-tap 比較ログ設定\n",
              "adaptive end")

    # fine-grained boundaries inside the restore block
    ds = sub1(ds,
              "            // ビット深度設定の読み込み (デフォルト0 = 自動/最大)\n",
              "            m236::mark(\"dt_end\");\n\n"
              "            // ビット深度設定の読み込み (デフォルト0 = 自動/最大)\n",
              "dt end")
    ds = sub1(ds,
              "            // ノイズシェーパー種類の読み込み (デフォルト0 = Current)\n",
              "            m236::mark(\"bd_end\");\n\n"
              "            // ノイズシェーパー種類の読み込み (デフォルト0 = Current)\n",
              "bd end")
    ds = sub1(ds,
              "            m236::mark(\"adaptive_begin\");\n",
              "            m236::mark(\"ns_end\");\n"
              "            m236::mark(\"adaptive_begin\");\n",
              "ns end")
    ds = sub1(ds,
              "            // オーバーサンプリング設定の読み込み (デフォルト0 = 自動)\n",
              "            m236::mark(\"log_end\");\n\n"
              "            // オーバーサンプリング設定の読み込み (デフォルト0 = 自動)\n",
              "log end")
    ds = sub1(ds,
              "            // 入力ヘッドルーム設定の読み込み (デフォルト -6.0dB)\n",
              "            m236::mark(\"os_end\");\n\n"
              "            // 入力ヘッドルーム設定の読み込み (デフォルト -6.0dB)\n",
              "os end")
    ds = sub1(ds,
              "            // Output Makeup設定の読み込み (デフォルト +12.0dB)\n",
              "            m236::mark(\"hr_end\");\n\n"
              "            // Output Makeup設定の読み込み (デフォルト +12.0dB)\n",
              "hr end")
    ds = sub1(ds,
              "            // ★ v14.0: Auto Gain Staging 設定の復元 (デフォルト true)\n",
              "            m236::mark(\"mk_end\");\n\n"
              "            // ★ v14.0: Auto Gain Staging 設定の復元 (デフォルト true)\n",
              "mk end")
    ds = sub1(ds,
              "            // フィルタタイプ設定の読み込み (デフォルト0 = IIR)\n",
              "            m236::mark(\"ags_end\");\n\n"
              "            // フィルタタイプ設定の読み込み (デフォルト0 = IIR)\n",
              "ags end")
    ds = sub1(ds,
              "            // Audio Thread Priority 設定の読み込み (デフォルト \"MMCSS\")\n",
              "            m236::mark(\"ot_end\");\n\n"
              "            // Audio Thread Priority 設定の読み込み (デフォルト \"MMCSS\")\n",
              "ot end")

    ds = sub1(ds,
              "            // Convolver state\n"
              "            if (auto* convXml = xml->getChildByName(\"Convolver\"))\n",
              "            // Convolver state\n"
              "            m236::mark(\"convolver_begin\");\n"
              "            if (auto* convXml = xml->getChildByName(\"Convolver\"))\n",
              "convolver begin")

    ds = sub1(ds,
              '            m236::mark("convolver_begin");\n',
              '            m236::mark("tp_end");\n            m236::mark("convolver_begin");\n',
              "tp end")

    ds = sub1(ds,
              "            return;\n        }\n    }\n",
              "            if (std::getenv(\"CONVOPEQ_M236_REINIT\") != nullptr)\n"
              "            {\n"
              "                m236::mark(\"reinit_begin\");\n"
              "                deviceManager.initialise (2, 2, xml.get(), false);\n"
              "                m236::mark(\"reinit_end\");\n"
              "            }\n"
              "            m236::mark(\"convolver_end\");\n"
              "            m236::mark(\"restore_end\");\n"
              "            return;\n        }\n    }\n",
              "file branch end")

    ds = sub1(ds,
              "    initialiseDefaultDevice();\n\n"
              "    // ★ [work70 v9.11] MMCSS policy 判定用にデバイス種類名をエンジンに通知\n",
              "    m236::mark(\"default_init_begin\");\n"
              "    initialiseDefaultDevice();\n"
              "    m236::mark(\"default_init_end\");\n\n"
              "    m236::mark(\"restore_begin\");\n\n"
              "    // ★ [work70 v9.11] MMCSS policy 判定用にデバイス種類名をエンジンに通知\n",
              "default init")

    ds = sub1(ds,
              "    ensureUsableChannelSelection(deviceManager);\n\n"
              "    engine.setDitherBitDepth(0); // 自動設定へ\n",
              "    m236::mark(\"channel_begin\");\n"
              "    ensureUsableChannelSelection(deviceManager);\n"
              "    m236::mark(\"channel_end\");\n\n"
              "    engine.setDitherBitDepth(0); // 自動設定へ\n",
              "default channel")

    ds = sub1(ds,
              "    engine.setOversamplingType(AudioEngine::OversamplingType::IIR); // デフォルトIIR\n\n}\n",
              "    engine.setOversamplingType(AudioEngine::OversamplingType::IIR); // デフォルトIIR\n"
              "    m236::mark(\"restore_end\");\n\n}\n",
              "default end")

    # S2 envelope in MainWindow::loadSettings
    ap = read(AP)
    if "M236Trace.h" not in ap:
        ap = sub1(ap, '#include "AudioEngineProcessor.h"',
                  '#include "AudioEngineProcessor.h"\n#include "M236Trace.h"', "proc include")
        ap = sub1(ap,
                  "void AudioEngineProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)\n{\n",
                  "void AudioEngineProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)\n{\n"
                  "    m236::mark(\"prep_begin\");\n", "prep begin")
        ap = sub1(ap,
                  "void AudioEngineProcessor::processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&)\n{\n",
                  "void AudioEngineProcessor::processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&)\n{\n"
                  "    static int cbCount = 0;\n"
                  "    if (cbCount < 5)\n"
                  "    {\n"
                  "        static const char* cbTags[] = { \"cb1\", \"cb2\", \"cb3\", \"cb4\", \"cb5\" };\n"
                  "        m236::mark(cbTags[cbCount]);\n"
                  "        ++cbCount;\n"
                  "    }\n", "cb first")
        ap = sub1(ap,
                  "        convo::publishAtomic(cachedTailLength, 0.0, std::memory_order_relaxed);\n    }\n}\n",
                  "        convo::publishAtomic(cachedTailLength, 0.0, std::memory_order_relaxed);\n    }\n"
                  "    m236::mark(\"prep_end\");\n}\n", "prep end")
        write(AP, ap)

    pp = read(PP)
    if "M236Trace.h" not in pp:
        pp = sub1(pp, '#include "DSPLifetimeManager.h"',
                  '#include "DSPLifetimeManager.h"\n#include "M236Trace.h"', "pp include")
        pp = sub1(pp,
                  "void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)\n{\n"
                  "    ASSERT_NON_RT_THREAD();\n",
                  "void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)\n{\n"
                  "    m236::mark(\"aeprep_begin\");\n    ASSERT_NON_RT_THREAD();\n", "aeprep begin")
        pp = sub1(pp,
                  "    reprepareUiConvolverForProcessingGeometry(safeSampleRate, bufferSize);\n",
                  "    m236::mark(\"rep_begin\");\n"
                  "    reprepareUiConvolverForProcessingGeometry(safeSampleRate, bufferSize);\n"
                  "    m236::mark(\"rep_end\");\n", "rep")
        pp = sub1(pp,
                  "    if (rateChanged || blockSizeChanged || !hasCurrentRuntime) {\n",
                  "    m236::mark(\"rb_begin\");\n"
                  "    if (rateChanged || blockSizeChanged || !hasCurrentRuntime) {\n", "rb begin")
        pp = sub1(pp,
                  "    diagLog(\"[DIAG] prepareToPlay: exit currentSR=\"",
                  "    m236::mark(\"rb_end\");\n"
                  "    diagLog(\"[DIAG] prepareToPlay: exit currentSR=\"", "rb end")
        pp = sub1(pp,
                  "    lifecycleRuntime_.leavePrepare(lifecycleToken);\n}\n",
                  "    lifecycleRuntime_.leavePrepare(lifecycleToken);\n    m236::mark(\"aeprep_end\");\n}\n",
                  "aeprep end")
        write(PP, pp)

    ma = read(MA)
    if "M236Trace.h" not in ma:
        ma = sub1(ma, '#include "MainApplication.h"',
                  '#include "MainApplication.h"\n#include "M236Trace.h"', "ma include")
        ma = sub1(ma,
                  "void MainApplication::initialise(const juce::String& commandLine)\n{\n",
                  "void MainApplication::initialise(const juce::String& commandLine)\n{\n"
                  "    m236::mark(\"app_init_start\");\n", "ma init start")
        ma = sub1(ma,
                  "    engine->getAffinityManager().applyMessageThreadPolicy();\n",
                  "    engine->getAffinityManager().applyMessageThreadPolicy();\n"
                  "    m236::mark(\"app_init_end\");\n", "ma init end")
        ma = sub1(ma,
                  "void MainApplication::shutdown()\n{\n",
                  "void MainApplication::shutdown()\n{\n"
                  "    m236::mark(\"shutdown_start\");\n", "ma shutdown start")
        ma = sub1(ma,
                  "    timeEndPeriod(1);\n",
                  "    timeEndPeriod(1);\n    m236::mark(\"shutdown_end\");\n", "ma shutdown end")
        write(MA, ma)

    mw = sub1(mw, '#include "MainWindow.h"',
              '#include "MainWindow.h"\n#include "M236Trace.h"', "mw include")
    mw = sub1(mw,
              "void MainWindow::loadSettings()\n{\n    DeviceSettings::loadSettings (audioDeviceManager, audioEngine);\n}",
              "void MainWindow::loadSettings()\n{\n"
              "    m236::mark(\"s2_begin\");\n"
              "    DeviceSettings::loadSettings (audioDeviceManager, audioEngine);\n"
              "    m236::mark(\"s2_end\");\n}",
              "mw s2")

    mw = sub1(mw,
              "    DeviceSettings::saveSettings (audioDeviceManager, audioEngine);\n",
              "    m236::mark(\"s4_begin\");\n"
              "    DeviceSettings::saveSettings (audioDeviceManager, audioEngine);\n"
              "    m236::mark(\"s4_end\");\n", "s4")

    write(DS, ds)
    write(MW, mw)
    print("instrumented")


def revert():
    for p in (DS, MW, AP, PP, MA):
        if os.path.exists(p):
            subprocess.run([GIT, "-C", W, "checkout", "--", os.path.relpath(p, W).replace("\\", "/")], check=True)
    if os.path.exists(HDR):
        os.remove(HDR)
    print("reverted")


if __name__ == "__main__":
    (apply if (len(sys.argv) > 1 and sys.argv[1] == "apply") else revert)()
