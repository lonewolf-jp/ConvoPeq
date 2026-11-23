//============================================================================
// MainWindow.cpp ── v0.2 (JUCE 8.0.12対応)
//
// メインウィンドウの実装
// UIコンポーネントの配置とオーディオデバイス管理を行う
//============================================================================
#include "MainWindow.h"
#include "MainApplication.h"
#include <cmath>
#include <limits>
#include "InputInspection.h"   // ★ B-05/B-06: XML のサイズ・深さ検査
#include "AsioBlacklistSetup.h"   // ★ B-17: ブラックリストの用意と通知判断
#include "audioengine/AtomicAccess.h"

namespace
{
    // ★ C-15: containsOnly() は空文字を許し、"1+2" のような文字種だけ合う文字列も
    //   通してしまう（getIntValue() は先頭の数値だけを返す）。CLI 入力は外部入力なので、
    //   「符号は先頭 1 個、残りは数字だけ」を明示的に検査する。
    bool tryParseIntOption(const juce::String& text, int& outValue)
    {
        const auto trimmed = text.trim();
        if (trimmed.isEmpty())
            return false;

        int start = 0;
        if (trimmed[0] == '+' || trimmed[0] == '-')
            start = 1;
        if (start >= trimmed.length())
            return false;

        for (int i = start; i < trimmed.length(); ++i)
            if (!juce::CharacterFunctions::isDigit(trimmed[i]))
                return false;

        const auto parsed = trimmed.getLargeIntValue();
        if (parsed < static_cast<juce::int64>(std::numeric_limits<int>::min())
            || parsed > static_cast<juce::int64>(std::numeric_limits<int>::max()))
            return false;

        outValue = static_cast<int>(parsed);
        return true;
    }

    bool tryParseFloatOption(const juce::String& text, float& outValue)
    {
        const auto trimmed = text.trim();
        if (trimmed.isEmpty())
            return false;

        int start = 0;
        if (trimmed[0] == '+' || trimmed[0] == '-')
            start = 1;

        bool seenDigit = false;
        bool seenDot = false;
        for (int i = start; i < trimmed.length(); ++i)
        {
            const auto ch = trimmed[i];
            if (ch == '.')
            {
                if (seenDot)
                    return false;
                seenDot = true;
                continue;
            }
            if (!juce::CharacterFunctions::isDigit(ch))
                return false;
            seenDigit = true;
        }

        if (!seenDigit)
            return false;
        outValue = trimmed.getFloatValue();
        return true;
    }

    juce::String normalizeCliValue(juce::String value)
    {
        return value.trim().toLowerCase().replace(" ", "").replace("_", "").replace("-", "").replace(">", "");
    }

    bool parseCliPhaseMode(const juce::String& value, ConvolverProcessor::PhaseMode& outMode)
    {
        const auto normalized = normalizeCliValue(value);
        if (normalized == "asis")
        {
            outMode = ConvolverProcessor::PhaseMode::AsIs;
            return true;
        }

        if (normalized == "mixed")
        {
            outMode = ConvolverProcessor::PhaseMode::Mixed;
            return true;
        }

        if (normalized == "minimum" || normalized == "min")
        {
            outMode = ConvolverProcessor::PhaseMode::Minimum;
            return true;
        }

        return false;
    }

    bool parseCliOrderMode(const juce::String& value, int& outModeId)
    {
        const auto normalized = normalizeCliValue(value);

        if (normalized == "conv" || normalized == "convolver")
        {
            outModeId = 1;
            return true;
        }

        if (normalized == "peq" || normalized == "eq")
        {
            outModeId = 2;
            return true;
        }

        if (normalized == "convpeq" || normalized == "convolverpeq")
        {
            outModeId = 3;
            return true;
        }

        if (normalized == "peqconv" || normalized == "eqconvolver")
        {
            outModeId = 4;
            return true;
        }

        return false;
    }

    bool parseCliNoiseShaper(const juce::String& value, AudioEngine::NoiseShaperType& outType)
    {
        const auto normalized = normalizeCliValue(value);

        if (normalized == "psycho" || normalized == "psychoacoustic")
        {
            outType = AudioEngine::NoiseShaperType::Psychoacoustic;
            return true;
        }

        if (normalized == "fixed4" || normalized == "fixed4tap")
        {
            outType = AudioEngine::NoiseShaperType::Fixed4Tap;
            return true;
        }

        if (normalized == "adaptive" || normalized == "adaptive9" || normalized == "adaptive9thorder")
        {
            outType = AudioEngine::NoiseShaperType::Adaptive9thOrder;
            return true;
        }

        if (normalized == "fixed15" || normalized == "fixed15tap")
        {
            outType = AudioEngine::NoiseShaperType::Fixed15Tap;
            return true;
        }

        return false;
    }

    bool parseCliLearningMode(const juce::String& value, convo::NoiseShaperLearningMode& outMode)
    {
        const auto normalized = normalizeCliValue(value);

        if (normalized == "1" || normalized == "shortest")
        {
            outMode = convo::NoiseShaperLearningMode::Shortest;
            return true;
        }

        if (normalized == "2" || normalized == "short")
        {
            outMode = convo::NoiseShaperLearningMode::Short;
            return true;
        }

        if (normalized == "3" || normalized == "middle" || normalized == "mid")
        {
            outMode = convo::NoiseShaperLearningMode::Middle;
            return true;
        }

        if (normalized == "4" || normalized == "long")
        {
            outMode = convo::NoiseShaperLearningMode::Long;
            return true;
        }

        if (normalized == "5" || normalized == "ultra")
        {
            outMode = convo::NoiseShaperLearningMode::Ultra;
            return true;
        }

        if (normalized == "6" || normalized == "continuous" || normalized == "cont")
        {
            outMode = convo::NoiseShaperLearningMode::Continuous;
            return true;
        }

        return false;
    }

    juce::String formatSaturationValue(float value)
    {
        return juce::String(value, 2);
    }

   #if JUCE_WINDOWS && JUCE_DEBUG
    void forceSoftwareRendererIfAvailable(juce::TopLevelWindow& window)
    {
        if (auto* peer = window.getPeer())
        {
            const auto engines = peer->getAvailableRenderingEngines();
            const int softwareIndex = engines.indexOf("Software Renderer");

            if (softwareIndex >= 0 && peer->getCurrentRenderingEngine() != softwareIndex)
            {
                peer->setCurrentRenderingEngine(softwareIndex);
                juce::Logger::writeToLog("[MainWindow] Forced Software Renderer for debug window: " + window.getName());
            }
        }
    }
   #endif

    class SettingsWindow : public juce::DocumentWindow
    {
    public:

        SettingsWindow (const juce::String& name, juce::Colour backgroundColour, int buttons)
            : DocumentWindow (name, backgroundColour, buttons)
        {
            setOpaque (true);
            setUsingNativeTitleBar (true);
        }

        void closeButtonPressed() override
        {
            if (onClose)
                onClose();

            setVisible (false);
        }

        std::unique_ptr<juce::AccessibilityHandler> createAccessibilityHandler() override
        {
           #if JUCE_WINDOWS && JUCE_DEBUG
            return createIgnoredAccessibilityHandler(*this);
           #else
            return juce::DocumentWindow::createAccessibilityHandler();
           #endif
        }

        std::function<void()> onClose;
    };

    // バージョン情報を表示するコンポーネント
    class AboutComponent : public juce::Component
    {
    public:
        AboutComponent()
        {
            setOpaque (true);
            setSize (400, 200);
        }

        void paint (juce::Graphics& g) override
        {
            g.fillAll (juce::Colours::darkgrey);

            auto area = getLocalBounds().reduced(20);

            g.setColour (juce::Colours::white);
            g.setFont (juce::FontOptions (24.0f, juce::Font::bold));
            g.drawText (juce::String(ProjectInfo::projectName), area.removeFromTop (40), juce::Justification::centred);
            g.setFont (juce::FontOptions (16.0f));
            g.drawText ("Version " + juce::String(ProjectInfo::versionString), area.removeFromTop (30), juce::Justification::centred);
            g.setColour (juce::Colours::lightgrey);
            g.setFont (juce::FontOptions (14.0f));
            g.drawText (juce::String(ProjectInfo::companyName), area.removeFromTop (20), juce::Justification::centred);
            g.drawText ("Xf policy: Sm=Smooth, Hd=HardReset, Dr=DryAsOld", area.removeFromBottom (18), juce::Justification::centredBottom);
            g.drawText ("Made with JUCE", area.removeFromBottom (20), juce::Justification::centredBottom);
        }
    };
}

//==============================================================================
MainWindow::MainWindow (const juce::String& name)
    : DocumentWindow (name,
                      juce::Desktop::getInstance().getDefaultLookAndFeel()
                          .findColour (juce::ResizableWindow::backgroundColourId),
                      DocumentWindow::allButtons)
{
    setOpaque (true);
    setUsingNativeTitleBar (true);
    setResizable (true, true);
    setResizeLimits (720, 760, 10000, 10000);
    setSize (960, 980);

    // ── ASIO Blacklist 初期化 ──
    auto exeDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
    const auto blacklistResolution = convo::asio::resolveBlacklistFile (
        exeDir, juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory));
    auto blacklistFile = blacklistResolution.file;

    if (blacklistResolution.status == convo::asio::BlacklistFileStatus::readyInUserDirectory)
    {
        // ★ B-13: exe ディレクトリが書込不可（Program Files 配下など）だと既定リストの
        //   作成に失敗する。ユーザー領域へフォールバックしたことを残す。
        juce::Logger::writeToLog ("[MainWindow] ASIO blacklist fell back to: "
                                  + blacklistFile.getFullPathName());
    }
    else if (blacklistResolution.status == convo::asio::BlacklistFileStatus::unavailable)
    {
        // ★ C-16: 書込み失敗を無視すると、存在しないファイルを loadFromFile して
        //   「ブラックリストが空」のまま起動する。失敗はログに残す。
        juce::Logger::writeToLog ("[ASIO] blacklist file could not be created: "
                                  + blacklistFile.getFullPathName());
        juce::Logger::writeToLog ("[MainWindow] ASIO blacklist unavailable in both locations");
    }

    // ★ B-17: 用意も読み込みもできなかった場合、除外が効かないまま起動していることを
    //   利用者へ伝える。従来はログだけだった。自動実行（--cli-* / 証跡モード）では
    //   ダイアログを出さない（CI とテストを止めないため）。
    const bool blacklistLoaded = asioBlacklist.loadFromFile (blacklistFile);

    if (! blacklistLoaded)
    {
        juce::Logger::writeToLog ("[MainWindow] ASIO blacklist could not be created or read: "
                                  + blacklistFile.getFullPathName());

        if (convo::asio::shouldShowBlacklistWarning (
                blacklistLoaded,
                convo::asio::isAutomationInvocation (juce::JUCEApplicationBase::getCommandLineParameters())))
        {
            juce::NativeMessageBox::showAsync (
                juce::MessageBoxOptions()
                    .withIconType (juce::MessageBoxIconType::WarningIcon)
                    .withTitle ("ASIO blacklist")
                    .withMessage ("The ASIO driver blacklist could not be created or read:\n"
                                  + blacklistFile.getFullPathName()
                                  + "\nUnstable drivers such as ASIO4ALL or BRAVO-HD may appear "
                                    "in the device list."),
                nullptr);
        }
    }

    DeviceSettings::applyAsioBlacklist (audioDeviceManager, asioBlacklist);

    // エンジンを先に初期化してデフォルトのサンプルレート(48kHz)を設定
    audioEngine.initialize();

    audioEngineProcessor = std::make_unique<AudioEngineProcessor>(audioEngine);
    audioProcessorPlayer.setDoublePrecisionProcessing(true);
    audioProcessorPlayer.setProcessor(audioEngineProcessor.get());
    audioEngine.addChangeListener (this);

    // ★ [work63] オーディオスレッドが開始される前に、前回の優先度設定を読み込む
    DeviceSettings::preloadThreadPriorityMode(audioEngine);

    audioDeviceManager.addAudioCallback (&audioProcessorPlayer);

    juce::Component::SafePointer<MainWindow> safeThis(this);
    audioEngine.setAdaptiveAutosaveCallback([safeThis]
    {
        juce::MessageManager::callAsync([safeThis]
        {
            if (safeThis == nullptr)
                return;

            DeviceSettings::saveSettings(safeThis->audioDeviceManager, safeThis->audioEngine);
        });
    });

    // 設定読み込み（ブラックリスト適用後に実行することで、除外されたデバイスの自動ロードを防ぐ）
    // この時点でrebuildが呼ばれても、有効なサンプルレートが設定されている
    loadSettings();

    // UIコンポーネントの作成
    createUIComponents();

    startTimer (500); // CPU使用率の更新頻度を上げる (500ms)
}

void MainWindow::showMainWindowAsync()
{
    juce::Component::SafePointer<MainWindow> safeThis(this);
    juce::MessageManager::callAsync([safeThis]
    {
        if (safeThis == nullptr)
            return;

        safeThis->setVisible(true);

       #if JUCE_WINDOWS && JUCE_DEBUG
        forceSoftwareRendererIfAvailable(*safeThis);
       #endif

        safeThis->toFront(true);
    });
}

void MainWindow::runCommandLineAutomation(const juce::String& commandLine)
{
    const auto trimmedCommandLine = commandLine.trim();
    if (trimmedCommandLine.isEmpty())
        return;

    juce::StringArray tokens;
    tokens.addTokens(trimmedCommandLine, true);
    tokens.trim();
    tokens.removeEmptyStrings();

    if (tokens.isEmpty())
        return;

    const auto hasFlag = [&tokens](const juce::String& flag)
    {
        return tokens.contains(flag, true);
    };

    const auto findValue = [&tokens](const juce::String& key) -> juce::String
    {
        for (int i = 0; i < tokens.size(); ++i)
        {
            if (!tokens[i].equalsIgnoreCase(key))
                continue;

            if (i + 1 < tokens.size())
                return tokens[i + 1];

            return {};
        }

        return {};
    };

    const bool hasAutomationFlags =
        hasFlag("--cli-run")
        || hasFlag("--cli-start-learning")
        || hasFlag("--cli-resume-learning")
        || !findValue("--cli-ir").isEmpty()
        || !findValue("--cli-device-type").isEmpty()
        || !findValue("--cli-buffer-samples").isEmpty()
        || !findValue("--cli-sample-rate-hz").isEmpty()
        || !findValue("--cli-phase").isEmpty()
        || !findValue("--cli-order").isEmpty()
        || !findValue("--cli-dither-bit-depth").isEmpty()
        || !findValue("--cli-noise-shaper").isEmpty()
        || !findValue("--cli-post-load-dither-bit-depth").isEmpty()
        || !findValue("--cli-post-load-delay-ms").isEmpty()
        || !findValue("--cli-ir-reload-count").isEmpty()
        || !findValue("--cli-ir-reload-interval-ms").isEmpty()
        || !findValue("--cli-bypass-burst-count").isEmpty()
        || !findValue("--cli-bypass-burst-interval-ms").isEmpty()
        || !findValue("--cli-bypass-burst-value").isEmpty()
        || !findValue("--cli-intent-burst-count").isEmpty()
        || !findValue("--cli-intent-burst-interval-ms").isEmpty()
        || hasFlag("--cli-rebuild")
        || !findValue("--cli-eq-band").isEmpty()
        || !findValue("--cli-eq-freq-hz").isEmpty()
        || !findValue("--cli-eq-gain-db").isEmpty()
        || !findValue("--cli-eq-q").isEmpty()
        || !findValue("--cli-eq-type").isEmpty()
        || !findValue("--cli-target-ir-sec").isEmpty()
        || !findValue("--cli-debounce-ms").isEmpty()
        || !findValue("--cli-f1-hz").isEmpty()
        || !findValue("--cli-f2-hz").isEmpty()
        || !findValue("--cli-learning-action").isEmpty()
        || !findValue("--cli-learning-mode").isEmpty()
        || !findValue("--cli-ir-alt-a").isEmpty()
        || !findValue("--cli-ir-alt-b").isEmpty()
        || !findValue("--cli-exit-ms").isEmpty()
        || !findValue("--cli-log-file").isEmpty()
        || !findValue("--cli-measure-ir").isEmpty()
        || !findValue("--cli-measure-async-load-count").isEmpty()
        || !findValue("--cli-measure-late-load-at-ms").isEmpty()
        || !findValue("--cli-measure-bypass-from-ms").isEmpty()
        || !findValue("--cli-measure-bypass-span-ms").isEmpty()
        || !findValue("--cli-measure-eq-from-ms").isEmpty()
        || !findValue("--cli-measure-eq-span-ms").isEmpty()
        || hasFlag("--cli-measure-enable-upgrade");

    // ★ v14.47: --cli-log-file <path> — 診断ログをファイルに出力
    if (const auto logFileValue = findValue("--cli-log-file"); !logFileValue.isEmpty())
    {
        const auto logFile = juce::File::getCurrentWorkingDirectory().getChildFile(logFileValue);
        // ★ C-16 / review: 以前は release() して所有権を捨てていた。
        //   MainWindow より長生きさせる必要がある（~AudioEngine のログも CLI ファイルへ
        //   出すため）ので、MainApplication へ所有権を移す。移せない場合だけ MainWindow が持つ。
        auto newLogger = std::make_unique<SizeCappedFileLogger>(logFile, "ConvoPeq CLI Log",
                                                                static_cast<juce::int64>(8) << 20);
        if (auto* app = dynamic_cast<MainApplication*>(juce::JUCEApplication::getInstance()))
        {
            app->adoptCliLogger(std::move(newLogger));
        }
        else
        {
            cliFileLogger = std::move(newLogger);
            previousLogger = juce::Logger::getCurrentLogger();
            juce::Logger::setCurrentLogger(cliFileLogger.get());
        }
        juce::Logger::writeToLog("[CLI] Log file: " + logFile.getFullPathName());
    }

    if (!hasAutomationFlags)
    {
        cliAutomationTelemetryLoggingEnabled = false;
        convo::publishAtomic(cliAutomationCallbacksEnabled, false, std::memory_order_release);
        audioEngine.setCliProcessingTelemetryEnabled(false);
        cliAudioSetupRequested = false;
        cliAudioSetupMismatchLogged = false;
        cliRequestedBufferSamples = 0;
        cliRequestedSampleRateHz = 0.0;
        return;
    }

    // ★ 計測専用（既定 OFF）: 走行中 Worker を作る opt-in 経路。
    //   --cli-measure-ir / --cli-measure-async-load-count / --cli-measure-enable-upgrade を
    //   指定したときだけ動く。通常実行では何もしない（production 挙動不変）。
    if (const auto measureIr = findValue("--cli-measure-ir"); !measureIr.isEmpty())
    {
        const auto irFile = juce::File::getCurrentWorkingDirectory().getChildFile (measureIr);
        const int asyncCount = findValue("--cli-measure-async-load-count").getIntValue();
        const int asyncIntervalMs = juce::jmax (50, findValue("--cli-measure-async-load-interval-ms").getIntValue());
        const bool enableUpgrade = hasFlag ("--cli-measure-enable-upgrade");
        juce::Component::SafePointer<MainWindow> safeThis (this);

        if (enableUpgrade)
            audioEngine.getConvolverProcessor().setEnableProgressiveUpgrade (true);

        for (int i = 0; asyncCount > 0 && i < juce::jlimit (1, 64, asyncCount); ++i)
        {
            juce::Timer::callAfterDelay (300 + i * asyncIntervalMs, [safeThis, irFile, enableUpgrade]()
            {
                if (auto* w = safeThis.getComponent())
                {
                    auto& proc = w->getAudioEngine()->getConvolverProcessor();

                    if (enableUpgrade)
                    {
                        // 設定ロードで false に戻されることがあるため、各回の直前に立て直す。
                        proc.setEnableProgressiveUpgrade (true);
                        proc.loadIR (irFile);          // → startProgressiveUpgrade（W6 を作る）
                    }
                    else
                    {
                        proc.loadImpulseResponse (irFile, false);   // → LoaderThread（W5 を作る）
                    }
                }
            });
        }

        // 計測専用の負荷: 指定時刻から bypass と EQ gain を交互に切り替える（競合状態の再現）。
        if (const auto fromMs = findValue("--cli-measure-bypass-from-ms"); !fromMs.isEmpty())
        {
            const int from = juce::jlimit (0, 600000, fromMs.getIntValue());
            constexpr int kIntervalMs = 5;
            const int kSpanMs = juce::jlimit (20, 5000,
                findValue ("--cli-measure-bypass-span-ms").getIntValue() > 0
                    ? findValue ("--cli-measure-bypass-span-ms").getIntValue() : 2000);
            for (int i = 0; i < kSpanMs / kIntervalMs; ++i)
            {
                juce::Timer::callAfterDelay (from + i * kIntervalMs,
                                             [safeThis, i]()
                {
                    if (auto* w = safeThis.getComponent())
                    {
                        // bypass のみを切り替える（EQ gain の切替は submitRebuildIntent を
                        // 誘発して進行中ロードをキャンセルするため、この条件には含めない）。
                        w->getAudioEngine()->setConvolverBypassRequested ((i % 2) == 0);
                    }
                });
            }
        }

        // 計測専用の負荷（EQ gain 切替）: submitRebuildIntent を誘発する条件。
        if (const auto eqFromMs = findValue ("--cli-measure-eq-from-ms"); !eqFromMs.isEmpty())
        {
            const int from = juce::jlimit (0, 600000, eqFromMs.getIntValue());
            constexpr int kIntervalMs = 5;
            const int spanMs = juce::jlimit (20, 5000,
                findValue ("--cli-measure-eq-span-ms").getIntValue() > 0
                    ? findValue ("--cli-measure-eq-span-ms").getIntValue() : 260);
            for (int i = 0; i < spanMs / kIntervalMs; ++i)
            {
                juce::Timer::callAfterDelay (from + i * kIntervalMs,
                                             [safeThis, i]()
                {
                    if (auto* w = safeThis.getComponent())
                        w->getAudioEngine()->setEQBandGain (0, ((i % 2) == 0) ? 0.0f : 3.0f);
                });
            }
        }

        // 終了直前に 1 回だけロードして、shutdown 時点で走行中の Worker を作る。
        if (const auto lateMs = findValue("--cli-measure-late-load-at-ms"); !lateMs.isEmpty())
        {
            const int atMs = juce::jlimit (100, 600000, lateMs.getIntValue());
            juce::Timer::callAfterDelay (atMs, [safeThis, irFile, enableUpgrade]()
            {
                if (auto* w = safeThis.getComponent())
                {
                    auto& proc = w->getAudioEngine()->getConvolverProcessor();
                    if (enableUpgrade)
                    {
                        proc.setEnableProgressiveUpgrade (true);
                        proc.loadIR (irFile);
                    }
                    else
                    {
                        proc.loadImpulseResponse (irFile, false);
                    }
                }
            });
        }
    }
    else if (hasFlag ("--cli-measure-enable-upgrade"))
    {
        audioEngine.getConvolverProcessor().setEnableProgressiveUpgrade (true);
    }

    juce::Logger::writeToLog("[CLI] Automation requested: " + trimmedCommandLine);
    cliAutomationTelemetryLoggingEnabled = true;
    convo::publishAtomic(cliAutomationCallbacksEnabled, true, std::memory_order_release);
    audioEngine.setCliProcessingTelemetryEnabled(true);
    audioEngine.setConvolverEnableProgressiveUpgrade(false);

    {
        if (const auto deviceTypeValue = findValue("--cli-device-type"); !deviceTypeValue.isEmpty())
        {
            juce::StringArray availableTypeNames;
            for (const auto* type : audioDeviceManager.getAvailableDeviceTypes())
            {
                if (type != nullptr)
                    availableTypeNames.add(type->getTypeName());
            }

            juce::Logger::writeToLog("[CLI_AUDIO_DEV_TYPES] available=" + availableTypeNames.joinIntoString(","));

            juce::String resolvedTypeName;
            for (const auto& typeName : availableTypeNames)
            {
                if (typeName.equalsIgnoreCase(deviceTypeValue)
                    || normalizeCliValue(typeName) == normalizeCliValue(deviceTypeValue))
                {
                    resolvedTypeName = typeName;
                    break;
                }
            }

            if (resolvedTypeName.isNotEmpty())
            {
                const auto currentType = audioDeviceManager.getCurrentAudioDeviceType();
                if (!currentType.equalsIgnoreCase(resolvedTypeName))
                {
                    audioDeviceManager.setCurrentAudioDeviceType(resolvedTypeName, false);
                    const auto switchedType = audioDeviceManager.getCurrentAudioDeviceType();

                    if (!switchedType.equalsIgnoreCase(resolvedTypeName))
                    {
                        juce::Logger::writeToLog("[CLI_AUDIO_DEV_SWITCH] failed requested="
                                                 + deviceTypeValue + " resolved=" + resolvedTypeName
                                                 + " current=" + switchedType);
                    }
                    else
                    {
                        juce::Logger::writeToLog("[CLI_AUDIO_DEV_SWITCH] success requested="
                                                 + deviceTypeValue + " resolved=" + resolvedTypeName
                                                 + " current=" + switchedType);
                    }
                }
                else
                {
                    juce::Logger::writeToLog("[CLI_AUDIO_DEV_SWITCH] skipped requested="
                                             + deviceTypeValue + " resolved=" + resolvedTypeName
                                             + " reason=already_current");
                }
            }
            else
            {
                juce::Logger::writeToLog("[CLI_AUDIO_DEV_SWITCH] unknown requested="
                                         + deviceTypeValue + " available=" + availableTypeNames.joinIntoString(","));
            }
        }

        bool hasRequestedAudioSetup = false;
        int requestedBufferSamples = 0;
        double requestedSampleRateHz = 0.0;

        if (const auto bufferValue = findValue("--cli-buffer-samples"); !bufferValue.isEmpty())
        {
            int parsedBuffer = 0;
            if (tryParseIntOption(bufferValue, parsedBuffer) && parsedBuffer > 0)
            {
                requestedBufferSamples = parsedBuffer;
                hasRequestedAudioSetup = true;
            }
            else
            {
                juce::Logger::writeToLog("[CLI] Invalid --cli-buffer-samples: " + bufferValue);
            }
        }

        if (const auto sampleRateValue = findValue("--cli-sample-rate-hz"); !sampleRateValue.isEmpty())
        {
            float parsedSampleRate = 0.0f;
            if (tryParseFloatOption(sampleRateValue, parsedSampleRate) && parsedSampleRate > 0.0f)
            {
                requestedSampleRateHz = static_cast<double>(parsedSampleRate);
                hasRequestedAudioSetup = true;
            }
            else
            {
                juce::Logger::writeToLog("[CLI] Invalid --cli-sample-rate-hz: " + sampleRateValue);
            }
        }

        if (hasRequestedAudioSetup)
        {
            juce::AudioDeviceManager::AudioDeviceSetup setup;
            audioDeviceManager.getAudioDeviceSetup(setup);
            const auto setupBefore = setup;

            if (auto* currentDevice = audioDeviceManager.getCurrentAudioDevice())
            {
                juce::StringArray bufferSizes;
                for (const auto size : currentDevice->getAvailableBufferSizes())
                    bufferSizes.add(juce::String(size));

                juce::StringArray sampleRates;
                for (const auto rate : currentDevice->getAvailableSampleRates())
                    sampleRates.add(juce::String(rate, 1));

                juce::Logger::writeToLog("[CLI_AUDIO_CFG_CAPS] deviceType="
                                         + audioDeviceManager.getCurrentAudioDeviceType()
                                         + " deviceName=" + currentDevice->getName()
                                         + " availableBufferSizes=" + bufferSizes.joinIntoString(",")
                                         + " availableSampleRatesHz=" + sampleRates.joinIntoString(","));
            }
            else
            {
                juce::Logger::writeToLog("[CLI_AUDIO_CFG_CAPS] deviceUnavailable=1");
            }

            if (requestedBufferSamples > 0)
                setup.bufferSize = requestedBufferSamples;
            if (requestedSampleRateHz > 0.0)
                setup.sampleRate = requestedSampleRateHz;

            cliAudioSetupRequested = true;
            cliAudioSetupMismatchLogged = false;
            cliRequestedBufferSamples = requestedBufferSamples;
            cliRequestedSampleRateHz = requestedSampleRateHz;

            juce::Logger::writeToLog("[CLI_AUDIO_CFG_REQ] requestedBufferSamples="
                                     + juce::String(requestedBufferSamples)
                                     + " requestedSampleRateHz=" + juce::String(requestedSampleRateHz, 1)
                                     + " setupBeforeBufferSamples=" + juce::String(setupBefore.bufferSize)
                                     + " setupBeforeSampleRateHz=" + juce::String(setupBefore.sampleRate, 1)
                                     + " setupTargetBufferSamples=" + juce::String(setup.bufferSize)
                                     + " setupTargetSampleRateHz=" + juce::String(setup.sampleRate, 1));

            if (const auto err = audioDeviceManager.setAudioDeviceSetup(setup, true); err.isNotEmpty())
            {
                juce::Logger::writeToLog("[CLI] Failed to apply audio setup override: " + err);
            }
            else
            {
                juce::AudioDeviceManager::AudioDeviceSetup setupAfter;
                audioDeviceManager.getAudioDeviceSetup(setupAfter);

                if (auto* device = audioDeviceManager.getCurrentAudioDevice())
                {
                    juce::Logger::writeToLog("[CLI] Applied audio setup override: sampleRateHz="
                                             + juce::String(device->getCurrentSampleRate(), 1)
                                             + " bufferSamples=" + juce::String(device->getCurrentBufferSizeSamples()));

                    juce::Logger::writeToLog("[CLI_AUDIO_CFG_COMMIT] setupAfterBufferSamples="
                                             + juce::String(setupAfter.bufferSize)
                                             + " setupAfterSampleRateHz=" + juce::String(setupAfter.sampleRate, 1)
                                             + " deviceBufferSamples=" + juce::String(device->getCurrentBufferSizeSamples())
                                             + " deviceSampleRateHz=" + juce::String(device->getCurrentSampleRate(), 1));
                }
                else
                {
                    juce::Logger::writeToLog("[CLI] Applied audio setup override (device unavailable for readback)");
                    juce::Logger::writeToLog("[CLI_AUDIO_CFG_COMMIT] setupAfterBufferSamples="
                                             + juce::String(setupAfter.bufferSize)
                                             + " setupAfterSampleRateHz=" + juce::String(setupAfter.sampleRate, 1)
                                             + " deviceUnavailable=1");
                }
            }
        }
        else
        {
            cliAudioSetupRequested = false;
            cliAudioSetupMismatchLogged = false;
            cliRequestedBufferSamples = 0;
            cliRequestedSampleRateHz = 0.0;
        }
    }

    if (const auto orderValue = findValue("--cli-order"); !orderValue.isEmpty())
    {
        int modeId = 0;
        if (parseCliOrderMode(orderValue, modeId))
        {
            orderModeBox.setSelectedId(modeId, juce::dontSendNotification);
            orderModeBoxChanged();
            juce::Logger::writeToLog("[CLI] Applied order mode: " + orderValue);
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Unknown --cli-order value: " + orderValue);
        }
    }

    if (const auto phaseValue = findValue("--cli-phase"); !phaseValue.isEmpty())
    {
        ConvolverProcessor::PhaseMode phaseMode {};
        if (parseCliPhaseMode(phaseValue, phaseMode))
        {
            audioEngine.setConvolverPhaseMode(phaseMode);
            juce::Logger::writeToLog("[CLI] Applied phase mode: " + phaseValue);
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Unknown --cli-phase value: " + phaseValue);
        }
    }

    if (const auto targetIRLengthValue = findValue("--cli-target-ir-sec"); !targetIRLengthValue.isEmpty())
    {
        float seconds = 0.0f;
        if (tryParseFloatOption(targetIRLengthValue, seconds))
        {
            audioEngine.setConvolverTargetIRLength(seconds, true);
            juce::Logger::writeToLog("[CLI] Applied target IR length (sec): " + juce::String(seconds, 3));
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-target-ir-sec: " + targetIRLengthValue);
        }
    }

    if (const auto debounceValue = findValue("--cli-debounce-ms"); !debounceValue.isEmpty())
    {
        int debounceMs = 0;
        if (tryParseIntOption(debounceValue, debounceMs))
        {
            audioEngine.setConvolverRebuildDebounceMs(debounceMs);
            juce::Logger::writeToLog("[CLI] Applied rebuild debounce (ms): " + juce::String(debounceMs));
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-debounce-ms: " + debounceValue);
        }
    }

    if (const auto f1Value = findValue("--cli-f1-hz"); !f1Value.isEmpty())
    {
        float hz = 0.0f;
        if (tryParseFloatOption(f1Value, hz))
        {
            audioEngine.setConvolverMixedTransitionStartHz(hz);
            juce::Logger::writeToLog("[CLI] Applied mixed f1 (Hz): " + juce::String(hz, 2));
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-f1-hz: " + f1Value);
        }
    }

    if (const auto f2Value = findValue("--cli-f2-hz"); !f2Value.isEmpty())
    {
        float hz = 0.0f;
        if (tryParseFloatOption(f2Value, hz))
        {
            audioEngine.setConvolverMixedTransitionEndHz(hz);
            juce::Logger::writeToLog("[CLI] Applied mixed f2 (Hz): " + juce::String(hz, 2));
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-f2-hz: " + f2Value);
        }
    }

    if (const auto ditherValue = findValue("--cli-dither-bit-depth"); !ditherValue.isEmpty())
    {
        int bitDepth = 0;
        if (tryParseIntOption(ditherValue, bitDepth))
        {
            audioEngine.setDitherBitDepth(bitDepth);
            juce::Logger::writeToLog("[CLI] Applied dither bit depth: " + juce::String(bitDepth));
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-dither-bit-depth: " + ditherValue);
        }
    }

    if (const auto noiseShaperValue = findValue("--cli-noise-shaper"); !noiseShaperValue.isEmpty())
    {
        AudioEngine::NoiseShaperType noiseShaperType {};
        if (parseCliNoiseShaper(noiseShaperValue, noiseShaperType))
        {
            audioEngine.setNoiseShaperType(noiseShaperType);
            juce::Logger::writeToLog("[CLI] Applied noise shaper: " + noiseShaperValue);
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Unknown --cli-noise-shaper value: " + noiseShaperValue);
        }
    }

    {
        auto learningMode = audioEngine.getNoiseShaperLearningMode();
        if (const auto learningModeValue = findValue("--cli-learning-mode"); !learningModeValue.isEmpty())
        {
            convo::NoiseShaperLearningMode parsedMode {};
            if (parseCliLearningMode(learningModeValue, parsedMode))
            {
                learningMode = parsedMode;
                audioEngine.setNoiseShaperLearningMode(learningMode);
                juce::Logger::writeToLog("[CLI] Applied learning mode: " + learningModeValue);
            }
            else
            {
                juce::Logger::writeToLog("[CLI] Unknown --cli-learning-mode value: " + learningModeValue);
            }
        }

        bool shouldStartLearning = false;
        bool resumeLearning = false;

        if (hasFlag("--cli-start-learning"))
        {
            shouldStartLearning = true;
            resumeLearning = false;
        }

        if (hasFlag("--cli-resume-learning"))
        {
            shouldStartLearning = true;
            resumeLearning = true;
        }

        if (const auto learningActionValue = findValue("--cli-learning-action"); !learningActionValue.isEmpty())
        {
            const auto action = normalizeCliValue(learningActionValue);
            if (action == "start")
            {
                shouldStartLearning = true;
                resumeLearning = false;
            }
            else if (action == "resume")
            {
                shouldStartLearning = true;
                resumeLearning = true;
            }
            else
            {
                juce::Logger::writeToLog("[CLI] Unknown --cli-learning-action value: " + learningActionValue);
            }
        }

        if (shouldStartLearning)
        {
            audioEngine.startNoiseShaperLearning(learningMode, resumeLearning);
            juce::Logger::writeToLog("[CLI] Requested learning start: resume="
                                     + juce::String(static_cast<int>(resumeLearning))
                                     + " mode=" + juce::String(static_cast<int>(learningMode)));
        }
    }

    if (const auto irValue = findValue("--cli-ir"); !irValue.isEmpty())
    {
        juce::File irFile;
        if (juce::File::isAbsolutePath(irValue))
            irFile = juce::File(irValue);
        else
            irFile = juce::File::getCurrentWorkingDirectory().getChildFile(irValue);

        if (irFile.existsAsFile())
        {
            // Defer IR load to allow audio device to stabilize first
            const int irLoadDelayMs = 200;
            juce::Logger::writeToLog("[CLI] Deferred IR load: " + irFile.getFullPathName()
                                     + " (delayMs=" + juce::String(irLoadDelayMs) + ")");
            juce::Timer::callAfterDelay(irLoadDelayMs, [safeThis = juce::Component::SafePointer<MainWindow>(this), irFile]
            {
                if (safeThis == nullptr) return;
                if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire)) return;

                safeThis->audioEngine.requestConvolverPreset(irFile);
                juce::Logger::writeToLog("[CLI] Loading IR: " + irFile.getFullPathName());

                const bool irLoaded = safeThis->audioEngine.getConvolverProcessor().isIRLoaded();
                const int irLen = safeThis->audioEngine.getConvolverProcessor().getIRLength();
                juce::Logger::writeToLog("[CLI_IR] isIRLoaded=" + juce::String(static_cast<int>(irLoaded))
                                         + " irLen=" + juce::String(irLen));
            });

            int reloadCount = 0;
            if (const auto reloadCountValue = findValue("--cli-ir-reload-count"); !reloadCountValue.isEmpty())
            {
                int parsedCount = 0;
                if (tryParseIntOption(reloadCountValue, parsedCount))
                    reloadCount = juce::jmax(0, parsedCount);
            }

            int reloadIntervalMs = 300;
            if (const auto reloadIntervalValue = findValue("--cli-ir-reload-interval-ms"); !reloadIntervalValue.isEmpty())
            {
                int parsedInterval = 0;
                if (tryParseIntOption(reloadIntervalValue, parsedInterval))
                    reloadIntervalMs = juce::jmax(1, parsedInterval);
            }

            if (reloadCount > 0)
            {
                juce::Logger::writeToLog("[CLI] Scheduled IR reload storm: count="
                                         + juce::String(reloadCount)
                                         + " intervalMs=" + juce::String(reloadIntervalMs));

                for (int i = 1; i <= reloadCount; ++i)
                {
                    const int delayMs = i * reloadIntervalMs;
                    juce::Timer::callAfterDelay(delayMs, [safeThis = juce::Component::SafePointer<MainWindow>(this), irFile, i]
                    {
                        if (safeThis == nullptr)
                            return;

                        if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire))
                            return;

                        safeThis->audioEngine.requestConvolverPreset(irFile);
                        juce::Logger::writeToLog("[CLI] IR reload iteration=" + juce::String(i)
                                                 + " file=" + irFile.getFullPathName());
                    });
                }
            }
        }
        else
        {
            juce::Logger::writeToLog("[CLI] IR file not found: " + irFile.getFullPathName());
        }
    }


    // ========================================================================
    // \u2605 WORK113 \u5909\u66f4 2 (DIAG): \u7570\u306a\u308b IR \u3078\u306e\u5b9f\u4ea4\u63db\u3092\u6e2c\u308b\u4ea4\u4e92\u8a66\u9a13\u3002
    //   A \u2192 B \u2192 A \u2192 ... \u3092\u5b8c\u4e86\u99c6\u52d5\u3067\u9032\u3081\u3001\u5404 step \u3067 5 \u6761\u4ef6\u3068 publish \u5897\u5206\u3092\u78ba\u8a8d\u3059\u308b\u3002
    // ========================================================================
    {
        const auto altAValue = findValue("--cli-ir-alt-a");
        const auto altBValue = findValue("--cli-ir-alt-b");

        if (!altAValue.isEmpty() || !altBValue.isEmpty())
        {
            auto toFile = [](const juce::String& v)
            {
                return juce::File::isAbsolutePath(v)
                    ? juce::File(v)
                    : juce::File::getCurrentWorkingDirectory().getChildFile(v);
            };

            const juce::File altA = toFile(altAValue);
            const juce::File altB = toFile(altBValue);

            int altSteps = 6;
            if (const auto v = findValue("--cli-ir-alt-steps"); !v.isEmpty())
            {
                int parsed = 0;
                if (tryParseIntOption(v, parsed))
                    altSteps = juce::jmax(2, parsed);
            }

            int altSettleMs = 2000;
            if (const auto v = findValue("--cli-ir-alt-settle-ms"); !v.isEmpty())
            {
                int parsed = 0;
                if (tryParseIntOption(v, parsed))
                    altSettleMs = juce::jmax(100, parsed);
            }

            auto readIrLength = [](const juce::File& f, double& srOut, juce::int64& framesOut) -> bool
            {
                juce::AudioFormatManager fm;
                fm.registerBasicFormats();
                std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(f));
                if (reader == nullptr)
                    return false;
                srOut = reader->sampleRate;
                framesOut = reader->lengthInSamples;
                return true;
            };

            bool preflightOk = true;
            juce::String preflightReason;

            if (!altA.existsAsFile() || !altB.existsAsFile())
            {
                preflightOk = false;
                preflightReason = "ir_alt_file_missing";
            }

            double srA = 0.0, srB = 0.0;
            juce::int64 framesA = 0, framesB = 0;
            if (preflightOk && (!readIrLength(altA, srA, framesA) || !readIrLength(altB, srB, framesB)))
            {
                preflightOk = false;
                preflightReason = "ir_alt_header_unreadable";
            }

            if (preflightOk)
            {
                juce::MemoryBlock dataA, dataB;
                const bool readA = altA.loadFileAsData(dataA);
                const bool readB = altB.loadFileAsData(dataB);
                if (!readA || !readB || dataA.getSize() == 0 || dataB.getSize() == 0)
                {
                    // ★ 比較できない入力を許可しない（fail-closed）。
                    preflightOk = false;
                    preflightReason = "ir_alt_content_unreadable";
                }
                else if (dataA == dataB)
                {
                    preflightOk = false;
                    preflightReason = "ir_alt_identical";
                }
            }

            if (!preflightOk)
            {
                juce::Logger::writeToLog("[CLI_IR_ALT] preflight failed reason=" + preflightReason);
            }
            else
            {
                struct AltState
                {
                    juce::File fileA, fileB;
                    int expectedLenA = 0;
                    int expectedLenB = 0;
                    double fileSrA = 0.0;
                    double fileSrB = 0.0;
                    int totalSteps = 6;
                    int settleMs = 2000;
                    int stepIndex = 0;
                    int phase = 0;                 // 0=request,1=waitLoad,2=beginWindow,3=baseline,4=requestRebuild,5=waitPublish,6=verify,7=settle,8=done
                    int baselineRetries = 0;
                    std::uint64_t requestId = 0;
                    std::uint64_t genBefore = 0;
                    std::uint64_t pubBefore = 0;
                    std::uint64_t deadlineMs = 0;
                    std::uint64_t settleDeadlineMs = 0;
                    bool windowOpen = false;
                    bool allValid = true;
                    juce::String failReason;
                    std::shared_ptr<std::function<void()>> self;
                };

                auto st = std::make_shared<AltState>();
                st->fileA = altA;
                st->fileB = altB;
                st->expectedLenA = (int) framesA;
                st->expectedLenB = (int) framesB;
                st->fileSrA = srA;
                st->fileSrB = srB;
                st->totalSteps = altSteps;
                st->settleMs = altSettleMs;

                juce::Logger::writeToLog("[CLI_IR_ALT] preflight ok A=" + altA.getFullPathName()
                                         + " lenA=" + juce::String(st->expectedLenA)
                                         + " B=" + altB.getFullPathName()
                                         + " lenB=" + juce::String(st->expectedLenB)
                                         + " steps=" + juce::String(altSteps)
                                         + " settleMs=" + juce::String(altSettleMs));

                auto selfFn = std::make_shared<std::function<void()>>();
                st->self = selfFn;

                auto failStep = [st](const juce::String& reason)
                {
                    st->allValid = false;
                    st->failReason = reason;
                    st->phase = 8;
                    juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                             + " phase=fail reason=" + reason);
                };

                *selfFn = [safeThis = juce::Component::SafePointer<MainWindow>(this), st, failStep]()
                {
                    if (safeThis == nullptr)
                        return;
                    if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire))
                        return;

                    auto& engine = safeThis->audioEngine;
                    auto& conv = engine.getConvolverProcessor();
                    const auto nowMs = juce::Time::getMillisecondCounter();

                    switch (st->phase)
                    {
                    case 0: // request
                    {
                        if (st->stepIndex >= st->totalSteps)
                        {
                            st->phase = 8;
                            break;
                        }

                        const bool useA = ((st->stepIndex % 2) == 0);
                        const juce::File& f = useA ? st->fileA : st->fileB;
                        const int expected = useA ? st->expectedLenA : st->expectedLenB;

                        // ★ デバイスの内部 SR は CLI 解析時には未確定。
                        //   step 開始時に確認し、一致しなければ不成立にする。
                        // ★ 期限は SR の取得結果に依存せず判定する（期限超過後に SR が確定しても不成立）。
                        if (st->deadlineMs == 0)
                            st->deadlineMs = nowMs + 60000;
                        if (nowMs >= st->deadlineMs)
                        {
                            failStep("device_sr_unavailable");
                            break;
                        }

                        const double deviceSr = engine.getSampleRate();
                        if (deviceSr <= 0.0)
                            break;
                        if (std::abs((useA ? st->fileSrA : st->fileSrB) - deviceSr) > 1.0e-6)
                        {
                            failStep("irlen_not_verifiable");
                            break;
                        }

                        st->requestId = 1000 + (std::uint64_t) st->stepIndex;
                        st->genBefore = engine.getActiveRuntimeSnapshot().generation;
                        st->pubBefore = engine.worldRetirementTelemetry().acquireObserved();
                        st->baselineRetries = 0;
                        st->windowOpen = false;

                        juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                                 + " phase=request requestId=" + juce::String((juce::int64) st->requestId)
                                                 + " file=" + f.getFileName()
                                                 + " irLenExpected=" + juce::String(expected)
                                                 + " genBefore=" + juce::String((juce::int64) st->genBefore)
                                                 + " publishBefore=" + juce::String((juce::int64) st->pubBefore));

                        engine.requestConvolverPreset(f, st->requestId);
                        st->deadlineMs = nowMs + 60000;
                        st->phase = 1;
                        break;
                    }
                    case 1: // wait load (commit record + IR length + target FFT)
                    {
                        ConvolverProcessor::DiagCommitRecord rec {};
                        const bool haveRecord = conv.peekCommitRecord(st->requestId, rec);
                        const bool lenOk = conv.getIRLength() == (st->stepIndex % 2 == 0 ? st->expectedLenA : st->expectedLenB);
                        const bool fftOk = haveRecord && rec.fftSize >= conv.getTargetUpgradeFFTSize();

                        if (haveRecord && lenOk && fftOk)
                        {
                            juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                                     + " phase=loaded irLen=" + juce::String(rec.irLength)
                                                     + " fft=" + juce::String(rec.fftSize)
                                                     + " requestGen=" + juce::String((juce::int64) rec.requestGenId));
                            st->phase = 2;
                            break;
                        }
                        if (nowMs >= st->deadlineMs)
                        {
                            failStep(haveRecord ? "upgrade_incomplete" : "load_timeout");
                            break;
                        }
                        break;
                    }
                    case 2: // begin window (open + quiescence + 2 phases)
                    {
                        const auto out = engine.beginDiagWindow();
                        if (out != AudioEngine::RebuildDispatchOutcome::Accepted)
                        {
                            failStep("begin_window_" + juce::String((int) out));
                            break;
                        }
                        st->windowOpen = true;
                        st->deadlineMs = nowMs + 2000;
                        st->phase = 3;
                        break;
                    }
                    case 3: // A -> baseline -> B under diagPhaseMutex
                    {
                        const auto remaining = (st->deadlineMs > nowMs)
                            ? std::chrono::milliseconds((int) (st->deadlineMs - nowMs))
                            : std::chrono::milliseconds(0);
                        if (remaining.count() <= 0)
                        {
                            failStep("not_quiescent");
                            break;
                        }

                        std::unique_lock<std::timed_mutex> lock(engine.diagPhaseMutex(), std::defer_lock);
                        if (!lock.try_lock_for(remaining))
                        {
                            failStep("not_quiescent");
                            break;
                        }
                        if (juce::Time::getMillisecondCounter() >= st->deadlineMs)
                        {
                            lock.unlock();
                            failStep("not_quiescent");
                            break;
                        }

                        AudioEngine::DiagConditionsSnapshot a {};
                        engine.diagSnapshotConditions(a);
                        if (!AudioEngine::diagConditionsMet(a))
                        {
                            lock.unlock();
                            if (++st->baselineRetries > 3) { failStep("not_quiescent"); break; }
                            break;
                        }

                        st->genBefore = engine.getActiveRuntimeSnapshot().generation;
                        st->pubBefore = engine.worldRetirementTelemetry().acquireObserved();
                        // ★ Private の計測は変更 3 の範囲。ここではベースラインの位置だけを残す。
                        const std::uint64_t privProxy = 0;

                        AudioEngine::DiagConditionsSnapshot b {};
                        engine.diagSnapshotConditions(b);
                        lock.unlock();

                        const bool same =
                            a.hasPendingTask == b.hasPendingTask
                            && a.workerBusy == b.workerBusy
                            && a.swapPending == b.swapPending
                            && a.publicationResidency == b.publicationResidency
                            && a.deferredAdmission == b.deferredAdmission
                            && a.recoveryPending == b.recoveryPending
                            && a.liveRecoveryObligations == b.liveRecoveryObligations
                            && a.recoveryAdmissionPending == b.recoveryAdmissionPending
                            && a.publishRetryReady == b.publishRetryReady
                            && a.pendingDiagTask == b.pendingDiagTask
                            && a.phaseTick == b.phaseTick
                            && a.workerActivitySeq == b.workerActivitySeq
                            && AudioEngine::diagConditionsMet(b);

                        if (!same)
                        {
                            if (++st->baselineRetries > 3) { failStep("baseline_changed"); break; }
                            break;
                        }

                        juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                                 + " phase=baseline ok tick=" + juce::String((juce::int64) b.phaseTick)
                                                 + " activity=" + juce::String((juce::int64) b.workerActivitySeq)
                                                 + " genBefore=" + juce::String((juce::int64) st->genBefore)
                                                 + " publishBefore=" + juce::String((juce::int64) st->pubBefore)
                                                 + " wsProxy=" + juce::String((juce::int64) privProxy));
                        st->phase = 4;
                        break;
                    }
                    case 4: // request the diagnostic rebuild
                    {
                        const auto out = engine.requestRebuildForDiag(st->requestId);
                        if (out != AudioEngine::RebuildDispatchOutcome::Accepted)
                        {
                            failStep("request_rebuild_" + juce::String((int) out));
                            break;
                        }
                        juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                                 + " phase=requestRebuild requestId=" + juce::String((juce::int64) st->requestId));
                        st->deadlineMs = nowMs + 30000;
                        st->phase = 5;
                        break;
                    }
                    case 5: // wait publish with our requestId
                    {
                        const auto snap = engine.getActiveRuntimeSnapshot();
                        if (snap.diagRequestId == st->requestId && snap.generation > st->genBefore)
                        {
                            juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                                     + " phase=published genAfter=" + juce::String((juce::int64) snap.generation)
                                                     + " irLoaded=" + juce::String((int) snap.irLoaded)
                                                     + " irLen=" + juce::String(snap.irLength)
                                                     + " requestId=" + juce::String((juce::int64) snap.diagRequestId)
                                                     + " sealed=" + juce::String((int) snap.diagProjectionSealed));
                            st->phase = 6;
                            break;
                        }
                        if (nowMs >= st->deadlineMs)
                        {
                            failStep("publish_timeout");
                            break;
                        }
                        break;
                    }
                    case 6: // publish increment must be exactly 1
                    {
                        const auto delta = engine.worldRetirementTelemetry().acquireObserved() - st->pubBefore;
                        const bool ok = (delta == 1);
                        juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                                 + " phase=settled publishDelta=" + juce::String((juce::int64) delta)
                                                 + " ok=" + juce::String((int) ok));
                        if (!ok)
                        {
                            failStep(delta == 0 ? "no_publish" : "external_publish");
                            break;
                        }
                        st->settleDeadlineMs = nowMs + (std::uint64_t) st->settleMs;
                        st->phase = 7;
                        break;
                    }
                    case 7: // settle
                    {
                        if (nowMs >= st->settleDeadlineMs)
                        {
                            // ★ step の終わりにウィンドウを閉じる（次の step が開けるようにする）。
                            if (st->windowOpen)
                            {
                                engine.endDiagWindow();
                                st->windowOpen = false;
                            }

                            juce::Logger::writeToLog("[CLI_IR_ALT] step=" + juce::String(st->stepIndex)
                                                     + " phase=result valid=1");
                            ++st->stepIndex;
                            st->deadlineMs = 0; // 次の step の期限を初期化する
                            st->phase = 0;
                        }
                        break;
                    }
                    default: // 8 = done
                        break;
                    }

                    if (st->phase == 8)
                    {
                        if (st->windowOpen)
                        {
                            engine.endDiagWindow();
                            st->windowOpen = false;
                        }
                        juce::Logger::writeToLog("[CLI_IR_ALT] result valid=" + juce::String((int) st->allValid)
                                                 + " steps=" + juce::String(st->stepIndex)
                                                 + " reason=" + st->failReason);
                        st->self.reset();
                        return;
                    }

                    if (st->phase != 0 || st->stepIndex < st->totalSteps)
                        juce::Timer::callAfterDelay(10, *st->self);
                };

                juce::Timer::callAfterDelay(300, *selfFn);
            }
        }
    }

    if (const auto postLoadDitherValue = findValue("--cli-post-load-dither-bit-depth"); !postLoadDitherValue.isEmpty())
    {
        int postLoadBitDepth = 0;
        if (tryParseIntOption(postLoadDitherValue, postLoadBitDepth))
        {
            int delayMs = 200;
            if (const auto postLoadDelayValue = findValue("--cli-post-load-delay-ms"); !postLoadDelayValue.isEmpty())
            {
                int parsedDelay = 0;
                if (tryParseIntOption(postLoadDelayValue, parsedDelay))
                    delayMs = juce::jmax(1, parsedDelay);
            }

            juce::Logger::writeToLog("[CLI] Scheduled post-load dither bit depth: "
                                     + juce::String(postLoadBitDepth)
                                     + " (delayMs=" + juce::String(delayMs) + ")");

            juce::Timer::callAfterDelay(delayMs, [safeThis = juce::Component::SafePointer<MainWindow>(this), postLoadBitDepth]
            {
                if (safeThis == nullptr)
                    return;

                if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire))
                    return;

                safeThis->audioEngine.setDitherBitDepth(postLoadBitDepth);
                juce::Logger::writeToLog("[CLI] Applied post-load dither bit depth: " + juce::String(postLoadBitDepth));
            });
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-post-load-dither-bit-depth: " + postLoadDitherValue);
        }
    }

    {
        int burstCount = 0;
        if (const auto burstCountValue = findValue("--cli-bypass-burst-count"); !burstCountValue.isEmpty())
        {
            int parsedCount = 0;
            if (tryParseIntOption(burstCountValue, parsedCount))
                burstCount = juce::jmax(0, parsedCount);
        }

        if (burstCount > 0)
        {
            int burstIntervalMs = 40;
            if (const auto burstIntervalValue = findValue("--cli-bypass-burst-interval-ms"); !burstIntervalValue.isEmpty())
            {
                int parsedInterval = 0;
                if (tryParseIntOption(burstIntervalValue, parsedInterval))
                    burstIntervalMs = juce::jmax(1, parsedInterval);
            }

            bool burstBypassValue = false;
            if (const auto burstValue = findValue("--cli-bypass-burst-value"); !burstValue.isEmpty())
            {
                int parsedValue = 0;
                if (tryParseIntOption(burstValue, parsedValue))
                    burstBypassValue = (parsedValue != 0);
            }

            // ★ A(D1): 交互バースト（毎回値を反転する）。同一値の連投では publish が
            //   起きない場合があるため、測定用の駆動として明示的に交互切替を選べるようにする。
            bool burstAlternate = false;
            if (const auto alternateValue = findValue("--cli-bypass-burst-alternate"); !alternateValue.isEmpty())
                burstAlternate = (alternateValue != "0");

            juce::Logger::writeToLog("[CLI] Scheduled bypass burst: count="
                                     + juce::String(burstCount)
                                     + " intervalMs=" + juce::String(burstIntervalMs)
                                     + " value=" + juce::String(static_cast<int>(burstBypassValue))
                                     + " alternate=" + juce::String(static_cast<int>(burstAlternate)));

            for (int i = 0; i < burstCount; ++i)
            {
                const int delayMs = i * burstIntervalMs;
                juce::Timer::callAfterDelay(delayMs, [safeThis = juce::Component::SafePointer<MainWindow>(this), burstBypassValue, burstAlternate, i]
                {
                    if (safeThis == nullptr)
                        return;

                    if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire))
                        return;

                    const bool requested = burstAlternate
                                             ? (((i % 2) == 0) ? burstBypassValue : ! burstBypassValue)
                                             : burstBypassValue;
                    safeThis->audioEngine.setConvolverBypassRequested(requested);
                });
            }
        }
    }

    {
        int intentBurstCount = 0;
        if (const auto intentBurstCountValue = findValue("--cli-intent-burst-count"); !intentBurstCountValue.isEmpty())
        {
            int parsedCount = 0;
            if (tryParseIntOption(intentBurstCountValue, parsedCount))
                intentBurstCount = juce::jmax(0, parsedCount);
        }

        if (intentBurstCount > 0)
        {
            int intentBurstIntervalMs = 25;
            if (const auto intentBurstIntervalValue = findValue("--cli-intent-burst-interval-ms"); !intentBurstIntervalValue.isEmpty())
            {
                int parsedInterval = 0;
                if (tryParseIntOption(intentBurstIntervalValue, parsedInterval))
                    intentBurstIntervalMs = juce::jmax(1, parsedInterval);
            }

            juce::Logger::writeToLog("[CLI] Scheduled structural intent burst: count="
                                     + juce::String(intentBurstCount)
                                     + " intervalMs=" + juce::String(intentBurstIntervalMs));

            for (int i = 0; i < intentBurstCount; ++i)
            {
                const int delayMs = i * intentBurstIntervalMs;
                juce::Timer::callAfterDelay(delayMs, [safeThis = juce::Component::SafePointer<MainWindow>(this)]
                {
                    if (safeThis == nullptr)
                        return;

                    if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire))
                        return;

                    safeThis->audioEngine.requestStructuredRebuildIntent(convo::RebuildKind::Structural);
                });
            }
        }
    }

    // ★ v14.47+: --cli-eq-* — EQ band settings for auto gain staging measurement
    {
        int eqBand = 0;
        float eqFreqHz = 1000.0f;
        float eqGainDb = 12.0f;
        float eqQ = 2.0f;
        int eqType = 1; // Peaking

        if (const auto eqBandValue = findValue("--cli-eq-band"); !eqBandValue.isEmpty())
        {
            int parsed = 0;
            if (tryParseIntOption(eqBandValue, parsed))
                eqBand = juce::jlimit(0, 19, parsed);
        }
        if (const auto eqFreqValue = findValue("--cli-eq-freq-hz"); !eqFreqValue.isEmpty())
        {
            float parsed = 0.0f;
            if (tryParseFloatOption(eqFreqValue, parsed) && parsed > 0.0f)
                eqFreqHz = parsed;
        }
        if (const auto eqGainValue = findValue("--cli-eq-gain-db"); !eqGainValue.isEmpty())
        {
            float parsed = 0.0f;
            if (tryParseFloatOption(eqGainValue, parsed))
                eqGainDb = parsed;
        }
        if (const auto eqQValue = findValue("--cli-eq-q"); !eqQValue.isEmpty())
        {
            float parsed = 0.0f;
            if (tryParseFloatOption(eqQValue, parsed) && parsed > 0.0f)
                eqQ = parsed;
        }
        if (const auto eqTypeValue = findValue("--cli-eq-type"); !eqTypeValue.isEmpty())
        {
            int parsed = 1;
            if (tryParseIntOption(eqTypeValue, parsed))
                eqType = juce::jlimit(0, 4, parsed);
        }

        if (!findValue("--cli-eq-gain-db").isEmpty() || !findValue("--cli-eq-freq-hz").isEmpty())
        {
            audioEngine.setEQBandEnabled(eqBand, true);
            audioEngine.setEQBandFrequency(eqBand, eqFreqHz);
            audioEngine.setEQBandGain(eqBand, eqGainDb);
            audioEngine.setEQBandQ(eqBand, eqQ);
            audioEngine.setEQBandType(eqBand, static_cast<EQBandType>(eqType));
            juce::Logger::writeToLog("[CLI_EQ] band=" + juce::String(eqBand)
                + " freq=" + juce::String(eqFreqHz, 1) + "Hz"
                + " gain=" + juce::String(eqGainDb, 1) + "dB"
                + " Q=" + juce::String(eqQ, 2)
                + " type=" + juce::String(eqType));
        }
    }

    // ★ v14.47+: --cli-rebuild — Force structural rebuild after IR load (bypasses telemetry suppression)
    if (hasFlag("--cli-rebuild"))
    {
        // Must fire after deferred IR load (200ms) to include IR in rebuild
        const int rebuildDelayMs = 500;
        juce::Logger::writeToLog("[CLI] Scheduled forced structural rebuild (delayMs="
                                 + juce::String(rebuildDelayMs) + ")");
        juce::Timer::callAfterDelay(rebuildDelayMs, [safeThis = juce::Component::SafePointer<MainWindow>(this)]
        {
            if (safeThis == nullptr)
                return;

            if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire))
                return;

            const bool irLoaded = safeThis->audioEngine.getConvolverProcessor().isIRLoaded();
            const int irLen = safeThis->audioEngine.getConvolverProcessor().getIRLength();
            juce::Logger::writeToLog("[CLI_REBUILD] isIRLoaded=" + juce::String(static_cast<int>(irLoaded))
                                     + " irLen=" + juce::String(irLen));

            safeThis->audioEngine.requestStructuredRebuildIntent(convo::RebuildKind::Structural);
            juce::Logger::writeToLog("[CLI_REBUILD] Forced rebuild intent submitted");
        });
    }

    if (eqPanel != nullptr)
        eqPanel->updateAllControls();
    if (convolverPanel != nullptr)
        convolverPanel->updateIRInfo();

    if (const auto exitValue = findValue("--cli-exit-ms"); !exitValue.isEmpty())
    {
        int exitMs = 0;
        if (tryParseIntOption(exitValue, exitMs) && exitMs > 0)
        {
            // ★ v14.47: Ensure minimum exit time when IR + rebuild are used
            const bool hasIr = !findValue("--cli-ir").isEmpty();
            const bool hasRebuild = !findValue("--cli-rebuild").isEmpty();
            const int minExitMs = (hasIr || hasRebuild) ? 3000 : 1000;
            if (exitMs < minExitMs)
            {
                juce::Logger::writeToLog("[CLI] Adjusted exit-ms from " + juce::String(exitMs)
                    + " to minimum " + juce::String(minExitMs));
                exitMs = minExitMs;
            }

            juce::Logger::writeToLog("[CLI] Auto-exit scheduled in " + juce::String(exitMs) + "ms");
            juce::Timer::callAfterDelay(exitMs, [safeThis = juce::Component::SafePointer<MainWindow>(this)]
            {
                // ★ v14.47: Flush log file before exit (P5 fix)
                juce::Logger::writeToLog("[CLI] Auto-exit flush: shutting down");
                // ★ D127-E (diagnostic only): ここでの logger 切断は teardown 全体を観測不能に
                //   していた（D119-BLOCKER-2 診断障害）。切断は MainApplication::shutdown() 末尾
                //   （LOGGER_DETACH）に移譲し、teardown 全程をファイルに記録する。
                //   production ownership semantics 変更なし（ログ出力タイミングのみ）。

                if (safeThis != nullptr)
                {
                    convo::publishAtomic(safeThis->cliAutomationCallbacksEnabled, false, std::memory_order_release);
                    safeThis->cliAutomationTelemetryLoggingEnabled = false;
                    safeThis->audioEngine.setCliProcessingTelemetryEnabled(false);
                }

                if (auto* app = juce::JUCEApplication::getInstance())
                    app->systemRequestedQuit();
            });
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-exit-ms: " + exitValue);
        }
    }
}

// ★ Live evidence: 実行中 engine の観測値を集める（値の生成・補完はしない）。
//   `--verify-runtime-evidence-live` の shutdown 時（engine 生存中）に 1 回だけ呼ばれる。
bool MainWindow::collectLiveRuntimeObservation(convo::isr::LiveRuntimeObservation& out)
{
    auto* device = audioDeviceManager.getCurrentAudioDevice();

    out.engineInitialized = (device != nullptr);
    out.sampleRateHz = audioEngine.getSampleRate();
    out.blockSamples = (device != nullptr) ? device->getCurrentBufferSizeSamples() : 0;
    out.audioCallbackCount = audioEngine.audioCallbackCount();

    if (device != nullptr)
    {
        out.deviceType = device->getTypeName().toStdString();
        out.deviceName = device->getName().toStdString();
    }

    const auto& convolver = audioEngine.getConvolverProcessor();
    out.irLoaded = convolver.isIRLoaded();
    out.irLength = convolver.getIRLength();

    return out.engineInitialized;
}

std::unique_ptr<juce::AccessibilityHandler> MainWindow::createAccessibilityHandler()
{
   #if JUCE_WINDOWS && JUCE_DEBUG
    return createIgnoredAccessibilityHandler(*this);
   #else
    return juce::DocumentWindow::createAccessibilityHandler();
   #endif
}

//--------------------------------------------------------------
// デストラクタ
//--------------------------------------------------------------
MainWindow::~MainWindow()
{
    convo::publishAtomic(cliAutomationCallbacksEnabled, false, std::memory_order_release);
    juce::Logger::writeToLog("[DIAG] ~MainWindow: ENTER (before setLookAndFeel)");
    orderModeBox.setLookAndFeel (nullptr);
    juce::Logger::writeToLog("[DIAG] ~MainWindow: after setLookAndFeel");

    // 【パッチ4】audioEngine の ChangeListener を最初に解除する
    // 理由: audioEngine はメンバ変数であり、このデストラクタ本体が完了した後に
    //       メンバの逆順破棄が始まる。もし audioEngine が本体完了後~audioEngine()
    //       呼び出し前に sendChangeMessage() を発火した場合、すでに破棄済みの
    //       UIコンポーネント (specAnalyzer / eqPanel 等) にアクセスする
    //       Use-After-Free が発生する。最初に removeChangeListener することで
    //       このウィンドウへの通知を即座に遮断し、安全にシャットダウンできる。
    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 1 removeChangeListener");
    audioEngine.removeChangeListener (this);
    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 2 setAdaptiveAutosaveCallback");
    audioEngine.setAdaptiveAutosaveCallback({});
    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 3 setCliProcessingTelemetryEnabled");
    audioEngine.setCliProcessingTelemetryEnabled(false);
    cliAutomationTelemetryLoggingEnabled = false;

    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 4 setProcessor(nullptr)");
    audioProcessorPlayer.setProcessor (nullptr);
    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 5 stopTimer");
    stopTimer();

    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 6 saveSettings");
    DeviceSettings::saveSettings (audioDeviceManager, audioEngine);

    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 7 removeAudioCallback");
    // 破棄される前にコールバックとしてAudioEngineの登録を解除
    audioDeviceManager.removeAudioCallback (&audioProcessorPlayer);

    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 8 closeAudioDevice");
    // アプリ終了時にASIOドライバを確実に閉じるための安全手順
    audioDeviceManager.closeAudioDevice();
    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 9 audioEngineProcessor.reset");
    audioEngineProcessor.reset();

    juce::Logger::writeToLog("[DIAG] ~MainWindow: step 10 UI reset");
    settingsWindow.reset();
    deviceSettings.reset();
    specAnalyzer.reset();
    eqPanel.reset();
    convolverPanel.reset();
    juce::Logger::writeToLog("[DIAG] ~MainWindow: complete");
    // ★ C-16: FileLogger は MainWindow が所有する。カレントロガーを元に戻してから
    //   メンバ破棄（unique_ptr）で解放する。戻さないと Logger が破棄済み領域を指す。
    if (cliFileLogger != nullptr)
        juce::Logger::setCurrentLogger(previousLogger);
}

//--------------------------------------------------------------
// 閉じるボタン押下時
//--------------------------------------------------------------
void MainWindow::closeButtonPressed()
{
    convo::publishAtomic(cliAutomationCallbacksEnabled, false, std::memory_order_release);
    cliAutomationTelemetryLoggingEnabled = false;
    audioEngine.setCliProcessingTelemetryEnabled(false);
    juce::JUCEApplication::getInstance()->systemRequestedQuit();
}

//--------------------------------------------------------------
// 変更通知コールバック
//--------------------------------------------------------------
void MainWindow::changeListenerCallback (juce::ChangeBroadcaster* source)
{
    if (source == &audioEngine)
    {
        DBG("[DIAG] MainWindow::changeListenerCallback enter (audioEngine)");
        juce::Logger::writeToLog("[DIAG] MainWindow::changeListenerCallback enter (audioEngine)");
        if (eqPanel != nullptr)
            eqPanel->updateAllControls();
        if (convolverPanel != nullptr)
            convolverPanel->updateIRInfo();

        // メインウィンドウ上のコントロールを更新 (プリセットロード時など)
        // UI のモード表示はユーザー意図（Requested）に合わせる。
        // Active は Audio Thread 反映タイミング依存のため、直後に旧値へ戻ることがある。
        const bool eqBypassed = audioEngine.isEqBypassRequested();
        const bool convBypassed = audioEngine.isConvolverBypassRequested();
        int modeId;
        if (eqBypassed && convBypassed)
            modeId = 5; // Bypass
        else if (!eqBypassed && convBypassed)
            modeId = 2; // Peq
        else if (eqBypassed && !convBypassed)
            modeId = 1; // Conv
        else if (!eqBypassed && !convBypassed
              && audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver)
            modeId = 4; // Peq->Conv
        else
            modeId = 3; // Conv->Peq
        orderModeBox.setSelectedId(modeId, juce::dontSendNotification);

        // ソフトクリップとサチュレーション
        softClipButton.setToggleState(audioEngine.isSoftClipEnabled(), juce::dontSendNotification);
        saturationValueLabel.setText(formatSaturationValue(audioEngine.getSaturationAmount()),
                                     juce::dontSendNotification);
        DBG("[DIAG] MainWindow::changeListenerCallback leave (audioEngine)");
        juce::Logger::writeToLog("[DIAG] MainWindow::changeListenerCallback leave (audioEngine)");
    }
}

void MainWindow::labelTextChanged(juce::Label* label)
{
    if (label != &saturationValueLabel)
        return;

    float value = label->getText().retainCharacters("0123456789.").getFloatValue();
    value = juce::jlimit(0.0f, 1.0f, value);
    audioEngine.setSaturationAmount(value);
    saturationValueLabel.setText(formatSaturationValue(audioEngine.getSaturationAmount()),
                                 juce::dontSendNotification);
}

void MainWindow::editorShown(juce::Label* label, juce::TextEditor& editor)
{
    if (label != &saturationValueLabel)
        return;

    editor.setInputRestrictions(5, "0123456789.");
    editor.setText(saturationValueLabel.getText(), false);
}

//--------------------------------------------------------------
// UIコンポーネント作成
//--------------------------------------------------------------
void MainWindow::createUIComponents()
{
    convolverPanel = std::make_unique<ConvolverControlPanel> (audioEngine);
    eqPanel        = std::make_unique<EQControlPanel> (audioEngine);
    specAnalyzer   = std::make_unique<SpectrumAnalyzerComponent> (audioEngine);

    juce::Component::addAndMakeVisible (convolverPanel.get());
    juce::Component::addAndMakeVisible (eqPanel.get());
    juce::Component::addAndMakeVisible (specAnalyzer.get());

    deviceSettings = std::make_unique<DeviceSettings> (audioDeviceManager, audioEngine);

    showDeviceSelectorButton.setButtonText ("Audio Settings");
    showDeviceSelectorButton.setColour (juce::TextButton::buttonColourId,
                                      juce::Colours::darkslategrey.withAlpha (0.8f));
    showDeviceSelectorButton.setColour (juce::TextButton::textColourOffId,
                                      juce::Colours::white);
    showDeviceSelectorButton.onClick = [safeThis = juce::Component::SafePointer<MainWindow>(this)]
    {
        juce::MessageManager::callAsync([safeThis]
        {
            if (safeThis != nullptr)
                safeThis->toggleDeviceSelector();
        });
    };
    juce::Component::addAndMakeVisible (showDeviceSelectorButton);

    // 処理モード選択
    orderModeBox.addItem("Conv", 1);
    orderModeBox.addItem("Peq", 2);
    orderModeBox.addItem("Conv->Peq", 3);
    orderModeBox.addItem("Peq->Conv", 4);
    orderModeBox.addItem("Bypass", 5);
    orderModeBox.setJustificationType(juce::Justification::centred);
    orderModeBox.setTooltip("Processing mode");
    orderModeBox.setLookAndFeel (&orderModeLookAndFeel);
    orderModeBox.onChange = [this] { orderModeBoxChanged(); };
    juce::Component::addAndMakeVisible(orderModeBox);

    // 保存/読み込みボタン
    saveButton.setButtonText ("Save");
    saveButton.onClick = [safeThis = juce::Component::SafePointer<MainWindow>(this)]
    {
        juce::MessageManager::callAsync([safeThis]
        {
            if (safeThis != nullptr)
                safeThis->savePreset();
        });
    };
    juce::Component::addAndMakeVisible (saveButton);

    loadButton.setButtonText ("Load");
    loadButton.onClick = [safeThis = juce::Component::SafePointer<MainWindow>(this)]
    {
        juce::MessageManager::callAsync([safeThis]
        {
            if (safeThis != nullptr)
                safeThis->loadPreset();
        });
    };
    juce::Component::addAndMakeVisible (loadButton);

    // CPU使用率ラベル
    cpuUsageLabel.setText ("CPU: --%", juce::dontSendNotification);
    cpuUsageLabel.setJustificationType (juce::Justification::centredRight);
    cpuUsageLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    juce::Component::addAndMakeVisible (cpuUsageLabel);

    latencyLabel.setText ("Lat: -- ms", juce::dontSendNotification);
    latencyLabel.setJustificationType (juce::Justification::centredRight);
    latencyLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    juce::Component::addAndMakeVisible (latencyLabel);

    // Aboutボタン
    aboutButton.setButtonText ("?");
    aboutButton.setTooltip ("About this application");
    aboutButton.onClick = [safeThis = juce::Component::SafePointer<MainWindow>(this)]
    {
        juce::MessageManager::callAsync([safeThis]
        {
            if (safeThis != nullptr)
                safeThis->showAboutDialog();
        });
    };
    juce::Component::addAndMakeVisible (aboutButton);

    // ソフトクリップボタン
    softClipButton.setButtonText("Soft Clip");
    softClipButton.setToggleState(audioEngine.isSoftClipEnabled(), juce::dontSendNotification);
    softClipButton.setTooltip("Enable/Disable Output Soft Clipper");
    softClipButton.onClick = [this] {
        audioEngine.setSoftClipEnabled(softClipButton.getToggleState());
    };
    juce::Component::addAndMakeVisible(softClipButton);

    // サチュレーションスライダー
    saturationValueLabel.setText(formatSaturationValue(audioEngine.getSaturationAmount()), juce::dontSendNotification);
    saturationValueLabel.setEditable(true);
    saturationValueLabel.setJustificationType(juce::Justification::centred);
    saturationValueLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    saturationValueLabel.setColour(juce::Label::outlineColourId, juce::Colours::grey);
    saturationValueLabel.setTooltip("Saturation Amount (0.0 - 1.0)");
    saturationValueLabel.addListener(this);
    juce::Component::addAndMakeVisible(saturationValueLabel);

    saturationLabel.setText("Sat:", juce::dontSendNotification);
    saturationLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    saturationLabel.setJustificationType(juce::Justification::centredRight);
    juce::Component::addAndMakeVisible(saturationLabel);

    // 初期選択をエンジン状態に同期
    changeListenerCallback(&audioEngine);
}

//--------------------------------------------------------------
// 処理モードドロップダウン
//--------------------------------------------------------------
void MainWindow::orderModeBoxChanged()
{
    const int mode = orderModeBox.getSelectedId();
    switch (mode)
    {
        case 1: /* Conv */
            audioEngine.setConvolverBypassRequested(false);
            audioEngine.setEqBypassRequested(true);
            break;
        case 2: /* Peq */
            audioEngine.setConvolverBypassRequested(true);
            audioEngine.setEqBypassRequested(false);
            break;
        case 3: /* Conv->Peq */
            audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::ConvolverThenEQ);
            audioEngine.setConvolverBypassRequested(false);
            audioEngine.setEqBypassRequested(false);
            break;
        case 4: /* Peq->Conv */
            audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::EQThenConvolver);
            audioEngine.setConvolverBypassRequested(false);
            audioEngine.setEqBypassRequested(false);
            break;
        case 5: /* Bypass */
            audioEngine.setConvolverBypassRequested(true);
            audioEngine.setEqBypassRequested(true);
            break;
        default:
            jassertfalse;
            break;
    }

    if (eqPanel != nullptr)
        eqPanel->updateAllControls();
    if (convolverPanel != nullptr)
        convolverPanel->updateIRInfo();
}

//--------------------------------------------------------------
// 設定読み込み
//--------------------------------------------------------------
void MainWindow::loadSettings()
{
    DeviceSettings::loadSettings (audioDeviceManager, audioEngine);
}

//--------------------------------------------------------------
// デバイス設定画面の表示切り替え
//--------------------------------------------------------------
void MainWindow::toggleDeviceSelector()
{
    juce::Component::SafePointer<MainWindow> safeThis(this);
    juce::MessageManager::callAsync([safeThis]
    {
        if (safeThis != nullptr)
            safeThis->toggleDeviceSelectorImpl();
    });
}

void MainWindow::toggleDeviceSelectorImpl()
{
    if (settingsWindow == nullptr)
    {
        auto background = juce::Desktop::getInstance().getDefaultLookAndFeel()
                              .findColour (juce::ResizableWindow::backgroundColourId);

        auto newSettingsWindow = std::make_unique<SettingsWindow> ("Audio Settings", background, DocumentWindow::allButtons);
        newSettingsWindow->setResizable (true, false);
        newSettingsWindow->setResizeLimits (440, 440, 900, 1040);
        newSettingsWindow->setContentNonOwned (deviceSettings.get(), false);
        newSettingsWindow->centreWithSize (560, 660);

        newSettingsWindow->onClose = [safeThis = juce::Component::SafePointer<MainWindow>(this)]
        {
            if (safeThis != nullptr)
                safeThis->showDeviceSelectorButton.setButtonText ("Audio Settings");
        };

        settingsWindow = std::move (newSettingsWindow);
    }

    if (settingsWindow->isVisible())
    {
        settingsWindow->userTriedToCloseWindow();
    }
    else
    {
        settingsWindow->setVisible (true);

       #if JUCE_WINDOWS && JUCE_DEBUG
        if (settingsWindow != nullptr)
            forceSoftwareRendererIfAvailable(*settingsWindow);
       #endif

        settingsWindow->toFront (true);
        showDeviceSelectorButton.setButtonText ("Hide Settings");
    }
}

//--------------------------------------------------------------
// リサイズ
//--------------------------------------------------------------
void MainWindow::resized()
{
    auto bounds = getLocalBounds();

    auto buttonRow = bounds.removeFromTop (28);

    // 右側: About / Audio Settings
    aboutButton.setBounds (buttonRow.removeFromRight (30).reduced (2, 2));
    showDeviceSelectorButton.setBounds (buttonRow.removeFromRight (130).reduced (2, 2));

    // 状態表示
    cpuUsageLabel.setBounds (buttonRow.removeFromRight (95).reduced (2, 2));
    latencyLabel.setBounds (buttonRow.removeFromRight (170).reduced (2, 2));

    // クリップ制御 (左→右: Soft Clip, Sat, 数値入力)
    saturationValueLabel.setBounds(buttonRow.removeFromRight(58).reduced(2, 2));
    saturationLabel.setBounds(buttonRow.removeFromRight(42).reduced(2, 2));
    softClipButton.setBounds(buttonRow.removeFromRight(90).reduced(2, 2));

    // 左側: 保存/読込 + 処理モード
    orderModeBox.setBounds (buttonRow.removeFromRight (145).reduced (2, 2));
    loadButton.setBounds (buttonRow.removeFromRight (46).reduced (2, 2));
    saveButton.setBounds (buttonRow.removeFromRight (46).reduced (2, 2));

    if (convolverPanel)
        convolverPanel->setBounds (bounds.removeFromTop (320));

    const int eqH = static_cast<int> (bounds.getHeight() * 0.48f);
    if (eqPanel)
        eqPanel->setBounds (bounds.removeFromTop (eqH));

    if (specAnalyzer)
        specAnalyzer->setBounds (bounds);
}

//--------------------------------------------------------------
// タイマーコールバック
//--------------------------------------------------------------
void MainWindow::timerCallback()
{
    double cpu = audioDeviceManager.getCpuUsage() * 100.0;
    cpuUsageLabel.setText ("CPU: " + juce::String (cpu, 1) + "%", juce::dontSendNotification);

    if (cliAutomationTelemetryLoggingEnabled && audioEngine.isCliProcessingTelemetryEnabled())
    {
        const auto cliPerf = audioEngine.consumeCliProcessingTelemetrySnapshot();
        juce::Logger::writeToLog(
            "[CLI_PERF_RAW] callbacks=" + juce::String(static_cast<juce::int64>(cliPerf.callbackCount))
            + " procTimeUsLast=" + juce::String(cliPerf.lastProcessTimeUs, 3)
            + " procTimeUsAvg=" + juce::String(cliPerf.avgProcessTimeUs, 3)
            + " procTimeUsMax=" + juce::String(cliPerf.maxProcessTimeUs, 3)
            + " blockSamples=" + juce::String(cliPerf.lastBlockSamples)
            + " sampleRateHz=" + juce::String(cliPerf.sampleRateHz, 1));

        if (cliAudioSetupRequested && !cliAudioSetupMismatchLogged)
        {
            const bool bufferRequested = (cliRequestedBufferSamples > 0);
            const bool sampleRateRequested = (cliRequestedSampleRateHz > 0.0);
            const bool bufferMismatch = bufferRequested && (cliPerf.lastBlockSamples != cliRequestedBufferSamples);
            const bool sampleRateMismatch = sampleRateRequested
                && (std::abs(cliPerf.sampleRateHz - cliRequestedSampleRateHz) > 1.0);

            if (bufferMismatch || sampleRateMismatch)
            {
                juce::Logger::writeToLog("[CLI_AUDIO_CFG_DRIFT] requestedBufferSamples="
                                         + juce::String(cliRequestedBufferSamples)
                                         + " requestedSampleRateHz=" + juce::String(cliRequestedSampleRateHz, 1)
                                         + " runtimeBlockSamples=" + juce::String(cliPerf.lastBlockSamples)
                                         + " runtimeSampleRateHz=" + juce::String(cliPerf.sampleRateHz, 1)
                                         + " bufferMismatch=" + juce::String(static_cast<int>(bufferMismatch))
                                         + " sampleRateMismatch=" + juce::String(static_cast<int>(sampleRateMismatch)));
                cliAudioSetupMismatchLogged = true;
            }
        }
    }

    if (cliAutomationTelemetryLoggingEnabled)
        return;

    // ISR Bridge Runtime: アクティブ判定は published runtime world の current DSP を基準とし、
    // placeholder スロット（getActiveRuntimeDSP）もフォールバックとして併用する。
    const bool hasActiveDsp = audioEngine.hasPublishedRuntimeDSP()
        || audioEngine.hasActiveRuntimeDSP();
    const auto breakdown = audioEngine.getCurrentLatencyBreakdown();
    const int latencySamples = breakdown.totalLatencyBaseRateSamples;
    const double sr = audioEngine.getSampleRate();
    const bool latencySrValid = hasActiveDsp && (sr > 0.0);
    const int latencyMsX10 = latencySrValid
        ? static_cast<int>(std::lround((static_cast<double>(latencySamples) * 10000.0) / sr))
        : 0;
    const bool latencySnapshotChanged =
        !hasLastLatencyLabelState
        || latencySamples != lastLatencySamples
        || latencyMsX10 != lastLatencyMsX10
        || latencySrValid != lastLatencySrValid;
    if (latencySnapshotChanged)
    {
        juce::String latencyText;
        if (latencySrValid)
        {
            if (latencySamples > 0 && latencyMsX10 < 10)
                latencyText = "Lat: <1ms (" + juce::String(latencySamples) + " smp)";
            else
                latencyText = "Lat: " + juce::String(static_cast<double>(latencyMsX10) / 10.0, 1)
                    + "ms (" + juce::String(latencySamples) + " smp)";
        }
        else
        {
            latencyText = hasActiveDsp
                ? "Lat: -- ms (" + juce::String(latencySamples) + " smp)"
                : "Lat: -- ms";
        }

        latencyLabel.setText(latencyText, juce::dontSendNotification);
        lastLatencySamples = latencySamples;
        lastLatencyMsX10 = latencyMsX10;
        lastLatencySrValid = latencySrValid;
        hasLastLatencyLabelState = true;
    }

}

//--------------------------------------------------------------
// プリセット保存
//--------------------------------------------------------------
void MainWindow::savePreset()
{
    launchFileChooser(true);
}

//--------------------------------------------------------------
// プリセット読み込み
//--------------------------------------------------------------
void MainWindow::loadPreset()
{
    launchFileChooser(false);
}

//--------------------------------------------------------------
// ファイル選択ダイアログ
//--------------------------------------------------------------
void MainWindow::launchFileChooser(bool isSaving)
{
    juce::Component::SafePointer<MainWindow> safeThis(this);
    juce::MessageManager::callAsync([safeThis, isSaving]
    {
        if (safeThis != nullptr)
            safeThis->launchFileChooserImpl(isSaving);
    });
}

void MainWindow::launchFileChooserImpl(bool isSaving)
{
    const juce::String title = isSaving ? "Save Preset" : "Load Preset";
    const juce::String wildcards = isSaving ? "*.xml" : "*.xml;*.txt";
    const int chooserFlags = isSaving ? (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles)
                                      : (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles);

    auto fileChooser = std::make_shared<juce::FileChooser>(title,
                                                           juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                                                           wildcards);

    // 安全性と整合性のためにSafePointerを使用
    juce::Component::SafePointer<MainWindow> safeThis(this);

    fileChooser->launchAsync(chooserFlags, [safeThis, isSaving, fileChooser](const juce::FileChooser& fc)
    {
        if (safeThis == nullptr)
            return;

        auto file = fc.getResult();
        if (file == juce::File())
            return;

        if (isSaving)
        {
            auto state = safeThis->audioEngine.getCurrentState();

            // ★ 保存失敗はユーザーに明示する。原因の詳細はログに残し、
            //   通知文言は原因ごとに増やさず統一する（保存が完了しなかったことを明示）。
            const auto notifySaveFailure = [] (const juce::String& detail)
            {
                juce::Logger::writeToLog ("[MainWindow] preset XML save rejected: " + detail);

                juce::NativeMessageBox::showAsync (
                    juce::MessageBoxOptions()
                        .withIconType (juce::MessageBoxIconType::WarningIcon)
                        .withTitle ("Preset not saved")
                        .withMessage ("The preset was not saved. The existing file was left unchanged.")
                        .withButton ("OK"),
                    nullptr);
            };

            if (auto xml = state.createXml())
            {
                const juce::String xmlText = xml->toString();

                // ★ 上限判定は読込側と同じ関数・同じ単位（UTF-8 バイト数）で行う。
                //   超過時は writeTo を呼ばない（既存ファイルに触れない）。
                if (! convo::safeinput::isXmlTextWithinLimit (xmlText))
                {
                    notifySaveFailure ("size " + juce::String (static_cast<juce::int64> (xmlText.getNumBytesAsUTF8()))
                                       + " bytes exceeds the limit of "
                                       + juce::String (convo::safeinput::kMaxXmlFileBytes) + " bytes");
                }
                else if (! xml->writeTo (file))
                {
                    notifySaveFailure ("write failed: " + file.getFullPathName());
                }
            }
            else
            {
                notifySaveFailure ("state could not be serialized");
            }
        }
        else // 読み込み中
        {
            if (file.existsAsFile())
            {
                if (file.hasFileExtension(".xml"))
                {
                    // ★ B-06: プリセット／セッション XML は解析前にサイズ・深さ・DOCTYPE を検査する。
                    //   拒否理由はログと UI の両方へ出す（silent skip にしない）。
                    juce::String xmlReason;

                    if (auto xml = convo::safeinput::parseXmlFile(file, &xmlReason))
                    {
                        auto state = juce::ValueTree::fromXml(*xml);
                        if (state.isValid())
                            safeThis->audioEngine.requestLoadState(state);
                    }
                    else
                    {
                        const juce::String why = xmlReason.isNotEmpty()
                                                     ? xmlReason
                                                     : ("Preset XML could not be read: " + file.getFileName());

                        juce::Logger::writeToLog("[MainWindow] preset XML rejected: " + why);

                        juce::NativeMessageBox::showAsync(
                            juce::MessageBoxOptions()
                                .withIconType(juce::MessageBoxIconType::WarningIcon)
                                .withTitle("Preset not loaded")
                                .withMessage(why)
                                .withButton("OK"),
                            nullptr);
                    }
                }
                else if (file.hasFileExtension(".txt"))
                {
                    safeThis->audioEngine.requestEqPresetFromText(file);
                }
            }
        }
    });
}

//--------------------------------------------------------------
// バージョン情報ダイアログ
//--------------------------------------------------------------
void MainWindow::showAboutDialog()
{
    juce::Component::SafePointer<MainWindow> safeThis(this);
    juce::MessageManager::callAsync([safeThis]
    {
        if (safeThis != nullptr)
            safeThis->showAboutDialogImpl();
    });
}

void MainWindow::showAboutDialogImpl()
{
    juce::DialogWindow::LaunchOptions options;
    options.content.setOwned (new AboutComponent());
    options.dialogTitle = "About " + juce::String(ProjectInfo::projectName);
    options.dialogBackgroundColour = juce::Colours::darkgrey;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.launchAsync();
}
