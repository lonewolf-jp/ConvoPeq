//============================================================================
// MainWindow.cpp ── v0.2 (JUCE 8.0.12対応)
//
// メインウィンドウの実装
// UIコンポーネントの配置とオーディオデバイス管理を行う
//============================================================================
#include "MainWindow.h"
#include <cmath>
#include "audioengine/AtomicAccess.h"

namespace
{
    bool tryParseIntOption(const juce::String& text, int& outValue)
    {
        if (!text.containsOnly("+-0123456789"))
            return false;

        outValue = text.getIntValue();
        return true;
    }

    bool tryParseFloatOption(const juce::String& text, float& outValue)
    {
        if (!text.containsOnly("+-0123456789."))
            return false;

        outValue = text.getFloatValue();
        return text.containsAnyOf("0123456789");
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
    auto blacklistFile = exeDir.getChildFile ("asio_blacklist.txt");

    // デフォルトのブラックリストファイルを作成（存在しない場合）
    // シングルクライアントASIOや不安定なドライバをデフォルトで除外
    if (! blacklistFile.existsAsFile())
    {
        blacklistFile.replaceWithText ("# ASIO Driver Blacklist\n"
                                       "# Add partial driver names to exclude them from the list.\n"
                                       "BRAVO-HD\n"
                                       "ASIO4ALL\n");
    }

    asioBlacklist.loadFromFile (blacklistFile);
    DeviceSettings::applyAsioBlacklist (audioDeviceManager, asioBlacklist);

    // エンジンを先に初期化してデフォルトのサンプルレート(48kHz)を設定
    audioEngine.initialize();

    audioEngineProcessor = std::make_unique<AudioEngineProcessor>(audioEngine);
    audioProcessorPlayer.setDoublePrecisionProcessing(true);
    audioProcessorPlayer.setProcessor(audioEngineProcessor.get());
    audioEngine.addChangeListener (this);
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
        || !findValue("--cli-target-ir-sec").isEmpty()
        || !findValue("--cli-debounce-ms").isEmpty()
        || !findValue("--cli-f1-hz").isEmpty()
        || !findValue("--cli-f2-hz").isEmpty()
        || !findValue("--cli-pre-ring-tau").isEmpty()
        || !findValue("--cli-exit-ms").isEmpty();

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

    if (const auto tauValue = findValue("--cli-pre-ring-tau"); !tauValue.isEmpty())
    {
        float tau = 0.0f;
        if (tryParseFloatOption(tauValue, tau))
        {
            audioEngine.setConvolverMixedPreRingTau(tau);
            juce::Logger::writeToLog("[CLI] Applied mixed pre-ring tau: " + juce::String(tau, 2));
        }
        else
        {
            juce::Logger::writeToLog("[CLI] Invalid --cli-pre-ring-tau: " + tauValue);
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

    if (const auto irValue = findValue("--cli-ir"); !irValue.isEmpty())
    {
        juce::File irFile;
        if (juce::File::isAbsolutePath(irValue))
            irFile = juce::File(irValue);
        else
            irFile = juce::File::getCurrentWorkingDirectory().getChildFile(irValue);

        if (irFile.existsAsFile())
        {
            audioEngine.requestConvolverPreset(irFile);
            juce::Logger::writeToLog("[CLI] Loading IR: " + irFile.getFullPathName());

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

            juce::Logger::writeToLog("[CLI] Scheduled bypass burst: count="
                                     + juce::String(burstCount)
                                     + " intervalMs=" + juce::String(burstIntervalMs)
                                     + " value=" + juce::String(static_cast<int>(burstBypassValue)));

            for (int i = 0; i < burstCount; ++i)
            {
                const int delayMs = i * burstIntervalMs;
                juce::Timer::callAfterDelay(delayMs, [safeThis = juce::Component::SafePointer<MainWindow>(this), burstBypassValue]
                {
                    if (safeThis == nullptr)
                        return;

                    if (!convo::consumeAtomic(safeThis->cliAutomationCallbacksEnabled, std::memory_order_acquire))
                        return;

                    safeThis->audioEngine.setConvolverBypassRequested(burstBypassValue);
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

                    safeThis->audioEngine.requestRebuild(convo::RebuildKind::Structural);
                });
            }
        }
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
            juce::Logger::writeToLog("[CLI] Auto-exit scheduled in " + juce::String(exitMs) + "ms");
            juce::Timer::callAfterDelay(exitMs, [safeThis = juce::Component::SafePointer<MainWindow>(this)]
            {
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
        int modeId = 3; // Conv->Peq
        if (!eqBypassed && convBypassed)
            modeId = 2; // Peq
        else if (eqBypassed && !convBypassed)
            modeId = 1; // Conv
        else if (!eqBypassed && !convBypassed
              && audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver)
            modeId = 4; // Peq->Conv
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
    if (mode == 1)
    {
        audioEngine.setConvolverBypassRequested(false);
        audioEngine.setEqBypassRequested(true);
    }
    else if (mode == 2)
    {
        audioEngine.setConvolverBypassRequested(true);
        audioEngine.setEqBypassRequested(false);
    }
    else if (mode == 3)
    {
        audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::ConvolverThenEQ);
        audioEngine.setConvolverBypassRequested(false);
        audioEngine.setEqBypassRequested(false);
    }
    else if (mode == 4)
    {
        audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::EQThenConvolver);
        audioEngine.setConvolverBypassRequested(false);
        audioEngine.setEqBypassRequested(false);
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

        newSettingsWindow->onClose = [this]
        {
            showDeviceSelectorButton.setButtonText ("Audio Settings");
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
        convolverPanel->setBounds (bounds.removeFromTop (280));

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

    const auto breakdown = audioEngine.getCurrentLatencyBreakdown();
    const int latencySamples = breakdown.totalLatencyBaseRateSamples;
    const double sr = audioEngine.getSampleRate();
    const bool latencySrValid = (sr > 0.0);
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
            latencyText = "Lat: -- ms (" + juce::String(latencySamples) + " smp)";
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
            if (auto xml = state.createXml())
            {
                xml->writeTo(file);
            }
        }
        else // 読み込み中
        {
            if (file.existsAsFile())
            {
                if (file.hasFileExtension(".xml"))
                {
                    if (auto xml = juce::XmlDocument::parse(file))
                    {
                        auto state = juce::ValueTree::fromXml(*xml);
                        if (state.isValid())
                            safeThis->audioEngine.requestLoadState(state);
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
