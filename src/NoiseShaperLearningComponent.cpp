#include "NoiseShaperLearningComponent.h"

#include <cstdio>

NoiseShaperLearningComponent::NoiseShaperLearningComponent(AudioEngine& engine)
    : audioEngine(engine), periodicSaver(engine)
{
    modeComboBox.addItem("Shortest", 1);
    modeComboBox.addItem("Short", 2);
    modeComboBox.addItem("Middle", 3);
    modeComboBox.addItem("Long", 4);
    modeComboBox.addItem("Ultra", 5);
    modeComboBox.addItem("Continuous", 6);

    int initialModeId = 2; // Default to Short
    switch (audioEngine.getNoiseShaperLearningMode())
    {
        case NoiseShaperLearner::LearningMode::Shortest:   initialModeId = 1; break;
        case NoiseShaperLearner::LearningMode::Short:      initialModeId = 2; break;
        case NoiseShaperLearner::LearningMode::Middle:     initialModeId = 3; break;
        case NoiseShaperLearner::LearningMode::Long:       initialModeId = 4; break;
        case NoiseShaperLearner::LearningMode::Ultra:      initialModeId = 5; break;
        case NoiseShaperLearner::LearningMode::Continuous: initialModeId = 6; break;
    }
    modeComboBox.setSelectedId(initialModeId, juce::dontSendNotification);

    addAndMakeVisible(modeComboBox);
    addAndMakeVisible(modeLabel);

    modeComboBox.onChange = [this]
    {
        NoiseShaperLearner::LearningMode mode = NoiseShaperLearner::LearningMode::Short;
        switch (modeComboBox.getSelectedId())
        {
            case 1: mode = NoiseShaperLearner::LearningMode::Shortest; break;
            case 2: mode = NoiseShaperLearner::LearningMode::Short; break;
            case 3: mode = NoiseShaperLearner::LearningMode::Middle; break;
            case 4: mode = NoiseShaperLearner::LearningMode::Long; break;
            case 5: mode = NoiseShaperLearner::LearningMode::Ultra; break;
            case 6: mode = NoiseShaperLearner::LearningMode::Continuous; break;
        }
        audioEngine.setNoiseShaperLearningMode(mode);
    };

    startButton.onClick = [this]
    {
        NoiseShaperLearner::LearningMode mode = NoiseShaperLearner::LearningMode::Short;
        switch (modeComboBox.getSelectedId())
        {
            case 1: mode = NoiseShaperLearner::LearningMode::Shortest; break;
            case 2: mode = NoiseShaperLearner::LearningMode::Short; break;
            case 3: mode = NoiseShaperLearner::LearningMode::Middle; break;
            case 4: mode = NoiseShaperLearner::LearningMode::Long; break;
            case 5: mode = NoiseShaperLearner::LearningMode::Ultra; break;
            case 6: mode = NoiseShaperLearner::LearningMode::Continuous; break;
        }
        audioEngine.startNoiseShaperLearning(mode, false);
    };
    addAndMakeVisible(startButton);

    stopButton.onClick = [this]
    {
        audioEngine.stopNoiseShaperLearning();
    };
    addAndMakeVisible(stopButton);

    resumeButton.onClick = [this]
    {
        NoiseShaperLearner::LearningMode mode = NoiseShaperLearner::LearningMode::Short;
        switch (modeComboBox.getSelectedId())
        {
            case 1: mode = NoiseShaperLearner::LearningMode::Shortest; break;
            case 2: mode = NoiseShaperLearner::LearningMode::Short; break;
            case 3: mode = NoiseShaperLearner::LearningMode::Middle; break;
            case 4: mode = NoiseShaperLearner::LearningMode::Long; break;
            case 5: mode = NoiseShaperLearner::LearningMode::Ultra; break;
            case 6: mode = NoiseShaperLearner::LearningMode::Continuous; break;
        }
        audioEngine.startNoiseShaperLearning(mode, true);
    };
    addAndMakeVisible(resumeButton);

    for (auto* label : { &statusLabel, &orderLabel, &sampleRateAndBitDepthLabel, &iterationLabel, &processCountLabel,
                         &segmentCountLabel, &bestScoreLabel, &latestScoreLabel, &messageLabel,
                         &elapsedLabel, &phaseLabel, &cmaesRestartsLabel, &coeffSafetyMarginLabel })
    {
        label->setJustificationType(juce::Justification::centredLeft);
        label->setColour(juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible(*label);
    }

    messageLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.72f));

    addAndMakeVisible(progressGraph);

    cmaesRestartsSlider.setRange(1, 10, 1);
    cmaesRestartsSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    cmaesRestartsSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    cmaesRestartsSlider.onValueChange = [this] {
        auto s = audioEngine.getNoiseShaperLearnerSettings();
        s.cmaesRestarts = static_cast<int>(cmaesRestartsSlider.getValue());
        audioEngine.setNoiseShaperLearnerSettings(s);
    };
    addAndMakeVisible(cmaesRestartsSlider);

    coeffSafetyMarginSlider.setRange(0.3, 0.95, 0.01);
    coeffSafetyMarginSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    coeffSafetyMarginSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    coeffSafetyMarginSlider.onValueChange = [this] {
        auto s = audioEngine.getNoiseShaperLearnerSettings();
        s.coeffSafetyMargin = coeffSafetyMarginSlider.getValue();
        audioEngine.setNoiseShaperLearnerSettings(s);
    };
    addAndMakeVisible(coeffSafetyMarginSlider);

    enableStabilityCheckButton.onClick = [this] {
        auto s = audioEngine.getNoiseShaperLearnerSettings();
        s.enableStabilityCheck = enableStabilityCheckButton.getToggleState();
        audioEngine.setNoiseShaperLearnerSettings(s);
    };
    enableStabilityCheckButton.setToggleState(true, juce::dontSendNotification);
    addAndMakeVisible(enableStabilityCheckButton);

    refreshFromEngine();
    startTimerHz(8);
    periodicSaver.startTimer(300000); // 5 minutes
}

void NoiseShaperLearningComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff20252b));

    g.setColour(juce::Colours::white.withAlpha(0.9f));
    g.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    // タイトル行の高さを32pxに拡張
    g.drawText("Adaptive 9th-order Noise Shaper", getLocalBounds().removeFromTop(32),
               juce::Justification::centredLeft);
}

void NoiseShaperLearningComponent::resized()
{

    auto area = getLocalBounds().reduced(12);
    area.removeFromTop(32); // タイトル行の高さを32pxに

    // ボタン行
    auto controlRow = area.removeFromTop(30);
    startButton.setBounds(controlRow.removeFromLeft(130).reduced(2));
    controlRow.removeFromLeft(6);
    stopButton.setBounds(controlRow.removeFromLeft(130).reduced(2));
    controlRow.removeFromLeft(6);
    resumeButton.setBounds(controlRow.removeFromLeft(130).reduced(2));

    // Learning mode行（新設）
    area.removeFromTop(4); // ボタン行とmode行の間に余白
    auto modeRow = area.removeFromTop(28); // mode行の高さ
    // 左: ラベル, 右: ComboBox
    auto labelWidth = 110;
    modeLabel.setBounds(modeRow.removeFromLeft(labelWidth).reduced(2, 0));
    modeComboBox.setBounds(modeRow.removeFromLeft(120).reduced(2, 0));

    area.removeFromTop(8); // mode行とstatus行の間に余白

    auto topStatusRow = area.removeFromTop(24);
    // Status: Idle | Format: ...
    auto statusRowLeft = topStatusRow.removeFromLeft(topStatusRow.getWidth() / 2);
    statusLabel.setBounds(statusRowLeft);
    sampleRateAndBitDepthLabel.setBounds(topStatusRow); // 右半分にFormat

    // 空行を削除し、Status/Format行の直後にElapsed/Phase行を配置
    auto timeRow = area.removeFromTop(24);
    elapsedLabel.setBounds(timeRow.removeFromLeft(timeRow.getWidth() / 2));
    phaseLabel.setBounds(timeRow);

    area.removeFromTop(4);

    auto generationRow = area.removeFromTop(24);
    iterationLabel.setBounds(generationRow.removeFromLeft(generationRow.getWidth() / 2));
    processCountLabel.setBounds(generationRow);

    area.removeFromTop(4);

    auto metricsRow = area.removeFromTop(24);
    segmentCountLabel.setBounds(metricsRow.removeFromLeft(metricsRow.getWidth() / 2));
    bestScoreLabel.setBounds(metricsRow);

    area.removeFromTop(4);

    latestScoreLabel.setBounds(area.removeFromTop(24));

    area.removeFromTop(8);
    auto restartsRow = area.removeFromTop(24);
    cmaesRestartsLabel.setBounds(restartsRow.removeFromLeft(labelWidth).reduced(2, 0));
    cmaesRestartsSlider.setBounds(restartsRow.reduced(2, 0));

    area.removeFromTop(4);
    auto marginRow = area.removeFromTop(24);
    coeffSafetyMarginLabel.setBounds(marginRow.removeFromLeft(labelWidth).reduced(2, 0));
    coeffSafetyMarginSlider.setBounds(marginRow.reduced(2, 0));

    area.removeFromTop(4);
    auto stabilityRow = area.removeFromTop(24);
    enableStabilityCheckButton.setBounds(stabilityRow.reduced(2, 0));

    area.removeFromTop(4);
    messageLabel.setBounds(area.removeFromTop(22));

    area.removeFromTop(10);
    // グラフの高さを現状の3倍（320→960px）に拡張
    constexpr int kGraphHeight = 960;
    progressGraph.setBounds(area.removeFromTop(kGraphHeight));
}

void NoiseShaperLearningComponent::ProgressGraph::setHistory(const double* values, int count)
{
    historySize = juce::jlimit(0, static_cast<int>(history.size()), count);
    for (int i = 0; i < historySize; ++i)
        history[static_cast<size_t>(i)] = values[i];

    repaint();
}

void NoiseShaperLearningComponent::ProgressGraph::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(6.0f);
    g.setColour(juce::Colour(0xff14181d));
    g.fillRoundedRectangle(bounds, 6.0f);

    g.setColour(juce::Colours::white.withAlpha(0.15f));
    g.drawRoundedRectangle(bounds, 6.0f, 1.0f);

    // ラベルを先に描画してから描画エリアを縮める
    auto labelArea = bounds;
    g.setColour(juce::Colours::white.withAlpha(0.65f));
    g.setFont(juce::FontOptions(12.0f));
    g.drawText("recent score", labelArea.removeFromTop(16.0f).toNearestInt(), juce::Justification::centredLeft);

    // ラベル分を除いた描画エリア
    auto plotArea = bounds.withTrimmedTop(18.0f);

    if (historySize <= 0)
    {
        g.setColour(juce::Colours::white.withAlpha(0.45f));
        g.drawText("No learning history yet", plotArea.toNearestInt(), juce::Justification::centred);
        return;
    }

    // ── 対数スケールで Y 軸を正規化 ──
    // スコアは 1e-17 〜 1e-15 程度の非常に小さな値で、かつ収束後は
    // double 精度で min == max になるため線形スケールでは常に 0 になる。
    // 対数スケールを使うことで収束の進行を可視化する。
    constexpr double kLogEpsilon = 1.0e-38; // 対数計算のゼロ除算防止

    // 最新の半分のデータでmin/maxを計算して、最近の変動にフォーカスする
    const int focusStart = historySize / 2;
    double logMin = std::log10(std::max(history[focusStart], kLogEpsilon));
    double logMax = logMin;
    for (int i = focusStart + 1; i < historySize; ++i)
    {
        const double logVal = std::log10(std::max(history[static_cast<size_t>(i)], kLogEpsilon));
        logMin = std::min(logMin, logVal);
        logMax = std::max(logMax, logVal);
    }

    // 値が 1 点だけ、または全て同じ場合は表示範囲を広げる
    if ((logMax - logMin) < 0.01)
    {
        double center = (logMax + logMin) * 0.5;
        logMax = center + 0.005;
        logMin = center - 0.005;
    }

    const double logRange = logMax - logMin;

    juce::Path linePath;
    const float xStep = (historySize > 1)
        ? (plotArea.getWidth() / static_cast<float>(historySize - 1))
        : 0.0f;

    for (int i = 0; i < historySize; ++i)
    {
        const double logVal = std::log10(std::max(history[static_cast<size_t>(i)], kLogEpsilon));
        double yNorm  = (logVal - logMin) / logRange; // 0=底, 1=頂
        yNorm = juce::jlimit(0.0, 1.0, yNorm);
        const float x = plotArea.getX() + static_cast<float>(i) * xStep;
        const float y = plotArea.getBottom() - (static_cast<float>(yNorm) * plotArea.getHeight());

        if (i == 0)
            linePath.startNewSubPath(x, y);
        else
            linePath.lineTo(x, y);
    }

    // 単点の場合は点として描画する
    if (historySize == 1)
    {
        const double logVal = std::log10(std::max(history[0], kLogEpsilon));
        const double yNorm  = (logVal - logMin) / logRange;
        const float x = plotArea.getCentreX();
        const float y = plotArea.getBottom() - (static_cast<float>(yNorm) * plotArea.getHeight());
        g.setColour(juce::Colour(0xff69b7ff));
        g.fillEllipse(x - 3.0f, y - 3.0f, 6.0f, 6.0f);
    }
    else
    {
        g.setColour(juce::Colour(0xff69b7ff));
        g.strokePath(linePath, juce::PathStrokeType(2.0f));
    }

    // Y 軸の最大値・最小値を対数表示
    g.setColour(juce::Colours::white.withAlpha(0.45f));
    g.setFont(juce::FontOptions(10.0f));

    auto formatLogLabel = [](double logVal) -> juce::String
    {
        // 例: -16.3 → "1e-16"
        const int exponent = static_cast<int>(std::floor(logVal));
        return "1e" + juce::String(exponent);
    };

    g.drawText(formatLogLabel(logMax),
               plotArea.removeFromTop(12.0f).toNearestInt(),
               juce::Justification::centredRight);
    g.drawText(formatLogLabel(logMin),
               plotArea.removeFromBottom(12.0f).toNearestInt(),
               juce::Justification::centredRight);
}

void NoiseShaperLearningComponent::timerCallback()
{
    refreshFromEngine();
}

void NoiseShaperLearningComponent::refreshFromEngine()
{
    const auto& progress = audioEngine.getNoiseShaperLearningProgress();
    const auto status = progress.status.load(std::memory_order_acquire);
    int iteration = progress.iteration.load(std::memory_order_relaxed);
    uint64_t totalGenerations = progress.totalGenerations.load(std::memory_order_relaxed);
    int processCount = progress.processCount.load(std::memory_order_relaxed);
    const int segmentCount = progress.segmentCount.load(std::memory_order_relaxed);
    double bestScore = progress.bestScore.load(std::memory_order_relaxed);
    const double latestScore = progress.latestScore.load(std::memory_order_relaxed);
    double elapsedSec = progress.elapsedPlaybackSeconds.load(std::memory_order_relaxed);
    int currentPhase = progress.currentPhase.load(std::memory_order_relaxed);
    const auto learningMode = static_cast<NoiseShaperLearner::LearningMode>(progress.learningMode.load(std::memory_order_relaxed));

    const double sr = audioEngine.getSampleRate();
    const int bd = audioEngine.getDitherBitDepth();

    bool canStart = true;
    bool canStop = false;
    bool canResume = false;

    NoiseShaperLearner::State savedState;
    const int bankIndex = AudioEngine::getAdaptiveCoeffBankIndex(sr, bd, static_cast<NoiseShaperLearner::LearningMode>(modeComboBox.getSelectedId() - 1));
    if (audioEngine.getAdaptiveNoiseShaperState(bankIndex, savedState) && savedState.iteration > 0)
    {
        canResume = true;
        if (status == NoiseShaperLearner::Status::Idle)
        {
            iteration = savedState.iteration;
            totalGenerations = savedState.totalGenerations;
            processCount = savedState.processCount;
            bestScore = savedState.bestScore;
            elapsedSec = savedState.elapsedPlaybackSeconds;
            currentPhase = savedState.currentPhase;
        }
    }

    statusLabel.setText("Status: " + statusToText(status), juce::dontSendNotification);
    // orderLabel.setText("Filter order: " + juce::String(NoiseShaperLearner::kOrder), juce::dontSendNotification); // 表示だけ消す（内部実装は残す）

    sampleRateAndBitDepthLabel.setText("Format: " + juce::String(sr, 0) + " Hz / " + juce::String(bd) + " bit", juce::dontSendNotification);
    sampleRateAndBitDepthLabel.setFont(phaseLabel.getFont());

    juce::String elapsedStr = juce::String(elapsedSec, 1) + " s";
    elapsedLabel.setText("Elapsed audio: " + elapsedStr, juce::dontSendNotification);

    juce::String phaseStr = "Phase " + juce::String(currentPhase);
    juce::String modeStr = "Short";
    if (learningMode == NoiseShaperLearner::LearningMode::Shortest) modeStr = "Shortest";
    else if (learningMode == NoiseShaperLearner::LearningMode::Middle) modeStr = "Middle";
    else if (learningMode == NoiseShaperLearner::LearningMode::Long) modeStr = "Long";
    else if (learningMode == NoiseShaperLearner::LearningMode::Ultra) modeStr = "Ultra";
    else if (learningMode == NoiseShaperLearner::LearningMode::Continuous) modeStr = "Continuous";
    phaseLabel.setText(phaseStr + " (" + modeStr + ")", juce::dontSendNotification);

    if (learningMode != NoiseShaperLearner::LearningMode::Continuous)
        iterationLabel.setText("Generation: " + juce::String(iteration) + " (Total: " + juce::String(totalGenerations) + ")",
                               juce::dontSendNotification);
    else
        iterationLabel.setText("Generation: " + juce::String(iteration) + " (continuous)",
                               juce::dontSendNotification);
    processCountLabel.setText("Process count: " + juce::String(processCount),
                              juce::dontSendNotification);
    segmentCountLabel.setText("Training segments: " + juce::String(segmentCount),
                              juce::dontSendNotification);
    bestScoreLabel.setText("Best score: " + formatScore(bestScore),
                           juce::dontSendNotification);
    latestScoreLabel.setText("Latest score: " + formatScore(latestScore),
                             juce::dontSendNotification);

    if (!cmaesRestartsSlider.isMouseButtonDown())
        cmaesRestartsSlider.setValue(audioEngine.getNoiseShaperLearnerSettings().cmaesRestarts, juce::dontSendNotification);
    else
        cmaesRestartsSlider.setValue(5, juce::dontSendNotification);

    if (!coeffSafetyMarginSlider.isMouseButtonDown())
        coeffSafetyMarginSlider.setValue(audioEngine.getNoiseShaperLearnerSettings().coeffSafetyMargin, juce::dontSendNotification);
    else
        coeffSafetyMarginSlider.setValue(0.85, juce::dontSendNotification);

    enableStabilityCheckButton.setToggleState(audioEngine.getNoiseShaperLearnerSettings().enableStabilityCheck, juce::dontSendNotification);

    juce::String message = "Press Start learning to begin adaptive optimization.";

    switch (status)
    {
        case NoiseShaperLearner::Status::WaitingForAudio:
            message = "Waiting for active stereo playback to accumulate training segments.";
            canStart = false;
            canStop = true;
            canResume = false;
            break;
        case NoiseShaperLearner::Status::Running:
            message = "Evaluating adaptive coefficients continuously until Stop learning is pressed.";
            canStart = false;
            canStop = true;
            canResume = false;
            break;
        case NoiseShaperLearner::Status::Completed:
            message = "Learning finished. The best adaptive coefficients are already live.";
            break;
        case NoiseShaperLearner::Status::Error:
            message = "Learning stopped due to an error.";
            if (const char* err = audioEngine.getNoiseShaperLearningError(); err && *err)
                message += " (" + juce::String(err) + ")";
            break;
        case NoiseShaperLearner::Status::Idle:
        default:
            if (processCount > 0)
                message = "Learning stopped. Press Start learning to run another pass.";
            break;
    }

    messageLabel.setText(message, juce::dontSendNotification);
    startButton.setEnabled(canStart);
    stopButton.setEnabled(canStop);
    resumeButton.setEnabled(canResume);

    const int points = audioEngine.copyNoiseShaperLearningHistory(historyBuffer.data(),
                                                                  static_cast<int>(historyBuffer.size()));
    progressGraph.setHistory(historyBuffer.data(), points);
}

juce::String NoiseShaperLearningComponent::statusToText(NoiseShaperLearner::Status status)
{
    switch (status)
    {
        case NoiseShaperLearner::Status::Idle: return "Idle";
        case NoiseShaperLearner::Status::WaitingForAudio: return "Waiting for audio";
        case NoiseShaperLearner::Status::Running: return "Running";
        case NoiseShaperLearner::Status::Completed: return "Completed";
        case NoiseShaperLearner::Status::Error: return "Error";
        default: break;
    }

    return "Unknown";
}

juce::String NoiseShaperLearningComponent::formatScore(double score)
{
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "%.6e", score);
    return juce::String(buffer);
}
