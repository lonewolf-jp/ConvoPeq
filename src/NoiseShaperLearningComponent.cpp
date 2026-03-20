#include "NoiseShaperLearningComponent.h"

#include <cstdio>

NoiseShaperLearningComponent::NoiseShaperLearningComponent(AudioEngine& engine)
    : audioEngine(engine)
{
    startButton.onClick = [this]
    {
        audioEngine.startNoiseShaperLearning();
    };
    addAndMakeVisible(startButton);

    stopButton.onClick = [this]
    {
        audioEngine.stopNoiseShaperLearning();
    };
    addAndMakeVisible(stopButton);

    for (auto* label : { &statusLabel, &orderLabel, &iterationLabel, &processCountLabel,
                         &segmentCountLabel, &bestScoreLabel, &latestScoreLabel, &messageLabel })
    {
        label->setJustificationType(juce::Justification::centredLeft);
        label->setColour(juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible(*label);
    }

    messageLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.72f));

    addAndMakeVisible(progressGraph);

    refreshFromEngine();
    startTimerHz(8);
}

void NoiseShaperLearningComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff20252b));

    g.setColour(juce::Colours::white.withAlpha(0.9f));
    g.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    g.drawText("Adaptive 9th-order Noise Shaper", getLocalBounds().removeFromTop(28),
               juce::Justification::centredLeft);
}

void NoiseShaperLearningComponent::resized()
{
    auto area = getLocalBounds().reduced(12);
    area.removeFromTop(28);

    auto controlRow = area.removeFromTop(30);
    startButton.setBounds(controlRow.removeFromLeft(130).reduced(2));
    controlRow.removeFromLeft(6);
    stopButton.setBounds(controlRow.removeFromLeft(130).reduced(2));

    area.removeFromTop(8);

    auto topStatusRow = area.removeFromTop(24);
    statusLabel.setBounds(topStatusRow.removeFromLeft(topStatusRow.getWidth() / 2));
    orderLabel.setBounds(topStatusRow);

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

    area.removeFromTop(4);
    messageLabel.setBounds(area.removeFromTop(22));

    area.removeFromTop(10);
    progressGraph.setBounds(area);
}

void NoiseShaperLearningComponent::ProgressGraph::setHistory(const float* values, int count)
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
    // float32 精度で min == max になるため線形スケールでは常に 0 になる。
    // 対数スケールを使うことで収束の進行を可視化する。
    constexpr float kLogEpsilon = 1.0e-38f; // 対数計算のゼロ除算防止

    float logMin = std::log10(std::max(history[0], kLogEpsilon));
    float logMax = logMin;
    for (int i = 1; i < historySize; ++i)
    {
        const float logVal = std::log10(std::max(history[static_cast<size_t>(i)], kLogEpsilon));
        logMin = std::min(logMin, logVal);
        logMax = std::max(logMax, logVal);
    }

    // 値が 1 点だけ、または全て同じ場合は表示範囲を ±0.5 decade 広げる
    if ((logMax - logMin) < 0.05f)
    {
        logMax += 0.5f;
        logMin -= 0.5f;
    }

    const float logRange = logMax - logMin;

    juce::Path linePath;
    const float xStep = (historySize > 1)
        ? (plotArea.getWidth() / static_cast<float>(historySize - 1))
        : 0.0f;

    for (int i = 0; i < historySize; ++i)
    {
        const float logVal = std::log10(std::max(history[static_cast<size_t>(i)], kLogEpsilon));
        const float yNorm  = (logVal - logMin) / logRange; // 0=底, 1=頂
        const float x = plotArea.getX() + static_cast<float>(i) * xStep;
        const float y = plotArea.getBottom() - (yNorm * plotArea.getHeight());

        if (i == 0)
            linePath.startNewSubPath(x, y);
        else
            linePath.lineTo(x, y);
    }

    // 単点の場合は点として描画する
    if (historySize == 1)
    {
        const float logVal = std::log10(std::max(history[0], kLogEpsilon));
        const float yNorm  = (logVal - logMin) / logRange;
        const float x = plotArea.getCentreX();
        const float y = plotArea.getBottom() - (yNorm * plotArea.getHeight());
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

    auto formatLogLabel = [](float logVal) -> juce::String
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
    const int iteration = progress.iteration.load(std::memory_order_relaxed);
    const int maxIteration = progress.maxIteration.load(std::memory_order_relaxed);
    const int processCount = progress.processCount.load(std::memory_order_relaxed);
    const int segmentCount = progress.segmentCount.load(std::memory_order_relaxed);
    const float bestScore = progress.bestScore.load(std::memory_order_relaxed);
    const float latestScore = progress.latestScore.load(std::memory_order_relaxed);

    statusLabel.setText("Status: " + statusToText(status), juce::dontSendNotification);
    orderLabel.setText("Filter order: " + juce::String(NoiseShaperLearner::kOrder), juce::dontSendNotification);
    if (maxIteration > 0)
        iterationLabel.setText("Generation: " + juce::String(iteration) + "/" + juce::String(maxIteration),
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

    juce::String message = "Press Start learning to begin adaptive optimization.";
    bool canStart = true;
    bool canStop = false;

    switch (status)
    {
        case NoiseShaperLearner::Status::WaitingForAudio:
            message = "Waiting for active stereo playback to accumulate training segments.";
            canStart = false;
            canStop = true;
            break;
        case NoiseShaperLearner::Status::Running:
            message = "Evaluating adaptive coefficients continuously until Stop learning is pressed.";
            canStart = false;
            canStop = true;
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

juce::String NoiseShaperLearningComponent::formatScore(float score)
{
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "%.6e", static_cast<double>(score));
    return juce::String(buffer);
}
