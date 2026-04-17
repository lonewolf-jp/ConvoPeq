//============================================================================
#pragma once
// ConvolverControlPanel.h  ── v0.2 (JUCE 8.0.12対応)
//
// Convolverコントロールパネル
//
// UI elements:
//   - Load IR ボタン
//   - Dry/Wet Mix スライダー
//   - 位相選択 (As-Is/Mixed/Minimum)
//   - IR波形表示
//   - IR情報表示
//============================================================================

#include "MixedPhaseOptimizationComponent.h"
#include <JuceHeader.h>
#include "AudioEngine.h"

class ConvolverControlPanel  : public juce::Component,
                              private juce::Button::Listener,
                              private juce::Slider::Listener,
                              private juce::Timer,
                              private ConvolverProcessor::Listener   // ← 追加
{
public:
    explicit ConvolverControlPanel(AudioEngine& audioEngine);
    ~ConvolverControlPanel() override;

    // ConvolverProcessor::Listener
    void convolverParamsChanged(ConvolverProcessor* processor) override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    //----------------------------------------------------------
    // UI更新
    //----------------------------------------------------------
    void updateIRInfo();

private:
    // 最適化進捗ウィンドウ（非モーダル）
    juce::Component::SafePointer<convo::MixedPhaseOptimizationWindow> optimizationProgressWindow = nullptr;
    void showOptimizationProgressWindow();   // MixedPhaseOptimizationWindow専用

    AudioEngine& engine;

    // 既存UIコンポーネント
    juce::TextButton loadIRButton{"Load IR..."};
    juce::TextButton irAdvancedButton{"IR Advanced..."};
    juce::TextButton convolverSettingsButton{"Conv Settings..."};
    juce::TextButton optimizationProgressButton{"Optimization Progress..."};
    juce::ComboBox phaseChoiceBox;
    juce::ToggleButton experimentalDirectHeadToggle;

    // リサンプリング位相モード選択
    juce::ComboBox resamplingPhaseBox;
    juce::Label   resamplingPhaseLabel;

    juce::Slider mixSlider;
    juce::Label mixLabel;

    juce::Slider smoothingTimeSlider;
    juce::Label smoothingTimeLabel;

    juce::Slider irLengthSlider;
    juce::Label irLengthLabel;

    juce::Slider rebuildDebounceSlider;
    juce::Label rebuildDebounceLabel;

    juce::Slider mixedF1Slider;
    juce::Label mixedF1Label;

    juce::Slider mixedF2Slider;
    juce::Label mixedF2Label;

    juce::Slider mixedTauSlider;
    juce::Label mixedTauLabel;

    juce::Label irInfoLabel;

    // 出力周波数フィルター UI (① コンボルバー最終段の場合に使用)
    juce::Label  hcfLabel;
    juce::TextButton hcfSharpButton   { "Sharp"   };
    juce::TextButton hcfNaturalButton { "Natural" };
    juce::TextButton hcfSoftButton    { "Soft"    };

    juce::Label  lcfLabel;
    juce::TextButton lcfNaturalButton { "Natural" };
    juce::TextButton lcfSoftButton    { "Soft"    };

    // ボタン状態更新ヘルパー
    void updateFilterModeButtons();

    //----------------------------------------------------------
    // イベントハンドラ
    //----------------------------------------------------------
    void buttonClicked(juce::Button* button) override;
    void sliderValueChanged(juce::Slider* slider) override;
    void mouseDown (const juce::MouseEvent& event) override;
    void timerCallback() override;

    void updateWaveformPath();
    void updateMixedPhaseControlsEnabled();
    void startAsyncIRLoadPreview(const juce::File& irFile);
    void finishAsyncIRLoadPreview(const juce::File& irFile,
                                  const ConvolverProcessor::IRLoadPreview& preview,
                                  int requestId);
    void setIRPreviewInProgress(bool isInProgress);
    void showIRAdvancedWindow();
    void showIRAdvancedWindowImpl();
    void showConvolverSettingsWindow();
    void showConvolverSettingsWindowImpl();
    // Convolver Input Trim スライダーの表示と値をエンジンの現在モードに同期する。
    // モード変更後 (バイパス切替・処理順序変更・プリセットロード) に呼ぶこと。
    void updateTrimSlider();

    void markConvolverParameterDirty();

        //----------------------------------------------------------
        // Convolver Input Trim (EQ→Conv 時のみ有効)
        //----------------------------------------------------------
        juce::Slider convTrimSlider;
        juce::Label  convTrimLabel;
    void applyPendingConvolverParameters();
    bool hasPendingConvolverParameters() const noexcept;
    void showOptimizationProgressWindowImpl();

    juce::Path waveformPath;
    juce::Rectangle<int> waveformArea;

    static constexpr int PARAMETER_RECALC_DEBOUNCE_MS = 3000;
    double lastParameterChangeMs = 0.0;
    bool pendingMixDirty = false;
    bool pendingSmoothingDirty = false;
    bool pendingIrLengthDirty = false;
    bool pendingMixedF1Dirty = false;
    bool pendingMixedF2Dirty = false;
    bool pendingMixedTauDirty = false;
    float pendingMixValue = 1.0f;
    float pendingSmoothingTimeSec = ConvolverProcessor::SMOOTHING_TIME_DEFAULT_SEC;
    float pendingIrLengthSec = ConvolverProcessor::IR_LENGTH_DEFAULT_SEC;
    float pendingMixedF1Hz = ConvolverProcessor::MIXED_F1_DEFAULT_HZ;
    float pendingMixedF2Hz = ConvolverProcessor::MIXED_F2_DEFAULT_HZ;
    float pendingMixedTau = ConvolverProcessor::MIXED_TAU_DEFAULT;
    std::atomic<int> irPreviewRequestId { 0 };
    bool irPreviewInProgress = false;
    juce::Component::SafePointer<juce::DialogWindow> irAdvancedWindow;
    juce::Component::SafePointer<juce::DialogWindow> convolverSettingsWindow;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverControlPanel)
};
