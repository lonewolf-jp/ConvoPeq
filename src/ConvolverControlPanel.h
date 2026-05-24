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
    std::unique_ptr<convo::MixedPhaseOptimizationWindow> optimizationProgressWindow;
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

        //----------------------------------------------------------
        // Convolver Input Trim (EQ→Conv 時のみ有効)
        //----------------------------------------------------------
        juce::Slider convTrimSlider;
        juce::Label  convTrimLabel;
    void showOptimizationProgressWindowImpl();

    juce::Path waveformPath;
    juce::Rectangle<int> waveformArea;

    // UI反映待ちの一時値（Timer debounce廃止後も、非同期IR解析直後の表示同期に利用）
    bool pendingMixDirty = false;
    double pendingMixValue = 0.0;
    bool pendingSmoothingDirty = false;
    double pendingSmoothingTimeSec = 0.0;
    bool pendingIrLengthDirty = false;
    double pendingIrLengthSec = 0.0;
    bool pendingMixedF1Dirty = false;
    double pendingMixedF1Hz = 0.0;
    bool pendingMixedF2Dirty = false;
    double pendingMixedF2Hz = 0.0;
    bool pendingMixedTauDirty = false;
    double pendingMixedTau = 0.0;

    std::atomic<int> irPreviewRequestId { 0 };
    bool irPreviewInProgress = false;
    static juce::ThreadPool irPreviewThreadPool;
    juce::Component::SafePointer<juce::DialogWindow> irAdvancedWindow;
    juce::Component::SafePointer<juce::DialogWindow> convolverSettingsWindow;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverControlPanel)
};
