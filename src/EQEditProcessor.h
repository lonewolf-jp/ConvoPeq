//============================================================================
#pragma once
//============================================================================
// EQEditProcessor.h — v2.3 Phase 6
//
// EQProcessor の互換ラッパークラス。
// EQProcessor を public 継承することで、既存 UI コードの一切の変更なしに
// 新 EQ アーキテクチャへの移行を可能にする。
//
// ■ 設計根拠:
//   - コンポジション案では旧リスナー型との互換性問題により
//     SpectrumAnalyzerComponent 等の変更が必要になる。
//   - 継承案では EQProcessor* として扱えるため、既存のリスナー登録・ポインタ比較が
//     全て変更不要で動作する。
//
// ■ デバウンス通知:
//   - 連続した EQ 変更は 50ms ごとにまとめて Snapshot コマンドとして発行する。
//
// ■ スレッド安全性:
//   - 全てのセッターは Message Thread からのみ呼ぶこと。
//   - timerCallback は JUCE の Message Thread で実行される。
//============================================================================

#include "EQProcessor.h"

class AudioEngine;

class EQEditProcessor final : public EQProcessor,
                              private juce::Timer
{
public:
    explicit EQEditProcessor(AudioEngine& engine);
    ~EQEditProcessor() override = default;

    EQEditProcessor(const EQEditProcessor&) = delete;
    EQEditProcessor& operator=(const EQEditProcessor&) = delete;

    // -----------------------------------------------------------------
    // バンドパラメータ setter（シャドウ：デバウンス追加）
    // -----------------------------------------------------------------
    void setBandFrequency  (int band, float freq);
    void setBandGain       (int band, float gainDb);
    void setBandQ          (int band, float q);
    void setBandEnabled    (int band, bool enabled);
    void setBandType       (int band, EQBandType type);
    void setBandChannelMode(int band, EQChannelMode mode);

    // -----------------------------------------------------------------
    // グローバルパラメータ setter（シャドウ：デバウンス追加）
    // -----------------------------------------------------------------
    void setTotalGain          (float gainDb);
    void setAGCEnabled         (bool enabled);
    void setNonlinearSaturation(float value) noexcept;
    void setFilterStructure    (FilterStructure mode) noexcept;

    // -----------------------------------------------------------------
    // バッチ操作（シャドウ：デバウンス追加）
    // -----------------------------------------------------------------
    void resetToDefaults();
    bool loadFromTextFile(const juce::File& file);
    void setState(const juce::ValueTree& state);

    // -----------------------------------------------------------------
    // バイパス（Snapshot に含まれるためデバウンスを経由する）
    // -----------------------------------------------------------------
    void setBypass(bool shouldBypass);

private:
    // デバウンスタイマー起動
    void scheduleDebounce();

    // juce::Timer コールバック
    void timerCallback() override;

    AudioEngine& audioEngine;

    std::atomic<bool> pendingSnapshot { false };

    static constexpr int kDebounceMs = 50;
};
