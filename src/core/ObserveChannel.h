#pragma once

namespace convo {

// ObserveChannel: 監査主体（観測カテゴリ）単位の固定スロット。
// Audio Thread / Message Thread / Publication / Worker（最大8）× Reserved（2）の合計13チャネル。
// 各チャネルは observeLastSeenGeneration_ / observeLastSeenSequenceId_ 配列のインデックスに対応する。
//
// ■ チャネル設計の原則
// チャネルは「監査主体（観測カテゴリ）」を表し、「スレッド」を表すわけではない。
// 同一スレッド上の異なる論理主体は、必要に応じて別チャネルに分離する。
//
// ■ Publication と Message の分離理由
// 両者は同一スレッド（Message Thread）で動作するが、監査主体としては意味が異なる。
// Publication は publish 発行側の観測を記録し、Message は Timer や制御側の観測を記録する。
// 統合すると publish 発行と Timer 読み取りの generation 更新が混線し、逆行検出の品質が低下する。
//
// ■ SpectrumAnalyzerComponent が ObserveChannel::Message を使用する理由
// SpectrumAnalyzerComponent は JUCE Timer コールバック（Message Thread）で動作し、
// AudioEngine::timerCallback() と同じ監査カテゴリ（Timer/制御側観測）に属する。
// そのため ObserveChannel::Message を使用する。両者が同一スロットを共有しても、
// 当該スロットは「Message カテゴリで最後に観測された generation」を記録するだけであり、
// 逆行検出の品質に影響しない（複数 Reader 間で最も進んだ generation が記録される）。
// もし個別監査が必要な場合は、SpectrumAnalyzer 専用のチャネルを追加する。
//
// ■ Worker1〜Worker7 は将来の Worker 追加用の予約枠
// 固定13チャネルである理由:
// 1. observeLastSeenGeneration_ / observeLastSeenSequenceId_ 配列は std::array で静的に確保される
// 2. チャネル追加は再コンパイルが必要だが、Worker は最大8までが現実的な上限
// 3. Reserved0/Reserved1 は拡張予備として確保
// 現時点で使用するのは Worker0（NoiseShaperLearner）のみ。
//
// ■ チャネル追加基準
// 新しい観測主体を追加する場合、以下の基準で判断する:
// - 同一スレッド上の異なる論理主体で、generation/sequence の逆行パターンが異なる可能性がある
// - 例: 将来 Worker が増えた場合は Worker1〜Worker7 を順次割り当てる
// - チャネル数が13を超える場合は Reserved スロットを解放し、上限を拡張する
enum class ObserveChannel : int {
    Audio       = 0,   // Audio Thread（getNextAudioBlock）
    Message     = 1,   // Message Thread + JUCE Timer
    Publication = 2,   // RuntimePublicationOrchestrator
    Worker0     = 3,   // NoiseShaperLearner（現時点で唯一の Worker）
    Worker1     = 4,   // 将来用予約
    Worker2     = 5,   // 将来用予約
    Worker3     = 6,   // 将来用予約
    Worker4     = 7,   // 将来用予約
    Worker5     = 8,   // 将来用予約
    Worker6     = 9,   // 将来用予約
    Worker7     = 10,  // 将来用予約
    Reserved0   = 11,  // 予約（Worker 上限超過時の拡張用）
    Reserved1   = 12,  // 予約
};

static constexpr int kObserveChannelCount = 13;

} // namespace convo
