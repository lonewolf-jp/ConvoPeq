#pragma once

#include "RCUReader.h"
#include "ObserveChannel.h"

namespace convo {

// RuntimeReaderContext: RCUReader と ObserveChannel を運用上束縛する軽量コンテキスト。
// 各クラスは使用時にこのコンテキストを構築し、RuntimePublishWorld へのアクセスに使用する。
//
// ■ 型安全性の限界
// C++ の型システムでは reader と channel の組み合わせの正当性は保証できない。
// 例えば以下の誤った組み合わせがコンパイルを通ってしまう:
//   RuntimeReaderContext{ messageThreadRcuReader, ObserveChannel::Audio }; // 誤りだがコンパイル可能
// このため「各クラスが構築時点で適切な組み合わせを選択する」運用に依存する。
//
// ■ 対策
// - ヘルパー関数（makeAudioReaderContext / makeMessageReaderContext 等）の使用を推奨
// - 各クラスは自身の RuntimeReaderContext をメンバ保持せず、使用時に都度構築する
// - コードレビューで reader と channel の対応を確認する
struct RuntimeReaderContext {
    RCUReader& reader;
    ObserveChannel channel;
};

// ヘルパー構築関数（省略記法）
inline RuntimeReaderContext makeAudioReaderContext(RCUReader& reader) noexcept
{
    return RuntimeReaderContext{ reader, ObserveChannel::Audio };
}

inline RuntimeReaderContext makeMessageReaderContext(RCUReader& reader) noexcept
{
    return RuntimeReaderContext{ reader, ObserveChannel::Message };
}

inline RuntimeReaderContext makePublicationReaderContext(RCUReader& reader) noexcept
{
    return RuntimeReaderContext{ reader, ObserveChannel::Publication };
}

inline RuntimeReaderContext makeWorkerReaderContext(RCUReader& reader, int workerIndex) noexcept
{
    // workerIndex は 0〜7（Worker0〜Worker7）の範囲であること
    const auto channel = static_cast<ObserveChannel>(
        static_cast<int>(ObserveChannel::Worker0) + workerIndex);
    return RuntimeReaderContext{ reader, channel };
}

} // namespace convo
