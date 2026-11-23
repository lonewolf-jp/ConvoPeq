#pragma once

// ★ B-15: 学習係数ファイル（learned_state.xml）の version 管理と明示移行。
//
//   背景:
//     - B-15 の修正前は、公開係数に tanh を二重に掛けていた。修正後は parcor を
//       そのまま保存する。保存形式に version が無かったため、既存ファイルが
//       旧形式（二重 tanh 済み）か新形式（parcor）かを値だけでは識別できない。
//     - 旧形式の値域は |c| < tanh(1) = 0.7616、新形式は |c| < 1.0 であり重なる。
//       したがって version 無しのファイルを自動変換してはいけない。
//
//   方針:
//     - 新規保存は version="2" / conversion="parcor" を付ける。
//     - version 無しは legacyUnknown として読み込み、値は変更せず警告する。
//     - 変換はユーザー／管理者の明示操作（migrateFile）でのみ行う。
//       変換は可逆（parcor = atanh(c)）で、検証とバックアップを伴う。
//     - CmaMean（optimizer の平均）は二重変換の影響を受けないため変換しない。

#include <JuceHeader.h>

namespace convo::learnedstate
{

inline constexpr int kCurrentVersion = 2;
inline constexpr const char* kVersionAttribute = "version";
inline constexpr const char* kConversionAttribute = "conversion";
inline constexpr const char* kConversionParcor = "parcor";
inline constexpr int kMaxCoefficients = 16;   // 学習器の kOrder(9) に対する安全上限

struct VersionInfo
{
    int version = 0;             // 0 = version 属性なし
    bool legacyUnknown = false;  // 属性なし: 旧形式か新形式か不明
    bool legacyV1 = false;       // version="1" が明示されている
};

VersionInfo readVersion (const juce::XmlElement& root) noexcept;

struct MigrationResult
{
    bool ok = false;
    bool changed = false;
    juce::String message;
};

// 明示操作専用。version < 2 のファイルだけを対象に、係数を atanh で戻して
// version=2 として保存する。失敗時は元ファイルを変更しない。
MigrationResult migrateFile (const juce::File& file);

} // namespace convo::learnedstate
