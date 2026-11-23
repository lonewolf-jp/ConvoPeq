#include "learner/LearnedStateMigration.h"

#include <cmath>

#include "InputInspection.h"

namespace convo::learnedstate
{

VersionInfo readVersion (const juce::XmlElement& root) noexcept
{
    VersionInfo info;

    if (root.hasAttribute (kVersionAttribute))
    {
        info.version = root.getIntAttribute (kVersionAttribute, 0);
        info.legacyV1 = (info.version == 1);
        return info;
    }

    // version 属性が無い = 旧形式か新形式かを値から決められない。
    info.version = 0;
    info.legacyUnknown = true;
    return info;
}

namespace
{
bool convertCoefficient (double stored, double& parcorOut) noexcept
{
    // 旧形式は c = tanh(parcor)。|c| >= 1 は旧形式ではあり得ず、atanh も取れない。
    if (! std::isfinite (stored) || std::abs (stored) >= 1.0)
        return false;

    const double parcor = std::atanh (stored);
    if (! std::isfinite (parcor))
        return false;

    // 逆変換の検証（浮動小数の丸め誤差のみ許す）。
    if (std::abs (std::tanh (parcor) - stored) > 1.0e-12)
        return false;

    parcorOut = parcor;
    return true;
}
} // namespace

MigrationResult migrateFile (const juce::File& file)
{
    MigrationResult result;

    if (! file.existsAsFile())
    {
        result.message = "learned state file not found: " + file.getFullPathName();
        return result;
    }

    juce::String reason;
    auto xml = convo::safeinput::parseXmlFile (file, &reason);
    if (xml == nullptr || ! xml->hasTagName ("ConvoPeqLearnedState"))
    {
        result.message = "not a ConvoPeq learned state file: "
                         + (reason.isNotEmpty() ? reason : file.getFileName());
        return result;
    }

    const auto info = readVersion (*xml);
    if (info.version > kCurrentVersion)
    {
        result.message = "unknown version " + juce::String (info.version)
                         + " (supported: <= " + juce::String (kCurrentVersion) + "); refusing to modify";
        return result;
    }

    if (info.version >= kCurrentVersion)
    {
        result.ok = true;
        result.changed = false;
        result.message = "already version " + juce::String (kCurrentVersion) + "; nothing to do";
        return result;
    }

    // 係数の変換（BestCoefficients のみ。CmaMean は対象外）。
    double converted[kMaxCoefficients] {};
    int convertedCount = 0;

    if (auto* coeffs = xml->getChildByName ("BestCoefficients"))
    {
        for (int i = 0; i < kMaxCoefficients; ++i)
        {
            const juce::String name = "c" + juce::String (i);
            if (! coeffs->hasAttribute (name))
                break;

            if (! convertCoefficient (coeffs->getDoubleAttribute (name), converted[i]))
            {
                result.message = "coefficient " + name
                                 + " cannot be converted (value out of range or not finite); file left unchanged";
                return result;
            }
            ++convertedCount;
        }
    }

    // 変換結果を書き戻す。
    if (auto* coeffs = xml->getChildByName ("BestCoefficients"))
    {
        for (int i = 0; i < convertedCount; ++i)
            coeffs->setAttribute ("c" + juce::String (i), converted[i]);
    }

    xml->setAttribute (kVersionAttribute, kCurrentVersion);
    xml->setAttribute (kConversionAttribute, kConversionParcor);

    // バックアップ（元データを失わない）。
    const auto backup = file.getSiblingFile (
        file.getFileName() + ".bak-" + juce::String (juce::Time::currentTimeMillis()));
    if (! file.copyFileTo (backup))
    {
        result.message = "failed to create backup: " + backup.getFullPathName();
        return result;
    }

    // 一時ファイルへ書いてから原子的に置換する。
    const auto temp = file.getParentDirectory()
                          .getNonexistentChildFile (file.getFileNameWithoutExtension() + "-migrate", ".tmp", false);
    if (! temp.replaceWithText (xml->toString()))
    {
        temp.deleteFile();
        result.message = "failed to write temporary file: " + temp.getFullPathName();
        return result;
    }

    if (! temp.replaceFileIn (file))
    {
        temp.deleteFile();
        result.message = "failed to replace the original file atomically; original kept, backup at "
                         + backup.getFullPathName();
        return result;
    }

    result.ok = true;
    result.changed = true;
    result.message = "migrated to version " + juce::String (kCurrentVersion)
                     + " (backup: " + backup.getFileName() + ")";
    return result;
}

} // namespace convo::learnedstate
