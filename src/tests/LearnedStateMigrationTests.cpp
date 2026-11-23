//============================================================================
// LearnedStateMigrationTests.cpp — B-15 の移行テスト
//
// 目的:
//   * version 無し（legacyUnknown）のファイルは自動変換されないこと。
//   * 明示移行で parcor = atanh(c) に戻り、tanh で元に戻ること（可逆）。
//   * 移行は冪等（2 回目は変更なし）で、バックアップが残ること。
//   * 未知 version・不正値・壊れた XML では元ファイルを変更しないこと。
//   * CmaMean は変換されないこと。
//============================================================================
#include "learner/LearnedStateMigration.h"

#include <cmath>
#include <cstdio>

namespace
{
int gFailures = 0;

void check (bool ok, const char* id, const juce::String& detail)
{
    std::printf ("[%s] %s %s\n", ok ? "PASS" : "FAIL", id, detail.toRawUTF8());
    if (! ok)
        ++gFailures;
}

juce::String legacyFileText (const double* stored, int count, int version, bool includeVersion)
{
    juce::XmlElement xml ("ConvoPeqLearnedState");
    if (includeVersion)
        xml.setAttribute ("version", version);
    xml.setAttribute ("bestScore", -1.25);
    xml.setAttribute ("sampleRate", 48000.0);
    xml.setAttribute ("bitDepth", 24);
    xml.setAttribute ("phase", 1);
    xml.setAttribute ("elapsedPlaybackSeconds", 12.5);

    auto* coeffs = xml.createNewChildElement ("BestCoefficients");
    for (int i = 0; i < count; ++i)
        coeffs->setAttribute ("c" + juce::String (i), stored[i]);

    auto* mean = xml.createNewChildElement ("CmaMean");
    for (int i = 0; i < count; ++i)
        mean->setAttribute ("m" + juce::String (i), 0.11 * (i + 1));

    return xml.toString();
}

juce::File makeTempDir (const juce::String& name)
{
    auto dir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                   .getNonexistentChildFile (name, {}, false);
    dir.createDirectory();
    return dir;
}

void testLegacyMigration()
{
    const auto dir = makeTempDir ("b15-legacy");
    const auto file = dir.getChildFile ("learned_state.xml");

    constexpr int kCount = 9;
    double parcor[kCount];
    double stored[kCount];
    for (int i = 0; i < kCount; ++i)
    {
        parcor[i] = -0.55 + 0.13 * i;           // |parcor| < 1
        stored[i] = std::tanh (parcor[i]);      // 旧形式（二重 tanh 済み）
    }

    file.replaceWithText (legacyFileText (stored, kCount, 0, false));

    const auto result = convo::learnedstate::migrateFile (file);
    check (result.ok && result.changed, "M-15-1", "legacy file migrates: " + result.message);

    auto xml = juce::XmlDocument::parse (file);
    check (xml != nullptr && xml->getIntAttribute ("version") == 2
               && xml->getStringAttribute ("conversion") == "parcor",
           "M-15-2", "version=2 / conversion=parcor is written");

    double maxError = 0.0;
    if (auto* coeffs = xml != nullptr ? xml->getChildByName ("BestCoefficients") : nullptr)
    {
        for (int i = 0; i < kCount; ++i)
        {
            const double migrated = coeffs->getDoubleAttribute ("c" + juce::String (i));
            maxError = juce::jmax (maxError, std::abs (migrated - parcor[i]));
        }
    }
    check (maxError <= 1.0e-12, "M-15-3", "converted values equal the original parcor (maxErr="
                                            + juce::String (maxError, 3) + ")");

    const bool backupExists = dir.getNumberOfChildFiles (juce::File::findFiles, "learned_state.xml.bak-*") > 0;
    check (backupExists, "M-15-4", "backup of the original file exists");

    if (auto* mean = xml != nullptr ? xml->getChildByName ("CmaMean") : nullptr)
        check (std::abs (mean->getDoubleAttribute ("m0") - 0.11) < 1.0e-12,
               "M-15-5", "CmaMean is not converted");
    else
        check (false, "M-15-5", "CmaMean missing");

    // 冪等性
    const auto second = convo::learnedstate::migrateFile (file);
    check (second.ok && ! second.changed, "M-15-6", "second migration is a no-op: " + second.message);

    dir.deleteRecursively();
}

void testUnknownVersion()
{
    const auto dir = makeTempDir ("b15-unknown");
    const auto file = dir.getChildFile ("learned_state.xml");
    const juce::String original = "<ConvoPeqLearnedState version=\"3\"><BestCoefficients c0=\"0.2\"/></ConvoPeqLearnedState>";
    file.replaceWithText (original);

    const auto result = convo::learnedstate::migrateFile (file);
    check (! result.ok && result.message.contains ("unknown version"),
           "M-15-7", "unknown version is refused: " + result.message);
    check (file.loadFileAsString() == original, "M-15-8", "original file is untouched");
    dir.deleteRecursively();
}

void testInvalidCoefficient()
{
    const auto dir = makeTempDir ("b15-invalid");
    const auto file = dir.getChildFile ("learned_state.xml");
    double stored[2] { 0.2, 1.5 };   // 2 番目は |c| >= 1 で旧形式ではあり得ない
    const juce::String original = legacyFileText (stored, 2, 0, false);
    file.replaceWithText (original);

    const auto result = convo::learnedstate::migrateFile (file);
    check (! result.ok, "M-15-9", "out-of-range coefficient is refused: " + result.message);
    check (file.loadFileAsString() == original, "M-15-10", "original file is untouched");
    check (dir.getNumberOfChildFiles (juce::File::findFiles, "*.tmp") == 0,
           "M-15-11", "no temporary file is left behind");
    dir.deleteRecursively();
}

void testCorruptFile()
{
    const auto dir = makeTempDir ("b15-corrupt");
    const auto file = dir.getChildFile ("learned_state.xml");
    const juce::String original = "<ConvoPeqLearnedState><BestCoefficients c0=";
    file.replaceWithText (original);

    const auto result = convo::learnedstate::migrateFile (file);
    check (! result.ok, "M-15-12", "corrupt XML is refused");
    check (file.loadFileAsString() == original, "M-15-13", "corrupt file is untouched");
    dir.deleteRecursively();
}

} // namespace

int main()
{
    testLegacyMigration();
    testUnknownVersion();
    testInvalidCoefficient();
    testCorruptFile();

    std::printf ("[SUMMARY] failures=%d\n", gFailures);
    return gFailures == 0 ? 0 : 1;
}
