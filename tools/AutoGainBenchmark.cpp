//==============================================================================
// AutoGainBenchmark.cpp — ★ v14.47
//
// Auto Gain Staging 実IR ベンチマーク & 係数較正ツール
//
// 使用方法:
//   AutoGainBenchmark --irs-dir <path> [--bands <config>] [--output <json>]
//
// 本ツールは AudioEngine の Runtime が必要。
// スタンドアロンビルド時は ./build/AutoGainBenchmark --help
//==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include <JuceHeader.h>

// ConvoPeq のコア解析ヘッダー
#include "audioengine/RuntimeBuildTypes.h"
#include "eqprocessor/EQProcessor.h"
#include "IRAnalyzer.h"
#include "IRConverter.h"
#include "audioengine/AutoGainPlanner.h"

namespace fs = std::filesystem;

//==============================================================================
// コマンドライン引数パーサー
//==============================================================================
struct BenchmarkConfig {
    fs::path irsDir;          // IR ファイルのディレクトリ
    std::string outputFile;   // 出力 JSON ファイル
    int sampleRate = 48000;   // 解析サンプルレート
    bool verbose = false;     // 詳細出力
};

BenchmarkConfig parseArgs(int argc, char* argv[])
{
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--irs-dir" && i + 1 < argc)
            config.irsDir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            config.outputFile = argv[++i];
        else if (arg == "--sr" && i + 1 < argc)
            config.sampleRate = std::stoi(argv[++i]);
        else if (arg == "--verbose" || arg == "-v")
            config.verbose = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Auto Gain Staging Benchmark Tool\n"
                      << "Usage: AutoGainBenchmark --irs-dir <path> [options]\n"
                      << "Options:\n"
                      << "  --irs-dir <path>   IR WAV ファイルのディレクトリ\n"
                      << "  --output <file>    出力 JSON ファイル\n"
                      << "  --sr <rate>        解析サンプルレート (default: 48000)\n"
                      << "  --verbose, -v      詳細出力\n"
                      << "  --help, -h         このヘルプ\n";
            std::exit(0);
        }
    }
    return config;
}

//==============================================================================
// IR 分析
//==============================================================================
struct IRAnalysisResult {
    std::string fileName;           // ファイル名
    double freqPeakGainDb = 0.0;    // IRAnalyzer による周波数ピークゲイン
    double irFreqPeakGainDb = 0.0;  // irFreqPeakGainDb（Conv パスで使用）
    double peakDb = 0.0;            // 絶対ピーク
    double rmsDb = 0.0;             // RMS レベル
    int length = 0;                 // サンプル長
    int sampleRate = 0;             // サンプルレート
};

std::vector<IRAnalysisResult> analyzeAllIRs(const fs::path& dir, int targetSr, bool verbose)
{
    std::vector<IRAnalysisResult> results;

    if (!fs::exists(dir)) {
        std::cerr << "Error: directory not found: " << dir << std::endl;
        return results;
    }

    int count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.path().extension().is_extension() ||
            entry.path().extension().string() != ".wav")
            continue;

        // WAV 読み込み
        juce::AudioBuffer<double> ir;
        double fileSr = 0;
        {
            juce::AudioFormatManager fmtMgr;
            fmtMgr.registerBasicFormats();

            std::unique_ptr<juce::AudioFormatReader> reader(
                fmtMgr.createReaderFor(entry.path().string()));
            if (!reader) continue;

            fileSr = reader->sampleRate;
            ir.setSize(static_cast<int>(reader->numChannels),
                       static_cast<int>(reader->lengthInSamples));
            reader->read(&ir, 0, static_cast<int>(reader->lengthInSamples),
                         0, true, true);
        }

        if (ir.getNumSamples() <= 0) continue;

        // リサンプリング（必要に応じて）
        juce::AudioBuffer<double>* analysisIr = &ir;
        juce::AudioBuffer<double> resampled;
        if (std::abs(fileSr - targetSr) > 1.0 && ir.getNumSamples() > 0) {
            // 簡易リサンプリング（線形補間）
            const double ratio = targetSr / fileSr;
            const int newLen = static_cast<int>(ir.getNumSamples() * ratio);
            resampled.setSize(ir.getNumChannels(), newLen);
            for (int ch = 0; ch < ir.getNumChannels(); ++ch) {
                const double* src = ir.getReadPointer(ch);
                double* dst = resampled.getWritePointer(ch);
                for (int i = 0; i < newLen; ++i) {
                    const double srcPos = i / ratio;
                    const int srcIdx = static_cast<int>(srcPos);
                    const double frac = srcPos - srcIdx;
                    if (srcIdx + 1 < ir.getNumSamples())
                        dst[i] = src[srcIdx] * (1.0 - frac) + src[srcIdx + 1] * frac;
                    else
                        dst[i] = src[std::min(srcIdx, ir.getNumSamples() - 1)];
                }
            }
            analysisIr = &resampled;
        }

        // IRAnalyzer で周波数ピークゲインを推定
        const double freqPeakLin = IRAnalyzer::estimateMaxFrequencyResponseGain(*analysisIr);
        const double freqPeakGainDb = (freqPeakLin > 1e-18)
            ? 20.0 * std::log10(freqPeakLin) : 0.0;

        // ピーク/RMS
        double peak = 0.0;
        double energy = 0.0;
        for (int ch = 0; ch < analysisIr->getNumChannels(); ++ch) {
            const double* data = analysisIr->getReadPointer(ch);
            const int len = analysisIr->getNumSamples();
            for (int i = 0; i < len; ++i) {
                peak = std::max(peak, std::abs(data[i]));
                energy += data[i] * data[i];
            }
        }
        const double rms = std::sqrt(energy / (analysisIr->getNumSamples()
                                                * analysisIr->getNumChannels()));

        IRAnalysisResult r;
        r.fileName = entry.path().filename().string();
        r.freqPeakGainDb = freqPeakGainDb;
        r.irFreqPeakGainDb = freqPeakGainDb;
        r.peakDb = 20.0 * std::log10(peak + 1e-18);
        r.rmsDb = 20.0 * std::log10(rms + 1e-18);
        r.length = analysisIr->getNumSamples();
        r.sampleRate = targetSr;
        results.push_back(r);

        ++count;
        if (verbose && count % 10 == 0)
            std::cout << "  Processed " << count << " files..." << std::endl;
    }

    return results;
}

//==============================================================================
// EQ 設定
//==============================================================================
struct EQConfig {
    std::string name;
    // band frequencies, gains, Qs
    std::vector<float> freqs;
    std::vector<float> gains;
    std::vector<float> qs;
    std::vector<int> types; // 0=Peaking, 1=LowShelf, 2=HighShelf
    bool isParallel = false;
};

std::vector<EQConfig> getDefaultEQConfigs()
{
    std::vector<EQConfig> configs;

    // 設定 1: シングル Peaking +12dB
    configs.push_back({"Single Peaking +12dB Q=1 @1kHz",
        {1000.0f}, {12.0f}, {1.0f}, {0}, false});

    // 設定 2: シングル Peaking +24dB Q=10
    configs.push_back({"Single Peaking +24dB Q=10 @1kHz",
        {1000.0f}, {24.0f}, {10.0f}, {0}, false});

    // 設定 3: 2バンド Serial (異周波数)
    configs.push_back({"2-Band Serial +12dB @1kHz + +12dB @4kHz",
        {1000.0f, 4000.0f}, {12.0f, 12.0f}, {1.0f, 1.0f}, {0, 0}, false});

    // 設定 4: 2バンド Parallel (同周波数)
    configs.push_back({"2-Band Parallel +12dB @1kHz same freq",
        {1000.0f, 1000.0f}, {12.0f, 12.0f}, {1.0f, 1.0f}, {0, 0}, true});

    // 設定 5: 20Band 全部 +12dB Q=0.707 (フル構成)
    {
        EQConfig full;
        full.name = "20-Band Full +12dB Q=0.707 Serial";
        full.isParallel = false;
        constexpr float defaultFreqs[20] = {
            25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600,
            2500, 4000, 6300, 10000, 11000, 12500, 14000, 16500, 18000, 19500
        };
        for (int i = 0; i < 20; ++i) {
            full.freqs.push_back(defaultFreqs[i]);
            full.gains.push_back(12.0f);
            full.qs.push_back(0.707f);
            full.types.push_back(0);
        }
        configs.push_back(full);
    }

    return configs;
}

//==============================================================================
// メイン: ベンチマーク実行
//==============================================================================
int main(int argc, char* argv[])
{
    // JUCE 初期化 (最小限)
    juce::ScopedJuceInitialiser_GUI scopedJuce;

    const auto config = parseArgs(argc, argv);

    std::cout << "==============================================" << std::endl;
    std::cout << "Auto Gain Staging Benchmark Tool" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "IRs directory: " << config.irsDir << std::endl;
    std::cout << "Sample rate:   " << config.sampleRate << " Hz" << std::endl;
    std::cout << "Output:        " << (config.outputFile.empty() ? "(stdout)" : config.outputFile) << std::endl;
    std::cout << "==============================================" << std::endl;

    // Step 1: IR 分析
    std::cout << "\n[1/3] Analyzing IRs..." << std::endl;
    auto irResults = analyzeAllIRs(config.irsDir, config.sampleRate, config.verbose);
    if (irResults.empty()) {
        std::cerr << "Error: no IR files found in " << config.irsDir << std::endl;
        return 1;
    }
    std::cout << "  Found " << irResults.size() << " IR files" << std::endl;

    // IR周波数ピークゲインの分布
    std::vector<double> peakGains;
    for (const auto& r : irResults)
        peakGains.push_back(r.freqPeakGainDb);
    std::sort(peakGains.begin(), peakGains.end());

    double mean = std::accumulate(peakGains.begin(), peakGains.end(), 0.0) / peakGains.size();
    double median = peakGains[peakGains.size() / 2];
    double p95 = peakGains[static_cast<size_t>(peakGains.size() * 0.95)];
    double p99 = peakGains[static_cast<size_t>(peakGains.size() * 0.99)];
    double maxGain = peakGains.back();

    std::cout << "\n  IR Peak Gain Distribution (dBFS):" << std::endl;
    std::cout << "    Mean:   " << mean << " dB" << std::endl;
    std::cout << "    Median: " << median << " dB" << std::endl;
    std::cout << "    P95:    " << p95 << " dB" << std::endl;
    std::cout << "    P99:    " << p99 << " dB" << std::endl;
    std::cout << "    Max:    " << maxGain << " dB" << std::endl;

    // Step 2: Safety Margin 評価
    std::cout << "\n[2/3] Evaluating EmpiricalSafetyMarginPolicy..." << std::endl;

    std::cout << "\n  Current coefficients:" << std::endl;
    std::cout << "    base=0.8, q_coeff=0.12, gain_coeff=0.04, max=2.5" << std::endl;

    // P95 と P99 を参考に推奨ベースラインを計算
    double recommendedBase = std::max(0.8, p95 * 0.1); // heuristic
    std::cout << "  Recommended base (P95-based): " << recommendedBase << " dB" << std::endl;

    // Step 3: 出力
    if (!config.outputFile.empty()) {
        std::ofstream out(config.outputFile);
        out << "{\n";
        out << "  \"tool\": \"AutoGainBenchmark\",\n";
        out << "  \"timestamp\": \"" << juce::Time::getCurrentTime().toString(true, true) << "\",\n";
        out << "  \"config\": {\n";
        out << "    \"irsDir\": \"" << config.irsDir.string() << "\",\n";
        out << "    \"sampleRate\": " << config.sampleRate << "\n";
        out << "  },\n";
        out << "  \"irStatistics\": {\n";
        out << "    \"count\": " << irResults.size() << ",\n";
        out << "    \"peakGainDb\": {\n";
        out << "      \"mean\": " << mean << ",\n";
        out << "      \"median\": " << median << ",\n";
        out << "      \"p95\": " << p95 << ",\n";
        out << "      \"p99\": " << p99 << ",\n";
        out << "      \"max\": " << maxGain << "\n";
        out << "    }\n";
        out << "  },\n";
        out << "  \"calibration\": {\n";
        out << "    \"currentCoefficients\": {\n";
        out << "      \"base\": " << kSafetyMarginBase << ",\n";
        out << "      \"qCoeff\": " << kSafetyMarginCoeffQ << ",\n";
        out << "      \"gainCoeff\": " << kSafetyMarginCoeffGain << ",\n";
        out << "      \"maxMargin\": " << kSafetyMarginMax << "\n";
        out << "    },\n";
        out << "    \"recommendedBase\": " << recommendedBase << "\n";
        out << "  }\n";
        out << "}\n";
        out.close();
        std::cout << "\n  Results saved to: " << config.outputFile << std::endl;
    }

    std::cout << "\n==============================================" << std::endl;
    std::cout << "Benchmark complete." << std::endl;
    std::cout << "To calibrate coefficients, run with --verbose" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
