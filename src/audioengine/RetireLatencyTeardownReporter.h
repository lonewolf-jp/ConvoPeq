//==============================================================================
// RetireLatencyTeardownReporter.h  (W-2, 2026-11-10 Owner 限定承認)
//
// 役割:
//   AudioEngine の「最初の非 static データメンバ」として置かれ、メンバ破棄が
//   すべて終わった時点（デストラクタ）で退役レイテンシ計測の postTeardown 段を
//   独立した JSON ファイルへ書く。
//
// 制約（設計書 w2_post_teardown_design_v2.md）:
//   - ISRDebugRuntime には依存しない（破棄時点で既に存在しないため）。
//   - 他の AudioEngine メンバにも依存しない。レコーダ（グローバル）だけを読む。
//   - 出力は evidence/retire_latency_post_teardown.json（独立ファイル）。
//   - 判定には一切関与しない（evaluateRetireDelay は runtime 段のみを使う）。
//
// 位置づけ:
//   postTeardown = AudioEngine のメンバ破棄完了時点の最終捕捉。
//   postProcess（プロセス終了時の静的ガード）は今回の承認対象外（未実装）。
//==============================================================================
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <system_error>

#include "ISRRetireLatency.h"
#include "../core/EpochDomain.h"        // ★ 計測: reader 保持時間の証跡
#include "../core/ScopedEpochReader.h"  // ★ 計測: guard インスタンス単位の入退場
#include "../core/ReaderMeasureRegistry.h"  // ★ 計測: RCUReader / 音声 reader

namespace convo
{
namespace isr
{

class RetireLatencyTeardownReporter
{
public:
    RetireLatencyTeardownReporter() noexcept = default;
    ~RetireLatencyTeardownReporter() noexcept
    {
        write();
    }

    RetireLatencyTeardownReporter (const RetireLatencyTeardownReporter&) = delete;
    RetireLatencyTeardownReporter& operator= (const RetireLatencyTeardownReporter&) = delete;

    // 明示的に無効化したい場合（テスト等）に使う。既定は有効。
    void setEnabled (bool enabled) noexcept { enabled_ = enabled; }

    // 出力先（既定: current_path()/evidence/retire_latency_post_teardown.json）
    void setOutputPath (std::filesystem::path path) { path_ = std::move (path); }

private:
    struct Values
    {
        std::uint64_t requests = 0;
        std::uint64_t completions = 0;
        std::uint64_t unmatched = 0;
        std::uint64_t cancelled = 0;
        std::uint64_t recordMisses = 0;
        std::uint64_t postShutdownRequests = 0;
        std::uint64_t postShutdownCompletions = 0;
        std::uint64_t overflow = 0;
        std::uint64_t idExhausted = 0;
        std::uint64_t pending = 0;
        std::uint64_t rtRequests = 0;
        bool valid = false;
    };

    static Values fromStage (const RetireLatencyRecorder::StageStats& s) noexcept
    {
        Values v;
        v.valid = s.valid;
        v.requests = s.requests;
        v.completions = s.completions;
        v.unmatched = s.unmatchedCompletions;
        v.recordMisses = s.recordMisses;
        v.postShutdownRequests = s.postShutdownRequests;
        v.postShutdownCompletions = s.postShutdownCompletions;
        v.overflow = s.overflow;
        v.idExhausted = s.idExhausted;
        v.pending = s.pending;
        v.rtRequests = s.rtRequests;
        return v;
    }

    static void writeStage (std::ofstream& f, const char* name, const Values& v, bool last)
    {
        f << "    \"" << name << "\": {\n";
        f << "      \"captured\": " << (v.valid ? "true" : "false") << ",\n";
        f << "      \"cumulative\": true,\n";
        f << "      \"requests\": " << v.requests << ",\n";
        f << "      \"completions\": " << v.completions << ",\n";
        f << "      \"unmatchedCompletions\": " << v.unmatched << ",\n";
        f << "      \"recordMisses\": " << v.recordMisses << ",\n";
        f << "      \"postShutdownRequests\": " << v.postShutdownRequests << ",\n";
        f << "      \"postShutdownCompletions\": " << v.postShutdownCompletions << ",\n";
        f << "      \"overflow\": " << v.overflow << ",\n";
        f << "      \"idExhausted\": " << v.idExhausted << ",\n";
        f << "      \"pending\": " << v.pending << ",\n";
        f << "      \"rtRequests\": " << v.rtRequests << "\n";
        f << "    }" << (last ? "\n" : ",\n");
    }

    static void writeDelta (std::ofstream& f, const char* name,
                            const Values& a, const Values& b, bool last)
    {
        const auto d = [](std::uint64_t x, std::uint64_t y) -> std::uint64_t {
            return (x >= y) ? (x - y) : 0u;
        };
        f << "    \"" << name << "\": {\n";
        f << "      \"requests\": " << d (b.requests, a.requests) << ",\n";
        f << "      \"completions\": " << d (b.completions, a.completions) << ",\n";
        f << "      \"unmatchedCompletions\": " << d (b.unmatched, a.unmatched) << ",\n";
        f << "      \"recordMisses\": " << d (b.recordMisses, a.recordMisses) << ",\n";
        f << "      \"postShutdownRequests\": " << d (b.postShutdownRequests, a.postShutdownRequests) << ",\n";
        f << "      \"postShutdownCompletions\": " << d (b.postShutdownCompletions, a.postShutdownCompletions) << ",\n";
        f << "      \"pending\": " << b.pending << "\n";
        f << "    }" << (last ? "\n" : ",\n");
    }

    void write() const noexcept
    {
        if (! enabled_)
            return;

        std::error_code ec;
        std::filesystem::path out = path_;
        if (out.empty())
            out = std::filesystem::current_path (ec) / "evidence" / "retire_latency_post_teardown.json";

        std::filesystem::create_directories (out.parent_path(), ec);

        std::ofstream f (out, std::ios::binary | std::ios::trunc);
        if (! f.is_open())
            return;

        const auto& rec = RetireLatencyRecorder::instance();

        Values runtime = fromStage (rec.runtimeStage());
        Values finalV = fromStage (rec.finalStage());
        const auto snap = rec.snapshot();               // 現時点の累積（postTeardown）
        Values teardown = fromStage (rec.finalStage()); // 型を借りて詰め替える
        teardown.valid = true;
        teardown.requests = snap.requests;
        teardown.completions = snap.completions;
        teardown.unmatched = snap.unmatchedCompletions;
        teardown.cancelled = snap.cancelled;
        teardown.recordMisses = snap.recordMisses;
        teardown.postShutdownRequests = snap.postShutdownRequests;
        teardown.postShutdownCompletions = snap.postShutdownCompletions;
        teardown.overflow = snap.overflow;
        teardown.idExhausted = snap.idExhausted;
        teardown.pending = (snap.requests > snap.completions) ? (snap.requests - snap.completions) : 0u;
        teardown.rtRequests = snap.rtRequests;

        f << "{\n";
        f << "  \"schema\": \"retire_latency_post_teardown_v1\",\n";
        f << "  \"capturePoint\": \"~AudioEngine member destruction completion (first non-static data member)\",\n";
        f << "  \"postTeardownRole\": \"final capture for AudioEngine-owned resources\",\n";
        f << "  \"postProcess\": { \"implemented\": false, \"reason\": \"out of scope for this approval\" },\n";
        f << "  \"verdictUsage\": \"not used (evaluateRetireDelay reads runtime stage only)\",\n";
        f << "  \"stages\": {\n";
        writeStage (f, "runtime", runtime, false);
        writeStage (f, "final", finalV, false);
        writeStage (f, "postTeardown", teardown, true);
        f << "  },\n";
        f << "  \"delta\": {\n";
        writeDelta (f, "finalMinusRuntime", runtime, finalV, false);
        writeDelta (f, "postTeardownMinusFinal", finalV, teardown, true);
        f << "  }\n";
        f << "}\n";

        // ★ 計測 (2026-11-10): reader 区画の最長保持時間（タグ別）を別ファイルへ
        const std::filesystem::path holds = out.parent_path() / "retire_latency_reader_holds.json";
        if (std::FILE* hf = std::fopen (holds.string().c_str(), "wb"))
        {
            convo::EpochDomain::dumpHoldWitness (hf);
            std::fclose (hf);
        }

        // ★ 計測 (2026-11-11): RCUReader / 音声 reader のレジストリ（postTeardown 段）
        const std::filesystem::path reg = out.parent_path() / "reader_registry_post_teardown.json";
        if (std::FILE* rf = std::fopen (reg.string().c_str(), "wb"))
        {
            convo::ReaderMeasureRegistry::dump (rf, "postTeardown");
            std::fclose (rf);
        }

        // ★ 計測 (2026-11-11 Owner 限定承認): guard インスタンス単位の入退場
        const std::filesystem::path guards = out.parent_path() / "retire_latency_guard_instances.json";
        if (std::FILE* gf = std::fopen (guards.string().c_str(), "wb"))
        {
            convo::ScopedEpochReader::dumpMeasure (gf);
            std::fclose (gf);
        }
    }

    bool enabled_ = true;
    std::filesystem::path path_;
};

} // namespace isr
} // namespace convo
