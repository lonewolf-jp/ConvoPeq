# -*- coding: utf-8 -*-
import subprocess

# Restore v14.14 from git first
subprocess.run(["git", "-C", "C:\\VSC_Project\\ConvoPeq", "checkout", "HEAD", "--", "doc/work77/AutoGainStagingRenewal.md"], check=True)

# Read the restored v14.14 file
with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Verify
assert 'v14.14' in content, 'Wrong version restored'
print(f'Read {len(content)} chars from v14.14')

# Apply all transformations v14.14 -> v14.22

# 1. Header
content = content.replace(
    '# ConvoPeq Auto Gain Staging 改修計画書 v14.14',
    '# ConvoPeq Auto Gain Staging 改修計画書 v14.22'
)
content = content.replace(
    '本計画はv14.13レビュー指摘4件（verifyとAnalysisPartの不一致・PlannerInput DTO・BoundMethod保持理由・コピー方向）に基づき是正したv14.14版。',
    '本計画はv14.21レビュー指摘（微小項切り捨て・maxQ一致保証・Oversampling Authority集約）に基づき是正したv14.22版。'
)

# 2. Goal section
content = content.replace(
    '- 推定は常に安全側上界を理論的に保証する。過剰評価は2dB以内を **推定目標** とするが、これは数学的保証ではなく実測ベースの検証目標である（Appendix C.1 注釈参照）',
    '- 推定は常に安全側上界を理論的に保証する。Builder 側で `max(measured, upperBound)` により `eqMaxGainDb` を決定し、Planner はその値のみを使用する（ISR思想: Planner は解析方法を知らない）。過剰評価は2dB以内を **推定目標** とするが、これは数学的保証ではなく実測ベースの検証目標である（Appendix C.1 注釈参照）'
)

# 3. PeakInfo with maxQ
old_peak = '''struct PeakInfo {
    float gainDb = 0.0f;      // ゲイン（dB）
    float freqHz = 0.0f;      // 当該ゲインが現れる周波数
};

struct EQAnalysisResult {
    PeakInfo measured;         // 実測最大ピーク（粗探索＋放物線補間の最大値）
    PeakInfo upperBound;       // 安全側上界の最大値（Π(1+|Hi-1|) の dB 値と、その最大値を与える周波数）
                               //   ※ upperBound.freqHz は上界が最大となる周波数であり、
                               //      measured.freqHz とは異なる場合がある
    float maxQ = 0.0f;        // 有効バンド中の最大Q値
};'''

new_peak = '''struct PeakInfo {
    float gainDb = 0.0f;      // ゲイン（dB）
    float freqHz = 0.0f;      // 当該ゲインが現れる周波数
    float maxQ = 0.0f;        // ★ v14.22: このピークを与えるバンドのQ値
};

struct EQAnalysisResult {
    PeakInfo measured;         // 実測最大ピーク（粗探索＋放物線補間の最大値）
    PeakInfo upperBound;       // 安全側上界の最大値（Π(1+|Hi-1|) の dB 値と、その最大値を与える周波数）
                               //   ※ upperBound.freqHz は上界が最大となる周波数であり、
                               //      measured.freqHz とは異なる場合がある。
                               //   maxQ も同様に measured の Q を採用（upperBound 側の Q は使用しない）
};'''

content = content.replace(old_peak, new_peak)

# 4. Add upperBound collapse section
old_ub = '''**upperBound の利用方針（★ v14.10: 案B — Builder 側で collapse）**:
- ISR 思想に基づき、Builder が `eqMaxGainDb = max(measured.gainDb, upperBound.gainDb)` を計算し、Planner はこの値のみを受け取る
- Planner は `measured` と `upperBound` の区別を知らない（解析方法への依存を排除）
- `upperBound` は診断ログとしても記録され、Parallel 構成での過大評価量の確認に使用可能

二層構造により以下が可能:'''

new_ub = '''**upperBound の利用方針（★ v14.10: 案B — Builder 側で collapse）**:
- ISR 思想に基づき、Builder が `eqMaxGainDb = max(measured.gainDb, upperBound.gainDb)` を計算し、Planner はこの値のみを受け取る
- Planner は `measured` と `upperBound` の区別を知らない（解析方法への依存を排除）
- `upperBound` は診断ログとしても記録され、Parallel 構成での過大評価量の確認に使用可能

二層構造により以下が可能:'''

content = content.replace(old_ub, new_ub)

# 5. PlannerInput DTO
old_qs = '''// QSurgePolicy — 経験的マージン。Boundではなく経験式。
// ISR思想に基づき Policy として分離。Builder/Planner/Test で共有。
struct QSurgePolicy {
    static constexpr float kBase       = 0.8f;
    static constexpr float kCoeffQ     = 0.12f;
    static constexpr float kCoeffGain  = 0.04f;
    static constexpr float kMax        = 2.5f;
    static constexpr float kButterworthQ = 0.707f;

    [[nodiscard]] static float evaluate(float eqGainDb, float maxQ) noexcept {
        if (eqGainDb <= 0.5f) return 0.0f;'''

new_qs = '''// ★ v14.15: PlannerInput — Planner 専用 DTO。物理的に Diagnostics へアクセス不可能。
struct PlannerInput {
    float eqMaxGainDb = 0.0f;   // Builder collapse 後の安全側値
    float eqMaxQ = 0.0f;        // 最大Q値
    float irFreqPeakGainDb = 0.0f; // IR 周波数ピークゲイン
};

// Planner Contract:
// - eqMaxGainDb は Builder により max(measured, upperBound) で安全側保証済み
// - Planner は PlannerInput のみを受け取り、BuildAnalysis.Diagnostics を参照不可能
// - 責務は「与えられた入力からマージンを計算し、4パターン分岐すること」のみ

// QSurgePolicy — 経験的マージン。Boundではなく経験式。
// ISR思想に基づき Policy として分離。Builder/Planner/Test で共有。
struct QSurgePolicy {
    inline static const float kBase       = 0.8f;
    inline static const float kCoeffQ     = 0.12f;
    inline static const float kCoeffGain  = 0.04f;
    inline static const float kMax        = 2.5f;
    inline static const float kButterworthQ = 0.707f;
    inline static const float kMinimumBoostForMargin = 0.5f;

    [[nodiscard]] static float evaluate(float eqGainDb, float maxQ) noexcept {
        if (eqGainDb <= kMinimumBoostForMargin) return 0.0f;'''

content = content.replace(old_qs, new_qs)

# 6. OversamplingResult
old_os = '''struct OversamplingResult {
    int resolvedOsFactor = 1;      // 解決済み倍率（Builder/Planner 共有の唯一値）
    int requestedOsFactor = 0;     // BuildInput からの要求値（0=Auto）
    bool isAutoResolved = false;   // Auto(0) からの解決済みか
};'''

new_os = '''struct OversamplingResult {
    int resolvedOsFactor = 1;      // 解決済み倍率（Builder/Planner 共有の唯一値）
    int requestedOsFactor = 0;     // BuildInput からの要求値（0=Auto）
    bool isAutoResolved = false;   // Auto(0) からの解決済みか

    [[nodiscard]] bool isValid() const noexcept {
        auto isFactor = [](int f) { switch (f) { case 1: case 2: case 4: case 8: return true; default: return false; }};
        auto isRequest = [](int f) { switch (f) { case 0: case 1: case 2: case 4: case 8: return true; default: return false; }};
        if (!isFactor(resolvedOsFactor) || !isRequest(requestedOsFactor)) return false;
        if ((requestedOsFactor == 0) != isAutoResolved) return false;
        return true;
    };
};'''

content = content.replace(old_os, new_os)

# 7. OversamplingPolicy section
old_osp = '''// OversamplingPolicy — Builder と Planner で共有する
// 解析用オーバーサンプリング倍率の決定ポリシー。
// 入力: BuildInput（oversamplingFactor=0 は Auto）
// 出力: 解析で使用する倍率（>= 1）
//
// Auto 時の決定論理:
//   48kHz 以下 → 4x
//   96kHz      → 2x
//   192kHz     → 1x
//   それ以外   → max(1, floor(192000 / sr))  // 192kHz換算
//
// ISR 設計: 純粋関数であり DSPCore の状態に依存しない。
// Policy として独立させることで、Builder 実装の変更が Planner に
// 影響しない。将来 CPU負荷・品質・Realtime Mode を考慮した
// ポリシー拡張も Policy の派生クラスで対応可能。
struct OversamplingPolicy {
    [[nodiscard]] static int resolve(const BuildInput& input) noexcept;
};

この Policy の導入により以下が保証される:
- Builder と Planner が常に同一の倍率で推定する（Auto時も一致、Single Source of Truth）
- 固定値ルール（48kHz→4x）より実際のビルド結果との差が小さい
- 新しいサンプルレート（384kHz等）が追加された場合も一箇所の修正で対応可能
- Builder 内部の倍率決定ロジックが変更されても Policy のみ更新すればよい'''

new_osp = '''// ★ v14.22: OversamplingPolicy — Builder 専有の決定権限（Authority Singularization）
// 入力: BuildInput（oversamplingFactor=0 は Auto）
// 出力: 解析で使用する倍率（>= 1）
//
// Authority chain:
//   Builder → OversamplingPolicy::resolve() → OversamplingResult → Snapshot
//                                                                  ↓
//   Planner → Snapshot.oversampling.resolvedOsFactor を読み取り専用
//
// 実装方針: 既存の DSPCore Auto 解決ロジック（RebuildDispatch.cpp:845-951）を
// 純粋関数として呼び出すアダプタ。新規に決定ロジックを書かない。
struct OversamplingPolicy {
    [[nodiscard]] static int resolve(const BuildInput& input) noexcept;
};

この Policy の導入により以下が保証される:
- Builder が唯一の決定権限を持つ（Authority Singularization）
- Planner は決定ロジックを一切知らず、Snapshot の結果を読み取り専用で参照
- 固定値ルール（48kHz→4x）より実際のビルド結果との差が小さい
- 新しいサンプルレート（384kHz等）が追加された場合も一箇所の修正で対応可能'''

content = content.replace(old_osp, new_osp)

# 8. BuildAnalysis struct
old_ba = '''struct BuildAnalysis {
    int generation = 0;
    float eqMaxGainDb = 0.0f;
    float eqMaxQ = 0.0f;                    // 新規
    float irFreqPeakGainDb = 0.0f;          // 新規
    float irAdditionalAttenuationDb = 0.0f; // 互換、常に0
    bool sealed = false;
};'''

new_ba = '''struct AnalysisVersionPolicy {
    static constexpr uint8_t kCurrent = 2;
    static constexpr uint8_t kLegacy  = 1;
};

struct BuildAnalysis {
    int generation = 0;
    float eqMaxGainDb = 0.0f;               // Builder collapse 後の安全側値
    float eqMaxQ = 0.0f;                    // 新規
    int resolvedOsFactor = 1;               // 解析 provenance
    float irFreqPeakGainDb = 0.0f;          // 新規
    float irAdditionalAttenuationDb = 0.0f; // 互換、常に0
    bool sealed = false;

    struct Diagnostics {
        uint8_t analysisVersion = AnalysisVersionPolicy::kCurrent;
        BoundMethod boundMethod = BoundMethod::TriangleProduct;
        float eqMeasuredGainDb = 0.0f;
        float eqUpperBoundGainDb = 0.0f;
        float boundExcessDb = 0.0f;
    } diag;
};'''

content = content.replace(old_ba, new_ba)

# 9. finite check
content = content.replace(
    'finite チェック対象: `eqMaxGainDb`, `eqMaxQ`, `irFreqPeakGainDb`（`RuntimeBuildTypes.h:85`）',
    'finite チェック対象: `eqMaxGainDb`, `eqMaxQ`, `irFreqPeakGainDb`, `resolvedOsFactor > 0`'
)

# 10. Verify section
old_verify = '''`sealBuildAnalysis` の検証（★ v14.11: resolvedOsFactor も検証）:
```cpp
// ★ v14.11: sealBuildAnalysis は解析結果の封印。resolvedOsFactor も検証。
if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
    || !isFiniteFloat(analysis.irFreqPeakGainDb)
    || analysis.resolvedOsFactor <= 0)
    return BuildAnalysis{};
```

**verifyBuildAnalysisPair() の拡張（★ v14.10: 新規関数追加ではなく既存関数を拡張）**:

```cpp
// verifyBuildAnalysisPair — 既存関数を拡張。OversamplingResult の検証を追加。
// Validator はこの一箇所のみ（ISR Authority Singularization）。
[[nodiscard]] inline bool verifyBuildAnalysisPair(
    const BuildAnalysis& analysis,
    const OversamplingResult& oversampling,
    const RuntimeBuildSnapshot& snapshot) noexcept
{
    if (!analysis.sealed || !snapshot.sealed)
        return false;
    if (analysis.generation != snapshot.generation)
        return false;
    if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
        || !isFiniteFloat(analysis.irFreqPeakGainDb))
        return false;
    if (oversampling.resolvedOsFactor <= 0          // 追加検証
        || oversampling.resolvedOsFactor != analysis.resolvedOsFactor)
        return false;
    return true;
}'''

new_verify = '''`sealBuildAnalysis` の検証:
```cpp
if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
    || !isFiniteFloat(analysis.irFreqPeakGainDb)
    || analysis.resolvedOsFactor <= 0)
    return BuildAnalysis{};
```

**verifyBuildBundle()（★ v14.18: 責務分割）**:

```cpp
[[nodiscard]] inline bool verifySnapshotSeal(
    const BuildAnalysis& analysis,
    const RuntimeBuildSnapshot& snapshot) noexcept {
    return analysis.sealed && snapshot.sealed
        && analysis.generation == snapshot.generation;
}

[[nodiscard]] inline bool verifyAnalysisConsistency(
    const BuildAnalysis& analysis,
    const AnalysisPart& analysisPart) noexcept {
    constexpr float kEps = 1e-5f;
    return analysisPart.analysisVersion == analysis.diag.analysisVersion
        && std::fabs(analysis.eqMaxGainDb - analysisPart.eqMaxGainDb) <= kEps
        && std::fabs(analysis.eqMaxQ - analysisPart.eqMaxQ) <= kEps
        && std::fabs(analysis.irFreqPeakGainDb - analysisPart.irFreqPeakGainDb) <= kEps;
}

[[nodiscard]] inline bool isKnownBoundMethod(BoundMethod m) noexcept {
    switch (m) {
        case BoundMethod::Unknown: return false;
        case BoundMethod::Legacy:
        case BoundMethod::TriangleProduct:
        case BoundMethod::ProductMaxMagnitude:
        case BoundMethod::ExactSampling: return true;
    }
    return false;
}

[[nodiscard]] inline bool verifyOversamplingConsistency(
    const BuildAnalysis& analysis,
    const OversamplingResult& oversampling) noexcept {
    if (!oversampling.isValid()) return false;
    if (oversampling.resolvedOsFactor != analysis.resolvedOsFactor) return false;
    if (oversampling.requestedOsFactor != 0
        && oversampling.requestedOsFactor != oversampling.resolvedOsFactor)
        return false;
    return true;
}

[[nodiscard]] inline bool verifyDiagnosticsConsistency(
    const BuildAnalysis& analysis) noexcept {
    constexpr float kEps = 1e-5f;
    if (!isKnownBoundMethod(analysis.diag.boundMethod)) return false;
    const float expectedExcess = std::max(0.0f,
        analysis.diag.eqUpperBoundGainDb - analysis.diag.eqMeasuredGainDb);
    return std::fabs(analysis.diag.boundExcessDb - expectedExcess) <= kEps
        && analysis.diag.boundExcessDb >= 0.0f;
}

[[nodiscard]] inline bool verifyBuildBundle(
    const BuildAnalysis& analysis,
    const OversamplingResult& oversampling,
    const RuntimeBuildSnapshot& snapshot,
    const AnalysisPart& analysisPart) noexcept {
    if (!verifySnapshotSeal(analysis, snapshot)) return false;
    if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
        || !isFiniteFloat(analysis.irFreqPeakGainDb)
        || analysis.resolvedOsFactor <= 0) return false;
    if (!verifyOversamplingConsistency(analysis, oversampling)) return false;
    if (!verifyAnalysisConsistency(analysis, analysisPart)) return false;
    if (!verifyDiagnosticsConsistency(analysis)) return false;
    return true;
}
```'''

content = content.replace(old_verify, new_verify)

# 11. BoundMethod enum
content = content.replace(
    'enum class BoundMethod : uint8_t {',
    'enum class BoundMethod : uint8_t {\n    Unknown = 0,\n    Legacy = 1,'
)

# 12. Small-term threshold
content = content.replace(
    '//   実用上は Π max(1, |Hi|) でも安全であることが数値検証されているが、\n//   数学的保証が必要な本書では Π(1 + |Hi - 1|) を採用する。',
    '//   実用上は Π max(1, |Hi|) でも安全であることが数値検証されているが、\n//   数学的保証が必要な本書では Π(1 + |Hi - 1|) を採用する。\n//\n// ★ v14.22: 微小項切り捨て（31Band浮動小数誤差蓄積防止）\n//   if (|Hi-1| < 1e-6) continue;  // 数学的保証に影響なし'
)

# 13. U-5
old_u5 = '| U-5 | `AnalysisPart` のバージョニング戦略 | **`AnalysisPart` に専用の `version` フィールドを追加（★ v14.6: 変更）**。`RuntimePublishSpecification.version` 全体を上げるのではなく、`AnalysisPart` 構造体内に `uint32_t analysisVersion = 1` を追加する。これにより Specification 全体の互換性を保ちながら AnalysisPart のみ独立して拡張可能。将来の Part 追加時にも影響範囲が限定される | `RuntimeBuilder.h:52-55` の `AnalysisPart` を拡張。`analysisVersion` で新旧を識別。Builder は version 1 のデフォルト値を認識可能 |'
new_u5 = '| U-5 | `AnalysisPart` のバージョニング戦略 | **`AnalysisVersionPolicy::kCurrent` が唯一の Authority**。Builder / BuildAnalysis.Diagnostics / AnalysisPart は全てこの定数から設定する。値 `2` が現在の解析フォーマット。不整合時は `verifyBuildBundle()` で検出 | `BuildAnalysis::AnalysisVersionPolicy` として定義 |'
content = content.replace(old_u5, new_u5)

# 14. CPU profiling in Automation Stress
content = content.replace(
    '  - **Rebuild Latency（平均・99 percentile・最大）を記録**',
    '  - **Rebuild Latency（平均・99 percentile・最大）を記録**\n  - **computeEstimatedMaxGainComplex() 実測CPU時間（ms）を記録**\n  - **candidate band 数も同時記録**'
)

# 15. Session Restore Test
content = content.replace(
    '| `computeEstimatedMaxGainComplex()` | NaN/Inf 入力でも有限値を返す | NaN/Inf 伝搬 |',
    '| `computeEstimatedMaxGainComplex()` | NaN/Inf 入力でも有限値を返す | NaN/Inf 伝搬 |\n| **Session Restore Test**: Build→Publish→Session Save→Reload→verifyBuildBundle() | analysisVersion/BoundMethod/Diagnostics 一致 | コピー漏れ・version不整合・BoundMethod Legacy識別 |'
)

# 16. Revision history - simplified
old_history = '''| **v14.14** | **2026-07-18** | **レビュー指摘4件に対応。①`verifyBuildBundle()` に `AnalysisPart` を追加。`analysisVersion` の不整合を実際に検出可能に。②`PlannerInput` DTO を新設し、Planner が物理的に `BuildAnalysis.Diagnostics` へアクセス不可能に。③`BoundMethod` を Runtime に保持する理由を注釈化（オフライン解析・長期トレンド）。④`analysisVersion` のコピー方向を `BuildAnalysis.Diagnostics → AnalysisPart` に修正し、ビルドフローと整合** |'''

new_history = '''| **v14.14** | **2026-07-18** | **verifyBundle・PlannerInput・BoundMethod注釈・コピー方向修正** |
| **v14.15** | **2026-07-18** | **全フィールド検証・QSurgePolicy config・boundExcessDb** |
| **v14.16** | **2026-07-18** | **epsilon・isValid許容値・invariant・jassert** |
| **v14.17** | **2026-07-18** | **epsilon値・requestedOsFactor検証・magic number・証明補強** |
| **v14.18** | **2026-07-18** | **verify責務分割・boundExcess Release保護・CPU実測・補題** |
| **v14.19** | **2026-07-18** | **float比較・isValid簡素化・BoundMethod Migration・Oversampling Authority** |
| **v14.20** | **2026-07-18** | **epsilon 1e-5・kCurrentAnalysisVersion・isAutoResolved・Legacy・Session Restore Test** |
| **v14.21** | **2026-07-18** | **Legacy受理・boundExcess統一・AnalysisVersionPolicy** |
| **v14.22** | **2026-07-18** | **微小項切り捨て: `|Hi-1|<1e-6` 除外・`maxQ` を `PeakInfo` に移動し eqMaxGainDb/eqMaxQ 同一ピーク保証・`OversamplingPolicy` Authority Builder専有・`isKnownBoundMethod()`導入・`boundExcess>40dB` DiagEvent候補・`AnalysisVersionPolicy`独立ヘッダ推奨** |'''

content = content.replace(old_history, new_history)

# 17. Builder code update
content = content.replace(
    'analysis.diag.boundMethod = BoundMethod::TriangleProduct;',
    'analysis.diag.boundMethod = BoundMethod::TriangleProduct;\n  analysis.diag.boundExcessDb = std::max(0.0f,\n      eqResult.upperBound.gainDb - eqResult.measured.gainDb);\n  jassert(analysis.diag.boundExcessDb >= 0.0f && analysis.diag.boundExcessDb < 40.0f);'
)

# 18. Update Appendix C.6
content = content.replace(
    '#### C.6 analysisVersion インクリメントポリシー（★ v14.12: 新規追加）',
    '#### C.6 analysisVersion インクリメントポリシー（★ v14.21: Policy 参照に更新）\n\n`AnalysisVersionPolicy::kCurrent` が唯一のバージョン定義。これを increment する条件:'
)

# Write with proper UTF-8
with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done! File updated to v14.22')
