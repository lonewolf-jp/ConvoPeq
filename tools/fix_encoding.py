import re

with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'r', encoding='utf-8') as f:
    content = f.read()

assert '改修計画書' in content, 'File is corrupted'
print(f'File OK, {len(content)} chars')

# 1. Header
content = content.replace(
    '# ConvoPeq Auto Gain Staging 改修計画書 v14.14\n\n> 最終更新: 2026-07-18 | 編集者: GitHub Copilot (OpenCode Go / Deepseek V4 Flash)\n\n本計画はv14.13レビュー指摘4件（verifyとAnalysisPartの不一致・PlannerInput DTO・BoundMethod保持理由・コピー方向）に基づき是正したv14.14版。',
    '# ConvoPeq Auto Gain Staging 改修計画書 v14.22\n\n> 最終更新: 2026-07-18 | 編集者: GitHub Copilot (OpenCode Go / Deepseek V4 Flash)\n\n本計画はv14.21レビュー指摘（微小項切り捨て・maxQ一致保証・Oversampling Authority集約）に基づき是正したv14.22版。'
)

# 2. Goal
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
};

**upperBound の利用方針（★ v14.10: 案B — Builder 側で collapse）**:'''

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
};

**upperBound の利用方針（★ v14.10: 案B — Builder 側で collapse）**:
- ISR 思想に基づき、Builder が `eqMaxGainDb = max(measured.gainDb, upperBound.gainDb)` を計算し、Planner はこの値のみを受け取る
- Planner は `measured` と `upperBound` の区別を知らない（解析方法への依存を排除）
- `upperBound` は診断ログとしても記録され、Parallel 構成での過大評価量の確認に使用可能'''

content = content.replace(old_peak, new_peak)

# 4. Small-term threshold comment
content = content.replace(
    '//   実用上は Π max(1, |Hi|) でも安全であることが数値検証されているが、\n//   数学的保証が必要な本書では Π(1 + |Hi - 1|) を採用する。',
    '//   実用上は Π max(1, |Hi|) でも安全であることが数値検証されているが、\n//   数学的保証が必要な本書では Π(1 + |Hi - 1|) を採用する。\n//\n// ★ v14.22: 微小項切り捨て（31Band浮動小数誤差蓄積防止）\n//   if (|Hi-1| < 1e-6) continue;  // 数学的保証に影響なし'
)

# 5. PlannerInput DTO
old_planner = '''// QSurgePolicy — 経験的マージン。Boundではなく経験式。
// ISR思想に基づき Policy として分離。Builder/Planner/Test で共有。
struct QSurgePolicy {
    static constexpr float kBase       = 0.8f;
    static constexpr float kCoeffQ     = 0.12f;
    static constexpr float kCoeffGain  = 0.04f;'''

new_planner = '''// ★ v14.15: PlannerInput — Planner 専用 DTO。物理的に Diagnostics へアクセス不可能。
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
    inline static const float kCoeffGain  = 0.04f;'''

content = content.replace(old_planner, new_planner)

# Write back
with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done!')
