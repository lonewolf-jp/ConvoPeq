# -*- coding: utf-8 -*-
with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the OversamplingPolicy section (4.5)
# Look for the section comment that starts the Policy code block
old_marker = '// ★ v14.24: OversamplingPolicy'
idx = content.find(old_marker)
if idx < 0:
    print('ERROR: OversamplingPolicy section not found')
else:
    print(f'Found OversamplingPolicy section at position {idx}')
    # Find the end of the section - look for '#### 4.6' or next heading
    end_marker = '#### 4.6'
    end_idx = content.find(end_marker, idx)
    if end_idx < 0:
        print('ERROR: 4.6 section end not found')
    else:
        old_section = content[idx:end_idx]
        new_section = '''// ★ v14.25: OversamplingPolicy — Builder 専有の決定ポリシー。\n// Planner は決定ロジックを一切知らず、Snapshot.oversampling.resolvedOsFactor を読み取り専用で参照。\n//\n// システム全体の制約:\n//   - 入力: 44.1kHz〜768kHz\n//   - 内部処理レート: 最大 768kHz（OS 後）\n//   - 出力: 44.1kHz〜768kHz\n//\n// 入力: BuildInput（oversamplingFactor=0 は Auto）\n// 出力: 解析で使用する倍率（>= 1）\n//\n// Auto 時の決定論理（入力 SR → 最大 OS 倍率）:\n//   44.1kHz 〜 96kHz   → x8  （96x8=768 ≤ 768k）\n//   96kHz   〜 192kHz  → x4  （192x4=768）\n//   192kHz  〜 384kHz  → x2  （384x2=768）\n//   384kHz  〜 768kHz  → x1  （OS なし）\n//\n// 実装方針:\n//   min(8, max(1, floor(768000 / sr)))  // 768kHz 上限、x8 上限\n//\n// ISR 設計: 純粋関数であり DSPCore の状態に依存しない。\nstruct OversamplingPolicy {\n    static constexpr double kMaxInternalRate = 768000.0;\n    static constexpr int kMaxFactor = 8;\n\n    [[nodiscard]] static int resolve(const BuildInput& input) noexcept;\n};\n\nこの Policy の導入により以下が保証される:\n- Builder が唯一の決定権限を持ち、Planner は Snapshot の結果を読み取り専用で参照する\n- 最大内部レート 768kHz の制約を満たす最大倍率が選択される\n- 新しいサンプルレートが追加された場合も一箇所の修正で対応可能\n- Builder 内部の倍率決定ロジックが変更されても Policy のみ更新すればよい\n\n'''
        content = content[:idx] + new_section + content[end_idx:]
        print('Replacement applied successfully')

with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
