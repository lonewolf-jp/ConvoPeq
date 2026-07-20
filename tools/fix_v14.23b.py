# -*- coding: utf-8 -*-
with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix Diagnostics section (add resolvedOsFactor, boundExcessDb, use Policy)
old_diag = '''        uint8_t analysisVersion = 2;         // BuildAnalysis → AnalysisPart へコピーされる
        BoundMethod boundMethod = BoundMethod::TriangleProduct;
        float eqMeasuredGainDb = 0.0f;       // collapse前の measured（診断用）
        float eqUpperBoundGainDb = 0.0f;     // collapse前の upperBound（診断用）
    } diag;
};

finite チェック対象: `eqMaxGainDb`, `eqMaxQ`, `irFreqPeakGainDb`, `resolvedOsFactor > 0`

`sealBuildAnalysis` の検証（★ v14.11: resolvedOsFactor も検証）:
```cpp
// ★ v14.11: sealBuildAnalysis は解析結果の封印。resolvedOsFactor も検証。
if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
    || !isFiniteFloat(analysis.irFreqPeakGainDb)
    || analysis.resolvedOsFactor <= 0)
    return BuildAnalysis{};'''

new_diag = '''        uint8_t analysisVersion = AnalysisVersionPolicy::kCurrent;
        int resolvedOsFactor = 1;
        BoundMethod boundMethod = BoundMethod::TriangleProduct;
        float eqMeasuredGainDb = 0.0f;
        float eqUpperBoundGainDb = 0.0f;
        float boundExcessDb = 0.0f;
    } diag;
};

finite チェック対象: `eqMaxGainDb`, `eqMaxQ`, `irFreqPeakGainDb`

`sealBuildAnalysis` の検証:
```cpp
if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
    || !isFiniteFloat(analysis.irFreqPeakGainDb))
    return BuildAnalysis{};'''

if old_diag in content:
    content = content.replace(old_diag, new_diag)
    print('Diagnostics: OK')
else:
    print('Diagnostics: NOT FOUND')

# Write back
with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
