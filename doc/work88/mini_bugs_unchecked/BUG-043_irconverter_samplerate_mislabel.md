# BUG-043: IRConverter::estimateMaxFrequencyResponseGain の sampleRate パラメータ誤表示

## 重要度
低 (Low) — 現在の動作に影響はないが、API の意図と実態が乖離しており保守性を損ねる。

## 対象ファイル
- `src/IRConverter.h:46-47`
- `src/IRConverter.cpp:394-399`

## 概要
`IRConverter::estimateMaxFrequencyResponseGain` は `double sampleRate` という名前の
パラメータを取るよう宣言されているが、実装では `/*sampleRate*/` とパラメータ名を
コメントアウトしており、引数を完全に無視して `IRAnalyzer::estimateMaxFrequencyResponseGain(ir)`
に委譲している。

## 問題の箇所
**IRConverter.h:46-47**
```cpp
static double estimateMaxFrequencyResponseGain(const juce::AudioBuffer<double>& ir,
                                               double sampleRate) noexcept;
```

**IRConverter.cpp:394-399**
```cpp
double IRConverter::estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir,
    double /*sampleRate*/) noexcept
{
    return IRAnalyzer::estimateMaxFrequencyResponseGain(ir);
}
```

コメントには「★ v14.0: 後方互換用デリゲート — IRAnalyzer に委譲」とあるが:
1. **パラメータ名が「sampleRate」と誤解を招く** — 周波数応答解析にサンプルレートは不要。
   内部で FFT を行い周波数ビンごとに振幅を計算するが、サンプルレート情報を全く使っていない。
2. **呼び出し側が誤った仮定を持つリスク** — サンプルレートを変更しても結果が変わらず、
   デバッグが困難になる。
3. **呼び出し側が無意味な値を渡す負荷** — どの値でも同じ結果が返るが、API 上は意味のある
   引数のように見える。

## 修正案
```cpp
// IRConverter.h
static double estimateMaxFrequencyResponseGain(const juce::AudioBuffer<double>& ir) noexcept;

// IRConverter.cpp
double IRConverter::estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir) noexcept
{
    return IRAnalyzer::estimateMaxFrequencyResponseGain(ir);
}
```

または、将来の周波数重み付け対応を見越してパラメータを残すのであれば、
「現在は未使用だが将来拡張用」であることを明示するコメントを付ける。
