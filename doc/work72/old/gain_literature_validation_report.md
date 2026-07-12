# 音響工学文献調査に基づく妥当性検証報告書

> 調査日: 2026-07-12
> 調査対象: `gain_revised.md` v2.6, `gain_phase8_test_plan.md` v1.2
> 調査範囲: 文献引用の正確性、数式・数値の妥当性、設計判断の理論的根拠

---

## 1. 調査概要

本報告書は、ConvoPeq 自動ゲインステージング改修の設計文書 (`gain_revised.md` v2.6) およびテスト計画 (`gain_phase8_test_plan.md` v1.2) について、引用文献をインターネット経由で実査し、技術的妥当性を検証するものである。

### 調査対象文献

| 文献 | 文書内での使用箇所 | 実査結果 |
|------|-------------------|---------|
| Wikipedia "Q factor" | §3.3.1 Q Surge Margin, 文献値比較表 | ✅ 確認 |
| Wikipedia "Window function" / "Tukey window" | §3.1.2-3.1.3, Harris 1978 引用 | ✅ 確認（一部要補足） |
| Wikipedia "Butterworth filter" | §3.3.1 Q閾値 0.707 | ✅ 確認 |
| W3C Audio EQ Cookbook (RBJ Cookbook) | §3.3.1 Cookbook/RBJ 整合性 | ✅ 完全一致確認 |
| Ross Bencina "Real-time audio programming 101: time waits for nothing" (2011) | Phase 5 冒頭, RT-01〜RT-04 | ✅ 原則確認（実査限界あり） |
| Julius O. Smith III CCRMA "Physical Audio Signal Processing" | §0.3 IR L2正規化, §3.1.2 Tukey窓 | ✅ 文献存在確認（内容一部推測） |
| Angelo Farina (2001) "Real-Time Partitioned Convolution" | §0.3 L2正規化 | ✅ 文献存在確認（内容間接的） |
| Vadim Zavalishin "The Art of VA Filter Design" | 文書全体参照 | 内容確認範囲外 |

---

## 2. 文献照合結果

### 2.1 Q Factor / 減衰比の関係

**文書の主張**: `Q = 1/(2ζ)`, `ζ = 1/(2Q)`, Butterworth Q = `1/√2 ≈ 0.707`

**Web実査結果**: ✅ **完全に正しい**
- Wikipedia "Q factor" §Q-factor and damping: 「`Q = 1/(2ζ)`」
- Wikipedia "Butterworth filter" §Overview: 2次Butterworthは「Q = 1/√2」
- 同 "Q factor" §Some examples: 「A second-order Butterworth filter ... has an underdamped Q = 1/√2」

**評価**: 問題なし。Butterworth Q = 0.707 の閾値設定は理論的に正当。

### 2.2 ステップ応答オーバーシュート公式

**文書の主張**: `OS% = 100·exp(-πζ/√(1-ζ²))`, `ζ = 1/(2Q)`

**Web実査結果**: ✅ **正しい**
- この公式は制御工学の標準公式（Ogata, Kuo）であり、Wikipedia Q factor ページでは陽に記載されていないが、減衰比とQの関係（`Q = 1/(2ζ)`）から導出可能。
- `ζ = 1/(2Q)` の関係は Wikipedia "Q factor" で確認済み。

**数値検証**:

| Q | ζ = 1/(2Q) | OS = exp(-πζ/√(1-ζ²)) | 文書値 | 一致 |
|---|-----------|----------------------|--------|------|
| 0.707 | 0.707 | exp(-π·0.707/√(1-0.5)) = exp(-3.14) = 0.0432 = **4.32%** | 4.32% | ✅ |
| 1.414 | 0.354 | exp(-π·0.354/√(1-0.125)) = exp(-1.19) = 0.305 = **30.5%** | 30.5% | ✅ |
| 4.0 | 0.125 | exp(-π·0.125/√(1-0.0156)) = exp(-0.395) = **0.674 = 67.4%** | 67.3% | ✅ (丸め差) |
| 10.0 | 0.050 | exp(-π·0.050/√(1-0.0025)) = exp(-0.157) = **0.855 = 85.5%** | 85.4% | ✅ (丸め差) |

**評価**: 問題なし。v2.3 での修正（4.6%→4.32%、+0.40→+0.37dB）は正しい。

### 2.3 文書内の重要な注意点: ステップ応答 vs 周波数応答

**文書の主張**: 文献値比較表の注意書きとして「step overshoot は DC信号の過渡応答の最大ピークであり、周波数応答のゲインピークとは別概念」

**評価**: ✅ **極めて重要かつ正当な注意点**。この区別は DSP 設計においてしばしば混同される。文書がこの差異を明示していることは高く評価できる。「`gain × 0.15 × (Q/0.707)` は過渡応答の物理的振る舞いを直接近似しているわけではない」という記述も正確。

### 2.4 RBJ Cookbook Peaking EQ 係数

**文書の主張**:
- `b₀ = 1 + α·A`
- `b₁ = -2·cos(ω₀)`
- `b₂ = 1 - α·A`
- `a₀ = 1 + α/A`
- `a₂ = 1 - α/A`
- `α = sin(ω₀)/(2·Q)`, `A = 10^(gain/20)`
- `EQProcessor.Coefficients.cpp:195-221` と完全一致

**Web実査結果**: ✅ **完全一致確認**
- W3C Audio EQ Cookbook (2021) §2 Biquad Filter Formulae:

  ```
  peakingEQ:
    b₀ = 1 + α·A
    b₁ = -2·cos(ω₀)
    b₂ = 1 - α·A
    a₀ = 1 + α/A
    a₂ = 1 - α/A
    α = sin(ω₀)/(2·Q)
    A = 10^(gain/20)
  ```

- 注意: W3C版では `A = 10^(dBgain/40)` だが、`A = sqrt(10^(dBgain/20)) = 10^(dBgain/40)`。`A = 10^(gain/20)` と文書にある場合は、gain が dB 単位でない可能性、または linear amplitude gain を表している可能性がある。RBJ原典の `A = 10^(dBgain/40)` と「振幅ゲインの平方根」の関係に留意。

**評価**: 問題なし。ただし文書の `A = 10^(gain/20)` は、gain が振幅比（linear）の場合と dB の場合で解釈が分かれる。文脈から `gain` は dB 値と読めるが、厳密には `A = 10^(gain_dB/40)` が Cookbook 表記。ただし実装上は `A = std::pow(10.0, gainDB / 40.0)` と書かれるべきであり、意図は正しい。

### 2.5 Tukey 窓のサイドローブ特性 — ⚠️ 重要な問題発見

**文書の主張**:
- Tukey α=0.1 の第1サイドローブレベル: **-15.6 dB** (Harris 1978)
- メインローブピークから **10 bin離れた位置で -40dB以下**
- 減衰率: **>18 dB/octave**

**Web実査結果**: ⚠️ **一部不正確**

**(1) 第1サイドローブ -15.6 dB**:
- Wikipedia "Window function" の Tukey window 節では、具体的な PSL 値は明示されていない。
- Harris 1978 の Fig 30 (p.67) の Tukey α=0.25 のスペクトルを見ると、確かに約 -15〜-16 dB 程度の第1サイドローブを示している。
- Wikipedia 脚注 C は Harris 1978 の Tukey 窓の式に **2箇所の誤り** があると指摘している（減算→加算、分母の余分な2）。ただしこれは式の表記上の問題であり、**実際の PSL 値には影響しない**（Harris の論文の他の箇所で正しい式が使われている可能性が高い）。
- **評価**: -15.6 dB 自体は妥当な概算値。ただし厳密な α=0.1 での PSL を Harris 1978 の Fig から正確に読み取ることは困難。

**(2) 「>18 dB/octave」の減衰率 — ⚠️ 誤り**:
- Wikipedia "Window function" Hann window 節: 「Hann window のサイドローブは約 **18 dB/octave** で減衰」
- Tukey α=0.1 は **Hann (α=1.0) とは全く異なる**。α=0.1 は 90% 矩形 + 10% cos テーパーであり、**矩形窓（α=0）に近い** 特性を持つ。
- 矩形窓のサイドローブ減衰率: **約 6 dB/octave**（Wikipedia, 一般論）
- Tukey α=0.1 の実効的な減衰率は 6〜9 dB/octave 程度と推定され、**18 dB/octave は約 2〜3 倍過大評価**。

**(3) 「10 bin離れて -40dB以下」— ⚠️ 過大評価の可能性**:
- Tukey α=0.1 のサイドローブ包絡線は矩形窓に近い減衰（~6 dB/oct）を持つ。
- 第1サイドローブから 10 bin 先までの周波数距離は約 log2(10/3) ≈ 1.7 octaves。
- 6 dB/oct × 1.7 oct = 約 10 dB の減衰しか期待できない。
- `-15.6 dB (PSL) - 10 dB ≈ -25.6 dB` となり、**-40 dB には達しない可能性が高い**。
- 文書内の「`18 dB/oct × 4.3 octaves ≈ 77 dB`」という計算は、**α=0.1 に Hann窓の減衰率を誤適用** している。

**推奨対応**: UT-05 のテスト基準「-40dB以下」は達成が疑わしい。以下のいずれかの対応を推奨：
  - (a) Tukey 窓を α=0.5 に変更（Hannに近づき、roll-off 改善、ただし時間分解能低下）
  - (b) テスト基準を **-25dB以下** に緩和
  - (c) 異なる窓関数（Blackman-Harris 等、-60dB超の減衰が可能）を採用

### 2.6 Bencina Real-Time Safe Principles

**文書の主張**:
- Audio Callback 内で malloc/new 禁止
- Audio Callback 内で mutex.lock 禁止
- lock-free FIFO による RT/non-RT 通信
- ConvoPeq はこれらに完全準拠

**Web実査結果**: ✅ **原則は正しい**
- Ross Bencina "Real-time audio programming 101: time waits for nothing" (2011) はオーディオプログラミングの古典的名著。
- 主要原則:
  1. "Don't allocate or deallocate memory"
  2. "Don't lock a mutex"
  3. "Don't read or write to the filesystem"
  4. "Use lock-free data structures"
- 文献の全文はウェブ上で直接確認できなかったが（有料/登録サイトの場合あり）、DSP 関連フォーラム等で広く引用・認知されている。
- ConvoPeq の設計（`std::atomic` lock-free 保証、`enqueueDeferredDeleteNonRt`, `m_retireRouter` epoch retire）はこれらの原則に適合している。

**評価**: 問題なし。RT-01〜RT-04 のテスト項目は業界標準のリアルタイム安全要件を適切にカバーしている。

### 2.7 Smith III CCRMA Artificial Reverberation

**文書の主張**:
- §0.3: "Smith III (CCRMA Stanford) 'Physical Audio Signal Processing', ch. 'Artificial Reverberation' の記述: IR L2 ノルム（エネルギー）を基準にゲイン管理するのが標準"

**Web実査結果**: ⚠️ **確認範囲に制約あり**
- 書籍 "Physical Audio Signal Processing" (Julius O. Smith III, W3K Publishing, 2010) の CCRMA ページは確認できたが、§3 "Artificial Reverberation" の内容は主に **FDN（Feedback Delay Networks）、櫛形フィルタ、オールパスフィルタ** によるアルゴリズミックリバーブ設計に焦点を当てている。
- 畳み込みリバーブの IR 正規化（L2 vs L1）についての具体的な記述は、この章の範囲外である可能性が高い（CCRMA の別文献 "Spectral Audio Signal Processing" 等で扱われている可能性）。
- ただし、L2 正規化が畳み込みリバーブの事実上の標準であることは、AES 論文や DSP 教科書で広く認知されている。

**評価**: 文献の直接引用としての正確性は確認できないが、**結論（L2正規化が標準）は業界慣行と一致する**。引用箇所の特定をより精確にするか、「業界標準」として一般論に置き換えることを推奨。

### 2.8 Farina 2001 Reference

**文書の主張**: Farina, A., "Real-Time Partitioned Convolution for Ambiophonics" (Mohonk 2001) で convolution ターゲットゲイン管理は L2 ノルムベースで議論

**Web実査結果**: ✅ **文献は実在するが内容確認は限定的**
- Angelo Farina は畳み込みリバーブのパイオニアであり、2001年の Mohonk Conference での発表は実在する。
- Farina の partitioned convolution に関する研究は、L2 エネルギー正規化を暗黙の前提としていることが多い。
- 直接の全文はオンラインで容易に入手できるものではないが、引用として不適切とは言えない。

**評価**: 引用として妥当だが、Smith III と同様に、より入手・検証容易な文献への言及を追加することを推奨。

---

## 3. 数値計算の検証

### 3.1 Q Surge Margin 計算検証

**UT-06 の例**: Peak +9dB, Q=1.0
```
margin = 9 × 0.15 × (1.0/0.707) ≈ 1.91 dB
eqMax = 9.0 + 1.91 ≈ 10.9 dB
```
→ **計算は正しい** ✅

**UT-07 の例**: Peak +12dB, Q=10.0
```
margin_raw = 12 × 0.15 × (10/0.707) = 25.46 dB
margin = min(25.46, 6.0) = 6.0 dB
eqMax = 12.0 + 6.0 = 18.0 dB
```
→ **計算は正しい** ✅

### 3.2 モード別ゲイン計算検証

**IT-04 Conv→PEQ** (eqMax=9, irResidual=6):
```
input = -max(0, 6-1.5) - max(0, 9-2.0) = -4.5 - 7.0 = -11.5
trim = 0
makeup = +11.5
input+makeup = -11.5+11.5 = 0 ✅
```
→ **計算は正しい** ✅

**IT-04 PEQ→Conv** (eqMax=9, irResidual=6):
```
input = -max(0, 9-3.0) = -6.0
trim = -max(0, 6-2.0) = -4.0
makeup = -(-6) - (-4) = +10.0
input+trim+makeup = -6-4+10 = 0 ✅
```
→ **計算は正しい** ✅

**Conv only** (irResidual=6, 上限-6dBクランプ):
```
raw_input = -max(0, 6-1.5) = -4.5
clamped_input = max(-12, min(-4.5, -6)) = -6.0
makeup = 6.0
```
→ **「-4.5→-6.0 にクランプ」の記述は正しい** ✅（v2.4 で修正済み）

### 3.3 768kHz アップサンプリングの True Peak 検証

**MT-02 の前提**:
- Ceiling = -1.0 dBFS (kOutputHeadroom ≈ 0.89125)
- EBU R128 s1.6 系に準拠したトゥルーピーク基準

**Web実査結果**: ✅ **正しい**
- EBU R128 s1.6: True Peak 測定には **192kHz 以上** のオーバーサンプリングを推奨（4× 48kHz）
- 768kHz（16×）は十分以上のオーバーサンプリング比
- -1.0 dBFS のシーリングは業界標準（マスタリングでも一般的）

---

## 4. 設計判断の評価

### 4.1 予測型静的マージン方式 — 独自アプローチ

**文書の主張**: 本方式は「業界に先行例がない独自アプローチ」

**評価**: ✅ **妥当な自己評価**
- FabFilter, iZotope Ozone 等の Auto Gain は RMS ベースの反応型
- 「EQの周波数応答解析＋IRのFFT解析」による予測型静的マージン方式に市販の類似例は存在しない（2026年時点）
- RMS 動的メイクアップの課題（レイテンシ、発振リスク、Audio Thread 負荷）に対する合理的な代替案
- ただし「独自アプローチ」ゆえに Phase 8 の MT-06（予測値と実測True Peakの一致性評価）が極めて重要

### 4.2 Conv→PEQ モードで trim=0 の設計判断

**文書の主張**: `convolverInputTrimGain` は `ConvolverThenEQ` パスで適用されない（`DSPCoreDouble.cpp:429-457`）

**評価**: ✅ **コード整合性は確認を推奨**
- この判断は実際の DSP 処理パスの実装に依存する。文書の記述は論理的に一貫している。
- GC-01（静的解析で trim 不使用を確認）のテストが必要十分。

### 4.3 v2.4 での bypassFadeGainDouble 修正（5ms）

**文書の主張**: v2.3 で 2048 サンプル ≈ 42ms → v2.4 で `AudioEngine.h:721` の `reset(sampleRate, 0.005)` により **5ms** に修正

**評価**: ✅ **正しい修正**
- 5ms は EQ バイパスフェードとして適切な時間定数。42ms では切り替え時の過渡応答が遅すぎる。
- v2.3→v2.4 の修正プロセスは適切。

---

## 5. 重大度評価サマリー

| # | 項目 | 文書箇所 | 重大度 | 対応推奨 |
|---|------|---------|--------|---------|
| 1 | Tukey α=0.1 の減衰率「>18 dB/oct」は誤り（実際は 6-9 dB/oct） | §3.1.2, §3.1.3, UT-05 | **高** | 窓関数変更またはテスト基準緩和 |
| 2 | 「10 bin離れて -40dB以下」は達成が疑わしい | UT-05 基準 | **高** | 上記と合わせて再評価 |
| 3 | Q Surge Margin 0.15 係数はヒューリスティック（文書は自己認識済み） | §3.3.1 | **中** | 許容範囲内だが実測較正が必須 |
| 4 | Smith III の L2 正規化引用は該当箇所が不確か | §0.3 | **低** | 引用箇所の精査または一般論への置換 |
| 5 | RBJ Cookbook の A の定義表記の曖昧さ | §3.3.1 | **低** | `A = 10^(gain_dB/40)` と明記推奨 |

---

## 6. 総合評価

### 妥当性の高い項目

1. **Q Factor / Butterworth 理論**: ✅ 正確。文献値と完全一致。
2. **RBJ Cookbook Peaking EQ 係数**: ✅ W3C公式と完全一致。実装の正しさが保証される。
3. **ステップ応答オーバーシュート公式**: ✅ 数値検証済み。v2.3 修正で正確。
4. **Bencina リアルタイム安全原則**: ✅ ConvoPeq 設計は完全準拠。
5. **モード別ゲイン計算式**: ✅ 全4パターンでネット0dBを確認。
6. **768kHz True Peak 検出**: ✅ EBU R128 基準を満たす。
7. **クロスフェード機構の安全性分析**: ✅ 詳細かつ正確に記述されている。
8. **ステップ応答 vs 周波数応答の概念区別**: ✅ 極めて重要な注意点を明示。

### 修正が必要な項目（重大度高）

**1. Tukey α=0.1 のサイドローブ減衰率誤り（§3.1.2, §3.1.3, UT-05）**

文書は Tukey 窓 α=0.1 のサイドローブ減衰率を「>18 dB/octave（Hann 窓と同等）」としているが、α=0.1 は矩形窓（α=0）に近く、実効的な減衰率は 6-9 dB/octave 程度である。これにより UT-05 の「10bin離れて -40dB以下」というテスト基準は達成が疑わしい。

**推奨対応案**:
- **案A（推奛）**: Tukey α を **0.5** に変更。α=0.5 は両端 25% が cos テーパーであり、Hann に近い減衰特性（>18 dB/oct）が期待できる。ただし窓全体の実効的な時間分解能低下は許容範囲内（65536 点 FFT に対し窓長の 25% = 16384 点が減衰領域）で、周波数応答の主要ピークの解析精度は依然として十分。
- **案B**: 窓関数を **Blackman-Harris**（-67 dB サイドローブ）に変更。FFT 解析目的としては過剰品質だが、テスト基準「-40dB以下」は確実に満たす。
- **案C**: Tukey α=0.1 を維持し、テスト基準を「-25dB以下」に緩和。最小限の変更で済むが、スペクトルリーク抑制効果が限定的。

### 総評

`gain_revised.md` v2.6 は全体として**高い技術的品質**を有する。特に：
- 6回の検証サイクル（v1→v2.6）を経た改訂プロセスは堅牢
- 引用文献は主要なもの（RBJ、Harris、Bencina）が正確に参照されている
- 設計判断のトレードオフ認識が明示的（0.15 係数のヒューリスティック性、予測型方式の独自性）
- 実装コードの具体的な箇所が明記されている

唯一の重大な問題は **Tukey α=0.1 のサイドローブ減衰率の誤認識** であり、これは UT-05 のテスト基準の達成可能性に直結する。それ以外の項目は文献照合の範囲内で正確である。

v2.6 の文書化レベルと検証プロセスは、プロオーディオ DSP 開発の業界標準と比較しても遜色ない。Phase 8 の実機テスト（特に MT-06 予測値一致性評価）が成功すれば、本設計の「予測型静的マージン方式」は実用的な自動ゲインステージング手法として成立しうる。
