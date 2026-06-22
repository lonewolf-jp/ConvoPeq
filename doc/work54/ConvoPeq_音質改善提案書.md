# ConvoPeq 音質改善提案書

対象: `ConvoPeq.md`（約239ファイル / 約6.7万行、最新版ソースコード一括抽出）
調査範囲: `EQProcessor.*`, `MKLNonUniformConvolver.*`, `AllpassDesigner.*`, `ConvolverProcessor.MixedPhase.cpp`,
`CustomInputOversampler.*`, `NoiseShaperLearner.*` / `LatticeNoiseShaper.h` / `MklFftEvaluator.h` / `PsychoacousticDither.h`,
`IRConverter.cpp` / `IRDSP.cpp`, `OutputFilter.*`, `UltraHighRateDCBlocker.h`, `InputBitDepthTransform.h`,
`DspNumericPolicy.h`, `AudioEngine.Processing.DSPCoreDouble.cpp` 他
調査方法: ソースコード直接解析 + DAFx/IEEE SPL/AES文献調査（Web検索）

---

## 0. 総評：現状アーキテクチャの評価

最初に明記しておきたいのは、ConvoPeqは既に**プロフェッショナルグレードを超える**水準のDSP実装を備えているという点である。一般的な商用EQ/コンボルバープラグインと比較しても下記は明確な強みであり、本提案書の改善案はこれらを「壊さず伸ばす」方向で設計した。

| サブシステム | 実装内容 | 評価 |
|---|---|---|
| 20バンドEQ | Zavalishin TPT SVF（Topology-Preserving Transform State Variable Filter）、Serial/Parallel構造選択可、L/R/Stereo個別チャンネルモード | 教科書（*The Art of VA Filter Design*）に忠実な正確な実装。係数のNaN/Inf防御も丁寧 |
| 畳み込みエンジン | Intel IPP FFTベースの3層非均一分割畳み込み（NUPC, Gardner方式）。直接形FIRヘッド（ゼイテンシ近似）＋ L0即時/L1・L2分散処理 | 業務用コンボルバーに匹敵する設計。CPUスパイク対策も実装済み |
| Mixed Phase処理 | 線形位相IRと最小位相IRの群遅延差をターゲットとし、2次オールパス縦列（AdaGrad / CMA-ES最適化）で近似する独自実装 | 学術文献レベルの高度な手法。市販プラグインでもここまでやる例は少ない |
| ノイズシェーパ学習 | 実プログラム素材を複数レベルでキャプチャし、MPEG Annex D方式の心理音響マスキングモデル（Bark尺度24帯域、tonal/noiseマスカー分類、拡散関数、ATH＝Terhardt式、A重み）でCMA-ES学習 | 事実上MP3心理音響モデル2相当を自前実装し、ノイズシェーパ係数最適化に応用。極めて高度 |
| ディザ | Intel MKL VSLによるベクトル化TPDF | 教科書的に正しい高品質実装 |
| IRリサンプリング | r8brain（線形位相、阻止域140dB、遷移帯域2.0）でオフライン高品質変換 | マスタリンググレード |
| オーバーサンプリング | Kaiser窓ハーフバンドカスケード（最大3段, 1023/255/63 tap）, AVX2 FMA最適化 | 設計は標準的かつ正確。ただし全段が線形位相固定（後述） |
| 数値精度 | パイプライン全体が64bit doubleで統一。Audio Thread内はlibm禁止・AVX2手書き近似で代替 | レイテンシ・量子化誤差の両面で極めて堅実 |

上記を踏まえ、本提案は「未実装の機能ギャップ」と「既存実装の精度・厳密性を一段引き上げる改修」の両方を扱う。

---

## 1. EQ部の改善提案

### 1.1 バンド別ノンリニアサチュレーションへのADAA導入（エイリアシング対策）

**現状（`src\eqprocessor\EQProcessor.Processing.cpp`）**

`processBand()` / `processBandStereo()` 内で、各バンドのSVF出力 `output = m0*v0 + m1*v1 + m2*v2` に対し、

```cpp
if (saturation > 0.0) {
    const double oneMinusSat = 1.0 - saturation;
    output = output * oneMinusSat + fastTanhScalarOutput(output) * saturation;
}
```

という「dry/wetブレンド型」のサチュレーションが適用されている。`fastTanhScalarOutput` は

```
f(x) = x*(27+x²)/(27+9x²)   (|x|<4.5),  f(x)=sign(x)  (|x|≥4.5)
```

という tanh の Padé(3,2) 近似である。重要な点は、**この非線形写像はSVFの内部状態（ic1eq/ic2eq）の更新には関与していない**（状態更新は飽和前の `v1`,`v2` から計算される）ことである。つまりこれは数学的に**メモリレス非線形性**であり、Bilbao/Esqueda/Parker/Välimäki の標準的ADAA（Antiderivative Antialiasing）理論が直接適用可能な、教科書的なケースに当たる過去には、アンチエイリアシングを低減した非線形静的関数を離散化する効果的な手法が、非線形関数の不定積分に基づいて提案されている。

現在この段は、AudioEngine側で `oversamplingFactor > 1` の場合のみオーバーサンプル領域で実行され、エイリアシング抑制はユーザーが指定したOS倍率に依存している（`AudioEngine.Processing.DSPCoreDouble.cpp` 441行目以降、`oversampling.processUp()` の中にEQ処理全体が含まれる構造）。OS=1（オーバーサンプリング無効）設定時は、ブースト＋高Saturation設定で**無対策のメモリレス非線形性が直接ナイキスト折り返しを生む**。

**提案: 1次ADAA（差分商法）の導入**

ADAAの核は、サンプル `x_n, x_{n-1}` に対し

```
ADAA1(x_n, x_{n-1}) = (F(x_n) - F(x_{n-1})) / (x_n - x_{n-1})
```

（`|x_n - x_{n-1}|` が小さい場合は数値的相殺を避けるため `f((x_n+x_{n-1})/2)` にフォールバック — Parker/Holters 系の実装で標準的に行われる安全策）で `f(x)` を置き換える手法である。

ここで実務上の最大の障害は、真の `tanh` のADAAが**多重対数関数（dilogarithm, Li₂）を必要とし**、C/C++での実装が煩雑になる点であるtanh関数の不定積分には対数関数と多重対数関数（dilogarithm）が必要であり、ディログは手動実装が難しい関数だが、Pythonのscipyや、C/C++向けにはGitHub上のpolylogarithmライブラリに実装例がある。しかし**ConvoPeqは既に真のtanhではなくPadé(3,2)有理関数近似を使っている**ため、この障害は実質的に消える。`f(x) = x/9 + (8/3)·x/(x²+3)` と部分分数分解できるため、不定積分は初等関数で閉形式が得られる：

```
F(x) = x²/18 + (4/3)·ln(x²+3)          (|x| < 4.5 の区間)
F(x) = |x| + C_edge                     (|x| ≥ 4.5 のクリップ区間。
                                          C_edge は x=4.5 での連続性から
                                          F(4.5⁻)=F(4.5⁺) を解いて定数化。
                                          値は約 0.8201 — 実装時は constexpr で
                                          倍精度のまま再計算すること)
```

つまり**dilogarithmは不要で、対数1回の評価のみでADAAが成立する**。これはConvoPeq固有のtanh近似を選んだことの副産物として非常に都合が良い。

**実装上の注意（ConvoPeqの「Audio Thread内libm禁止」規約との整合）**

`EQProcessor.Coefficients.cpp` のコメントにある通り、本コードベースは「Audio Thread内でのlibm呼び出し禁止」を一貫したポリシーとしており（`std::tan`/`std::pow`はMessage Thread専用、`fastTanh`系もすべて多項式/有理関数近似でlibmを回避している）。ADAAは原理上 `std::log` を毎サンプル要求するため、素朴な実装はこの規約に抵触する。

対応策は2つある。

1. **ブロック単位ベクトル化（推奨）**: `processBand()` の既存のサンプル毎再帰（ic1eq/ic2eq更新）はそのまま維持しつつ、ループ内でのサチュレーション適用を「生の `output[n]` をスクラッチバッファに格納する」だけに変更する。ブロック処理完了後、`x2[n] = output[n]^2` をAVX2で計算し、`Intel MKL VML` の `vdLn`（または既にIPPを使っている箇所であれば `ippsLn_64f`）で `ln(x2[n]+3)` を**1回のベクトル呼び出しで一括計算**し、その後ADAA差分式を適用する。これはConvoPeqが既に`MKLNonUniformConvolver.cpp`で`vdMul`/`cblas_ddot`等をAudio Thread内で使用している前例（=「MKL呼び出しはOK、生libmスカラー呼び出しはNG」という実質的な規約）に完全に合致する。
2. **AVX2多項式近似のlnを自作**: MKLへの依存をEQProcessor.Processing.cppに追加したくない場合、指数部のビット抽出＋次数5程度のmin-max多項式補正による高速log近似（このコードベースの`besselI0`や`fastTanh`と同系統の手法）を自作する。相対誤差1e-6程度で十分（量子化誤差よりずっと小さい）。

いずれの場合も、`output[n]` と前サンプル `prevSaturatorInput`（バンド・チャンネルごとに新規追加する永続状態。`UltraHighRateDCBlocker`の`m_state[]`と同じ思想で、`filterState`配列に1要素追加するだけで実装できる）からADAA出力を計算し、最終的に既存の `output*(1-sat) + ADAA(output)*sat` のdry/wetブレンドへ差し替える。

**期待効果**: Bilbao et al. の報告では、tanh系メモリレス非線形性に対し1次ADAAだけで20–30dB程度のエイリアシング抑制が得られ、これは2×～4×オーバーサンプリングに匹敵する。ConvoPeqの場合、(a) OS=1運用時のユーザーに対して「タダで」エイリアシング対策を提供できる、(b) OS>1運用時も「ADAA＋低めのOS倍率」の組み合わせで「素のOS倍率を上げる」よりCPU効率よく同等以上の音質が得られる、という二重のメリットがある。

---

### 1.2 最終段ソフトクリッパー（`musicalSoftClipScalar`/`softClipBlockAVX2`）のエイリアシング対策

**現状（`src\audioengine\AudioEngine.Processing.DSPCoreDouble.cpp`）**

EQ・コンボルバーの後段に独立した「musical soft clip」ステージが存在する。これは閾値`threshold`、ニー幅`knee`、非対称性`asymmetry`を持つ smoothstep ブレンド型クリッパーで、リニア領域とtanh近似クリップ領域を3次エルミート（smoothstep: `t²(3-2t)`）で滑らかに混合する設計である。

ソースコード中のコメントに非常に重要な記述がある：

```cpp
// [P3] midVec事前平均化ブロックを完全削除
// このブロックは threshold レベルでのハードリミッティングを引き起こし、
// SoftClip本来の滑らかなKnee特性を損なっていた。
```

これは、過去にAVX2/スカラー間の不整合バグ（メモリ記録にある「`prevScalar`不整合」）を修正する過程で、**事前に試みられていた何らかの簡易アンチエイリアシング処理（隣接サンプル平均）が誤動作を理由に完全削除**されたことを示している。結果として、このクリッパーは現在**EQ部のサチュレーション同様、無対策のメモリレス非線形性**になっている。

**提案A（厳密解）**: 1.1と同じ枠組みでADAAを導出する。本関数はリニア領域・smoothstepブレンド領域・飽和（tanh近似）領域の3区間からなる区分多項式＋有理関数であり、各区間の不定積分は閉形式で求まる：
- リニア領域: 元々非線形性がないため対策不要（`F(x)=x²/2`、通常の積分）。
- ブレンド領域: `ks(t)=t²(3-2t)` は `t` の3次多項式であり、`t` 自体が `x` の1次関数なので、`ks` は `x` の3次多項式。これと「リニア成分」「クリップ成分（1.1と同型のPadé-tanh）」の積を含む式の不定積分は、多項式項＋対数項（`ln(arg²+...)`の形）の組み合わせで閉形式になる。
- 飽和領域: 1.1と同様 `F(x)=±x+C`。

導出はやや煩雑になるため、本書では戦略のみを示す。実装は次のいずれかが現実的である。

**提案B（実用解・推奨）**: 厳密ADAAの代わりに、このステージだけをローカルに2倍オーバーサンプリングする。`CustomInputOversampler` が既に持つ Kaiser窓ハーフバンドFIR設計（`prepareStage`/`interpolateStage`/`decimateStage`）をそのまま再利用し、わずか1段（taps=31程度で十分、最終クリッパー用途なら阻止域90dB程度で要件を満たす）の軽量half-bandを専用に1個追加するだけで実装できる。これにより、**グローバルなオーバーサンプリング設定（OS=1）の有無に関係なく**、最終段クリッパーのエイリアシングを常時抑制できる。計算コストは1段ハーフバンド（31 tap）×ブロック分のみで無視できるレベルである。

どちらを採るにせよ、**「過去に削除された誤った対策」を「正しい対策」に置き換える**ことは、サウンドクオリティ上のリグレッション解消として優先度が高い。

---

### 1.3 Mid/Side（M/S）EQ処理モードの追加

**現状**: `EQChannelMode` は `Stereo / Left / Right` の3値のみ（`EQProcessor.h` 55–60行目）。ステレオイメージに関わる帯域別処理（例：低域はM/S関係なくセンターに寄せ、高域だけサイド成分を持ち上げて広がりを作る、といったマスタリング定番のテクニック）が構造的に不可能。

**提案**: `EQChannelMode` に `Mid`, `Side` を追加し、バンドループの前後に

```
M = (L+R) * 0.5
S = (L-R) * 0.5
... 各バンドをmode別（Stereo/L/R/Mid/Side）に適用 ...
L = M + S
R = M - S
```

のエンコード/デコードを挟む。実装上は `processBlock` 用に既存の `parallelInputBuffer` 等と同様のスクラッチバッファ（M/S用に2本）を新規確保するだけで、既存のSerial/Parallel構造・バンドアクティブ判定ロジックとシームレスに統合できる。Mid/Side変換は完全に可逆な線形演算（位相・振幅特性に副作用なし）なので、追加によるリスクは極めて小さい。

---

### 1.4 ダイナミックEQ（マルチバンド・ダイナミクス）の追加

**現状**: ダイナミクス処理は `EQProcessor` 内の **ブロードバンドAGC**（`processAGC`、RMSエンベロープ追従、アタック0.2s/リリース2.0s）のみで、帯域別の動的ゲインは存在しない（コード全体を検索しても compressor/sidechain/multiband dynamics に該当する実装は確認できなかった）。

**提案**: 既存のAGCが既に持つ「エンベロープ追従（アタック/リリース時定数テーブル, `agcAttackCoeffTable`等）」のロジックをバンド単位に再利用し、各バンドのSVF出力レベルに応じてそのバンドの実効ゲイン（`m1`,`m2`、もしくは段後のゲイン係数）を動的に変調する「ダイナミックEQ」として拡張する。具体的には:

1. 各バンドに `threshold`, `ratio`, `attack`, `release`, `range`（最大減衰/増幅量）パラメータを追加。
2. バンド出力（または専用のサイドチェイン用ナローバンドフィルタ出力）のRMS/ピークを既存のAGC由来のエンベロープ追従ロジックで検出。
3. しきい値超過分を `ratio` で圧縮し、目標ゲインを `LinearRamp`（既存の`convo::LinearRamp`をそのまま使える）でスムージングしてバンドゲインに乗算。

これは既存コードの再利用率が高く（AGCの数学的枠組みをそのまま転用可能）、新規実装範囲が比較的小さい割に音質的価値（リミッターのようなブロードバンド処理より遥かに自然な周波数選択的ダイナミクス制御）が大きい。

---

### 1.5 True Peakリミッタ + LUFSラウドネスメータリングの追加

**現状**: コードベース全体を検索した結果、`LUFS` / `True Peak` / `Limiter` / `K-weighting` に該当する実装は確認できなかった。出力段にあるのは前述の「musical soft clip」（固定アルゴリズムのウェーブシェーパ）のみで、規格に基づくラウドネス管理・トゥルーピーク管理機構は存在しない。

**提案**:
- **True Peakリミッタ**: ITU-R BS.1770-4 Annex 2に規定される4倍オーバーサンプリング（ポリフェーズFIR補間）による"true peak"検出を実装し、検出値に基づくルックアヘッド・リミッティングを追加する。`CustomInputOversampler`の半帯域カスケード設計コード（Kaiser窓設計部分）はそのまま4倍補間フィルタの設計に転用可能であり、新規実装は「検出専用の軽量インターポレータ＋ルックアヘッドゲインリダクション」のみで済む。
- **LUFSメータリング**: EBU R128 / ITU-R BS.1770-4のKフィルタ（シェルビング＋ハイパスの2段Biquad）+ ゲーティングアルゴリズムによるIntegrated/Short-term/Momentary LUFS計測を追加する。Kフィルタの係数計算は既存の`OutputFilter`のBiquad設計パターン（RBJ Cookbook式の`makeLPF`/`makeHPF`と同型）を流用できる。

この2つは「マスタリング用スタンドアロンアプリ」という製品コンセプトに対して欠落している標準機能であり、優先度は高いと考える。

---

## 2. コンボルバー（畳み込みエンジン）部の改善提案

### 2.1 パーティション構成のGarcia/Wefers最適化

**現状（`src\MKLNonUniformConvolver.cpp` `SetImpulse()`）**: レイヤー構成は固定的な等比則で決定されている。

```cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
const int l1Part = l0Part * tailL1L2Mult;   // デフォルト ×8
const int l2Part = l1Part * tailL1L2Mult;   // デフォルト ×8
```

つまり3層・各層×8固定の幾何級数的パーティションスキームであり、`kL0MaxParts=32`, `kL1MaxParts=64` という固定上限もハードコードされている。これは実装が簡潔で堅牢という利点がある一方、IR長・ブロックサイズ・CPU予算に応じた**計算量最小化の観点では最適とは限らない**。

非均一分割畳み込みの計算コストを最小化する問題は、García (2002)が動的計画法（Viterbi的探索）で解く"Optimal Filter Partition"として定式化しており本ライブラリは均一分割、2段（double FDL）分割、および一般的な非均一分割をサポートし、引用論文で記述されているViterbiアルゴリズムを実装して最適な分割を求める、Wefers (2014)の博士論文ではリアルタイム室内音響オーラリゼーション向けにこれを一般化している。さらに近年の実装例（`zones_convolver`）では修正Garcia最適分割と時間分散変換を実装した非均一分割畳み込み(NUPC)方式により、追加レイテンシなしで可変ブロックサイズに対しても大きな負荷スパイクなしに単一スレッドで動作できる、ConvoPeqが既に実装している「L1/L2をpartsPerCallbackずつ分散処理してCPUスパイクを抑える」という設計思想と本質的に同じ方向性をさらに体系化している。

**提案**: 固定×8幾何級数の代わりに、IR長・目標レイテンシ・CPU予算（許容ピーク負荷）を入力としたGarcia型動的計画法（あるいは簡略版として、層数とその境界点を対数尺度で網羅探索する程度でも十分実用的な改善が見込める）でパーティション境界を決定するモードを追加する。特に**非常に長いIR（数秒～十数秒の大聖堂リバーブIRなど）**では、現行の固定スキームより総FFT演算数を削減できる可能性が高く、同じCPU予算でより長いIRを扱える、または同じIR長でCPU使用率を下げられる。

実装は既存の`Layer`構造体・`processLayerBlock()`をそのまま使い、`SetImpulse()`内のレイヤー境界計算ロジック（`l0Len`/`l1Len`/`l2Len`の決定部）のみを置き換える形で統合できるため、既存アーキテクチャへの侵襲は小さい。

### 2.2 "Air Absorption"モードの物理的精度向上（連続時間-周波数減衰への変更）

**現状（同ファイル 902–940行目）**: `tailMode==0`（Air Absorption）時、L1とL2の**レイヤー全体**に対し、`layerWeight`（L1=1.0, L2=1.6固定）でスケールした単一のガウス型HFダンピングを一括適用している：

```cpp
const double layerWeight = (li == 1) ? 1.0 : 1.6;
const double dampingCoeff = dampingBase * layerWeight;
...
const double hfTilt = std::exp(-dampingCoeff * fNorm * fNorm);
```

つまりHFダンピングの時間方向の粒度は「L1か、L2か」という**2段階の離散ステップ関数**でしかなく、各レイヤー内部の個々のパーティション（＝個々の時間区間）には一律に同じダンピングが適用される。

実際の大気吸収（air absorption）はISO 9613-1で規定されるように、周波数のほぼ2乗（古典吸収＋分子緩和吸収の合成）に比例して連続的に増大し、当然ながら時間（＝音源からの伝搬距離に相当するIRのタップ位置）に対しても連続的に効果が積算される現象である。現行の「2値ステップ」近似は、特に L1→L2 の境界（既定で約0.68秒@48kHz、`l1Part=l0Part*8`の累積）で**HFダンピングの不連続なジャンプ**を生み、IRの聴感上の自然さをわずかに損なう可能性がある。

**提案**: `layerWeight`を2値ではなく、**パーティション単位（あるいは時間軸上で連続的な減衰関数からサンプリングした値）**にする。具体的には、各パーティション `p`（レイヤー `li`）の時間オフセット `t_p = (l1Offset または l2Offset) + p * partSize` を計算し、

```
dampingCoeff(p) = dampingBase * g(t_p)     // g(t) は時間に対して滑らかに増加する関数
                                            // 例: g(t) = (t / tailStartSec)^β  (β は1.0前後)
```

としてパーティション毎に異なるHFダンピング係数を適用する。`applySpectrumFilter()`と同型のループ構造をそのまま使えるため実装コストは低く、聴感上のテール自然さの改善が期待できる。

### 2.3 既存AllpassDesigner資産のオーバーサンプラーへの転用（レイテンシ削減）

**現状**: `AllpassDesigner`（CMA-ES/AdaGradによる目標群遅延への2次オールパス縦列フィッティング）はIRのMixed Phase変換専用に使われている。一方、`CustomInputOversampler`の全フィルタ（Kaiser窓ハーフバンド）は`isLinearPhaseFIR = true`固定であり、`Preset::IIRLike`であってもタップ数を減らした**線形位相**FIRに過ぎない（真の最小位相/IIR的低レイテンシ特性ではない）。

**提案**: 既に社内に存在する「目標群遅延に対しオールパス縦列でフィッティングする」資産（`AllpassDesigner::design` / `designWithCMAES`）を転用し、ハーフバンドFIRの振幅応答を保持したまま群遅延を最小位相相当まで圧縮する設計オプションを`CustomInputOversampler`に追加する。具体的な手順：

1. 各ステージのKaiser窓係数から複素ケプストラム法（あるいは既存の`MklFftEvaluator`のFFT基盤を使ったHilbert変換）で最小位相版の振幅応答を保持したIRを得る。
2. その最小位相版と元の線形位相版の群遅延差を「ターゲット群遅延」として`AllpassDesigner`に渡し、既存のオールパス縦列設計ロジックをそのまま適用する。
3. ステージ毎に「Linear Phase（現状）／Minimum Phase（新規）／その中間のMixed Phase」を選択可能にする。

これにより、**新規アルゴリズムを1行も書かずに**（既存のMixed Phase基盤をオーバーサンプラーに転用するだけで）、オーバーサンプリング由来のレイテンシを大幅に削減できる可能性がある（線形位相ハーフバンド3段カスケードのレイテンシは合計タップ数の半分に達するため、1023+255+63タップ構成では相当な遅延になる）。プリアンプ的なリアルタイム用途や低レイテンシモニタリングが重要なユーザーに対する差別化要素になる。

---

## 3. 入力処理段の改善提案

### 3.1 ハードクランプからソフトニーリミッティングへ（`InputBitDepthTransform.h`）

**現状**: `sanitizeAndLimit()` はNaN/Inf/デノーマル除去と同時に、入力を無条件に**`std::clamp(v, -1.0, 1.0)`でハードクリップ**している。これはNaN/Inf耐性という観点では正しい防御だが、0dBFSをわずかに超えるピーク（インターサンプルピークや、上流プラグインからの僅かなオーバーシュート等、実務ではしばしば発生する）に対しては、**角の立ったハードクリップによる可聴な歪み**を生む。

**提案**: NaN/Inf/デノーマル除去の論理はそのまま維持し、`[-1,1]`の単純クランプの代わりに、閾値付近（例: ±0.98以上）でのみ作用する短いニーを持つソフトサチュレーション（1.2節のクリッパーと同系統の数式を再利用可能）に置き換える。これにより、異常値に対する安全性は変えずに、僅かなオーバーシュートに対する音質劣化を低減できる。

---

## 4. ノイズシェーピング/ディザリング部の改善提案

`NoiseShaperLearner` / `MklFftEvaluator` の心理音響モデルは、Bark尺度24帯域・MPEG Annex D型拡散関数・Terhardt式ATH・A重み相当の`bandWeightForHz`を統合した、実質的にMP3心理音響モデル2級の精緻な実装であり、大規模な変更は不要と判断する。1点のみ、明確に改善余地がある箇所を指摘する。

### 4.1 JND重み付け曲線のISO 226等価ラウドネス曲線化

**現状（`src\MklFftEvaluator.h` `computeJndDb`）**:

```cpp
const double lowPeak = kJndLowPeak * std::exp(-0.5 * (f - 0.5) * (f - 0.5));   // 0.5kHz付近にガウシアンピーク
const double highShape = kJndHighSlope * (f - 3.0) * (f - 3.0);                // 3kHz基準の二次関数
```

これは人間の聴覚感度（中域で鋭敏、低域・高域で鈍感）を模した**ヒューリスティックなガウシアン/二次関数近似**であり、ATH（`computeAthSplDb`、Terhardt 1979式）やA重み（`bandWeightForHz`）が標準規格に基づく一方、このJND項だけは恣意的な定数（`kJndLowPeak=1.0`, ピーク0.5kHz, `kJndHighSlope=0.2`、基準3.0kHz）に依存している。

**提案**: このJND項を、ISO 226:2003で規定される等価ラウドネスレベル曲線（再生想定SPL、例えば`kReferenceSplDb=90.0`で既に定義されている基準レベル）から導出した重みに置き換える。等価ラウドネス曲線は周波数ごとの「同じラウドネスに感じるために必要なSPL」を与えるため、その勾点（周波数に対する微分）が「その周波数での量子化雑音の可聴性の鋭敏さ」に直接対応する、より理論的根拠のある重み付けになる。具体的な再生レベル（90dB SPL基準は既存コードに既にあるので、ISO 226のその近傍の等高線データを補間して使えばよい）を使うことで、現状の「中域だけ鋭敏」という単純化されたヒューリスティックより、実際の聴感に近い学習目標関数になることが期待される。

---

## 5. 実装優先度マトリクス（提案者所感）

| # | 提案 | 想定インパクト | 実装コスト | 既存資産再利用度 |
|---|---|---|---|---|
| 1.1 | EQバンドサチュレーションのADAA化 | 中〜高（OS=1時に特に大） | 低〜中 | 高（既存fastTanh近似をそのまま閉形式積分） |
| 1.2 | 最終段クリッパーのエイリアシング対策 | 中 | 低（提案B採用時） | 高（既存Oversamplerコード転用） |
| 1.3 | Mid/Side EQ | 中（ワークフロー価値大） | 低 | 高 |
| 1.4 | ダイナミックEQ | 高 | 中 | 高（AGCの枠組み再利用） |
| 1.5 | True Peak Limiter + LUFS | 高（製品完成度として） | 中 | 中 |
| 2.1 | パーティション最適化（Garcia/Wefers） | 中（長尺IR時のCPU効率） | 中〜高 | 高 |
| 2.2 | Air Absorption連続化 | 低〜中（聴感の僅かな自然さ） | 低 | 高 |
| 2.3 | オーバーサンプラー最小位相化 | 中（レイテンシ重視ユーザー） | 中 | 非常に高（AllpassDesigner転用） |
| 3.1 | 入力段ソフトニー化 | 低〜中（エッジケース耐性） | 低 | 高 |
| 4.1 | JND曲線のISO226化 | 低〜中（学習品質の微改善） | 低 | 高 |

優先順位としては、**1.1（ADAA、自社近似式の特性を活かせる低コスト施策）→ 1.3（M/S, 低コスト高価値）→ 1.5（True Peak/LUFS, 製品完成度）→ 1.4（Dynamic EQ）→ 2.3（最小位相オーバーサンプラー）→ その他**、という順で着手するのが投資対効果として合理的と考える。

---

## 6. 参考文献

- Vadim Zavalishin, *The Art of VA Filter Design*（TPT SVFの理論的基盤。既存EQ実装が参照）
- Stefan Bilbao, Fabián Esqueda, Julian D. Parker, Vesa Välimäki, "Antiderivative Antialiasing for Memoryless Nonlinearities," *IEEE Signal Processing Letters*, Vol. 24, No. 7, July 2017
- Julian D. Parker, Vadim Zavalishin, Efflam Le Bivic, "Reducing the Aliasing of Nonlinear Waveshaping Using Continuous-Time Convolution," DAFx-16, Brno, 2016
- Martin Holters, "Antiderivative Antialiasing for Stateful Systems," DAFx-19
- Martin Vicanek, "Note on Alias Suppression in Digital Distortion"（有理関数近似のADAA実装上の数値安定性に関する実務的知見）
- Guillermo García, "Optimal Filter Partition for Efficient Convolution with Short Input/Output Delay," *Journal of the Audio Engineering Society*, 2002
- Frank Wefers, "Partitioned Convolution Algorithms for Real-Time Auralization," Ph.D. thesis, RWTH Aachen, 2015
- ISO 9613-1:1993, *Acoustics — Attenuation of sound during propagation outdoors — Part 1: Calculation of the absorption of sound by the atmosphere*
- ISO 226:2003, *Acoustics — Normal equal-loudness-level contours*
- ITU-R BS.1770-4, *Algorithms to measure audio programme loudness and true-peak audio level*
- EBU R128, *Loudness normalisation and permitted maximum level of audio signals*
- E. Terhardt, "Calculating Virtual Pitch," *Hearing Research*, 1979（既存ATH実装が参照する近似式の出典）
- ISO/IEC 11172-3 Annex D（MPEG-1 Audio Layer I/II 心理音響モデル2。既存マスキング拡散関数の出典）

---

*本書はソースコードの静的解析に基づく分析であり、提案の妥当性は実機でのA/Bリスニングテスト・自動回帰テスト（既存`src\tests\`配下のISR系コントラクトテストとは独立した、DSP出力波形/スペクトルの回帰テストを別途整備することを推奨）による検証を前提とする。*





ConvoPeq.mdの最新版を確認した上で評価すると、この提案書は**DSP理論の理解はかなり深いものの、ConvoPeqの実装実態との整合性にばらつきがあります。**

全体評価を先に示します。

| 項目           | 評価      |
| ------------ | ------- |
| DSP理論の正確性    | A       |
| 文献引用の妥当性     | A       |
| ConvoPeq実装理解 | B+      |
| 実装コスト見積り     | C+      |
| 投資対効果評価      | B       |
| 実際に採用すべき割合   | 約30～40% |

特に、

* ADAA
* True Peak/LUFS
* Mid/Side EQ

は検討価値があります。

一方、

* Dynamic EQ
* Garcia最適パーティション
* オーバーサンプラ最小位相化

は、ConvoPeqの現在の設計思想とかなり衝突します。

以下、項目ごとに詳細に検証します。

---

# 1.1 ADAA導入

結論：

**理論的には正しい。**
しかし提案書は実装コストを過小評価しています。

---

まず提案書の前提、

> EQのサチュレーションはメモリレス非線形

これは正しいです。

実際にConvoPeqでは、

```cpp
output = output * oneMinusSat
       + fastTanhScalarOutput(output) * saturation;
```

型の後段ウェーブシェーパになっています。

また状態変数

```cpp
ic1eq
ic2eq
```

へのフィードバックには非線形が入っていません。

従ってBilbaoらのADAA対象です。 ([Edinburgh Research][1])

---

さらに提案書の

```text
Padé近似だからLi₂不要
```

も数学的には正しいです。

提案されている

```text
x*(27+x²)/(27+9x²)
```

は有理関数なので、

不定積分は

```text
x²
ln(x²+3)
```

だけで閉じます。

これは良い着眼点です。

---

ただし問題があります。

ConvoPeqは

```cpp
NUM_BANDS = 20
```

です。 

さらに

* Stereo
* Parallel/Serial
* 最大8x OS

があります。

ADAAを各バンドへ導入すると

* 前サンプル保持
* 分岐
* log近似

が追加されます。

特に

```cpp
Audio Thread内libm禁止
```

ポリシーが存在します。 

そのため

> MKL VML vdLnで解決

という提案は少し楽観的です。

なぜならEQ処理は現在サンプル逐次再帰であり、

ADAAだけブロック後処理にすると

キャッシュ構造が変わります。

---

私の評価：

### 採用価値

高い

### 優先度

中

### 実装難易度

提案書の「低～中」ではなく

**中～高**

---

# 1.2 SoftClipへのADAA/局所OS

結論：

**こちらの方がADAAより優先度が高い**

---

理由。

EQサチュレーションは

* 20バンドに分散
* 使用者限定

です。

一方SoftClipは

出力直前で常時動作します。

---

提案書の

> 2倍局所オーバーサンプリング

はかなり現実的です。

ConvoPeqは既に

```cpp
CustomInputOversampler
```

を持っています。 

また

```cpp
31 tap
```

級HalfbandならCPU負荷は非常に小さいです。

---

私の評価

### 妥当性

非常に高い

### 実装リスク

低い

### 優先度

高

---

# 1.3 Mid/Side EQ

結論：

**最も費用対効果が高い改善案**

---

現状のConvoPeqは

```cpp
Stereo
Left
Right
```

しかありません。 

また実際の処理側も

```cpp
Stereo
Left
Right
```

前提です。 

---

M/S追加は

DSP的には

```text
M=(L+R)/2
S=(L-R)/2
```

のみ。

音質リスクもありません。

---

マスタリング用途では

非常に価値があります。

FabFilter Pro-Q
DMG Equilibrium
TDR Nova GE

など上位製品では標準機能です。

---

評価

### 妥当性

極めて高い

### 優先度

最優先候補

---

# 1.4 Dynamic EQ

結論：

提案書は実装コストを大幅に過小評価しています。

---

Dynamic EQは

単なる

```text
AGC流用
```

ではありません。

必要になるのは

* バンド毎検波器
* Attack/Release
* Knee
* Ratio
* Lookahead
* GUI
* Preset保存

です。

20バンドEQなら

実質

```text
20個のコンプレッサ
```

です。

---

ConvoPeqの主目的は

```text
Convolution + Parametric EQ
```

です。

Dynamic EQまで入れると製品の方向性自体が変わります。

---

評価

### 技術的妥当性

高い

### 投資対効果

低い

### 優先度

低

---

# 1.5 True Peak + LUFS

結論：

非常に妥当

---

これは提案書で最も納得感があります。

実際にコードベースには

* LUFS
* BS1770
* True Peak

が見当たりません。

提案書の指摘は概ね正しいです。

---

BS.1770は

* K-weighting
* gating
* integrated loudness
* short-term
* momentary

を定義しています。 ([mediaspecs.rtl.de][2])

True Peakも4倍OS測定が標準です。 ([mediaspecs.rtl.de][2])

---

ConvoPeqは

明らかに

```text
マスタリング用途
```

を意識した設計なので、

この機能の欠落は確かにあります。

---

評価

### 妥当性

非常に高い

### 優先度

高

---

# 2.1 Garcia/Wefers最適化

結論：

理論的には正しいが優先度は低い

---

提案書は

Garcia
Wefers

の研究内容を正しく引用しています。

ただし。

ConvoPeqの現状は

既に

* 非均一分割
* 3層
* 時間分散

を実装しています。

提案書自身も認めている通り、

業務用レベルです。

---

長尺IRで数％改善する可能性はあります。

しかし

ISR Bridge
RCU
World再構築

ほど価値はありません。

---

評価

低優先

---

# 2.2 Air Absorption

結論：

正しいが聴感差は極小

---

これはDSP研究者が好む改善です。

しかし実運用では

ABXで区別できるかかなり怪しいです。

---

優先度は非常に低いです。

---

# 2.3 最小位相オーバーサンプラ

結論：

提案書最大の問題点

---

提案書は

```text
AllpassDesignerを転用
```

と言います。

しかしConvoPeqでは

オーバーサンプラ遅延計算が

```cpp
static_assert(CustomInputOversampler::isLinearPhaseFIR)
```

前提です。 

また

```cpp
static constexpr bool isLinearPhaseFIR = true;
```

がコードに固定されています。 

---

つまり現在のレイテンシ管理は

```text
線形位相
対称FIR
```

を前提に構築されています。

---

ここを崩すと

* PDC
* レイテンシ表示
* ISR Bridge再構築

まで影響します。

---

評価

### DSP理論

○

### ConvoPeq適合性

×

---

# 3.1 InputBitDepthTransform

結論：

優先度低

---

提案内容は理解できます。

しかし

入力クランプは

「音質処理」

ではなく

「防御コード」

です。

---

ConvoPeqの思想からすると

```text
異常入力は切り捨てる
```

方が合理的です。

---

私は変更不要と考えます。

---

# 4.1 ISO226 JND

結論：

研究テーマとしては面白い

しかし実用効果は不明

---

提案書は

現在のJNDモデルが

ヒューリスティック

であることを正しく指摘しています。

しかし。

NoiseShaperLearnerは既に

* Bark
* ATH
* Tonal/Noise masker
* Spread function

を持っています。

提案書自身も認めています。

---

そのため

JNDだけISO226へ変えても

改善量はかなり小さいと思われます。

---

# 総合結論

私がConvoPeq開発者なら優先順位は次になります。

| 優先度 | 提案                |
| --- | ----------------- |
| S   | Mid/Side EQ       |
| S   | True Peak Meter   |
| A   | LUFS Meter        |
| A   | SoftClip局所2x OS   |
| B   | EQサチュレーションADAA    |
| C   | Garcia最適化         |
| C   | Air Absorption連続化 |
| D   | Dynamic EQ        |
| D   | ISO226 JND        |
| E   | オーバーサンプラ最小位相化     |

特に最新版ConvoPeqを見る限り、現在の最大テーマは依然として **Practical Stable ISR Bridge Runtime の完成度維持とランタイム堅牢性** です。提案書の中でその前提を崩さず導入できるものは、

1. Mid/Side EQ
2. True Peak/LUFS
3. SoftClip局所OS
4. ADAA

の4件です。

それ以外はDSPとしては興味深いものの、現行アーキテクチャに対するリスクや開発コストの方が大きいと評価します。

[1]: https://www.research.ed.ac.uk/en/publications/antiderivative-antialiasing-for-memoryless-nonlinearities/?utm_source=chatgpt.com "Antiderivative antialiasing for memoryless nonlinearities - University of Edinburgh Research Explorer"
[2]: https://mediaspecs.rtl.de/glossar/itu-r-bs1770-4?utm_source=chatgpt.com "Glossar ITU-R BS.1770-4 - RTL Support Portal"
