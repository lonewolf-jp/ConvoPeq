# ConvoPeq コンボルバー実装 — 妥当性検証レポート

**対象**: `lonewolf-jp/ConvoPeq` — 添付 `ConvoPeq.md`（連結ソース、約67,600行）
**検証範囲**: コンボルバーサブシステム一式
（`MKLNonUniformConvolver`, `ConvolverProcessor.*`, `AllpassDesigner`, `IRConverter`/`IRDSP`, `AlignedAllocation`, `PreparedIRState` 等）
**方法**: ソースコードの全文精査（該当ファイルを行単位で抽出・通読） + 音響工学／DSP文献の調査（Gardner 1995, Garcia 2002, Wefers 2015 等）との対比検証

---

## 0. 要旨（エグゼクティブサマリー）

ConvoPeqのコンボルバーは、**Gardner型 非一様分割畳み込み（Non-Uniform Partitioned Convolution, NUC）**を、Overlap-Save法・IPP FFT・AVX2 SIMD複素MAC・RCUベースのIRホットスワップという現代的なスタックの上に正しく実装したものである。コア畳み込み数学（オーバーラップセーブの境界条件、FDLとIRパーティションの内積、レイヤー分割の遅延予算）は**文献と整合し、検証可能な誤りは見つからなかった**。

一方で、コア畳み込みの「外側」にあるサブシステム（Mixed-Phase IR設計、r8brainリサンプリング、レガシーなサイジング計算）には、**実際に値を追跡することで初めて発見できる具体的な不整合**が複数存在する。以下はその一覧（詳細は本文）。

| # | 箇所 | 種別 | 確信度 | 一言で |
|---|---|---|---|---|
| F1 | `ConvolverProcessor.MixedPhase.cpp` | **設計の方向性が逆転している疑い** | 高 | Mixed Phaseのデフォルト挙動が「低域=Linear / 高域=Minimum」になっており、文献の確立された慣行（低域=Minimum / 高域=Linear）と逆 |
| F2 | `ConvolverProcessor.MixedPhase.cpp` | デッドパラメータ | 確定（メモリ記載済み事項を本検証でも再確認） | `mixedPreRingTau` がキャッシュキー以外で一切使われず、音には無関係 |
| F3 | `IRDSP.cpp` / `IRDSP::resampleIR` | バッファマージン不足 | 中〜高（r8brain仕様から論理的に導出） | `+2.0` サンプルしかマージンを取らずに r8brain の内部レイテンシをフラッシュしており、フィルタが長くなる設定（140dB/2%）では尾部切り捨てのリスクがある |
| F4 | `ConvolverProcessor.Internal.h::computeMasteringSizing` | デッドコード（無効化された配線） | 確定 | `firstPartition`/`maxFFTSize` を計算して保持するが、`SetImpulse()` はこれらを受け取らないため実際のNUC構成に一切影響しない |
| F5 | `AllpassDesigner::applyAllpassToIR` | 未使用コード | 確定 | 公開APIとして定義されているが呼び出し箇所が無く、Mixed-Phase経路は同等処理をインライン実装で重複している |
| F6 | レイヤースケジューリング（L0/L1/L2, 固定倍数=8） | 設計判断（バグではない） | — | Gardner流の実用的ヒューリスティックであり、Garcia/Wefersの動的計画法による厳密最適スケジュールではない（性能上は妥当な簡略化） |

コア畳み込みエンジン自体（F1〜F6に含まれない部分）は、**設計判断として極めて手堅く、AVX2実装・FDL構成・スレッド安全性のいずれにも実装上の欠陥は確認できなかった**。

---

## 1. アーキテクチャ全体像

```
入力 ─┬─→ [Dry経路: 遅延補償バッファ + Catmull-Rom分数遅延補間] ─┐
      │                                                          ├→ Equal-Power Mix → 出力
      └─→ [Wet経路: StereoConvolver::process()]                 ┘
                ├─ ch0: MKLNonUniformConvolver (L0/L1/L2)
                └─ ch1: MKLNonUniformConvolver (L0/L1/L2)

IRロード経路（Message Thread / LoaderThread, 非リアルタイム）:
  ファイル読込(IRConverter::loadAudioFile)
    → リサンプリング(IRDSP::resampleIR, r8brain)
    → 長さ決定/トリム/非対称Tukeyウィンドウ
    → [PhaseMode] AsIs / Minimum / Mixed(AllpassDesigner + CMA-ES)
    → スケール係数計算(IRConverter::computeScaleFactor)
    → finalizeNUCEngineOnMessageThread()
        → StereoConvolver::init() → MKLNonUniformConvolver::SetImpulse() ×2ch
    → RCU経由でAudio Threadへ公開(エポックベース遅延解放)
```

1チャンネルあたり1つの `MKLNonUniformConvolver` インスタンスを使い、ステレオは完全に独立した2エンジンで処理する（`StereoConvolver` がラップ）。IRの読み込み・差し替えはMessage Thread / LoaderThreadで行い、Audio Threadへは**RCU（epoch-based reclamation）**を介してロックフリーに公開される。Audio Thread内でのメモリ確保は一切行わない設計（`SetImpulse()` 時点でワークバッファを事前確保）であり、これはリアルタイムオーディオの基本要件（malloc-freeなオーディオコールバック）を満たしている。

---

## 2. コア畳み込みエンジン（`MKLNonUniformConvolver`）の検証

### 2.1 アルゴリズム分類: Gardner型 非一様分割畳み込み

`MKLNonUniformConvolver.h` のコメント、および実装から判断すると、本エンジンは以下の3層構成を取る：

- **L0（即時層）**: `partSize = nextPow2(max(blockSize, 64))`。ホストのI/Oブロックサイズに直接追従し、**毎コールバックで全パーティションを処理**する低レイテンシ層。
- **L1**: `partSize_L1 = partSize_L0 × tailL1L2Multiplier`（既定値8、UI範囲2〜16）。コストをコールバック間で分散処理（amortize）する層。
- **L2**: `partSize_L2 = partSize_L1 × tailL1L2Multiplier`。さらに大きいパーティションで残りのIRテイルを処理。

これは William G. Gardner の "Efficient Convolution without Input-Output Delay" (J. Audio Eng. Soc. 43(3), 127-136, 1995) で提案された、**小さいブロックを直接畳み込みに近い形で即時処理し、大きいブロックの計算をコールバック間に分散させることでI/Oレイテンシをゼロに保つ**というハイブリッド畳み込み手法の直接の系譜にある。Frank Wefers の博士論文 *Partitioned convolution algorithms for real-time auralization* (RWTH Aachen, 2015) も、この方式を "Gardner's partitioning scheme"（6.4.6節）として分類しており、ConvoPeqの3層構成はこの記述と整合する。

各レイヤーは内部的に **Overlap-Save法**（FFTサイズ = パーティションサイズ×2、すなわち50%ゼロ詰め）で実装されている。これはStockham (1966) の "High-speed convolution and correlation" に始まる標準的構成であり、円状畳み込みによる時間領域エイリアシングを避けるための条件（FFTサイズ ≥ 入力ブロック長 + フィルタブロック長 − 1）を満たしている。

### 2.2 FDL（Frequency-Domain Delay Line）の「二重バッファ線形化」トリックの数学的検証

`MKLNonUniformConvolver.cpp` の `processLayerBlock()` 内で使われている最適化を検証した。実装は、サイズ `numParts` の循環FDLバッファを**2倍（`2×numParts`）に確保し、新しいブロックを書き込む際に同じデータを `fdlIndex` と `fdlIndex+numParts` の両方に書き込む**（ミラー書き込み）。さらにIRパーティション配列を**事前に逆順に並べ替える**（`SetImpulse()` 内、コメント「[最適化2] IRパーティションを逆順に並び替える」）。

これにより、内積計算ループが
```
for p in 0..numPartsIR-1:
    acc += FDL[ fdlIndex - numPartsIR + 1 + numParts + p ] × IR_reversed[p]
```
という**剰余演算（mod）を一切使わない単純な前方走査**で実現できる。

筆者はこれを式変形によって検証した。求めたい量は標準的なブロック畳み込みの式
`Y[n] = Σ_{p=0}^{P-1} X[n-p] · H[p]`
である。IR配列を反転した結果、`IR_reversed[p]` は元の `H[P-1-p]` に等しく、対応する `FDL` 参照位置は `fdlIndex-(P-1-p)`、すなわち `fdlIndex-P+1+p` で**pに対して単調増加**となる。これに `numParts` を加算したオフセット（`linStart`）は、`fdlIndex∈[0,numParts-1]`、`numPartsIR(=P) ≤ numParts` の制約下で常に `[0, 2·numParts-1]` の範囲に収まることを確認した（境界値を両端で計算し検証済み）。ミラー書き込みにより `[0,numParts)` と `[numParts,2numParts)` は常に同一内容を保持するため、この「2倍バッファ＋オフセット読み出し」は循環インデックスの剰余演算と数学的に等価である。

**結論**: このトリックは正しい。文献上の specific な出典は確認できなかったが（Gardner論文、Wefers論文ともにFDL実装の具体的なメモリレイアウト最適化までは踏み込んでいない）、メモリを2倍消費して分岐/剰余を除去するというトレードオフは、SIMDの内側ループにおける一般的な高速化パターンとして妥当である。

### 2.3 SIMD実装（AVX2 split-complex MAC）の検証

複素積和（`(ar+i·ai)(br+i·bi)` の蓄積）をAVX2でSoA（Structure of Arrays、実部/虚部を別配列）形式で処理しており、これはAoS（実部・虚部を交互配置）形式に比べてシャッフル命令を排除でき、4要素同時処理（256bit÷64bit=4 doubles）が素直に効く構成である。IR側スペクトルは「マスター表現（AoS、IFFT用）」と「演算用ワーク表現（SoA）」を併存させる設計になっており、`SetImpulse()` 時に一度だけ変換コストを払うことで、Audio Thread側のホットループからは変換コストを排除している。これは妥当な設計である。

FFTプランは `IppFFTPlanCache`（静的・プロセス全体で共有）経由でオーダー（サイズ）ごとにキャッシュされ、`fftSpec`（読み取り専用記述子）はチャンネル間で共有しつつ、可変な `fftWorkBuf` はレイヤーごとに個別確保されている。IPPのFFT記述子は一般にスレッドセーフな読み取り専用ディスクリプタとワークバッファを分離する設計のため、この共有方式自体は妥当である。

### 2.4 レイヤースケジューリング（固定倍数=8）の妥当性 — Garcia/Wefers文献との比較

`tailL1L2Multiplier`（既定値8、UI範囲2〜16）による**固定比率の幾何級数的レイヤー成長**は、計算コストの観点では厳密に最適ではない。

Guillermo Garcia の "Optimal Filter Partition for Efficient Convolution with Short Input/Output Delay" (AES 113th Convention, 2002) は、**動的計画法（Viterbiアルゴリズム）を用いて、指定された入出力遅延とフィルタ長に対する計算コストを厳密に最小化する非一様パーティション**を導出する手法を提示している。同論文（および対応する米国特許 US6625629 のクレーム記述）によれば、最適解は単純な「毎段倍々」ではなく、**多くの場合パーティションサイズの遷移が2倍ではなく4倍以上になる**ことが示されている（"often the optimal partition does not include transitions to blocks twice as long (but four times as long or greater)"）。これは、ConvoPeqが採用する「固定倍数8」という比較的大きい成長率が、最適解の傾向（小さい倍率を避け大きく飛ぶ）と**方向性としては整合する**ことを示唆する。

一方、Wefers (2015) は第6章で非一様分割畳み込みを "Gardner's partitioning scheme"（基本ヒューリスティック、6.4.6節）と "Optimized filter partitions"（最適化問題として解く方式、6.5節、"Minimal-load partitions" と "Practical partitions" に細分）に明確に分けている。ConvoPeqの実装は前者（Gardner流の実用的ヒューリスティック）に該当し、ハードウェア固有のベンチマークに基づくコストモデルを用いた厳密最適化（Garcia/Wefersが提示するクラス）には踏み込んでいない。

**結論**: これは実装上の誤りではなく、**意図的な簡略化として妥当な設計判断**である。3層・固定倍数というシンプルな構成は実装・デバッグ・パラメータ調整の容易さで優れるが、理論上の下限（厳密最適スケジュール）に対しては計算コストに若干の余裕（slack）を残している可能性がある。これは「ユーザのメモリに記録されている既存の調査テーマ（García/Wefers最適スケジューリング）」を裏付け、文献的な裏付けを与えるものである。

### 2.5 〔新規発見・F4〕`computeMasteringSizing()` のデッドコード化

`ConvolverProcessor.Internal.h` の `computeMasteringSizing(internalBlockSize, irLength)` は、`firstPartition = clamp(nextPow2(internalBlockSize×4), 4096, 16384)` と `maxFFTSize` を計算し、`LoaderThread.cpp`（`buildConvolverFromTrimmed`, 223行目）と `Lifecycle.cpp`（221行目）の両方で呼び出されている。

しかし、これらの値が実際に消費される先を追跡すると：

```
firstPartition/maxFFTSize
  → buildConvolverFromTrimmed() の引数
  → initializeConvolverSynchronously() / queueFinalizeOnMessageThread()
  → StereoConvolver::init(..., maxFFTSize, knownBlockSize, firstPartition, ...)
  → storedMaxFFTSize / storedFirstPartition に保存
  → StereoConvolver::clone() で再度同じ値を init() に渡すのみ
```

`MKLNonUniformConvolver::SetImpulse(impulse, irLen, blockSize, scale, enableDirectHead, filterSpec)` の**シグネチャには `firstPartition` も `maxFFTSize` も存在しない**。実際にL0のパーティションサイズを決めるのは `blockSize`（= `knownBlockSize` = 実ホストブロックサイズをpower-of-2化した値）のみであり、`firstPartition`/`maxFFTSize` はクローン操作の引き継ぎ用に保存されるだけで、**畳み込みエンジンのFFTサイズ・パーティション数・レイテンシのいずれにも一切影響しない**。

検証のため全243ファイル中でこれらのフィールドの参照箇所を網羅的に検索したが、`ConvolverProcessor.h` / `ConvolverProcessor.Internal.h` / `ConvolverProcessor.Lifecycle.cpp` / `ConvolverProcessor.LoaderThread.cpp` の4ファイル以外に参照は無く、上記の配線で完結していることを確認した。おそらく旧アーキテクチャ（NUCがこれらの値を直接受け取っていた版）からのリファクタリング残骸と思われる。**動作上の不具合は生じないが、計算（`computeMasteringSizing` のロジック自体、log2/clamp等）が完全に無駄であり、保守者が「ここでFFTサイズが決まっている」と誤解するリスクがある。**

---

## 3. Mixed-Phase IR設計（`AllpassDesigner` / `ConvolverProcessor.MixedPhase.cpp`）の検証

### 3.1 群遅延式の数学的検証 — 正しい

`AllpassDesigner::sectionGroupDelayRhoTheta()` は、極半径ρ・極角度θで表現された2次オールパスセクション（共役極ペア `ρe^{±jθ}`）の群遅延を

```
τ(ω) = (1-ρ²)/(1-2ρcos(ω-θ)+ρ²) + (1-ρ²)/(1-2ρcos(ω+θ)+ρ²)
```

として計算している。これは実係数2次オールパス（共役極ペアを持つ）の群遅延に関する標準的な閉形式であり（単一実極の群遅延式 `(1-r²)/(1-2r cos(ω-θ)+r²)` を共役対の両方について加算した形）、Oppenheim & Schafer *Discrete-Time Signal Processing* 等の教科書に準拠する。**数式・実装ともに正しい。**

### 3.2 設計手法の文献的位置づけ

`AllpassDesigner::design()`（Greedy + AdaGrad）および `designWithCMAES()` は、**目標群遅延カーブにオールパスカスケードをフィッティングする**という手法を取っている。これは古典的には A. G. Deczky の "Synthesis of recursive digital filters using the minimum p-error criterion" (IEEE Trans. Audio Electroacoust., 1972) に始まる**オールパス群遅延等化（allpass group-delay equalization）**の系譜にあり、現代的にはルームコレクション製品（Dirac Live、Audiolenseなど）が「Mixed Phase」フィルタを構成する際に用いる手法と概念的に同種である。CMA-ESという進化計算による最適化を採用している点は新規性があるが、設計問題そのもの（離散周波数点での目標群遅延への適合）は文献的に確立されたアプローチである。

### 3.3 〔新規発見・F1・要確認〕周波数クロスオーバー方向が文献の確立された慣行と逆転している疑い

`convertToMixedPhaseAllpass()`（`ConvolverProcessor.MixedPhase.cpp`, 303-311行目）の重み付けは：

```cpp
double wLinear = 1.0;
if (freq >= transitionHiHz)      wLinear = 0.0;
else if (freq > transitionLoHz)  wLinear = 0.5*(1+cos(π·x));   // raised-cosine crossfade
const double wMinimum = 1.0 - wLinear;
```

これは **`freq ≤ transitionLoHz`（既定200Hz）で `wLinear=1`（完全Linear Phase）、`freq ≥ transitionHiHz`（既定1000Hz）で `wLinear=0`（完全Minimum Phase）** と読める。つまり、**低域はLinear Phase、高域はMinimum Phase** という構成である。

これは、ルームコレクション／スピーカーEQ分野で確立されている「Mixed Phase」設計の慣行とは**逆**である。調査した複数の音響工学的文献・専門資料：

- Dirac Research の技術解説 *"On Room Correction and Equalization of Sound Systems"* は、ミックスフェーズ補正において **零点ごとの慎重な反転判断**を行いつつ、200Hz付近を境界として挙動が変化する例を示し、低域での不適切なMinimum-Phase反転がリンギングを生むリスクを論じている。同時に、リニアフェーズ補正のプリリンギングが「ピークから-60dB」程度に収まるよう設計する必要性を述べている。
- ホームオーディオ／プロオーディオの実務解説（HomeAudioFidelity、AVNirvana、Audio Science Reviewのフォーラム議論等）は一致して、**「プリリンギングは自然界に存在しない（因果律違反的な)アーティファクトであり、特に低域での長時間プリリンギングはトランジェントを著しくスメアする」**ため、**低域はMinimum Phase（プリリンギング無し）、高域はLinear Phase（プリリンギングが極めて短時間で聴覚の時間分解能以下になるため実質無害）**にする、という設計指針を共有している。
- この理由は、線形位相フィルタのプリリンギング持続時間が、要求される遮断特性（dB/oct）を実現するために必要なフィルタ長（＝波長に比例する周期数）にスケールするためであり、低い周波数ほど同じ急峻さを得るのに物理的に長い時間（多くの波長分）を要する。

ConvoPeqのデフォルト値（`MIXED_F1_DEFAULT_HZ=200`, `MIXED_F2_DEFAULT_HZ=1000`、UI許容範囲 `F1:100-400Hz` / `F2:700-1300Hz`）は、まさにこの文献群が言及する典型的な「ミックスフェーズ・クロスオーバー帯域」と一致している。これは、**実装者の意図が文献の慣行（低域Minimum/高域Linear）に沿ったものであったと推測する強い状況証拠**であり、現在のコードはその意図と重みが入れ替わっている可能性が高い。

**実害の推定**: 現状のままだと、IRの低域成分（多くの自然なIR・リバーブ・スピーカーキャビネットIRにおいて支配的なエネルギーを持つ帯域）に対して、ピーク位置を基準とした一定群遅延（線形位相）が強制される。これは、低域でのプリリンギング（聴感的に最も問題視される帯域でのトランジェント・スメア）を温存したまま、逆にプリリンギングの影響が元々軽微な高域だけをMinimum Phase化する、という**「Mixed Phaseモードが本来意図した効果と正反対の挙動」**になっている可能性がある。

*（注: これはコードの挙動として確認した事実〔重みの数式と既定パラメータ〕に基づく強い推測であり、実際の音響的影響は実機でのIR波形・周波数解析による確認が望ましい。意図的にこの方向性を選んだ可能性も論理的には排除できないため、「確定したバグ」ではなく「要検証の高確信度の疑義」として報告する。）*

### 3.4 〔既知事項の再確認・F2〕`mixedPreRingTau` パラメータのデッドコード化

`convertToMixedPhaseAllpass()` 内で `tau` パラメータは：

- キャッシュキー（`IRCacheKey::tau`, `MixedPhasePersistentCache`）として**保存のみ**される（98, 124, 676行目）。
- 群遅延ターゲット生成（`targetGroupDelay` の計算）、オールパス設計（`AllpassDesigner::Config`）、最終的な周波数応答合成のいずれにも**一度も登場しない**。
- フォールバック関数 `convertToMixedPhaseFallback()` では明示的に `(void)tau;` とされ、未使用であることが意図的に明示されている。

UI上は `MIXED_TAU_MIN=4.0` 〜 `MIXED_TAU_MAX=256.0`（既定32.0）として「Pre-Ring Tau」という名前で公開されており、ユーザがこの値を変更すると（キャッシュキーが変わるため）IR再設計がトリガーされるが、**得られるIRは数学的に完全に同一**になる。本検証で再確認した結果、これは既にユーザの調査メモにある「`mixedPreRingTau` UIパラメータが実際のコードパスでゼロの音響的効果しか持たない」という既知の発見と完全に一致する。

### 3.5 〔新規発見・F5〕`AllpassDesigner::applyAllpassToIR` の未使用化

`AllpassDesigner.h` で宣言・`AllpassDesigner.cpp` (598-742行目) で実装されている静的関数 `applyAllpassToIR()` は、MKL DFTIを用いてオールパスカスケードをIRに正しく適用するユーティリティだが、**コードベース全体で呼び出し箇所が一つも存在しない**（`grep` で定義・宣言以外のヒット無し）。実際の `convertToMixedPhaseAllpass()` は、同等の処理（周波数応答計算→複素乗算→逆FFT）を**インラインで再実装**している（555-567行目）。機能上の不具合は無いが、ロジックの二重化により将来の保守時に「片方だけ修正される」リスクを抱えている。

---

## 4. リサンプリング（r8brain）の検証

### 4.1 設定パラメータの評価

`IRDSP.h` の `ResampleConfig` 既定値（`transBand=2.0`, `stopBandAtten=140.0dB`, `phase=fprLinearPhase`）は、r8brain（Aleksey Vaneev氏の r8brain-free-src）の公開仕様（transition bandは入力/出力帯域幅に対する割合で0.5%〜45%、stop-band attenuationは49〜218dB）の中でも**かなり高品質側**の設定である。IR用リサンプリングとしては妥当、というよりむしろ慎重に高品質を選んでいる。

### 4.2 〔新規発見・F3〕出力バッファマージン不足による IR テイル切り捨てリスク

`IRDSP::resampleIR()`（`IRDSP.cpp`）は出力バッファ長を

```cpp
const double expectedLen = static_cast<double>(inLength) * ratio + 2.0;
const int maxOutLen = static_cast<int>(expectedLen);
```

として確保し、その後 `while (done < maxOutLen ...)` というフラッシュループで r8brain の残存出力を排出する。

r8brainの公開ドキュメント／ソース（GitHub `avaneev/r8brain-free-src`）を調査した結果、以下が判明した：

- トップレベルの `r8b::CDSPResampler::getLatency()` は常に **0を返す**ことが明記されている。これはAPIドキュメントにある「リサンプラーは初期処理レイテンシを自動的に除去する」という説明と一致する——すなわち、出力ストリームの先頭から、内部フィルタの群遅延分はあらかじめ取り除かれた状態で出てくる。
- その代償として、**入力ストリームの終端を処理した後も、内部フィルタが「先読み」していた分のサンプルを排出するためのフラッシュ処理（`process(nullptr, 0, ...)` の繰り返し呼び出し）が必要**であり、この排出量は内部フィルタの長さ（≒レイテンシ）にほぼ一致する。
- README には「transition bandを広げ、attenuationを下げるとフィルタ長が短くなり、それに伴い *input before output delay* も小さくなる」という記述があり、**transition band/attenuationの設定値とフィルタ内部レイテンシが直結している**ことが確認できる。

ConvoPeqの設定（`stopBandAtten=140dB`, `transBand=2.0`〔=2%相当〕）は、窓関数法FIRの経験則（Harrisの近似式 `N ≈ Atten_dB / (22·Δf)`、Δfは正規化遷移帯域幅）に当てはめると、単段相当で **概算300タップ程度**に達する規模であり、r8brainの多段（half-band cascade + 補間段）構成でも、合計の内部レイテンシは数十〜100サンプル超の規模になりうる。

これに対し、`IRDSP::resampleIR()` が確保しているマージンは **わずか2.0サンプル分**であり、r8brainのいかなるレイテンシ照会APIも呼び出していない。したがって：

- フラッシュループは `done < maxOutLen` という条件で停止するため、**r8brainがまだ排出可能な正当な出力サンプルを残したまま、ループが先に終了する**可能性が高い。
- その結果、**リサンプリング後のIRの末尾（テイル）が、内部フィルタのレイテンシに相当する分だけ切り捨てられる**——リバーブテイルの減衰末尾やエコーの後半部分が静かに失われる、という形の劣化が生じうる。

これは、ユーザの調査メモに記載されている既存の未解決事項（「r8brainリサンプリングのバッファマージンによるIRテイル切り捨てリスク」）について、**r8brainの公開API仕様（`getLatency()`が常に0を返す設計であること、フィルタ長と内部レイテンシが設定値に直結すること）から論理的に妥当性を裏付ける**ものである。実害の正確な量（サンプル数）はビルドして実測するのが望ましいが、設定値（140dB/2%）から見て「無視できる程度」とは考えにくい規模である。

**推奨対応の方向性**: `expectedLen` の計算に、r8brainのフィルタ設計から導出される実レイテンシ相当のマージン（最低でも数百サンプル、または `transBand`/`stopBandAtten` から動的に見積もった値）を加える。あるいは、`maxOutLen` を「フラッシュが完全に終わるまで」可変長で確保し、完了後に実際の `done` 値でトリムする方式に変更するのが安全である。

---

## 5. 周辺実装の健全性

以下は検証の結果、**問題が見つからなかった**項目である（網羅的に確認した範囲内）。

- **非正規化数（denormal）対策**: Audio Thread (`juce::ScopedNoDenormals`)、LoaderThread (`_MM_SET_FLUSH_ZERO_MODE`/`_MM_SET_DENORMALS_ZERO_MODE`/`vmlSetMode(VML_FTZDAZ_ON)`)、Mixed-Phase処理スレッドのいずれにも一貫してFTZ/DAZが設定されており、リアルタイムDSPの標準的なプラクティスに合致する。
- **メモリ管理**: `AlignedAllocation.h` は `mkl_malloc`/`mkl_free` を64バイトアラインメントで一貫して使用し、`ScopedAlignedPtr`/`aligned_unique_ptr` によるRAII管理で漏れの無い設計になっている。`PreparedIRState` のムーブセマンティクスも自己代入・多重解放を正しく回避している。
- **オートゲイン（`IRConverter::computeScaleFactor`）**: IRエネルギーに基づく正規化（`1/√energy`）に -6dB（`10^(-6/20)=0.50119`）のセーフティマージンを掛け、さらにピーク/RMSの絶対クランプ、IR切り替え時の急激なレベル変化を制限する「ジャンプリミッタ」（4倍超の変化を抑制）を備えており、リアルタイムIR差し替え時の音量飛びを防ぐ妥当な設計である。
- **Dry/Wetミックスとレイテンシクロスフェード**: Equal-Power（sin/cos）クロスフェードと、レイテンシ変更時のCatmull-Rom 3次補間による分数遅延読み出し（AVX2実装）は、クリックノイズの無いシームレスな切り替えのための標準的かつ正しい手法である。
- **IRの非対称Tukeyウィンドウ（`applyAsymmetricTukey`）**: ピーク前は固定5%、ピーク後はIR長の対数に応じて可変（5%〜25%）のテーパーを適用しており、IRトリム時の高周波クリックノイズ（スペクトルリーケージ）対策として妥当である。
- **RCU / スレッド安全性**: `EpochManager`/`RCUReader`/`DeferredDeletionQueue` によるIRホットスワップ機構、Audio Thread内でのメモリ確保ゼロ方針（`SetImpulse()`時点での事前確保）は、過去の調査で確立された設計（メモリ記載のRCU移行作業）と一致し、本検証範囲内でも矛盾は見られなかった。

---

## 6. 結論と優先度付き推奨事項

**総合評価**: ConvoPeqのコンボルバーは、**コア畳み込みアルゴリズム（NUCエンジン本体）に関しては音響工学的に妥当**であり、Gardner (1995) に始まる非一様分割畳み込みの確立された理論を正しく実装している。FDLの最適化、SIMD実装、レイテンシ予算、RCUベースのスレッド安全性のいずれも、検証した範囲で論理的な誤りは見つからなかった。

一方で、**コア畳み込みの「周辺」にある機能（Mixed-Phase設計、リサンプリング、レガシーなサイジング配線）には、実際の信号や数値を追跡することで初めて見える具体的な不整合が複数存在する**。優先度は以下の通り提案する。

| 優先度 | 項目 | 理由 |
|---|---|---|
| **高** | F1: Mixed Phase クロスオーバー方向の検証・修正 | 機能の音響的な目的（プリリンギング抑制）と実際の挙動が逆転している可能性があり、ユーザが期待する音響的効果が得られていない疑いが強い |
| **高** | F3: r8brainリサンプリングのテイル切り捨て対策 | サンプルレート変換を伴う全てのIRロードに影響しうる潜在的な音質劣化（リバーブテイルの消失） |
| 中 | F2: `mixedPreRingTau` の実装または UI からの削除 | ユーザが効果の無いパラメータを操作してしまう体験上の問題（既知事項） |
| 低 | F4: `computeMasteringSizing` 周りのデッドコード整理 | 動作には影響しないが、保守性・可読性のリスク |
| 低 | F5: `applyAllpassToIR` の重複実装解消 | 将来の保守時にロジック分岐リスク |
| 情報 | F6: レイヤースケジューリングの最適化余地 | バグではないが、Garcia/Wefers型の厳密最適スケジュールに対し計算コストの最適化余地が残っている可能性（既存調査テーマと一致） |

---

## 参考文献

1. Gardner, W. G. (1995). *Efficient Convolution without Input-Output Delay*. Journal of the Audio Engineering Society, 43(3), 127–136.
2. Garcia, G. (2002). *Optimal Filter Partition for Efficient Convolution with Short Input/Output Delay*. AES 113th Convention, Los Angeles. (および対応特許 US6625629)
3. Wefers, F. (2015). *Partitioned convolution algorithms for real-time auralization*. PhD thesis, RWTH Aachen University / Logos Verlag Berlin. (特に第6章 "Non-uniformly partitioned convolution")
4. Stockham, T. G. (1966). *High-speed convolution and correlation*. AFIPS Conference Proceedings.
5. Egelmeers, G. P. M., & Sommen, P. C. W. (1996). *A new method for efficient convolution in frequency domain by nonuniform partitioning for adaptive filtering*. IEEE Trans. Signal Processing.
6. Rosseel, H., & van Waterschoot, T. (2025). *A State-of-the-Art Review on Acoustic Preservation of Historical Worship Spaces through Auralization*. arXiv:2503.18022. （6.1.1.2節 "Partitioned Convolution" を参照）
7. Deczky, A. G. (1972). *Synthesis of recursive digital filters using the minimum p-error criterion*. IEEE Trans. Audio and Electroacoustics.
8. Dirac Research AB. *On Room Correction and Equalization of Sound Systems* (technical whitepaper).
9. Oppenheim, A. V., & Schafer, R. W. *Discrete-Time Signal Processing*（オールパス群遅延の標準的導出）.
10. Vaneev, A. *r8brain-free-src* — High-quality pro audio resampler library. GitHub: avaneev/r8brain-free-src（`CDSPResampler::getLatency()` の実装・ドキュメントを参照）.

---

*本レポートはConvoPeqのソースコード（添付ConvoPeq.md）の静的解析と、上記文献調査に基づく。F1・F3は「コードの実際の挙動」としては確認済みの事実だが、「設計意図との一致・不一致」については推測を含む。実機での音響測定（IRの周波数応答・群遅延・テイル波形の確認）による裏付けを推奨する。*
