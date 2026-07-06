# ConvoPeq: ASIO入力 → コンボルバー/PEQ処理 詳細検証レポート

対象範囲: 外部ASIO機器（Voicemeeter Virtual ASIO等）からのオーディオ到達 〜 `processInputDouble()` 〜 オーバーサンプリング 〜 コンボルバー/EQのいずれか先に実行される段まで。
検証方法: 実ソースコードを直接追跡（推測に基づく記述は明示）。

---

## 1. ASIO入力からDSPCoreコールバックに至る経路

### 1-1. デバイス層〜コールバック接続

- `MainWindow.cpp`: `juce::AudioProcessorPlayer audioProcessorPlayer;` を用い、`audioProcessorPlayer.setDoublePrecisionProcessing(true)` を無条件設定した上で `setProcessor(audioEngineProcessor.get())`。
- 実運用では ASIO からは float32 で到達するが、JUCEの`AudioProcessorPlayer`が毎コールバック float→double 変換を行い、**`AudioEngineProcessor::processBlock(double&, ...)` → `AudioEngine::processBlockDouble()` のみが実行経路**になる（`processBlock(float&)`/`getNextAudioBlock`は事実上不使用）。
- **重要な理解**: double精度化は内部演算（フィルタ状態・畳み込みFFT正規化・アキュムレータ）の丸め誤差を減らす効果はあるが、ASIOから来る信号自体の量子化分解能（float32、実効的にはオーディオI/Fのbit深度に依存）を遡って向上させるものではない。「入力の実力以上の解像度」を生み出すわけではない点は認識しておく必要がある。

### 1-2. チャンネルレイアウト検証

`AudioEngineProcessor::isBusesLayoutSupported()`:
```cpp
if (outSet.size() < 1 || outSet.size() > 2) return false;
if (!inSet.isDisabled() && (inSet.size() < 1 || inSet.size() > 2)) return false;
```
入出力とも1〜2chに制限。Voicemeeter Virtual ASIOのような多チャンネルデバイスからも、デバイス選択UI側（`AudioDeviceSelectorComponent`, min/max 1-2ch設定）で実際に使うチャンネル対を選ばせる設計であり、製品スコープ（ステレオEQ+コンボルバー）として妥当。

### 1-3. バッファサイズ超過に対する安全策

`AudioEngine::processBlockDouble()`:
```cpp
if (numSamples > dsp->maxSamplesPerBlock)
{
    buffer.clear();
    return;
}
```
ASIOドライバ（特にVoicemeeterのような仮想ドライバ）が想定外に大きいブロックを要求してきた場合でも、バッファオーバーランせずサイレンスを返す設計になっている。安全側の実装として妥当。

### 1-4. サンプルレート整合性ガード（重要・良好な設計）

同関数内、DSP取得直後:
```cpp
const double engineSampleRate = getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef, 0.0);
if (engineSampleRate <= 0.0 || absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
{
    buffer.clear();
    return;
}
```
**現在アクティブなDSPCoreが「準備された時のサンプルレート」と「現在のエンジン全体のサンプルレート」が一致しているかを毎コールバック検証し、不一致ならサイレンスを返す。** これは後述(4章)のサンプルレート変更時の過渡状態に対する極めて重要な安全策であり、非常に良い設計判断である。

---

## 2. `processInputDouble()` の検証（要約）

既に確認済みの内容の要約（詳細は前回レポート参照）:

1. フォーマット変換（`convertDoubleToDoubleHighQuality`）
2. モノラル→ステレオ展開
3. NaN/Inf sanitize
4. 入力レベル計測（ヘッドルームゲイン適用前＝生入力を計測、妥当）
5. Analyzerタップ
6. 入力ヘッドルームゲイン
7. DCブロッカー（入力、3Hz、2段カスケード）

この並びは音響工学的に妥当と判断済み（詳細は前回レポート参照）。

---

## 3. オーバーサンプリングと処理レートの整合性

### 3-1. `processingRate` の定義

`DSPCore::prepare()`:
```cpp
const double processingRate = newSampleRate * static_cast<double>(oversamplingFactor);
```
畳み込みエンジン・EQ・DCブロッカー(オーバーサンプル域)・OutputFilter は全て `processingRate`（デバイスのサンプルレート × オーバーサンプリング倍率）で `prepare` される:

```cpp
convolverState->prepare(owner, processingRate, processingBlockSize);
eqState->prepare(processingRate, internalMaxBlock);
dcBlockers().init(newSampleRate, processingRate);
outputFilter.prepare(processingRate);
```

一方 `truePeakDetector`/`loudnessMeter` は `newSampleRate`（ベースレート）で prepare される。これは処理順序上、これらのメーターが**ダウンサンプリング後のベースレート信号**を計測する設計だからであり、整合性は取れている。

### 3-2. オーバーサンプラー自体の設計

前回確認済み: Kaiser窓ポリフェーズ・ハーフバンドFIR、2x/4x/8x多段構成、90〜160dB減衰。設計上の問題なし。

---

## 4. 【本検証の中心】サンプルレート／オーバーサンプリング係数変更時の、コンボルバーIR再構築の整合性

ASIO機器（特にVoicemeeterのような仮想ドライバ）は、接続タイミングやユーザーのWindows側サンプルレート設定変更によって、セッション途中で提示サンプルレートが変わり得る。また `oversamplingFactor` はユーザーがConvoPeq内で変更可能な設定である。**このどちらも `processingRate` を変化させ、コンボルバーの再構築（rebuild）をトリガーする。** ここに、単体で見ると重大な不整合に見える実装があったため、実際に音声出力へ影響するかを多段階で追跡した。

### 4-1. `ConvolverProcessor::prepareToPlay()` のインライン再構築ロジック

```cpp
const bool rateChanged = (std::abs(currentSampleRate - sampleRate) > 1e-6);
...
if ((rateChanged || blockChanged) && conv->irDataLength > 0)
{
    auto irL = convo::makeAlignedArray<double>(conv->irDataLength);
    auto irR = convo::makeAlignedArray<double>(conv->irDataLength);
    std::memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double));
    std::memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double));
    ...
    newConv->init(irL.release(), irR.release(), conv->irDataLength, sampleRate, ...);
}
```

**この分岐は、現在アクティブなエンジンが既に保持しているIRサンプル配列を、リサンプリングせずそのまま`memcpy`し、新しい`sampleRate`ラベルを付けて`init()`に渡しているだけ**である。`StereoConvolver::init()` → `MKLNonUniformConvolver::SetImpulse()` の実装を確認したところ、**畳み込みエンジンはサンプル数（`irLen`/`blockSize`）のみを扱い、サンプルレートという概念を一切持っていない**（`SetImpulse`のシグネチャに`sampleRate`引数自体が存在しない）。

したがって、この分岐だけを単独で見た場合: 例えば元々48kHz(オーバーサンプリング無効, processingRate=48000)用にリサンプリング済みのIRが、ユーザーが4倍オーバーサンプリングを有効化した瞬間（processingRate=192000に変化）、**同じサンプル配列がそのまま「192kHzのデータ」として扱われ、IRの時間軸が実質1/4に圧縮される**（インパルス応答の全ての周波数特性が4倍にシフトし、減衰時間も1/4になる）という、致命的な音響的破綻を招く実装に見える。

### 4-2. しかし: この不整合は2段階の下流セーフガードによって現在は表面化しない

詳細に呼び出し経路を追跡した結果、以下が判明した。

**(a) 正しいリサンプリング機構が別途存在し、必ず後続で実行される**

`ConvolverProcessor::rebuildAllIRsSynchronous()`:
```cpp
const double processingSampleRate = currentSampleRate; // 新しいprocessingRate
LoaderThread loader(*this, *(state->ir), state->sampleRate, processingSampleRate, ...);
                             // ↑ 元々の"ソースIR"          ↑ ソースの真のサンプルレート
loader.runSynchronously();
```
`state->ir`/`state->sampleRate` は、**ユーザーが最初にIRファイルを読み込んだ時点の「元データ・元サンプルレート」を独立して保持している`IRState`から取得**されており、`ConvolverProcessor::prepareToPlay()`が参照する「現在アクティブなエンジンの（既に加工済みの）irData」とは別物である。この関数はソースIRから正しくターゲットレートへリサンプリング（r8brain経由、前回確認済みの`getMaxOutLen()`使用で安全）した上でエンジンを再構築する。

**(b) サンプルレート/オーバーサンプリング係数変更時は、必ず (a) を経由するパイプラインのみが本番反映される**

`AudioEngine::rebuildThreadLoop()`（バックグラウンドの構造リビルド専用スレッド）:
```cpp
convo::BuildResult buildResult = runtimeBuilder.build(task.runtimeBuildSnapshot.buildInput, task.convolverBuildSnapshot);
// ↑ ここで DSPCore::prepare() → ConvolverProcessor::prepareToPlay() が呼ばれる（4-1のインライン再構築が発生）
...
if (newDSP->convolverRt().getIRLength() > 0)
{
    newDSP->convolverRt().rebuildAllIRsSynchronous(isObsolete);
    // ↑ 直後に必ず正しいリサンプリングで上書きされる
}
...
// 6. Commit on Message Thread ← ここで初めてRTスレッドに公開される
```
`AudioEngine::setOversamplingFactor()`や、`AudioEngine::prepareToPlay()`内のサンプルレート変更検知（`rateChanged`）は、いずれも最終的に`submitRebuildIntent(RebuildKind::Structural, ...)` → `requestRebuild()` → 上記`rebuildThreadLoop`の非同期パイプラインを経由する。**このパイプライン内では、4-1の（単体では不正確な）インライン再構築の結果は、RTスレッドに公開される前に必ず`rebuildAllIRsSynchronous()`によって正しいリサンプリング結果で上書きされる。**

**(c) さらに、RTスレッド側に独立したサンプルレート整合性チェックが存在する（1-4節参照）**

`AudioEngine::processBlockDouble()`の`dsp->sampleRate == engineSampleRate`チェックにより、**たとえ何らかの理由で(a)(b)のパイプラインが破綻し、リサンプリングされていない中間状態のDSPCoreが公開されてしまったとしても**、その中間状態のDSPCoreが持つ`sampleRate`メンバは（4-1のコードで`newConv->init(..., sampleRate, ...)`に渡された、変更後の新しい値になっているはずなので）、この整合性チェック自体はすり抜けてしまう可能性がある点には注意が必要（このガードは「DSPCoreが今の全体エンジンレートに追従しているか」を見ているのであって、「DSPCore内部のIRが正しくリサンプリングされているか」までは検証していない）。つまり(c)は(a)(b)が想定通り機能する前提の設計であり、IRリサンプリング漏れそのものに対する直接的な保険ではない。

### 4-3. 結論と評価

- **現在の実装において、サンプルレート/オーバーサンプリング係数の変更が実際の音声出力に対してIRの時間軸破壊を引き起こすことは確認できなかった。** これは(a)(b)の「prepare→rebuildAllIRsSynchronous→commit」という2段階シーケンスが、現状全ての到達経路で正しく守られているためである。
- ただし、この安全性は**型システムやアサーションによって強制されたものではなく、「`ConvolverProcessor::prepareToPlay()`を呼んだら必ず直後に`rebuildAllIRsSynchronous()`を呼ぶ」という、複数ファイルにまたがる暗黙の呼び出し規約**に依存している。`ConvolverProcessor::prepareToPlay()`単体は、この規約を知らないコードから将来呼び出された場合（例えば新しい「サンプルレートだけ即座に反映したい」という最適化パスが将来追加された場合など）、無条件にIR破壊を引き起こす関数として存在し続けている。
- **推奨対応**: `ConvolverProcessor::prepareToPlay()`のインライン`rateChanged`分岐は、(1) 完全に削除し常に`rebuildAllIRsSynchronous()`相当の経路のみを使う、(2) 少なくとも関数コメント＋実行時アサーション（例:「この関数の直後に必ずrebuildAllIRsSynchronousが呼ばれることを呼び出し元が保証すること」を`jassert`や静的解析コメントで明示）を付与し、暗黙の規約を明示的な契約に格上げする、のいずれかを推奨する。現状の安全性はコードレビューの副産物であり、意図された設計としてドキュメント化されていない。

---

## 5. まとめ

| # | 項目 | 評価 |
|---|------|------|
| 1 | チャンネルレイアウト制限(1-2ch) | 妥当 |
| 2 | バッファサイズ超過時のサイレンス出力 | 妥当・安全 |
| 3 | DSPCore⇔エンジンのサンプルレート整合性チェック | 優れた設計 |
| 4 | `processInputDouble`内の処理順序 | 妥当（前回確認済み） |
| 5 | オーバーサンプラーのフィルタ設計 | 優れた設計（前回確認済み） |
| 6 | `processingRate`の畳み込み/EQ/OutputFilterへの伝播 | 正しく整合 |
| 7 | **`ConvolverProcessor::prepareToPlay()`のIR再利用ロジック単体** | **設計として脆弱（単体では誤り）** |
| 8 | **上記が実際の出力に影響するか** | **現状は下流の2段階セーフガードにより非顕在化。ただし暗黙の呼び出し規約に依存しており、リファクタ時のリグレッションリスクを抱える** |

全体として、ASIO入力からコンボルバー/EQに至る経路は、RT安全性・サンプルレート整合性・オーバーサンプリング設計のいずれにおいても高い水準で作り込まれている。唯一の懸念は4章で述べた「正しさが暗黙の呼び出し順序に依存している」設計上の脆さであり、機能面のバグではなく保守性・将来のリグレッション耐性の観点からの指摘である。
