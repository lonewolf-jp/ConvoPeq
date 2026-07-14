## 第1回レビュー: B14「Retire Queue のデータ競合 (MPSC Queue Data Race)」

今回は最優先項目である **B14** を詳細に検証します。

---

# 結論

現時点での評価は次のとおりです。

| 項目        | 評価         |
| --------- | ---------- |
| バグの存在     | **可能性は高い** |
| 報告内容の妥当性  | ★★★★☆      |
| 修正案の妥当性   | ★★★☆☆      |
| そのまま適用推奨か | **いいえ**    |

つまり

> **問題の指摘自体はかなり妥当ですが、修正方法は慎重に再設計すべき**

という評価になります。

---

# 1. ソースコード確認

最新版 ConvoPeq では RetireRuntime に

* retireIntentQueue_
* fallbackQueue_
* acknowledgeGeneration_
* overflowCount
* fallbackCount

など多数の共有状態があります。

さらに

```cpp
escalateAllRetires()
```

では

```cpp
retireIntentQueue_
```

全体を走査しています。

コメントには

> Shutdown中は単一スレッドアクセス

と書かれています。

つまり通常運転時には

* Audio Thread
* Timer
* Worker

からアクセスされることを設計者自身も前提にしています。

---

# 2. MPSC Queueとして正しいか

ここが最重要です。

MPSC Queueでは

Producer

```
payload
↓

release

↓

tail++
```

Consumer

```
tail load(acquire)

↓

payload読む
```

という順序が必要になります。

もし

```
payloadを書いている途中

↓

Consumerが読む
```

なら

未初期化データ

古いgeneration

nullptr

を読む危険があります。

---

# 3. 現状コード

現状コードでは

```
RetireIntent
```

に

```
bool isValid
priority
generation
slot
...
```

があります。

Shutdown時は

```
reinterpret_cast<std::atomic<bool>>
```

で読んでいます。

この点は少し危険です。

理由は

```
bool

↓

reinterpret_cast

↓

atomic<bool>
```

は

**C++標準では保証されません。**

これはstrict aliasing以前に

atomic objectではないものを

atomicとして扱っています。

---

# 4. ただし

報告書では

「データ競合」

と断定しているようですが、

ここは慎重であるべきです。

現状コードには

```
publishAtomic()

consumeAtomic()

fetchAddAtomic()
```

というラッパがあります。

もし

```
enqueue()

↓

publishAtomic()

↓

dequeue()

↓

consumeAtomic()
```

になっているなら

data raceではありません。

つまり

レビュー本文だけでは

断定できません。

ここは

enqueue実装

dequeue実装

まで確認する必要があります。

---

# 5. 修正案について

報告書では

おそらく

```
全部atomicに変更
```

あるいは

```
memory_order変更
```

を提案していると思われます。

私はこの修正はそのままでは勧めません。

理由は

Audio Threadでは

```
atomic<uint64>

atomic<bool>

atomic<size_t>

...
```

が大量に増えると

キャッシュライン競合が増えます。

ISR設計では

**atomicを増やすこと自体がコスト**になります。

---

# 6. ISR設計から見た改善案

ISRなら

RetireIntentそのものをatomic化するより

```
struct Slot
{
    RetireIntent payload;

    std::atomic<uint64_t> sequence;
}
```

という

Vyukov MPSC

あるいは

sequence lock

方式の方が自然です。

つまり

```
payload

↓

release

↓

sequence++
```

だけatomicにします。

これなら

payload全体をatomicにする必要がありません。

Audio Threadにも優しい設計になります。

---

# 7. Shutdownコードについて

ここは別問題があります。

```
reinterpret_cast<std::atomic<bool>>
```

という実装は

Shutdown専用でも

設計としてかなり危険です。

Shutdownだから問題が起きにくいだけで、

可読性・保守性は良くありません。

ここは

```
std::atomic<bool> isValid;
```

へ変更した方が安全です。

ただし

payload全体をatomicにする必要はありません。

---

# 総合評価

| 項目      |           評価 |
| ------- | -----------: |
| バグの存在   | ★★★★☆（高い可能性） |
| 報告書の解析  |        ★★★★☆ |
| 原因分析    |        ★★★★☆ |
| 修正案     |        ★★★☆☆ |
| ISRとの整合 |        ★★★☆☆ |

### 私の結論

* **B14の問題提起自体は概ね妥当**です。
* ただし、現時点で確認できた範囲だけでは「MPSC Queue にデータ競合が存在する」とまでは断定できません。`enqueue`/`dequeue` の実装と `publishAtomic`・`consumeAtomic` の使用箇所まで確認する必要があります。
* 修正は **atomicを増やす方向ではなく、release/acquire による公開プロトコル（sequence番号や公開フラグ）を明確化する方向**が、ISRおよびリアルタイムDSP設計に適しています。



## 第2回レビュー: B20「TruePeakDetector Rチャンネル計測欠落」

これは DSP 品質に直接影響するため、B14と並んで優先度が高い項目です。報告書では **P0（即時修正）** と評価されています。 

### 結論

| 項目            | 評価              |
| ------------- | --------------- |
| Rチャンネル処理欠落の指摘 | **妥当である可能性が高い** |
| BS.1770との整合性  | ★★★★★           |
| 「即時修正」の優先度    | ★★★★☆           |
| 修正案           | ★★★☆☆           |

私の評価は、

> **Rチャンネルが実際に Stage1 で処理されていないのであれば、これは確実に BUG です。**

になります。

---

## 1. True Peakとは

ITU-R BS.1770 の True Peak Meter は

```
L
R
```

それぞれを

オーバーサンプリング

↓

FIR補間

↓

最大値検出

します。

つまり

```
Lだけ測る

Rは測らない
```

という実装は規格に適合しません。

---

## 2. 報告内容

報告書では

```
Stage0

L
↓

interpolate()

R
↓

interpolate()

Stage1

Lのみ

↓

Peak
```

となっており

**Stage1でRが欠落**

していると主張しています。

もし本当にそうなら

R側は

```
2x

↓

4x

↓

Peak
```

が実施されません。

つまり

インターサンプルピークを

Rだけ見逃します。

---

## 3. BS.1770との比較

BS.1770では

チャンネル毎に

```
oversample

↓

true peak

↓

max
```

です。

左右を混ぜません。

したがって

R側のStage1欠落は

**規格違反**

になります。

ここはDSP仕様としてかなり重大です。

---

## 4. どんな症状になるか

例えば

```
L = -3dBTP

R = +0.8dBTP
```

なら

現在のメーターは

```
Lだけ表示

↓

-3dBTP
```

となります。

実際には

```
+0.8dBTP
```

なのに

Limiter

Meter

Clip警告

すべて誤ります。

プロ用ルーム補正ソフトでは好ましくありません。

---

## 5. バッファオーバーフローについて

報告書はさらに

```
buffer overflow
```

も指摘しています。

これは

R欠落より危険です。

ただし

現時点で取得できた情報では

オーバーフロー箇所までは確認できません。

従って

ここは

**現時点では未確認**

と評価します。

つまり

| 項目       | 評価     |
| -------- | ------ |
| R欠落      | ほぼBUG  |
| Overflow | 追加検証必要 |

です。

---

## 6. 修正案について

報告書は

おそらく

```
L

↓

Stage1

R

↓

Stage1
```

を追加するだけの修正を提案しています。

この方向性は正しいです。

ただし

さらに確認すべき点があります。

### ① work配列

```
work

work + up1Samples
```

で

左右を共有しています。

ここで

Stage2まで

十分なサイズがあるか

検証が必要です。

---

### ② Peak保持

```
peak=max(L)

↓

peak=max(R)
```

ではなく

```
peak=max(peak,L)

peak=max(peak,R)
```

になっているか

確認する必要があります。

---

### ③ SIMD

AVX2実装なら

左右を

```
LLLL

RRRR
```

で分けているか

```
LRLR
```

なのか

確認したいところです。

SIMD最適化次第では

修正位置が変わります。

---

## 7. 優先度

私は

P0評価には概ね賛成です。

理由は

クラッシュではないものの

**メーターが誤表示**

するからです。

プロ用ソフトでは

```
True Peak

Loudness

Limiter

Meter
```

は

信頼性が最優先です。

---

# 総合評価

| 項目     |    評価 |
| ------ | ----: |
| バグの存在  | ★★★★★ |
| 報告書の解析 | ★★★★☆ |
| 修正方針   | ★★★★☆ |
| そのまま適用 | ★★★☆☆ |

### 私の結論

* **B20 の問題提起は妥当であり、実装が報告書どおりであれば BUG と判断できます。**
* Rチャンネルの Stage1 補間が欠落している場合、ITU-R BS.1770 の True Peak 測定として不完全です。
* 一方で、**「バッファオーバーフロー」については、提示された抜粋だけでは十分に確認できません。** この点は `processBlock()` と `interpolateStage()` の全体実装を確認したうえで最終判断すべきです。
* 修正時は、Rチャンネル処理の追加だけでなく、作業バッファ容量、ピーク値の集約方法、SIMD経路との整合性も合わせて検証することを推奨します。



## 第3回レビュー: B13「Non-Uniform Partitioned Convolution (NUPC) の遅延アラインメント不整合」

B13 は ConvoPeq の音質に直接関わるため、DSP の観点から非常に重要です。特に以前の解析で確認した **MKLNonUniformConvolver** の3層構成（64/512/4096 サンプル分割）とも密接に関係します。

---

# 結論

| 項目            | 評価       |
| ------------- | -------- |
| 問題提起          | **概ね妥当** |
| 実際に音質へ影響する可能性 | ★★★★★    |
| 報告書の修正案       | ★★★☆☆    |
| そのまま適用推奨      | いいえ      |

私の評価は

> **「問題は実在する可能性が高いが、修正方法はDSP全体を考慮して設計し直すべき」**

です。

---

# 1. NUPCとは

ConvoPeqは

```
Layer0
64 samples

Layer1
512 samples

Layer2
4096 samples
```

という典型的な

**Non-Uniform Partitioned Convolution**

になっています。

これは

* Gardner (1995)
* Garcia (2002)
* Wefers (2015)

でも採用されている代表的な方式です。

重要なのは

各Layerが

```
FFT

↓

IFFT

↓

Overlap-Add
```

を行うだけではなく、

**時間軸が完全一致していること**

です。

---

# 2. 報告内容

報告書では

概ね

```
Layer0

↓

Layer1

↓

Layer2
```

の

**開始位置（delay alignment）が一致していない**

と指摘しています。

つまり

Layer1

Layer2

が

数サンプル〜数百サンプル

ずれて合成される可能性があります。

---

# 3. これは本当に問題か

はい。

かなり重要です。

例えば

```
Layer0

Impulse

0 sample

Layer1

Impulse

+64 sample
```

なら

合成すると

インパルスが二重になります。

これは

```
Frequency response

↓

Ripple

Phase error

Comb filtering
```

になります。

---

# 4. Wefers論文との比較

Wefers (2015)

では

各Partitionは

```
partitionStart

↓

FFT

↓

IFFT

↓

OLA
```

を

厳密に一致させています。

Layerごとに

```
offset
```

を持ちます。

これは

数学的に必要です。

つまり

offsetが曖昧なら

NUPCは成立しません。

---

# 5. Gardner方式との比較

Gardnerも

```
small partition

large partition
```

の切り替え時に

Delay Compensation

を必須としています。

つまり

Layer毎に

```
FFTサイズが違う

↓

遅延も違う

↓

補償必要
```

です。

ここは理論的にも一致しています。

---

# 6. 修正案について

報告書では

おそらく

```
Layer開始位置を補正
```

程度の修正になっています。

方向性は正しいです。

しかし

私は

もっと根本的に

```
Layer

↓

logical start sample

↓

physical buffer offset
```

を

完全分離した方が良いと思います。

---

# 7. ISR設計との関係

ISRでは

RuntimeWorldに

IR情報

Partition情報

Delay情報

があります。

Delayだけ

別管理すると

将来

```
Crossfade

↓

RuntimePublish

↓

Delay
```

がずれる危険があります。

従って

Delayも

RuntimeWorldに含めるべきです。

---

# 8. 副作用

ここが重要です。

単純に

```
delay += 64
```

などとすると

Crossfade中に

```
旧IR

↓

新IR
```

で

位相が一致しません。

すると

```
Click

Comb

Dip
```

が発生します。

つまり

修正は

Crossfade全体を考慮する必要があります。

---

# 9. 修正方針

私なら

Layer毎ではなく

```
RuntimeBuilder

↓

PartitionDescriptor

↓

delayOffset

↓

RuntimeWorld
```

として

Builderが

全Layerを計算して

Runtimeへ渡します。

DSPは

Runtimeを読むだけ

にします。

ISRにも一致します。

---

# 総合評価

| 項目        |    評価 |
| --------- | ----: |
| 問題の存在     | ★★★★★ |
| 報告書の分析    | ★★★★☆ |
| DSP理論との一致 | ★★★★★ |
| 修正案       | ★★★☆☆ |

## 私の結論

* **B13 の問題提起は、NUPC の理論（Gardner・Garcia・Wefers）と整合しており、十分妥当です。**
* レイヤ間の遅延アラインメントが不一致であれば、周波数応答のリップルや位相誤差、クロスフェード時のアーティファクトなど、音質上の問題が発生する可能性があります。
* ただし、**修正を単純なオフセット調整だけで済ませるのは危険**です。遅延情報は RuntimeBuilder で一元的に算出し、RuntimeWorld の一部として管理するほうが、ISR の設計思想とも整合し、クロスフェードや将来の保守性の面でも安全です。



## 第4回レビュー: B17「Crossfade 中の Runtime 切替／世代管理の不整合」

B17 は DSP アルゴリズムというより、**ISR（Immutable Snapshot Runtime）設計の整合性**に関する指摘です。ConvoPeq の現在のアーキテクチャ（RuntimeWorld・RuntimeBuilder・CrossfadeAuthority・RetireRuntime）を考えると、非常に重要なレビュー項目です。

---

# 結論

| 項目        | 評価              |
| --------- | --------------- |
| 問題提起      | **妥当である可能性が高い** |
| 実際に発生しうるか | ★★★★☆           |
| 修正案       | ★★☆☆☆           |
| そのまま適用推奨  | いいえ             |

私の結論は

> **問題意識は正しいが、修正方法は ISR の基本設計に合わせて再設計すべき**

です。

---

# 1. ISR の原則

Practical Stable ISR Bridge Runtime の原則は

```
Build

↓

Publish

↓

Audio Threadは読むだけ

↓

Retire
```

です。

Audio Thread が

```
RuntimeA

↓

RuntimeB

↓

RuntimeA
```

のように途中で戻ることは絶対に許されません。

つまり

**単調増加（Monotonic）**

が重要になります。

---

# 2. 報告内容

報告書では概ね

```
Crossfade中

↓

Runtime切替

↓

generation管理

↓

Retire
```

の順序が曖昧で、

古い Runtime と新しい Runtime が同時に参照される可能性を指摘しています。

もしこれが事実なら

Audio Thread が

```
旧Filter

↓

新Filter

↓

旧Filter
```

という状態を観測する可能性があります。

---

# 3. なぜ危険か

例えば

```
Runtime 100

↓

Publish

↓

Crossfade開始

↓

Runtime101 Publish

↓

Runtime100 Retire
```

なら問題ありません。

しかし

```
Runtime100

↓

Crossfade開始

↓

Runtime100 Retire

↓

まだCrossfade中
```

なら

Crossfade が

解放済み Runtime を参照します。

これは

Use-after-free

になります。

---

# 4. Epochとの関係

現在 ConvoPeq は

EpochDomain

RetireRuntime

Generation

を持っています。

以前の設計レビューでも

Authority が

```
Epoch

Crossfade

RuntimeWorld
```

に分散していることを指摘しました。

つまり

Crossfade が

Retire を決める

Epoch が

Retire を決める

Runtime が

Generation を持つ

という

Authority の分散があります。

これは ISR の思想と一致しません。

---

# 5. 修正案について

報告書では

おそらく

```
generationを増やす

↓

retireタイミング変更
```

程度の修正です。

これは

対症療法です。

本質的には

Authority が分散したままになります。

---

# 6. ISRとして望ましい構造

私なら

```
RuntimeBuilder

↓

RuntimeWorld(generation)

↓

publish()

↓

CrossfadePlan

↓

Retire
```

という流れにします。

つまり

Crossfade は

Runtime を所有しません。

Crossfade は

```
oldGeneration

newGeneration
```

だけ知ります。

Retire は

RuntimeStore

だけが決定します。

---

# 7. Crossfade の責務

Crossfade は

```
gainOld

gainNew
```

だけ持つべきです。

例えば

```
oldRuntime
```

そのものを保持すると

寿命管理まで担当することになります。

ISRでは

これは責務違反です。

---

# 8. 発生確率

通常運転では

```
IR変更

EQ変更

SR変更
```

が同時に来なければ

問題は起きにくいでしょう。

しかし

ストレステストでは

```
IR連続変更

Automation

Preset連打
```

で顕在化する可能性があります。

以前実施された

Crossfade Storm テスト

とも整合する指摘です。

---

# 9. 私ならどう直すか

ISR設計なら

```
RuntimeWorld
    generation

↓

publish()

↓

CrossfadeState
    fromGeneration
    toGeneration

↓

RetireManager
```

とします。

Crossfade は

Generation を参照するだけです。

Runtime ポインタの寿命は

RuntimeStore

だけが管理します。

これにより

Authority Singularization

が完成します。

---

# 総合評価

| 項目       |    評価 |
| -------- | ----: |
| 問題の存在    | ★★★★☆ |
| ISRとの整合性 | ★★★★★ |
| 報告書の分析   | ★★★★☆ |
| 修正案      | ★★☆☆☆ |

## 私の結論

* **B17 の問題提起は、ISR の観点から十分妥当**です。Crossfade と Runtime の寿命管理が分離されていない場合、世代管理や Retire のタイミングに起因する不整合が発生する可能性があります。
* ただし、**Generation の更新順序や Retire タイミングだけを修正するのでは根本解決にはなりません。**
* 根本的には、以前から検討されている **Authority Singularization** を徹底し、

  * RuntimeStore が Runtime の寿命管理を唯一担当する
  * Crossfade はフェード状態（from/to generation とゲイン）のみ保持する
  * Retire 判定も RuntimeStore に集約する
    という構成にするのが、ISR の設計思想と最も整合する改善策です。



## 第5回レビュー: B21「リミッター／ゲイン処理のクリッピング・ヘッドルーム管理」

B21 は DSP アルゴリズムそのものに関する指摘であり、音質・安全性・測定値の信頼性に直結します。

---

# 結論

| 項目       | 評価       |
| -------- | -------- |
| 問題提起     | **一部妥当** |
| 重大度      | ★★★★☆    |
| 報告書の修正案  | ★★★☆☆    |
| そのまま適用推奨 | いいえ      |

私の評価は、

> **問題提起自体は概ね妥当ですが、報告書の修正案は「対症療法」に留まっており、Limiter 全体のゲイン構造を見直すべきです。**

---

# 1. Limiter の役割

プロ用オーディオソフトの Limiter は一般的に

```text
入力

↓

Pre Gain

↓

True Peak Detection

↓

Gain Computer

↓

Lookahead

↓

Limiter

↓

Output Gain
```

という構成です。

重要なのは

Limiter は

**最後の保険**

であることです。

---

# 2. 報告書の指摘

報告書では概ね

* Gain の適用順序
* Limiter の判定位置
* Headroom の不足
* Clip 判定

について問題を指摘しています。

これは DSP として十分あり得る指摘です。

---

# 3. Headroom

例えば

```text
EQ

+9dB

↓

Output Gain

+3dB
```

なら

合計

```text
+12dB
```

になります。

Limiter が

```text
最後
```

だけなら

途中の演算では

かなり大きな値になります。

double 処理なら

数値的には問題ありません。

しかし

True Peak Detector

との位置関係が重要になります。

---

# 4. True Peakとの関係

もし

```text
Limiter

↓

TruePeak
```

なら

Limiter後のピークしか測れません。

逆に

```text
TruePeak

↓

Limiter
```

なら

Limiter が追従できる設計になります。

この順序は

BS.1770

や

ITU推奨実装

でも重要です。

---

# 5. 報告書の修正案

報告書では

おそらく

```text
Headroom追加

Limiter閾値変更
```

程度になっています。

方向性は間違っていません。

しかし

これだけでは

将来

```text
EQ

↓

Crossfade

↓

Limiter
```

になった場合

また問題になります。

---

# 6. 根本原因

Limiter の問題は

ほとんどの場合

```text
Gain構造
```

が原因です。

つまり

```text
Input Gain

EQ Gain

Auto Gain

Output Gain
```

が

複数箇所に分散しています。

ISRでは

これは好ましくありません。

---

# 7. ISRとの整合

RuntimeWorldには

DSP係数

Crossfade情報

Oversampling情報

があります。

私なら

さらに

```text
GainStructure
```

を

RuntimeWorld

へ持たせます。

例えば

```cpp
struct GainStructure
{
    inputGain;

    makeupGain;

    limiterThreshold;

    outputGain;

    headroom;
};
```

とします。

---

# 8. Crossfadeとの関係

IR変更中

```text
旧IR

↓

新IR
```

では

Energy が変わります。

Limiterだけが

旧IR

を見ていると

新IRでは

Overshoot

する可能性があります。

つまり

Limiter は

Runtime世代に追従すべきです。

---

# 9. プロ用ソフトとの比較

FabFilter Pro-L2

DMG Limitless

Weiss

などでは

Limiter は

DSPチェーン全体の

最後にあります。

さらに

True Peak

Lookahead

Oversampling

が

一体設計になっています。

単純な

```text
threshold
```

変更だけではありません。

---

# 10. 推奨修正

私なら

```text
RuntimeBuilder

↓

GainStructure生成

↓

LimiterRuntime

↓

AudioThread
```

にします。

Audio Threadでは

```cpp
limiter.process(runtime->gainStructure);
```

だけにします。

こうすると

ISRとも一致します。

---

# 総合評価

| 項目       |    評価 |
| -------- | ----: |
| 問題の存在    | ★★★★☆ |
| DSPとの整合性 | ★★★★★ |
| 報告書の分析   | ★★★★☆ |
| 修正案      | ★★★☆☆ |

## 私の結論

* **B21 の問題提起は、Limiter と Gain Structure の設計という観点では妥当**です。
* ただし、**閾値やヘッドルームを局所的に修正するだけでは根本解決にはなりません。**
* Gain・Limiter・True Peak・Headroom を RuntimeWorld の一部として一元管理し、Limiter が Runtime 世代と同期して動作する構造にすることが、ISR の設計思想と整合し、将来的な保守性・音質・リアルタイム安全性の面でも望ましい改善策です。


設計者判断：B21は別途改修を行うので、バグに含めない。



## 第6回レビュー: B11「Crossfade のフェード係数・補間方式の問題」

B11 は音質に関わる指摘ですが、B13（NUPC 遅延補償）や B17（Runtime 世代管理）ほど構造的ではありません。しかし、クロスフェード品質には直接影響するため、プロ用オーディオソフトでは重要なポイントです。

---

# 結論

| 項目     | 評価       |
| ------ | -------- |
| 問題提起   | **概ね妥当** |
| 音質への影響 | ★★★★☆    |
| 重大度    | ★★★☆☆    |
| 修正案    | ★★★☆☆    |

私の評価は

> **フェード方式に改善余地はあるが、報告書の修正案だけでは十分ではない**

です。

---

# 1. Crossfade の目的

Crossfade は

```text
旧IR

↓

新IR
```

を滑らかにつなぐためにあります。

単純な

```text
Old = 1-x

New = x
```

では、

エネルギーが一定になりません。

---

# 2. Linear Fade の問題

例えば

```text
Old = 0.5

New = 0.5
```

では

振幅は一定に見えます。

しかし

エネルギーは

```text
0.5² + 0.5²

=

0.5
```

です。

つまり

中央で約 -3 dB のディップになります。

---

# 3. Equal-Power Fade

一般には

```text
Old = cos θ

New = sin θ
```

または

```text
Old = √(1-x)

New = √x
```

を使います。

すると

```text
Old² + New²

=

1
```

となり

音量変化が起きません。

これは

DAW

IR Loader

Convolver

でも一般的です。

---

# 4. 報告書の指摘

報告書では

Crossfade の係数計算が

Linear

になっている、

あるいは

正規化が不足している可能性を指摘しています。

この方向性は十分理解できます。

---

# 5. ただし

ここで重要なのは

IR の Crossfade は

通常の Audio Fade と少し違います。

IR が

```text
旧

↓

新
```

で

位相が大きく違う場合

Equal Power

でも

Comb Filter

が出ます。

つまり

フェード係数だけでは

完全には解決しません。

---

# 6. ConvoPeq の場合

以前解析した

MKLNonUniformConvolver

では

Layer毎に

```text
Partition

↓

FFT

↓

IFFT
```

しています。

もし

Layer単位で

Crossfadeしているなら

Layer毎の

Delay

Partition Offset

も一致している必要があります。

つまり

B11 は

B13 と独立ではありません。

---

# 7. Runtime との関係

ISRでは

Crossfade は

Runtime 切替時だけ発生します。

したがって

フェード係数も

RuntimeWorld

から供給されるべきです。

例えば

```cpp
CrossfadePlan
{
    duration;

    curve;

    samplesRemaining;
}
```

という形が望ましいでしょう。

DSP 側は

係数を計算するのではなく

Runtime の情報を消費するだけにします。

---

# 8. 修正案

報告書は

おそらく

```text
Linear

↓

Equal Power
```

への変更を提案しています。

方向性は正しいです。

しかし

私はさらに

* Crossfade Curve を RuntimeBuilder が決定
* Audio Thread は LUT（ルックアップテーブル）を参照
* `sqrt()` や `cos()` を Audio Thread で毎サンプル計算しない

という構成を推奨します。

リアルタイム性能にも有利です。

---

# 9. プロ用ソフトとの比較

主要なコンボルバーでは

* Equal-Power Crossfade
* フェード係数の事前計算
* LUT 利用
* Runtime 中の分岐最小化

が一般的です。

Audio Callback 内で毎回三角関数を評価する設計は通常採用されません。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★☆ |
| 報告書の分析  | ★★★★☆ |
| 修正案     | ★★★☆☆ |
| ISRとの整合 | ★★★★☆ |

## 私の結論

* **B11 の問題提起は概ね妥当**です。Linear Crossfade のままであれば、フェード中の音量変動が生じる可能性があります。
* ただし、**フェード係数を Equal-Power に変更するだけでは十分ではありません。** IR の位相差やレイヤ間遅延（B13）も合わせて考慮する必要があります。
* ConvoPeq の ISR アーキテクチャに合わせるのであれば、Crossfade のカーブや係数は RuntimeBuilder 側で事前決定し、Audio Thread は事前計算済みの係数列または LUT を参照する構成が、リアルタイム性・保守性・音質のバランスが最も良いと考えられます。



## 第7回レビュー: B10「Crossfade 完了判定・状態遷移の不整合」

B10 は、DSP アルゴリズムではなく **Crossfade の状態機械（State Machine）** の設計に関する指摘です。ConvoPeq は Runtime の Publish/Retire と Crossfade が密接に結び付いているため、この種のバグは B17 と関連性があります。

---

# 結論

| 項目       | 評価       |
| -------- | -------- |
| 問題提起     | **概ね妥当** |
| 重大度      | ★★★★☆    |
| 修正案      | ★★★☆☆    |
| そのまま適用推奨 | いいえ      |

私の結論は、

> **Crossfade の完了判定を局所的に修正するだけでは不十分で、状態遷移全体を単調（monotonic）な状態機械として再設計すべき**

です。

---

# 1. Crossfade の状態遷移

理想的には、Crossfade は次のような単方向の状態遷移になります。

```text
Idle
   │
   ▼
Prepare
   │
   ▼
Running
   │
   ▼
Complete
   │
   ▼
Retire Old Runtime
   │
   ▼
Idle
```

重要なのは、

* **逆戻りしないこと**
* **同じ状態に二度入らないこと**

です。

---

# 2. 報告書の指摘

報告書では、

* Complete 判定
* Remaining Sample
* Crossfade Active
* Runtime の切替

のタイミングにズレがあり、

例えば

```text
Running

↓

Complete

↓

次フレームで再びRunning
```

のような状態になる可能性を指摘しています。これは状態機械として好ましくありません。

---

# 3. なぜ問題になるのか

例えば

```text
remaining = 1
```

で

最後のサンプルを処理した後、

```text
remaining = 0
```

になったにもかかわらず

```text
active = true
```

のままなら、

次ブロックでも

Crossfade が走ります。

逆に

```text
remaining > 0
```

なのに

```text
active = false
```

なら、

最後までフェードされません。

どちらも音質に影響します。

---

# 4. ISR との関係

ISRでは

Crossfade は

Runtime の寿命とは独立した

**状態**

であるべきです。

つまり

```cpp
Runtime
```

と

```cpp
CrossfadeState
```

は

別オブジェクトです。

状態遷移だけが

```text
Runtime Generation
```

を参照します。

この責務分離が重要です。

---

# 5. 修正案について

報告書は

おそらく

```cpp
if (remaining <= 0)
```

のような

条件修正を提案しています。

これは

症状を直す可能性はあります。

しかし

状態遷移の保証にはなりません。

---

# 6. 状態機械として再設計すべき

私なら

```cpp
enum class CrossfadeState
{
    Idle,
    Preparing,
    Running,
    Completed
};
```

として

遷移を

```text
Idle
↓

Preparing
↓

Running
↓

Completed
↓

Idle
```

だけに制限します。

これなら

逆遷移は起きません。

---

# 7. 完了判定

完了条件は

例えば

```cpp
remainingSamples == 0
```

だけではなく

* Gain が最終値
* Runtime Publish 済み
* Old Runtime が Retire 可能

まで含めて

一つの

```text
Complete Event
```

として扱う方が安全です。

---

# 8. B17との関係

このレビューは

B17

と非常に関係があります。

B17では

Runtime の寿命管理

今回は

Crossfade の状態管理

です。

両方とも

現在の設計では

責務が分散している可能性があります。

---

# 9. ISR設計なら

私なら

```text
RuntimeStore
        │
        ▼
CrossfadeController
        │
        ▼
CrossfadeState
        │
        ▼
RetireManager
```

という構成にします。

Audio Thread は

現在の

```text
CrossfadeState
```

だけを読む構造です。

---

# 総合評価

| 項目          |    評価 |
| ----------- | ----: |
| 問題の存在       | ★★★★☆ |
| 状態機械としての妥当性 | ★★★★★ |
| 報告書の分析      | ★★★★☆ |
| 修正案         | ★★★☆☆ |

## 私の結論

* **B10 の問題提起は妥当**です。Crossfade の完了判定と状態遷移が局所的なフラグ管理に依存している場合、状態の不整合やフェード終了タイミングの誤りが発生する可能性があります。
* ただし、**完了条件の if 文だけを修正するのは根本対策ではありません。**
* ISR アーキテクチャに合わせるなら、Crossfade を単調な状態機械として設計し、Runtime の寿命管理（B17）とは責務を分離することで、保守性・リアルタイム安全性・将来的な拡張性を大きく向上させることができます。



## 第8回レビュー: B09「Crossfade 開始シーケンスの競合・再入（Reentrancy）問題」

B09 は B10・B17 と密接に関連しますが、こちらは **「Crossfade を開始する瞬間」** に焦点を当てたレビューです。Runtime の Publish と Crossfade 開始要求が重なった場合の設計が適切かを問う内容です。

---

# 結論

| 項目       | 評価              |
| -------- | --------------- |
| 問題提起     | **妥当である可能性が高い** |
| 重大度      | ★★★★☆           |
| 報告書の修正案  | ★★★☆☆           |
| そのまま適用推奨 | いいえ             |

私の結論は、

> **Crossfade の開始要求が複数同時に発生することを前提とした設計になっていない場合、この指摘は妥当です。**

---

# 1. Crossfade開始とは

通常の処理は

```text
Build Runtime

↓

Publish Runtime

↓

Crossfade開始

↓

Crossfade終了

↓

Old Runtime Retire
```

です。

しかし実際には

```text
IR変更

EQ変更

Output Gain変更

Oversampling変更
```

などが短時間に連続して発生します。

---

# 2. 問題になるケース

例えば

```
Runtime100

↓

Crossfade開始

↓

Runtime101 Publish

↓

Runtime102 Publish

↓

まだRuntime100→101のCrossfade中
```

となるケースです。

ここで

```
101→102
```

を開始すると

```
100→101

101→102
```

が重なります。

---

# 3. 報告書の指摘

報告書では、

Crossfade の開始条件が十分に排他制御されておらず、

* 二重開始
* 開始途中の再開始
* Runtime の取り違え

が起こり得ると指摘しています。

これは ISR の Publish モデルでも十分起こり得るシナリオです。

---

# 4. Publishは単調でもCrossfadeは単調ではない

ISRでは

```
Runtime100

↓

Runtime101

↓

Runtime102
```

という Publish は単調です。

しかし Crossfade は

```
100→101
```

が終わっていない状態で

```
101→102
```

が始まる可能性があります。

つまり

Runtime は単調でも

Crossfade は単調とは限りません。

ここを切り分ける必要があります。

---

# 5. 修正案について

報告書は

おそらく

```
if (!crossfadeActive)
```

のような条件追加を提案しています。

これは十分ではありません。

例えば

```
Runtime101 Publish

↓

crossfadeActive=true

↓

Runtime102 Publish
```

なら

Runtime102 が失われます。

---

# 6. ISR設計なら

私なら

Crossfade は

```
Pending Runtime
```

を1つ持ちます。

例えば

```
Running

↓

Pending Runtime102

↓

Complete

↓

102開始
```

です。

あるいは

もっとISRらしく

```
Runtime100

↓

Publish101

↓

Publish102

↓

Audio Threadは常に最新Runtimeだけ取得
```

として

Crossfade 自体が

```
Current Runtime

Next Runtime
```

だけを見る設計にします。

---

# 7. Queue化する方法

もし頻繁な更新を想定するなら

```
Publish Queue

↓

Crossfade Controller

↓

Audio Thread
```

という

シリアライズ構造が望ましいです。

Audio Thread が

複数の開始要求を直接処理するべきではありません。

---

# 8. Crossfade Stormとの整合

以前検討した QA 計画には

**Crossfade Storm**

試験が含まれていました。

この試験では

```
IR変更

↓

IR変更

↓

IR変更
```

を高速で繰り返します。

B09 が正しければ

このテストで

* フェード飛び
* Runtime取り違え
* 古いIRへの逆戻り

などが発生する可能性があります。

---

# 9. 根本対策

ISRの設計思想に合わせるなら

```
RuntimeBuilder

↓

RuntimeStore

↓

Publish Queue

↓

CrossfadeController

↓

RetireManager
```

という責務分離が望ましいです。

Crossfade は

Publish の結果を消費するだけで、

Publish 自体を制御してはいけません。

---

# 総合評価

| 項目       |    評価 |
| -------- | ----: |
| 問題の存在    | ★★★★☆ |
| ISRとの整合性 | ★★★★★ |
| 報告書の分析   | ★★★★☆ |
| 修正案      | ★★★☆☆ |

## 私の結論

* **B09 の問題提起は概ね妥当**です。Crossfade 開始要求が再入可能な状態になっている場合、連続 Publish や高速パラメータ変更時に不整合が発生する可能性があります。
* ただし、**`crossfadeActive` のようなフラグだけで開始を抑制する修正は不十分**です。更新の取りこぼしや最新 Runtime の反映遅延を招く恐れがあります。
* ISR の設計思想に沿うなら、Publish を単調なイベント列として扱い、Crossfade はそのイベントを順序どおり消費する専用コンポーネントにすることで、再入・競合・責務分散を避けられます。

---

### ここまで（B09〜B21）の総括

ここまでレビューした項目を重要度で整理すると、私の評価は次のようになります。

| 項目                 | 評価                 |
| ------------------ | ------------------ |
| B20（True Peak）     | ★★★★★（修正優先度最高）     |
| B13（NUPC Delay）    | ★★★★★（DSP品質に重大）    |
| B17（Runtime世代管理）   | ★★★★★（ISR設計の根幹）    |
| B14（Retire Queue）  | ★★★★☆（並行性の詳細確認が必要） |
| B09（Crossfade開始）   | ★★★★☆              |
| B10（Crossfade状態遷移） | ★★★★☆              |
| B11（Crossfade係数）   | ★★★★☆              |
| B21（Limiter/Gain）  | ★★★★☆              |

この中では、**B17・B09・B10 は個別に修正するより、「CrossfadeController の状態機械」と「RuntimeStore の唯一の寿命管理」を再設計することで、一括して解決できる可能性が高い**と考えます。



## 第9回レビュー: B08「Runtime Publish の可視性・メモリオーダーの問題」

B08 は、DSP アルゴリズムではなく **C++ メモリモデル** と **ISR(Runtime Publish)** の実装に関するレビューです。ConvoPeq は RuntimeWorld を Immutable Snapshot として publish する設計であるため、この部分は設計の根幹になります。

---

# 結論

| 項目       | 評価           |
| -------- | ------------ |
| 問題提起     | **非常に妥当**    |
| 重大度      | ★★★★★        |
| 修正案      | ★★★★☆        |
| そのまま適用推奨 | 概ね可（詳細確認は必要） |

今回レビューした項目の中では、

**B17と並ぶ最重要項目**

です。

---

# 1. ISR Publishの原則

Immutable Runtimeでは

```text
Builder

↓

Runtime完成

↓

Publish

↓

Audio Thread読む
```

となります。

重要なのは

**完成する前に絶対読まれない**

ことです。

---

# 2. C++メモリモデル

Publishは

```cpp
runtime.store(ptr,
    std::memory_order_release);
```

Readerは

```cpp
runtime.load(
    std::memory_order_acquire);
```

になります。

これは

C++標準でも

RCU

Hazard Pointer

Epoch

ISR

すべて同じ考えです。

---

# 3. なぜ必要か

例えば

Builderが

```cpp
runtime->eq

runtime->ir

runtime->convolver
```

を構築している途中で

Audio Threadが

```cpp
runtime.load(relaxed)
```

すると

途中状態を読む可能性があります。

つまり

```text
EQだけ新しい

IRは古い

Convolver未完成
```

という

絶対に起きてはいけない状態になります。

---

# 4. 報告書の指摘

報告書では

Publish時の

* release不足
* acquire不足
* generation更新順序

について指摘しています。

この方向性は非常に妥当です。

---

# 5. よくある誤り

例えば

```cpp
generation++;

runtime.store(ptr,
    release);
```

では

Readerは

```text
Generationだけ新しい
```

可能性があります。

逆に

```cpp
runtime.store()

↓

generation++
```

でも

整合しません。

Generationも

Publish対象の一部

として扱う必要があります。

---

# 6. Runtimeは一体でPublishすべき

ISRなら

RuntimeWorldに

```cpp
struct RuntimeWorld
{
    generation;

    eq;

    ir;

    gain;

    crossfade;
};
```

があります。

これ全体を

一度だけ

Publishします。

Generationだけ

atomicで更新する設計は

ISRとは相性がよくありません。

---

# 7. 修正案

報告書では

おそらく

```cpp
release

acquire
```

を追加する方向です。

これは

基本的には正しいです。

しかし

もっと重要なのは

**Publish点を一箇所に限定すること**

です。

例えば

```cpp
publishRuntime(world);
```

だけが

release store

を行うべきです。

---

# 8. Authority Singularizationとの一致

以前から検討している

Authority Singularizationでは

```text
RuntimeBuilder

↓

RuntimeStore

↓

publish()

↓

Audio Thread
```

です。

Publish APIが一つなら

memory orderも

そこだけ見ればよくなります。

これは

保守性が非常に高い設計です。

---

# 9. relaxedは禁止か

報告書では

relaxed使用を問題視しているようです。

ここは少し補足が必要です。

**すべて acquire/release にする必要はありません。**

例えば

```cpp
statisticsCounter
```

や

```cpp
diagnosticCounter
```

は

```cpp
memory_order_relaxed
```

で問題ありません。

重要なのは

Runtime Publish

Generation

Runtime Pointer

です。

つまり

**Publish境界**

だけを厳密にすべきです。

---

# 10. 私ならどう設計するか

```cpp
Builder

↓

Runtime完成

↓

publish(world)
{
    runtime.store(world,
        release);
}

↓

Audio

↓

world =
runtime.load(acquire);
```

これ以外に

PublishするAPIを作りません。

Generationも

RuntimeWorldの中にあります。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★★ |
| 報告書の分析  | ★★★★★ |
| 修正案     | ★★★★☆ |
| ISRとの整合 | ★★★★★ |

## 私の結論

* **B08 の問題提起は非常に妥当**です。Runtime Publish の可視性は ISR の根幹であり、release/acquire による公開・取得の保証が不可欠です。
* 修正案の方向性（release/acquire の適用）は適切ですが、**個々の atomic 操作を修正するだけでなく、「Publish は RuntimeStore の単一 API のみが担当する」という設計に集約することが重要**です。
* また、`memory_order_relaxed` は統計や診断用途では有効ですが、**Runtime Pointer・Generation・Publish 境界には使用すべきではありません。**
* このレビュー項目は **B17（世代管理）・B14（Retire Queue）・B09/B10（Crossfade）と一体で設計を見直すべき中核的な課題**であり、個別修正よりも Publish/Retire の責務統一を優先することを推奨します。



## 第10回レビュー: B07「RuntimeWorld の Immutable 性が破られる可能性」

B07 は ConvoPeq が採用している **Immutable Snapshot Runtime (ISR)** の基本原則そのものに関するレビューです。

これは単なる実装上のバグではなく、

> **設計不変条件 (Design Invariant)**

が維持されているかを問うレビューです。

---

# 結論

| 項目       | 評価           |
| -------- | ------------ |
| 問題提起     | **非常に妥当**    |
| 重大度      | ★★★★★        |
| 修正案      | ★★★☆☆        |
| そのまま適用推奨 | いいえ（設計変更が必要） |

今回レビューした中では

**B08・B17と並ぶ最重要クラス**

です。

---

# 1. ISRの定義

Immutable Runtimeとは

```text
Builder

↓

Runtime生成

↓

Publish

↓

Audio Thread参照

↓

絶対に変更されない
```

ことです。

つまり

Publish後に

```cpp
runtime->gain = ...
```

のような変更は

一切許されません。

---

# 2. なぜ重要か

例えば

```text
EQ係数

IR

Limiter

Crossfade
```

のうち

EQだけを書き換えると

Audio Threadは

```text
EQ = 新

IR = 旧
```

を見る可能性があります。

ISRでは

これを

**部分更新 (Partial Update)**

と呼びます。

最も避けるべき状態です。

---

# 3. 報告書の指摘

報告書では

RuntimeWorld内部または

Runtimeから到達可能なオブジェクトが

Publish後にも変更される可能性を指摘しています。

もし本当に

```cpp
runtime->xxx = ...
```

が存在するなら

ISR違反です。

---

# 4. Mutableメンバー

例えば

```cpp
struct Runtime
{
    std::vector<float> coeffs;
};
```

だけなら問題ありません。

しかし

Publish後に

```cpp
coeffs[3] = ...
```

すると

Immutableではありません。

つまり

const Runtime*

だけでは

十分ではありません。

---

# 5. constでも壊れる

ここは重要です。

例えば

```cpp
const Runtime*
```

でも

中に

```cpp
shared_ptr<Data>
```

があり

その

```cpp
Data
```

を書き換えれば

Immutableは破れます。

つまり

深い意味での

Immutable

が必要です。

---

# 6. 修正案について

報告書では

おそらく

```cpp
const
```

を増やす方向です。

これは

改善になります。

しかし

十分ではありません。

---

# 7. ISRなら

私なら

Builderで

```cpp
RuntimeBuilder

↓

RuntimeWorld完成

↓

freeze()

↓

publish()
```

という考え方にします。

つまり

Publish前に

完全完成

させます。

---

# 8. freezeの意味

例えば

```cpp
RuntimeBuilder
```

は

mutableです。

しかし

```cpp
RuntimeWorld
```

は

生成後

一切変更しません。

Builderだけが

編集できます。

これは

LLVM

Rust

ゲームエンジン

RCU

でも一般的な設計です。

---

# 9. Runtime内部

さらに

RuntimeWorld内も

```cpp
const FilterGraph

const Convolver

const GainStructure

const IR
```

だけを持つようにします。

つまり

Builder以外は

誰も変更できません。

---

# 10. RuntimeCache

一点だけ注意があります。

DSP内部には

例えば

```cpp
FFT scratch

delay line

history

filter state
```

があります。

これらは

当然

mutableです。

しかし

これは

Runtimeではありません。

DSP State

です。

つまり

```text
Runtime

↓

DSP State
```

を完全分離すべきです。

---

# 11. Authority Singularizationとの一致

以前から検討している

```text
Builder

↓

RuntimeWorld

↓

Publish

↓

DSP
```

という流れなら

Builderだけが

Runtimeを書けます。

Audio Threadは

読み取り専用です。

これが

Authority Singularization

そのものです。

---

# 12. 私ならどうするか

```cpp
RuntimeBuilder

↓

RuntimeWorld

↓

freeze()

↓

publish()

↓

Audio Thread

↓

Retire
```

freeze後は

static_assertレベルで

変更できない設計にします。

例えば

```cpp
const RuntimeWorld

std::shared_ptr<const RuntimeWorld>
```

まで徹底します。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★★ |
| ISRとの整合 | ★★★★★ |
| 報告書の分析  | ★★★★☆ |
| 修正案     | ★★★☆☆ |

## 私の結論

* **B07 の問題提起は非常に妥当**です。Publish 後に RuntimeWorld またはその配下オブジェクトが変更される可能性があるなら、ISR の前提そのものが崩れます。
* ただし、**`const` を追加するだけでは不十分**です。深い意味での Immutable 性（Deep Immutability）を保証する必要があります。
* Runtime と DSP の実行時状態（ディレイライン、FFT ワーク領域、フィルタ状態など）は明確に分離し、**RuntimeBuilder が完成・凍結（freeze）した RuntimeWorld のみを Publish する**構造にすることが、ISR・リアルタイム安全性・保守性のすべてにおいて最も望ましい設計です。

### 補足

ここまでレビューした項目を総合すると、**B07・B08・B17 は互いに独立した問題ではありません。**

* **B07**: Runtime が Immutable であること
* **B08**: Immutable Runtime を正しく Publish すること
* **B17**: Publish された Runtime の寿命を正しく管理すること

この3つは ISR アーキテクチャの中核を構成しており、個別修正ではなく、一体として設計・実装・検証することを推奨します。



## 第11回レビュー: B06「RuntimeBuilder と RuntimeWorld の責務分離の不完全性」

B06 は B07 の一段階手前に位置する問題であり、**「Immutable Runtime を生成する Builder の責務が十分に分離されているか」** を問うレビューです。

---

# 結論

| 項目       | 評価       |
| -------- | -------- |
| 問題提起     | **概ね妥当** |
| 重大度      | ★★★★★    |
| 修正案      | ★★★☆☆    |
| そのまま適用推奨 | いいえ      |

私の結論は

> **B06 は単なるリファクタリングではなく、ISR の成立条件に関わる設計上の問題です。**

---

# 1. Builderの責務

ISRでは

```text
Parameters

↓

Builder

↓

RuntimeWorld

↓

Publish
```

となります。

Builderだけが

```
可変
```

です。

RuntimeWorldは

```
完成品
```

です。

---

# 2. Builderがやるべきこと

Builderは

* IR生成
* FFT生成
* Filter生成
* Gain計算
* CrossfadePlan生成

など

すべて終えてから

RuntimeWorldを返します。

つまり

Builderは

```
Factory
```

です。

---

# 3. 報告書の指摘

報告書では

BuilderとRuntimeWorldの責務が混在し、

Runtime生成後も

Builder相当の処理が残っている可能性を指摘しています。

もし

```cpp
RuntimeWorld
{
    rebuildSomething();
}
```

のようなメソッドがあるなら

Builderの責務です。

---

# 4. なぜ危険か

例えば

```text
Runtime生成

↓

あとでIRを追加

↓

あとでGain更新
```

が可能なら

Immutableではありません。

Builderが

完成させていません。

---

# 5. RuntimeWorldはDTOに近い

RuntimeWorldは

設計的には

```text
Data Object
```

に近い存在です。

例えば

```cpp
struct RuntimeWorld
{
    FilterGraph;

    IRData;

    GainStructure;

    CrossfadePlan;
};
```

程度です。

ここに

```cpp
updateFilter()
```

などがあると

責務が混ざります。

---

# 6. 修正案について

報告書では

Builderへ処理を移すことを提案しているようです。

方向性は正しいです。

しかし

もっと徹底すべきです。

Builder以外は

Runtimeを生成できないようにします。

---

# 7. ISRとして望ましい構造

私なら

```text
Parameter Snapshot

↓

RuntimeBuilder

↓

Validation

↓

Freeze

↓

RuntimeWorld

↓

Publish
```

という段階を明確に分けます。

Validationが失敗したら

RuntimeWorldは生成しません。

---

# 8. Validationの位置

これは以前の

Oversampling解析でも問題になりました。

Auto Oversamplingが

```
0
```

のまま

Publishされそうになっていました。

本来は

Builder内で

```text
Resolve

↓

Validate

↓

Build
```

すべきです。

---

# 9. RuntimeBuilderは唯一の生成器

ISRでは

```cpp
RuntimeWorld(...)
```

を

誰でも呼べる設計は避けるべきです。

例えば

```cpp
RuntimeBuilder::build()
```

だけが

生成します。

---

# 10. Authority Singularizationとの一致

以前から検討している

```
Builder

↓

RuntimeStore

↓

DSP
```

という流れなら

Authorityは

Builder

だけになります。

RuntimeWorldは

生成されるだけです。

---

# 11. 将来の保守性

この構造にすると

将来

* 新しいDSP
* 新しいLimiter
* 新しいIR形式

が追加されても

Builderだけ修正すれば済みます。

RuntimeWorldは

変更されません。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★☆ |
| ISRとの整合 | ★★★★★ |
| 報告書の分析  | ★★★★☆ |
| 修正案     | ★★★☆☆ |

## 私の結論

* **B06 の問題提起は妥当**です。RuntimeBuilder と RuntimeWorld の責務が曖昧なままでは、Immutable Runtime の前提が崩れやすくなります。
* RuntimeWorld は「完成済み・読み取り専用」のデータ構造に徹し、Builder が構築・検証・凍結（Freeze）を完了してから Publish するべきです。
* また、RuntimeWorld の生成経路を RuntimeBuilder に一本化し、Validation・Build・Freeze を Builder の責務として明確化することで、ISR 設計・リアルタイム安全性・保守性を同時に向上させることができます。

---

### ここまでのレビューの総括

ここまでレビューした **B06〜B21** を俯瞰すると、問題は大きく4つのグループに分類できます。

| グループ        | 該当項目               |   優先度 |
| ----------- | ------------------ | ----: |
| ISRアーキテクチャ  | B06, B07, B08, B17 | ★★★★★ |
| Crossfade制御 | B09, B10, B11      | ★★★★☆ |
| DSPアルゴリズム   | B13, B20, B21      | ★★★★★ |
| 並行処理・Retire | B14                | ★★★★☆ |

この分類を見ると、**個々のバグを個別に修正するよりも、「ISR の責務整理」「Crossfade の状態機械」「DSP の演算品質」の3本柱として再設計する方が、修正量を抑えつつ根本的な改善につながる**と評価できます。



## 第12回レビュー: B05「Parameter Snapshot の一貫性（Snapshot Consistency）の欠如」

B05 は ISR のさらに上流、**RuntimeBuilder に渡される Parameter Snapshot が本当に一貫した状態なのか**という問題です。

これは ConvoPeq のこれまでの解析（Auto Oversampling の不具合や RuntimePublishWorld の設計）とも非常に関係が深い項目です。

---

# 結論

| 項目       | 評価                    |
| -------- | --------------------- |
| 問題提起     | **非常に妥当**             |
| 重大度      | ★★★★★                 |
| 修正案      | ★★★★☆                 |
| そのまま適用推奨 | 概ね可（Builder全体との整合が必要） |

私の評価では、

> **B05 は B06〜B08 よりさらに上流にある「ISR の入口」の問題であり、設計上の重要度は非常に高い**

となります。

---

# 1. Snapshotとは何か

ISRでは

```text
UI Parameter

↓

Snapshot Capture

↓

RuntimeBuilder

↓

RuntimeWorld
```

という流れになります。

Builderは

**Snapshotを絶対に信用する**

という前提です。

---

# 2. Snapshotに矛盾があるとどうなるか

例えば

```text
SampleRate = 48000

Oversampling = 4x

ProcessingRate = 96000
```

となっていたら

矛盾しています。

本来

```text
48000 × 4

=

192000
```

になるべきです。

Builderは

どちらを信じればいいか分かりません。

---

# 3. 以前のOversampling問題

以前解析した

Auto Oversamplingの問題では

```text
manualOversamplingFactor

=

0
```

のまま

Builderへ渡され

Publishが拒否されていました。

これは

**Snapshotが完成していなかった**

ことを意味します。

つまり

B05の指摘と一致します。

---

# 4. 報告書の指摘

報告書では

Snapshot取得中に

複数のParameterを別々に取得しており、

取得途中でUI変更が入ることで

不整合なSnapshotが生成される可能性を指摘しています。

これはマルチスレッドDSPでは

典型的な問題です。

---

# 5. なぜ危険か

例えば

```text
Band1 Gain

↓

UI変更

↓

Band2 Gain
```

なら

Snapshotは

```text
Band1 = 新

Band2 = 旧
```

になります。

Builderは

これを正常な状態と判断してしまいます。

---

# 6. SnapshotはAtomicであるべき

ここでいうAtomicとは

CPU命令ではありません。

意味としては

```text
全部取得

↓

一括で完成
```

です。

途中状態が存在してはいけません。

---

# 7. 修正案

報告書は

Snapshot取得を

一箇所へ集約する方向を提案しています。

これは非常に良い方向です。

ただし

私はさらに

Snapshot取得後

```text
Resolve

↓

Validate

↓

Freeze
```

まで行うべきだと思います。

---

# 8. Resolveの重要性

例えば

```text
Auto Oversampling
```

なら

Snapshotでは

```text
Auto
```

だけ持ちます。

Builderで

```text
Auto

↓

8x
```

へ変換します。

これで

Snapshotは

意味的に一貫します。

---

# 9. Validation

Builderは

Snapshotに対して

例えば

```text
SampleRate

Oversampling

ProcessingRate

IR

FilterCount
```

が

矛盾していないことを確認します。

Validation失敗なら

Publishしません。

---

# 10. Freeze

Validation成功後

```text
Snapshot

↓

Freeze

↓

RuntimeWorld
```

になります。

Freeze後は

変更できません。

ここで

B06

B07

へ繋がります。

---

# 11. ISR設計との一致

私なら

Builderを

```text
Capture

↓

Resolve

↓

Validate

↓

Freeze

↓

Build Runtime
```

の5段階に分けます。

これにより

責務が非常に明確になります。

---

# 12. 実運用での効果

この構造なら

例えば

* Automation
* MIDI Learn
* Preset読込
* UI変更
* CLI

すべて

同じSnapshot経路を通ります。

Builderは

入力元を気にしません。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★★ |
| ISRとの整合 | ★★★★★ |
| 報告書の分析  | ★★★★★ |
| 修正案     | ★★★★☆ |

## 私の結論

* **B05 の問題提起は非常に妥当**です。Parameter Snapshot が一貫していなければ、その後の RuntimeBuilder・RuntimeWorld・Publish がすべて不整合な状態を引き継ぐことになります。
* 修正は Snapshot の取得箇所を集約するだけでなく、**Capture → Resolve → Validate → Freeze → Build** の明確なパイプラインとして設計することを推奨します。
* このレビュー項目は、これまで ConvoPeq で発生した **Auto Oversampling の Publish 失敗**とも整合しており、実際の不具合事例から見ても設計改善の優先度は非常に高いと評価できます。

---

### 現時点での再評価

ここまで（B05〜B21）のレビューを通して見ると、報告書の中で**最も設計改善効果が高い項目**は次の5件です。

1. **B05**：Snapshot の一貫性保証（ISR の入口）
2. **B06〜B08**：Builder・Immutable Runtime・Publish の責務整理（ISR の中核）
3. **B17**：Runtime の寿命管理と Authority の統一
4. **B13**：NUPC の遅延アラインメント（DSP 品質）
5. **B20**：True Peak Detector のチャンネル処理（測定精度）

この5点は個別修正ではなく、**ISR 全体の設計見直しとしてまとめて取り組む価値がある**と判断します。



## 第13回レビュー: B04「Parameter Capture の責務分散・Single Source of Truth 違反」

B04 は B05（Snapshot Consistency）よりさらに上流のレビューです。

B05 が

> 「Snapshot が一貫しているか」

であるのに対し、

B04 は

> **「Snapshot を誰が作るのか」**

という **Single Source of Truth（SSOT）** の問題です。

これは、これまで ConvoPeq で目標としてきた **Authority Singularization** と完全に一致するテーマです。

---

# 結論

| 項目       | 評価            |
| -------- | ------------- |
| 問題提起     | **非常に妥当**     |
| 重大度      | ★★★★★         |
| 修正案      | ★★★★☆         |
| そのまま適用推奨 | 概ね可（責務の整理が必要） |

今回レビューした中では

**B05と同等に重要**

です。

---

# 1. Parameter Captureとは

ISRでは

```
UI

Automation

CLI

Preset

↓

Capture

↓

Snapshot

↓

Builder
```

となります。

重要なのは

**Captureは一箇所**

であることです。

---

# 2. SSOT

例えば

```
GUI

↓

gain
```

と

```
Preset

↓

gain
```

が

別々にSnapshotを書き換えると

Authorityが

二つになります。

これは

Single Source of Truthではありません。

---

# 3. 報告書の指摘

報告書では

Parameter Capture が

複数箇所に分散しており、

それぞれが独自に Snapshot を構築している可能性を指摘しています。

この問題提起は、ConvoPeq の設計目標（Authority Singularization）と整合しています。

---

# 4. なぜ危険か

例えば

```
GUI

↓

capture()
```

と

```
Automation

↓

capture()
```

が

別実装なら

片方だけ

新しいParameterが追加される

という事故が起きます。

---

# 5. 以前の解析との一致

以前解析した

Auto Oversamplingも

```
GUI

↓

Auto
```

↓

別箇所

↓

Builder

という流れで

0

が残っていました。

これも

Capture責務が曖昧だったことが

原因の一つです。

---

# 6. 修正案

報告書では

Capture処理を

一箇所へ集約することを提案しています。

これは非常に良い方向です。

---

# 7. 私なら

Capture専用クラスを作ります。

例えば

```cpp
ParameterCapture
{
    capture();
}
```

だけです。

Builderは

```cpp
Snapshot s =
capture.capture();
```

しか呼びません。

---

# 8. CaptureとResolve

Captureでは

まだ

```
Auto Oversampling
```

は

Auto

のままです。

Resolveは

Builderで

```
Auto

↓

8x
```

へ変換します。

Captureが

意味変換までやるべきではありません。

---

# 9. Captureの責務

Captureは

ただ

```
現在値を読む
```

だけです。

例えば

```
GUI

Automation

Preset
```

を

全部読んで

Snapshotへ詰めます。

それ以上のことはしません。

---

# 10. Builderとの境界

境界は

```
Capture

↓

Snapshot

↓

Builder
```

です。

Captureが

IR生成

FFT生成

Validation

などを始めると

責務が崩れます。

---

# 11. Authority Singularization

以前から検討している

```
ParameterCapture

↓

Snapshot

↓

RuntimeBuilder

↓

RuntimeWorld

↓

RuntimeStore
```

という構造なら

Authorityは

きれいに分離できます。

---

# 12. 将来の拡張

この構造なら

例えば

新しい

```
MIDI Mapping
```

を追加しても

Captureだけ修正します。

Builderは

変更不要です。

---

# 総合評価

| 項目       |    評価 |
| -------- | ----: |
| 問題の存在    | ★★★★★ |
| SSOTとの整合 | ★★★★★ |
| 報告書の分析   | ★★★★☆ |
| 修正案      | ★★★★☆ |

## 私の結論

* **B04 の問題提起は非常に妥当**です。Parameter Capture が複数経路に分散している場合、Single Source of Truth が崩れ、Snapshot の一貫性（B05）にも直接影響します。
* Capture の責務は「現在のパラメータ状態を一括取得すること」に限定し、Resolve・Validate・Build は RuntimeBuilder の責務として明確に分離するべきです。
* ConvoPeq の設計目標である **Authority Singularization** に照らしても、**ParameterCapture → Snapshot → RuntimeBuilder → RuntimeWorld → RuntimeStore** という単方向のデータフローに整理することが最も保守性・安全性・拡張性に優れた構成です。

### ここまでの全体像

ここまでレビューした B04〜B21 を俯瞰すると、報告書は大きく次の3層を改善対象として整理できます。

1. **入力層（Parameter Pipeline）**

   * B04：Capture の一元化
   * B05：Snapshot の一貫性

2. **ISR 基盤層**

   * B06：Builder の責務
   * B07：Immutable Runtime
   * B08：Publish の可視性
   * B17：Runtime の寿命管理

3. **DSP 実行層**

   * B09〜B11：Crossfade 制御
   * B13：NUPC 遅延補償
   * B20：True Peak
   * B21：Limiter/Gain

この分類は、ConvoPeq の現在の ISR リファクタリング計画とも整合性が高く、実装計画を立てる際の優先順位付けにも有用です。



## 第14回レビュー: B03「Parameter の正規化（Normalization）・意味解決（Semantic Resolution）の責務混在」

B03 は B04・B05 のさらに上流であり、「取得したパラメータを誰が正規化・意味解決するのか」を扱っています。

これは ConvoPeq で以前問題になった **Auto Oversampling（0 → 実倍率）** の設計と極めて密接に関係するレビューです。

---

# 結論

| 項目       | 評価              |
| -------- | --------------- |
| 問題提起     | **非常に妥当**       |
| 重大度      | ★★★★★           |
| 修正案      | ★★★★☆           |
| そのまま適用推奨 | 概ね可（責務境界の整理が必要） |

私の評価では、

> **B03 は B04・B05 を成立させるための前提条件であり、ISR パイプラインの意味論（Semantic Layer）に関する重要なレビューです。**

---

# 1. Normalizationとは

Parameterには

```text
GUI値

Auto

Default

None

%
Hz
dB
```

など

UI都合の値が含まれます。

DSPは

これらを直接扱うべきではありません。

---

# 2. 例

例えば

GUIでは

```text
Oversampling

Auto
```

ですが

DSPでは

```text
8x
```

が必要です。

つまり

```text
Auto

↓

8x
```

という

意味解決

(Semantic Resolution)

が必要になります。

---

# 3. 以前のAuto Oversampling問題

以前解析した

```text
manualOversamplingFactor

=

0
```

問題は

まさに

Semantic Resolution

が

遅すぎたことが原因でした。

Builderに

```text
0
```

が届いてしまったわけです。

この実例は

B03の指摘を裏付けています。

---

# 4. 報告書の指摘

報告書では、

Parameter が UI 表現のまま Builder や Runtime に渡されており、

意味解決（Normalization / Resolution）の責務が曖昧になっている可能性を指摘しています。

この方向性は非常に妥当です。

---

# 5. Captureではやるべきではない

ここが重要です。

Captureは

```text
読むだけ
```

です。

例えば

```text
Auto
```

なら

そのまま

Snapshotへ保存します。

Captureは

意味を知るべきではありません。

---

# 6. Builderで解決する

Builderでは

例えば

```text
Sample Rate

48000
```

なら

```text
Auto

↓

8x
```

へ変換します。

Builderは

DSP都合を知っています。

ここが責務です。

---

# 7. Runtimeには完成形だけ

RuntimeWorldには

```text
Auto
```

は存在しません。

代わりに

```text
oversamplingFactor = 8
```

だけがあります。

Runtimeは

DSP実行体だからです。

---

# 8. 修正案

報告書は

Normalization層を

Builderへ寄せることを提案しています。

私は

これに概ね賛成です。

ただし

Builder内部でも

段階を分けます。

---

# 9. 私なら

Builderは

```text
Snapshot

↓

Normalize

↓

Resolve

↓

Validate

↓

Build
```

にします。

Normalizeでは

例えば

```text
%

↓

Linear Gain
```

なども行います。

---

# 10. Resolveとの違い

Normalize

```text
GUI単位

↓

DSP単位
```

Resolve

```text
Auto

↓

8x
```

Validation

```text
矛盾検査
```

という三段階です。

---

# 11. Single Source

Normalizeが

GUIにも

Builderにも

DSPにも

あると

三重管理になります。

つまり

SSOT違反です。

Normalizeも

Builderだけに置くべきです。

---

# 12. ISRとの一致

私なら

```text
Capture

↓

Snapshot

↓

Normalize

↓

Resolve

↓

Validate

↓

Freeze

↓

RuntimeWorld
```

という

パイプラインにします。

この構造なら

各段階の責務が

非常に明確になります。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★★ |
| 設計上の重要度 | ★★★★★ |
| 報告書の分析  | ★★★★☆ |
| 修正案     | ★★★★☆ |

## 私の結論

* **B03 の問題提起は非常に妥当**です。GUI 向けの表現（Auto、%、dB、既定値など）をそのまま Runtime や DSP に持ち込むべきではありません。
* 意味解決（Semantic Resolution）と単位変換（Normalization）は Builder の責務として集約し、RuntimeWorld には「DSP がそのまま実行できる完成済みの値」だけを保持するべきです。
* 以前確認した **Auto Oversampling の不具合**も、この責務分離が不十分だったことと整合しており、**Capture → Snapshot → Normalize → Resolve → Validate → Freeze → RuntimeWorld** という明確なパイプラインを構築することで、ISR の設計思想とも一致した改善が期待できます。

---

## 現時点での総括（B03〜B21）

ここまでのレビューを見ると、報告書は次のような層構造で問題を整理していることが分かります。

| 層     | 対象レビュー             | 評価    |
| ----- | ------------------ | ----- |
| 意味解決層 | B03                | ★★★★★ |
| 入力層   | B04, B05           | ★★★★★ |
| ISR基盤 | B06, B07, B08, B17 | ★★★★★ |
| DSP制御 | B09, B10, B11      | ★★★★☆ |
| DSP品質 | B13, B20, B21      | ★★★★★ |
| 並行処理  | B14                | ★★★★☆ |

この構成は、**Parameter Pipeline → Runtime Pipeline → DSP Pipeline** という一方向のデータフローとして捉えると整合性が高く、ConvoPeq の ISR リファクタリング計画を整理するうえで有用な観点です。



## 第15回レビュー: B02「Parameter モデルと DSP モデルの責務混在」

B02 は、これまでレビューしてきた B03〜B08 のさらに上位にある設計レビューです。

B03 が「意味解決 (Semantic Resolution)」だったのに対し、B02 は

> **「そもそも UI の Parameter モデルと DSP の Runtime モデルを分離できているか」**

という、アーキテクチャ全体に関わる指摘です。

---

# 結論

| 項目       | 評価           |
| -------- | ------------ |
| 問題提起     | **非常に妥当**    |
| 重大度      | ★★★★★        |
| 修正案      | ★★★★☆        |
| そのまま適用推奨 | 概ね可（設計変更を伴う） |

これは B03〜B08 を包含する設計上の問題です。

---

# 1. Parameter Model と DSP Model

プロ用DSPでは通常、

```
GUI

↓

Parameter Model

↓

DSP Model

↓

DSP
```

という二層構造になります。

つまり

GUIが持つ情報と

DSPが持つ情報は

全く別物です。

---

# 2. Parameter Model

例えば

```
Gain = +3 dB

Oversampling = Auto

IR Path = xxx.wav

Preset = Rock
```

などです。

これは

**ユーザーの意図**

を表しています。

---

# 3. DSP Model

一方

DSP側は

```
linearGain = 1.41254

oversamplingFactor = 8

IR Pointer

Convolver

FilterGraph
```

になります。

こちらは

**実行状態**

です。

---

# 4. 報告書の指摘

報告書では、Parameter と Runtime（DSP 実行モデル）の境界が曖昧で、UI の概念や設定値が Runtime 側へ直接持ち込まれている可能性を指摘しています。

この問題提起は、ISR の設計思想とよく一致しています。

---

# 5. なぜ危険か

例えば

Runtimeが

```cpp
bool autoOversampling;
```

を持っていたら

DSPは

Auto

という概念を知ることになります。

しかし

DSPには

```
8x
```

しか必要ありません。

---

# 6. AutoはUI概念

例えば

```
Auto

Default

Factory

Preset
```

などは

すべて

UI都合です。

DSPが知るべきではありません。

---

# 7. ParameterとRuntimeの境界

理想は

```
Parameter

↓

Normalize

↓

Resolve

↓

Runtime
```

です。

Runtimeには

GUI由来の情報は

残りません。

---

# 8. 修正案

報告書は

Parameter構造と

Runtime構造を

完全分離する方向です。

私は

これにほぼ賛成です。

---

# 9. 私なら

例えば

```
ParameterState
```

と

```
RuntimeWorld
```

を

完全に別型にします。

例えば

```cpp
ParameterState
{
    autoOversampling;

    gainDb;

    presetName;
}
```

---

```cpp
RuntimeWorld
{
    gainLinear;

    oversamplingFactor;

    filterGraph;

    convolver;
}
```

です。

共通メンバーはありません。

---

# 10. Builderだけが橋渡し

Builderだけが

```
ParameterState

↓

RuntimeWorld
```

へ変換します。

これが

Semantic Layer

になります。

DSPは

ParameterStateを

一切知りません。

---

# 11. Authority Singularization

この構造なら

Authorityは

```
Parameter

↓

Builder

↓

Runtime
```

だけになります。

途中で

DSPが

Parameterを読むことはありません。

---

# 12. 将来のメリット

この構造では

GUI変更

Preset

CLI

Automation

などを

全部

ParameterStateへ入れるだけです。

DSPは

全く修正不要になります。

これは

保守性が非常に高い設計です。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★★ |
| 設計上の重要度 | ★★★★★ |
| 報告書の分析  | ★★★★★ |
| 修正案     | ★★★★☆ |

## 私の結論

* **B02 の問題提起は非常に妥当**です。Parameter モデル（ユーザーの意図）と Runtime モデル（DSP の実行状態）は責務が異なるため、同一の構造や概念を共有すべきではありません。
* RuntimeWorld には「DSP がそのまま実行できる情報」のみを保持し、GUI やプリセットに由来する概念（Auto、dB 表記、プリセット名など）は ParameterState 側に閉じ込めるべきです。
* **RuntimeBuilder を唯一の変換レイヤ**として、ParameterState → RuntimeWorld の変換を担わせる構造は、ISR・Single Source of Truth・Authority Singularization のすべてと整合します。

---

## このレビューに対する補足

ここまでのレビューでは、設計思想（ISR や SSOT）に基づく改善案も述べました。ただし、**実際の修正方針の妥当性を最終判断するには、最新版の `ConvoPeq.md` の該当実装（ParameterState、RuntimeBuilder、RuntimeWorld、Capture パイプライン）が報告書の指摘どおりになっているかをコードレベルで照合する必要があります。**

現時点では、報告書の問題提起自体は設計原則と整合していますが、「実際に ConvoPeq の最新版でその問題が残っているか」と「提案された修正が最適か」は、ソースコードとの照合によって初めて確定できます。これは残りの B01 を含めた最終総括でも重要な確認ポイントになります。



## 第16回レビュー: B01「アーキテクチャ全体の Authority（責務・所有権）の分散」

B01 は、このレビュー全体の総括に相当する項目です。

B02〜B21 が個別の問題点を扱っていたのに対し、B01 は

> **「ConvoPeq 全体のアーキテクチャとして Authority（責務・所有権）が分散していないか」**

という最上位の設計レビューです。

---

# 結論

| 項目       | 評価            |
| -------- | ------------- |
| 問題提起     | **非常に妥当**     |
| 重大度      | ★★★★★         |
| 修正案      | ★★★★☆         |
| そのまま適用推奨 | 概ね可（段階的移行を推奨） |

これは今回のレビュー全体の中で、

**最も重要な設計レビュー**

です。

---

# 1. Authorityとは

Authorityとは

> **「誰が最終決定権を持つか」**

です。

例えば

```text
Oversampling
```

を

GUI

Builder

Runtime

DSP

の全員が変更できるなら

Authorityは

4つあります。

これは危険です。

---

# 2. 理想構造

ISRなら

Authorityは

```text
GUI

↓

ParameterState

↓

Builder

↓

RuntimeWorld

↓

DSP
```

だけになります。

逆流はありません。

---

# 3. 報告書の指摘

報告書では、

Authority が

* Parameter
* Capture
* Builder
* Runtime
* Crossfade
* Retire

など複数箇所に分散しており、

結果として

Single Source of Truth が成立していない可能性を指摘しています。

この問題提起は、これまでの B02〜B17 の内容を包括するものです。

---

# 4. 以前の解析との一致

これまで私たちが解析してきた内容でも

Authority分散は

何度も現れていました。

例えば

以前指摘した

```text
PublicationIntent

CrossfadeAuthority

RuntimeStore

Epoch
```

の

四重Authority

です。

これは

まさに

B01で言う

Authority分散です。

---

# 5. Auto Oversampling

以前解析した

Auto Oversamplingも

```text
GUI

↓

Builder

↓

Validator
```

で

誰が

```text
0

↓

8x
```

へ変換するか

曖昧でした。

これも

Authorityが

一つではありません。

---

# 6. Crossfade

Crossfadeも

以前解析したように

```text
Crossfade

Retire

Runtime
```

が

寿命管理を

共有していました。

これも

Authority分散です。

---

# 7. Publish

Publishも

以前レビューした

B08

B17

では

```text
Runtime

Generation

Retire
```

が

複数箇所にありました。

これも

単一Authorityではありません。

---

# 8. 修正案

報告書では

Authorityを

一元化する方向です。

私は

概ね賛成です。

ただし

Authorityを

一箇所へ集めるだけではなく

**階層化**

すべきです。

---

# 9. 私なら

最終的には

```text
ParameterState
        │
        ▼
ParameterCapture
        │
        ▼
RuntimeBuilder
        │
        ▼
RuntimeWorld
        │
        ▼
RuntimeStore
        │
        ▼
Audio Thread
        │
        ▼
Retire
```

という

一本のパイプラインにします。

途中で

逆流はありません。

---

# 10. DSPは読むだけ

DSPは

```text
RuntimeWorld
```

しか知りません。

Parameterも

Builderも

知りません。

これが

ISRです。

---

# 11. RuntimeStore

RuntimeStoreだけが

```text
Publish

Swap

Retire
```

を担当します。

Crossfadeは

寿命管理をしません。

---

# 12. Builder

Builderだけが

```text
Parameter

↓

Runtime
```

へ変換します。

Normalize

Resolve

Validate

Freeze

も

Builderだけです。

---

# 13. Capture

Captureは

```text
読むだけ
```

です。

意味解決をしません。

---

# 14. Parameter

GUI

Preset

Automation

CLI

全部

ParameterStateへ

集約されます。

DSPは

存在を知りません。

---

# 15. このレビュー全体との一致

ここまでレビューした

B02〜B21

は

すべて

この構造に収束します。

つまり

```text
Authority Singularization
```

が

共通テーマです。

これは

以前策定した

**Practical Stable ISR Bridge Runtime**

とも一致しています。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| 問題の存在   | ★★★★★ |
| 設計上の重要度 | ★★★★★ |
| 報告書の分析  | ★★★★★ |
| 修正案     | ★★★★☆ |

## 私の結論

* **B01 はレビュー全体の総括として非常に妥当な問題提起**です。Authority が複数箇所に分散していると、Single Source of Truth が崩れ、B02〜B21 で指摘された多くの問題（Snapshot、Publish、Crossfade、Retire、True Peak など）の温床になります。
* RuntimeBuilder・RuntimeStore・ParameterCapture の責務を明確に分離し、**Authority を単方向のパイプラインへ集約する設計**は、ISR の基本思想とも一致します。
* ただし、**Authority の一元化は段階的に進めるべき**です。既存コードとの互換性や移行期間を考慮し、各段階で動作検証を行いながら責務を整理していくことを推奨します。

---

# 全16項目をレビューした総括

全項目を通して見ると、報告書の内容は次のように分類できます。

| 分類                       | 項目            |    評価 |
| ------------------------ | ------------- | ----: |
| アーキテクチャ（Authority / ISR） | B01〜B08, B17  | ★★★★★ |
| Crossfade 制御             | B09〜B11       | ★★★★☆ |
| DSP 品質                   | B13, B20, B21 | ★★★★★ |
| 並行処理・Retire              | B14           | ★★★★☆ |

一方で、**各修正案については慎重な評価が必要**です。

* **問題提起（バグの存在や設計上の懸念）**は、多くの項目で妥当性が高く、ConvoPeq が目指している ISR アーキテクチャとも整合しています。
* **修正方法**については、局所的な `if` 文の追加や `const` の付与だけでは不十分なものが多く見られます。特に B01〜B08・B17 は、個別修正ではなく **Authority Singularization** を軸にした設計全体の見直しとして取り組む方が、結果的に保守性・リアルタイム安全性・将来の拡張性の面で優れています。

最後に重要な点として、今回の評価は**報告書の内容そのものの妥当性**を中心に行いました。各項目について「最新版 ConvoPeq.md の実装が本当にその問題を含んでいるか」「修正案が現在のコードベースに最適か」は、最新版ソースコードとの照合によって個別に確認する必要があります。特に B13、B14、B20 のような DSP や並行処理に関わる項目は、実装レベルでの検証が最終判断には不可欠です。



