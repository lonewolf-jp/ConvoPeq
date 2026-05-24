# bug4 監査メモ

対象: `doc/work/bug4.md`

このメモは、bug4.md の各指摘について、コード上の確認結果を踏まえて **再現条件** と **影響範囲** を整理したものです。判定は「有効 / 条件付きで有効 / 不成立」に分類しています。

---

## 1. `cblas_dscal` の Audio Thread 呼び出し

- 判定: **有効**
- 再現条件:
  - `AudioEngine::DSPCore::processInput()` または `processInputDouble()` が入力変換を実行する
  - `convo::input_transform::applyHighQuality64BitTransform()` が `gain != 1.0` で実行される
  - その内部で `cblas_dscal()` が呼ばれる
- 影響範囲:
  - Audio Thread 上で MKL BLAS を踏むため、初回呼び出し時の内部初期化やロックの可能性を排除できない
  - レイテンシ増加、ジッタ、最悪の場合は音切れのリスク
  - 入力変換経路全体に波及するため、影響は広い
- 補足:
  - これは bug4 の中で最も優先度が高い指摘です

## 2. `RuntimePublishWorld` の生ポインタ観測に RCU ガードがない

- 判定: **条件付きで有効**
- 再現条件:
  - `RuntimeStore::observe()` で返った生ポインタを、保護ガードなしで長く保持する
  - 同時に publish/retire が進み、旧 world が解放される
- 影響範囲:
  - `RuntimePublishWorld` の参照が UAF に近い状態になる可能性
  - `getRuntimePublishView()` 由来の view をガードなしで跨いで使うコードが増えると危険度が上がる
- 補足:
  - 現状、音声処理側では `RCUReaderGuard` を先に張る箇所があり、即時 UAF と断定はできない
  - ただし `observe()` 単体は契約依存が強く、設計上の穴は残る
  - 現在は `AudioEngine::ObservedRuntime` / `RuntimePublishView` により、control reader 側の観測寿命をスコープ化する経路も存在する

## 3. `SafeStateSwapper::tryReclaim` の複数スレッド呼び出しリスク

- 判定: **現時点では不成立**
- 再現条件:
  - もし `tryReclaim()` を複数スレッドから同時に呼ぶ構造へ変更した場合
- 影響範囲:
  - 同一 retired エントリの重複回収、double-free、head/tail の不整合
- 補足:
  - `tryReclaim()` には単一 consumer 前提のコメントがあり、Debug build では呼び出しスレッドの `jassert` も入っている
  - 現実装では `DeferredFreeThread` と `releaseResources()` 側の後処理が中心で、同時多発呼び出しの形には見えない
  - 将来の拡張リスクとしては残るが、現状のバグとしては弱い

## 4. `AllpassDesigner::applyAllpassToIR` の未使用パラメータと安全ガード不備

- 判定: **現状のバグとしては不成立**
- 再現条件:
  - この関数が実運用経路から呼ばれる場合
- 影響範囲:
  - もし使われれば、引数の意味と実装の齟齬が設計不整合になる
- 補足:
  - 現在のコードベースではこの関数の呼び出し箇所が見当たらない
  - したがって、現時点の実害は確認できない

## 5. DSP lifecycle の ownership 分散

- 判定: **有効な設計リスク**
- 再現条件:
  - `activeDSP`、`fadingOutDSP`、`runtimeStore`、`retireDSP` の経路が今後の改修で少しでもずれる
  - publish/retire の順序や退役先がズレる
- 影響範囲:
  - UAF、二重解放、古い DSP の残留、状態遷移の取り違え
  - 音声切替、フェード、再ビルド、releaseResources まで広く影響する
- 補足:
  - 現時点で壊れているとは言い切れないが、将来変更時の事故要因としては大きい

## 6. `NoiseShaperLearner` の `candidatePopulation` 同期不足

- 判定: **不成立**
- 再現条件:
  - `optimizer.sample()` と評価が同時に同じバッファを触る場合
- 影響範囲:
  - 候補係数の破損、学習結果の非決定化、まれなクラッシュ
- 補足:
  - 実コードでは `sample → evaluate → update` の順で進み、評価中に別スレッドが同じ母集団を書き換える構造は確認できない
  - ワーカーは読み取り専用で、指摘どおりの競合は現状では見えない

## 7. `LinearRamp::skip` の浮動小数点誤差・ゼロステップバグ

- 判定: **不成立**
- 再現条件:
  - `step == 0` の状態で `skip()` が呼ばれる場合
- 影響範囲:
  - ランプ進行のズレ、終端未収束、値の取り残し
- 補足:
  - 現実装では `numSamples >= remaining` の分岐で `current = target` に収束する
  - ゼロステップのケースも即座に破綻する構造ではない

## 8. `ConvolverProcessor::prepareToPlay` の例外安全性

- 判定: **不成立**
- 再現条件:
  - `makeAlignedArray` などで `std::bad_alloc` が発生する場合
- 影響範囲:
  - 起動失敗、既存エンジンの破棄、ホストクラッシュ
- 補足:
  - 実装には `std::bad_alloc` の捕捉とロールバックが既にある
  - bug4 の指摘は現状のコードには当たらない

## 9. `dspCrossfadeArmed_RT` の未初期化可能性

- 判定: **不成立**
- 再現条件:
  - `initialize()` 前に `armCrossfadeIfPending()` が走る場合
- 影響範囲:
  - 不定値読み取り、クロスフェード誤作動
- 補足:
  - メンバは宣言時に `false` 初期化され、`initialize()` でも再初期化される
  - 未初期化の再現条件は確認できない

## 10. `MKLNonUniformConvolver` の `applySpectrumFilter` メモリ断片化

- 判定: **低優先の最適化候補**
- 再現条件:
  - Message Thread で `applySpectrumFilter()` を繰り返し呼ぶ
  - `gain` / `gainIL` の一時バッファ確保が頻発する
- 影響範囲:
  - 断片化というより、確保コストの積み上がり
  - 実害は限定的で、即バグではない
- 補足:
  - RT ではなく Message Thread 専用のため、リアルタイム障害とは別枠
  - 現行実装では `reusableGain` / `reusableGainInterleaved` によりバッファ再利用が入っており、
    単純な「毎回確保・解放」状態ではない

## 11. コメント誤り

- 判定: **情報レベル**
- 再現条件:
  - コメントだけを信じて保守した場合
- 影響範囲:
  - 保守時の読み違い、誤修正の誘発
- 補足:
  - 実行時バグではない
  - ただしドキュメント品質としては修正候補

---

## 総合評価

- **即修正優先**: 1
- **設計改善として有効**: 2, 5
- **現状バグとしては弱い/不成立**: 3, 4, 6, 7, 8, 9, 10, 11

### 監査上の結論

bug4.md は、**1 と 5 を中心に妥当性が高い**一方で、**いくつかの項目は現コードでは既に解消済み、または呼び出し経路がなく実害が確認できない** という結果です。

したがって、監査メモとしては以下の扱いが妥当です。

1. まず 1 を修正対象として確定する
2. 2 と 5 を設計改善候補として別管理する
3. 3, 4, 6, 7, 8, 9, 10, 11 は「観察メモ」または「将来リスク」として保持する
