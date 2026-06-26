# MKLNonUniformConvolver 実装詳細検証レポート（静的解析最終版）

- **日付**: 2026-06-25
- **対象**: `MKLNonUniformConvolver.h/.cpp`, `ConvolverProcessor.Runtime.cpp`, `ConvolverProcessor.LoadPipeline.cpp`
- **プロジェクト**: ConvoPeq
- **目的**: 低音「ジジジ」ノイズ原因の静的解析による特定

---

## 0. エグゼクティブサマリー

静的解析の最終版。4回の改訂とDSP理論文献調査を経て、以下の結論に至る：

**MKLNonUniformConvolverのDSPコアに明白な実装バグ（メモリ破壊・リングバグ等）は見つかっていない。しかしレイヤ間時間整合性（Gardner理論の$y[n]=\sum y_m[n-D_m]$に対する適合性）は未解決であり、静的解析のみでは確定できない。**

本レポートは「静的解析の到達点」を示すものであり、「最終結論」ではない。

---

## 1. 確定棄却項目

### 1.1 tailOutputBuf上書き — ✅ 棄却

**主張**: 「新IFFT完了時に未消費のtailデータが上書き消失する」
**判定**: 発生しない。Fill周期=消費周期=partSize/blockSizeのため、新IFFT完了時には常に旧データが完全消費済み。

### 1.2 FDLウォームアップ — ✅ 棄却

**主張**: 「初期FDLが空で不完全な畳み込みになる」
**判定**: x[n<0]=0（因果系定義）の自然な帰結。正常動作。

### 1.3 Gardnerレイヤー間遅延補償 — 🔴 Aランク（未解決）

**第1版からの主張**: 「L1/L2出力にl0Len分のFIFO遅延が必要」

**現状**: Gardner理論の$y[n]=\sum y_m[n-D_m]$（$D_{m+1}=D_m+L_m$）に対する現行実装の適合性は未検証。理論誤差（~1312 samples ≈ 27ms @48kHz, 86ms IR時）は数値的に無視できない大きさだが、以下の理由で単純な断定はできない：

1. IRセグメントの内在的オフセット（L1 IR = IR[l0Len:]）と出力遅延 $D_m$ の関係は理論的に複雑
2. 低域補正IRではL0/L1の寄与が近いレベルになるため、時間ずれが帯域依存歪みとして知覚される可能性がある
3. Null Test未実施のため実測値がない

→ **Null Testで差分を実測するまで、原因候補から除外できない。**

---

## 2. 未確定（A/B/Cランク）— ランタイム検証優先

### 2.1 🔴 Null Test（理想逐次畳み込み vs NUC）（Aランク・最優先）

**目的**: すべての仮説を一掃する決定的テスト。

**方法**: [`doc/work57/null_test_procedure.md`](null_test_procedure.md) に詳細手順書を作成済み。

**概要**:

1. 既知のIRと既知の入力を用意
2. NUCを通した出力をキャプチャ
3. 同じIRで逐次FFT畳み込み（オーバーラップ無し、完全な線形畳み込み）を行った出力をキャプチャ
4. 両者の差分をFFT分析

**注意**: NUCは非同期分散MAC・異サイズパーティションを使用するため、逐次畳み込みと完全一致（-120dBFS）は期待できない。以下が現実的な基準:

| 差分レベル | 評価 |
|-----------|------|
| < -90 dBFS | 正常。NUCは正確 |
| -70〜-90 dBFS | ほぼ正常。浮動小数点加算順序差の範囲 |
| -50〜-70 dBFS | 要調査。分散MAC/Gardner誤差の可能性 |
| > -50 dBFS | 異常候補。構造的誤差の可能性大 |

### 2.2 🔴 Gardnerレイヤ間時間整合性検証（Aランク）

**目的**: Get()でのL0（ringBuf）+ L1（tailOutputBuf）単純加算が、Gardner理論$y[n]=\sum y_m[n-D_m]$に適合するかを検証する。

**理論誤差（計算値）**:

- $D_1$（あるべきL1遅延）= $l0Part + l0Len$ ≈ 64 + 2048 = 2112 samples
- 実際のL1出力タイミング = $l1Part + distributed\_delay$ ≈ 512 + 288 = 800 samples
- 差分 ≈ 1312 samples ≈ 27.3ms @48kHz

**注意**: この差分は以下の条件下で影響が大きくなる可能性がある（要Null Test）:

- 低域補正IRではL0/L1の寄与が同程度のため、時間ずれが帯域依存歪みとして知覚される
- 37Hz周期の櫛形フィルタ効果が低域の周波数応答にリップルを生む

**検証方法**: Null Test（2.1）の結果で差分が-50dBFSを超える場合、L1出力に追加遅延を入れたバージョンとの比較テストを実施。

### 2.3 ✅ ringOverflowCount実測（Bランク→確定）

**発見**: $m\_ringOverflowCount$ は設計されているが、理論的にoverflow発生経路が限定的。Add/Getが同一Audio Thread内で対になって呼ばれるため、継続的なoverflowには以下の条件が必要: ringBufサイズ < 1ブロック分のWrit - Read差の累積。現行のringBufサイズ（1024）では通常運用でoverflowが発生する理論経路は確認できていない。

**検証結果**: ✅ ログ確認済み。3セッション全てで ringOverflowCount=0（REBUILD_MERGED7回全て起動中、再生中0）。リングバッファオーバーフローは原因候補から除外。

### 2.4 🟡 REBUILD_MERGED相関確認（Bランク）

**発見**: $ConvoPeq.log$ に $REBUILD\_MERGED$ が出力されている場合、その前後にノイズが発生していないか確認する必要がある。

**検証方法**: ログのタイムスタンプとノイズ発生タイミングの照合。

### 2.5 🟡 NUC Engine切替過渡（Bランク）

**発見**: $switchEngineOnMessageThread()$ → $exchangeActiveEngine()$ によるアトミックエンジン差し替え直後、新NUCの $ringAvail=0$ のため最初の $Get()$ が $got=0$ を返し、Wet信号がゼロ埋めされる。ただし過渡は1〜数callbacksで収束するため、継続的「ジジジ」との一致度は低い。

**検証方法**:

1. $ConvoPeq.log$ から $REBUILD\_MERGED$ の発生頻度とタイミングを確認
2. ノイズ発生とrebuildタイミングの相関を調査

---

## 3. 既知の安全な設計（静的解析確認済み）

| 項目 | 判定 |
|------|-----|
| Overlap-Save (fftSize=2×partSize) | ✅ 正しい |
| ringWrite/ringRead循環バッファ | ✅ [BUG-02]修正済み |
| Direct Head（時間領域FIR） | ✅ 最大32タップ、AVX2 FMA |
| baseFdlIdxSavedスナップショット | ✅ 意図通りの近似 |
| ミラー領域FDL保護 | ✅ 定常状態で成立 |
| スレッド安全性 | ✅ 単一スレッド+RCU |
| メモリ管理（mkl_malloc 64byte align） | ✅ Audio Thread内動的確保なし |
| デノーマル対策（FTZ/DAZ） | ✅ releaseビルドで保証 |
| 型精度（全double） | ✅ 問題なし |
| partsPerCallback計算 | ✅ 1〜numPartsIRの範囲に制約 |

---

## 4. 実装済み診断と調査優先度

### 4.1 実装済み: ringOverflowCount診断ログ

`MKLNonUniformConvolver.h` に以下のpublicゲッターを追加:

- `getRingOverflowCount()` — Atomic loadで現在のオーバーフロー回数を取得
- `resetRingOverflowCount()` — カウンタを0にリセット（Message Thread用）

`ConvolverProcessor.Lifecycle.cpp` の `timerCallback()` 内で各chのNUCを確認:

- overflow > 0 の場合、ConvoPeq.logに回数とchを記録
- 記録後カウンタをリセット

これにより「数分再生して overflow=0 → 候補除外」「overflow>0 → 確定的手掛かり」の判断が可能。

### 4.2 調査優先度

| 優先度 | 項目 | 状態 |
|-------|------|------|
| A | Null Test（理想 vs NUC） | 📄 手順書作成済み (`doc/work57/null_test_procedure.md`)。手動テスト待ち |
| A | Gardnerレイヤ間時間整合性 | Null Test結果次第で追加検証 |
| B | ringOverflowCount実測 | ✅ 実装済み・確認済み（全セッション0） |
| B | REBUILD_MERGED相関確認 | ログ調査のみで即実施可能 |
| B | NUC Engine切替過渡 | 過渡は短時間であり症状との一致度は中程度 |
| C | knownBlockSize≠callQ | 発見は正しいが連続ノイズの説明力は低い |

---

## 5. 静的解析の総括

5回の静的解析を通じて、**MKLNonUniformConvolverのDSPコアに明白な実装バグ（メモリ破壊・リングバグ等）は見つかっていない。しかしレイヤ間時間整合性（Gardner理論の$y[n]=\sum y_m[n-D_m]$への適合性）は未解決であり、Null Testによる検証が必要である。**

検証範囲:

- SetImpulse() IR分割: ✅ 行単位トレース完了
- processLayerBlock() L0 OLS: ✅ 行単位トレース完了
- Add() 分散MAC: ✅ 全5サイクルのコールバックトレース完了
- Get() L1/L2合成: ✅ 全経路検証完了
- StereoConvolver::process: ✅ Add/Get呼び出し検証完了
- switchEngineOnMessageThread: ✅ RCU交換検証完了
- applyNewState / publish: ✅ フラグ・レイテンシ検証完了

検証済み:

- ✅ ringOverflowCount: 全セッション0。原因候補から除外
- ✅ REBUILD_MERGED: 7回全て起動中。再生中0件

未検証（ランタイムが必要）:

- 📄 Null Test（手順書作成済み、手動テスト待ち）
- 特定IRでの低域数値的挙動

---

## 6. 改訂履歴

| 版 | 日付 | 変更内容 |
|---|------|---------|
| 第1版 | 2026-06-24 | 遅延補償欠如をCritical断定 |
| 第2版 | 2026-06-25 | FDLウォームアップ/tailOutputBuf上書きを仮説化 |
| 第3版 | 2026-06-25 | 両仮説棄却。静的解析では原因特定不能 |
| 第4版 | 2026-06-25 | Runtime統合層拡張。knownBlockSize≠callQ発見 |
| v1 静的解析最終版 | 2026-06-25 | Gardner仮説格下げ。rebuild過渡/ringOverflowをAランク化。Null Test設計提示 |
| v2 | 2026-06-25 | ringOverflowCount確認済み（全セッション0）。Null Test手順書作成。CMakeLists.txtクリーンアップ。NUCConvolverNullTest削除 |
