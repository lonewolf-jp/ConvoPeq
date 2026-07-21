# ConvoPeq ソースコード調査報告書（Part 9）

`AudioEngine.Processing.DSPCoreFloat.cpp`を精読し、既に精読済みの`DSPCoreDouble.cpp`と1行ずつ突き合わせて実装乖離（コピペミス）がないか検証しました。**新規バグなし**（良好な結果）です。

---

## 1. 検証内容

Float版とDouble版の`process()`/`processDouble()`は、コード中のコメント自体が「Double 版と同一パターン」と相互参照しているため、意図的に同一ロジックを2系統（内部処理はどちらも64bit double、入出力段のみFloat/Double差異）で保守する設計です。以下を突き合わせました:

- 入力段（`processInput`/`processInputDouble`）、bypassランプ設定、dry信号保存、オーバーサンプリング、DCブロッカー適用、EQ/Convolver処理順序分岐: **構造・変数名まで含めほぼ一致**。
- **bypassブレンド適用部分**（wet/dry線形クロスフェード）: 一見コードの入れ子構造が異なっている箇所を発見しました。

  - Double版: `if (oversamplingFactor > 1) { processDown(); ... if (bypassBlendRequested) { ...numSamplesでブレンド... } }`（downsampleとブレンドが同じifブロック内）
  - Float版: `if (oversamplingFactor > 1) { processDown(); ... }` の後、別ブロックで `if (oversamplingFactor > 1 && bypassBlendRequested) { ...numSamplesでブレンド... }`（downsampleとブレンドが別々のifブロック）

  中括弧の入れ子は異なりますが、**実行順序は完全に同一**（downsample→ブレンドの順で、両条件とも`oversamplingFactor>1`の場合のみ実行）であり、動作に差はないことを確認しました。単なるリファクタリング時のコードスタイル差で、バグではありません。
  - `oversamplingFactor==1`時のブレンドは`numProcSamples`、`oversamplingFactor>1`時のブレンドは`numSamples`を使用という一見不統一に見える変数選択も、`numProcSamples`はダウンサンプル前のオーバーサンプル後カウントのまま更新されない設計のため、`oversamplingFactor>1`時に正しい値を得るには`numSamples`（ダウンサンプル後の実際のサンプル数）を使う必要があり、**両版とも正しく使い分けられている**ことを確認しました。

## 2. 結論

Float/Double間の実装乖離（メモリに記載のある「AVX2分岐とスカラー分岐の動作乖離」と同種のコピペミスパターン）を意識して重点的に突き合わせましたが、**今回確認した範囲（入力〜bypassブレンドまで）では発見されませんでした**。EQ/Convolver処理より後段（出力段、レベルメーター更新等）はまだ未突き合わせです。

---

## 3. 調査範囲の更新

- `AudioEngine.Processing.DSPCoreFloat.cpp`（471行）: 精読完了。`DSPCoreDouble.cpp`との突き合わせは入力〜bypassブレンドまで完了、出力段以降は未実施。
- `ISRRuntimePublicationCoordinator.cpp`: `precheckPublish`, 状態遷移系（`markTransitionStart`/`markTransitionCommitted`）, `ShutdownScheduler`, `PriorityScheduler::escalateAllRetires`（未実装のプレースホルダと自己文書化されており問題なし）, `PublicationBuffer`（mutex+vector使用だが呼び出し元が存在しないデッドコードと判明）を確認。新規の確定バグなし。

---

## 4. 次のステップ（提案）

1. `DSPCoreFloat.cpp`/`DSPCoreDouble.cpp`の出力段以降（レベルメーター、ノイズシェーパー適用部分）の突き合わせ継続
2. `DSPCoreIO.cpp`/`DSPCoreToBuffer.cpp`
3. `ConvolverProcessor.*`一式
4. `ISRRetireRuntimeEx.cpp`
