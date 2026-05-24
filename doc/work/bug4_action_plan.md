# bug4 修正優先順位付きアクションプラン

対象: `doc/work/bug4.md`

このプランは、監査メモ `doc/work/bug4_audit_memo.md` をもとに、**実施順序** と **依存関係** が分かるように整理したものです。

## 現在の実施状況メモ

- P0-1 `cblas_dscal` の Audio Thread 呼び出しは、`src/InputBitDepthTransform.h` で AVX2 / スカラーループ置換済み。
- P1-2 `RuntimePublishWorld` 観測契約は、`ObservedRuntime` / `RuntimePublishView` により既存の RCU ガード経路が確認済み。
- P1-3 DSP ライフサイクルの所有権経路は、`activeDSP` / `fadingOutDSP` / `retireDSP()` のコメントを強化して追跡しやすくした。
- P2-4 `SafeStateSwapper::tryReclaim()` は既に単一コンシューマ前提の `jassert` / コメントが入っており、追加コード修正は不要。
- P2-10 `applySpectrumFilter()` のバッファ再利用は既に実装済みで、追加修正は不要。

---

## 目的

- Audio Thread のリアルタイム制約違反を先に除去する
- 設計上の所有権分散を減らし、将来の UAF / 二重解放リスクを下げる
- 現時点で不成立だった項目は、必要に応じて後続の観察・再監査対象として管理する

---

## 優先順位一覧

### P0: 即修正

> ステータス: ✅ 実施済み（2026-05-24）

#### 1. `cblas_dscal` の Audio Thread 呼び出しを置換する

- 対象: `src/InputBitDepthTransform.h`
- 対応方針:
  - `applyHighQuality64BitTransform()` 内の `cblas_dscal()` を廃止する
  - AVX2 またはスカラーループでゲイン適用に置換する
  - Audio Thread で MKL BLAS を踏まない構造にする
- 依存関係:
  - なし
- 完了条件:
  - `processInput()` / `processInputDouble()` 経路で MKL BLAS 呼び出しが残らない
  - Audio Thread からの初回実行でもブロッキング要因がない
- 検証:
  - `src/InputBitDepthTransform.h` のコードレビュー
  - `get_errors` / ビルド確認
  - 可能なら `cblas_dscal` 残存検索でゼロ確認

---

### P1: 高優先

> ステータス: ✅ 実施済み（2026-05-24）

#### 2. `RuntimePublishWorld` の観測契約を明文化・ガード化する

- 対象: `src/core/RuntimeStore.h`, `src/audioengine/AudioEngine.h`
- 対応方針:
  - `observe()` の raw pointer 利用に対して、RAII ガード付きの観測経路を優先する
  - `getRuntimePublishView()` を経由する呼び出し側が、観測寿命を跨がないよう整理する
  - 必要なら `ObservedRuntime` 風のラッパーを導入する
- 依存関係:
  - P0 の修正とは独立
- 完了条件:
  - 生ポインタ観測の契約が曖昧なまま使われる箇所を減らす
  - 実運用経路で観測寿命が明示される
- 検証:
  - `observe()` 呼び出し箇所の棚卸し
  - 音声スレッド／Timer スレッドでの利用経路確認

#### 3. DSP ライフサイクルの所有権経路を整理する

- 対象: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Commit.cpp`, `src/audioengine/AudioEngine.Timer.cpp`
- 対応方針:
  - `activeDSP` / `fadingOutDSP` / `runtimeStore` / `retireDSP` の役割を文書化する
  - 退役経路を集約し、どのスレッドが何を所有するかを明示する
  - 将来のリファクタ候補として、所有権の単一化方針を検討する
- 依存関係:
  - P1-2 と並行可
- 完了条件:
  - publish / retire / clear の責務分担が追える状態になる
  - レビュー時に所有権の流れを追跡しやすい
- 検証:
  - 退役経路の一覧化
  - publish / retire 系関数のコメント整備

---

### P2: 中優先（観察メモからの必要分のみ）

> ステータス: ✅ 実施済み（2026-05-24）

#### 4. 将来の拡張リスクを抑えるガードを追加する

> ステータス: ✅ 実質対応済み（単一コンシューマ前提のコメント・Debug assert あり）

- 対象: `src/SafeStateSwapper.h`
- 対応方針:
  - `tryReclaim()` が単一スレッド前提であることをコメントまたは `jassert` で明示する
  - 将来複数スレッド化する場合の前提を文書化する
- 依存関係:
  - P1-3 とは独立
- 完了条件:
  - 誤った並列利用が入り込んだときに早期検知できる
  - 現行コードが既にこの条件を満たしていることが分かる
- 検証:
  - `tryReclaim()` 呼び出し元の再確認

#### 5. `MKLNonUniformConvolver` の一時バッファ運用を見直す

> ステータス: ✅ 実装済み確認済み（バッファ再利用あり）

- 対象: `src/MKLNonUniformConvolver.cpp`
- 対応方針:
  - `applySpectrumFilter()` の一時バッファ確保コストを確認する
  - 必要なら再利用化を検討する
- 依存関係:
  - なし
- 完了条件:
  - 性能改善余地が明確になる
- 検証:
  - Message Thread での負荷確認
  - 実害が小さい場合は最適化見送りを明記

#### 6. コメント不整合を整理する

> ステータス: ✅ 実施済み（2026-05-24）

- 対象: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, その他の該当箇所
- 対応方針:
  - 実装と一致しないコメントを修正する
  - 保守時に誤解を招く文言を消す
- 依存関係:
  - なし
- 完了条件:
  - コメントと実装の齟齬が減る
- 検証:
  - コードレビュー

---

### P3: 保留・観察対象

> ステータス: ⏳ 継続監視中

#### 7. 現状不成立だった項目は再監査候補として保持する

- 対象:
  - `SafeStateSwapper::tryReclaim` の複数スレッド化懸念
  - `AllpassDesigner::applyAllpassToIR`
  - `NoiseShaperLearner` の `candidatePopulation` 同期
  - `LinearRamp::skip`
  - `ConvolverProcessor::prepareToPlay`
  - `dspCrossfadeArmed_RT`
- 対応方針:
  - 現時点では修正対象にしない
  - 将来の改修や仕様変更時に再評価する
- 完了条件:
  - 観察対象として明示されている
- 検証:
  - 次回の関連改修時に再チェック

---

## 推奨実施順序

1. **P0-1**: `cblas_dscal` の Audio Thread 置換
2. **P1-2**: `RuntimePublishWorld` 観測契約の明確化
3. **P1-3**: DSP ライフサイクルの所有権整理
4. **P2-4**: `SafeStateSwapper` の利用前提を明示
5. **P2-5**: `MKLNonUniformConvolver` の一時バッファ見直し
6. **P2-6**: コメント不整合の整理
7. **P3-7**: 観察項目の継続管理

---

## 実施時の注意

- `JUCE/` と `r8brain-free-src/` は編集しない
- Audio Thread ではブロッキング、alloc/free、ロック、I/O、例外、MessageManager アクセスを避ける
- 変更は最小単位で行い、各段階でビルドとエラー確認を入れる
- 設計改善は、即時修正と分けて小さく進める

---

## 完了時の判定基準

- P0 が完了している
- P1 の設計リスクが整理されている
- 観察対象が「修正済み」「保留」「継続監視」に分類されている

## 現在の結論

- 実装上の即修正項目は `bug4.md` の項目 1 と 5。
- 項目 2 は設計改善候補として継続観察。
- それ以外は現コードで不成立、または情報レベルとして整理済み。
