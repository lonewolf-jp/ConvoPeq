# work58 音飛び原因解析報告（2026-06-25）

## 📋 概要

音楽再生中に複数回の音飛び（オーディオドロップアウト）が発生したため、ログを詳細に解析した結果、根本原因を特定しました。

### ログ情報

- **ログファイル**: `C:\Users\user\Desktop\ConvoPeq\develop\ConvoPeq.log`
- **解析日時**: 2026-06-25
- **サンプルレート**: 192kHz（最高設定）
- **バッファサイズ**: 1024 samples
- **内部オーバーサンプリング**: 192kHz x 4 = 768kHz（最高）

---

## 🔍 ログ解析結果

### 問題のあるイベントの統計

```
REBUILD_MERGEDイベントの発生回数: 7回
すべてのイベントで400msレイテンシが発生

イベントパターン:
- intentId: 7-10, 16, 17, 20
- reason: same_as_pending_would_merge
- class: Snapshot (一部 Structural)
- latencyMs: 400.000 (すべてのイベント)
- finalizeState: phase5_reduce_target
```

### 重要なログ出力例

```
[REBUILD_TELEMETRY] event=REBUILD_REQUESTED intentId=7 reason=enqueue_snapshot_command
                    class=Snapshot policy=Replaceable hash=0x0 fingerprint=0x4750cac7cf4d7390
                    finalizeState=N/A decision=accepted

[REBUILD_TELEMETRY] event=REBUILD_MERGED intentId=7 reason=same_as_pending_would_merge
                    class=Snapshot policy=Replaceable hash=0x0 fingerprint=0x4750cac7cf4d7390
                    finalizeState=phase5_reduce_target decision=merged latencyMs=400.000

[AudioEngine] setNoiseShaperType: newType=2 wasAdaptive=1
```

### 再発パターンの特徴

1. **連続したスナップショット要求**: `enqueue_snapshot_command` が短時間に複数回発行
2. **同一フィンガープrint**: `0x4750cac7cf4d7390` が繰り返し（同一スナップショットの連続要求）
3. **400ms待機**: マージのために400ms待機してから処理
4. **処理継続**: 400ms待機後、通常通りリビルドが継続

---

## 🔴 根本原因特定

### 原因の特定

**`ConvolverProcessor.h:173` で定義されている `REBUILD_DEBOUNCE_DEFAULT_MS = 400` ms**

```cpp
// src/ConvolverProcessor.h:173
static constexpr int REBUILD_DEBOUNCE_DEFAULT_MS = 400;
```

### 問題のメカニズム

1. **スナップショットの連続要求**: UI操作や内部イベントにより、短時間に複数のスナップショット要求が発行される
2. **デバウンス適用**: [AudioEngine.RebuildDispatch.cpp](src/audioengine/AudioEngine.RebuildDispatch.cpp#L204-L209) で、`same_as_pending_would_merge` の理由により、デバウンス時間が適用される
3. **400ms待機**: `REBUILD_DEBOUNCE_DEFAULT_MS` の値（400ms）だけ待機してからマージ
4. **オーディオドロップアウト**: 400ms は192kHzで約 **77,000サンプル** に相当し、明らかな音飛びを引き起こす

### 400msレイテンシの影響（内部オーバーサンプリング考慮）

| 入力サンプルレート | オーバーサンプリング | 処理レート | 400ms に相当するサンプル数 | オーディオ影響 |
|------------------|------------------|----------|--------------------------|--------------|
| 48kHz            | x8               | 384kHz   | 153,600 サンプル          | **致命的**    |
| 192kHz           | x4               | 768kHz   | **307,200 サンプル**      | **致命的**    |
| 192kHz           | x2               | 384kHz   | 153,600 サンプル          | **致命的**    |

**重要**: 最高設定（192kHz x 4 = 768kHz）では、400ms は **307,200 サンプル** に相当し、これは致命的なオーディオドロップアウトを引き起こします。

---

## 🔧 推奨解決策

### 解決策1: `REBUILD_DEBOUNCE_DEFAULT_MS` を下げる（推奨）

オーディオ処理のベストプラクティスに従い、デバウンス時間を **5-10ms** に下げます。

```cpp
// src/ConvolverProcessor.h:173
static constexpr int REBUILD_DEBOUNCE_DEFAULT_MS = 10; // 400ms → 10ms
```

#### 修正のメリット

- 10ms は 192kHz x 4 = 768kHz で **7,680 サンプル** に相当し、オーディオ処理の許容範囲内
- UIバースト吸収機能を維持しつつ、オーディオ品質を確保
- 既存のマージロジックを変更する必要なし

#### 修正のデメリット

- なし（オーディオ処理観点）

### 解決策2: スナップショットの連続要求を抑制する（補助）

スナップショット要求が連続して発行されないように、発行ロジックを改善します。

#### 実装方法

- スナップショット要求の発行前に、直近の発行時刻をチェック
- 短時間（例: 10ms以内）の連続要求を抑制

#### 修正のメリット

- 不必要なリビルド要求を削減
- CPU使用率の削減

#### 修正のデメリット

- 複雑度が増加
- デバウンス機能と二重管理の可能性

### 解決策3: マージ待機のロジックを改善する（代替）

[AudioEngine.RebuildDispatch.cpp](src/audioengine/AudioEngine.RebuildDispatch.cpp#L176-L178) で、スナップショットクラスに対しても、Structural クラスと同様の短いデバウンス時間（例: 10ms）を適用します。

#### 実装方法

```cpp
// src/audioengine/AudioEngine.RebuildDispatch.cpp:176-L178
latestWinsWindowMs = (kind == convo::RebuildKind::Structural)
    ? std::max(1, uiConvolverProcessor.getRebuildDebounceMs())
    : std::max(1, uiConvolverProcessor.getRebuildDebounceMs()); // Structuralと同じデバウンスを適用
```

ただし、このアプローチは推奨しません。なぜなら、既存のデバウンス時間（400ms）自体が問題であり、その根本原因を修正すべきだからです。

---

## 📊 推奨設定

### デバウンス時間の推奨値（内部オーバーサンプリング考慮）

| 入力サンプルレート | オーバーサンプリング | 処理レート | 推奨デバウンス時間 | 最大サンプル数 | オーディオ影響 |
|------------------|------------------|----------|------------------|--------------|--------------|
| 48kHz            | x8               | 384kHz   | 5-10ms           | 1,920-3,840 サンプル | ほぼ影響なし  |
| 192kHz           | x4               | 768kHz   | 5-10ms           | 3,840-7,680 サンプル | 許容範囲内    |
| 192kHz           | x2               | 384kHz   | 5-10ms           | 1,920-3,840 サンプル | ほぼ影響なし  |

### 推奨設定

```cpp
// src/ConvolverProcessor.h:173
static constexpr int REBUILD_DEBOUNCE_DEFAULT_MS = 10; // 最も安全な設定
```

**注**: 10ms は 192kHz x 4 = 768kHz で **7,680 サンプル** に相当し、これはオーディオ処理の許容範囲内です。

---

## 🎯 実装プラン

### Phase 1: 即時修正（推奨）

1. **`REBUILD_DEBOUNCE_DEFAULT_MS` の値を変更**
   - ファイル: `src/ConvolverProcessor.h`
   - 行番号: 173
   - 修正: `400` → `10`

2. **再ビルドとテスト**
   - コマンド: `cmake --build build --config Debug`
   - テスト: 音楽再生中にパラメータを変更し、音飛びが発生しないことを確認

### Phase 2: 検証（推奨）

1. **ログ解析**
   - `REBUILD_MERGED` イベントのレイテンシが 10ms 以下になっていることを確認
   - オーディオドロップアウトが発生しないことを確認

2. **性能評価**
   - CPU使用率が適切であることを確認
   - レイテンシが許容範囲内であることを確認

### Phase 3: 最適化（任意）

1. **UIパラメータ調整**
   - UIで `rebuildDebounceMs` を調整できるようにする
   - ユーザーがデバウンス時間をカスタマイズできるようにする

2. **自動調整**
   - サンプルレートに応じてデバウンス時間を自動調整
   - バッファサイズに応じてデバウンス時間を自動調整

---

## ✅ 検証方法

### テストシナリオ

1. **基本テスト**
   - 音楽再生中にパラメータを連続して変更
   - 音飛びが発生しないことを確認

2. **ストレステスト**
   - 高負荷状態でパラメータを連続して変更
   - 音飛びが発生しないことを確認

3. **長時間テスト**
   - 長時間（数時間）の音楽再生で音飛びが発生しないことを確認

### ログ確認

```
# REBUILD_MERGED イベントのレイテンシを確認
Get-Content "ConvoPeq.log" | Select-String "REBUILD_MERGED" | Select-String "latencyMs"

# 期待される出力: latencyMs=10.000, 20.000 など（400.000ではない）
```

### パフォーマンス指標

| 指標 | 修正前 | 修正後 | 目標 |
|------|--------|--------|------|
| REBUILD_MERGED レイテンシ | 400ms | 5-10ms | < 10ms |
| オーディオドロップアウト | 有り | 無し | 0 |
| CPU使用率 | 正常 | 正常 | 変化なし |

---

## 📚 関連ドキュメント

### 既知の問題と修正履歴

- **work56**: 音飛び原因分析（oversamplingFactor=0 問題の修正）
  - ファイル: `doc/work56/audio_dropout_design_resolution_jp.md`
  - 備考: work56 の修正は Phase1 で実施済み。Phase2（強く推奨）の修正が未実施の場合、同様の問題が再発可能性

### アーキテクチャドキュメント

- **Rebuild Telemetry**: リビルドの遠隔測定システム
  - ファイル: `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - 関連: `RebuildTelemetryEvent::Merged`, `RebuildTelemetryReason::SameAsPendingWouldMerge`

- **Snapshot 機能**: スナップショットの作成と管理
  - ファイル: `src/convolver/ConvolverProcessor.StateAndUI.cpp`
  - 関連: `enqueue_snapshot_command`

---

## 🔧 実装チェックリスト

### 修正前の確認

- [ ] 現在の `REBUILD_DEBOUNCE_DEFAULT_MS` の値を確認（400ms）
- [ ] ログファイルを保存（修正前の状態を記録）
- [ ] テスト計画を作成

### 修正の実施

- [ ] `src/ConvolverProcessor.h:173` の値を `400` → `20` に変更
- [ ] コードレビューを実施
- [ ] コミットメッセージを作成

### 修正後の検証

- [ ] 再ビルドを実行
- [ ] 単体テストを実行
- [ ] 統合テストを実行
- [ ] 音楽再生テストを実行
- [ ] ログ解析を実行（400ms レイテンシが消滅していることを確認）

### ドキュメント更新

- [ ] 修正内容を `doc/work58/` に記録（本ファイル）
- [ ] 変更ログを更新
- [ ] ユーザーマニュアルを更新（必要な場合）

---

## 📞 問題報告者へのフィードバック

### 問題の回答

音飛びの原因は、`ConvolverProcessor.h` で定義されている `REBUILD_DEBOUNCE_DEFAULT_MS = 400` ms という過度に長いデバウンス時間です。

400ms は192kHzで約77,000サンプルに相当し、これは明らかなオーディオドロップアウトを引き起こします。

### 推奨アクション

この値を 5-10ms に下げることで、オーディオドロップアウトを防ぐことができます。修正は簡単で、`src/ConvolverProcessor.h:173` の値を変更するだけです。

### 次のステップ

1. 修正を実施
2. 再ビルドとテスト
3. 音飛びが発生しないことを確認
4. 本番環境にデプロイ

---

## 📝 変更履歴

| 日付 | バージョン | 変更内容 | 著者 |
|------|-----------|---------|------|
| 2026-06-25 | 1.0 | 初版作成 | GitHub Copilot |

---

## 🔗 参考リンク

### ソースファイル

- [ConvolverProcessor.h](src/ConvolverProcessor.h#L173) - `REBUILD_DEBOUNCE_DEFAULT_MS` の定義
- [AudioEngine.RebuildDispatch.cpp](src/audioengine/AudioEngine.RebuildDispatch.cpp#L204-L209) - REBUILD_MERGED ロジック

### 関連メモリ

- [work56 音飛び原因分析 — 確定事項](/memories/repo/work56_audio_dropout_findings.md) - 以前の音飛び原因分析
- [core](/memories/repo/core.md) - プロジェクトのコア情報
- [rt_snapshot_notes](/memories/repo/rt_snapshot_notes.md) - ランタイムスナップショットに関するノート

---

## 📄 付録

### ログ分析スクリプト

```powershell
# REBUILD_MERGED イベントの抽出と解析
$logPath = "C:\Users\user\Desktop\ConvoPeq\develop\ConvoPeq.log"
Get-Content $logPath | Select-String "REBUILD_MERGED" | ForEach-Object {
    $line = $_.Line
    if ($line -match "intentId=(\d+).*latencyMs=([\d\.]+)") {
        $intentId = $matches[1]
        $latency = $matches[2]
        Write-Output "intentId=$intentId, latency=$latency"
    }
}

# 400ms レイテンシのカウント
Get-Content $logPath | Select-String "latencyMs=400" | Measure-Object
```

### 関連コードスニペット

```cpp
// src/ConvolverProcessor.h:173
static constexpr int REBUILD_DEBOUNCE_DEFAULT_MS = 400; // 修正前

// src/ConvolverProcessor.h:173
static constexpr int REBUILD_DEBOUNCE_DEFAULT_MS = 20;  // 修正後
```

---

**報告作成日**: 2026-06-25
**バージョン**: 1.0
**ステータス**: 原因特定済み、修正待ち
