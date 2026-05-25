# P0-1 ObserveToken formalization 専用PR 差分設計

作成日: 2026-05-24
対象ブランチ: `main`
上位計画: `doc/work/bridge_runtime_migration_plan.md`
厳守規約: `doc/work/ISR_Bridge_Runtime_AI_暴走防止規約.md`

---

## 1. このPRの責務（1PR=1責務）

本PRの責務は **ObserveToken の責務形式化のみ** とする。

- 許可: observe enter/exit と generation pin の責務明文化
- 禁止: publish/retire/graph mutation/cache ownership の導入
- 禁止: crossfade / retire / validator / CI ロジック変更

---

## 2. 変更しないこと（非スコープ固定）

以下は本PRで変更しない。

- Audio処理アルゴリズム
- crossfade 遷移・遅延整合ロジック
- RuntimePublicationCoordinator / RetireRuntime の挙動
- `RuntimeGraph` の責務
- cleanup 実行

---

## 3. 現行コード根拠（as-is）

### 3.1 token 実体

- `src/core/ObservedRuntime.h`
  - `ObservedRuntime` は
    - `EpochDomainReaderGuard guard`
    - `const GlobalSnapshot* ptr`
    - `ownerThreadId`
    を保持する move-only オブジェクト

### 3.2 生成点

- `src/core/SnapshotCoordinator.h`
  - `observeCurrentRuntime(int)` が `ObservedRuntime` を返す

### 3.3 使用点

- `src/audioengine/AudioEngine.h`
  - `observeCurrentRuntime()` public API
  - `RuntimePublishView` が `ObservedRuntime` を保持
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - Audio callback で `m_coordinator.observeCurrentRuntime(kAudioEpochReaderIndex)` を使用
- `src/audioengine/AudioEngine.Snapshot.cpp` / `AudioEngine.Timer.cpp`
  - control thread で observe 利用
- `src/SpectrumAnalyzerComponent.cpp`
  - UI thread で `engine.observeCurrentRuntime()` を使用

---

## 4. 差分方針（最小差分・非破壊）

本PRは **挙動変更なし** を原則とし、型責務の形式化だけを行う。

### 方針A（採用）: 互換エイリアス方式

1. `ObservedRuntime` を現行どおり維持
2. `ObserveToken` という概念名を型エイリアスで導入
3. ドキュメントコメントで「許可責務 / 禁止責務」を明文化
4. 既存呼び出し側のシグネチャ変更は実施しない

この方式により、bridge runtime への影響をゼロに近づける。

### 方針B（不採用）: 全呼び出し側の一括リネーム

- 理由: 「1PR=1責務」「大規模置換禁止」に反するため不採用

---

## 5. 予定差分（ファイル単位）

## 5.1 `src/core/ObservedRuntime.h`（変更）

- 追加:
  - `ObserveToken` 概念の型エイリアス（non-breaking）
  - 責務コメント（許可/禁止）
- 維持:
  - move-only 制約
  - owner thread 検査
  - `get()` / `operator bool()` の挙動

## 5.2 `src/core/SnapshotCoordinator.h`（変更候補）

- 変更候補は最小2案。
  - 案1: コメントのみ更新（API変更なし）
  - 案2: `using ObserveToken = ...` を使った戻り値型の明示
- 既存API互換を優先し、呼び出し側変更を発生させない案を採用

## 5.3 `src/audioengine/AudioEngine.h`（変更しない / コメントのみ候補）

- 実装・シグネチャ変更は行わない
- 必要なら `observeCurrentRuntime()` の責務コメントを補強

---

## 6. 機械判定可能な完了条件

P0-1完了は次の条件を満たすこと。

1. `ObserveToken`（または同義型）の責務定義がコード上で明文化されている
2. token 型に publish/retire/graph mutation/cache ownership 相当 API が追加されていない
3. 既存 observe 呼び出し点の件数が増加していない（新規 observe path 追加 0）
4. ビルド成功（Debug）

---

## 7. 検証計画（P0-1 PR内）

1. 静的確認
   - `ObservedRuntime` / `ObserveToken` の公開API差分を確認
   - 呼び出し点増加の有無を検索で確認
2. ビルド確認
   - 既存 Debug ビルドタスクでコンパイル成功
3. 診断確認
   - 対象ファイルの diagnostics がゼロ

---

## 8. rollback 設計

本PRは挙動非変更のため rollback は単純。

- 方法: `ObservedRuntime.h` / `SnapshotCoordinator.h` の差分取り消し
- 影響面: 型名コメント/エイリアスの除去のみ

---

## 9. レビュー観点（固定）

1. observe固定（IR-A）を壊していないか
2. runtime 挙動が変わっていないか
3. dual authority 統制に影響を与えていないか
4. 新規 mutable/state を増やしていないか
5. purity志向の過剰抽象化を入れていないか

---

## 10. 実装順（このPR内）

1. `ObservedRuntime.h` へ概念名と責務制約コメントを追加
2. 必要最小限で `SnapshotCoordinator.h` の表現を整合
3. 呼び出し側は原則無変更
4. 機械判定項目・ビルド・診断を実施

---

## 11. 判断メモ

P0-1は「ObserveToken formalization」であり、機能追加フェーズではない。
したがって本PRは **構造定義の明文化だけ** で閉じる。
