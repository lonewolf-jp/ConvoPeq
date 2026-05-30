# AI実装統治規約 v1.1（改善版）

- 作成日: 2026-05-30
- 対象: Practical Stable ISR Bridge Runtime（M1〜M6）
- 目的: `implementation_priorities_bridge_runtime.md` に基づくAI実装の統治基準を明確化する。
- v1.1の改善点:
  - RT禁止事項の補強
  - Publication失敗時fail-safeの分岐表化
  - 完了判定の回復性を数値基準で固定
  - Reject条件の適用対象を明確化

---

## 第1章 基本原則

### G-1 目的固定原則

AIは常に以下を最優先目的とする。

- semantic divergence（意味論ドリフト）の低減
- 実運用安定性の向上
- 既存Runtimeの安全な段階移行

以下を主目的としてはならない。

- Final ISR Runtime の完成
- RuntimeWorld 全面移行
- 理論的純粋性の追求
- アーキテクチャの美化
- 将来拡張のみを理由とした大規模再設計

### G-2 非破壊移行原則

Bridge Runtime安定化フェーズでは既存動作維持を最優先とする。

- 単一マイルストーン/単一タスクの遂行を原則とする。
- M1実施中にM5実装へ踏み込んではならない。

---

## 第2章 Authority Governance

### A-1 Authority Source 固定原則

実装前に必ず T2-1 Semantic Source Table を参照する。

- 定義されていない authority の新設を禁止する。

### A-2 Authority 増殖禁止

Authority View導入後も authority 数を増やしてはならない。

- 形式的な集約構造体を導入しても、callback配下で別経路再取得が残る場合は違反。

### A-3 Callback 再取得禁止

Audio callback 配下で以下の追加取得を禁止する。

- `readAudioRuntimeView(...)`
- `getRuntimeSnapshot(...)`
- `consumeCrossfadePreparedSnapshot(...)`

Authority取得は callback 冒頭で1回のみ許可される。

### A-4 Observe Path Collapse 完了条件

以下をすべて満たすこと。

- callback本体のauthority取得経路が実質1経路
- 補助関数へのauthority引渡しが単一路線
- callback配下でauthority再取得が0件
- RuntimeGraph / RuntimeSnapshot / CrossfadePreparedSnapshot の直接参照が排除される

---

## 第3章 RT Safety Governance

### R-1 RT禁止事項

AudioThread経路で以下を禁止する。

#### メモリ割当/解放

- `new` / `delete`
- `malloc` / `free`
- `vector::resize`（RT経路）

#### ロック/同期

- `std::mutex` / `std::recursive_mutex` / `std::shared_mutex`
- `std::unique_lock` / `std::lock_guard`
- `std::future` / `std::async` / `std::promise`
- `MessageManager` アクセス

#### I/O・リソース

- `std::fstream`
- `std::filesystem`
- コンソール/ファイルログ出力（RT経路）

#### 例外・重計算（プロジェクト実態反映）

- `try-catch`（RT経路）
- `std::shared_ptr` の新規持ち込み（RT経路）
- `libm`重依存計算の追加（RT経路）

### R-2 RT変更許可条件

Audio callback 修正時は以下を証明すること。

- lock増加なし
- allocation増加なし
- wait増加なし
- block操作増加なし

証明できない変更は禁止。

### R-3 RT性能退行禁止

callback内で以下を増加させてはならない。

- authority取得回数
- atomic読み取り回数
- atomic書き込み回数
- virtual dispatch回数
- DSP実行前準備コスト

---

## 第4章 Publication Governance

### P-1 PublicationInvariant v1 強制

以下 I-1〜I-6 を必須とする。

- I-1: world frozen
- I-2: world sealed
- I-3: closure valid
- I-4: publication state legal
- I-5: fade transition legal
- I-6: retire registration complete

### P-2 Publish入口増殖禁止（外部入口）

新規の**外部**publish入口を追加してはならない。

許可されるpublish入口:

- Commit
- Timer
- PrepareToPlay
- ReleaseResources

注: 既存入口内部の共通化（共通関数化/ラップ）は許可。

### P-3 PublicationInvariant bypass 禁止

以下を禁止する。

- Invariant無視
- 強制publish
- 一部条件のみ検証
- debug時のみ検証
- 経路ごとの独自ルール

### P-4 Publish失敗方針（fail-safe分岐表）

| 失敗種別 | 必須動作 | 禁止動作 |
| --- | --- | --- |
| I-1 / I-2 失敗（frozen/sealed不整合） | publish中止、現行runtime継続、診断記録 | 強制publish |
| I-3 失敗（closure無効） | publish中止、safe fallback、診断記録 | dangling参照を伴う継続 |
| I-4 失敗（state遷移違反） | publish中止、state保全、診断記録 | 遷移無視で続行 |
| I-5 失敗（fade違法遷移） | publish中止またはlegal policyへ丸め、診断記録 | 不正fadeのまま実行 |
| I-6 失敗（retire未登録） | publish中止、retire登録修復を優先 | retire抜けでpublish |

---

## 第5章 Retire Governance

### T-1 閾値変更管理

以下変更時は理由を文書化すること。

- `retireHighWatermark_`
- `retireLowWatermark_`
- `retireSaturationActive_`

### T-2 Telemetry 削除禁止

以下監視項目を削除してはならない。

- `retireQueueDepth`
- `fallbackQueueDepth`
- `retireHighWatermark`
- `retireLowWatermark`

### T-3 回復性維持原則

飽和検出だけでなく回復確認を必須とする。

- saturation detection
- saturation recovery
- queue drain confirmation

### T-4 Baseline 管理

以下は変更管理対象とする。

- metric名称
- metric単位
- sampling interval
- aggregation window
- threshold値

---

## 第6章 M5保護規約

### W-1 RuntimeWorld先行実装禁止

M5完了前にRuntimeWorldをauthority化してはならない。

禁止:

- RuntimeGraph除去
- DSPCore参照除去
- ExecutionTopology導入（本実装）
- RuntimeWorld authority化

許可:

- 棚卸し
- 依存分析
- 設計スパイク
- ドキュメント作成

### W-2 Bridge Runtime優先

将来利便性のみを理由にM5実装を前倒ししてはならない。

---

## 第7章 実装レビュー規約

各タスク完了時に以下を提出する。

- R-Report-1: 変更ファイル一覧
- R-Report-2: 対象タスク（例: T1-1, T3-1）
- R-Report-3: DoD達成証拠
- R-Report-4: RT影響評価（allocation/lock/wait/atomic read/write）
- R-Report-5: Authority Flow変更（該当時のみ）
- R-Report-6: 新規Invariant（該当時のみ）
- R-Report-7: 計画外変更（「なし」または詳細）

---

## 第8章 自動却下条件

以下に該当する実装はレビュー前に却下する。

- Reject-1: M1実施中にM5内容へ踏み込んだ
- Reject-2: Authority Sourceを増殖させた
- Reject-3: Audio callback内でauthorityを再取得した
- Reject-4: RT lockを追加した
- Reject-5: RT allocationを追加した
- Reject-6: PublicationInvariantをbypassした
- Reject-7: 計画外アーキテクチャ変更を行った
- Reject-8: Telemetryを削除した
- Reject-9: 運用試験ゲートを省略した（実装変更を伴うタスクに適用）

---

## 第9章 完了判定

Practical Stable ISR Bridge Runtime 完了判定は、以下をすべて満たした場合のみ認める。

### 構造条件

- Observe Path Collapse 完了
- Snapshot Semantic Unification 完了
- PublicationInvariant Centralization 完了
- Retire Governance 文書化完了

### 運用試験条件

#### Long Run IR Switch Test

- 30分以上連続実行で異常なし

#### Publish Burst Test

- 100回以上連続publish実施
- invariant逸脱なし
- `retireQueueDepth` が許容レンジへ回復
- `fallbackQueueDepth` が許容レンジへ回復

#### Retire Queue Saturation Test

- 飽和発生後に正常回復すること

#### Crossfade Stress Test

以下が発生しないこと。

- click
- pop
- state inconsistency
- semantic divergence

### 回復判定の数値基準（v1.1追加）

- 観測窓: `T4-3` で定義した aggregation window を使用
- 許容レンジ: baseline ± 設定閾値（`T4-3` 定義値）
- 収束条件: 許容レンジ内サンプルが連続M回（Mは `T4-3` で固定）

---

## 統治原則（再確認）

Bridge Runtimeフェーズでは、

「理想アーキテクチャを作ること」ではなく、

「既存Runtimeを壊さず semantic divergence を減らすこと」

を最優先とする。

Final ISR Runtime への移行は、本計画完了後の次段階とする。
