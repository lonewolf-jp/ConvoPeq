# ConvoPeq Practical Stable ISR Runtime 詳細設計 v1.2

本書は `v1.1` の後続改訂であり、**監査結果の MUST 以外（Should / 実装可能性ギャップ）**を解消するための詳細設計である。

- 参照元:
  - `doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_1.md`
  - `doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_0_監査結果.md`
- 上位拘束:
  - `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`
  - `doc/work5/ISR_Runtime_実装統治規約_v1_1.md`

---

## 0. この版で追加する解消対象（MUST以外）

v1.1 で解消済みの MUST 6項目は維持し、以下を追加で解消する。

1. `RuntimeCoordinator` 状態機械の未定義（実装可能性ギャップ）
2. Authority Inventory 更新の自動化不足（運用ギャップ）
3. Tier × PR SLA の未定義（運用品質ギャップ）
4. Soak failure taxonomy と Phase DoD の紐付け不足（追跡可能性ギャップ）
5. Runtime Safety Regression 判定の定量規則不足（判定可能性ギャップ）

---

## 1. 既存方針（v1.1）継承

- Safety-First 優先順位
- 5単一（observe / authority / publication / generation / retire）
- Documentation Scope Rule
- Governance Budget
- State Addition Exception 5条件
- RT-side retire 失敗時挙動
- Crossfade 境界
- Phase DoD ↔ verifier 紐付け

上記は v1.1 の規定をそのまま有効とし、v1.2 は追加詳細のみ規定する。

---

## 2. RuntimeCoordinator 状態機械（新設）

### 2.1 目的

`RuntimeCoordinator` を「authority 判定・publish 判定・retire 制御」の責務に限定しつつ、遷移条件を機械的に検証可能にする。

### 2.2 状態定義

| 状態 | 意味 | 許可操作 | 禁止操作 |
| --- | --- | --- | --- |
| `Bootstrapping` | 初回 world 構築前 | 初期 inventory 読込、初期 world 構築 | Audio Thread 起動 |
| `Ready` | 有効 world が1つ存在、通常運転 | consume / publish候補評価 / retire drain | 強制再初期化 |
| `Publishing` | 新 world 構築完了〜publish完了まで | single-unit publish / generation進行 | field-level publish |
| `Transitioning` | crossfade意味属性切替中 | overlap policy 適用（reject/coalesce/restart） | semantic merge |
| `Pressure` | retire backlog 圧力状態 | coalesce / throttle / deferred drain | RT側の挙動変更 |
| `ShuttingDown` | 停止・解放フェーズ | shutdown reclaim / retire 完了 | 新規 publish |
| `Faulted` | fail-closed 保護状態 | diagnostics 記録 / 安全停止 | 運転継続 |

### 2.3 遷移規則

| From | Event | To | ガード条件 | 失敗時 |
| --- | --- | --- | --- | --- |
| `Bootstrapping` | initial world built | `Publishing` | world non-null + immutable 完了 | `Faulted` |
| `Publishing` | publish success | `Ready` | generation monotonic | `Faulted` |
| `Ready` | transition request | `Transitioning` | overlap policy 決定済み | request reject |
| `Transitioning` | transition committed | `Ready` | semantic source singular | `Faulted` |
| `Ready` | queue pressure detected | `Pressure` | backlog slope > threshold | `Ready` 継続 |
| `Pressure` | backlog normalized | `Ready` | slope ≤ threshold for N windows | `Pressure` 継続 |
| `*` | shutdown requested | `ShuttingDown` | stop token set | `Faulted` |
| `ShuttingDown` | reclaim complete | terminal | outstanding world = 0 | `Faulted` |

### 2.4 不変条件（Coordinator Invariants）

1. `Ready` / `Transitioning` 中、Audio Thread が observe する world は常に non-null
2. publish 後 mutation 0
3. generation は strict monotonic
4. retire owner は常に単一
5. executor-local state は Coordinator から publish/export しない

### 2.5 実装境界（責務固定）

Coordinator は以下を持たない。

- DSP 実体 ownership
- IO / UI / persistence
- 再試行ポリシーの実行ロジック（方針判定のみ）

---

## 3. Inventory 自動化（新設）

### 3.1 目的

`Current/Post-Migration Authority Inventory` を手作業依存から CI 一貫性チェックへ移行する。

### 3.2 生成物

- `storage/isr_inventory/current_authority_inventory.json`
- `storage/isr_inventory/post_authority_inventory.json`
- `storage/isr_inventory/inventory_diff_report.json`

### 3.3 自動化フロー

1. PR pre-check で current inventory 生成
2. 変更後に post inventory 生成
3. diff 生成（追加/削除/authority_class 変更/observe_path 変更）
4. Budget / Legacy / Scope rule と突合
5. 不整合時 fail-closed

### 3.4 CI 判定ルール

| ルール | 判定 |
| --- | --- |
| 未分類 state 存在 | Fail |
| `LegacyTemporary` 追加で manifest 未登録 | Fail |
| authority source 増加 > 0 | Fail |
| observe path 増加 > 0（許可無） | Fail |
| semantic duplication > 2 | Fail |
| retirement_owner 不整合 | Fail |

### 3.5 監査トレーサビリティ

各 PR で必須添付:

- inventory diff 抜粋
- 影響した state 一覧
- 例外運用時は BreakGlassOverride ID

---

## 4. Tier × PR SLA（新設）

### 4.1 SLA 目的

変更リスクに応じて検証深度とマージ可否の時間条件を統一する。

### 4.2 PR クラス分類

| PR クラス | 代表変更 | 最低 Tier | 追加条件 |
| --- | --- | --- | --- |
| `Class-S` | 文書のみ / コメントのみ | `smoke` | runtime code 変更ゼロ |
| `Class-A` | runtime semantics 非変更リファクタ | `standard` | inventory diff が構造不変 |
| `Class-B` | world schema / generation / crossfade 境界 | `exhaustive` | soak 短時間（30分） |
| `Class-C` | retire lane / pressure / shutdown reclaim | `exhaustive` | soak 長時間（4h） |
| `Class-D` | break-glass を伴う例外変更 | `exhaustive` | 承認 + rollback 明記 |

### 4.3 SLA ルール

- `Class-S`: 1営業日以内に判定
- `Class-A`: 2営業日以内に判定
- `Class-B/C/D`: exhaustive 完了前に merge 不可
- release 直前は全クラス `exhaustive` 強制

### 4.4 SLA 逸脱時

- 判定期限超過は PR に `needs-revalidation` 付与
- 再実行で最新 commit を再評価
- 例外運用は BreakGlassOverride と同等の承認フロー必須

---

## 5. Runtime Safety Regression 判定（定量化）

### 5.1 ベースライン

- baseline は「直前の main merge 成功時点」
- 比較窓は同一 host 条件・同一シナリオ

### 5.2 判定指標

| 指標 | 記号 | 合格条件 |
| --- | --- | --- |
| XRUN count | $X$ | $X_{new} \le X_{base}$ |
| stale observe count | $S$ | $S_{new} \le S_{base}$ |
| retire backlog slope | $B$ | $B_{new} \le B_{base}$ |
| world leak count | $L$ | $L_{new} = 0$ かつ $L_{new} \le L_{base}$ |
| publication latency p99 | $P_{99}$ | $P_{99,new} \le P_{99,base} \times 1.05$ |

### 5.3 総合判定

$$
\text{SafetyPass} = (X \land S \land B \land L \land P_{99})
$$

`SafetyPass = false` は fail-closed。

### 5.4 例外許容

- 計測ノイズ等の一時上振れは 1 指標のみ、かつ +3% 以内、再計測 3 回で中央値判定
- それ以外は BreakGlassOverride 必須

---

## 6. Soak Failure Taxonomy ↔ Phase DoD マッピング（新設）

### 6.1 マッピング表

| Failure Class | 主要症状 | 主監視 Phase | DoD 連動 |
| --- | --- | --- | --- |
| Class-A audio corruption | 破音 / クリック | Phase 4, 5 | crossfade 境界逸脱 / publication 完全性 |
| Class-B generation drift | 世代不一致 | Phase 1, 5 | generation singularization / atomic publish |
| Class-C stale observe | 古い world 参照 | Phase 2, 5 | observe 単一化 / publish 単一化 |
| Class-D backlog divergence | 退避キュー増大 | Phase 6 | pressure governance |
| Class-E retention leak | world 解放漏れ | Phase 6 | shutdown reclaim / retire owner 単一 |
| Class-F authority duplication | 二重 authority | Phase 1, 3 | authority freeze / legacy removal |

### 6.2 失敗時アクション

- Class-A/B/C: 即 fail（merge block）
- Class-D/E: 連続 2 窓悪化で fail
- Class-F: 検出即 fail（許容期間なし）

---

## 7. 運用手順追補

### 7.1 PR 作成時チェックリスト（追加）

1. PR クラス（Class-S/A/B/C/D）を宣言
2. 目標 Tier（smoke/standard/exhaustive）を宣言
3. inventory current/post/diff を添付
4. safety regression 判定表を添付
5. taxonomy 該当 Class と監視結果を添付

### 7.2 マージ前ゲート（追加）

- Tier 達成
- SafetyPass true
- inventory diff 整合
- BreakGlassOverride（該当時）有効期限内

---

## 8. 受入基準（v1.2 追加分）

v1.1 の受入基準に加え、次を満たす。

1. Coordinator 状態機械が状態・遷移・不変条件として定義済み
2. inventory が current/post/diff の3点で CI 判定可能
3. PR クラス別 Tier/SLA が定義済み
4. Runtime Safety Regression が定量判定式で定義済み
5. failure taxonomy が Phase DoD に追跡可能

---

## 9. v1.1 → v1.2 差分サマリ

| 追加項目 | v1.1 | v1.2 |
| --- | --- | --- |
| RuntimeCoordinator 状態機械 | 未記載 | §2 で状態/遷移/不変条件を定義 |
| Inventory 自動化 | 方針のみ | §3 で生成物・CI判定ルールを定義 |
| Tier × PR SLA | 未記載 | §4 で PR クラス別規定を定義 |
| Safety Regression 定量判定 | 概念のみ | §5 で指標・式・例外条件を定義 |
| Taxonomy と Phase 連動 | 暗黙 | §6 で明示マッピング |

---

## 10. 最終判断原則

採用可否は次の2軸で判定する。

1. 5単一への収束を促進するか
2. SafetyPass を満たしたまま運用可能か

両方が Yes の場合のみ採用候補とする。
