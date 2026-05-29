# ISR Runtime 詳細設計 v1.0 監査結果

- 監査対象: `doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_0.md`
- 上位拘束:
  - 基本計画: `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`
  - 実装統治規約: `doc/work5/ISR_Runtime_実装統治規約_v1_1.md`
- 反映先: `doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_1.md`
- 結論: **条件付き合格**。v1.1 で Must 6項目を反映済み。

---

## 1. 総合評価

| 観点 | 判定 | 備考 |
| --- | --- | --- |
| Safety-First との整合 | 合格 | §0.1 優先順位を明示 |
| 5単一（observe/authority/publication/generation/retire） | 合格 | §0.2 で収束ゴールを明示 |
| Audio Thread 禁止事項との整合 | 合格 | §2.2 で再掲 |
| Governance Hardened 仕様（v3.1 §5） | **不足** | Budget が DoD に紐付いていない |
| 実装統治規約 v1.1 との整合 | **不足** | Documentation Scope / State Addition Exception / Tier 紐付け |
| 内部整合性 | **不足** | Consume 初期化前未定義 / Retire RT 失敗時挙動 / Crossfade 境界 |
| 実装可能性（implementability） | **不足** | RuntimeWorld フィールド / Coordinator 状態機械 / verifier 紐付け |

---

## 2. v3.1 基本計画とのギャップ

### 2.1 Governance Budget が DoD に紐付いていない（Must）

- v3.1 §5.11 が定義する 4 種 Budget（Authority Migration / Observe Growth / Legacy Lifetime / Semantic Duplication）が v1.0 §6 の Phase DoD に数値として組み込まれていない。
- 結果として、Phase 完了の機械判定が不能。
- 対応: v1.1 §6.1〜§6.6 に Budget を全 Phase へ統合済み。

### 2.2 Soak failure taxonomy の参照不足（Should）

- v3.1 で定義された Class A〜F が v1.0 §8 で網羅されているが、Phase DoD との対応が無い。
- 対応: v1.1 §8.3 で taxonomy を維持。Phase ごとの紐付けは将来拡張（Should）。

---

## 3. 実装統治規約 v1.1 とのギャップ

### 3.1 Documentation Scope Rule 未取り込み（Must）

- 規約 §17 が要求する「コードのみ変更で PR を閉じない / 同一 PR で更新すべき文書群」が詳細設計に未明記。
- 対応: v1.1 §9.4 に必須更新対象 7 種を明文化。

### 3.2 State Addition Exception の必須条件不足（Must）

- 規約 §9.2 で要求される BreakGlassOverride / deadline / removal_phase / CI guard / soak validation の 5 条件のうち、v1.0 では一部しか言及されていない。
- 対応: v1.1 §9.2 に 5 条件＋LegacyTemporary 登録を必須化。

### 3.3 Tier 紐付け不足（Must）

- 規約 §12.3 の Tier 1-to-1 mapping（smoke / standard / exhaustive と具体スクリプト集合）が、Phase DoD へ紐付いていない。
- 対応: v1.1 §7.4 に Phase × 必須 verifier 表を追加。

### 3.4 Break-glass 期限到来時の挙動（Must）

- 規約 §11 は `expiration` 超過を fail-closed と規定するが、v1.0 では明文化されていない。
- 対応: v1.1 §10.4 に fail-closed（warning 不可）を明記。

### 3.5 Runtime Safety Regression 判定方法不明（Should）

- 規約 §1 の Safety-First を満たすか否かを実装 PR で判定する方法（メトリクス・ベースライン）が薄い。
- 対応: v1.1 §8.2 で「直前マージ時点ベースライン以下」を明文化。詳細指標の自動化は将来拡張。

---

## 4. 内部整合性（弱点）

### 4.1 Consume 戻り値 non-null の前提が未定義（Must）

- v1.0 §4.1 は「常に non-null world を返す」と記述するが、初回 publish 前の挙動が未定義。
- 対応: v1.1 §4.1 で「初回 publish 完了前は Audio Thread を起動しない」を明示。

### 4.2 RT 側 Retire 失敗時の挙動が未定義（Must）

- v1.0 §4.3 は enqueue 失敗時の RT 側挙動が空白。`QueueFull` / `Shutdown` で RT が何をするかが不明確。
- 対応: v1.1 §4.3 を表化:
  - `QueueFull`: RT は block/alloc/log/代替publish 不可。直前 world 継続 observe のみ。Non-RT で overflow lane 回収。silent drop 禁止
  - `Shutdown`: RT は新規 publish を期待せず最後の world で終了まで継続。Non-RT が全 outstanding を回収。

### 4.3 Crossfade 境界の不明確さ（Must）

- v1.0 §6.4 は「semantic merge 0」を要求するが、具体的にどの state が world 内/外かが列挙されていない。
- 対応: v1.1 §3.3 に列挙:
  - world 内: 「fade有無」「fade先 graph」など authoritative な事実
  - world 外: fade progression / interpolation phase / サンプルカウント / エンベロープ位置
- v1.1 §6.4 で再固定。判定原則も §3.3 に追加。

---

## 5. 実装ギャップ（実装可能性）

| 項目 | v1.0 状態 | v1.1 対応 |
| --- | --- | --- |
| RuntimeWorld フィールド構造 | 概念のみ | §3.3 に世界内/外境界＋判定原則 |
| RuntimeCoordinator 状態機械 | 未記載 | 未充足（Should・別書で詳細化） |
| Crossfade 境界列挙 | 抽象記述 | §3.3 / §6.4 で確定 |
| Inventory 自動化 | 提出物のみ | §5.1 にフィールド明示。自動化は CI 側で別途 |
| Break-glass 機械化 | 言及のみ | §4.4 と §10.4 で fail-closed 化 |
| Tier × PR SLA | 未記載 | §7.4 で Phase 必須 verifier 化（SLA 自体は Should） |

Should 項目（Coordinator 状態機械詳細・PR SLA・Inventory 自動化）は v1.1 採用後の別タスクで段階対応する。

---

## 6. 是正一覧（Must 6項目）

| # | 項目 | 反映先 |
| --- | --- | --- |
| 1 | Documentation Scope Rule の取り込み | v1.1 §9.4 |
| 2 | Governance Budget を Phase DoD に統合 | v1.1 §6.1〜§6.6 |
| 3 | State Addition Exception 5 条件の明示 | v1.1 §9.2 |
| 4 | RT 側 Retire QueueFull/Shutdown 挙動の確定 | v1.1 §4.3 |
| 5 | Crossfade 境界の列挙と再固定 | v1.1 §3.3 / §6.4 |
| 6 | Phase DoD ↔ verifier 表 | v1.1 §7.4 |

副次（Should）として §10.4（Break-glass 期限超過 fail-closed）も追加。

---

## 7. 受入結論

- v1.0 単体では Must 6項目の不足により採用不可。
- v1.1（差分マージ）で全 Must 反映済み。lint 0、上位2文書と矛盾なし。
- 次工程として Phase 1（Authority Freeze）のタスク分解と、Should 項目（Coordinator 状態機械、PR SLA、Inventory 自動化）の段階対応を推奨。

---

## 付録: 監査範囲外（明示）

- 音響アルゴリズム品質
- UI/UX
- vendor ソース（`JUCE/`, `r8brain-free-src/`）
- Coordinator 内部実装の具体コード設計（次工程）
