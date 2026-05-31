# Practical Stable ISR Bridge Runtime

## AI実装統治規約 v1.2（評価結果統合版）

- 対象: `Practical_Stable_ISR_Bridge_Runtime_完全実装_詳細設計_2026-05-31.md`
- 由来: v1.1に妥当性評価結果（採用/修正/追加）を統合した版
- 目的: AI実装時の解釈余地を排除し、authority逆流・部分更新・観測逸脱・retire破綻を fail-closed で防止する
- 優先順位: 本規約 > 実装都合 > 暫定最適化

---

## 0. 適用範囲

本規約は以下に適用する。

- RuntimeWorld の構築/公開/観測/退役
- Publication queue / coordinator / verifier / evidence
- Runtime semantic（topology/routing/execution/publication/overlap/retire）

非対象（参考）:

- UI装飾ロジック
- Runtime authorityに触れない補助スクリプト

---

## 1. 絶対原則（Override不可）

1. fail-closed
   - 未検証・未確定は reject。
2. RuntimeWorld 単一authority
   - 意味決定源は RuntimeWorld のみ。
3. Partial semantic update 禁止
   - published world の field 更新を禁止。
   - `new world -> publish` の置換のみ許可。
4. Self-contained
   - RuntimeWorld は意味決定に外部 mutable state を参照してはならない。
5. Immutable published world
   - publish 後 mutation 不可。

> 修正反映: 「fallback全面禁止」ではなく、**authority bypass fallback 禁止**とする。`publish reject -> previous world continue` は必須。

---

## 2. RuntimeWorld 契約

### 2.1 構造契約

RuntimeWorld は以下 semantic を持つ。

- topology
- routing
- execution
- publication
- overlap
- retire

### 2.2 Self-Contained Contract

禁止:

- `world + external mutable state` による意味決定
- singleton mutable state / global registry / external authority table 依存

許可:

- value
- immutable snapshot
- runtime-owned immutable object

### 2.3 Identity Contract

- `worldId`: world instance identity
- `generation`: 公開世代識別子（strict monotonic）
- `semanticHash`: 内容指紋（補助）

Authoritative semantic が変わる場合は必ず新 world を生成する。

### 2.4 Canonical Semantic Form

One Semantic = One Canonical Representation を強制する。

- 例: `fadeSamples` と `fadeMs` の同時 authority 化を禁止
- 変換は projection でのみ許可

### 2.5 Derived Semantic Non-Persistence

Derived semantic を persisted authority として保持してはならない。

---

## 3. Semantic Dependency Graph 契約

依存方向を以下に固定する。

```text
Topology
 ↓
Routing
 ↓
Execution
 ↓
Publication
 ↓
Retire
```

禁止:

- 逆依存（例: publication -> routing）
- 循環依存

---

## 4. Publication 契約

### 4.1 単一入口

- `publish(world)` のみ許可
- graph/snapshot/intent 単位 publish を禁止

### 4.2 Queue Ordering Contract

PublicationIntent Queue を通る publish は以下を満たす。

- publish順 = generation順
- enqueue順と publish順の整合
- 後着世代公開後に旧世代公開する逆転禁止

### 4.3 Replacement Atomicity

必須順序:

1. new world visible
2. old world retire start

禁止順序:

1. old world retire
2. new world visible

### 4.4 Completeness + Validity + Admission

publish前に必ず以下の順で検証する。

1. Completeness
2. Validity
3. Admission
4. Publish

#### Completeness

- required authoritative semantic が全て存在

#### Validity

- 値域/符号/整合条件を満たす
- 例: overlap disabled かつ fadeDuration > 0 は conflict

#### Admission

- topology valid
- routing valid
- semantic complete
- no semantic conflict
- no authority leak

---

## 5. Observe 契約

1. AudioThread observe source は RuntimeWorld のみ
2. observe は read-only
3. observer side mutation / lazy update 禁止

禁止型参照（AudioThread）:

- RuntimeGraph*
- RuntimeBuildSnapshot*
- PublicationIntent*
- TransitionState*

> 修正反映: 「snapshot完全排除」ではなく、**snapshot/crossfade の authority化禁止**。ExecutorLocal/diagnostic用途は許可。

---

## 6. Retire 契約

### 6.1 Bounded starvation

- `maxRetireDeferralEpochs`
- `maxRetireWallClockMs`

### 6.2 Retire escalation

しきい値超過時は即reclaimしない。まず escalation へ遷移する。

reclaim実行は以下成立時のみ許可:

- No Reader
- No Executor Reference
- No Pending Transition

### 6.3 Destruction ordering

`Published -> Retiring -> Retired -> Destroyed` を固定する。

---

## 7. Overlap 契約

- overlap authority は `world.overlap` のみ
- CrossfadePreparedSnapshot は ExecutorLocal としてのみ許可
- ExecutorLocal が authority write してはならない

---

## 8. Memory Ordering / Ownership / ABA 契約

### 8.1 Memory ordering

- publish path は release/acquire 必須
- contract-critical経路で relaxed を禁止
- 診断/統計の非意味データのみ relaxed を許可（要明示）

### 8.2 Ownership transfer matrix

- Build -> Publish
- Publish -> Retire
- Retire -> Destroy

二重所有禁止。

### 8.3 ABA hazard

識別には `worldId + generation + epoch` の組を使用する。

---

## 9. Semantic Transaction 契約

許可状態:

- Building
- Validated
- Committed
- Published
- Rejected

禁止:

- PartiallyBuilt / PartiallyCommitted の外部可視化

Rejected は evidence 出力を必須とする。

---

## 10. 探索・実装プロトコル（強制）

### Step 1: 探索

- grep探索
- serena系構造探索
- codegraph系経路探索

### Step 2: 3ソース一致確認

- 結果差分がある場合は実装禁止、再調査

### Step 3: 影響解析

- authority
- observe path
- publication ordering
- retire safety

### Step 4: verifier更新

- 必須 verifier を追加/更新

### Step 5: 実装

- 置換型変更（new world publish）のみ

### Step 6: 検証

- Unit / Integration / Soak tier で検証

#### ツール利用不可時の代替プロトコル

- ツール障害時は手動探索ログ（対象、検索語、経路、未確定点）を残す
- 代替証跡なしの実装は禁止

#### 新規実装時のグラフ整合

- `graphに存在しない修正` は即禁止ではなく、
  1) 実装
  2) 再インデックス
  3) 差分検証
  を必須化する

---

## 11. 必須 verifier セット

1. schema completeness verifier
2. semantic validity verifier
3. runtime admission verifier
4. self-contained world verifier
5. semantic dependency graph verifier
6. publication queue ordering verifier
7. replacement atomicity verifier
8. generation monotonicity verifier
9. visibility monotonicity verifier
10. observe path verifier
11. observe forbidden-type verifier
12. overlap authority verifier
13. retire safety verifier
14. retire escalation verifier
15. hidden authority verifier
16. authority exhaustiveness verifier
17. semantic alias verifier
18. multi-writer prohibition verifier
19. memory ordering contract verifier
20. ownership transfer verifier
21. ABA hazard verifier
22. deterministic build verifier
23. semantic conflict verifier
24. semantic equivalence verifier

---

## 12. Severity Table（必須）

- Warning: 診断補助、実行継続可
- Error: PR gate fail
- Fatal: 即停止、再設計フェーズへ戻す

最低限以下は Fatal:

- RuntimeWorld以外のauthority起点
- partial semantic update
- generation/visibility rollback
- queue ordering逆転
- retire safety違反でのreclaim

---

## 13. 禁止事項（抜け穴封じ）

- 互換のための旧authority経路維持
- snapshot/graph/intent/transition の authority化
- 性能理由による authority bypass
- observer side mutation
- lazy retire による無期限滞留
- hash単独での semantic 同値判定

---

## 14. 失敗条件（即停止）

以下が1つでも成立した場合、実装停止。

- 探索3系統のいずれか未実施（または代替証跡なし）
- authority 未分類
- RuntimeWorld以外からobserve
- queue ordering違反
- partial publish/partial update存在
- graph/snapshot authority残存

---

## 15. 最終品質ゲート

全て満たすこと。

- RuntimeWorld単一authority
- publish/observe/retire の契約分離
- self-contained達成
- hidden authority = 0
- authority exhaustiveness 達成
- deterministic build 成立
- verifier coverage 定義済み分母に対して閾値達成

### coverage定義（明示）

- 分母: `Authority + Invariant + Verifier + ContractTest + EvidenceHook`
- tier別閾値（本版で固定）:
  - PR: 85%以上
  - Nightly: 95%以上
  - Release: 100%

---

## 16. v1.1からの追加/修正条文（評価反映）

1. fallback文言を明確化（authority bypass禁止 + 前world継続必須）
2. snapshot/crossfadeは「完全排除」ではなく「authority化禁止」へ統一
3. retireは即reclaim禁止、escalation後に安全条件成立時のみreclaim
4. queue orderingを三者整合（enqueue/generation/publish）で規定
5. graph未存在修正の扱いを再インデックス前提へ修正
6. relaxed使用を contract-critical経路で禁止、統計のみ条件付き許可へ明確化
7. verifier coverageの分母とtier閾値を具体化

---

## 17. 改訂履歴

- 2026-05-31: v1.1 初版（差し替え条文つき完全版）
- 2026-05-31: v1.2 作成（妥当性評価結果の統合、運用閾値明確化）
