# Practical Stable ISR Bridge Runtime

## AI実装統治規約 v1.1（差し替え条文つき完全版）

- 対象: `Practical_Stable_ISR_Bridge_Runtime_完全実装_詳細設計_2026-05-31.md`
- 目的: AI実装時の解釈分岐を排除し、authority逆流・部分更新・観測経路逸脱・retire破綻を fail-closed で防止する。
- 優先順位: 本規約 > 実装都合 > 暫定最適化。

---

## 0. 適用範囲

本規約は以下に適用する。

- RuntimeWorld 構築/公開/観測/退役
- Publication queue / coordinator / verifier / evidence
- Runtime semantic（topology/routing/execution/publication/overlap/retire）

非対象（参考運用のみ）:

- UI装飾ロジック
- 開発補助スクリプト（Runtime authorityに触れないもの）

---

## 1. 絶対原則（Override不可）

1. fail-closed
   - 未検証・未確定は reject。
2. RuntimeWorld 単一authority
   - 意味決定源は RuntimeWorld のみ。
3. Partial semantic update 禁止
   - published world の field更新禁止。
   - `new world -> publish` の置換のみ許可。
4. Self-contained
   - RuntimeWorld は意味決定に外部 mutable state を参照してはならない。
5. Immutable published world
   - publish後は mutation 不可。

---

## 2. RuntimeWorld 契約

## 2.1 構造契約

RuntimeWorld は以下 semantic を持つ。

- topology
- routing
- execution
- publication
- overlap
- retire

## 2.2 Self-Contained Contract

禁止:

- `world + external mutable state` による意味決定
- singleton mutable state / global registry / external authority table 依存

許可:

- value
- immutable snapshot
- runtime-owned immutable object

## 2.3 Identity Contract

- `worldId`: world instance identity
- `generation`: 公開世代識別子（strict monotonic）
- `semanticHash`: 内容指紋（補助）

Authoritative semantic が変わる場合は必ず新 world を生成する。

## 2.4 Canonical Semantic Form

One Semantic = One Canonical Representation を強制する。

- 例: `fadeSamples` と `fadeMs` を同時authority化しない。
- 変換は projection 層でのみ許可。

## 2.5 Derived Semantic Non-Persistence

Derived semantic は persisted authority にしてはならない。

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

## 4.1 単一入口

- `publish(world)` のみ許可。
- graph/snapshot/intent単位 publish を禁止。

## 4.2 Queue Ordering Contract

PublicationIntent Queue を通る publish は以下を満たす。

- publish順 = generation順
- enqueue順と publish順の整合
- 後着世代公開後に旧世代公開する逆転禁止

## 4.3 Replacement Atomicity

必須順序:

1. new world visible
2. old world retire start

禁止順序:

1. old retire
2. new visible

## 4.4 Publication Completeness + Validity + Admission

publish前に必ず以下の順で検証する。

1. Completeness
2. Validity
3. Admission
4. Publish

### Completeness

- required authoritative semantic が全て存在。

### Validity

- 値域/符号/整合条件を満たす。
- 例: overlap disabled かつ fadeDuration > 0 は conflict。

### Admission

- topology valid
- routing valid
- semantic complete
- no semantic conflict
- no authority leak

---

## 5. Observe 契約

1. AudioThread observe source は RuntimeWorld のみ。
2. observe は read-only。
3. observer side mutation / lazy update 禁止。

禁止型参照（AudioThread）:

- RuntimeGraph*
- RuntimeBuildSnapshot*
- PublicationIntent*
- TransitionState*

---

## 6. Retire 契約

## 6.1 Bounded starvation

- `maxRetireDeferralEpochs`
- `maxRetireWallClockMs`

## 6.2 Retire escalation（矛盾回避）

しきい値超過時は即reclaimしない。まず escalation へ遷移する。

reclaim実行は以下成立時のみ許可:

- No Reader
- No Executor Reference
- No Pending Transition

## 6.3 Destruction ordering

`Published -> Retiring -> Retired -> Destroyed` の順序を固定する。

---

## 7. Overlap 契約

- overlap authority は `world.overlap` のみ。
- CrossfadePreparedSnapshot は ExecutorLocal としてのみ許可。
- ExecutorLocal が authority write してはならない。

---

## 8. Memory Ordering / Ownership / ABA 契約

## 8.1 Memory ordering

- publish path: release/acquire を必須。
- contract-critical経路で relaxed を禁止。
- 診断/統計の非意味データのみ relaxed を許可（要明示）。

## 8.2 Ownership transfer matrix

- Build -> Publish
- Publish -> Retire
- Retire -> Destroy

二重所有禁止。

## 8.3 ABA hazard

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

## Step 1: 探索

- grep探索
- serena系構造探索
- codegraph系経路探索

## Step 2: 3ソース一致確認

- 結果差分がある場合は実装禁止、再調査。

## Step 3: 影響解析

- authority
- observe path
- publication ordering
- retire safety

## Step 4: verifier更新

- 必須 verifier を追加/更新。

## Step 5: 実装

- 置換型変更（new world publish）のみ。

## Step 6: 検証

- Unit / Integration / Soak tier で検証。

### ツール利用不可時の代替プロトコル

- ツール障害時は手動探索ログ（対象ファイル、検索語、経路、未確定点）を残す。
- 代替証跡なしの実装は禁止。

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
- PR/Nightly/Release で閾値を別定義する。

---

## 16. v1.0 からの差し替え条文（必須反映）

### 16.1 fail-closed の文言差し替え

旧: fallback全面禁止
新: authority bypass fallback禁止。`publish reject -> previous world continue` は必須。

### 16.2 snapshot/crossfade の扱い差し替え

旧: snapshot完全排除 / crossfade snapshot禁止
新: snapshot/crossfade の authority化禁止。ExecutorLocal/diagnostic用途は許可。

### 16.3 retire 強制回収の差し替え

旧: しきい値超過で強制reclaim
新: しきい値超過で escalation。reclaimは safety条件成立時のみ。

### 16.4 publication queue 差し替え

旧: publish入口のみ定義
新: queue ordering（enqueue順・generation順・publish順）整合を必須化。

### 16.5 memory ordering 差し替え

旧: relaxed全面禁止
新: contract-critical経路で禁止。診断/統計のみ条件付き許可。

### 16.6 coverage差し替え

旧: verifier coverage 100%（分母不明）
新: 分母を定義し、tier別閾値で運用。

---

## 17. 改訂履歴

- 2026-05-31: v1.1 初版（差し替え条文つき完全版）
