# ConvoPeq ISR Bridge Runtime 実装規約（AI向け）

## 0. 最上位原則

AI は：

- ISR purity
- abstraction beauty
- theoretical completeness

を目的にしてはならない。

最優先目的は：

```text
bridge runtime の事故率を下げること
```

である。

以下は禁止：

- 大規模rewrite
- subsystem 一括置換
- ownership model の全面刷新
- crossfade path の同時全面変更

---

## 1. 実装単位規約（最重要）

### 1.1 1PR = 1責務

1PRで変更してよい責務は1つのみ。

例:

- ObserveToken formalization のみ
- RTLocalState 分離のみ
- facade bypass CI のみ

禁止:

- ObserveToken + crossfade redesign
- RTLocalState + retire redesign
- validator redesign + publish redesign

### 1.2 “cleanup” 単独PR禁止

cleanup は必ず：

```text
trigger達成確認
```

と同一PRで行う。

禁止:

```text
「古そうだから削除」
```

### 1.3 動作変更と構造変更を混在禁止

禁止例:

- field rename + synchronization変更
- facade導入 + retire挙動変更
- mutable移動 + callback順序変更

---

## 2. Authority Migration 規約

### 2.1 authority transfer 前に旧authority削除禁止

必須順序:

1. 新authority導入
2. read path切替
3. write path切替
4. metrics確認
5. trigger達成確認
6. 旧authority削除

順序逆転禁止。

### 2.2 dual authority 期間を正式許容

AI は dual authority を：

```text
「即削除すべき異常状態」
```

として扱ってはならない。

bridge phase は許容状態。

ただし：

- metrics
- CI
- trigger

下で統制する。

---

## 3. Observe Topology 規約

### 3.1 callback observe固定を最優先

AI は：

- abstraction整理
- ownership整理

より先に：

```text
callback中 snapshot固定
```

を壊さないこと。

IR-A は最優先 invariant。

### 3.2 observe path 増殖禁止

新規 observe path 追加は禁止。

observe は：

```text
RuntimeExecutionView{snapshot, local}
```

へ収束させる。

### 3.3 RT-visible mutable 新設禁止

新規 mutable state を追加する場合：

- RTLocalState
- RTAuxMutable

以外は禁止。

---

## 4. Crossfade 規約（超重要）

### 4.1 crossfade は独立 subsystem として扱う

crossfade は：

- lifetime
- latency
- generation
- ownership
- timing

が交差する危険領域。

他フェーズと混ぜない。

### 4.2 crossfade rewrite 禁止

禁止:

- crossfade path 全面置換
- latency alignment 全面変更
- ramp algorithm 一括変更

許可:

- observable state 固定
- metrics追加
- generation drift 検出
- 局所移行

### 4.3 音響品質優先

ISR purity より：

- click
- pop
- latency discontinuity
- XRUN

を優先。

---

## 5. RuntimeGraph 規約

### 5.1 RuntimeGraph の責務追加禁止

RuntimeGraph に追加禁止:

- retire coordination
- ownership repair
- synchronization workaround
- validator cache
- rollback logic

### 5.2 immutable purity 強制禁止

AI は RuntimeGraph を：

```text
今すぐ完全immutable化
```

してはならない。

まず禁止するのは：

```text
RT-visible mutation
```

のみ。

---

## 6. Validator 規約

### 6.1 validator hard dependency 維持禁止

validator は：

- release safety net
- runtime dependency

になってはならない。

### 6.2 validator 削除禁止

一方で：

```text
「不要そう」
```

という理由で削除禁止。

validator は：

- smoke
- standard
- exhaustive

tier維持を優先。

---

## 7. CI / Trigger 規約

### 7.1 trigger は必ず機械判定

禁止:

```text
「十分整理されたら削除」
```

許可:

```text
AST rule 0件
runtime metric 0
symbol usage 0
```

### 7.2 grep rule 永続化禁止

grep は暫定。

AI は：

- AST
- symbol reference

への移行前提で設計すること。

### 7.3 CI rule self-test 必須

新規 CI rule は：

```text
rule自身のテスト
```

を持たなければならない。

---

## 8. RTAuxMutable 規約

### 8.1 RTAuxMutable の肥大化禁止

許可:

- counter
- timing
- telemetry
- debug scalar

禁止:

- pointer
- ownership
- cache
- graph
- DSP handle

### 8.2 convenience mutable 禁止

RTAuxMutable を：

```text
便利mutable倉庫
```

として使用禁止。

---

## 9. Rollback 規約

### 9.1 rollback 不可能変更禁止

AI は：

```text
flag無しの不可逆変更
```

を行ってはならない。

### 9.2 subsystem rollback 維持

以下は独立rollback可能を維持:

- publication
- retire
- crossfade

---

## 10. Metrics 規約

### 10.1 metrics explosion 禁止

metric追加時は必須:

- owner
- threshold
- retention
- action

### 10.2 「取るだけmetrics」禁止

運用actionを持たないmetricは禁止。

---

## 11. Cleanup 規約

### 11.1 deferred 永続化禁止

cleanup deferred には必須:

- trigger
- owner
- expiry

### 11.2 cleanup 優先実装禁止

cleanup は：

```text
安全性向上後
```

にのみ実施。

---

## 12. AI禁止事項（重要）

AI は以下を行ってはならない。

- 全面rewrite
- global mutable purge
- full ISR rewrite
- crossfade全面刷新
- RuntimeGraph肥大化
- validator削除
- metric大量追加
- flag依存複雑化
- facade多層化
- template abstraction 増殖
- “generic runtime framework” 化

---

## 13. 実装レビュー時の最優先評価軸

レビュー優先順位は固定。

1. XRUN悪化がないか
2. click/pop悪化がないか
3. rollback可能か
4. dual authority暴走していないか
5. observe path増殖していないか
6. RT-visible mutation増えていないか
7. cleanup先走りしていないか
8. purity追求になっていないか

---

## 14. 最重要メッセージ

AI は：

```text
「理想ISRを作る」
```

のではない。

作るべきは：

```text
「bridge runtime を長期間崩壊させない統制システム」
```

である。
