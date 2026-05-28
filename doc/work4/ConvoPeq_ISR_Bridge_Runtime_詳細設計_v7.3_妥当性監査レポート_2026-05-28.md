# ConvoPeq ISR Bridge Runtime 詳細設計 v7.3 妥当性監査レポート

- Date: 2026-05-28
- Target: `doc/work4/ConvoPeq_ISR_Bridge_Runtime_詳細設計_v7.3_2026-05-28.md`
- Basis:
  - `doc/work4/ConvoPeq_ISR_Bridge_Runtime_Operational_Survivability_基本設計_v7_2026-05-28.md`
  - `doc/work4/ConvoPeq_ISR_Bridge_Runtime_AI詳細設計・実装統制規約_v7.3_2026-05-28.md`
  - `doc/work4/ConvoPeq_ISR_Bridge_Runtime_v7.3_CIチェック実装案_grep_lint_2026-05-28.md`

---

## 0. 監査範囲と前提

本監査は、以下 4 軸で実施した。

1. 規約適合性（v7 / v7.3 とのトレーサビリティ）
2. 論理整合性（文書内矛盾・運用破綻リスク）
3. 実装可能性（現行コードフックとの接続可能性）
4. CI 機械検証可能性（fail-closed で運用可能か）

補足:

- `file:///c%3A/VSC_Project/ConvoPeq/doc/work4` の取得は「無効な URL」応答となり、Web取得では参照不可。
- 代替としてローカルワークスペース上の対象文書・関連実装を直接照合した。

---

## 1. 総合判定

- 文書妥当性（設計として）: **A-（高い）**
- 規約トレーサビリティ: **良好**
- 実装適合性（現行コード準拠）: **中程度（移行ギャップあり）**
- CI 実装可能性: **高い**

評価式（要約）:

$$
\text{Doc Validity}=\text{High},\quad
\text{Operational Enforceability}=\text{High},\quad
\text{Current Code Conformance}=\text{Medium}
$$

---

## 2. 妥当と判断した主要根拠

### 2.1 Authority 境界の固定が明確

- admission / drain / reclaim / residency の単一 authority が明示され、運用時の判断分岐が減る。

### 2.2 Tier0 優先の順序が運用リスクに一致

- inflow 封鎖 → bounded reclaim → ownership 固定の順が、実運用の破綻連鎖遮断に整合。

### 2.3 waitForDrain の意味論が過保証を回避

- bounded convergence observation API として定義され、shutdown 完全静止保証との混同を回避。

### 2.4 Snapshot/Seal 順序が機械検証可能

- `capture -> finalize -> seal -> publish` が順序契約として固定。

### 2.5 CI トレーサビリティが具体化

- Check ID と Rule の対応が明示され、運用・保守時の検証導線がある。

---

## 3. 実装照合で確認できた整合点

### 3.1 admission / drain フックの存在

- `submitRebuildIntent(...)`: 実装あり
- `isFullyDrained()`: 実装あり
- `waitForDrain(...)`: 実装あり（non-RT assertion + bounded timeout）

### 3.2 hysteresis 関連の存在

- `retireHighWatermark_`
- `retireLowWatermark_`
- `retireSaturationActive_`

上記が実装上存在し、設計意図と接続可能。

---

## 4. 指摘事項（改善推奨）

## 4.1 P0（高優先）

### P0-1: Rule-1M の境界定義を明文化すること

現状懸念:

- funnel 外（例: UI/Convolver 変化ハンドラ）でも `Suppressed` テレメトリが発火する箇所があり、
  「rebuild suppression 権限」と「snapshot/queue suppression」の境界が曖昧に読める。

推奨:

- 文書上で suppression を二層定義する。
  - `RebuildIntentSuppression`（admission funnel 専属）
  - `SnapshotQueueSuppression`（キュー/バッファ保護）

### P0-2: direct `requestRebuild` 移行状態の暫定規約を追加

現状懸念:

- 現行コードには `requestRebuild(sr, bs)` 呼び出しが複数残る。

推奨:

- Tier0 移行完了までの暫定 allowlist を文書に明記し、
  CI fail 条件を段階適用（warn→fail）で設計する。

## 4.2 P1（中優先）

### P1-1: Residency boundedness の定量化

- 各 residency に上限/警告域/強制ドレイン条件を追記すると、CI 機械判定が容易。

### P1-2: Telemetry owner マップの具体化

- policy JSON に counter→owner を具体名で追記すると Rule-6E 違反検知精度が上がる。

### P1-3: suppression reason table の内包

- v7.3 OPS-6 と対応する reason table を詳細設計本体にも明示すると、規約参照の往復が減る。

---

## 5. CI 実装可能性判定

判定: **実装可能（高）**

理由:

- `CI-ADMISSION-001..003` / `CI-SHUTDOWN-001` / `CI-RECLAIM-001` / `CI-RESIDENCY-001` / `CI-TELEMETRY-001` の設計粒度が、grep/lint + policy JSON で実装可能。
- definition site 除外・allowlist 除外方針が既に定義されている。

留意:

- allowlist の肥大化を防ぐため、`owner/issue/rationale/expiry` の必須化を維持すること。

---

## 6. 最終結論

対象詳細設計は、v7 基本設計および v7.3 統制規約に対して高い整合性を持ち、
Tier0/Tier1 実装のガイドとして妥当である。

一方で、現行実装との移行境界（特に suppression authority と direct `requestRebuild`）は、
規約文面の追加明確化により誤検知・解釈差をさらに低減できる。

結論:

- **採用可（要: 境界明文化の追記）**
- 優先対応は P0-1, P0-2

---

## 7. 追加評価（実運用観点・採用判断補強）

本監査結果は、ConvoPeq の現フェーズに対して次の観点で妥当と再確認した。

- 実運用適合性: 高い
- Tier0/Tier1 移行方針: 採用推奨
- 方向性: purity より survivability を優先

中核判断:

1. single admission authority を最優先で固定
2. drain authority を SSOT 化
3. reclaim を bounded cadence に収束
4. inflow bypass を先に閉塞

これは「理想化された ISR purity」よりも、運用時の破綻因子を優先度順に潰す方針であり、
ConvoPeq の現状構造に適合する。

---

## 8. 実装前に追加で固定すべき事項

### 8.1 suppression 定義域の分離

混同防止のため、少なくとも以下を分離して定義する。

- rebuild intent suppression
- queue admission suppression
- snapshot drop suppression

### 8.2 residency authority table のコード化

文書定義に加え、コード上も enum/contract で固定する。

```cpp
enum class ResidencyAuthority
{
  PublicationCoordinator,
  DeferredDeleteFallback,
  EpochRetire,
  ShutdownDrain,
};
```

### 8.3 suppression reason の固定

Tier0 では reason の無秩序な増殖を禁止し、最低集合を固定する。

- Saturation
- Duplicate
- Shutdown
- Obsolete
- InvalidState

### 8.4 requestRebuild の mechanical 封鎖

「禁止宣言」ではなく、機械検証まで接続して封鎖する。

- allowlist
- grep/lint
- CI fail
- PR block

---

## 9. 最終採用判断（補強版）

実運用で破綻しにくい ISR Bridge Runtime を目標とする限り、本監査の方向性は妥当である。

特に優先すべき順序は次で固定する。

1. bypass を閉じる
2. authority を一本化
3. reclaim を bounded 化
4. shutdown determinism を固定

結論（補強版）:

- **採用推奨**（Tier0/Tier1 の実装方針として有効）
- ISR purity の理想化は Tier2 以降で扱う
