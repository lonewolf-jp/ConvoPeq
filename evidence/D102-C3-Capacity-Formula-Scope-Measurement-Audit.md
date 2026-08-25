# D102-C3 — Capacity Formula / Scope Measurement & Contract Eligibility Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0** / 数値採用 **0**）
- **判定**: **PASS**（構造的 compatibility 確認完了・数値適用は次フェーズ）
- **基準**: ConvoPeq.md **2026-08-25 20:06 再生成版**

---

## 1. M_scope の定義をソース上で再確認

### numerator が何を数えるか

```text
M_scope ≜ sup_t [B_true(t) − est(t)]
        ≜ sampler の観測から漏れる true outstanding の最大超過量

semantic unit: published World outstanding count（整数・無次元）
```

### O_denom と同一 semantic unit か

✅ **同一**。両者とも「acquire event 数 − release event 数」の差分であり、
単位は「published World outstanding count」。

### scope 境界

```text
scope ≜ bootstrap commit 後 〜 shutdown clear 前
      （定常運転区間に限定・lifecycle 遷移時の burst は別フェーズとして扱う）
```

---

## 2. `ceil(M_scope / O_denom)` 入力値の source-observable 測定

| 入力 | 値 | source | 状態 |
|---|---:|---|---|
| M_scope | 4120 | D101-35-C §2 証明（K + λ_prod_bound × G_bound + N_timer） | ✅ MEASURED |
| O_denom | ≥ 1 | bootstrap invariant（post-first-commit で est ≥ 1 構造保証） | ✅ ELIGIBLE |
| M_scope > 0 | ✅ 4120 > 0 | λ_prod_bound ≥ 1 + N_timer ≥ 2 より自明 | ✅ |
| O_denom > 0 | ✅ bootstrap invariant | 構造的に保証 | ✅ |

### K_min 計算

```text
K_min = ceil(M_scope / O_denom) = ceil(4120 / O_denom)

O_denom ≥ 1 より K_min ≤ 4120
O_denom = 1 の場合 K_min = 4120（最大）
```

---

## 3. 実装上の bounded storage と照合

### publication path（M_scope/O_denom の対象 domain）

| storage | capacity | 対象 |
|---|---:|---|
| `intentQueue_` (MPSC ring) | 4096 | Publish Intent 待ち行列 |
| `ownerChannel_` | transfer channel | 所有権移譲済み world |

### retirement path（World 破壊までの保持）

| storage | capacity | 対象 |
|---|---:|---|
| DeferredDeletionQueue (D) | 4096 | 全 type の deletion entry |
| RetireQuarantineStore (Q) | 512 | quarantine fallback entry |
| EmergencyQ (E) | 512（別 instance 同型） | D+Q full 時の第3退避 |
| TerminalReclaimAuthority (T) | growable vector | 全 bounded store 枯渇後の最終安全装置 |

### capacity 合計

```text
retirement path 固定 bounded subtotal = 4096(D) + 512(Q) + 512(E) = 5120
T = growable → 固定上限なし（構造的に obligation 消失不可能）

publication path 固定 bounded subtotal = 4096 (intentQueue_)
```

---

## 4. capacity の二重計上チェック

| 区分 | 意味 | M_scope/O_denom との関係 |
|---|---|---|
| physical storage capacity | 各 queue/store の slot 数 | M_scope/O_denom とは**別 concept**。物理的な保持上限 |
| logical obligation capacity | admitted logical obligation の同時存在数 | D14.2 reservation-first モデルの budget。INV-X1-5 |
| outstanding World count | committed − released の差分 | **M_scope/O_denom と同一 semantic unit** ✅ |

⚠️ physical storage capacity と logical obligation capacity は関連するが同一ではない。
D14.2/D15.2 の契約では「1 logical obligation = exactly 1 reservation」のため、
reservation 会計が正確なら physical capacity ≥ logical obligation count が従う。

本監査では logical obligation capacity（= outstanding World count）を対象とし、
physical storage capacity は implementation detail として区別。

---

## 5. I4 D14 / D15 契約整合監査

### D14.2 reservation-first model

```text
Transport admission
    ↓
Logical obligation reservation（最初に取得・単一 budget から）
    ↓
placement: Transport / DurablePending / Building / Stalled

不変式: transportCount + durableCount + buildingCount + stalledCount ≤ kMaxLogicalRecoveryObligations
```

✅ **placement 間の二重計上なし**: reservation-first により各 obligation は正確に 1 回のみカウント。

### D15.2 ownership conservation

```text
admittedLogicalObligationCount（coalesce/supersede で増えない）
```

✅ ownership conservation 成立。消失理由は Success / Superseded / ShutdownDiscard のみ。
terminal-failure を disappearance reason として使用していないことを確認。

### M_scope/O_denom への影響

D14/D15 の invariant は Recovery domain の契約であるため、
Publication domain の M_scope/O_denom には直接適用されない。
ただし、両 domain とも「obligation 消失理由の制限」という設計思想を共有しており、
M-bound 証明の信頼性を支える基盤として整合している。

---

## 6. 判定表

| 項目 | 判定 |
|---|---|
| M_scope semantic unit = O_denom semantic unit | ✅ 同一（published World outstanding count） |
| M_scope source-observable | ✅ 4120（D101-35-C §2 証明値） |
| O_denom source-observable | ✅ ≥ 1（bootstrap invariant） |
| K_min = ceil(M_scope/O_denom) 導出可能 | ✅ symbolic 確定 |
| bounded storage mapping | ✅ 完了 |
| 二重計上 | ✅ なし（排所有移動チェーン） |
| D14/D15 契約整合 | ✅ reservation-first + ownership conservation 整合 |
| terminal-failure を disappearance reason に使用していない | ✅ 確認済み |
| Numerical compatibility | **PENDING**（λ/G/O_denom 値決定後） |

# VERDICT: D102-C3 = **PASS**

---

## 7. 次ステップ

```text
D102-C3 PASS（構造的 compatibility 確認完了）
      ↓
ユーザー意思決定: λ/G/O_denom 具体値
      ↓
Numerical compatibility 判定（R_required vs R_cap 比較）
      ↓
Phase I NO-GO 解除判断
```
