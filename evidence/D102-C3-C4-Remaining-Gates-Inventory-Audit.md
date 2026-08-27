# D102-C3/C4 — Remaining Gates Inventory & Implementation-Prerequisite Audit

- **実施日**: 2026-08-26
- **基準版**: `ConvoPeq.md Generated: 2026-08-26 20:06:47` ＋ `doc/work88/I4_DESIGN_CONTRACT.md` 2026-08-15（Phase I-Design-5）
- **作業種別**: **read-only / audit-only**（production 変更 **0** / test 変更 **0** / CMake 変更 **0** / contract 変更 **0**）
- **固定前提**（C2-7 decision values、CLOSED として扱う）:
  `M_scope=4120` / `O_denom=1` / `K_min=4120` / `R_required=4121` / `R_cap=5120` / `TerminalDep=0` / `Headroom=999` / D102 numeric **GO**

---

## 1. 最新 ConvoPeq.md の D102 C3/C4 全文検索結果

検索クエリ（rg -n, WSL）:

```
D102-C3 | D102-C4 | D102      →  5 ヒット: OdenomCampaign コメントのみ（C2-3 harness）
D102-C2                       →  同上（C2-3 のみ）
R_required | R_cap | M_scope | O_denom →  C2-3 数値適用コードのみ
C3 / C4（単独）              →  該当なし（D102 文脈の C3/C4 ラベルはソースに存在しない）
INV-PUB-[1-4] / INV-PUB      →  Phase I-T1/T2 コメントのみ（D101-35 系）
Phase I                      →  T1 型準備 / I-T2/R 監査コメントのみ
implementation gate          →  該当なし
```

**結論**: ConvoPeq.md に **D102-C3 / C4 というラベルのソース記述は存在しない**。
C3/C4 は「監査定義上の gate 名称」であり、I4 契約および evidence 上の数値・実装前提条件に対応する。
最新の数値チェーン（C2-6/7 で GO 確定）はソースに反映済みであり、ConvoPeq.md の該当コード（OdenomCampaignTests.cpp:1-420）は C2-3 までで閉じている。

---

## 2. C3/C4 棚卸し — 「未完了 / 既完了 / moot」分類

本表では「C3」を *数値の実装適用*、「C4」を *bounded-retirement の production 実装前提* と解釈して棚卸しする（I4 契約の D11/D14/D15 との対応で命名が揺れるため、機能別に再定義）:

| Gate | 想定内容 | 現在の状態 | 根拠 | C2-7 の影響 | 次アクション |
|---|---|---|---|---|---|
| **C3-a** `M ≤ M_scope` の構造証明 | **CLOSED** | `D101-35-C ProofClosureAudit` で `M ≤ 4120 < ∞` を証明済み。`M_scope=4120` は C2-6/7 で GO 確定 | なし（前提） | 不要 |
| **C3-b** `O_denom` 実測確定 | **CLOSED** | `D102-C2-3` warmup1+10 窓で `O_denom=1`、全 G1-G15 PASS（`evidence/D102-C2-3-*`） | C2-6/7 で GO | 不要 |
| **C3-c** `R_required` 算出 | **CLOSED** | `R_required=4121`（C2-6/7、独立再計算・parametric table 全 PASS） | C2-7 で GO | 不要 |
| **C4-a** `R_required ≤ R_cap(5120)` の bounded 充足 | **CLOSED** | `4121 ≤ 5120`（headroom 999、TerminalDep=0）— C2-6/7 GO | C2-7 で GO | 不要 |
| **C4-b** bounded-retirement 実装（admission 制御等） | **OPEN ではない — 該当なし** | R_cap 5120 は現行コードの D/Q/E 合計容量そのものであり、新規実装を要しない。D8-2-C/D で現行 retire path が全 disposition で conserved であることは PASS 済み | D2-4 で Terminal growable 維持確定のため対象外 | 不要 |
| **C4-c** `F / G-B2`（heap/allocator bound） | **OPEN** だが **moot** | D38/D39: heap container は直接 bound で CLOSED、F/G-B2 は `B_existing_measured` 実測 gate として OPEN のまま | C2-7 と無関係（別 budget 項） | `B_existing_measured` 実測は可能だが D102 bounded 判定を block しない |
| **C4-d** `P2 / G2 / W1`（T_min / G_max / H_max 静的 bound） | **OPEN** | D40: コード静的保証なし。P2/G2/W1 は探索で未導出（`T_min>0` / `G_max<∞` を build duration / callback scope から静的証明できず） | C2-7 と無関係（代替 N_retired 静的 bound 経路） | 「measure or constrain」決定待ち（D40.5）— D102 bounded には不要 |

**重要**: `I4_DESIGN_CONTRACT.md`（2026-08-15）に残る古い記述
「`Phase I implementation NO-GO`」「`D102 numeric NO-GO`（当時）」は
D101-35-C（M 有限証明）〜 D102-C2-6/7（数値 GO 確定）により **後続 evidence で解消済み**であり、
現在状態として機械的に再オープンしない（下記 §4 参照）。

---

## 3. 「数値 gate と実装 gate」の分離 — D102 numeric GO ≠ implementation GO

```
数値 gate（D102-C2）:  R_required=4121 ≤ R_cap=5120  →  GO（C2-6/7 確定）
     │
     │  I4 でも「M-bound は構造上証明済みで λ/G の契約値決定が残る」と明記
     │  → C2-6/7 で契約値決定が完了し、数値的前提は閉じた
     ▼
実装 gate（Layer D）:  bounded-retirement を runtime で強制する仕組みが必要か？
     │
     ├─ 現行 D/Q/E（4096/512/512）は既に合計 5120 で bounded
     ├─ Terminal は growable だが bounded 判定は Terminal 依存 0 で PASS（§1 D2-4）
     ├─ retire path の ownership conservation は D8-2-C/D で PASS
     └─ → **実装 gate は「新規コードを書く」対象が存在しない**
```

したがって:

> **D102-C2 の GO は「bounded 5120 で数値的に feasible」であることの証明**であり、
> **production 実装を開始すべきという合図ではない**。実装は既に bounded 容量で存在し、
> 数値 GO は「現行容量で足りる」ことの事後検証として機能する。

I4 契約に残る OPEN invariant（Phase I NO-GO）との関係は §5 で切り分ける。

---

## 4. 古い OPEN の機械的再オープン禁止 — 検証結果

| 旧契約 OPEN | 最新検証 | 混同回避 |
|---|---|---|
| `N_retired_max` | D39: 構造 CLOSED（`ceil(grace/T_build)` 導出構造は確立）、数値は実測 gate。D40: `N_retired_world` 静的 bound は OPEN（P2/G2 未導出）。矛盾ではなく**構造/数値の分離** | 機械的に「OPEN のまま」と扱わない。数値は C2-6/7 の `R_required` 静的 bound（O_denom 実測）で代替済み |
| `T_build` / `grace_lifetime_max` | 実測 gate（D38/D39/D40 共通）。`R_required` 導出に `T_build` は使わず `O_denom` 実測を使用したため、現在 D102 の blocking gate ではない | 再オープンしない |
| `F / G-B2` | D38: B_existing_measured 実測 gate として OPEN のまま。だが heap container 直接 bound で CLOSED。D102 bounded 判定とは別 budget 項 | D102 実装の prerequisite ではない（§5 Layer C） |
| `Phase I implementation NO-GO` | I4 2026-08-15 時点で NO-GO 継続。**原因は D9 domain-based supersession**（canSupersede の十分条件未固定）— D102 数値系とは無関係 | D102 GO をもって Phase I NO-GO を GO と誤読しない |

---

## 5. Implementation Prerequisite Audit — 4 Layer 分類

### Layer A: 数学 / 数値

| 項目 | 状態 |
|---|---|
| `M ≤ M_scope < ∞`（D101-35-C） | ✅ CLOSED |
| `O_denom = 1`（D102-C2-3, 10 窓 campaign-wide max） | ✅ CLOSED |
| `R_required = 4121`（C2-6/7） | ✅ CLOSED — worst-case O=1 で bounded PASS |
| `R_required ≤ R_cap` | ✅ GO（headroom 999） |

→ **Layer A: READY**（全 blocking gate CLOSED/GO）

### Layer B: ownership / lifetime / authority invariant

| 項目 | 状態 |
|---|---|
| Q/E cap 512 / overflowCount / health 監視 | ✅ 現行で bounded・監視あり |
| D→Q→E→T ownership conservation（D8-2-C 9/9 PASS） | ✅ CLOSED |
| DSPLifetimeManager caller 契約（D8-2-D PASS, T9 CLOSED） | ✅ CLOSED |
| INV-PUB-1（recovery publish 直列化・D39） | ✅ CLOSED |
| pendingReclaim overflow（512 bounded・INV-QOWN） | ✅ CLOSED |
| **D9: `canSupersede()` semantic target containment** | ❌ **OPEN** — I4 2026-08-15 の最重要 NO-GO。domain containment ≠ semantic containment の分離、「equality 保守的」方針の実装前固定が必要。**D102 とは別系列** |

→ **Layer B: D102 bounded-retirement に限れば READY。Phase I 全体としては D9 により CONDITIONAL。**

### Layer C: runtime / realtime safety

| 項目 | 状態 |
|---|---|
| RT=D only / NonRT=D→Q→E→T boundary（D8-2-C C5） | ✅ CLOSED |
| grace 有限性（INV-GRACE-1 / D39） | ✅ CLOSED（新 reader が旧 generation 再取得不能） |
| shutdown closure（drainAll 強制解放） | ✅ CLOSED（C6, D8-2-C C6） |
| **P2/G2/W1 静的 bound**（T_min/G_max/H_max） | ⚠️ OPEN だが **D102 の N_retired 静的 bound 代替経路**。D40 で「コード静的保証なし」を確定し、「measure or constrain」決定待ち。D102 数値 GO には不要（O_denom 実測で代替済み） |

→ **Layer C: D102 前提では READY。静的 N_retired bound を別途追求する場合のみ追加 gate。**

### Layer D: implementation readiness

「source 変更を開始してよいか」— 上記 A〜C を総合:

- D102-C2 が意図する実装（もしあるとすれば、admission 制御の bounded 保証）は
  **既に現行コードの D/Q/E 容量で充足**しており、新規 production 変更の対象が存在しない。
- 残る implementation NO-GO は **Phase I D9（canSupersede）** に起因し、
  D102 数値系の外側にある。
- D40 の P2/G2/W1 OPEN は「N_retired を静的に bound する」代替経路の未完であり、
  D102 の O_denom 実測経路とは独立。

---

## 6. 最終判定

```
IMPLEMENTATION PREREQUISITE = CONDITIONAL
```

### 理由

- **D102 bounded-retirement に限れば READY**:
  `M_scope=4120 / O_denom=1 / R_required=4121 ≤ R_cap=5120 / TerminalDep=0`
  は全て CLOSED/GO。bounded 5120 で数値的 feasible であり、retire path の
  ownership も D8-2-C/D で PASS 済み。実装を開始すべき新規コードは存在しない
  （現行容量が既に十分）。

- **Phase I 全体としては NOT READY ではないが CONDITIONAL**:
  blocking gate は **D9 domain-based supersession**（`I4_DESIGN_CONTRACT.md` 2026-08-15、
  `canSupersede()` の十分条件固定）のみ。D102-C2 の GO をもって Phase I 実装を
  GO とすることはできない。逆に D9 をもって D102 の GO を否定することも誤り
  （別系列の形式閉包課題）。

### blocking gate 明示（CONDITIONAL の内訳）

| blocking gate | why blocking | required evidence | next audit/measurement |
|---|---|---|---|
| **D9: `canSupersede()` semantic target containment** | I4 最重要 NO-GO。domain containment（必要条件）と semantic target containment（十分条件）の分離が未固定のまま実装すると誤った supersession で obligation を消失させる恐れ | D9 十分条件（equality 保守的）の実装前固定とコード対応付け | `I4_DESIGN_CONTRACT.md` D9 節の設計レビュー（ユーザー Design-5 指示の範囲） |
| P2/G2/W1 静的 bound（参考・D102 非 blocking） | `N_retired_world ≤ floor(G_max/T_min)+1` の静的 bound は P2/G2 未導出のため OPEN。ただし D102 は O_denom 実測で代替済みのため blocking ではない | 「measure or constrain」決定（D40.5）。静的 bound を追求するなら T_min/G_max のコード保証または制約追加 | D40 追補（measure vs constrain 選択）— D102 とは独立 |

**D102-C2 のみをスコープとする実装判断は READY**。
**Phase I 全体の実装判断は D9 により CONDITIONAL（NOT READY ではない — D9 以外は READY）**。

---

## 変更有無

```
production source : 0
test source       : 0
CMake             : 0
contract          : 0
telemetry         : 0
Terminal/K_terminal/QueueFull : 再検討なし（D2-4 継承）
```

## 参照

- `evidence/D102-C2-6-*` / `evidence/D102-C2-7-*`（R_required 4121 / headroom 999 / TerminalDep 0）
- `evidence/D102-C2-5-D8-2-C-*`（disposition 9/9 PASS）
- `evidence/D102-C2-5-D8-2-D-*`（DSPLifetimeManager PASS / T9 CLOSED）
- `doc/work88/I4_DESIGN_CONTRACT.md` 2026-08-15（Phase I NO-GO / D9 / D40 P2/G2/W1 OPEN）
- `ConvoPeq.md` 2026-08-26 20:06:47（最新ソース反映済み）
