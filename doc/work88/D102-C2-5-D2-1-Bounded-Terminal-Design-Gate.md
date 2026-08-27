# D102-C2-5-D2-1 — Bounded Terminal Design Gate

- **実施日**: 2026-08-26
- **作業種別**: **design-only / read-only**（production 変更 0 / test 変更 0 / contract 変更 0 / 実装 0）
- **基準版**: ConvoPeq.md `Generated: 2026-08-26 20:06:47`
- **前提監査**: D8-2-B-2 (tests PASS) / D8-2-C (PASS 9/9) / D8-2-D (PASS)

---

## 1. Current Terminal Contract（事実固定）

`ISRRetireRouter.cpp:22-25`（TerminalReclaimAuthority 実装ヘッダコメント）:

> ★ P-4 (15-P-4): entries_ is GROWABLE (std::vector). store() ALWAYS succeeds,
> so there is NO "store full" failure path. This guarantees the ownership
> invariant: **enqueueWithRetry() never returns with ptr unowned.**

```text
D (DeferredDeletionQueue, lock-free MPMC)
  ↓ QueuePressure
retry ×2 (tryReclaim + drainEmergencyAndTerminal)
  ↓
Q (RetireQuarantineStore, cap 512, full時 deleter 実行せず false)
  ↓
E (EmergencyQuarantineStore, cap 512, 同上)
  ↓
T (TerminalReclaimAuthority, growable std::vector, store() 常に true)
  ├─ epoch-safe && NonRT → synchronous delete（即時破棄）
  └─ otherwise           → T 内保持（drain()/drainAll() で解放）
```

補助事実:
- **telemetry 既存**: `terminalPeakResident()` / `terminalStoreCount()` /
  `terminalDrainEntryCount()` / `terminalDrainAllCount()`（`ISRRetireRouter.h:104-119`）
- **backpressure は上流で完結設計**: `quarantineResidentCount()` は「Q + EmergencyQ のみで
  Terminal を含まない」（`AudioEngine.Threading.cpp:146` コメント明記）—
  Terminal を backpressure 監視対象から意図的に除外している
- **I4 契約**: `INV-X1-7`（I4 D15.2）— 「logical obligation は Success / Superseded /
  ShutdownDiscard のいずれかでのみ消失しうる」。**terminal-failure は消失理由として認めない**
- **D15.2 conservation 等式**: transport + durable + building + stalled + superseded +
  shutdownDiscard == admittedLogicalObligation
- **先行監査**: `phase-d101-9-step2-terminal-candidate-d-safety-audit.md` — epoch-unsafe ptr への
  synchronous destruction（Candidate D）を P-4/I4 違反として既に棄却済み
- **D2-0 Semantic Freeze**: `QueueFull` は既存 6 状態への再計上のみ許容。
  D15.2 右辺（disappearance states）への追加は禁止と確定済み

---

## 2. Problem Definition

`K_terminal` を導入し T full を発生させた場合、`store()` が拒否すると
ownership は Terminal に移転せず **caller に残存**する。しかし現行 `enqueueWithRetry()`
は D/Q/E/T のいずれかで ownership を完結させる単一契約であり、全 production caller
（AudioEngine wrapper / DSPLifetimeManager / EQProcessor / RuntimePublicationCoordinator /
SnapshotCoordinator）がこの契約を前提に実装済み（D8-2-C/D で検証・PASS 済み）。

**中心問題**: 「T bounded 化」と「P-4 ownership contract 維持」は両立するか。

---

## 3. Candidate A — Caller Retains Ownership + Backpressure

```text
D full → Q full → E full → T full → Terminal rejects → caller retains ptr → caller retry/escalation
```

### 3.1 Ownership

| 項目 | 評価 |
|---|---|
| P-4 | ❌ **破壊**。「never returns with ptr unowned」の不変条件が崩れる |
| D15.2 | ⚠️ caller-retained は実質第 7 の状態。D2-0 冻結方針（右辺追加禁止）と衝突 |
| INV-1 (exactly one owner) | ✅ 維持可能（Caller=1）だが新 state 導入が必要 |

### 3.2 Backpressure

- **caller ごとの再試行プロトコルが 5 系統すべてに必要**になる:
  - `DSPLifetimeManager::retire/retireByHandle`: QueueFull 後の ptr 保持 → 再試行キュー新設
  - `EQProcessor.Core.cpp:65-67`: `false` 返却先の上位 caller でも同様の処置が必要
  - `ISRRuntimePublicationCoordinator.cpp:163`: result 転送のみ → 上位で処理責務発生
  - `AudioEngine.h:4201-4204` bool wrapper: `QueueFull==false` の既存意味が
    「caller が保持」に変わるが、wrapper 自体は何も保持できない（ptr は引数で渡された非所有ポインタ）
- 各系統で再試行の順序保証・重複排除を個別実装する必要があり、統一性の担保が困難

### 3.3 RT Safety

- RT 経路（`retireRT` → `enqueueRetire`、D のみ）は構造的に T に到達しないため直接的影響なし（INV-4 ✅）
- ただし D full 時の RT 挙動は現行と同一であり、Candidate A によって改善しない

### 3.4 Caller Compatibility — **最重要**

> **`DSPLifetimeManager::ignoreUnused(result)` が成立しなくなる。**

```text
DSPLifetimeManager::retire()
    → router_->enqueueWithRetry(dsp, &destroyDSPCoreNode, ...)
    → QueueFull（Candidate A 下で到達可能になる）
    → ignoreUnused(result)          ← dsp への参照を破棄
    → DSPCore* の ownership が caller（retire 関数ローカル）に残ったまま関数終了
    → **orphan（leak）確定**
```

これは D8-2-D で PASS 判定した契約（「result 返却時点で必ず Router 内 authority が所有」）の
**前提そのものを無効化する**。対処には `ignoreUnused` の削除 + caller 側 fallback 実装が必要であり、
ユーザー指示の禁止事項（ignoreUnused 修正・caller fallback 実装）に直接抵触する。

さらに `SnapshotCoordinator::quarantineRetireSink`（cpp:28-29）は Q full 時 `assert(false)` —
Q/E/T 全部が full になりうる世界では assert ではなく回復手順が要求されるが、これも現行契約外の新規実装。

### 3.5 I4 Compatibility

- `INV-X1-7`: T full による caller retention は消失ではないため直接違反ではないが、
  D15.2 等式に第 7 項（callerRetainedCount）を追加する**契約改定が必要**
- `D14.3`（budget 枯渇 = backpressure、terminal-failure 撤廃の方針）と逆行

### Candidate A 小結: **NO-GO**
P-4 不変条件の破壊 + D8-2-D 検証済み契約の無効化 + 5 caller 系統の個別 backpressure 実装 +
D15.2 契約改定。コストが大きく、便益が未証明。

---

## 4. Candidate B — Shutdown-only Bounded

```text
通常 runtime path: D → Q → E → T（growable のまま維持）
shutdown 時のみ Terminal を bounded として扱う
```

### 4.1 Ownership

- 通常経路は完全に現行契約 → B-2/C/D の全 PASS 結果を温存 ✅
- ただし shutdown 中の `terminalReclaim` は Audio Thread 停止済みのため
  `epochSafe && !isRt` → **即時削除分岐が通常成立**し、T store に入ること自体が稀
  （`ISRRetireRouter.cpp:517-527`、`AudioEngine.h:4216` コメント「shutdown 中は … 即時破棄される」）

### 4.2 Shutdown Closure

- shutdown 時の最終回収は `drainAll()` → `drainAllQuarantineStore()` が
  Q+E+T を **容量無関係に強制解放**（`ISRRetireRouter.cpp:593-601, 444-451`）
- shutdown 中に T full 拒否が発生した場合、その ptr の受け皿は存在しない
  （以後何も動かない = INV-5 違反の leak 直行）
- → **bounded 化が最も closure が要求される局面で拒否経路を作るという自己矛盾**

### 4.3 RT Safety

- 影響なし（shutdown は NonRT のみ）

### 4.4 I4 Compatibility

- `ShutdownDiscard` は既に消失理由として認可済みのため等式整合は可能
- ただし「bound を切るために discard を増やす」方向性は conservation を悪化させる

### Candidate B 小結: **採用価値なし（NO-GO）**
技術的には実装可能だが、(a) runtime ownership protocol を一切改善せず、
(b) shutdown closure（INV-5）をむしろ悪化させ、(c) 便益ゼロで複雑度のみ追加。
「bounded Terminal の問題を runtime protocol に持ち込まない案」としては、
そもそも持ち込むべき問題（必要性）が示されていない。

---

## 5. Ownership Matrix

| 状態 | D | Q | E | T | Caller | ownership 消失 |
|---|--:|--:|--:|--:|--:|---|
| D accepted | 1 | 0 | 0 | 0 | 0 | No |
| Q accepted | 0 | 1 | 0 | 0 | 0 | No |
| E accepted | 0 | 0 | 1 | 0 | 0 | No |
| T accepted | 0 | 0 | 0 | 1 | 0 | No |
| T immediate delete | 0 | 0 | 0 | 0 | 0 | Destroyed（正当: deleter 実行） |
| **T full — Candidate A** | 0 | 0 | 0 | 0 | **1** | No — ただし **新 state 導入 + D15.2 改定 + ignoreUnused 契約崩壊** |
| **T full — Candidate B** | 0 | 0 | 0 | 1 | 0 | No（runtime は現行維持／shutdown 拒否分岐は INV-5 と衝突） |
| **T full — 現行 (growable)** | 0 | 0 | 0 | **1** | 0 | No — **常に成立、変更不要** |

核心行の判定: **現行契約（最終行）が唯一、全制約を追加 state なしで満たす。**

---

## 6. Invariant Analysis

| Invariant | 現行 growable | Candidate A | Candidate B |
|---|---|---|---|
| **INV-1** exactly one owner ∈ {0,1} | ✅ | ⚠️ Caller=1 で成立するも新 state 必要 | ✅（ただし shutdown 拒否時に不成立リスク） |
| **INV-2** no silent disappearance | ✅（T が最終受領） | ⚠️ caller 保持漏れ = leak リスク（ignoreUnused 問題） | ✅ runtime / ❌ shutdown 拒否時 |
| **INV-3** retry safety（multiple enqueue 禁止） | ✅（Router 内で完結、caller 再試行不要） | ❌ caller 側再試行で same ptr multiple-enqueue の危険を各系統で防御する必要 | ✅ |
| **INV-4** RT exclusion | ✅（RT=D only、jassert ガード） | ✅（変化なし） | ✅ |
| **INV-5** shutdown closure | ✅（drainAll 強制解放） | ⚠️ caller 保持 ptr は drain 対象外 → 最終 sweep API が新規必要 | ❌ 拒否分岐が closure と正面衝突 |

---

## 7. K_terminal Sizing — methodology only（Phase B 保留）

Phase A が NO-GO のため sizing は実施しない。将来再評価時の手順のみ確定:

1. `terminalPeakResident()` を実運用/soak test で継続観測（既存 telemetry で取得可能）
2. 持続的増加（drain を上回る store レート）が確認された場合のみ bounded 化を再審
3. K 候補 = 観測 peak の p99.9 × 安全係数、かつ Q/E 合計 (1024) に対して有意に小さい値
4. その際は本 gate の Candidate A/B 分析を出発点として再実施

---

## 8. QueueFull Semantics

- 現行: dead enum value（enqueueWithRetry が返さない — D8-2-C C1/C2 確定）
- D2-0 Semantic Freeze: QueueFull を disappearance として扱わない方針は**維持**
- 本 gate の結論により QueueFull は **dead のまま維持**が正解。
  削除も実装も本次期は行わない（enum 変更は contract 変更に該当）
- Candidate A 採用時のみ QueueFull が live になるが、それは同時に P-4/D15.2 改定を意味する

---

## 9. noexcept / allocation observations（観察事項のみ・変更提案なし）

- `TerminalReclaimAuthority::store()` は `noexcept` だが内部で `entries_.push_back()`（allocation）を実行
  （`ISRRetireRouter.cpp:33-34`）。allocation failure 時は `std::terminate` 経由で異常終了する
- **区別の確定**:
  - *logical capacity exhaustion*（K_terminal 設問）= 本 gate の対象 → **NO-GO 判定**
  - *allocator failure*（OOM）= 別問題。process 全体が OOM 状態での terminate は
    音声アプリの fail-fast として現行設計の範囲内と判断し、本次期は変更しない
- 将来 fixed-capacity 化する場合（Phase C、実現すれば allocation を排除できる）は
  この観察が再評価の出発点になりうる

---

## 10. Candidate Comparison（総括）

| 評価軸 | Candidate A | Candidate B | 現行 growable |
|---|---|---|---|
| P-4 (never unowned return) | ❌ 破壊 | ✅ | ✅ |
| D15.2 / INV-X1-7 | ⚠️ 契約改定必要 | ✅ | ✅ |
| D8-2-D ignoreUnused 契約 | ❌ 無効化 | ✅ | ✅ |
| INV-5 shutdown closure | ⚠️ 新 sweep API 必要 | ❌ 正面衝突 | ✅ |
| caller 影響 | 5 系統全部 | shutdown 経路のみ（負の便益） | なし |
| 必要性の証明 | なし | なし | — |
| 実装複雑度 | 高 | 中 | 現状維持（ゼロ） |

---

## 11. Final Verdict

# **NO-GO**

理由（確定根拠）:

1. **必要性が未証明** — bounded 化の動機となる Terminal 滞留の実測が存在しない。
   既存 telemetry (`terminalPeakResident` 等) によるデータ収集が先行措置であり、
   現時点で増加傾向の証拠はない
2. **Candidate A は現行 authority architecture を破壊する** —
   P-4「never returns with ptr unowned」の不変条件、および D8-2-B/C/D で
   実証・監査済みの `ignoreUnused(result)` caller 契約を無効化する
3. **Candidate B は便益ゼロ** — shutdown は epoch 安全即時削除 + 容量無関係強制 drain で
   既に閉じており、bounded 化は INV-5 を悪化させるだけ
4. **backpressure は既に上流で完結** — Q/E cap 512 + overflowCount +
   `quarantineResidentCount` ベースの health 監視が存在し、Terminal は
   「決して失敗しない最後の authority」として意図的に設計されている
   （Threading.cpp:146 — Terminal を backpressure 監視から除外するコメント）

D8-2 シリーズで確認された最大の資産——
**「Terminal が必ず ownership を受け取り、caller に ownership を残さない」という
単純で検証可能な契約** ——を、未証明の便益と交換しない。

---

## 12. Required Next Gate

- **Phase B/C（K_terminal sizing / implementation）: 起動条件付き保留**
  - 起動条件: soak/実運用で `terminalPeakResident()` の持続的増加が観測され、
    かつメモリ圧迫が health telemetry で裏付けられた場合
  - 再開時は本報告書 §10 の比較表を出発点とし、Candidate A 前提での
    D15.2 改定 + caller backpressure API 設計を別 gate として実施
- **直近の推奨アクション（design-only の範囲内）**:
  - soak test（ISRSoakTests 系）に `terminalPeakResident` 観測を含めることを
    次回テスト設計時に検討（production 変更なしで取得可能な既存 counter）

```text
D2-1 判定サマリ:
  Phase A (protocol feasibility) : NO-GO — bounded Terminal 不採用
  Phase B (K_terminal sizing)    : 保留（telemetry 起動条件付き）
  Phase C (implementation)       : 保留（Phase B 未着手のため）
  QueueFull enum                 : dead のまま維持（変更なし）
  production / test / contract   : 変更 0（本 gate は design-only のため最初から 0）
```
