# D102-C2-5-D1 — QueueFull Semantics + Caller-Retain Closure Audit（read-only, production変更0）

- **実施日**: 2026-08-26 01:00 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines）
- **目的**: D0 で仮定した `QueueFull == caller retains` が 14 caller 全体で ownership conservation を成立させるかを、**最終所有者で A/B/C/D 再分類**し、**enqueueWithRetry の閉ループが retry 責務を caller へ移譲する状態**であることを証明する。`QueueFull` と `TerminalPressure` の比較、D full と Terminal full の ambiguity 有無も確定する
- **判定**: **CONDITIONAL PASS** — `QueueFull` は generic bounded-retire storage failure として caller-retain に再定義可能。ただし 6重点 caller のうち **5 caller で戻り値無視（D）** が残り、単純な `return QueueFull` だけでは **ownership closure しない**。D2-D4 で caller 側の `quarantine/retry/backpressure` 閉ループを併せて修正する必要がある

---

## D1-1. `QueueFull` vs `TerminalPressure` 比較

### 現行 enum（`src/audioengine/ISRAuthorityClass.h:28`）

```text
Success, QueuePressure, QueueFull, Shutdown, TerminalReclaim
```

- `QueuePressure` — Q/E が所有（`ISRRetireRouter.cpp:346,356`）
- `TerminalReclaim` — T が所有
- `QueueFull` — 現行 `ISRRetireRouter::enqueueRetire` では **未使用**（`QueuePressure` に統合、`ISRRetireRouter.cpp:257` の retry 後に `QueuePressure` のみ返す）
- `Shutdown` — shutdown reclaim 失敗時のみ caller-retain

### 過去 Candidate A 検討（`evidence/phase-d101-9-step2-terminal-candidate-d-safety-audit.md:223`）

- Candidate A: Bounded Terminal + Caller Retains Ownership — `store() == false → QueueFull`
- 過去監査で `TerminalPressure` として分離する案も検討されたが、**新規 enum 値は D14/D15 contract の改訂を要する**ため本 task では禁止
- 本 D1 推奨: **`QueueFull = generic bounded-retire storage failure = caller retains`** として再定義する

**理由:**
- `QueueFull` は現行 production `enqueueRetire()` で使用されておらず、意味を caller-retain に再定義できる
- `QueueFull` は `QueuePressure` と明確に対照（`QueuePressure==transferred / QueueFull==caller retains`）
- `TerminalPressure` を新設すると 14 caller の分岐を新規値対応に全改修する必要があり、**既存 `QueueFull` を流用する方が最小変更**

### D full vs Terminal full の ambiguity

```text
現行: D full → Q → E → Terminal（全て所有移譲、区別は RetireEnqueueResult の値で表現）
Candidate A: D full → Q/E で QueuePressure（所有移譲）、Terminal full → QueueFull（caller retains）
```

- `QueueFull` を **generic bounded failure** と定義すると、**D full と Terminal full を同じ値で表現することになる**が、**呼び出し元にとってはどちらも `caller retains`（backpressure）として扱えばよい**ため ambiguity は生じない
- caller が `D full` と `Terminal full` を区別して別処理をする必要はない（どちらも `Q/E/T` への移送失敗を意味し、所有は caller に残る）
- 区別が必要な場合でも、ログの `reason` 文字列（`"QueuePressure"` vs `"TerminalReclaim"`）で診断可能

**D1-1 判定: `QueueFull` を generic bounded failure として採用してよい — TerminalPressure 分離は不要**

---

## D1-2. 14 Caller を「最終所有者」で A/B/C/D 再分類（6重点 caller 含む）

### 分類定義（D1 指示）

| 分類 | 意味 | GO条件 |
|---|---|---|
| A | callerが保持し、その場でretry | retry先が明確（`tryReclaim` 後の再 enqueue 等） |
| B | callerが保持し、quarantine等へ移送 | 移送成功を確認（`quarantineRetireSink` 等） |
| C | callerが保持し、上位へbackpressureを返す | ownership を保持したまま上位へ伝播（`return QueueFull`） |
| D | 戻り値を無視 | **原則 NO-GO** |

### 14 Caller 現行 Control-Flow（`QueueFull` 到達時を想定）

| # | Caller | 現行 `QueueFull` 時の最終所有者 | 分類 | ptr が手元に残るか | GO |
|---|---|---|---|---|---|
| 1 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult` (`h:4205`) | `result == QueueFull` を return → 上位 caller へ伝播 | **C** | Yes (`ptr` は引数、return 後に caller が保持) | **GO**（上位が backpressure 処理すれば） |
| 2 | `AudioEngine::enqueueDeferredDeleteNonRt` (`h:4198` `bool`) | `result != Shutdown` → `true`（現行誤認）→ **D** | **D** | Yes だが `true` を返すため caller は手放したと誤認 | **NO-GO**（wrapper 要修正） |
| 3 | `SnapshotCoordinator::switchImmediate` (`h:94` `result == Shutdown` 分岐) | `result == QueueFull` → 現行分岐では `Shutdown` 以外は成功扱い → **D** | **D** | Yes だが分岐で区別されない | **NO-GO** |
| 4 | `SnapshotCoordinator::discardSnapshot` (`h:107` `if (!enqueueWithRetry)`) | `QueueFull → false` → `quarantineRetireSink` | **B** | Yes → B へ移送 | **GO** |
| 5 | `SnapshotCoordinator::retireSnapshot` (`h:176` `if (!enqueueWithRetry)`) | 同上 | **B** | Yes | **GO** |
| 6 | `SnapshotCoordinator::retireSnapshot` (`h:181` 2回目) | 同上 | **B** | Yes | **GO** |
| 7 | `SnapshotCoordinator.cpp:57` `startFade` (`oldTarget`) | `const auto result = enqueueWithRetry` → **未使用** | **D** | Yes だが `result` を捨てる → 保持するが処理されない | **NO-GO** |
| 8 | `SnapshotCoordinator.cpp:114` `completeFade` (`old`) | 同上 未使用 | **D** | 同上 | **NO-GO** |
| 9 | `DSPLifetimeManager.cpp:49` `retireDSPCore` (1) | `ignoreUnused(result)` | **D** | Yes だが `result` を捨てる | **NO-GO** |
| 10 | `DSPLifetimeManager.cpp:96` `retireDSPCore` (2) | 同上 | **D** | 同上 | **NO-GO** |
| 11 | `EQProcessor.Core.cpp:61` (`stackRouter.enqueueWithRetry`) | `return Success||QueuePressure||TerminalReclaim` → `QueueFull` は `false` | **C** | Yes → `false` を上位へ伝播 | **GO** |
| 12 | `ISRRuntimePublicationCoordinator.cpp:162` | `if (result != Success) return result` → `QueueFull` を上位へ伝播 | **C** | Yes | **GO** |
| 13 | `AudioEngine.Cache.cpp:16,41` (`enqueueDeferredDeleteNonRt` via `bool`) | `bool==true` → **D** | **D** | 同 #2 | **NO-GO** |
| 14 | `ConvolverProcessor.Lifecycle.cpp:57,70` (`enqueueDeferredDeleteNonRt`) | 同 #2 | **D** | 同 #2 | **NO-GO** |

**集計:**

| 分類 | 件数 | 具体例 | GO |
|---|---|---|---|
| A (retry) | 0 | 現行 retry は `enqueueRetire` 内の `kMaxRetry 2` のみ、Terminal 到達後は retry しない | — |
| B (quarantine) | 3 | #4,5,6 | **GO** |
| C (backpressure 伝播) | 4 | #1,11,12,13? (EQ) | **GO**（#1 は上位依存） |
| D (無視) | 7 | #2,3,7,8,9,10,13,14 | **NO-GO** |

### 6重点 caller の詳細（D1-2 指示）

| Caller | `QueueFull` なら ptr が手元に残るか | control-flow が最後まで保持できるか |
|---|---|---|
| `SnapshotCoordinator::startFade` (`cpp:57`) | Yes (`oldTarget` は局所変数、関数終了まで生存) | **No** — `result` を捨てるため保持するが処理されない。`D` |
| `SnapshotCoordinator::completeFade` (`cpp:114`) | 同上 (`old`) | **No** — 同上 |
| `DSPLifetimeManager` (`cpp:49,96`) | Yes (`dsp` は引数) | **No** — `ignoreUnused(result)` |
| `EQProcessor` (`Core.cpp:61`) | Yes (`ptr` は引数) | **Yes** — `return false` で上位へ伝播 (`C`) |
| `AudioEngine.Cache` (`Cache.cpp:16`) | Yes (`map`/`old`) | **No** — `enqueueDeferredDeleteNonRt` の `bool` が `true` を返すため保持とみなされない |
| `ConvolverProcessor` (`Lifecycle.cpp:57`) | 同上 | **No** — 同上 |

**D1-2 判定: 14 caller のうち 7 が D（戻り値無視）で ownership closure しない — D2-D4 で caller 側の `quarantine/retry/backpressure` 閉ループを併せて修正する必要がある**

---

## D1-3. `enqueueWithRetry()` の閉ループ確認（最重要）

### 現行（`src/audioengine/ISRRetireRouter.cpp:303`）

```text
D (enqueueRetire)
 ↓ Success → return (D owns)
 ↓ QueuePressure
tryReclaim → re-enqueue (kMaxRetry 2)
 ↓ Success → return
 ↓ QueuePressure
Q (quarantine) → QueuePressure → return (Q owns)
 ↓ Q full
E (emergency quarantine) → QueuePressure → return (E owns)
 ↓ E full
Terminal (tstored → TerminalReclaim) → return (T owns)
```

- **Retry 責務は D のみ**（`kMaxRetry 2` の `tryReclaim` ループ）。Q/E/Terminal は **移送** であり retry ではない。
- `QueueFull` を返す候補は **Terminal full 時のみ**（将来 bounded 化後）。現行 growable では `tstored` は常に true のため `QueueFull` は返らない。

### Candidate A 変更後

```text
Terminal full → tstored==false → QueueFull → return QueueFull
```

**確認事項:** `QueueFull` は **retry 可能な状態**か **retry 責務を caller へ移譲した状態**か

**答え: 後者（移譲）**

- `enqueueWithRetry` 自身は `QueueFull` を返した時点で **処理終了**し、所有を caller に戻す。`enqueueWithRetry` 内部で `QueueFull` を再試行するループは存在しない。
- Caller は `QueueFull` を受け取った後、**`quarantineRetireSink` へ移送**するか、**上位へ `QueueFull` を伝播して backpressure** するか、**retry** するかを選択する責務を負う。
- したがって `QueueFull` は **「retry 責務を caller へ移譲した状態」** であり、`enqueueWithRetry` が自律的に retry する状態ではない。

**この二つの区別は全く違う — 前者なら `enqueueWithRetry` が所有を保持したまま retry するが、後者なら所有は caller に戻る。現行設計は後者であり、D14.3 の backpressure（budget 枯渇で admission を BLOCK）と整合する。**

---

## D1 総合判定（G1-G8）

| Gate | 条件 | 判定 |
|---|---|---|
| G1 | `QueueFull` semantics 明確（`generic bounded failure = caller retains`） | **PASS** |
| G2 | `tstored==false → QueueFull` の一意な伝播 | **PASS**（`ISRRetireRouter.cpp:365` で確定） |
| G3 | `bool` wrapper が caller-retain を表現 | **CONDITIONAL PASS**（`AudioEngine::enqueueDeferredDeleteNonRt` は `result != Shutdown && result != QueueFull` への修正が必須） |
| G4 | 14 caller 全て ownership closure | **NO-GO**（7 caller が D） |
| G5 | return-value-ignore caller = 0 | **NO-GO**（5 caller が無視） |
| G6 | retry/backpressure 責務が一意（移譲として明確） | **PASS**（`QueueFull` は移譲） |
| G7 | shutdown で最終的に閉じる（`ShutdownDiscard` 経由） | **PASS**（`ISRRuntimePublicationCoordinator.cpp:822` 等で `recoveryShutdownDiscardCount` として記録） |
| G8 | D15.2 ownership conservation と矛盾しない | **CONDITIONAL PASS**（`QueueFull` を disappearance としてカウントしなければ PASS。caller が `QueueFull` を正しく quarantine/backpressure すれば conservation 維持） |

**D1 全体: CONDITIONAL PASS**

- `QueueFull` 導入自体は `generic bounded failure` として **PASS**
- しかし **G4/G5 が NO-GO** のため、**bounded Terminal 実装（D5）へ進む前に D2-D4 の caller closure 修正が必須**
- 次工程は `D2 tstored propagation → D3 bool wrapper → D4 all caller closure` の順で、**D5 bounded Terminal は最後**に実施する

---

## 参照

- `src/audioengine/ISRAuthorityClass.h:28` / `src/audioengine/ISRRetireRouter.cpp:303,339,365` / `src/audioengine/AudioEngine.h:4198` / `src/core/SnapshotCoordinator.h:151` / `doc/work88/I4_DESIGN_CONTRACT.md:154,179` / `evidence/phase-d101-9-step2-terminal-candidate-d-safety-audit.md:223`

*本監査は read-only であり、production source 変更 0、D14/D15 改訂なし、Terminal bounded 化なしで実施された。*
