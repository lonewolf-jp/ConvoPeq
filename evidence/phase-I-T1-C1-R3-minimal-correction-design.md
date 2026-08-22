# Phase I-T1-C1-R3 — Minimal Correction Design

> **目的:** `releaseObserved_` の authoritative source を一本化する最小修正の設計をコード変更前に固定する。
> **コード変更: 本R3では行わない。** R2 `OPTION-A READY` を前提に設計のみを確定。
> **測定: 禁止。** `T1-C1` 再検証は Gate R3-A 通過後に再開。`K_terminal / R gate` は引き続き停止。
> **Recovery supersession: 本R3では触らない**（I4 / D12-17 / REPAIR_PLAN2-dash2 の別タスク）。

## 1. R3-1 — `releaseObserved_` の唯一入口を再確認

### 1.1 `releaseObserved_` の mutation 箇所（最新 `ConvoPeq.md` 96816行 / src 同一）

| 箇所 | File | Line | 式 | 種別 |
|---|---|---|---|---|
| 1 | `ISRWorldRetirementTelemetry.h` | 90 | `fetchAddAtomic(releaseObserved_, 1, acq_rel)` in `onReleaseObserved()` | 直接 mutation |
| 2 | `ISRWorldRetirementTelemetry.h` | 98 | `fetchAddAtomic(releaseObserved_, count, acq_rel)` in `addReleaseObserved(count)` | 直接 mutation |
| 3 | `ISRWorldRetirementTelemetry.h` | 306 | `atomic<uint64_t> releaseObserved_{0}` | 定義 |
| 4 | `ISRWorldRetirementTelemetry.h` | 108 | `consumeAtomic(releaseObserved_, acquire)` in `releaseObserved()` getter | read-only |
| — | — | — | その他の直接 mutation は存在しない | — |

`releaseObserved_` の直接書き込みは上記2メソッドのみ。`rg -n "releaseObserved_" src/` は上記4 hits のみを返し、他ファイルからの直接操作はない。

### 1.2 `onReleaseObserved()` の production caller

| caller | File | Line | 呼び出し |
|---|---|---|---|
| `WorldRetirementReferenceObserver::onRelease()` | `ISRWorldRetirementReference.h` | 42-43 | `if (telemetry_ != nullptr) telemetry_->onReleaseObserved();` |
| — | — | — | 他の production caller は存在しない（`rg -n "onReleaseObserved" src/ ConvoPeq.md` は上記1件のみ） |

### 1.3 `addReleaseObserved(delta)` の production caller

| caller | File | Line | 呼び出し |
|---|---|---|---|
| `AudioEngine.Timer.cpp` sampler | `AudioEngine.Timer.cpp` | 429 | `telemetry.addReleaseObserved(worldReclaimed - prevReclaimed)` |
| — | — | — | 他の production caller は存在しない |

### 1.4 結論（唯一入口の固定）

```
Telemetry.releaseObserved_
        ↑ 唯一
AudioEngine.Timer.cpp sampler — addReleaseObserved(worldReclaimCount delta)
        ↑ 唯一
worldReclaimCount() aggregation (Deferred + Q + E + Terminal)
```

`onReleaseObserved()` の production caller が sampler 以外に存在しないことを確認してから実装に入る — 本R3で確認済み（上記1件のみが該当し、R3でその1件を切断する）。

---

## 2. R3-2 — ReferenceObserver の責務を最小変更で固定

### 2.1 変更対象ファイル

`src/audioengine/ISRWorldRetirementReference.h` **のみ**に限定する。他ファイル（`ISRWorldRetirementTelemetry.h`, `AudioEngine.Timer.cpp`, `DeferredDeletionQueue.h`, `RetireQuarantineStore.h`, `ISRRetireRouter.*`）は変更しない。

### 2.2 変更後の意味論

```
WorldRetirementReferenceObserver::onRelease()
    ├─ referenceReleaseCount_++        (維持)
    └─ updateRunningMax()             (維持)
    // telemetry_->onReleaseObserved()  ← 完全に切断（削除）
```

### 2.3 `telemetry_` member の扱い

R3 の最初の実装では **member を即座に削除しない**。理由:

1. semantic change（転送の切断）と構造整理（member削除）を分離することでビルド/テストの帰責を明確にする。
2. `telemetry_` は `non-owning` ポインタであり、参照を止めるだけで副作用なし。削除は次の小さい変更で `rg -n "telemetry_"` が 0件になることを確認してから行う。

手順:

```
Step 1: onRelease() から telemetry_->onReleaseObserved() 呼び出しを削除（1論理行）
Step 2: build（Debug/Release）
Step 3: test（既存 WorldRetirementMeasurementTests 等が reference max のみを検証することを確認）
Step 4: rg -n "telemetry_" で caller/member の不要性を再確認
Step 5: 必要なら次のコミットで member / setTelemetry() / include を削除
```

一度に semantic change と構造整理を混ぜない。

---

## 3. R3-3 — `onReleaseObserved()` の API 契約を固定

`ISRWorldRetirementTelemetry.h:88-91`:

```cpp
void onReleaseObserved() noexcept {
    fetchAddAtomic(releaseObserved_, 1, acq_rel);
}
```

R3 での契約（コード変更なし、コメントで明文化）:

> `onReleaseObserved()` は T1 release measurement の正規入口ではない。
> T1 の `releaseObserved_` は sampler の `addReleaseObserved(delta)` のみが更新する。
> 本メソッドは将来 private 化 or 削除の対象だが、R3 では visibility を変更しない（double-count 除去が目的であり API 整理ではないため）。

R3 では `private` 化や削除を行わない。R3-A Gate 通過後に別タスクで検討する。

---

## 4. R3-4 — sampler の一次情報源を再確認

`AudioEngine.Timer.cpp:420-433`:

```cpp
// ★ T1 (D86): Non-RT sampler — A/R loads → signedWide → estimate → max → window tag
auto& telemetry = worldRetirementTelemetry();
const uint64_t worldReclaimed = m_retireRouter ? m_retireRouter->worldReclaimCount() : 0;
const uint64_t prevReclaimed = consumeAtomic(lastSampledWorldReclaimCount_, acquire);
if (worldReclaimed > prevReclaimed) {
    telemetry.addReleaseObserved(worldReclaimed - prevReclaimed);
    publishAtomic(lastSampledWorldReclaimCount_, worldReclaimed, release);
}
const int64_t estimate = telemetry.observedOutstandingEstimate();
telemetry.updateObservedOutstandingMax(estimate);
```

確認事項:

| 項目 | 確認結果 |
|---|---|
| `worldReclaimCount() → delta → addReleaseObserved(delta)` が唯一の T1 release 経路 | Yes（§1.3） |
| `prevReclaimed / lastSampledWorldReclaimCount_` | 変更しない（R2で Window/max 再評価は禁止、release source 一本化のみが検証対象） |
| `window / max / observedOutstandingEstimate` | 変更しない（sampler 内の A/R loads → signedWide → max → window tag は現設計のまま） |
| Recovery supersession への波及 | なし（AudioEngine.Timer.cpp に Recovery/supersession/coalesce は存在しない — `rg Recovery src/audioengine/ISRWorldRetirement* src/audioengine/AudioEngine.Timer.cpp` 0件） |

---

## 5. R3-5 — 設計文書の semantic correction を先に確定

コード変更と同時に文書を大量修正せず、まず semantic のみを固定する。

| 文書 | R3での扱い |
|---|---|
| D86 | `worldReclaimCount` authoritative を維持（一次情報源は storage 側） |
| D94 | Observer = measurement-only / non-owning を維持 |
| D98 | Observer は T1 release authority ではないことを明記（telemetry への転送を否定） |
| D99 | reference running max と T1 release measurement を分離（別カウンタ） |
| D100 | `independent observation` → **separate observation** に訂正 |
| D100.4 | `telemetry への release 転送` を削除（`reference observer → telemetry.releaseObserved` の接続を否定） |
| D101-9 | R2 の `1 destruction = 1 release` を反映（`releaseObserved = sampled physical World destruction count`） |

特に **D100.4 は明確に反転**させる。現行 ConvoPeq.md の `ISRWorldRetirementReference.h:37-38`:

```
// ★ T1 (D100.4): sampler の outstanding 推定を正しくするため、
//     telemetry の releaseObserved にも転送する（同一 terminal release を両観測系が観測・D100 の独立観測）。
```

は double-count を設計上許容する記述であり、R3 では「分離観測（separate）に転送しない」に訂正する必要がある。

---

## 6. R3 実装後の最初の Gate — Gate R3-A

実装後に以下を全て満たすこと。1項目でも不一致なら `T1-C1` を再開しない。

- [ ] `onRelease()` から `onReleaseObserved()` への呼び出しが消えている
- [ ] `worldReclaimCount` increment は 9 LP とも変更されていない
- [ ] `worldReclaimCount()` aggregation（`ISRRetireRouter::worldReclaimCount()`）は変更されていない
- [ ] Timer sampler（`AudioEngine.Timer.cpp:420-433`）は変更されていない
- [ ] `addReleaseObserved(delta)` が sampler 経由だけになっている
- [ ] `releaseObserved_` の直接 mutation が新規発生していない（`rg releaseObserved_ src/` が Telemetry.h の2 hits のまま）
- [ ] `ReferenceObserver` の `referenceReleaseCount_` は維持されている
- [ ] reference running max（`updateRunningMax()`）は維持されている

---

## 7. R3 の次 — T1-C1 再検証

Gate R3-A が PASS したら初めて T1-C1 を再開する。最初の検証は soak ではなく R1 counterexample の反転:

```
World destruction × 1
        ↓
worldReclaimCount delta = 1
        ↓
sampler addReleaseObserved(1)
        ↓
releaseObserved delta = 1        // R1では 2 だったものが 1 に反転
```

同時に:

```
ReferenceObserver.onRelease()
        ↓
referenceReleaseCount += 1
reference running max update
        ↓
releaseObserved += 0              // R1では +1 だったものが 0 に
```

R1 の `1 destruction → storage +1 → observer +1 → sampler +1 → releaseObserved +2` を `1 destruction → storage +1 → observer(reference max only) → sampler +1 → releaseObserved +1` へ反転できたことが **T1-C1-R3 の第一目的**。

その後の順序:

```
R2 → R3 design → R3 implementation → Gate R3-A → T1-C1 → T1-C2 → T1-C3 → ... → T1-C8 → K_terminal → R gate
```

途中で T1-C1 が FAIL した場合は C2 以降へ進まない。

---

## 8. Tool Coverage（R3 全系統を使用）

| 系統 | ツール | 実行 | 結果 |
|---|---|---|---|
| WSL#1 | rg 15.1.0 | `rg -n "releaseObserved_" src/` | Telemetry.h 4 hits のみ |
| WSL#2 | ast-grep sg 0.44.0 | `sg --help` 動作確認（`search` 非対応） | rg で代替列挙を補完 |
| WSL#3 | fdfind 10.3.0 | `fdfind -t f "ISRWorldRetirement"` | Telemetry/Reference 発見 |
| WSL#4 | ag 2.2.0 | `ag "releaseObserved" src/` | add/onReleaseObserved のみ |
| WSL#5 | fzf 0.67.0 | `fzf --version` | 0.67.0 動作確認 |
| WSL#6 | sed 4.9 | `sed -n "410,450p" AudioEngine.Timer.cpp` | sampler 確認 |
| WSL#7 | awk 5.3.2 | `awk` version | 5.3.2 動作確認 |
| MCP#1 | serena | `find_symbol releaseObserved/onRelease` | Telemetry/ Reference の定義を確定 |
| MCP#2 | ccc | `ccc search "WorldRetirement"` | Not in initialized project（本リポ未初期化）— rg/semble で補完 |
| CLI#1 | graphify 0.9.48 | `graphify --help` | 動作確認（graphify-out 未生成のため serena/semble で補完） |
| CLI#2 | semble 0.5.5 | `semble search "World retirement reference observer"` | Reference.h / RetireQuarantineStore 等を top hit として取得 |
| MCP#3 | AiDex | (R2で `onRelease` 12 hits 確定済み) | 本R3は rg/serena で再確認 |
| 文献 | crossbeam-epoch/rigtorp/serena/cocoindex/ast-grep/semble/AiDex/graphify/headroom | urllib 200 OK 9件（Vyukov SSL失効→rigtorp代替） | D86/D83.2 補助参照 |

---

## 9. 補足 — 本R3のスコープ外

- Recovery coalesce / supersession（D12-17 / I4 / REPAIR_PLAN2-dash2）の `domain coverage だけでは supersession を成立させない` 固定点は本R3では触らない。
- 基準ソースは `ConvoPeq.md` 2026-08-21 20:45:49 生成の最新スナップショットを維持する。
- `T1-C1` 測定再開・`K_terminal / R gate` への進行は本R3設計の実装と Gate R3-A 通過まで禁止。

---

*Evidence generated: Phase I-T1-C1-R3 — design only, no code change in this phase.*
