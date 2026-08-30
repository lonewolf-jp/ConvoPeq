# D129-0 — Crossfade Completion State-Machine Audit

**Date:** 2026-08-29
**性質:** read-only 状態機械監査。**production source 変更 0**
**目的:** M2（crossfade completion 駆動）を実装契約に落とし込む前に、completion の状態遷移・identity・overlap・liveness を line-level で証明する
**判定:** **CONDITIONAL — bool 型 completion は不十分（G2/G5）。M2 契約の再設計が必要 → 直ちに D129 実装へ進まない**

---

## 1. 現行 state machine（実コード確定）

### CrossfadeRuntime（CrossfadeRuntime.h）

| 状態/ API | 挙動 | thread |
| --- | --- | --- |
| `start(fadeTimeSec, sr)` | `gain_.reset`（totalSteps 確定のみ）+ `pending_=true` + `fadeStartTimestampUs_` 記録 + generation bump。**gain の 0 リセットは行わない**（BUG-028） | NonRT（DSPTransition） |
| `armCrossfadeIfPending`（AudioEngine.h:4003） | RT が `applyImmediateValueRT(0.0)` + `setTargetValue(1.0)` で ramp 開始（pending_ AND 条件で無限再Arm防止） | RT |
| `getGain().getNextValue()/skip()` | ramp 進行。`remaining` 1→0 で `current=target` 収束 | RT |
| `getGain().isSmoothing()` | `remaining > 0`（DspNumericPolicy.h — RT 専用読み、非atomic） | RT |
| `notifyFadeComplete(id)` / `notifyRampComplete()` | `CompletedFadeEvent` を SPSC queue（容量32）へ push。full 時は drop + dropCount | RT |
| `consumeCompletedFade(ev)` | Timer が消費 | Timer |
| `complete()` | pending_ 等 stale flag クリア + generation bump | NonRT |

### LinearRamp（DspNumericPolicy.h:313-435）

- `remaining` は **非 atomic・RT 専用**（`ASSERT_AUDIO_THREAD` 全 mutator）
- `getNextValue()`: `if (--remaining <= 0) current = target` — **1→0 エッジは ramp 生存中に正確に 1 回**
- `isSmoothing()` = `remaining > 0` — **NonRT から読むとデータレース**（BUG-028 の教訓）→ 方式 B（Timer が isSmoothing を監視）は構造的に不成立

### CrossfadeAuthorityRuntime（ISRDSPHandle.cpp:286-305）

- `registerCrossfade(from,to)`: `nextId_` fetch-add で id 発行 + `records_.push_back({id, from, to, 0, true})` + `fadingRuntimeDSPHandle_ = from`
- `unregisterCrossfade(id)`: 該当 record の `active=false`（**records_ は erase されず滞積** — 長時間運用での成長要因・要確認）
- `getActiveCrossfades()`: active record のみ返す（**mutex 使用 — RT から呼べない**）

## 2. 全 writer / reader

| field | writer | reader |
| --- | --- | --- |
| `gain_` (LinearRamp) | RT: armCrossfadeIfPending（applyImmediateValueRT/setTargetValue）/ mix loop（getNextValue） | RT: isSmoothing / getNextValue |
| `pending_` | NonRT: start/complete/reset | RT: isPending（arm 判定） |
| `completedFadeQueue_` | RT: notifyFadeComplete / notifyRampComplete | Timer: consumeCompletedFade |
| `fadingRuntimeDSPHandle_` | beginCrossfade（=from）/ activate（=null）/ endCrossfade（=null） | Timer retire 提出ブロック / DSPTransition storeReceipt / 検証関数 |
| `fadingRuntimeDSPSlot`（DSPCore\*） | claimFadingRuntimeDSP（CAS）/ Timer・onTransitionComplete の CAS clear | getActiveRuntimeDSP 系 |
| `crossfadeRecords_` | CoordinatorLoop: register / Timer: endCrossfade（active=false） | Timer: getActiveCrossfades（mutex） |

## 3. G1 — Completion source: **PASS（条件付き）**

- `remaining` 1→0 エッジは ramp 生存中に正確に 1 回（`--remaining <= 0` 収束、DspNumericPolicy.h:389-396）
- RT block で `wasSmoothing`（block 開始）→ mix → `!isSmoothing()`（block 終了）のエッジ検出で exactly-once 発火可能
- 発行 primitive: 既存 SPSC `completedFadeQueue_` push — **zero alloc / lock-free / non-blocking** ✓（G7）
- **Timer 側からの remaining 推測は不成立**（非atomic・データレース）→ RT 1→0 edge が唯一の completion source

## 4. G2 — Completion identity: **FAIL（bool は不十分 → id 携帯が必須）**

- `CrossfadeId` は `nextId_` fetch-add で発行（ISRDSPHandle.cpp:288）— 世代タグ付き record と対になる
- **fade A 完了と fade B 開始が重なる場合**（高速連続 publish）、id なしの bool signal では Timer が「どの crossfade の完了か」を判定できない → **B の record を誤 endCrossfade する ABA 危険**
- 対策: RT がエッジ検出時に id を付与する必要があるが、RT は `getActiveCrossfades()`（mutex）を呼べない → **id を RT が読める経路**が必要:
  - 案 i: `CrossfadePreparedSnapshot`（world 経由で RT に到達）に xfadeId を載せる（beginCrossfade 発行 id を world publish 時に埋め込む）
  - 案 ii: `fading` DSPCore\* は RT が既に保持（AudioBlock.cpp:348）→ イベントに handle を添え、Timer 側で `resolve` 検証（generation-safe）
- いずれも既存 primitive の組合せで実装可能だが、**bool 単独は不認可**

## 5. G3 — Record resolution: **PASS**

`CrossfadeId → records_ 走査（mutex, NonRT のみ）→ record.fromHandle → resolve(handle).instance（generation 検証済み）→ DSPCore*` — 一意に解決。generation 不一致は stale として排除（ISRDSPHandle.cpp:67-70）。

## 6. G4 — Exactly-once retire: **PASS（条件付き）**

- 二重 completion: 1 回目の endCrossfade で record.active=false → 2 回目は active 走査で不一致 → no-op
- 二重 retire: `retireDSPHandleForRuntime` が map erase 済みで false → DSPLifetimeManager::retire は静かに return（冪等）
- **completion 喪失時の liveness**: 現状セーフティネットなし → (a) fade 超過タイムアウト（`getFadeAgeUs` — Practical-2 で既存）による強制 retire decision、または (b) claim 失敗時の即時 retire（現行 DSPTransition 分岐 — 既存）が担う。**タイムアウト net を D129 契約に追加すること**

## 7. G5 — A→B→C overlap 状態表: **FAIL（現行設計に滞留パス）**

| イベント | fadingRuntimeDSPSlot | crossfadeRecords_ | 結果 |
| --- | --- | --- | --- |
| B publish（A→B crossfade） | CAS 成功 → **A** | A→B record（active） | A は fading、B active |
| A→B completion（M2 実装後） | CAS clear → null / A retire | A→B inactive | 正常 |
| **C publish（A→B 未完了時に発生）** | claim(B) → **CAS 失敗**（slot=A 占有） | — | **即時 `lifetime.retire(B)`** — B は commit 直後の current であり EBR 保護で destroy は安全だが、**slot に残った A は誰にも retire されず永久滞留** |

- **A→B→C で A が漏出する**（claim 失敗分岐が「新旧いずれか 1 体」しか処理しないため）
- 修正契約（D129 へ）: claim 失敗時、**slot の現占有者（A）と oldDSP(B) の双方を retire 対象にする**（exchange semantics — 旧 `exchangeFadingRuntimeDSP` の復活または retire チェーン化）。A/B 取り違え防止は DSPCore\* 比較（CAS のため pointer identity は自明）と resolve 検証で担保
- `fadingRuntimeDSPHandle_` の A/B/C 取り違え: C の beginCrossfade が fading handle を B で上書きするが、A の retire は handle に依存せず DSPCore\*（slot 値）で行うため混同しない設計にする

## 8. G6 — Deferred admission liveness: **PARTIAL（診断 1 点残存）**

- deferred publish は `evaluateDeferred`（Shutdown → TTL 30s → Generation → Sequence）で判定 — **fade 完了を条件にした deferral ではない**
- D127 実測の 15,796 RetryReady wake（≈450Hz・30s+継続）の具体的 defer 理由は**未特定**（TTL 30s を超えても discard されず 滞留 DSPCore が残存 — TTL 判定が機能した形跡なし）
- **fade completion 後の forward progress**: deferral 条件が fade 非依存であるため、completion とは独立に進むはずだが、**gen-5 が 30s+ 未 commit だった実測と矛盾** → submitPublishRequest/deferred 経路の拒否理由トレース（macro-gated 1 点）を D129 診断に追加するまで「forward progress 保証」は未証明

## 9. G7 — RT 制約: **PASS**（上記 4 のとおり — SPSC push のみ・alloc/lock/blocking なし）

## 10. G8 — production source 変更 0: **PASS**

---

## VERDICT: **CONDITIONAL — 3 項目の契約修正後に D129 実装可**

| # | 判定 | D129 契約への反映 |
| --- | --- | --- |
| G1 | PASS | RT 1→0 edge を唯一の completion source とする |
| G2 | **FAIL → 設計修正** | completion event は **CrossfadeId（または fading handle）を携帯**。bool は不認可。id の RT 到達経路（world 経由 or handle 添付）を D129 仕様に明記 |
| G3 | PASS | id → record → fromHandle → resolve → DSPCore\* の一意解決 |
| G4 | PASS + 追加 | 冪等 retire 確認 + **fade 超過タイムアウトによるセーフティネット retire** を契約追加 |
| G5 | **FAIL → 設計修正** | claim 失敗時の exchange-retire semantics（slot 占有者 + 新 oldDSP の双方 retire）を契約追加。さもなくば overlap 時に必ず 1 体漏出 |
| G6 | PARTIAL | fade 非依存の deferral 条件であることを確認済み。forward progress の実証に admission-reject reason trace（macro-gated 1 点）を追加 |
| G7 | PASS | RT 制約遵守（SPSC push のみ） |
| G8 | PASS | production 変更 0 |

## D129（実装）に持ち込む最終スコープ

```text
M1  activate 公開（DSPTransition normal/emergency）
M2  completion: RT 1→0 edge → CrossfadeId 携帯イベント → Timer consume → endCrossfade
    + fading slot exchange-retire（overlap 対応）
    + fade 超過タイムアウト net（getFadeAgeUs）
M3  wake reason 分離 + stale task build ガード
M5  Observe retire 誘発撤去
M6  shutdown 診断（logger 保持）
+   admission reject reason trace（macro-gated 診断 1 点）
```

**M2 が単独で最大リスク**（D119/D123 の教訓）— 実装時は M1/M3 同時適用 + Gate 3 での TASK_WAKE/TRANS/RETIRE 対合を必須とする。

## 生成物

- 本ファイル（`evidence/D129_0_CROSSFADE_COMPLETION_STATE_MACHINE_AUDIT.md`）
- production source 変更: **0**
