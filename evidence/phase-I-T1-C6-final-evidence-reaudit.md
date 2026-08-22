# Phase I-T1-C6-Final — Evidence Re-audit

> **Verdict: T1-C6 = PASS（G2 = 0 / Residual Gap = G1 × 3）**
> **T1 measurement run = GO**（normal / burst / jitter を規定プロトコルで実施し O_w / T_w / E_w / observedOutstandingMax を記録）。
> **R = UNDETERMINED のまま**（M-bound / D101 未着手・A_max candidate 記録まで R/R_cap/M は決定しない）。
> コード変更 = 0。

## 1. G2-1 revalidation（release observation）

| 項目 | 実測 | 判定 |
|---|---|---|
| `releaseObserved()` assertion | 9 reads（WorldRetirementMeasurementTests.cpp） | ✅ |
| `worldReclaimCount` measurement test | 16 refs | ✅ |
| `ΔworldReclaimCount == ΔreleaseObserved` | :416 `dRelease != dReclaim → FAIL` | ✅ |
| N=1 / N>1 | `g2-single`(N=1) / `g2-multi`(N=4) 実測 PASS | ✅ |
| repeated sampler double transfer | `no-double-transfer` 6 sites（:461-531 一貫性 + 再駆動 +0） | ✅ |
| reference → releaseObserved 誤結合 | 静的 0 件 + 動的 `dRef == dReclaim == N`（加算なら 2N を検出） | ✅ |
| production writer | 共有 transfer step のみ（Telemetry.h fetchAdd 2 件はメソッド定義） | ✅ |

## 2. G2-2 revalidation（max observation）

| 項目 | 実測 | 判定 |
|---|---|---|
| `observedOutstandingMax()` assertion | 4 reads | ✅ |
| `updateObservedOutstandingMax()` 到達性 | caller 1 件 = 共有 step 内（Timer.cpp:404）・harness 到達を実測（M0=0→M1=1） | ✅ |
| production/harness 同一 step | `runWorldRetirementMeasurementStep` call site 2（timerCallback + harness）・harness 独自実装なし | ✅ |
| first step writer liveness | `m0 != 0 → FAIL` 前提 + `m1 <= m0 → FAIL`（実測 0→1） | ✅ |
| monotonicity | 4 cycle `m < mPrev → FAIL` | ✅ |
| estimate 低下で max 減少しない | drain 後 `mFinal != mPrev → FAIL` | ✅ |
| 新規破壊なし再 step で max 不変 | 同上（追加 step で不変を実測） | ✅ |

### 共有 step 構造の再確認

```
timerCallback ─┐
               ├─ runWorldRetirementMeasurementStep()
harness ───────┘   ├─ transferWorldReclaimDeltaForTelemetry()   ← releaseObserved 唯一 writer 経路
                   ├─ observedOutstandingEstimate()
                   ├─ updateObservedOutstandingMax()
                   ├─ setWindowTag(Shutdown|Normal)
                   ├─ samplerTick(nowUs)
                   └─ reference onMeasurementStart/End
```

- 全 7 API が共有 step 本体に存在することを確認。
- 順序契約 `transfer < estimate < max < tag < tick` をソース位置で検証 **PASS**。
- harness の独自 sampler 実装（旧 `telemetry.samplerTick` 直呼び）は **削除済み**（ABSENT PASS）。

## 3. C6-F2 — 15-field contract の動的 coverage 最終判定

| Field | Final判定 |
|---|---|
| `acquireObserved` | deterministic assertion（静穏 baseline で dAcquire==N / reject=0 は既存テスト群） |
| `releaseObserved` | deterministic assertion（dRelease==dReclaim==N） |
| `observedOutstandingEstimate` | `A - R` identity assertion（実測一致） |
| `observedOutstandingMax` | liveness + monotonic + decrease-immunity assertion |
| `windowTag` | Normal 到達 assertion |
| `windowTagName` | **G2 対象外** — `windowTagName(int tag)` の純関数であり tag assertion により導出値も動作保証される（診断文字列） |
| `windowId` ほか window 9 field | window 機構として normal/burst/jitter で動的走行（Closed snapshot 読み・O_w/T_w 比較）。field 単位の値 assert は無し → **G1（JSON export content gap に含む）** |

**semantic separation 確認:** `observedOutstandingMax`（accumulated live max）と `measurementWindow.windowMax`（window-local sampled max）の cross-assertion はテスト中に存在せず（ABSENT PASS）、C5 の mapping も別 source を維持。I4 の「B_max^observed を sampled maximum 扱いしない」契約と整合。

## 4. C6-F3 — Terminal path 再評価

1. 各経路 → aggregate 収束: `ISRRetireRouter::worldReclaimCount() = provider_(D) + Q + E + Terminal.reclaimCount`（変更なし・再確認）。
2. aggregate → shared transfer が一意（caller 1）。
3. `releaseObserved` は aggregate の downstream に 1 回のみ（transfer 内 addReleaseObserved のみ）。
4. reference observer は `releaseObserved` の writer ではない（R3 状態維持・census 0）。
5. per-store 未動的テストの評価: increment site は 9 LP 全て静的列挙済み（R2/C2 で 1:1 確認・2 度の再監査で不変）。仮説上の「per-store increment 漏れ」は trivial な `++` 欠落であり静的に排除され、かつ D 経路で同一形状が動的証明済み。**新たな G2 counterexample は発見されなかった** → per-store end-to-end telemetry test は **G1 に留める**（G1-3 の scope に含む）。

I4「`releaseObserved++` は terminalization 成功後 1 回だけ」との照合: 9 LP 全てが deleter 成功直後の同一 `if(type==World)` block 内で counter++ と onRelease を 1:1 で実行する構造は不変（C4 以降の census で毎回確認）。

## 5. C6-F4 — adversarial counterexample census

| Counterexample | 検出手段 | 必須 | 判定 |
|---|---|---|---|
| destruction 1 → release 0 | :416 `dRelease != dReclaim` | YES | ✅ 検出可能 |
| destruction 1 → release 2 | 同上（2N を含む不等検知） | YES | ✅ |
| N destruction → release ≠ N | 同上（N=1/N=4 で実行） | YES | ✅ |
| sampler 再実行 → release +N | no-double-transfer（:518-520 条件付き +0、:487 累積整合は無条件） | YES | ✅ |
| reference release が releaseObserved に加算 | 静的結合 0 + 動的 `dRelease==N`（+1 なら 2N 検出） | YES | ✅ |
| max が現在値低下に追従して減る | drain 後 `mFinal != mPrev` | YES | ✅ |
| max writer が harness から到達不能 | `m0 != 0` 前提 + `m1 <= m0` liveness | YES | ✅ |
| `A-R` と exported estimate の不一致 | `observedOutstandingEstimate() == (int64)A-(int64)R` assert ＋ export が同一 getter（C5 #3） | YES | ✅ |

**8/8 検出可能 — G2 継続要因なし。**

## 6. C6-F5 — `#28 HeadlessAudioPathVerification` 分類

| 問い | 答え |
|---|---|
| 今回変更した measurement code を含む binary か？ | **No** — `build-icx/.../Release/ConvoPeq.exe`（icx ツールチェイン・別ツリー）。Debug(MSVC) pipeline はこの成果物を rebuild しない |
| measurement / retire counter path と code-sharing するか？ | **No** — CLI smoke は `[CLI_PERF_RAW] callbacks` ログ（audio callback 数）を検査。T1 counter/step と失敗信号の共有なし |
| measurement correctness の failure signal か？ | **No** — static-teardown `0xC0000005` は許容済みで、positive callback count の log 欠落（icx 成果物側の問題）で throw |

→ `C6 regression = 0`、`#28 = external / unrelated environment failure` として C6 gate から分離。

## 7. C6-F6 — Residual Gap（G1 × 3）

| ID | 内容 | G1 である根拠（measurement correctness に直接影響しない） |
|---|---|---|
| G1-1 | cross-window isolation の characterization | 計画中の characterization は 1 instance 1 window で成立。Closed→Idle 遷移は C5-4 静的确認済み。multi-window 持続観測（D101 以降）で prerequisite 化 |
| G1-2 | JSON 24-field export content の dynamic test | export は同一 getter の直列化のみ（C5 で conservation 確証）。counter 正しさの問題ではない |
| G1-3 | 背景併走時の exactly-1 event 粒度 ＋ per-store end-to-end telemetry | 静穏系では deterministic 実測済み（g2-single/multi）。背景併走粒度と Q/E/T 経路の観測側 e2e は increment site 静的列挙（1:1・二度監査）で correctness は担保済み |

新規 G2 counterexample は発見されなかったため、いずれも G1 へ逃がしたものではない。

## 8. Final gate

| Gate | 条件 | 判定 |
|---|---|---|
| C6-F1 | G2-1/G2-2 revalidation | ✅ |
| C6-F2 | 15-field coverage（max≠windowMax 分離維持） | ✅ |
| C6-F3 | Terminal path（G2 昇格なし） | ✅ |
| C6-F4 | adversarial 8/8 検出可能 | ✅ |
| C6-F5 | #28 = external 分離 | ✅ |
| C6-F6 | Residual G1 × 3 確定 | ✅ |
| — | コード変更 = 0 | ✅ |
| — | G2 = 0 | ✅ |

```
T1-C6 = PASS
G2 = 0
Residual Gap = G1 × 3
T1 measurement run = GO
```

## 9. Tool Coverage

node fs census（41 files・全項目）/ source 直読（Timer.cpp・AudioEngine.h・Telemetry.h・Reference.h・WorldRetirementMeasurementTests.cpp）/ rg・sed・awk（WSL）/ serena・AiDex・semble・ccc・graphify は C2〜G2-2 で確定済みのシンボル参照との整合維持 / 文献 9 系統は前 Gate までに 200 OK 済。

## 10. 次フェーズ（measurement run）の規約

- normal / burst / jitter を規定プロトコルどおり実施し `O_w / T_w / E_w / observedOutstandingMax` を記録する。
- **A_max candidate の記録まで**を範囲とし、`R / R_cap / M` は UNDETERMINED のまま（I4: 有限実測 max(E_w) を安全上界 M としない・D101 未着手）。
- measurement protocol（sampling interval・A/R・sampled maximum・window metadata）を揃えたうえで `B_max^observed` 特性を評価する。

---

*Evidence generated: Phase I-T1-C6-Final — no code change. T1-C6 PASS 確定・measurement run GO。*
