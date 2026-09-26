# P1-5-IR-P2 — Step 5-AJ / P3-5-R28: Origin-aware Publish Measurement / C-1・C-2 Observability Closure

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R28）
- **種別**: read-only measurement / analysis gate。production 0・test 0・CMake 0・new counter 0・new getter 0。
  Build 0（R27 binary を使用）。F/R 本測定・P3-1-D・原因帰属なし。
- **判定**: **R28-B（Observability insufficient）**。
  Main baseline は `cmt / coord / seq / drp` の同一 window 取得と **C-1方向の観測例**を再現可能な形で与えた。
  しかし **Recovery-origin publish は既存 vehicle では発生させられない**（R28 §4 の必須課題が未達）。
  新 counter／新 test を追加せずに閉じる（R28 §13）。

---

## 1. State Freeze（PASS）

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix = true（P1PolyphaseGainCharacterization.cpp:43）
CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（build/CMakeCache.txt:232 BOOL=OFF 実測）
保持: R16 take／R19 build-result／R22 commit／R27 origin plumbing／R10 accessor／T6/T9／R23 vehicle
ConvoPeq.md 再生成済み（2026-09-23 22:41:29・5,525,709 B・generator output_sourcecode_markdown.py）
R27 binary 使用（build/Release/AudioEngineHarness.exe）
  SHA-256 425158012318dde394340b9118b23ffeedff22d15c617ea1a08b17743f7dbd55（R27 と同一）
working tree = R27 implementation state（追加変更 0・本 Step の source 差分は本ドキュメントのみ）
```

- `ConvoPeq(2).md` は本環境のいずれの探索 root（VSC_Project／Downloads／Desktop／Documents／Temp）にも
  存在しなかった（§2）。代替として generator で `ConvoPeq.md` を再生成し、それを source authority とした。

## 2. Latest ConvoPeq source reconciliation（PASS・R27差分なし）

再生成 `ConvoPeq.md` 上で R28 §2 の要求 marker を確認（すべて R27 implementation state と一致）：

| marker | 確認結果 |
| --- | --- |
| `recoveryObligationId` | 存在（PublishPayload NSDMI・Intent 搬送・req 設定） |
| `trySubmitImpl` | 存在（admission→build→publish の単一実装） |
| `PublicationExecutor::publish` | 存在（`recoveryObligationId` 引数追加済み） |
| `commitRuntimePublication` | 存在（`recoveryObligationId` default 0 中継） |
| `enqueueRuntimePublicationFireAndForget` | 存在（`intent.payload.publish.recoveryObligationId = recoveryObligationId`） |
| `processIntent` | 存在（Publish pop で `coord` Main-only 加算） |
| `coordinatorTakeCount_` | 存在（decl 1／getter 1／writer 1） |
| `lastCommittedPublicationSequence_` | 存在（commit 成功で publish） |
| `lastDroppedGeneration` | 存在（commit 側 monotonicity reject のみ） |

- **R27実装との差分は source 上で発見されず**（測定続行）。追加 source 変更もしていない。

## 3. Measurement vehicle / command

```text
vehicle : build/Release/AudioEngineHarness.exe（R27 binary・rebuild なし）
Run M   : AudioEngineHarness.exe --p1-char（R23 vehicle・kP15G0OS1Only・Main-only・Recovery 非発生）
integrity: AudioEngineHarness.exe（引数なし＝publish pipeline＋deferred 回帰）
```

- `coord` を出力する唯一の既存 vehicle は `--p1-char` の `emitQDelta`
  （`req/que/dup/take/bld/cmt/coord/drp/seq/blo`）。
- 窓は pre/post の6点＋その間の gap（post→次 pre）を保存（R28 §8）。

## 4. Main baseline（Run M・no-recovery）

```text
flag_macro=0（OFF build）／summary cases=1 failures=1
qdelta:
  p1 pre :  req=2  que=2  dup=0 take=2 bld=2 cmt=2 coord=3 drp=0 seq=5 blo=0
  p1 post:  req=5  que=5  dup=0 take=4 bld=3 cmt=2 coord=3 drp=0 seq=5 blo=0
  p2 pre :  req=6  que=6  dup=0 take=5 bld=4 cmt=3 coord=5 drp=0 seq=8 blo=0
  p2 post:  req=8  que=8  dup=0 take=6 bld=4 cmt=3 coord=5 drp=0 seq=8 blo=0
  p3 pre :  req=10 que=10 dup=0 take=8 bld=7 cmt=5 coord=5 drp=0 seq=8 blo=0
  p3 post:  req=12 que=12 dup=0 take=9 bld=7 cmt=5 coord=5 drp=0 seq=8 blo=0
```

- R27 の同 vehicle 結果と **cmt/coord/seq/drp が完全一致**（cmt=2,2,3,3,5,5／coord=3,3,5,5,5,5／
  seq=5,5,8,8,8,8／drp=0 全点）。`take` のみ run 間で ±1 変動（非決定・既知）。
- 形状保存：gainDb_sc0=sc1=**-13.0276**（R23/R27 と同一）／gen=8・seq=8／ab0_fade=0.0600／
  `publish not confirmed os=1 sc=0`／failures=1／crash・assert 0。
- **Main bucket 背景の実測**：最初の pre で既に `coord=3`。vehicle 自身の最初の publish 前に
  3 回の Main-origin pop（bootstrap/idle publish）が発生済み。
  → `coord` は main-site rebuild 専用ではなく obligationId==0 全体（R26 §4.3・R27 §6 の再確認）。

### Integrity run（default harness）

```text
"AudioEngineHarness: all publish pipeline tests PASS"（deferred 回帰含む）
Failure/abort/exception 0
```

## 5. Recovery-origin run（Run R）— **実行不能（vehicle 不在）**

### 5.1 試行結果

```text
Recovery-origin publish attempt : 0（発生させられず）
Recovery-origin Coordinator pop : 0（同上）
→ R28 §4 の必須課題は未達。
```

### 5.2 到達不能の source 根拠（exhaustive search）

Recovery-origin publish は以下の一本のみ：
`quarantine → QuarantineIntentHandler → submitRecoveryIntent → submitRecoveryRequest →
Builder Work Queue → enqueuePublicationIntentForRuntimeCommit(..., obligationId != 0)`。

production の quarantine 発火点は3箇所、いずれも既存 vehicle から到達不能：

| 発火点 | source | 到達性 |
| --- | --- | --- |
| retire handle mismatch（PublishViolation） | Timer.cpp:1975（`retirePublishedDSP` CAS 不一致分岐） | 非決定・mismatch を誘発する vehicle なし |
| receipt reset（ReceiptReset） | Timer.cpp:2013 | **`resetReceipt()` の caller が 0**（def＋decl のみ・dead path） |
| retire deferral timeout（RetireDeferralTimeout） | Commit.cpp:623／643（`quarantineSlot`） | `quarantineSlot` は AudioEngine private。`pending.dspSlot != UINT32_MAX`（実 DSP slot）が必要。ca 経路の soak は `RegistrationContext::none()` のため slot なし |

- `submitRecoveryIntent` の生存 caller は QuarantineIntentHandler のみ
  （RecoveryIntentHandler は「誰も intentQueue_ に Recovery Intent を push しない dead code」と
   source 自身が明記）。
- **既存 test／harness に full AudioEngine で quarantine を起こすものが存在しない**：
  `src/tests` の quarantine 言及 186 件のファイルは
  `D8_2_B_2_Tests / ISRSemanticValidationTests / ISRSoakTests / PriorityIntegrationTests /
   RetireGraceSemanticsTests / RuntimeHealthMonitorTierTests / StuckReaderFallbackDrainTests /
   invariant_INV3_INV5` の8件のみで、いずれも**コンポーネント単体**（AudioEngine 非使用）。
  harness（AudioEngineHarness）配下の quarantine 言及は **0 件**。
- 回復系 getter も harness から不可視：`liveLogicalRecoveryObligationCount()` は **caller 0**、
  `recoveryCoalescedCount()` 等の test reader なし。
- soak の publish vehicle（`publishOne`）は `buildRuntimePublishWorld(nullptr,nullptr,…)`＋
  `RegistrationContext::none()` の **idle-only** であり quarantine を惹起しない。
  `--soak` 経路は `coord` を出力しないため、いずれにせよ測定寄与なし。

### 5.3 判定

```text
Recovery-origin transport / coord 除外 / D105 動的 completion は、
既存観測・既存 vehicle では 発生も観測もできない（vehicle limitation）。
新 counter／新 test 追加は R28 §13 で禁止 → R28-B として閉じる。
```

## 6. D105 dynamic verification

```text
D105 dynamic winner = UNOBSERVABLE
```

- Recovery-origin publish を 1 件も発生させられないため（§5）、
  async completion（onPublishCommitted → resolve）と sync completion（trySubmitImpl 後続）の
  winner 反転を動的に確認する手段がない。
- 既存観測にも **winner を表す値が存在しない**：
  `resolveRecoveryObligation` の `won` 戻り値は両 call site で破棄、
  `liveCount_`／terminal state は test 可視 getter なし、
  `liveLogicalRecoveryObligationCount()` は caller 0、Run M の Release build は
  `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` OFF のため recovery log も出ない。
- したがって **静的 PASS（R27 §3 の 9 check）を動的 PASS に置換しない**。
  R27 の静的結論は維持されるが、本 Step で動的裏付けは得られていない。

## 7. cmt / coord / seq / drp windows（Run M・同一 window 取得）

Δ（連続する pre/post 点の差）：

| window | reqΔ | queΔ | takeΔ | bldΔ | cmtΔ | coordΔ | drpΔ | seqΔ | 読み |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| p1-sc0（pre→post） | +3 | +3 | +2 | +1 | 0 | 0 | 0 | 0 | B2後・commit前（Case B 形） |
| gap1（p1post→p2pre） | +1 | +1 | +1 | +1 | +1 | **+2** | 0 | **+3** | publish 到達あり |
| p2-sc0（pre→post） | +2 | +2 | +1 | 0 | 0 | 0 | 0 | 0 | 窓内 publish なし |
| gap2（p2post→p3pre） | +2 | +2 | +2 | +3 | **+2** | **0** | 0 | **0** | **C-1 方向**（§9） |
| p3-sc1（pre→post） | +2 | +2 | +1 | 0 | 0 | 0 | 0 | 0 | 窓内 publish なし |

不変条件（R28 §7 の表に照合）：

```text
coord monotonic  : 3 → 3 → 5 → 5 → 5 → 5（非減少・reset なし）
coord >= cmt     : 3>=2, 3>=2, 5>=3, 5>=3, 5>=5, 5>=5（Main bucket ⊇ main-site rebuild）
seq  : 5 → 8（origin 非依存・publish 成功のみで前進）
drp  : 全点 0（commit 側 monotonicity reject のみを意味する・§11）
impossible 行（`>0 / <0`, `>0 / >0 / <0`）: なし（measurement error なし）
```

## 8. gap analysis

- **gap は必須**（R28 §8）。本 Run M でも gap1／gap2 に窓外 trailing event が出た。
  - gap1：cmt+1・coord+2・seq+3（窓内では seq 不変）→ publish は gap で進む（R23/R27 と同型）。
  - gap2：cmt+2・coord+0・seq+0（窓内 seq 不変＋gap でも不変）。
- gap1 の `coord+2 > cmt+1` は、Main bucket に main-site 以外（idle/bootstrap）が混在する
  差分として**そのまま保持**する（R28 §9：補正・差し引き禁止。「background=1」と断定しない）。
- window-boundary race は R23 §10 と同じく残る（post は瞬間値・trailing は gap に落ちる）。

## 9. C-1 / C-2 observability analysis

### 9.1 観測境界の定義（再現可能な形）

```text
Case C: cmtΔ > 0 かつ seqΔ = 0 の window に対し、
  coordΔ == 0  → C-1方向（main-site commit-enqueue 到達後に
                 obligationId==0 の Coordinator pop が観測されない）
  coordΔ >  0  → C-2方向（Main-origin pop は観測されたが publish/seq 未到達）
```

- 本 Run M で **C-1方向の観測例（gap2：cmt+2／coord+0／seq+0）** を取得した。
- C-2方向の観測例は本 Run M では未取得（`coordΔ>0 & seqΔ=0` の窓なし）。
- Case B 形（bld 到達・cmt 未到達）は p1-sc0 で再現（R23 と同型）。

### 9.2 限界（R28-B の根拠）

1. **per-task 対応不能**（R28 §12 R28-B の定義そのもの）：
   `coord` は Main bucket 集約であり、`cmt` のどの項目が `coord` のどの項目に対応するかを
   保証できない。よって `cmtΔ>0 & coordΔ>0 & seqΔ=0` を得ても、
   **「その cmt に対応する publish が C-2」とは確定しない**（aggregate evidence に留まる）。
2. **Main bucket 混在**：`coord` は idle/bootstrap publish を含む（§4 実測）。
   `coordΔ` を Main rebuild に帰属できない（R28 §9：補正禁止）。
3. **coordΔ==0 の多義性**：C-1（未到達）の他に、admission rejection（StaleGeneration／
   NotFinalized／Pressure／Shutdown／RejectedLowPriority）・deferred 滞留・
   executor failure・queue-full が同一観測（cmt+／coord0／seq0／drp0）を生む。
   既存観測では相互に区別できない（R26 §5.2 の indistinguishability set のまま）。
4. **drp の非包含**：`drpΔ==0` は「rejection なし」を意味しない（admission rejection は drp 非加算）。

### 9.3 結論

```text
- Case C を「C-1方向／C-2方向」に分ける window-level の観測境界は定義できた（§9.1・再現可能）。
  Run M で C-1方向の観測例を 1 件取得。
- ただし individual main-site publish の C-1/C-2 確定は不能（集約のみ・起源ラベルは pop で Main bucket に潰れる）。
- Recovery-origin は発生自体が不能のため、Recovery を挟む closure は未達。
→ R28-B（Observability insufficient）。
```

## 10. Limitations

```text
L1. Recovery-origin publish は既存 vehicle で発生不能（§5.2 の exhaustive 根拠）。
    新 counter／新 test 追加が禁止のため、本 Step では解消不能。
L2. D105 dynamic winner は観測不能（§6）。R27 静的 PASS は動的 PASS に置換しない。
L3. C-1/C-2 は window-level の方向判定まで。per-task 確定は generation linkage 禁止のため不能。
L4. coord の Main bucket は idle/bootstrap を含む（起源ラベルは R26-A の 2値止まり）。
L5. Run M は単一 episode・n=1。C-2方向の観測例は未取得。
L6. Release build は RUNTIME_DIAGNOSTICS OFF のため recovery 系 diagLog が出ない。
L7. gap の trailing event・window-boundary race は R23 §10 から不変。
```

## 11. R28 Gate

```text
R28-A： 非該当
  [NG] Recovery-origin publish を実走確認      → §5 発生不能
  [OK] origin transport が維持される           → 静的（R27）＋ source reconciliation（§2）
  [NG] Recovery pop が coord へ混入しない      → Recovery pop 自体が未発生（混入は構造的に否定できないが実測なし）
  [NG] D105 single-completion dynamic behavior → §6 UNOBSERVABLE
  [OK] Main baseline で coord/cmt 混在を再確認 → §4（coord>=cmt・coord=3 at start）
  [OK] cmt/coord/seq/drp を同一 window で取得  → §7
  [OK] C-1/C-2 の観測境界を再現可能な形で定義  → §9.1（Run M で C-1方向 1 例）
  [OK] gap による trailing event を分離        → §8
  [OK] drp の意味を逸脱していない              → §7／§9.2-4
  [OK] 新規 production instrumentation 0       → §1／§13
R28-B： ADOPTED
  Recovery-origin vehicle 不在＋per-task 対応不能により、C-1/C-2 の closure は
  aggregate evidence（C-1方向 1 例・C-2方向 0 例）に留まる。
  D105 dynamic winner = unobservable。
R28-C： 非該当
  Recovery id の 0 化・Recovery の coord 計上・CAS invariant 崩れ・deferred origin 変化は
  いずれも source 上発見されず（本 Step は測定のみ・source 差分 0）。
```

- R28 §13 の禁止事項遵守：coordinatorTakeCount_ 変更 0／新 counter 0／新 getter 0／
  Intent struct 変更 0／recoveryObligationId semantics 変更 0／deferred 経路変更 0／
  admission・retry・defer・drop reason counter 0／per-task generation linkage 0／
  F/R 本測定・P3-1-D・buzz/limiter/NUC 帰属 0。
- source は R27 vehicle のまま保持する（本 Step の差分は ConvoPeq.md 再生成と本ドキュメントのみ）。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。

### 次 Step 候補（R28-B の解消に必要なもの・本 Step では着手しない）

```text
- Recovery-origin を決定論的に発生させる vehicle（quarantine 誘発の test-only 手段）の設計 gate
  （R28 §13 により本 Step では追加しない）
- または D105 winner／per-task 対応を既存契約内で観測可能にする設計監査
```
