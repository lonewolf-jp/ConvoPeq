# D144 — P4 Selected Contract Implementation Plan / Proof Obligations (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only disproof-first verification of D143's I-HS. **Production source changes: 0. Test source changes: 0.**
**基準:** `ConvoPeq.md Generated: 2026-08-31 13:27:23`（D142 後・D143 と同一、ツリー無変更確認）。
**結論先出し: CONDITIONAL / CONTRACT REVISION — I-HS 単独は Case C を反証できない。修約 I-HS2（下記）で 7 判定基準すべてが証明可能。D145 は I-HS2 を対象とすべき。**

---

## D144-1 — linearization point 表（ordering と conflicting-access elimination の分離）

| 操作 | thread | linearization point | 備考 |
|---|---|---|---|
| RecoveryFailure 発生 | RebuildThread | event ring への push（CAS 予約） | adjudication 自体は行わない |
| Failure adjudication | CoordinatorLoop | event consume → delivery/counter/resolve（CL 内直列） | processIntent→redrive の前段で同一 tick 完結 |
| durable attach | CoordinatorLoop | **payload 書込 → CAS(NoAdmission→DurablePending, release)** | NoAdmission を acquire 観測した時だけ payload に触れる |
| durable take | RebuildThread | **CAS(DurablePending→Building, acquire)** 成功 = lease 取得 | 成功後にのみ payload を読む |
| durable settle(success) | RebuildThread | payload reset → CAS(Building→NoAdmission, release) | lease 保持者だけが reset する |
| durable settle(failure) | RebuildThread | CAS(Building→DurablePending, release) | payload 不変 |
| redrive attach | CoordinatorLoop | attach と同一（CAS publish） | |
| redrive repair | CoordinatorLoop | state acquire → oblId 読 → delivery 再同期（**payload 非書込**） | |
| shutdown discard | shutdown thread | **両スレッド join 後**（下記 D144-7 実測） | 単一スレッド化された reset |

**分離の明示**: 「state が atomic で release/acquire がある」は *ordering* を与えるが、*conflicting access の排除*は与えない。排除には「**payload を書く権利が排他的に誰にあるか**」（mutation exclusion）が別途必要 — これが Case C の論点。

## D144-2 — durable slot 9 field × ペア全列挙（現行コード = I-HS 適用後も同じ判定）

凡例: 書込者スレッド／読者スレッド、`×`=conflicting pair、判定は **I-HS（atomic state のみ）** と **I-HS2（修約）** で併記。

| field | 書込（場所・thread） | 読（場所・thread） | I-HS 判定 | I-HS2 判定 |
|---|---|---|---|---|
| `state` | CL 998/1159・Builder 1227/1261/1266・shutdown 1244 | CL 991/1158/1100・Builder 1214/1260/1243 | 全 transition を CAS 化すれば atomic 上で完結 → **SAFE** | SAFE |
| `recoveryObligationId` | CL 1006/1167・reset 1266/1244 | CL 992/1185/1101・Builder 1221 | **UB**: CL overwrite(1006) は Building 中にも走り得る（991 ガードは oblId 一致を通す）→ Builder take の oblId 読(1221) と read-write 競合 | SAFE（overwrite 禁止で書込は pre-publish/reset のみ、読は acquire 経由） |
| `buildSource` | CL 1001/1162・reset 1266/1244 | Builder 1222(take) | **UB**: CL overwrite × Builder take の read-write（~200B 構造体の途中書込） | SAFE |
| `handle`/`epoch`/`intentId`/`recoveryGeneration`/`pending`/`reservationOwned` | CL 1000-1006/1160-1167・reset | Builder take（1218-1225）/pending・reservationOwned は production 読 0 | **UB**（同上の overwrite 経路） | SAFE |
| （delivery は durable slot 外だが同表に含めると: W5 移動で単一書込者化 → SAFE） | | | | |

**要点**: 競合の発生源は単一 — **submit の same-oblId durable overwrite（cpp:991-908 の「同一 oblId なら state 値を問わず payload 上書き」）**。これを閉じれば全 field の conflicting access が消える。

## D144-3 — Case C 反証（I-HS 単独の失敗）と修約

### 反証（実コード経路・非 shutdown・生存スレッド 2 本）
```text
t0  O=Live, slot=DurablePending(O), payload=P1        （redrive/submit が attach 済み）
t1  Builder take(O): state acquire==DurablePending → payload P1 読開始 → CAS Building
t2  同一 {h,target} の再 submit（queue full）:
    cpp:991 ガード: state!=NoAdmission ∧ oblId==O → 通過（**Building でも通過する**）
    cpp:998-1008: payload := P2（buildSource/handle/... を上書）+ state:=DurablePending
t3  Builder は lease 中に payload が書き換わる（read-write / write-write 競合）
    + state 値が Builder の CAS 後に CL へ戻される（lease 剥奪の可視化）
```
→ **atomic state + release/acquire ≠ single mutation authority**（指示の命題が成立）。I-HS は Case C を除去できない。**CONTRACT REVISION 確定。**

### 修約 I-HS2（最小追加契約）
1. **state を atomic<uint8_t> 化し、全遷移を CAS 化**（take=CAS(DurablePending→Building)、settle=CAS、attach=CAS(NoAdmission→DurablePending)）。
2. **attach/初期書込は「NoAdmission を acquire 観測した時だけ」payload に触れる**（payload 書込 → CAS publish の順序規約）。
3. **same-oblId overwrite を禁止**: submit の durable 経路は `state==NoAdmission` のときのみ payload を書く。同一 oblId が既に slot に在る（DurablePending/Building）場合は**何も書かず true を返す**（表現は既存 durable が保持）。
   - 意味論的安全性: 同一 oblId ⇒ 同一 CoalesceIdentity ⇒ 5 semantic 値 + buildInput（target 構成要素）が一致。durable payload の「最新化」は recovery 再 build の意味を変えない（build 時に現在設定を再読する設計 — cpp:809-811）。**唯一の挙動差分**であり D145 テストで固定。
   - 副次効果: D136-C（Building 中 overwrite）と P5 の本命が本修約の定理になる（P5 は検証 Gate へ縮小）。
4. **reset は lease 保持者（Building の owner）または post-join の discard のみ**（現状維持、CAS 化で明示）。

修約後、payload への書込権限は「NoAdmission を観測した CL」か「Building を CAS 取得した Builder」か「join 後 shutdown thread」のいずれか排他 → **mutation exclusion 成立**。

## D144-4 — rearmRecoveryRetry の最終扱い: **Option A（Builder 側に残す）**

証明: rearm の唯一の production caller は Orchestrator:413（RejectedPressure）で、`submitPublishRequest` の実行 thread は RebuildThread（Commit.cpp:822 ← rebuildThreadLoop / processDeferredAdmission ← RebuildDispatch:914）。`state==Building` は take（RebuildThread）しか生成しない（修約 3 で CL の Building 中書込は消滅）。よって rearm の `state==Building ∧ oblId==O` 読 → settle(true) は**同一スレッドの lease 内操作**であり cross-thread conflict 不存在。Option B（イベント化）は Orchestrator が decision 返却前に再武装を必要とする同期 lease semantics を壊す → 棄却。
（settle(true) は修約 1 により CAS(Building→DurablePending) — 同一スレッド CAS で意味不変。）

## D144-5 — RecoveryFailure transport 仕様（確定）

- **専用 ring**: `MpscBoundedRing<RecoveryFailureEvent, 64>`（intentQueue_ 共有は避ける — Intent 4096 の backpressure/publish 経路と結合してしまう。専用 ring が証明が容易）。
- Event: `{ std::uint64_t obligationId; std::uint64_t seq; std::uint8_t siteId; }`（siteId は telemetry 判別用のみ、adjudication 意味論に非関与）。
- **Producer**: 失敗サイト 6 箇所のみ（全て RebuildThread）。**Consumer**: CoordinatorLoop のみ（runCoordinatorPhase、processIntent 内 or redrive 直前に drain → 同一 tick で adjudication→redrive→wake）。
- **6 サイト 1:1**: RebuildDispatch:1006/1033（P1 transport build/warmup 失敗）、:1091/1115（durable build/warmup 失敗）、Orchestrator:311（publish 失敗）/:401（RejectedPressure）。現行の直接呼び出しを push に置換（1 失敗観測 = 1 push）。
- **overflow / 非 drop**: primary 64 + **fallback ring 64**（quarantineFallbackQueue_ の前例 cpp:1352）。**構造的 bound**: 未処理イベント数 ≤ 4×L_max = 4×32 = 128（obligation あたり枯渇 K=4 まで、再付着には adjudication が必要）。64+64=128 が上界と一致 → **overflow は構成上発生し得ない**（D145 で再導出確認）。両 ring 満杯は到達不能だが、防御として drop counter + HealthEvent 昇格（silent loss 禁止 INV-5）を残す。
- **telemetry**: `recoveryFailureEventCount_`（post）、`recoveryFailureAdjudicatedCount_`（consume）、既存 recoveryRetry*Count_ と突合可能にする。
- **shutdown**: CL join → Builder join → discard 順序（D144-7 実測）で、残留イベントは「obligation 側は ShutdownDiscarded 済み」→ adjudication は Live チェックで no-op。drain-and-count で回収（silent loss なし）。

## D144-6 — exactly-once failure posting proof（サイト別）

| サイト | 1 失敗=1 post の根拠 | duplicate/stale/post-terminal |
|---|---|---|
| :1006 build 失敗 | pop 済み intent 1 件↔失敗 1 回、continue 前に 1 push | 再 pop は別 intent（別失敗観測=正当な +1、現行意味論と同じ） |
| :1033 warmup 失敗 | 同上（DSP 破棄後 1 push） | 同上 |
| :1091/:1115 durable 失敗 | take 1 件↔settle(true)+1 push（settle と post は同一 lease 内・順序固定） | spin break 後も slot は DurablePending、再 take は再失敗時のみ post |
| Orch :311 publish 失敗 | trySubmitImpl 1 回↔1 push | destroyRolledBackDSP と独立、decision 分類非依存（現行コメント 306-310 と同義） |
| Orch :401 RejectedPressure | decision 1 回↔1 push + rearm（Builder 側・D144-4） | rearm は delivery を触らない（settle のみ）→ counter と直交 |
- adjudication 側: Live チェック（現 cpp:1073 相当）で stale/unknown/post-terminal は no-op。K 到達の 4 回目は他と区別不要（counter 論理は現行 markTransientFailure をスレッド移設のみ、意味不変 — R17-4/R20-3/R21 契約維持）。
- **duplicate 懸念の潰し方**: 同一失敗の二重 post はサイト構造上ない（各サイトは 1 回の失敗観測で 1 回だけ通る）。異なる失敗の多重 post は現行の直接呼び出しでも成立する挙動で、exactly-once とは「1 失敗観測=1 加算」の対応関係 — 維持される。

## D144-7 — shutdown / discard の HB（実測に基づく）

停止順序（両経路で確認）:
```
AudioEngine.CtorDtor.cpp:126-127 / ReleaseResources.cpp:201-202:
    shutdownCoordinatorLoop()   // CoordinatorLoop join（attach/overwrite/repair/adjudication 停止）
    stopRebuildThread()         // exit flag → notify → rebuildThread.join()（take/settle/rearm 停止）
        → discardRecoveryRequestsOnShutdown()（RebuildDispatch:812）
        → discardPendingRecoveryAdmission()（:818、join 後）
```
- discard 実行時点では **CL・Builder とも join 済み**（コメント 806-808 が明示、コード順序で実証）。thread join は完了スレッドの全書込に HB を与える（[thread.sync.alg]）→ discard の payload reset は**単一スレッド操作**であり、`CL attach × discard`、`Builder settle × discard` の競合は**全順序で存在しない**（「通常 concurrent でない」仮定ではなく、join の HB と呼び出し順序の実測による）。
- 残存する理論競合（Builder join 前に take 済み Building のまま join されるケース）: join 完了=Builder の最終動作まで HB → discard の reset はその後。OK。

## D144-8 — P3 repair 互換の再証明（I-HS2 下）

- repair の 2 段読: `state.load(acquire)` → 非 NoAdmission → `oblId 読`。修約により oblId の書込は (a) CL の pre-publish（NoAdmission 観測時）または (b) lease 内 reset のみ。Builder の reset は state=NoAdmission の release に sequenced-before なので、**acquire した state が非 NoAdmission なら oblId 読はその state と対になる古い値**（reset 済みなら state は NoAdmission が見える）。D140 §3 の stale strand は消滅。
- same-holder repair / different-holder fallback / repeated redrive / failure→redrive / wake latch: すべて不変（repair は payload 非書込、delivery+latch のみ）。T-P3-1..6 の期待値は修約後も成立（failure→redrive 循環で durable 単一表現が保たれることは修約 3 でさらに強化）。

## D144-9 — 専用テスト仕様（**実装しない**、D145 の受け入れ条件として固定）

**HB/race（実スレッド統合、AudioEngineHarness 層 or 専用ハーネス）:**
- H-1 CL attach → Builder take: take が読む payload は attach 前書込と一致（撕裂なし）
- H-2 Builder settle(false) → CL repair/attach: NoAdmission 観測後の attach が旧 payload を読まない
- H-3 Builder failure → CL adjudication → 同一 tick redrive/wake の連鎖
- H-4 shutdown × durable: join 順序の回帰（discard が単一スレッドで完結）
**event semantics:** E-1 exactly-once（1 失敗=1 加算）、E-2 duplicate 非発生（サイト構造）、E-3 stale（terminal 後イベント no-op）、E-4 unknown oblId no-op、E-5 ring full→fallback、E-6 fallback 到達不能性の bound 検証（128 上限の組成的テスト）、E-7 shutdown 残留イベント回収+counter 整合
**retry:** R-1..R-3 failure 1/2/3 で Live 維持・delivery=None・同一 tick 再付着、R-4 K=4 terminal、R-5 failure→redrive 循環で表現 ≤1、R-6 stranded Transport 修復、R-7 terminal 後失敗無視
**compatibility:** 既存 40 テスト + R18 群 + RLOE C1-C16 + G-4.3/G-4.4-P2/P3 群 + **修約 3 の挙動差分テスト**（same-oblId 再 submit 時に payload 非書込・true 返却・transport 再 push 経路の維持）

## 最終判定

```text
GO criteria 評価（I-HS 単独 → I-HS2）:
  1. delivery ownership            I-HS: ✓（イベント化）      I-HS2: ✓
  2. durable mutation exclusion    I-HS: ✗（Case C 反証）     I-HS2: ✓（修約 1-4）
  3. payload snapshot consistency  I-HS: ✗                    I-HS2: ✓
  4. HB                            I-HS: 部分                 I-HS2: ✓（CAS+release/acquire+join）
  5. exactly-once adjudication     I-HS: ✓（条件付）          I-HS2: ✓（D144-6）
  6. shutdown safety               I-HS: ✓（join 実測）       I-HS2: ✓
  7. P3 repair compatibility       I-HS: ✗（strand 残存）     I-HS2: ✓（D144-8）

Verdict: CONDITIONAL / CONTRACT REVISION
  Selected contract（修約後）: I-HS2 =
    I（RecoveryFailure を専用 MPSC ring 64+64 で CL に転送、adjudication 単一権限化）
    + atomic state（全遷移 CAS）
    + payload 書込権限規約（NoAdmission 観測時のみ／lease 内 reset のみ）
    + same-oblId overwrite 禁止（no-op true 返却）
    + rearm は Builder 側維持（Option A、同一スレッド lease 証明）

NO-GO 項目の残存: なし（lost event=構造的 bound 128 で非 drop、duplicate=サイト 1:1、
  plain payload 競合=修約 3 で排除、shutdown=join HB、P3 stale=消滅、lease=不変、RT/ISR=非接触）

→ D145（P4 Implementation）は I-HS2 を対象に進めてよい。ただし D145 の最初の成果物は
   修約 3 の挙動差分（same-oblId 再 submit）の既存契約適合証明とすること。
```

**STOP — 実装 0。P5/P6 非着手。D145 の指示を待つ。**
