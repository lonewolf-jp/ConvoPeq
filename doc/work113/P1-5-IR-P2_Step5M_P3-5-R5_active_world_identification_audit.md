# P1-5-IR-P2 — Step 5-M / P3-5-R5: Active World Identification Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R5）
- **種別**: source-only／read-only。実行・build・変更なし。
- **目的**: 既存の production API／test harness だけで、測定時の active world を
  target snapshot と照合できるかを確定する（R5-A/B/C）。
  「active world が target だった」とは推定しない。
- **前提**: R4 維持（測定値有効／publish は A3 未証明／active 未同定／
  stale Possible／Δ未帰属）。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
F vehicle＋R1 diagnostic     保持（f4723815…／a00d140d…）
production/CMake/JUCE        0 diff（P1 TU以外）
```

## 2. Publication evidence API audit

### getLastCommittedPublicationSequence（:1744-1747・public）

- `lastCommittedPublicationSequence_` の acquire-read。
  書込みは `onRuntimePublishedNonRt` のみ（Commit.cpp:402：
  `lastCommittedPublicationSequence_ = world.publication.sequenceId`）。
- 意味：**実際に commit された RuntimeWorld の publication sequence**
  （requested rebuild の sequence ではない）。
- `seqBefore == seqAfter`＝window 内 commit ゼロの強い証拠（R1 の使い方・維持）。
  `seqAfter > seqBefore`＝何らかの commit 発生（内容不問）。
  target一致には snapshot 照合が別途要る（§4・§5）。

### getRuntimeLifecycleDiagnostics（:1732-1742・public）

- `{runtimePublishCount, runtimeRetireCount, runtimeReclaimCount,
  lastCommittedRuntimeGeneration_, lastCommittedPublicationSequence_,
  lastDroppedGeneration_}`。
- 意味：commit 観測の累積カウンタ＋最終commit mirrors。
  per-case 前後差分で当該区間の publish／retire／drop 有無が分かる。
  既存 getter のため test-only から直接読める（新規APIなし）。

## 3. Rebuild evidence API audit

### getRebuildDispatchDiagnostics（:2057-2069・public）

8 counters（宣言順＝`requestCount／queuedCount／blockedPendingDuplicateCount／
blockedRecentDuplicateCount／runtimeQueueFullCount／drainedCommandCount／
matchedRuntimeCommandCount／taskSnapshotFallbackCount`）は
プロセス累積値である。per-case 前後差分により当該区間の
request→queue→duplicate-suppress の到達点が分かる。
ただし **queued／drained は publish evidence ではない**
（queue は worker 投入、drained は command 消費。commit とは別段・指示どおり）。

## 4. Backpressure API audit

### getRuntimeBackpressureTelemetry（:1762-1794・public）

- `rebuildBacklog` と `publicationBacklog` は別物（再確認・混同禁止）。
  前者は rebuild queue 側、後者は `runtimePublicationBridge_` 側（:3644-3647）。
- `publicationRejectCount／retirePressureLevel／saturationEnter-Exit` 等は
  commit 拒否・retire 圧の区間差分として読める（既存 getter）。
-  pressure／health 自体の直接読値は本 API 群にない（U1 の直接観測は不可のまま）。

## 5. RuntimeWorld / RuntimeReadHandle audit（中心）

### 問い1：test-only から現在の RuntimeWorld を取得できるか

**YES**（production 変更なし）。`makeRuntimeReadHandle`（:3308）・
`getRuntimeWorldFromReadHandle`（:3405）・`getRuntimeSnapshotFromReadHandle`（:3464）は
public 領域（2259 public〜3729 private）にあり、test-only（`h.engine()` 経由）から呼べる。
既存使用例（CtorDtor.cpp:75-77）と同じ形で Message channel を使う：
`RuntimeReaderContext{ messageThreadRcuReader, ObserveChannel::Message }`。
ただし `messageThreadRcuReader` 自体は private（:4940）のため、
test-only は既存 public helper 経由に限る（＝上記3関数のみ。新規 getter 追加はしない）。

使用制約（次gate設計への申送り）：read handle は RCU epoch を pin する。
capture 全区間の保持は retire／reclaim を停滞させ pressure を誘発しうる。
よって capture 前後の**瞬間 read に限定**し、区間保持しないこと。

### 問い2：target identification に十分な field か

**YES**（IRファイル名を除く）。RuntimeState（＝RuntimePublishWorld・:334）の
Authoritative／Derived field 群：

```text
generation／worldId／publication.sequenceId（§6の三相関キー）
routing（convBypassed・eqBypassed・processingOrder）
automation（softClipEnabled・saturationAmount・headroom／makeup／trim gains）
timing／overlap（fadeTimeSec）／topology／engine／execution／graph
dspProjection{irLoaded・irFinalized・structuralHash・oversamplingFactor・sampleRate}
```

- os 同定：`dspProjection.oversamplingFactor`。
- conv／EQ／softClip／sat／staging 同定：routing＋automation。
- IR assay：world にファイル名はない。vehicle 単一 IR＋既存 `[IR_*]` geometry ログ
  （`[IR_TAIL_GEOM]`／`F_scale`／`[L0_WRITE]`）との複合キーで足りる。
- SealedObject は mutation 規律であり const-read を妨げない。
  audio thread が毎 callback 同一 read を行う設計である。

### 問い3：existing public API only で不可能か

**可能である**（§5 Q1/Q2）。Q3 は該当なし。

## 6. generation / worldId / publicationSequence correlation

4者は別 numbering であり等価を仮定しない（指示どおり）：

```text
rebuildRequestGeneration： request counter（task.generation＝++値・:676）
RuntimeWorld::generation： world identity（graph-generation 系予約値）
publicationSequence：     commit 順序（reserve時 fetch-add :3621-3623。
                          reserve≠commit のため未commit gap がありうる）
worldId：                 build 毎の一意値（runtimeWorldIdGenerator_.next() :3620）
```

相関 key は `(sequenceId, generation, worldId)` の三つ組＋snapshot field とし、
単一 counter の一致で同一視しない。commit 観測の正本は
`onRuntimePublishedNonRt` の代入対（:401-402）である。

## 7. A3 target-pair evidence table（Known／Unknown／Inferred 明示）

### am=-20（A3 L235／Forward L235／F-vehicle pair1）

| field | A3 | Forward | F-vehicle |
| --- | --- | --- | --- |
| requested snapshot | Known（g0/os1/convBypass=false/sat=1.0/sc0+sc1/EQ identity） | Known（同左） | Known（同左） |
| seqBefore | Unknown（行に sequence なし） | Unknown（同左） | Known（5・R1 diag） |
| seqAfter | Unknown | Unknown | Known（5・flat・gap付き） |
| committed generation | Unknown | Unknown | Unknown（未読） |
| active snapshot | Unknown | Unknown | Unknown（未読） |
| target一致性 | Unknown | Unknown | Unknown |
| measurement | Known（-14.5035・3 run一致） | Known（同左） | （pair無効・測定なし） |

### am=-6（A3 L356／Forward L356／F-vehicle pair2）

| field | A3 | Forward | F-vehicle |
| --- | --- | --- | --- |
| requested snapshot | Known（ampDbのみ-6） | Known | Known |
| seqBefore／seqAfter | Unknown（waitなし） | Unknown | Unknown（継承分岐・waitなし） |
| committed generation／active snapshot／target一致性 | Unknown | Unknown | Unknown |
| measurement | Known（-13.0276・3 run一致） | Known | Known（-13.0276・sc0/sc1一致） |

補完なし。Inferred 行は置かない（merge継承等の機構は §5 Q2 の将来観測で検証する）。

## 8. Existing-only feasibility

- publish 確認：既存 sequence 差分で可（R1 実証済み）。
- active snapshot 読出し：public read-handle API で可（§5・要build検証は次gate）。
- target identity 照合：world field＋dspProjection＋既存 IR ログで可。
- rebuild 経路内訳：dispatch diagnostics 差分で可（publish 証明には使わない）。
- 不足：pressure／health 直接読値（U1 直接観測は不可のまま。拒否计数の差分で代替）。
- RCU pin 制約（§5）を守る限り、production 変更は不要である。

## 9. R5-A/B/C classification

```text
R5-A（Existing-only sufficient）: ADOPTED
  publish＋active snapshot＋target identity が既存 public API のみで同定可能（§5・§8）。
  次は R5 execution vehicle の test-only 実装監査へ。
R5-B（sequence 可・identity 不可）: REJECTED（identity 可のため）
R5-C（観測経路自体不足）: REJECTED（経路ありのため）
```

## 10. Next-gate recommendation

1. 次は **R5 execution vehicle の test-only 実装監査**（read-only・実装なし）。
   観測点：各 pass の capture 前後での sequence 差分＋scoped read-handle snapshot
   （routing／automation／dspProjection／generation／sequenceId／worldId）＋
   dispatch／lifecycle／backpressure 差分。既存 `[P1CHAR]` 行への field 追加は
   test-log 変更として次gate承認事項とする（新規 prefix・production logger なし）。
2. RCU pin は瞬間 read に限定（§5 制約）。
3. 単一 counter（特に rebuild==world の等価）での同一視を禁止（§6）。
4. sequence 前進のみでの R4-C 閉鎖は禁止（§4 の3段階規律を維持）。
5. R vehicle 実行・F/R 比較・P3-1-D・limiter 計装は保留維持。
   R4 境界（測定値有効／publish 未証明／active 未同定／stale Possible／Δ未帰属）を維持する。
