# D135-8/9 Gate G-4.4 — Phase-I Durable Admission Audit (read-only)

**Date:** 2026-08-31 00:0x–00:5x (+09:00)
**Type:** read-only structural / concurrency / ownership audit. Production source changes: **0**. Test source changes: **0**.
**Inputs:** working tree @ HEAD `5f6f48c` + uncommitted G-4.1…G-4.3-T deltas（無変更確認済み: `git status` = cpp/h/tests の 3 ファイルのみ、ConvoPeq.md `Generated: 2026-08-30 23:47:13` と一致）
**Build evidence:** `evidence/g44_ctest.log` — Debug full build → CTest `100% tests passed out of 40`（DBG_CTEST_EXIT=0）/ Release full build → CTest `100% tests passed out of 40`（REL_CTEST_EXIT=0）
**Verdict: CONDITIONAL**（Safety 成立 / Liveness 残存事項あり — 下記 D10・観察 2-4）

---

## 中心命題の証明

> **logical obligation の identity / ownership は admission table が保持し、durable slot はその delivery representation に過ぎない**

### D1 — Durable slot は logical obligation を新規生成しない → 成立
追跡対象: `pendingRecoveryAdmission_`（h:952-968、plain struct + atomic predicate `recoveryAdmissionPending_` h:969）/ `recoveryObligationId` / `RecoveryAdmissionTable`（h:376-453）/ `liveCount_`（h:450）/ `ObligationDeliveryState`（h:325-329）/ `reservationOwned`（h:962）。

- `RecoveryAdmissionTable::tryInsert`（唯一の +1、h:392-413）の呼び出し箇所は **cpp:946 の 1 箇所のみ**（submitRecoveryRequest の NEW 分岐）。durable 経路（submit 側 cpp:998-1008 / redrive 側 cpp:1158-1170）は tryInsert を呼ばない。
- cpp 全体で `liveCount_` への直接触达は 0（コメント cpp:555 の言及のみ）。+1/−1 は table 内部（h:408/430）に閉じている。
- durable 書込は常に**既存 oblId の写像**: submit 側は `recoveryObligationId = oblId`（cpp:1006、oblId は COALESCE 再利用 or tryInsert 済みの値）、redrive 側は `recoveryObligationId = obligationId`（cpp:1167）＋「admit THIS obligation, not a new one」（cpp:1157）＋ table から id/buildSource/generation を復元（cpp:1154-1156）。
- 構造: `NEW obligation → tryInsert → liveCount+1 → Transport/Durable/Deferred` のうち durable は最後の**住処の付与**であり、生成段（tryInsert）とは別経路。redrive は ΔL=0 を明文化（cpp:1108-1109、C15 テストで実証済み）。

**∎ durable admission は logical obligation を生成しない。**

### D11 — coalesce（admission 問題）と durable fallback（delivery residency 問題）の分離 → 成立
- coalesce 判定は `findByKey(cid)` + `slot.state` CAS Live→Live（cpp:926-933）— **table の state/identity のみ**を参照。durable slot を読まない。
- durable 書込は coalesce 決定**後**の delivery 段（cpp:976-1009）で行われ、`pendingRecoveryAdmission_` は CoalesceIdentity を一切再構築しない（grep: durable 経路に findByKey/CoalesceIdentity 構築なし）。
- durable slot が tryInsert を経由して obligation を追加生成する経路なし（D1）。coalesce を bypass して logical obligation が増えることもなし（+1 サイト単一性）。

**∎ 層分離はコード上で成立。**

---

## D2 — distinct obligation の blind overwrite 不在 → 成立

ガード（cpp:991-996）:
```cpp
if (pendingRecoveryAdmission_.state != PendingRecoveryAdmission::State::NoAdmission
    && pendingRecoveryAdmission_.recoveryObligationId != oblId) {
    recoveryAdmissions_.slot(slotIdx).delivery = ObligationDeliveryState::None;   // deferred
    convo::fetchAddAtomic(recoveryRetryDeferredCount_, ...);
    return true;   // obligation stays Live; delivery re-driven when the occupied slot frees
}
```
- **distinctness は `recoveryObligationId`（obligationId）で保護**されており、handle 比較ではない（handle は同一でも target 違いなら別 oblId → 保護される。逆も同様）。
- 期待構造との対応: durable empty → admit A（cpp:991 条件不成立 → 998 で書込）/ durable A + incoming A → 上書き（同一 oblId のみ、D3）/ durable A + incoming B → **上書きしない**、B は Live・delivery=None・`recoveryRetryDeferredCount_`+1、後続 redrive（毎 tick、Threading.cpp:270）で再試行。
- redrive 側はさらに保守的: durable が**非 NoAdmission なら oblId 照合なしに** transport へフォールバック（cpp:1158/1173-1182）— durable の中身が誰であろうと上書きしない。
- I4 backpressure contract（既存 obligation の evict 禁止・blocked admission の park 可）と整合: どの分岐も既存 durable 保持者の delivery を剥奪しない。

## D3 — 同一 oblId 上書きの正当性 → 成立（1 窓を除く、観察 3 参照）

上書き可能フィールド（cpp:998-1007）: state/pending/recoveryGeneration/buildSource/reservationOwned/handle/epoch/intentId/recoveryObligationId — すべて**同一 oblId のときだけ**到達（cpp:991 ガード）。

- **buildSource 更新の正当性（D18.8 照合）**: 同一 oblId に到達した submit は `findByKey(cid)` 一致 = **CoalesceIdentity {handle, SemanticRecoveryTarget} が一致**（5 semantic 値 + handle 同一）。canonical identity は G-4.3 で固定済みで、durable layer は別 semantics を発明していない（durable 書込は cid を再構築しない — D11）。同一 semantic target に対して snapshot-level metadata（sampleRate 等、target 非関与フィールド）だけ進んだ値へ更新するのは「同一 recovery 意味論の最新ペイロード」の写像であり、D18.8 の「同一 target → coalesce」と矛盾しない。Recovery の build 自体は build 時に現在設定を再読する設計（cpp:809-811: IR 実体は transferIRStateFrom で現在値取得）。
- recoveryGeneration: cpp:1000 は `intent.recoveryGeneration` — COALESCE 時は table の不変 ordinal（cpp:974 経由）なので再生成なし。
- handle/epoch/intentId: handle は identity 一致により不変値の再書込。epoch は emit 時の最新へ更新（FIFO/epoch 検証用メタデータ）。intentId は診断シーケンスの更新（durable slot 側の診断値であり、table slot の intentId は coalesce で更新されない — G-4.3-RF 不変）。
- reservationOwned=true: 単一スロット自体が予約（INV-X1-5「1 admission = at most 1 reservation」h:943）— 再書込は冪等で二重予約にならない（pendingIntentCount_ は durable 経路で増減しない: cpp:976 の +1 は push 試行前、push 失敗時 cpp:983 で即 −1）。
- **例外窓（観察 3）**: 上書きガードは sub-state を見ないため、Builder が lease 保持中（Building）の同一 oblId へ上書きし得る。詳細は下記。

## D4 — Reservation conservation → 成立

3 台帳の分離（実コード再構成）:
- **logical**: `liveCount_`（table 内部、+1 h:408 / −1 h:430 のみ）。
- **transport residency**: `pendingIntentCount_` — reservation-before-push（cpp:976/1174 fetchAdd → push → 失敗時 cpp:983/1179 fetchSub）、pop 時 cpp:1261 fetchSub。
- **durable residency**: `pendingRecoveryAdmission_` スロット占有（reservationOwned）+ `recoveryAdmissionPending_`。

invariant「1 logical obligation ⇒ active delivery representation ≤ 1」:
- `ObligationDeliveryState` は単一 enum（None/Transport/Durable — 排他）。書込は全 site CoordinatorLoop 単独（h:324「CoordinatorLoop-only field — non-atomic by design」; 書込 site cpp:978/993/1008/1169/1176/1078）。
- 二重取得: transport push 成功で Transport、durable 占有で Durable — 同一 obligation が両方を同時に示す書込順序は通常経路に存在しない。**ただし transient-failure 経路で一時的二重住処窓あり（観察 2）**。
- distinct 上書きによる喪失: D2 ガードで否定。
- retry 増殖: settle(true)（cpp:1241-1244）は同一スロットの state 遷移のみ、markTransientFailure（cpp:1066-1092）は delivery=None + counter+1 のみで tryInsert しない。
- terminal 取り残し: resolve() は durable slot を触らないが、terminal 化された obligation の durable 表現は Builder に**一度消費されて消える**（take は state 無関係に DurablePending を拾い、settle(false)/discard で解放）。liveCount_ は二重減算されない（resolve 冪等、h:418-419）。shutdown では discardPendingRecoveryAdmission（cpp:1222-1230、唯一の production 呼び出し = RebuildDispatch:818 の shutdown drain）がスロットを確実に解放。

## D5 — Durable lease lifecycle → 成立

```
DurablePending --take(cpp:1193-1210)--> Building --settle(true)(cpp:1241-42)--> DurablePending
                                            |
                                            +--settle(false)(cpp:1246-47)--> NoAdmission (+ predicate false)
任意状態 --discardPendingRecoveryAdmission(cpp:1222-30, shutdown 専用)--> NoAdmission (+ discard count)
```
- take は**クリアしない**（lease — cpp:1207-1208、二十六次レビュー必須修正1）。obligationId/recoveryGeneration/buildSource をスロットから復元して返す（cpp:1198-1206）→ ownership 消滅なし。
- build failure（RebuildDispatch:1070-1084 / 1091-1108）: settle(true) → DurablePending 復帰 + markTransientFailure（obligation 側 counter）。durable obligation は失われない。Builder-local spin 上限 `kMaxRecoveryConsecutiveFailures=4`（RebuildDispatch:1056）到達時は break して次サイクル委譲 — durable は DurablePending のまま残る（RebuildDispatch:1051-1055 コメントと実コード一致）。
- retry は `liveCount_` を変更しない（settle/markTransientFailure/redrive のいずれも resolve/tryInsert を呼ばない。exhaustion 時のみ resolve(Failed) — cpp:1085-1087、唯一の公認 Failed 経路）。
- 最終解放は success（settle(false) RebuildDispatch:1122）/ invalid payload discard（RECOVERY-6、RebuildDispatch:1062-1065）/ shutdown discard のみ。
- `rearmRecoveryRetry`（cpp:1096-1103）: Building かつ同一 oblId のときだけ settle(true) — QueuePressure retry の guarded re-arm（Orchestrator:411-413、RejectedPressure 分岐）。distinct obligation の上書きなし。

## D6 — Producer / Consumer 境界 → 成立（形式的注記 1）

- **Producer（attach/overwrite 書込）= CoordinatorLoop 単独**: submitRecoveryRequest（cpp:998-1008）と redriveDeferredRecovery（cpp:1158-1170）は同一スレッド（ISRCoordinatorLoop.cpp:39 → runCoordinatorPhase、1ms fallback tick）。QuarantineIntentHandler 経由の submit も CoordinatorLoop 上（cpp:812-815 の単一 Producer 不変条件）。
- **Consumer（state 遷移書込）= RebuildThread 単独**: take/settle（RebuildDispatch:1058-1122）と rearmRecoveryRetry（Orchestrator:413 ← submitPublishRequest の RejectedPressure — RebuildThread 上）。
- 同一スレッドからの二重 mutation: なし（producer 側は DurablePending への attach のみ、consumer 側は Building 起点の遷移のみ — 交差は観察 3 の窓に限る）。
- **publication ordering**: submit 経路は durable payload 書込（CoordinatorLoop）→ `recoveryAdmissionPending_` release（cpp:1007）→ AudioEngine.h:4500-4504 で `rebuildMutex` lock 下 `recoveryPending=true` + notify → Builder の CV wait が同一 mutex を acquire → **happens-before 成立**。redrive 経路の書込も同一 CoordinatorLoop の program order にあり、後続の submit 経路 mutex 経由 wake で可視化される（同一スレッドの全先行書込が unlock で発行される）。`hasPendingRecoveryAdmission()`（cpp:1212-1216）は predicate の acquire 読み + state 確認。
- 形式的注記: `take()` は `recoveryAdmissionPending_` を acquire せずに plain `state` を読む（RebuildDispatch:1058）。実際には wake 経路の mutex エッジが全 producer 書込を発行するため x86-TSO で問題顕在化せず、CTest 40/40・運用観察でも症状なし。設計文書化または take 前の predicate ゲートが望ましい（G-4.4 実装時の検討事項、今回は変更しない）。

## D7 — Terminal race（resurrection 否定）→ 成立

interleaving 列挙（A = Live・DurablePending、Builder 消費中、terminal 化、redrive）:
1. **terminalize(A) → redrive(A)**: redrive は `state==Live` を acquire 読みで確認（cpp:1114/1140-1141）→ terminal 化済み A は候補にならない。**resurrection 経路なし**。
2. **Builder 消費中（Building）に terminalize(A)**: resolve() の CAS Live→terminal（h:428）は state atomic のみを変更。durable slot の Building は Builder の settle で解決（success→settle(false) で clear、failure→settle(true）で DurablePending 復帰）。復帰した場合、Builder は terminal 化した A のペイロードを再 take して build/publish を試みるが、`resolveRecoveryObligation(A, Published)` は CAS 失敗で **no-op（二重 −1 なし、h:418-419 冪等性）**。liveCount_ は不変。A の slot が別 obligation C に再利用されても id モノトニック（nextId_）で ABA 不成立（h:416-418、C9 テスト実証済み）。
3. **terminal 化後に durable が A を保持したまま B/C が deferred**: durable busy（state != NoAdmission）なので redrive は transport フォールバック（cpp:1173）— B/C の剥奪なし。A の durable 表現は Builder に一度消費されて消える（上記 2）。
4. **shutdown**: discardPendingRecoveryAdmission（RebuildDispatch:818、Builder 停止後）+ 各 obligation の ShutdownDiscarded resolve（cpp:1036/1045-1047）— 台帳とスロットが独立に閉じる。

**∎ terminalized obligation は durable slot に復活しない。所有保存（liveCount_）は全 interleaving で維持。**

## D8 — Redrive × durable slot 相互作用 → 期待構造と一致（1 窓は未検証、観察 2）

`redriveDeferredRecovery`（cpp:1126-1183）:
```
Live + delivery=None
  → durable free（NoAdmission）        → Durable（cpp:1158-1170、delivery=Durable）
  → durable busy（同一 oblId でも異 oblId でも）→ transport 試行（cpp:1174-1178）
       push 成功 → Transport / 失敗 → delivery=None 据置・ΔL=0・recoveryRetryRedriveFailureCount_+1（cpp:1179-1182）
```
- 既存 regression contract との対応: **C11**（deferred→durable 復帰）/ **C12**（redrive 冪等・二重送信なし）/ **C13**（durable busy→transport）/ **C14**（both busy→deferred 維持 ΔL=0）/ **C15**（liveCount 不変）/ **C16**（deferred 同一 key 再 submit→coalesce）— 全登録・両構成 PASS（本監査 CTest）。
- 指示の期待構造「durable busy with same oblId → no duplicate」との差分: 現行は same oblId-busy でも transport を試行する。delivery=None に**なり得る**唯一の durable 保持中経路は markTransientFailure（cpp:1078）で、このとき durable は同一 oblId を保持したまま（settle(true) 済み）→ transport 付与で**一時的二重住処**になり得る（観察 2）。C11-C16 はこの窓をカバーしていない。

## D9 — 容量分離 → 成立（誤推論なし）

- `Q_max = 256`: `kRecoveryIntentQueueCapacity = 256`（h:927、LockFreeRingBuffer）。
- `L_residency_max = 257`: I4_DESIGN_CONTRACT.md:943/984 — transport queue（256）+ durable slot（1）の**物理的 residency**。コード上の単一定数ではなく合成値として整合。
- `L_logical_max = 32`: `kMaxLogicalRecoveryObligations = 32`（h:363、INV-CAP-7）— table 独立、queue/durable と別台帳（pendingIntentCount_ 256 と liveCount_ 32 の混同なしは Gate F-4 で既確認）。
- 「durable slot が 1 個だから logical obligation が 1 個しか存在できない」は**成立しない**: table は 32 スロット独立（h:449）、durable を持てない obligation は delivery=None で park し毎 tick redrive で再試行される（C14/C15 が L=32 下での deferred 共存を実証）。episode decomposition は Phase-II deferred のまま（D12）。

## D10 — Liveness / stalled obligation → **Safety 成立 / Liveness 残存（修正しない）**

**Safety（成立）**: durable 保持 obligation は失われない（INV-X1-2、lease、D2 ガード）。stalled しても liveCount_・reservation・identity は不変。shutdown で discard カウント付き回収（silent loss なし）。

**Liveness（残存リスク — 本 Gate で修正禁止、以下を確定）**:
1. **redrive 経路に Builder wake が配線されていない**: `runCoordinatorPhase` の毎 tick redrive（Threading.cpp:266-270）は durable/transport 住処を付与するが、`recoveryPending` set + notify を行わない。wake 配線は submit 経路のみ（AudioEngine.h:4481-4483/4500-4504、RebuildDispatch:857 predicate）。したがって **Builder がアイドルで、以後無関係な wake（新 recovery submit / rebuild task / publishRetryReady）が来ない場合、redrive が付着させた表現は次の wake まで未消費で滞留し得る**。影響: quarantined DSP の復旧 build が遅延（音声は除外構成で継続 — 安全側の劣化）。システムが活動的なら wake は頻発するため、実運用では遅延として顕在化しにくい。
2. Coordinator tick 自体は event-driven + 1ms fallback（ISRCoordinatorLoop.cpp:41-45）で保証 — redrive 機会の欠如はなし。
3. dedicated retry timer は**存在しない**（grep 確認: durable 専用の再試行タイマーなし。再試行は tick redrive + Builder while ループ + wake predicate の組合せ）。
4. Builder settle 後の wake: while ループ内で即再 take するため同一サイクル内は自走。break（spin 上限）後は「次回 rebuild wake / 新規 Recovery で再処理」（RebuildDispatch:1053-1054）— 1. と同じ wake 依存。

**Verdict 分離**: Safety = PASS / Liveness = 上記 1 の配線ギャップ（G-4.4 実装設計で「redrive 付着時の wake」または等価機構を検討すべき対象。本監査では変更 0）。

---

## 観察（非ブロッキング・今回未変更）

1. **コメントドリフト（h:960-961）**: 「rebuildRequestGeneration（coalesce 判定用）」「latest（coalesce で更新）」— 現実は recoveryGeneration は専用 ordinal で coalesce 判定非使用、buildSource は coalesce 自体ではなく同一 oblId 再 submit の durable 経路で更新。G-4.2 からのドリフト（G-4.3-T-R 観察 1/2 と同一系統）。
2. **一時的二重住処窓（D4/D8）**: markTransientFailure の delivery=None（P-B stranded repair、cpp:1075-1078）は durable スロットが同一 oblId を保持したまま発生し得る（settle(true) 済み）。以後の redrive/再 submit は durable busy → transport 付与となり、同一 obligation に durable+transport の 2 表現が並存し得る。結果は有界の重複 build/publish（resolve 冪等・ABA 安全・L 不変・世界整合は publish 側ガード維持）。C11-C16 はこの窓をカバーせず。
3. **Building 中の上書き窓（D3/D7）**: cpp:991 ガードは sub-state を見ないため、Builder lease 中（Building）の同一 oblId へ durable 上書きが到達し得る。success 経路（settle(false)）は terminal 化と整合して問題なし。**invalid payload discard（RECOVERY-6）が重なると**、fresh 表現が clear され delivery=Durable のまま住処消失 → obligation stranded（Live・L 台帳は無傷・shutdown で回収）。無効ペイロード（null handle / !sealed）と同一 oblId 再 submit の同時発生が必要で、到達確率は極めて低。G-4.4 実装時は「上書きは DurablePending のみ許可」または discard の state ゲートが設計論点。
4. **take() の predicate 非acquire（D6）**: 実害なし（mutex エッジで可視性確保）、文書化推奨。

## 判定: **CONDITIONAL**

- **Safety**: durable single-slot は Phase-I obligation-table contract と完全整合。blind overwrite（D2）・ownership loss（D4/D5）・resurrection（D7）・double representation の**恒久的喪失**はいずれも構造的に否定。identity/layer 分離（D1/D11）・容量分離（D9）・Phase-II 非汚染（D12）成立。
- **残存（限定的）**: Liveness 配線ギャップ（D10-1: redrive→Builder wake なし）、一時的二重住処窓（観察 2）、Building 上書き窓（観察 3）、コメントドリフト（観察 1）。いずれも liveCount_・reservation・世界整合を壊さない。
- **NO-GO 不成立の理由**: distinct Live obligation の delivery 喪失経路は存在せず（D2 ガード + redrive の保守的フォールバック）、terminal/所有保存の欠陥も観測されない（CTest 40/40×2 + C9/C11-C16 実証）。

**STOP — G-4.4 implementation / durable-table rewrite / retry redesign には進まない。** verdict 報告をもって監査完了。
