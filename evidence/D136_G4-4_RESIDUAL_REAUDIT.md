# D136 — G-4.4 Residual Liveness / Representation-Race Re-audit (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only structural / concurrency / ownership audit. Production source changes: **0**. Test source changes: **0**.
**基準:** working tree @ HEAD `5f6f48c` + 未コミット G-4.1…G-4.3-T delta（`git status` = cpp/h/tests の 3 ファイルのみ、G-4.4 時と無変更）。`ConvoPeq.md` = `Generated: 2026-08-30 23:47:13`（marker `using ObligationDomains` 1 件、現ツリーと一致する最新生成版。ユーザー参照の 23:43:26 版は本ファイルではない — 本監査は 23:47:13 版＋実ツリーを基準）。
**Build evidence:** `evidence/g44_ctest.log`（同一ツリー、本日取得: Debug 40/40 / Release 40/40、exit 0×2）。
**Overall: CONDITIONAL**（Safety/台帳は成立。ただし wake gap・double representation・memory-order 契約欠陥を**欠陥として確定**。新規発見 1 件含む）

---

## D136-A — Builder wake gap: **FAIL**

1. **対象関数**: `redriveDeferredRecoveryObligations`（cpp:1110-1120）/ `redriveDeferredRecovery`（cpp:1126-1183）/ `runCoordinatorPhase`（Threading.cpp:258-325）/ `CoordinatorLoop::run`（ISRCoordinatorLoop.cpp:31-49）/ Builder CV wait（RebuildDispatch.cpp:854-859）
2. **実コードの状態遷移**: redrive は `delivery=None` の Live obligation に durable（cpp:1158-1170、`recoveryAdmissionPending_` release  cpp:1168）または transport（cpp:1174-1178、`pendingIntentCount_` +1）を付着させる。**付着後に Builder への wake を発火させる文は存在しない**。
3. **thread ownership**: 付着 = CoordinatorLoop（1ms tick、ISRCoordinatorLoop.cpp:39/47）。消費 = RebuildThread（Builder）。
4. **atomic ordering**: `recoveryAdmissionPending_` release（cpp:1168）は isFullyDrained（cpp:549）に可視化するだけで、rebuildCV を起こさない。
5. **具体的 interleaving**: A が durable を占有中（DurablePending）に B が submit→queue full→deferred（delivery=None、cpp:991-996）。Builder が A を消費・settle(false) して while 終了出来る→CV wait 復帰。次 tick redrive が B を durable 付着（cpp:1158）→**notify なし**。Builder は `hasPendingTask || publishRetryReady || recoveryPending || exit`（RebuildDispatch:855-858）のいずれかが成立するまで無限に眠る。
6. **invariant への影響**: 安全性（liveCount_・reservation・喪失）は無傷。I4 の「blocked admission は park 可」と整合するが、**「有限時間内に観測・処理される」保証はコード上に存在しない**。
7. **既存テストによる実証**: なし（単一スレッドテストのみ、`std::thread` 0 件）。
8. **反例**: 上記 interleaving 自体が反例 — 全 wake 源を網羅しても redrive 由来のものは無い:
   - `rebuildCV.notify` 全 4 サイト: RebuildDispatch:744（requestRebuild task — MessageThread）/ 799（stop）/ Threading:295（**watchdog のみ** — `hasDeferredRequest()` かつ 100 tick 経過後だけ。F6-5 で毎 tick 通知は廃止済み、h:2719-2724。h:2714 の「毎 tick set + notify」コメントは陳腐化）/ AudioEngine.h:4504（submitRecoveryIntent — 新規 submit 限定）。
   - `recoveryRetryReady`（h:2728）は「provenance only; not part of rebuildCV wake predicate」（h:2726-2727）。
   - `rebuildCV.wait` は無期限（timed wait なし）。dedicated retry timer なし（grep 確認）。
9. **修正が必要か**: **要**（次 Gate 入力: redrive 付着時の wake 配線、または等価の有限時間保証 — 例: 付着時に recoveryPending 相当を立てる／Builder に周期再点検を入れる。本 Gate では禁止）。
10. **次 Gate 要求**: 「redrive→Builder wake」機構の設計 + T3 相当の統合テスト（Builder スレッド実在環境）。

### 新規発見（G-4.4 の 4 項目外）— transport recovery build/warmup failure の stranded obligation
- RebuildDispatch.cpp:996-1000（build null → `diagLog; continue;`）と同 1009-1023（warmup 失敗 → DSP 破棄 → `continue;`）は、**`markTransientFailure()` を呼ばない**。durable 側（1075-1084 / 1102-1107: settle(true)+markTransientFailure）と非対称。
- 結果: transport 表現を pop 済みの obligation は Live のまま delivery=Transport（cpp:978 で設定、pop は delivery を変えない）に固定され、redrive 候補（delivery==None のみ、cpp:1116）から**恒久的に除外**される。retry も terminal も起きない（Orchestrator:400-401 の markTransientFailure は publish 拒否経路のみで、この build 失敗は publish に到達しない）。
- 影響: 台帳は不変（L 水増しなし・喪失なし・shutdown で回収）だが、**build が決定的に失敗する状況で recovery が静かに永久停止**する。liveness 欠陥として G-4.4 残存項目より深刻（再現が時間窓でなく決定論的）。
- 判定帰属: D136-A 系（liveness）。次 Gate 実装入力に追加必須。

## D136-B — transient double representation: **FAIL**（契約違反として確定。影響は有界）

1. **対象関数**: `settlePendingRecoveryAdmission`（cpp:1237-1248）/ `markTransientFailure`（cpp:1066-1092）/ `redriveDeferredRecovery`（cpp:1158-1182）/ Builder durable ループ（RebuildDispatch:1056-1124）
2. **実コードの状態遷移（指示 interleaving の形式化）**:
   ```
   t0  A Live, delivery=Durable, durable=DurablePending(A)
   t1  Builder take(A): state→Building（cpp:1208）           [delivery=Durable 不変]
   t2  build 失敗 → settle(true): Building→DurablePending（cpp:1241-42）
   t3  markTransientFailure(A): delivery→None（cpp:1078）、counter+1（1080-82）
       ── この時点で durable=DurablePending(A) かつ delivery=None ──
   t4  Coordinator redrive(A): state!=NoAdmission → transport 試行（cpp:1173-74）
       push 成功 → delivery=Transport（cpp:1176）
   t5  A の表現: durable slot（DurablePending, A）＋ queue intent（A）= **二重**
   ```
   線形化点: settle(true)=state 書込（cpp:1242、RebuildThread 単独）/ markTransientFailure の delivery=None=plain 書込（cpp:1078、CoordinatorLoop 単独）/ redrive の transport 付与=push（ring の release/acquire）。t3-t4 の順序はスレッド間自由 → 窓は t3 以降**無期限**（Builder が spin 上限 break（RebuildDispatch:1082-83）で眠ると、次の wake まで t5 が持続）。
3. **thread ownership**: delivery/state 書込は前述の通り CoordinatorLoop / RebuildThread に分離だが、t3 の delivery=None 化が「durable 保持中」に到達する唯一経路（P-B stranded repair の副作用）。
4. **atomic ordering**: delivery は plain（h:324 単一書込者前提）— 順序問題ではなく**意味論**の問題。
5. **具体的 interleaving**: 上記 t0-t5。到達条件は「durable 経路の transient failure ×（同一 obligation の）以後の redrive」— spin 上限 break 経路では窓が無限に伸びる。
6. **invariant への影響**:
   - **delivery uniqueness（§5 / C16 single-representation）違反**: 「delivery != None なら再送しない」保証が delivery=None 化で回避され、durable 側は delivery enum が示さない住処を保持し続ける。**enum ⟺ 実住処の対応が壊れる**（Transport 表示中に Durable 実在）。
   - **INV-X1-5**: 「1 logical admission = at most 1 reservation」— durable 占有 + transport reservation（pendingIntentCount_ +1）で 1 admission あたり 2 予約。**違反**（h:943 の文言に照らす。coalesce 由来ではないが、予約数の意味論では同一）。
   - **INV-X1-6**: 「durable admission は queue residency と二重計上しない」— **カウンタレベルでは成立**（pendingIntentCount_ は queue のみ、durable は非計上のまま）。壊れるのは residency 実態の方。
   - liveCount_ / terminal 所有: 無傷（resolve CAS 冪等、h:423-436）。
7. **既存テスト**: C11-C16 は「durable 保持中に delivery=None が生じる」前提を作らない（redrive 系は settle/markTransientFailure を組み合わせない）。R18 系（T-R18-3/4 等、test:1422/1451）は markTransientFailure 単体。**未カバー**。
8. **反例**: t0-t5 が反例そのもの。到達可能（durable 経路 transient failure は本番で起こり得る: build null / warmup 失敗）。
9. **修正が必要か**: **要**。分類: 「単なる transient duplicate」ではなく**delivery uniqueness / INV-X1-5 の契約違反**（ただし結果は有界重複 build/publish に限定され、台帳・世界整合を壊さない — 「resolve 冪等だから安全」で終わらせないこと、との指示に従い、違反は違反として確定）。
10. **次 Gate 要求**: markTransientFailure の durable 保持中 delivery 処理の設計（例: durable 保持中は None にしない／settle(true) 側で delivery=Durable を再宣言）、または redrive の same-oblId-busy 時の transport 試行禁止。world 側は重複 publish 時に同一 generation で 2 回 commit され得る（RebuildDispatch:1035-1036/1115-1116 は enqueue 時現在値を採番 → 2 回目も obsolete 判定されず冗長 commit）— 許容範囲だが重複退 retire を通す。

## D136-C — Building 中 durable overwrite race: **CONDITIONAL**

1. **対象**: `submitRecoveryRequest` durable 部（cpp:991-1008）/ `takePendingRecoveryAdmission`（cpp:1193-1210）/ `settlePendingRecoveryAdmission`（cpp:1237-1248）/ RECOVERY-6（RebuildDispatch:986-987 transport、1062-1065 durable）
2. **状態遷移**: ガード cpp:991 は `state != NoAdmission && oblId != oblId` のみ。**sub-state（Building/DurablePending）を見ない**ため、`state==Building && 同一 oblId` → cpp:998-1008 の上書きが**到達可能**（条件: queue full + 同一 {handle,target} 再 submit + Builder lease 中）。single-writer ordering では防がれない（書込者は確かに CoordinatorLoop 単独だが、**読者/共作者 RebuildThread が Building を保持している**）。
3. **thread ownership**: 上書き=CoordinatorLoop、take/settle=RebuildThread — 同一 plain フィールド群への並行書込（state: DurablePending 書込 vs Building 読取/settle 書込）。
4. **atomic ordering**: 形式上データ競合（D136-D 参照）。
5. **interleaving（指示シナリオの決着）**:
   ```
   A Building → 同一 A resubmit（queue full）→ 上書き: state=DurablePending, delivery=Durable（cpp:998-1008）
   → Builder 側後続処理:
      (a) build success → settle(false) クリア + resolve(Published) 終端 → **無害**（fresh 表現は terminal と共に消える）
      (b) transient failure → settle(true): state==Building でないため no-op（cpp:1241）→ DurablePending 保持 → markTransientFailure: delivery=None → 以後 D136-B と同じ窓 → **有界**
      (c) RECOVERY-6 discard → settle(false) が fresh 表現を消し delivery=Durable のまま住処消失 → **stranded**
   ```
6. **invariant への影響**: (c) のみ実害。しかし **(c) は本番で到達不能と確定**: discard 条件は `qHandle.isNull() || !buildSource.sealed`（RebuildDispatch:1062）で、本番 buildSource は `getCurrentBuildSnapshotForRecovery()`（AudioEngine.h:4456-4460）= `currentBuildSnapshot_` の値コピー、その書込元は Commit.cpp:800 の `sealedSnapshot`（sealRuntimeBuildSnapshot で sealed=true、RebuildDispatch:144-148/657/1039/1119）**のみ**。handle も ProcessIntent.cpp:137 で `!request.handle.isNull()` ゲート済み。→ RECOVERY-6 は本番 payload に対して dead branch。
7. **既存テスト**: testRecoveryDurableAdmission（test:690-803）は take/settle 基本のみ。Building 中 resubmit 併発テストなし。
8. **反例**: (c) は本番不到達（sealed/non-null 不変条件に依存）。(a)(b) は無害/有界。**stranding の反例は production では成立しない**。
9. **修正が必要か**: 即時不要だが**設計論点として要**: 「上書きは DurablePending のみ許可」または discard の state ゲート化を次 Gate で明文化すべき（現状の安全性は RECOVERY-6 不到達という**外部条件**に依存しており、payload 妥当性ゲートが変わると即座に危険化するため）。
10. **次 Gate 要求**: Building 中上書きの禁止（または settle(false) の Building ゲート）を契約として固定する設計 + 反証テスト。

## D136-D — take() memory-order / predicate correctness: **FAIL**（memory-order contract defect として記録・未修正）

1. **対象**: `recoveryAdmissionPending_` 全 store/load、`pendingRecoveryAdmission_.state` 全 read/write、take/hasPending、CV predicate、rebuildMutex。
2. **全 store/load 列挙**（grep 実測）:
   - `recoveryAdmissionPending_` store: cpp:1007（release、submit）/ 1168（release、redrive）/ 1228（release、discard）/ 1247（release、settle false）。load: cpp:549（acquire、isFullyDrained）/ 1214（acquire、hasPendingRecoveryAdmission — **production 呼び出し元は shutdown/drain 系とテストのみ**）。
   - `state` 書込: cpp:998（CoordinatorLoop 上書き）/ 1159（CoordinatorLoop 付着）/ 1208（RebuildThread take）/ 1242（RebuildThread settle true）/ 1246（RebuildThread settle false）/ 1227（shutdown discard）。読: 1195（take）/ 991（submit）/ 1158（redrive）/ 1241（settle）/ 1224（discard）/ 1215（hasPending）。
3. **thread ownership**: 2 スレッド（CoordinatorLoop / RebuildThread）が同一 plain struct を交互に read/write。
4. **atomic ordering — 形式証明の成否**:
   - **成立する経路**: submit 経路の wake（AudioEngine.h:4500-4504）は CoordinatorLoop が `rebuildMutex` を lock してから recoveryPending=true → Builder の wait が同一 mutex を acquire → **CoordinatorLoop の program-order 先行書込（submit 経路の durable 書込のみならず、同一スレッドの redrive 書込 cpp:1158-1168 も）が全て可視**。watchdog 経路（Threading:292-295）も CoordinatorLoop lock 由来なので同様。
   - **成立しない経路**: Builder が **MessageThread の requestRebuild**（RebuildDispatch:741-744、task 書込は rebuildMutex 下）で起床した場合、mutex エッジが順序付けるのは MessageThread の書込のみ。CoordinatorLoop の redrive 付着（cpp:1158-1168）との間に synchronizes-with 関係は**存在しない**。この状態で Builder の take()（cpp:1195）が plain `state` を読むと、CoordinatorLoop の書込と**形式上のデータ競合（C++ 標準上 UB）**。`take()` は `recoveryAdmissionPending_` を acquire しない（release/acquire ペアは cpp:1168↔1214 として存在するが、Builder 消費経路は 1214 を通らない）。
   - 同様に D136-C の上書き（cpp:998）vs Builder の take/settle 読書（1195/1241）もエッジなし。
5. **具体的 interleaving**: redrive 付着（CoordinatorLoop）∥ requestRebuild 起床→take()（RebuildThread）。
6. **invariant への影響**: 実装上の保証文（h:945-946「SPSC — 競合なし」/ h:968「atomic 不要」）は**単一書込者の同時性**を述べておらず、**可視性の順序**を述べていない。順序保証は wake 経路の mutex に暗黙依存しており、全 wake 経路で成立しない。
7. **既存テスト**: なし（単一スレッド）。TSan 系テストも存在しない。
8. **反例**: 上記 5 が反例（形式的）。実害（stale read による誤判定）は x86-TSO + MSVC の実装上、aligned enum store のため観測されない — だが**判定基準はコード上の保証**であり、「x86-TSO で問題ない」は採用しない（指示）。
9. **修正が必要か**: **記録必須（production 変更なし）**。`memory-order contract defect`: durable slot の plain 書込に対する全消費経路の順序付けが、CoordinatorLoop 由来 wake の場合にのみ成立する。
10. **次 Gate 要求**: (i) take() 前に `hasPendingRecoveryAdmission()`（acquire）でゲート、または (ii) durable 書込を atomic 化（state を atomic<uint8_t> + release/acquire）、または (iii) 「durable 消費は CoordinatorLoop wake 限定」を契約化し requestRebuild 経路で take しない。いずれか 1 つを設計で確定。

## D136-E — 容量・reservation conservation: **PASS**

1. **対象**: `pendingIntentCount_` / `liveCount_` / durable 占有 / delivery の全書込。
2. **全 fetchAdd/fetchSub 追跡（recovery 関連）**: +1 cpp:976（submit push 前）/ 1174（redrive push 前）；−1 cpp:983（submit push 失敗）/ 1179（redrive push 失敗）/ 1261（popRecoveryRequest）。`liveCount_`: +1 h:408（tryInsert のみ）/ −1 h:430（resolve CAS 成功時のみ）。`markTransientFailure`（cpp:1066-1092）: **カウンタ操作 0**（delivery=None + consecutiveFailureCount + 枯渇時 resolve のみ）。`settle`/`redrive`/`take`: liveCount_ 触达 0。
3. **指示経路 `markTransientFailure → delivery=None → redrive → transport`**: pendingIntentCount_ は +1（1174）→ pop で −1（1261）の対が 1 回のみ — **予約の増殖なし**。durable 占有はカウンタ非計上（INV-X1-6 のカウンタ側は成立）。liveCount_ 不変（ΔL=0）。
4. **混同チェック**: 32（h:363 table）/ 256（h:927 queue）/ 1（durable）/ 257（I4:943/984 の合成値）— 相互参照なし。D136-B の二重住処は**物理 residency 257 枠内**（queue 1 + durable 1）に収まり、容量不変は維持。
5. **判定**: 台帳・カウンタは全 interleaving で整合。**PASS**（違反は D136-B の residency 意味論側に限定され、カウンタ側には波及しない）。

## D136-F — 既存テスト coverage: **GAP**

- **T1**（DurablePending→take→Building→settle(true)→markTransientFailure→redrive→transport 併存）: **未カバー**。R18 系（test:1374-1597）は markTransientFailure を durable take/settle と併用せず、C11-C16 は transient-failure 由来の delivery=None×durable 保持窓を作らない。二重住処の挙動（characterization）を固定するテストなし。
- **T2**（Building + 同一 obligation resubmit + RECOVERY-6 discard）: **未カバー**。testRecoveryDurableAdmission（690-803）は単一スレッドで take/settle のみ。stranding は本番不到達（D136-C）だが、その不到達性自体を固定するテスト（sealed payload 前提の回帰）もない。
- **T3**（idle Builder + redrive-only wake の eventually-consumed）: **未カバーかつ現ハーネスで構成不能** — ISRSemanticValidationTests は `std::thread` 0 件の単一スレッド設計。Builder スレッド実在の統合テスト（AudioEngineHarness 系）にも redrive-only wake シナリオなし。
- 追加テストは**行っていない**（指示）。

---

## Overall: **CONDITIONAL**

| 項目 | 判定 |
|---|---|
| D136-A Builder wake gap | **FAIL**（有限時間保証なしを確定 + 新規: transport build/warmup 失敗の stranded obligation） |
| D136-B transient double representation | **FAIL**（delivery uniqueness / INV-X1-5 違反として確定。影響有界・台帳無傷） |
| D136-C Building overwrite | **CONDITIONAL**（到達可能、本番 stranding は不到達と決着。安全性が RECOVERY-6 不到達に外部依存） |
| D136-D memory-order | **FAIL**（memory-order contract defect 記録。修正なし） |
| D136-E capacity/reservation | **PASS** |
| D136-F test coverage | **GAP**（T1/T2/T3 全未カバー） |

**NO-GO 不成立の理由**: distinct obligation の delivery 喪失・台帳破壊・resurrection はいずれも成立せず（D136-E PASS、D136-C 本番不到達、resolve 冪等）。**PASS 不成立の理由**: wake 保証欠如・契約違反窓・形式順序欠陥が**理論窓ではなくコード上確定の欠陥**であり、うち 1 件（transport build 失敗 stranding）は決定論的に到達する。

**修正対象として確定できたもの（次 Gate 実装入力）**:
1. redrive 付着時の Builder wake 配線（A-1）。
2. transport recovery build/warmup 失敗時の markTransientFailure 欠落（A-新規、**最優先候補** — 決定論的）。
3. delivery uniqueness 修復: markTransientFailure の durable 保持中 None 化、または redrive same-oblId-busy の transport 試行禁止（B）。
4. durable 消費の memory-order 契約確定（D、3 案から選択）。
5. Building 中上書きの契約化（C）。
6. T1/T2/T3 の回帰テスト設計（F）。

**STOP — 実装・テスト追加・設計変更なし。verdict 報告をもって D136 完了。**
