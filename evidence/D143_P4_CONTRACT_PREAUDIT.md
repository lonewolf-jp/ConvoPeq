# D143 — P4 Repair Contract Selection / Pre-Audit (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only structural / concurrency / ownership audit. **Production source changes: 0. Test source changes: 0.**
**基準:** `ConvoPeq.md Generated: 2026-08-31 13:27:23`（D142 後・実測同期確認）。ツリー = P1+P2+P3+D142 状態（`git status` 5 ファイル、D142 以降無変更）。
**命名注記:** INV-P4-x は本監査用の契約名。現行文書の対応実は `INV-X1-1..7`（h:939-944、963）および D105-R17/R18/R20/R21 契約。両者を併記する。

---

## D143-1 現行 ownership 再確定

### delivery（plain `std::uint8_t`、h:356、lock 無し）

| # | 場所 | 値 | thread（呼び出し連鎖で確定） | 関係 |
|---|---|---|---|---|
| W1 | h:402 tryInsert | None | CoordinatorLoop（submit NEW のみ） | slot 新規＝id/identity と同時書込 |
| W2 | cpp:978 submit push 成功 | Transport | CoordinatorLoop | push(977) 後 |
| W3 | cpp:993 submit defer | None | CoordinatorLoop | 異 oblId ガード通過時 |
| W4 | cpp:1008 submit durable | Durable | CoordinatorLoop | payload 書込(998-1006)+predicate release(1007) 後 |
| **W5** | **cpp:1078 markTransientFailure** | **None** | **RebuildThread**（RebuildDispatch:1006/1033/1091/1115 + Orchestrator:311/401 ← enqueuePublicationIntentForRuntimeCommit=Commit.cpp:822 / processDeferredAdmission、全て RebuildThread） | state 非触达・counter は別 atomic |
| W6 | cpp:1169 redrive durable 付着 | Durable | CoordinatorLoop | predicate release(1168) 後 |
| W7 | cpp:1194 redrive transport 付着 | Transport | CoordinatorLoop | push 後 |
| W8 | cpp:1186 P3 repair 再同期 | Durable | CoordinatorLoop | 実体生成なし |

Reader: cpp:900 / 943 / 1116 / 1142 — **全て CoordinatorLoop**。RebuildThread 側の delivery 読み取りは 0（W5 は無条件書込）。`hasPendingRecoveryAdmission` の production caller は 0（test 専用と実測確認）。

### durable slot — field 個別（「struct だから一括安全」を退け field 単位で判定）

| field | CL 書込 | RebuildThread 書込 | shutdown 書込 | CL 読 | RebuildThread 読 |
|---|---|---|---|---|---|
| `state` | 998, 1159 | 1227(take), 1261(settle true), 1266(settle false reset) | 1244(discard reset) | 991, 1158, 1100(rearm 経由読※) | 1214(take), 1260(settle), 1100(rearm※), 1243(discard) |
| `recoveryObligationId` | 1006, 1167 | 1266 reset | 1244 reset | 992, 1185 | 1101(rearm), 1221(take) |
| `pending` | 999, 1160 | 1266 reset | 1244 reset | — | — （production 読取 0・predicate は atomic 側） |
| `recoveryGeneration` | 1000, 1161 | 1266 reset | 1244 reset | — | 1225(take) |
| `buildSource` | 1001, 1162 | 1266 reset | 1244 reset | — | 1222(take) |
| `handle` | 1003, 1164 | 1266 reset | 1244 reset | — | 1218(take) |
| `epoch` | 1004, 1165 | 1266 reset | 1244 reset | — | 1219 |
| `intentId` | 1005, 1166 | 1266 reset | 1244 reset | — | 1220 |
| `reservationOwned` | 1002, 1163 | 1266 reset | 1244 reset | — | —（production 読取 0） |

※ `rearmRecoveryRetry`（cpp:1096-1103）は Orchestrator:413（RejectedPressure）経由 = **RebuildThread が state+oblId を読み settle(true)=state 書込を行う**。D140 で見落としていた第 3 の cross-thread 接触点であり、durable slot の「Consumer 専用」想定をさらに崩す（rearm のガードは state==Building ∧ oblId==O の 2 段 plain 読）。

**確定**: delivery の単一ならざる書込者 = W5（RebuildThread）。durable slot の `state`/`recoveryObligationId`/payload 群は **CL・RebuildThread・shutdown thread の 3 系統**が lock 無しで接触。

## D143-2 HB graph 完全化

```
[CL→Builder] 成立:
  CL の全 durable 書込(attach/overwrite/repair) → 同スレッドで rebuildMutex lock
  （submit: AudioEngine.h:4500-4504 / redrive 付着・repair: P2 wake Threading.cpp:279-286）
  → unlock → Builder の wait が同一 mutex acquire → program order 経由で全先行書込に HB。
  ※ P2 以降「付着・repair には必ず wake」なので、この方向は全経路カバー（overwrite は admitted=true→submit 経路 wake）。

[Builder→CL] 不在（非枯渇）:
  W5 の delivery=None 書込後、Builder は state を触らず、fetch_add(consecutiveFailureCount, acq_rel) は
  別メモリ位置で CL の redrive 経路はそれを読まない。take/settle の state 書込（1227/1261/1266）の後に
  Builder が発行する release も、CL が対応する acquire を取らない（1158/991 の state 読は plain）。
  → synchronizes-with 無し = HB 無し = conflicting plain access = 形式 UB（D140 追認・更新）。

[Builder→CL] 成立（枯渇のみ）:
  markTransientFailure 枯渇 → resolve CAS(acq_rel, h:428) → CL の state acquire 読(1114) が当該値を
  reads-from → HB。この経路のみ SAFE。

[requestRebuild の wake/notify について]（指示の重要点）:
  requestRebuild（MessageThread）は rebuildMutex 下で hasPendingTask/pendingTask を書換→notify。
  Builder の wait は同一 mutex を acquire するので **MessageThread の書込に対する HB は実在**する。
  しかし CoordinatorLoop の durable-slot 書込は別スレッドの別データであり、この mutex エッジは
  それらを順序付けない（MessageThread の unlock が CL の先行書込を発行しない）。
  → 「mutex を使っている／notify している」だけでは SAFE 不成立。CL↔Builder の durable データの
  HB は上記 [CL→Builder]（recoveryPending 経路の mutex）と [Builder→CL]（不在）で決まる。

[modification order / coherence]:
  単一 byte/word のため実装上 tearing は起きない（x86-64 TSO・MSVC）。しかし [intro.races] 上は
  非同期 plain 競合 = UB。観測される最終値はどの順序でも論理的に収束（None 後 Durable=整合、
  Durable 後 None=窓再開で冪等）だが、それは「たまたま」であり契約にできない。
```

## D143-3 候補比較（実コード照合）

### Candidate I — CoordinatorLoop adjudication（第一候補）
- **delivery 集約**: 可能。W5 の全 caller（RebuildThread 6 箇所）を「RecoveryFailure{oblId} イベント」に変え、既存 **intentQueue_（MpscBoundedRing・Vyukov・複数 producer 実証済み：submitObserve cpp:634 / submitQuarantine cpp:1340 が別スレッド producer、processIntent=CoordinatorLoop が consumer）**へ post。adjudication（delivery=None + counter + 枯渇 resolve）は CL のみで実行 → **delivery 全 writer = CL 単一**（W1-W4/W6-W8 も CL）。
- **durable slot 集約**: **take/settle は CL へ移せない**（Builder が build 中に同期的に slot を claim/return する lease 設計の本体。移すと RebuildDispatch:1058-1124 の while ループが成立しない）。よって純粋な I だけでは INV-P4-2 を満たさない（state/oblId の 3 系統書込が残る）。
- **必須随伴要素**: `state` を **atomic<uint8_t> + release/acquire ハンドシェイク**へ昇格し、規約「payload/oblId 書込は state の release 書込に sequenced-before」を課す。これにより (state, oblId, payload) の一貫観測が両方向に成立（take/settle の release を CL の attach 判定 acquire が読む／CL の attach release を Builder の take acquire が読む — 既存 mutex エッジは補助として残る）。これは III の最小部分の取り込みであり、**選択契約は「I + state ハンドシェイク」**として明示する（純 I でも純 III でもない）。
- **RT/ISR**: 両スレッド NonRT、MPSC push は lock-free。ISR 制約不変（INV-P4-10 ✓）。
- **K=4**: ロジックはスレッド移動のみで不変（INV-P4-7 ✓）。ただし **exactly-once posting** が新要件（後述 D143-5）。
- **liveness**: processIntent（runCoordinatorPhase の redrive **前**、Threading:263→270）で adjudication されるため、**同一 tick 内で None→redrive 再付着→P2 wake** が完結 — 現行（次 tick 遅れ）より改善。

### Candidate III — atomic state-domain 統合（全面再設計）
- 3 値（state/delivery/oblId）+ payload 構造体を単一 linearization に収めるには：単一語に収まらない（oblId 64bit + payload ~200B）→ seqlock（単一書込者制約が take/settle+attach の 2 方向と衝突）/ mutex（hot path 追加・P4 範囲外）/ immutable record pointer-swap（大改修）。
- **個別 atomic を並べただけでは 3 値整合は得られない**: `read state(DurablePending) → Builder reset → read oblId(0)` の stale combination は別 atomic の interleaving で残存（D140 §3 の 2 段読と同一）。指示どおり「atomic を複数置くだけで SAFE」は不採用。
- 判定: 全面 III は P4 のスコープ超過。必要な部分（state ハンドシェイク）のみ I の随伴として採用。

### Candidate II — delivery のみ atomic（反証）
1. durable slot の plain race はそのまま: CL attach(998/1159) × Builder settle(false) reset(1266) の write-write、CL 判定読(991/1158/1185) × Builder 書(1227/1261/1266) の read-write — delivery とは無関係に UB 残存。
2. P3 repair の 2 段 stale 読（1158→1185）は state/oblId の問題で、delivery atomic 化では消えない。
3. ownership protocol 不変: W5 は RebuildThread のまま、h:324 の虚偽コメントも残る。
4. delivery は単一 byte で tearing 自体が実務上無害 — II が解くのは実質ゼロ。
→ **delivery atomic ≠ durable slot consistency ≠ state/id snapshot consistency ≠ ownership protocol**（反証完了）。

## D143-4 P3 repair との整合（I + state ハンドシェイク採用時）
repair（cpp:1185-1189、CL 実行）は `state(acquire)==非NoAdmission → oblId` 読となる。規約により Builder 側は oblId/payload を reset してから state=NoAdmission を release するため、**acquire した state が DurablePending/Building なら oblId 読は必ずその state と対になる**（D140 §3 の stale strand モードが構造的に消滅）。same-holder repair / different-holder fallback / repeated redrive / failure→redrive / wake latch の論理は不変（INV-P4-5/8 ✓）。P3 生産コード自体は本 Gate で変更しない。

## D143-5 liveness / lost-event proof（Candidate I）
```
Builder failure → event post → intentQueue_（MPSC）→ processIntent 消費 → adjudication → 同一 tick redrive → P2 wake
```
- **event loss（queue full）**: 現行 quarantine は fallback ring + HealthEvent 昇格の前例あり（cpp:1352、h:998-1004）。RecoveryFailure は**非 drop 保証が必須**（失うと delivery=Transport 固定 = pre-P1 stranding の再発）。契約: 専用 fallback（小型 ring or obligation 側の pending フラグ）+ drop 時は telemetry 必須。
- **duplicate event**: 同一 failure の二重 post は counter 二重加算（R20-3「exactly once failure counting」違反）。契約: 失敗観測箇所（6 箇所）で 1 回のみ post（現行の中央化呼び出しと 1:1）。
- **stale event**: 消費時に Live チェック（現行 markTransientFailure と同一）で no-op。
- **ordering inversion**: adjudication 直列化により submit/coalesce と同一スレッド順に帰着 — 反転なし。
- **shutdown race**: 停止時残存イベントは obligation の ShutdownDiscarded resolve と整合（adjudication 未達でも terminal 化され取り残しなし）。
- **deferred 意味の維持**: delivery=None=deferred、redrive の再付着設計は不変（INV-P4-6 ΔL 不変）。

## D143-6 invariant 行列（現行実との対応併記）

| INV | 内容（現行文書実） | 現行 | I 単独 | **I+state HS** | III 全面 | II のみ |
|---|---|---|---|---|---|---|
| P4-1 delivery 単一書込者 | （h:324 の主張、実 W5 で偽） | ✗ | ✓ | ✓ | ✓（delivery 集約前提） | ✗ |
| P4-2 durable slot 単一変異権限 | （h:945 SPSC 主張、実 3 系統） | ✗ | ✗（take/settle 残存） | ✓（state HS + 規約で 2 方向直列化） | ✓ | ✗ |
| P4-3 一貫 snapshot 観測 | （新） | ✗ | ✗ | ✓（release/acquire + sequenced-before 規約） | △（実装依存） | ✗ |
| P4-4 非 HB conflicting access ゼロ | （新） | ✗ | ✗ | ✓ | ✓ | ✗ |
| P4-5 delivery XOR（INV-P3-2'） | D139 契約 | ✓ | ✓ | ✓（D143-4） | ✓ | ✓ |
| P4-6 ΔL 意味不变 | INV-X1-7/liveCount | ✓ | ✓ | ✓ | ✓ | ✓ |
| P4-7 K=4 exhaustion | R17-4/R18・kMaxObligationConsecutiveFailures=4 | ✓ | ✓（exactly-once 条件付） | ✓ | ✓ | ✓ |
| P4-8 P1/P2/P3 liveness | D137/D139 契約 | ✓ | ✓（同一 tick 化で改善） | ✓ | ✓ | ✓ |
| P4-9 shutdown/discard | INV-X1-1・discard 経路 | ✓ | ✓ | ✓ | ✓ | ✓ |
| P4-10 RT/ISR 制約 | Decision Authority 境界 | ✓ | ✓ | ✓ | ✓ | ✓ |

## D143-7 最終決定

```text
Candidate I    GO（ただし単独では P4-2/3/4 を満たさない — 随伴要素が条件）
Candidate III  NO-GO（全面再設計はスコープ超過；個別 atomic 配置は 3 値整合を与えないと証明）
Candidate II   NO-GO（反証済み: durable-slot race・2 段 stale 読・ownership 不変）

Selected contract:
    Candidate I + state handshake（I-HS）
    (a) RecoveryFailure{obligationId} を intentQueue_（既存 MPSC）経由で CoordinatorLoop に転送し、
        delivery/counter/枯渇 resolve の adjudication を CoordinatorLoop 単一権限に集約。
        非 drop 保証（fallback + telemetry）と exactly-once posting を契約に含む。
    (b) pendingRecoveryAdmission_.state を atomic<uint8_t>（release/acquire）へ昇格し、
        「payload/oblId/pending 等の書込は対応する state release に sequenced-before」規約で
        (state, recoveryObligationId, payload) の一貫観測を担保。take/settle の Consumer 所有は維持。

Reason:
    D143-1/2 の実測で delivery の W5 と durable slot の 3 系統接触が確定。delivery 単独 atomic（II）は
    実害ゼロの層だけ塗り、本質（ownership protocol）を残す。全面 III は payload 構造体を収められず
    過剰。I は既存 MPSC 輸送・既存 wake プロトコル・既存 K=4 意味論をそのまま再利用でき、
    state ハンドシェイクを加えることで P4-1..4 を同時に満たす最小契約となる。

Remaining unresolved（D144 で確定すべき事項）:
    1. RecoveryFailure 輸送の具体形（intent 型追加か専用 ring か）と fallback/overflow policy の数値。
    2. exactly-once posting の各失敗サイト（6 箇所）での証明義務と telemetry 名。
    3. state atomic 化時の discard/shutdown 経路の release 順序詳細。
    4. rearmRecoveryRetry の扱い（Orchestrator:413 の RejectedPressure もイベント化するか、
       state HS 下で Builder 実行のままか）— D144 で選ぶ。
    5. 回帰テスト設計（既存 40 + P4 専用 HB/直列化テストの構成、D142-9 相当の TEMP 管理）。
```

**P4 audit 最終判定: DATA RACE（B）確定のまま — 修復契約は I-HS に選定。**
**STOP — 実装 0。D144（Selected Contract Implementation Plan / Proof Obligations）の指示を待つ。P5/P6 非着手。**
