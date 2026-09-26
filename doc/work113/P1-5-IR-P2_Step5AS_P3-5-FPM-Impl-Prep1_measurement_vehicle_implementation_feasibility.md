# P1-5-IR-P2 — Step 5-AS / P3-5-FPM-Impl-Prep1: Measurement Vehicle Implementation Feasibility Gate

- **作成**: 2026-09-24 / work113 P1-5-IR Phase 2（P3-5-FPM-Impl-Prep1）
- **種別**: **read-only feasibility gate**。
  production 0・test 0・CMake 0・build 0・test 実行 0・実計測 0・M0/M1/M2 execution 0。
- **判定**: **FPM-IMPL-PREP-PASS**。
  T3b 5 点は既存 public API のみで取得可能。M0/M1/M2 の実行経路は既存 test infrastructure 上で
  構成可能。production／getter／counter／shutdown／epoch／reclaim／Recovery semantics の
  変更は不要。次は **P3-5-FPM-Impl-1 — Measurement Vehicle Implementation Gate** へ進める。
- **入力**: R33-A-PASS＋R33-B-PASS＋R34-PASS（判定 C／P3-5 CLOSED）＋FPM-PREP-1-PASS（測定プロトコル固定）。
- **STOP 後停止**: 本 gate 完了後は実装に進まず STOP。vehicle 作成・build・run は次の別 gate。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a（R33-A／R33-B／R34／FPM-PREP-1 と同一）
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（継承）
R16/R19/R22/R27 counters／R23 vehicle／R30・R32 vehicle 保持（変更なし）
ConvoPeq.md = 2026-09-23 23:33:38, Length 5535334（本 Step で再実測・同一）
  ※ ConvoPeq(3).md（2026-09-23 23:49 頃の更新物）は本環境に存在しない。
     Get-ChildItem -Filter 'ConvoPeq*' で確認できるのは ConvoPeq.md（23:33:38）と
     ConvoPeq.code-workspace のみ。したがって同一性確認は「live source が
     FPM-PREP-1 の authority（23:33:38）と一致すること」の再照合で代替した（§2）。
     live source の 9 項目再確認はいずれも FPM-PREP-1 §2 と一致したため、
     (3).md 不在による判定支障はない。
production diff = R27 のみ／test diff = R23+R30+R32 vehicle／CMake clean（継承・本 Step 無変更）
FPM-Impl-Prep1 の差分 = 本ドキュメントのみ（read-only）
```

検証手段は Read／Grep のみ。build・test 実行・計測は行っていない。

---

## 2. Latest ConvoPeq.md Reconciliation

指示の 9 項目を live source で再確認（いずれも FPM-PREP-1 §2 の authority と一致・変更なし）：

```text
[1] getPhase()：ShutdownRuntime::getPhase（ISRShutdown.h:199／ISRShutdown.cpp:57-60）。
    既存 public・Non-RT 呼出可。R30/R32 vehicle（P1PolyphaseGainCharacterization.cpp:1254）と
    D167（PublishPipelineIntegrationTests.cpp:647）が同一 API で h.stop() 後に取得済み。
[2] collectResult()：ShutdownRuntime::collectResult（ISRShutdown.h:224／ISRShutdown.cpp:161-181）。
    completed＝phase 別名のみ。blockingReason／transitionViolations／lateCallback／
    postStopEnqueue を同時取得。既存 public・R30/R32・D167 が同一 API で取得済み。
[3] isFullyDrained()：AudioEngine::isFullyDrained（AudioEngine.h:1600／Threading.cpp:153-213）。
    既存 public noexcept。R30/R32 vehicle が p1RecDrainFields（:1104-1120）で h.stop() 後に
    取得済み（fullyDrained フィールド）。
[4] collectDrainAudit()：AudioEngine::collectDrainAudit（AudioEngine.h:1604／Threading.cpp:109-151）。
    既存 public noexcept。P2／P8 非露出は継承（T3b authority には使わない・diagnostic のみ）。
[5] start()/stop()：AudioEngineHarness::start（AudioEngineHarness.cpp:20-37・initialize＋
    prepareToPlay＋audio thread 起動）／stop（:38-53・stopAudioOnly→requestTerminalRelease→
    releaseResources terminal pass）。D167／D169／R30/R32 の全 vehicle が同一入口を使用。
[6] submitRecoveryIntent()＋snapshot：AudioEngine::submitRecoveryIntent（AudioEngine.h:4686-4729・
    authoritative runtime 不在時は silent absorb）＋getCurrentBuildSnapshotForRecovery
    （:4677-4681）。R30 D1 vehicle が同一 API で episode 駆動済み（:1189-1194）。
    handle 調達は registerDSPHandleForRuntime（:4520）＋observePublishedWorld（:3846-3847）。
[7] R30/R32 episode-driving：P1PolyphaseGainCharacterization.cpp:1122-1246
    （pre snap→submit→seq 前進待ち（20s budget）→post snap→dSeq/dCoord/dCmt/dTake/dBld 確認）。
    実行順序は R31-B 契約（Running 中 waitForDrain なし・terminalization は h.stop() 後 T3）。
[8] M0/M1/M2 実行入口：AudioEngineHarness 単一 executable（CMakeLists.txt:1904-2003）。
    CLI 分岐は PublishPipelineIntegrationTests.cpp:1124-1209
    （--p1-recovery-origin→runP1RecoveryOrigin 等・T1-T4/measurement 先例あり）。
    新規 scenario フラグの追加先は同分岐（test 変更は次 gate・本 gate では行わない）。
[9] build／test target／収容場所：add_executable(AudioEngineHarness …)（:1904-1921）＋
    add_test(NAME AudioEngineHarness …)（:2003）。新規 vehicle は同 executable 配下の
    新規 CLI mode として収容可能（CMake 新規 target 不要・次 gate 判断）。
```

---

## 3. T3b Capture Feasibility

### 3.1 取得子（既存 public API のみ・新規 getter 不要）

```text
phase:                e.isrShutdownRuntime().getPhase()            （ISRShutdown.h:199）
completed:            e.isrShutdownRuntime().collectResult(h,0)    （ISRShutdown.h:224）
blockingReason:       同上 .blockingReason
transitionViolations: 同上 .transitionViolations
isFullyDrained:       e.isFullyDrained()                           （AudioEngine.h:1600）
diagnostic:           e.collectDrainAudit()                        （AudioEngine.h:1604）
```

- `isrShutdownRuntime()`（AudioEngine.h:1595-1596）は NonRT admission-control accessor として
  既存 public。Orchestrator（RuntimePublicationOrchestrator.cpp:70-76）も同一 accessor を使用。
- R30/R32 vehicle（:1253-1276）は h.stop() 後に getPhase＋collectResult＋drain audit を
  既存 public API のみで取得済み。T3b 化に必要な追加は `isFullyDrained()` の 1 呼追加のみ
  （既存 public・R30 が p1RecDrainFields で取得済みの値を判定に使うだけ）。

### 3.2 同一 episode 性（timing skew 監査）

懸念：`collectResult()` と `isFullyDrained()` の取得タイミングずれで別 episode の値を混ぜないか。

判定：**混ざらない。** 理由（source）：

1. 両取得は `h.stop()` 復帰後（ShutdownComplete 到達後）に行う。R30/R32 は同一位置で取得済み。
2. ShutdownComplete 到達後は terminal pass が完了しており、新規 publish／retire／epoch 前進の
   producer は存在しない（producer join 済み・admission Closed）。`isFullyDrained()` の述語値は
   凍結状態であり、取得順序（collectResult→isFullyDrained／逆）で episode が変わることはない。
3. 唯一の例外は `~AudioEngine`（dtor）の forced drain（CtorDtor.cpp:228-292）だが、
   T3b capture は `h.stop()` 直後・dtor 前に行う（R30/R32 と同一順序）。dtor 後の値は読まない。
4. `collectResult()` 自体が phase／blockingReason／TV の atomic snapshot であり、
   取得中の phase 遷移はない（ShutdownComplete は terminal・後続遷移なし）。

⇒ 同一 shutdown episode の 5 点として取得可能。production getter 追加は不要。

---

## 4. M0 Feasibility

```text
start（h.start）→ startup settle（authoritative runtime＋seq 前進＋backlog 0）→ stop（h.stop）
→ T3b capture（§3.1）→ drain audit
```

- settle 条件は R30 の Precondition 1（:1138-1152・observePublishedWorld＋
  hasAuthoritativePublishedRuntime）＋ backlog 0（P1PolyphaseGainCharacterization.cpp:92-96
  の waitPublishSettled 相当）の再利用で構成可能。
- M0 は Recovery・publish・crossfade・retire の追加操作なし。既存 infrastructure のみで完結。
- R32 D0 実績（Control でも Unknown）のため baseline INVALID を許容する設計は FPM-PREP-1 §5
  で固定済み。vehicle 側の追加分岐は不要（§9 分類で記録するだけ）。

判定：**FEASIBLE（既存 infrastructure のみ）。**

---

## 5. M1 Feasibility

```text
start → settle → submitRecoveryIntent()（1 episode）→ seq advancement（20s budget・R30 同一）
→ dCoord/dCmt/dTake/dBld confirmation（R30 期待形 dSeq≥1・dCoord=dCmt=dTake=dBld=0）
→ settle（sleepPump 500・R32 同一）→ stop → T3b capture
```

- episode-driving は R30 の :1185-1242 を kEpisodes=1 にした部分流用で構成可能。
  snap 関数（p1RecReadSnap：:1077-1093・req／que／dup／take／bld／cmt／coord／drp／seq／blo）は
  既存のまま再利用可（変更不要）。
- silent absorb 経路（submitRecoveryIntent :4695-4699）は settle 済み authoritative runtime の
  ため発火しない（R30 Precondition 1 が保証）。
- **R31-C の E2 winner は追加しない**（指示継承・winner／liveCount の露出は不要）。
- Running 中の waitForDrain／drain_pre／drain_post は組込まない（R31 STOP-1 契約・R30 R32 修正済み）。

判定：**FEASIBLE（既存 episode-driving の流用・production 変更なし）。**

---

## 6. M2 Feasibility

```text
publish（SR 変更 re-prepare 等）→ crossfade settle（isPending()==false）→ recovery
→ publish → settle → shutdown → T3b
```

- publish 操作：D167 の SR 変更 re-prepare（PublishPipelineIntegrationTests.cpp:585-600・
  seq 前進待ち）が既存手順として存在。duplicate-prepare collapse 回避は SR 交互パターン
  （:607-623）を流用。
- crossfade settle：crossfadeRuntime_.isPending() の engine 側観測は
  collectDrainAudit.activeCrossfadeCount（Threading.cpp:119）に既存露出。
  M2 の settle 条件は同 audit 値==0 で構成可能（新規 getter 不要）。
- recovery 操作：M1 と同一（§5）。
- retire は publish／recovery の旧 World／DSP retire として自動発生する（DSPLifetimeManager 経由）。
  明示的 retire 操作の追加は不要。
- 音質・buzz・limiter・NUC は扱わない（FPM-PREP-1 §7・§10 継承）。

判定：**FEASIBLE（既存操作の組合せ・新規 production 経路なし）。**

---

## 7. Existing Vehicle Reuse Map

| 資産 | 流用内容 | 流用先 | 変更要否（次 gate） |
| --- | --- | --- | --- |
| R30 episode-driving（:1185-1242） | pre／post snap＋seq 待ち＋期待形判定 | M1（kEpisodes=1） | test 追加のみ（B なし・C なし） |
| R30 snap（p1RecReadSnap :1077-1093） | req／que／dup／take／bld／cmt／coord／drp／seq／blo | M1／M2 episode 観測 | 変更なし（そのまま呼出） |
| R30 drain emit（p1RecDrainFields :1104-1120） | fullyDrained＋audit 13 項目 | M0/M1/M2 diagnostic | T3b 判定に isFullyDrained() を使うだけ（emit 維持） |
| R30 terminal capture（:1254-1273） | getPhase＋collectResult＋admission | M0/M1/M2 T3b capture | terminalOk 式を T3b 5 点に置換（test のみ） |
| R30 control（:1176-1184 control_notrigger） | trigger なし seq 自発前進確認 | M1 の control 対照（任意） | 変更なし |
| D167 settle／re-prepare（:540-630） | admission Open 確認＋SR 変更 publish | M2 publish 群 | 変更なし |
| D167 terminal（:635-669） | h.stop＋admission／phase／completed／TV | M0/M1/M2 の雛形 | blockingReason＋isFullyDrained 参照を追加（test のみ） |
| Harness start／stop（AudioEngineHarness.cpp） | engine lifecycle | M0/M1/M2 共通 | 変更なし |
| CLI 分岐（PublishPipelineIntegrationTests.cpp:1124-1209） | --p1-recovery-origin 等の先例 | 新規 --fpm-m0／--fpm-m1／--fpm-m2（命名は次 gate） | test の分岐追加のみ |
| CMake 収容（CMakeLists.txt:1904-2003） | AudioEngineHarness executable＋add_test | 同 executable 配下 | CMake 変更不要見込み（新規 target 不要） |

---

## 8. Required File Changes（次 gate の見込み・本 gate では変更しない）

```text
変更候補（分類 A・test のみ）：
  src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp
    または新規 FPM vehicle cpp（配置は次 gate 判断）
    - M0/M1/M2 の run 関数（R30 runP1RecoveryOrigin の構成を流用）
    - T3b 5 点 capture＋§9 分類 emit
    - CLI mode 追加（--fpm-*・命名は次 gate）
  変更なし見込み：
    src/audioengine/（全 production）
    CMakeLists.txt（新規 target 不要・既存 AudioEngineHarness 配下）
    build.bat／CMakePresets.json
```

---

## 9. Change Classification A / B / C

```text
A. 既存 test infrastructure の追加 test vehicle → 次の Implementation Gate で許可候補
   - §8 の test 変更候補が該当。R30/R32/D167 の流用であり、新規 production 経路を含まない。
B. production source／public API／getter／counter の変更 → 即 STOP、別 gate
   - 本 gate では該当なし。T3b 取得に新規 getter／counter は不要（§3）。
C. shutdown／epoch／reclaim／Recovery semantics の変更 → 即 STOP、別 gate
   - 本 gate では該当なし。M0/M1/M2 は既存 semantics 上で構成可能（§4-§6）。
```

---

## 10. Implementation Boundary（次 gate への申送り）

```text
許可候補（FPM-Impl-1）：分類 A の test vehicle 追加のみ。
  - 新規 counter／getter を作らないこと。
  - shutdown／epoch／reclaim／Recovery semantics に触れないこと。
  - R31-C（per-episode E2 winner）・R33-C（P2／P8／per-entry）を混ぜないこと。
  - buzz／limiter／NUC を混ぜないこと。
即 STOP 条件（FPM-Impl-1 開始後に発覚した場合）：
  - 分類 B／C の変更が必要と判明 → 実装を中断し別 gate へ分離。
```

---

## 11. Gate Result

```text
[1] T3b 5 点を既存 public API だけで取得可能 … PASS（§3・新規 getter 不要・同一 episode 性確認）
[2] M0 の実行経路を既存 test infrastructure 上で構成可能 … PASS（§4）
[3] M1 の実行経路を既存 episode-driving 流用で構成可能 … PASS（§5・E2 winner 追加なし）
[4] M2 の実行経路を既存操作の組合せで構成可能 … PASS（§6・音質等除外）
[5] production 変更不要 … PASS（§8・B 該当なし）
[6] getter／counter 追加不要 … PASS（§3・§8）
[7] shutdown／epoch／reclaim／Recovery semantics 変更不要 … PASS（C 該当なし）
→ FPM-IMPL-PREP-PASS
```

---

## 12. STOP / Next Gate

```text
FPM-Impl-Prep1 PASS（本 gate・read-only・変更 0）
    │
    └── P3-5-FPM-Impl-1 — Measurement Vehicle Implementation Gate（別 gate）
          - 分類 A の test vehicle 追加のみ（§8・§10 の境界内）
          - build／run／実計測の扱いは同 gate の指示に従う（本 gate では行わない）
```

- source は R27 production＋R23/R30/R32 test vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7 解釈制約・H-B 対象外を維持する。
- 本 gate はここで停止する。
