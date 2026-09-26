# P1-5-IR-P2 — Step 5-U / P3-5-R13: First-Case Timeout Residual Observability Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R13）
- **種別**: read-only source audit。実装・build・run なし。
- **目的**: R12-B の残余（U1／U4／U6／fade）を既存 API／既存診断だけで
  どこまで観測可能か確定する。
- **結論**: **R13-B（既存観測限界が確定）**。pre-queue／post-queue の境界で
  可観測性が分かれることを確定した（§7）。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
F vehicle＋R1 diagnostic 保持／tmp/p35_F.exe／tmp/p35_R1.exe／両 run log 保持
production／CMake／JUCE／measurement 変更 0／revert なし
```

## 2. U6：rebuildBacklog／pending／generation／diagnostics 全列挙

### A. rebuildBacklog_（全 write site・確定）

```text
=1： queue 成功時（RebuildDispatch.cpp:755）のみ
=0： worker take 時（:921・build 前）のみ
```

- notify_all（:765）／predicate（:884-889・hasPendingTask／publishRetryReady／
  recoveryPending／exit）／take＋所有権移動（:906-909）は正規 pattern。
- build 開始前・終了後・exception 時・obsolete／drop 時・shutdown 時の
  backlog 書込みは存在しない（上記2件のみ）。
- 帰結：bridge-backlog とは別物であり（R2 訂正の維持）、
  rebuildBacklog_ 自体も vehicle 非出力（test 側 getter なし）である。

### B. pendingTask（lifecycle・確定）

```text
producer： queue 成功時（:753 pendingTask = task・hasPendingTask=true）
replacement： 重複でない新 task（currentToRelease 旧所有権は retire へ・:665-671）
release／consume： worker take 時に所有権移動＋pending クリア（:906-908）
clear： shutdown 時（:894-895）
currentDSP 所有権： take 時に worker へ移動する（:906）
```

### C. generation 対応表（≠ を維持）

```text
rebuildRequestGeneration： request counter。queue 時に ++（:676）→ task.generation
RuntimeWorld::generation： world identity（graph-generation 系予約値。別 counter）
publicationSequence：      commit 順序（reserve時 fetch-add。reserve≠commit）
worldId：                  build 毎一意値（generator next()）
```

対応は commit 経路でのみ成立し、counter 間の等価を仮定しない（R5 §6維持）。

### D. 既存 diagnostics の U6 証明力（直接／間接の判定）

- `getRebuildDispatchDiagnostics()` の8 counters のうち live は3件のみ：
  requestCount（:536）／queuedCount（:764）／blockedPendingDuplicateCount（:785）。
  残り5件（RecentDuplicate／QueueFull／Drained／Matched／Fallback）は
  writer が存在せず恒久ゼロである（exhaustive grep 確定）。
- よって区間差分で証明可能なのは **request 到達・queue 到達・duplicate-block** まで。
  **worker consumption の直接証明は不可能**（consume counter が存在しない。
  DrainedCommand は dead field である）。
- `getRuntimeLifecycleDiagnostics`（commit 観測）・bridge-backlog は間接証拠に留まる。
- U6 の verdict：queue-state は既存のみで分離可能／consumption は不可。
  U6 を単体で扱わず U6-pre（分離可）と U6-post（不可）に分割する（§6）。

## 3. U4：worker exception（possible／happened の分離）

- worker try/catch：rebuildThreadLoop try（:869）＋ `catch (std::exception)`（:1410）＋
  `catch (...)`（:1415）。いずれも DBG のみ＋loop 継続（thread 生存は確定）。
- `RuntimeBuilder::build` は noexcept（:454-455）。内部 `try`（:467）の
  `bad_alloc → ResourceUnavailable`／`catch(...) → InternalError`（:527-535）、
  IR contract REFUSED（:510-519）、InvalidInput はいずれも
  worker 側で diagLog のみ＋`continue`（:1219-1226）→ release 沈黙。
- 帰結：exception **可能**（catch 器あり）。しかし発生時の既存観測値への痕跡は
  存在しない（counter・log とも release 非出力）。
  よって **U4-happened は OBSERVATIONALLY INDISTINGUISHABLE**。
  possible≠happened を分離記録する（指示どおり）。
  thread-death 形は排除（catch は継続する）。

## 4. U1：pressure／health／fading（入力→条件→reject/defer→後処理）

### pressure（実際の admission 入力の確定）

- `PublicationAdmission::evaluate` の入力は：sealedSnapshot（irLoaded／irFinalized）・
  req.generation 対 currentGen・health ref・`retirePressurePublicationThrottleActive_`
  flag・fading uuid。**telemetry struct・backlog counter・saturation count・
  reject count は入力ではない**（source 確定）。
- set-path は全て fault／recovery／retire-critical 前提（R3 維持）。
  first-case での成立 positive support なし → NOT ESTABLISHED（§6）。

### health（遷移条件＋first-case 適用性）

- Critical／Degraded は monitor Error 状態の収束が必要
  （RuntimeHealthMonitor.cpp:380-418：retire／publication／overflow／reader／age）。
  clean fresh run での成立経路なし → NOT ESTABLISHED。
  直接読値の vehicle proxy はない（残余）。

### fading（値の data-flow）

- `m_phaseFadeTimeSec{0.060}` 既定 → policy → CrossfadeAuthority max() 選択 →
  snapshot → `world.overlap.fadeTimeSec`（観測 0.0600 と一致）。
- ただし gate 役割（arm／canCrossfade／DeferredFadingActive）は存在・uuid ベースであり
  **fade 値自体は判定入力ではない**。値の一致は由来証明にならない。

## 5. fade 独立トラック（U1 と混同しない）

```text
1 default（m_phaseFadeTimeSec 0.060）： source 確定
2 active world 保存値（0.0600）： 観測確定
3 crossfade 開始条件への関与： なし（bool／uuid ベース）→ ELIMINATED
4 fading-defer 条件への関与： なし（同上）→ ELIMINATED
5 rejection 条件への関与： なし → ELIMINATED
```

fade 値は未解決のまま残るが（publisher 不明）、timeout 経路への関与は排除する。
「fade=0.06だからtimeout」は主張しない（指示どおり）。

## 6. R12 U分類の更新

```text
U1 pressure／health／fading： NOT ESTABLISHED
  （成立 positive support なし。直接観測手段なし。commit 側入力は確定§4）
U4 worker exception： OBSERVATIONALLY INDISTINGUISHABLE（発生時）
  ＋ thread-death 形は ELIMINATED。possible≠happened を分離（§3）
U6： U6-pre（queue／duplicate-block）は既存差分で分離可能（残余でない）。
  U6-post（worker consumption）は OBSERVATIONALLY INDISTINGUISHABLE。
  （consume counter 不在・Drained dead field 確定による）
fade： timeout 経路への関与は ELIMINATED（§5）。値由来は R12-unresolved 維持。
```

## 7. R13-A/B/C 判定

- **R13-A**：U6-unconsumed／U4-happened／U1-active／fade-involved のいずれも
  具体的に確定できず → 非該当。
- **R13-B**：**ADOPTED**。既存観測限界が確定した：
  pre-queue（request／queue／duplicate）は差分で分離可能、
  post-queue（consume／build／commit-drop）の内部は既存手段で不可分。
- **R13-C**：非該当（経路自体は追跡可能）。

## 8. Next gate（R13-B の帰結）

- 既存観測だけでの区別不能が確定したため、次段は
  「U6-post／U4／U1-commit-side を分離する最小 test-only observability vehicle の
  設計監査」へ進む（実装は承認後）。
- 設計材料（本監査で確定）：dispatch 3 counters の区間差分で pre-queue を確定させる；
  post-queue 側は worker 側可視性なしでは不可分のため、観測点は queue 境界に置く；
  60／120 s 実行・F/R 再実行・production 計装・logger 追加・accessor 追加・build・
  P3-1-D・帰属結論はいずれも本 Step では行わない。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
