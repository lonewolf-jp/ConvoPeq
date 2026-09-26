# P1-5-IR-P2 — Step 5-T / P3-5-R12: First-Case Publish Timeout Source Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R12）
- **種別**: read-only source audit。実装・build・run なし。
- **目的**: first-case（g0/os1/am=-20）の Structural intent が publication に
  到達しない理由を既存 source だけで分解する。30 s 延長はしない。
- **結論**: **R12-B SOURCE-ONLY RESIDUAL**（§11）。副産物として
  headroom mismatch の source 解決を得た（§9）。

---

## 1. Scope / State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
F vehicle＋R1 diagnostic 保持／tmp/p35_F.exe／tmp/p35_R1.exe／両 run log 保持
production／CMake／JUCE 0 diff／revert なし
```

## 2. R11 Evidence Freeze（事実固定・推測なし）

```text
F first case： requested g0/os1/am=-20・seqBefore=5・seq 30 s flat・
  backlog 0・wait timeout（pair1 無効・failures=1）
window終了〜pair2 T6： sequence 5→8（3 publishes・主体／内容は未観測）
pair2 am=-6： gainDb -13.0276／-13.0276・active gen／wid／seq=8・
  sc=0（sc1 pass 含む）・hr=0.5012・os=1・irl=1／irf=1・fade=0.0600
```

stale／limiter／crossfade の分類を行わない（指示どおり）。

## 3. First-Case Intent Construction（実コード単位・表）

| 項目 | 判定（source） |
| --- | --- |
| `seqBefore=5` | harness 始動時の bootstrap／idle publish 群の到達値。runCase :411 で取得（ensureTestIr 後・configureChain 前）。前提正常（R1-C 維持） |
| 各 setter の intent | setEqBypass／setConvolverBypass／setSoftClipEnabled は guard なし → 必ず発行。setOversamplingFactor(1)は default 0(AudioEngine.h:2665)≠1 → 発行。endBulk は Structural 要求を発行。sat 1.0 は default 0.1 ≠ → 発行 |
| Structural→requestRebuild 到達 | MT 直接経路（R1 確定）。sr／bs は prepareToPlay 設定済み（:142-143）のため Deferred 不成立 |
| duplicate 適用 | 要 pending task。first-case 時点の pending は loader 由来以外になし（§4） |
| `forceMustExecute` | **false**（全 intent が Replaceable。`requestRebuild(sr,bs,policy==MustExecute)` :361）。よって重複抑止が有効 |
| `sameAsPending` | pending 存在＋全 signature 一致＋窓内が条件。startup task が消費済みなら不成立 |
| `rebuildRequestGeneration` | queue 時に `++`（:676）。commit 側の generation／worldId／sequence とは別 numbering（R5 §6維持） |
| worker queue→consume | queue で `rebuildBacklog_=1`（:755）→ `notify_all`（:765）→ worker が take 時に backlog 0（:921）→ build |
| RuntimeBuilder | transfer＋prepare＋rebuildAllIRs＋executePendingCommit（finalize 含む）→ validateWarmup |
| commit→publish | `enqueuePublicationIntentForRuntimeCommit` → coordinator 消費 → sequence bump（Commit.cpp:402）。失敗・沈黙 path あり（§7） |
| failure path | warmup-fail（R4 訂正後は conditional — 下記）、obsolete、exception、commit-side reject／defer（§7） |

## 4. Duplicate Suppression Audit（Q1–Q4 回答）

- **Q1** `endBulk` の `forceMustExecute`：**false**（Replaceable のため）。
- **Q2** `blockedAsDuplicate` の成立条件：`hasPendingTask==true` かつ
  snapshot 全一致かつ debounce 窓内（:651-667）。force によらず判定自体は走るが、
  pending 不在時は不成立。first-case で pending たりうるのは loader 由来 task のみ
  （下記 Q3）。`allowDuplicateSuppression=false` の経路（MustExecute）は本 vehicle にない。
- **Q3** first-case 時点の pending：loader は DSPCore rebuild を submit しない
  （engine IR 直接適用＋`rebuildPendingAfterLoad` は engine 側再構築。R12 §3 項）。
  よって pending は startup 残渣のみ。startup 5 publishes は消費済み
  （sequence 前進の既成事実）のため、残渣 task の存在証拠はない。
  ただし bridge-backlog は publication-stage の値であり rebuild-pending の
  直接証拠にはならない（R2 訂正の維持）。
- **Q4** pre-existing pending の排除：完全排除は不可（rebuild-pending 状態の
  vehicle 可視 proxy がない）。ただし存在した場合でも params 相違により
  replace＋queue へ進むため、単独では flat を説明しない（下流 failure との複合が必要）。

## 5. endBulk の扱い（維持）

`endBulkParameterRestore(true)` は Structural／RequestRebuildKindEntry／
Replaceable の要求であり、独立 publish の保証ではない（上記 §3–§4 のとおり
merge／duplicate／worker／commit の各関門に従う）。

## 6. Rebuild Queue / Worker Consumption

- queue→notify→take→backlog-clear（:921）→build の正常系は source 上完結する。
- worker thread の永続退出は shutdown／shouldExit のみ（window 内なし・R3 維持）。
- predicate＋notify pattern は正規（lost wakeup なし）。
- 30 s window 内の非消費を positive に示す材料はなく、消費の直接証拠もない
  （bridge-backlog の語義限界）。U6 は下記 §8 の残余形で保持する。

## 7. RuntimeBuilder / Validation・Commit / Publication

- build 成功後は executePendingCommit が finalize する（R4 訂正の維持）。
  よって validateWarmup は通過しうる。R3-D には戻らない（指示どおり）。
- commit 側関門（PublicationAdmission.cpp:11-58）：
  shutdown／generation-stale／**not-finalized（sealed snapshot 基準）**／
  health Critical-Degraded／pressure throttle／fading-defer。
  sealed snapshot は ENGINE UI 側値（loaded＋finalized）のため本ケースでは通過見込み。
  health／pressure／fading は vehicle 非観測（残余）。
- failure／defer はいずれも bridge-backlog・sequence に現れない設計であり、
  R1 観測（flat＋0）と両立する。
- BuildErrorPolicy の retry は bounded（≤80 ms×3）のため retry 累積では 30 s を埋めない。

## 8. U1–U6 Reclassification（U2≠R3-D の厳守）

```text
U1 pressure／health： 残余（全 set-path が fault-class 前提。positive support なし。
  commit 側 health／pressure／fading も vehicle 非観測のため残余に含める）
U2 warmup／validation： R3-D 形は排除（R4 訂正）。その他 validation-fail
  （IRRate／IRBlockMismatch 等）は本ケース条件（48k／1024 一致）で不成立 → 排除。
  WarmupFailed 一般形の残余は commit 側 not-finalized と統合して扱う（下記）。
U3 obsolete： 排除（window 内 newer-generation 源なし。R3 維持）
U4 worker exception： 形式的残余（support なし。thread 生存は確定）
U5 build＞30s： 単独原因として排除（bounded loop＋retry上限）。
  ただし worker-start／build-start／completion／commit／publish の時刻は
  既存 evidence（seq／bridge-backlog のみ）では観測不能。
  よって「U5 を U4／U6 と区別できない」ことを明記する（指示どおり）。
  60／120 s 実行による確認は行わない。
U6 worker not consumed： 狭義残余（task-state の vehicle 可視 proxy なし）。
  スレッド死亡説・predicate 誤り説は排除済み（R3 維持）。
```

## 9. sc／headroom／fade mismatch source tracing

### headroom（解決・source 確定）

`setInputHeadroomDb` は conv-first 時に -6.0 dB 上限で clamp する
（Parameters.cpp:228-241：`convIsFirst → maxDb=-6.0`）。
本 vehicle（convBypass=false・ConvolverThenEQ）は該当し、要求 0.0 dB は
**-6.0 dB（gain 0.5012）に clamp される**。default -6.0 と一致するため
setter は no-op（intent なし）だが、値は要求どおりではない。
5-way 分離の結論：(1) capture failure でも (2)–(5) でもなく、
**「要求自体が clamp により -6 dB だった」**（指示の5択外の確定分岐として記録）。
よって active hr=0.5012 は requested-effective と一致し、mismatch ではない。
（A3 の Δ に対しては両行 common-mode のため non-differentiator。）

### softClip sc=0（未解決・記録のみ）

requested sc1=true に対し active sc=false。merge 継承と整合するが、
 redispatch 分岐・late-publish 分岐も排除不能のため原因を決めない。

### fadeTimeSec=0.0600（R12-unresolved）

`m_phaseFadeTimeSec {0.060}`（AudioEngine.h:2474）との値一致はあるが、
world.overlap への格納 branch（CrossfadeAuthority の max 選択）は
publisher 不明のため確定不能。指示どおり **R12 unresolved** として残す。
「fade=0.06だからtimeout」は主張しない。

## 10. Residual Unknowns

- first-case task の queue／consume／drop の内訳（vehicle 可視 proxy なし）
- seq 5→8 の 3 publishes の発行主体・内容（gap 内・非観測）
- U1／U4／commit-side health-pressure-fading の直接値
- worker/build 各 stage の時刻（既存 evidence では観測不能・§8）
- fade 0.06 の格納 branch（§9）

## 11. R12 Gate Decision

```text
R12-A（specific …確定）： 該当なし。
  headroom clamp は確定したが timeout 原因ではなく mismatch 解決である。
R12-B SOURCE-ONLY RESIDUAL： ADOPTED
  残余＝{U1（pressure／health／commit-side）・U4（formal）・
  U6（task-state）・fade branch}。U2（R3-D形）・U3・U5（単独）・
  admission／filter／defer／async／MixedPhase は排除済み。
R12-C（追跡不能）： 非該当（経路自体は追跡可能）。
```

## 12. Next Gate

- 60／120 s 延長・F/R 再実行・accessor 変更・logger 追加・production 計装・
  P3-1-D・limiter／crossfade／stale 帰属はすべて保留維持。
- R11 の Case-B 証拠（sc・hr・fade）に hr 解決（§9）を反映する。
  残る Case-B は sc のみ＋fade 未解決である。
- §7解釈制約・H-B 対象外を維持する。
