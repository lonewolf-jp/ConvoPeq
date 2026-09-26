# P1-5-IR-P2 — Step 5-I / P3-5-R1: Publish Timeout Diagnostic（read-only audit＋最小diagnostic実行）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R1）
- **目的**: P3-5 F の第1ケース publish timeout（30 s）が cold-start 遅延（R1-A）か
  publish 未発生（R1-B）か前提相違（R1-C）かを、test-only vehicle 変更前に切り分ける。
  order comparison・R実行はしない。30 s→60/120 s 延長はしない。
- **方法**: 最新 source の publish 経路 read-only audit＋
  既存 API（sequence／backlog）のみを使う最小 test-only diagnostic の1回実行。
- **raw evidence**: `tmp/p35_R1_run.log`、`tmp/p35_R1.exe`
 （`a00d140d479184d4`・F vehicle＋diag）。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
source                       P3-5 F vehicle（revertなし・保持）
tmp/p35_F.exe                f472381550414f51（保持）
production/CMake/JUCE        0 diff
```

## 2. latest ConvoPeq.md source audit（R1-1）

### 2-1. seqBefore 取得位置（A・確定）

`runPair` :664 ensureTestIr → `runCase` :411 `seqBefore` 取得 → :412 `configureChain`。
よって `seqBefore` は IR-load の publish より後、このケースの configureChain より前であり、
「このケースが要求した publish より前」の sequence になっている（前提正常）。

### 2-2. configureChain 実行スレッド（B・確定）

- harness は console アプリだが `AudioEngine : … private juce::AsyncUpdater`
  （AudioEngine.h:586-591）。AsyncUpdater 基底構築時に MessageManager が
  main thread 上に生成されるため、main thread＝message thread である。
- よって `submitRebuildIntent`（RebuildDispatch.cpp:151-403）は
  `isMessageThread=true` で Structural intent を**直接経路**
  `requestRebuild(sr, bs, …)`（:343-363、sr/bs>0 条件付き）に送る。
  非MT分岐（setRebuildReason＋triggerAsyncUpdate :378-402）は本経路では不使用。
- 対象4 setter（setEqBypassRequested／setConvolverBypassRequested／
  setSoftClipEnabled／endBulkParameterRestore(true)）はいずれも当該直接経路に入る。
  うち前3者は guard なし（P3-1-B §2-1維持）のため intent は必ず発行される。

### 2-3. Replaceable merge 条件（C・再確認・P3-1-B と一致）

`rebuildOutstanding && pending.valid && kind/class/policy/fingerprintVersion/
structuralHash/fingerprint/deferCategory 一致`（:201-209）＋
Replaceable＋debounce窓内（:211-218）→ `REBUILD_MERGED` で return（:271-285）。
第1ケース時点では outstanding なしが既定（backlog 0 を後述診断で確認）のため、
merge は成立しにくく、直接 dispatch が既定動作である。

### 2-4. endBulkParameterRestore(true)（D・維持）

Structural／RequestRebuildKindEntry／Structural／Replaceable の rebuild request であり、
「必ず独立 publish を発生させる」とは仮定しない（指示どおり）。
merge 規則（§2-3）の適用対象である。

## 3. seqBefore acquisition（再掲・正常）

§2-1 のとおり順序・値は正常。診断 t=0 行の `before=5` は harness 始動時の
bootstrap／idle publish 群（initialize・prepareToPlay 経路の既存 publish）と整合する。

## 4. configureChain thread/context（再掲）

main thread＝message thread（§2-2）のため直接 `requestRebuild` 経路。
async bridge（triggerAsyncUpdate→handleAsyncUpdate）は本ケースの経路ではない。

## 5. submitRebuildIntent path（再掲）

直接経路（MT）：merge 非該当 → tryAdmit → `requestRebuild(sr,bs)`。
抑止分岐（いずれも publish を生まない）：shutdown／pressure（:241-269）／
KindFiltered（:301-313）／AdmissionClosed（:324-337）／sr-bs欠落時Deferred（:365-374）。

## 6. Replaceable merge analysis（再掲）

第1ケースは outstanding 空のため merge 既定動作は dispatch 側である（§2-3）。
merge による publish 欠落説は本ケースでは成立しにくい。

## 7. waitWorldPublished semantics（R1-2・固定）

worktree 実装（:94-111）：`seq!=before && backlog==0` → **`sleepPump(250)`** →
再確認（seq不変＋backlog 0）→ PASS。worktree 値は 250 ms である
（指示文の「300 ms」は概数として扱い、監査は source 値を引用する）。
失敗 mode は3分岐：(i) publish 未発生（seq不変）、(ii) backlog 非ゼロ継続、
(iii) publish 後も安定再確認が通らない（flapping）。WARN 1行だけでは区別不能
（指示どおり「30秒超＝build遅延」とは断定しない）。

## 8. diagnostic vehicle diff（R1-3・1ファイル限定）

対象：`src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp` のみ。
production／CMake／JUCE＝0（確認済み・§1）。

```text
a. waitWorldPublished 内に TU-local flag 付き sampling（5 s 刻み・既存WARN行のみ）
   `[P1CHAR] WARN pubdiag t=<ms> before=<N> seq=<cur> backlog=<bl>`
b. runCase strict-honor 分岐で flag on/off（第1ケースのみ観測）
c. 新 production accessor なし・新 production logger なし・
   timeout値変更なし（30000維持）・post-settle／sleepPump(800)不変
```

build: Release／OFF／当該1 TU再compile／BUILD_EXIT=0。
identity: `a00d140d479184d4dec166cb51c5a21d3adb6fd551114f2ab09c22bfc1a3cb3b`
（F `f4723815…` と意図どおり相違・diag分）。`tmp/p35_R1.exe` に保存。

## 9. F fresh-process diagnostic result

```text
[P1CHAR] WARN pubdiag t=0     before=5 seq=5 backlog=0
[P1CHAR] WARN pubdiag t=5005  before=5 seq=5 backlog=0
[P1CHAR] WARN pubdiag t=10009 before=5 seq=5 backlog=0
[P1CHAR] WARN pubdiag t=15011 before=5 seq=5 backlog=0
[P1CHAR] WARN pubdiag t=20008 before=5 seq=5 backlog=0
[P1CHAR] WARN pubdiag t=25002 before=5 seq=5 backlog=0
[P1CHAR] WARN publish not confirmed os=1 sc=0
[P1CHAR] p15ir id=g0_os1_am-6 … gainDb_sc0=-13.0276 gainDb_sc1=-13.0276 …
         limitingEngaged 0/0・hardClamp 0/0・clipEngagement=1437・clipEngMax=0.000000
[P1CHAR] summary cases=1 failures=1（clean exit・crashなし）
```

- 30 s 全域で `seq=5` 固定・`backlog=0` 継続。途中 advance・backlog>0 episode なし。
- am-6 行は再現（sc0/sc1 exact 一致・clipEngagement のみ 1400→1437 と jitter）。

## 10. R1-A / R1-B / R1-C classification

```text
R1-A（30 s超のcold-start/build latency）: REJECTED
  30 s以内の中途 publish（N→N+1）も backlog>0 episode も観測されず。
  よって待機budget延長（60/120 s）は根拠なし。本延長は行わない。
R1-B（merge/defer等で publish まで進まず）: ADOPTED
  seq=flat・backlog=0 の30 s完全静止は「単なる待機不足」ではない。
  直接経路の Structural intent が publish を生まなかった。
  残る分岐は admission（tryAdmit／pressure）・sr-bs defer・
  requestRebuild→worker→publish 間のいずれかであり、要追加監査。
R1-C（seqBefore/backlog/configureChain前提相違）: NO VIOLATION FOUND
  seqBefore=5・順序・初期値は想定どおり。前提修正は不要。
```

## 11. next gate

```text
P3-5-R1（R1-B）
   ↓
publish/rebuild経路の read-only 追加監査
（admission／pressure／defer／requestRebuild→worker→commit のどこで止まるか。
  PUBLISH/REBUILD_TELEMETRY 非出力の制約下で source のみから特定する）
   ↓
R実行には進まない（R1で分離できるまで固定・指示どおり）
```

- R vehicle build／R実行／F/R比較／Pattern分類／P3-1-D はすべて保留を維持。
- 30 s 延長・sleep/post-settle 変更・logger/accessor 追加は行わない。
- F vehicle source（＋diag）は保持・revert しない。
- §7解釈制約（新core／publish戻り値／envelope肯定・否定禁止）を維持。
  H-B 因果主張なし。
