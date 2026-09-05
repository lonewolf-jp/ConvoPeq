# D162-2-I1-D — Profile D Completion Record（Device cycle ×6・D-2〜D-7）

```text
Date:     2026-09-05（実行 00:24〜00:27・6 runs・間隔 2s）
Type:     Profile D 完遂（production 変更 0 / test 0 / CMake 0 / build 0）
Baseline: ConvoPeq.md Generated 2026-09-04 23:38:10（R2 A′ 修復込み）
Binary:   build-diag RelWithDebInfo ConvoPeq.exe（Sep 4 23:09 = R2 build・A/B/C と同一）
条件:     --cli-run --cli-log-file D{n}.log --cli-device-type WindowsAudio --cli-exit-ms 15000
          （IR / burst / rebuild なし・run 間隔 2s）
判定:     **crash 系 12 gate = 6/6 全 PASS・lifecycle 会計 gate 1 件に D profile 固有の
          未処分個体（reconfigure orphan）を検出 → PASS/STOP 判定はユーザー監査に委ねる**
          （本記録は実測事実 + root cause 分析を提示）
```

---

## 0. 実行上の訂正記録（2 回の無効 batch → 3 回目で条件成立）

`--cli-device-type "Windows Audio"`（スペース入り）の引数伝達に 2 回失敗した。

| batch | 引数 | 結果 | 保管先 |
| --- | --- | --- | --- |
| 1 回目 | `Windows Audio`（裸） | Start-Process がスペースで分割 → `requested=Windows` unknown → **switch 不成立**（default device 継続 = D-1 probe と同一の防御経路） | `invalid-device-type-arg-quoting/` |
| 2 回目 | `"Windows Audio"`（リテラル引用符） | 引用符が JUCE 側まで残留（`requested="Windows Audio"`）→ equalsIgnoreCase 不一致 → unknown | `invalid-device-type-literal-quotes/` |
| **3 回目（正式）** | `WindowsAudio`（スペース除去） | `normalizeCliValue()` 正規化経路（MainWindow.cpp:33/445）で解決 → **6/6 `success requested=WindowsAudio resolved=Windows Audio current=Windows Audio`** | `D2.log`〜`D7.log`（正式） |

- 3 回目はコードが正式サポートする一致形式（normalize 経路・h:444-445 の第 2 条件）。
- 無効 2 batch も 12 run 分の control data として有用（unknown fallback 経路でも
  exit 0x0 / zone clean / bootstrap DSP destroy 鎖 6/6 完遂 — D-1 §1.3 防御実証の反復確認）。

## 1. 4 大集計（指示 §6 明示 4 点）

```text
D-2〜D-7:
  6/6 exit=0                → ✓ 実測（全 run 0x00000000）
  6/6 clean shutdown        → ✓ 実測（SHUTDOWN_BEGIN → reset completed → LOGGER_DETACH/END 3 行完備）
  6/6 residual=0            → ✓ 実測（I0 定義: remaining≠0 0 件・destroy 済み DSP の remaining=0 6/6）
  6/6 lifecycle 1:1         → ✗ **不成立（pointer identity）** — 詳細 §3
```

**lifecycle の実態（run あたり DSP 2 個体・破壊 1 個体）**:

| 個体 | 構築 | retire | EBR | destroy | 残置 |
| --- | --- | --- | --- | --- | --- |
| bootstrap DSP（log 開始前構築・map 登録済み・gen=4※） | log 外 | **V-D retired=1**（reconfigure 内） | enqueue=0 epoch=9 | **D117_DESTROY → remaining=0** ✓ | なし |
| reconfigure placeholder（`phase=construct` 1 件・**map 未登録**） | **in-log**（prepareToPlay） | retired=0（map MISS → no-op） | なし | **なし** | **~108MB orphan（process exit で OS 回収）** |

※ D4 のみ gen=3（bootstrap DSP の採番差・会計に影響なし）。

## 2. Gate 別集計表（13 項目 × 6 run）

| Gate | D-2 | D-3 | D-4 | D-5 | D-6 | D-7 | 集計 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exit | 0x0 | 0x0 | 0x0 | 0x0 | 0x0 | 0x0 | **6/6 ✓** |
| crash dump（新規） | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 ✓**（CrashDumps 10 不変） |
| shutdown zone 3 行 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6 clean** |
| device lifecycle（switch success → callbacks → closeAudioDevice → reset） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6 ✓** |
| DSP lifecycle（constructed = destroyed） | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **6/6 ✗**（§1・§3） |
| residual（remaining≠0） | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0 ✓** |
| EBR final pend / ovf | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | **6/6 ✓** |
| E-3（INV-D162-8） | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0 ✓** |
| registered direct destroy（`[D162-2B_DESTROY]`） | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0 ✓** |
| stale-map forbidden HIT | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0 ✓**（MISS 1 / HIT 0 ×6） |
| Signature A / B / C | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | **6/6 ✓**（dump 0・exit 0x0・crash chain 0） |
| duplicate destroy | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0 ✓** |
| shutdown-window XRUN | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0 ✓**（総 XRUN 0） |

補助実測（run ごと）:

| run | log 行数 | session callbacks | Priv final | bootstrap DSP destroy 鎖 | placeholder orphan |
| --- | --- | --- | --- | --- | --- |
| D-2 | 867 | 83 | 394MB | VD retired=1 → epoch=9 → DESTROY → remaining=0 | 000001E296694080 retired=0 |
| D-3 | 864 | 85 | 399MB | 同型 6/6 | 000001EBBA4B1080 retired=0 |
| D-4 | 854 | 83 | 400MB | 同型（gen=3） | 000002168D97C080 retired=0 |
| D-5 | 859 | 80 | 399MB | 同型 | 0000021A12E1E080 retired=0 |
| D-6 | 865 | 84 | 398MB | 同型 | 0000026117A63080 retired=0 |
| D-7 | 858 | 84 | 393MB | 同型 | 000001E3BC52A080 retired=0 |

- E-4 会計: CREATE 0 = CONSUME 0 + OVERWRITE 0 + CLEAR 0 + DISCARD 0 × 6/6
  （IR/rebuild なしで deferred churn ゼロ・C profile と同型の自明収支）。
- CLEAR_SHUTDOWN_DISPOSITION = 0 × 6/6（slot 空で shutdown — 契約上正常・C profile と同型）。
- S3 retired=1 不発（slot 空）— A′ 無条件呼出（ReleaseResources.cpp:565-566）は毎 run 到達・
  slot 空 no-op 帰還（冪等性の追加実証）。
- session 形状: `[WORLD] Active=0`（publish なし・placeholder bypass）・procTimeUsAvg 3-5μs。

## 3. D profile 固有の発見 — reconfigure orphan DSP（root cause 確定）

### 3.1 現象（6/6 決定論的）

device type 明示 switch を行うと、run あたり **DSPCore 2 個体**が存在し、
**1 個体（bootstrap）のみが authority 経由で破壊**される。switch 後に
prepareToPlay が構築する placeholder（`[DSP_FOOTPRINT] phase=construct`・~108MB）は
handle registry に登録されず、shutdown で `retired=0`（map MISS no-op）のまま
**物理破壊されずに process exit**（D117_DESTROY / DSP_DESTROY_FOOTPRINT /
DSP_FOOTPRINT_RELEASED の 3 出力とも不在）。crash / corruption はなし（exit 0x0・dump 0）。

### 3.2 root cause chain（コード確定）

```text
(1) MainWindow.cpp:457  setCurrentAudioDeviceType("Windows Audio")
      → JUCE device change → 同一 AudioEngine に releaseResources + prepareToPlay
(2) ReleaseResources.cpp:87   shutdownRuntime_.closeAdmission()
      → INV-LIFE-9（ISRShutdown.h:299「Closed→Open は存在しない」）で admission 永久 Closed
(3) PrepareToPlay.cpp:241/264 placeholder 生成 → aligned_unique_ptr::release() で
      active slot へ所有権移転 → :277 commitRuntimePublication(needsRegistration)
(4) AudioEngine.h:4613  tryAdmit(1) 失敗（admission Closed）
      → :4614 return {Failed, CallerDestroy} — side-effect zero（:4633 の
        registerDSPHandleForRuntime に到達しない = 未登録）
      → pubResult は ignoreUnused（PrepareToPlay.cpp:280）で失敗が握り潰される
(5) session 15s: publish なし（rev=0 / currentUuid=0 / dspReady=0・[WORLD] Active=0）・
      rebuild 要求 5/5 queued のまま 1 回も build されない（Build 経路も admission gate 対象）
(6) shutdown: ReleaseResources.cpp:331 retire(slot ptr) → map MISS → retired=0 →
      enqueue せず return（DSPLifetimeManager.cpp:48-49 early return = 破壊しない）。
      V-D（:465-467）は activeHandle null で不発。~AudioEngine（CtorDtor.cpp:204-214）は
      handle authority のみ（E-2 で pointer-value retirement 廃止済み）で不発
(7) placeholder ~108MB が未処分のまま process exit（OS 回収）
```

### 3.3 分類

- **D162-2 の回帰ではない**（S3 / V-D / A′ は全て本設計どおり閉包 — bootstrap DSP の
  retire → EBR → destroy → remaining=0 が 6/6 完璧・E-4 会計成立・crash 0）。
- **reconfigure 経路の構造的 gap**（長期残留）: admission close 後の同一エンジン再準備で
  placeholder が registry に登録されない設計漏れ。I1 の全 profile（A/B/C）は
  mid-session reconfigure を実施しないため未観測だった。D profile（device cycle）が
  初めてこの経路を exercise して顕在化させた。
- 深刻度評価の材料:
  - CLI run では process exit で回収されるため実害なし（crash / UAF / corruption なし）。
  - 対話運用では **device type / sample rate / buffer 変更のたびに同一経路が動く**ため
    「switch 後に publish が復活しない（bypass 継続）+ switch 1 回あたり ~108MB 残置」
    が繰り返され得る（repair 対象として I2 候補）。
- STOP 条件の字義との突合: 指示の STOP トリガ列挙（`remaining≠0` 行出力・shutdown-window
  XRUN・crash 系）は **いずれも不発**。一方「DSP lifecycle constructed = destroyed」gate は
  pointer identity で **6/6 不成立**。判定（I1-D PASS/STOP・I1 final 可否）はこの 1 点に
  焦点を当てて実施すること。

## 4. D profile 観点（指示 §「特に見るべき点」5 項目）

1. **device shutdown/close と AudioEngine teardown の競合** — なし。`~MainWindow step 8
   closeAudioDevice`（:834）は releaseResources 完了（:830 ABOUT_TO_EXIT_SCOPE）後、
   step 9 audioEngineProcessor.reset → ~AudioEngine の順で直列完遂。shutdown window 内の
   audio callback 0（最終 CBSUMMARY は SHUTDOWN_BEGIN 前に確定）。
2. **A′（clearDeferredForShutdown 無条件）の device shutdown 経路への副作用** — なし。
   ReleaseResources.cpp:565-566 が毎 run 到達・slot 空 no-op（E-4 CLEAR 0・
   CLEAR_SHUTDOWN_DISPOSITION 0）。reconfigure orphan への作用も構造的に不可
   （A′ は deferredSlot_ のみ対象・orphan は registry/slot 問題）。
3. **device lifecycle 後も DSP → authority retire → EBR → destroy が閉じるか** —
   bootstrap DSP は 6/6 で閉包（V-D retired=1 → enqueue epoch=9 → D117_DESTROY →
   remaining=0）。**ただし switch 後 placeholder は authority に到達すらしない**
   （未登録のため）— §3。
4. **remaining≠0 の出現** — 0 件（`remaining=[1-9]` grep 6 log で 0 hit）。
5. **shutdown window XRUN** — 0 件（総 XRUN も 0・timestamp 確認不要）。

## 5. 付帯観察（gate 外・記録）

- `[MMCSS-ASIO] FAILED: primary err=1552 task=Pro Audio` × 2/run（D profile のみ・
  A/B/C/invalid batch は 0）: reconfigure で新規生成された audio thread の MMCSS 登録が
  1552 で失敗するもの。AudioEngine.Mmcss.cpp:139-144 の期待値分類に 1552 が
  `ERROR_NO_MORE_ITEMS` として盛り込まれているが、実際の winerror 定数値と不一致の可能性
  （1552 が分岐に catch されず FAILED に到達）。機能影響なし（false → NativeRT/no priority
  フォールバック・AFFINITY pin は実行済み :142-145）。前例: D133-1_6burst_diag.log:111
  （同一経路・同一 err）。MmcssPolicy 誤割り当て（Windows Audio なのに ASIO tag）も
  同時観測 — DeviceSettings.cpp:1259-1260 の device type 通知が reconfigure 後に再実行
  されない構造。I2 調査候補（gate 判定には無関係）。
- セッション全体が unpublished（`[WORLD] Active=0`・procTimeUs 3-5μs = bypass）で動作 —
  §3 (5) の帰結。CLI 15s 運用では音響的無影響。
- 無効 2 batch（12 runs）は unknown fallback 経路の clean shutdown 反復実証として
  保存済み（D-1 §4 の防御実証と整合）。

## 6. 成果物

- 正式ログ: `evidence/D162-2-I1/D2.log`〜`D7.log`（6 file）
- 無効 batch: `invalid-device-type-arg-quoting/D2-D7.log`・
  `invalid-device-type-literal-quotes/D2-D7.log`
- 実行 script: `evidence/D162-2-I1/D_runs.ps1`（最終版 = WindowsAudio normalize 経路）
- production / test / CMake / build 変更: **0**（binary Sep 4 23:09 を A/B/C から継続使用）
