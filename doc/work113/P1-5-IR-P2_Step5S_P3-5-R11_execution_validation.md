# P1-5-IR-P2 — Step 5-S / P3-5-R11: Execution Validation Report（Fのみ・R停止）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R11）
- **判定**: **R11-A observability PASS ＋ STOP（F first-case timeout のため R 停止）**。
  比較・P3-1-D・帰属には進まない。
- **raw evidence**: `tmp/p35_R11_F_run.log`（166行・summary有・clean exit）。
  使用 binary：`tmp/p35_R10.exe`（`a7bfb513a7d7eea7`・再buildなし）。

---

## 1. State Freeze＋実行条件（実行前確認）

```text
HEAD 1e9e63e3／kP15FullMatrix true／CONVOPEQ_CORRECT OFF
R10 binary SHA a7bfb513a7d7eea7d2bc198530f7d2b55125a207d00a0dfd1f76d278228e18ba（MATCH）
production／CMake／JUCE 0 diff／残存プロセスなし（clean）
vehicle 条件不変（30s wait・sleepPump(800)・500ms post-settle・capture・warmup・
IR・DFT・g0／os1・kP15・OFF のすべて維持）
```

## 2. F execution log（全文・要旨）

```text
[P1CHAR] flag_macro=0
[pubdiag t=0..25009] before=5 seq=5 backlog=0（全6点 flat・R1再現）
[P1CHAR] WARN publish not confirmed os=1 sc=0（pair1 am-20 無効化・failures=1）
[P1CHAR] p15ir id=g0_os1_am-6 … gainDb_sc0=-13.0276 gainDb_sc1=-13.0276 …
         limitingEngaged 0/0・hardClamp 0/0・clipEngagement=1411・clipEngMax=0.000000
         ＋ ab0／aa0／ab1／aa1（§3）
[P1CHAR] summary cases=1 failures=1
exit： ラッパー回収済みのため exit code 未記録（gapとして明記）。
      summary 存在＋event log（Application ID 1000）に crash なし → clean exit。
```

## 3. ab／aa snapshot（4 pass 全同一値）

```text
ab0＝aa0＝ab1＝aa1：
gen=8 wid=8 seq=8 ord=0 eqb=0 cvb=0 sc=0 sat=1.0000
hr=0.5012 mu=1.0000 tr=1.0000 os=1 irl=1 irf=1
hash=-8368842959456362885 fade=0.0600
```

## 4. Case 判定（§4 規律）

- **Case A（取得可否）**: triple 非ゼロ（8／8／8）→ **R11-A observability PASS**。
  accessor は published world を観測できる。triple は別識別値として扱う。
- **Case B（mismatch・差分記録のみ）**:
  1. sc1 pass（ab1／aa1）の `sc=0` ≠ requested sc=1。
  2. `hr=0.5012` ≠ requested 1.0（0.0 dB）。
     `hr=0.5012` は engine default `inputHeadroomDb=-6.0f`
     （AudioEngine.h:2670／:5017・10^(-6/20)=0.501187…）と一致する。
     よって active world は vehicle の headroom 要求を反映していない。
  3. `fade=0.0600` ≠ 既定 0.030。由来は unknown（記録のみ）。
  4. os=1／cvb=0／eqb=0／sat=1.0 は requested と一致するが、
     default との区別は未検証のため「一致」ではなく「非矛盾」と記録する。
  5. irl=1／irf=1＋hash：IR-associated world（ファイル同定は IR ログ複合に委ねる）。
  6. 「publish failure」とは断定しない（指示どおり）。
- **Case C（zero）**: 非該当（non-zero）。
- **Case D（行欠落）**: 非該当（ab／aa 全出力）。
- gen／wid／seq の「8」一致は観測値の一致であり、numbering 等価の証拠にしない（R5 §6維持）。

## 5. R11-B（F first-case・R1 との接続）

- seqBefore=5・30 s flat・timeout は R1 と同一（再現）。
- 新規：pair2 T6 時点で seq=8。pair1 window 終了〜T6 の gap 内に 3 publishes が発生した。
  発行主体・内容は vehicle 非観測域（gap として明記。推測しない）。
- active（gen／wid／seq=8）は pair1 timeout 後の world であり、
  §4 の mismatch（sc・hr）を持つ。stale-world 問題の Case-B 型証拠として記録する。

## 6. R11-C（R vehicle・STOP）

- STOP 条件「F first-case 30s timeout」が発火したため **R を実行しない**。
  R build（adb 1 token）も実施しない。source は F vehicle のまま保持する。
- よって R execution log なし。F/R 比較なし。Pattern 分類なし。

## 7. STOP条件照合（他項目）

```text
R10 SHA mismatch：なし／production・CMake・JUCE diff：なし／
kP15・OFF 変更：なし／crash・heap・ASan：なし（event log clean）／
ab/aa 欠落：なし／POD corruption 兆候：なし（全field範囲内・4 pass一致）／
ownership漏洩兆候：なし（値のみ・handle非出力）／
generation/worldId/sequence不整合：なし（観測値として記録のみ）
```

## 8. 禁止事項遵守

P3-1-D・crash再分類・production fix・accessor改訂・P1/measurement/DFT/
normalization変更・compensation・correction なし。

## 9. 次gate申送り

1. accessor 観測能力は実証された（R11-A）。active 身元照合が実行可能になった。
2. pair1 strict-timeout の再発は確定的挙動（2/2 run）。第1ケース publish 確立には
   timeout 原因（R2-U 残余）の解消か、vehicle 前提の再設計が要る。60/120 s 延長は
   根拠なしとして行わない（R1-B 維持）。
3. pair2 の active（default-headroom・sc=0 on sc1）は Case-B 証拠として保持し、
   原因（limiter／crossfade／stale）に決め打ちしない。
4. R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
