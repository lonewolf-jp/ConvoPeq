# P1-5-IR-P2 — Step 5-K / P3-5-R3: Worker Failure Residual Audit（source-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R3）
- **種別**: source-only／read-only。instrumentation追加・build・実行なし。
- **目的**: R2-U の U1〜U6 を追加計装なしで縮小する。
- **結論**: **停止条件1に到達（単一原因を source 上で確定）**。
  R3-D（warmup failure）が決定的機構である。下記に全排除表を添える。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
F vehicle＋R1 diagnostic     保持（revertなし）
production/CMake/JUCE        0 diff
```

## 2. 決定的機構（先に結論）

新 DSPCore の convolver における `irFinalized` の生涯（source-exhaustive）：

```text
既定 false（ConvolverProcessor.h:986）
transferIRStateFrom → updateIRState：IR buffer のみ複写。irFinalized 非接触
  （Lifecycle.cpp:31-63 に irFinalized 書込みなし）
rebuildAllIRsSynchronous：同期 LoaderThread 再構築。irFinalized 書込みなし
  （Rebuild.cpp:44-108 全文確認・書込みなし）
irFinalized=true の全src書込み：LoadPipeline.cpp:579／:835 の2件のみ。
  いずれも ENGINE 自身の load pipeline final commit であり、
  新 DSPCore の convolver に対する書込みではない（exhaustive grep 確定）。
```

よって IR転送後の新coreは `isIRLoaded()==true ＋ isIRFinalized()==false` が確定し、
`validateWarmup`（RuntimeBuilder.cpp:539-547）：

```cpp
if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
    return BuildError::WarmupFailed;
return BuildError::None;
```

により **WarmupFailed が決定的**となる。以降は retry ×3（backoff 10→80 ms上限・
BuildErrorPolicy.h:97-109）→ Exhausted／NoRetry → 無言 `continue`
（RebuildDispatch.cpp:1300-1339）→ commit・publish なし。
backlog（bridge）は 0 のまま、sequence 不変。**R1 観測（seq-flat＋backlog-flat）と完全一致**し、
追加仮定を要しない。

適用確認（第1ケース g0/os1/sc0/am=-20）:
engine convolver は ensureTestIr で g0 finalized 済み → transfer 成立（buffer有）→
新core loaded=true／finalized=false → 上記経路。**成立**。

## 3. U1 pressure / health（排除）

- pressure 入力は `retirePressureAdmissionStrict_`／Health-Critical のみ
  （Threading.cpp:49-60）。backlog ではない。
- 全 set-path は fault／recovery／retire-critical 前提
  （Retire.cpp:272,414／Timer.cpp:1657,1691,1829,1879）。
  本 run は fault マーカー 0・clean exit・pair2 正常であり、前提が成立しない。
- health Critical は retire／publication／overflow／reader／age の Error 状態由来
  （RuntimeHealthMonitor.cpp:380-418）。同上、前提なし。
- 判定：**U1 eliminated**。残余は「無音transient」の形式的余地のみで positive support なし。
  （「fresh processだから」ではなく set-path 前提分析による排除である。）

## 4. U6 worker未消費（排除・スレッド生存は source 確定）

- worker thread の永続退出 path は `rebuildThreadShouldExit`（:891）と
  shutdown（:892-898）のみ。window 内に shutdown はない（`h.stop()` は run 末尾）。
  よって window 内の生存は source 確定（publish実績推論を使わない）。
- `rebuildCV` predicate（:884-889）・notify（:765）は正規 pattern（lost wakeup なし）。
- `rebuildBacklog_` の全 write site は :755（=1・queue時）と :921（=0・wake時）の2件のみ。
  worker 以外からの 0 書込みは存在しない。
- 30 s busy-stuck 説：stuck しうる先行 task が存在しない
  （load は finalize 済み＝完了。§2 機構がなければ初回 task が唯一）。
- 判定：**U6 eliminated**（standalone 原因として）。

## 5. U5 build > 30 s（排除・単独原因として）

- g0/os1 case の build 内 loop は全 bounded（1-sample IR・小 partition・固定grid EQ解析）。
  warmup retry も backoff 上限 80 ms・3回（BuildErrorPolicy.h:97-109）。
- 単一 build が 30 s を超える path は source 上成立しない。
  retry-storm 累積も Exhausted（§2）で数秒以内に沈黙する。
- 判定：**U5 eliminated**（単独原因として）。

## 6. U2 warmup failure（CONFIRMED・表）

| failure | retry? | count | Exhausted? | NoRetry? | 行き先 |
| --- | --- | --- | --- | --- | --- |
| WarmupFailed（loaded＋!finalized） | WarmupRetryAction により Schedule（scheduler経由再投入） | max 3（`kMaxWarmupConsecutiveRetries`） | → terminal diag 1回後 silent | disposition により即 silent も有り | **いずれも commit せず `continue`** |
| その他 BuildError | 分類表に従う（BuildErrorPolicy.h:45-56） | 同上ドメイン | 同上 | 同上 | commit なし |

- `failure → commit` する path は存在しない（commit は warmup 通過後のみ :1408）。
- 本ケースでは transfer-IR により WarmupFailed が**毎回 deterministic**に成立する
  （§2）。retry 再投入も同一転送を受けるため同結果を反復し、Exhausted で沈黙する。

## 7. U3 obsolete（排除）

- predicate：`isRebuildObsolete(task.generation)`（:1178-1180）＝
  より新しい generation の要求／commit との比較。
- window 内の newer-generation 源は存在しない（自 intent 群は同一 task・
  loader 静止・retry/recovery 未発火・fault なし）。
- 判定：**U3 eliminated**。

## 8. U4 exception（形式的残余として記録・主因としない）

- worker の catch（:1410-1418）は loop 継続（break／return／throw なし）。
  release では DBG のみで沈黙する。thread は生存する（§4）。
- build／rebuildIR 中の throw は warmup 評価前に drop され、同観測と両立する。
  ただし positive support はなく、§2 の deterministic 機構が追加仮定なしで
  観測を完全に説明する。よって **R3-F は形式的残余**とし、主因としない。

## 9. R3 分類

```text
R3-A  pressure/health confirmed suppression： eliminated（§3）
R3-B  worker-not-consumed： eliminated（§4）
R3-C  slow build： eliminated（§5）
R3-D  warmup failure： CONFIRMED（§2・§6。停止条件1に到達）
R3-E  obsolete： eliminated（§7）
R3-F  worker exception / termination： 形式的残余（§8。support なし）
R3-U  residual ambiguity： なし（R3-F の形式的余地を除き空）
```

## 10. 停止条件の充足

**停止条件1（単一原因まで source 上で確定）に到達**したためここで停止する。
条件2（分離不能）・条件3（構造的矛盾発見）には該当しない。
E3 承認判断は不要となった（計装なしで確定したため）。

## 11. 次gate・申送り

- R vehicle・F/R比較・P3-1-D は引き続き保留。
- 申送り（帰属判断ではなく事実として）：
  本機構により、IR-load 後の DSPCore rebuild は publish 不能が既定動作となる。
  すなわち A3 を含む全 p15ir 行は stale world 上の決定的測定だった可能性が高く、
  M3 の runtime-state 系候補と整合する。帰属の再評価は次gateの判断事項とする。
- 修正（production 変更）は本 Step の範囲外。実施しない。
- §7解釈制約維持。H-B 因果主張なし。
