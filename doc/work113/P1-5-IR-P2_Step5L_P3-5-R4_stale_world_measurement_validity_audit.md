# P1-5-IR-P2 — Step 5-L / P3-5-R4: Stale-World Measurement Validity Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R4）
- **種別**: source＋既存ログの read-only 再評価。新規取得・修正・実行なし。
- **目的**: R3-D 確定後に「各測定が何を測定していたか」を再分類する。
  原因帰属はしない。production 修正・R vehicle・F/R 比較・P3-1-D は実施しない。
- **最重要訂正（本監査の主成果）**: R3-D（warmup-fail 決定的機構）は
  §2 の hinge 再検証により**成立しない**。下記に訂正記録を残し、
  以降の分類は訂正後の機構理解に基づく。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
F vehicle＋R1 diagnostic     保持（f4723815…／a00d140d…）
production/CMake/JUCE        0 diff
```

## 2. R3-D confirmed mechanism（訂正：不成立）

R3（Step5K §2）は「転送IR→loaded&&!finalized→WarmupFailed決定的」とした。
hinge 再検証の結果：

```text
transferIRStateFrom → IR buffer のみ（irFinalized 非接触）          …維持
applyBuildSnapshot → irLength 等メタのみ（irFinalized 非接触）      …維持
rebuildAllIRsSynchronous → LoaderThread.runSynchronously
  → applyNewState → executePendingCommit（同期）
  → Phase 3 engine swap ＋ Phase 4 Publish（irFinalized=true :835）
```

すなわち worker が task を消費し build が完走すれば、新coreは
`loaded && finalized` となり `validateWarmup` は **None を返す**。
R3-D の「決定的 WarmupFailed」は成立しない。R3 の停止条件1の根拠は崩れる。

残る確定事実（R3-D から継承できる部分）:
`irFinalized=true` の全src書込みは load pipeline の2件のみ
（LoadPipeline.cpp:579／:835）。transfer／rebuild経路に finalize はないが、
worker 経路の executePendingCommit が finalize する。
よって publish 可否は worker 到達＋build 完走＋commit 承認の各関門に依存し、
warmup 単独では決まらない。

## 3. Definition of publish evidence（定義固定）

```text
publish evidence ＝ publication sequence の前進（commit の単調増加）の観測、
  または commit 経路の成功を示す既存ログ。
ログ行の存在（[P1CHAR] p15ir 行）は publish evidence ではない。
[IR_RATE_GEN]／[L0_WRITE] 等は loader／engine 側の出力であり world commit を意味しない。
```

この定義の下で、sequence を出さない A3／Forward の P1CHAR 行は
それ自体では P0 にも P1 にもならない。

## 4. A3 p15ir case classification（P0/P1/P2）

| section | publish可否（機構） | ログ上の sequence | 分類 |
| --- | --- | --- | --- |
| G preset（IR未load） | 可能（warmup素通りの余地） | なし | **P2** |
| H p15eq（IR未load） | 可能（同上） | なし | **P2** |
| I p15ir（engine IR有） | 可能（§2訂正後。warmupは通過しうる） | なし | **P2** |
| J/K/K2/L/M | 可能（同上） | なし | **P2** |

- **P0（publish済みと証明可能）**: A3 全行で空集合。sequence を出す行が存在しない。
- **P1（publishされていないことが証明される）**: A3 ログ単独では該当なし。
  例外は F-vehicle 初ケースの window 限定 P1（R1 観測：30 s commit ゼロ。
  post-window 数秒の gap 付き。§5）。
- すなわち A3 の publish 状態は davranış ではなく **P2（判定不能）**である。
  R3 申送りの「全p15ir行 stale」は P1 ではなく Possible に格下げする（§7）。

## 5. g0/os1/-20, -6 target pair reconstruction

### am=-20（A3 L235／Forward L235／F-vehicle pair1）

```text
seqBefore        A3/Forward： unknown（ログに sequence なし）
                 F-vehicle： 5（pubdiag t=0 before=5・R1）
publish sequence A3/Forward： unknown（P2）
                 F-vehicle： 5 のまま30 s（commit ゼロ・P1 window限定）
active world generation： unknown（全vehicle・P2）
target snapshot  g0／os1／convBypass=false／sat=1.0（要求・source確定）
measurement      gainDb=-14.5035（3 run一致・決定的測定値として有効）
```

### am=-6（A3 L356／Forward L356／F-vehicle pair2）

```text
seqBefore／publish／active world： unknown（P2。pair2はwait自体なし・継承分岐）
target snapshot  同上（ampDbのみ-6）
measurement      gainDb=-13.0276（3 run一致・決定的測定値として有効）
```

- merge-no-dispatch 機構（P3-3 §3-3）は R3-D と独立に成立する：
  am-6 は am-20 と同一パラメータのため、自前の dispatch が merge されうる。
  この場合 am-6 の world は am-20 の world の継承である（redispatch 分岐では
  同一内容の新 world。いずれも source 上の既定動作・タイミング依存）。
- 測定値自体（gainDb 2行・Δ=+1.4759）は vehicle・run を跨いで決定的であり、
  有効な測定記録として扱ってよい。問題は active world の同定のみである。

## 6. Active-world / requested-world distinction

- ログ行の存在と要求 world の publish を同一視しない（§3定義）。
- active world の候補集合は「I 以前（G/H/bootstrap）の最終 publish」から
  「I 以降の各 publish」までの広がりを持ち、現 evidence では一点に定まらない。
- ただし要求 snapshot（g0/os1）は source 確定しており、
  active が要求と一致したことの証明も不一致の証明もない（P2）。

## 7. P3-2 M3 re-evaluation（事実／仮説の分離）

```text
確定：
・+1.4759 dB は測定式・窓・正規化だけでは説明できない（P3-2 M1棄却・維持）。
・R1 window では commit ゼロ（F-vehicle 初ケース・gap付きP1）。
・ログ行の存在は publish の証拠にならない（§3）。
・merge-no-dispatch による world 継承は source 上の既定動作（P3-3 §3-3・維持）。

未確定：
・観測された +1.4759 dB が stale world によって発生した（Possible に格下げ）。
・+1.4759 dB の直接原因が limiter／crossfade／convolver tail 等のどれか。
・A3 各行の publish 有無・active world 同定（P2）。
```

M3（測定妥当／runtime-state候補）は維持される。
M3 の根拠は R3-D ではなく P3-2 の M1棄却＋R1 の commit ゼロ観測である。

## 8. P3-1-C crash separation（変更なし）

- R3-D：不成立に訂正（§2）。warmup-fail 説は crash 説明にも使えない。
- P3-1-C の `0xc0000005` と rebuild／publish 経路の直接因果は未証明のまま。
- crash attribution は C4 のまま変更しない。

## 9. Evidence-grade classification

```text
Confirmed：
・R1 window の commit ゼロ（F-vehicle 初ケース・post-window gap付き）
・測定式・窓・正規化だけではΔを説明できない（M1棄却）
・merge-no-dispatch の既定動作（am-6 の world 継承構造）
・log存在≠publish（定義）
・R3-D 不成立（executePendingCommit が finalize する）
Probable：
・A3／Forward／F-am6 の gainDb 決定性（同一条件で同一値。
  clipEngagement の metric-level jitter を伴う）
Possible：
・全p15ir行の stale-world 測定（R3申送りから格下げ）
・limiter／crossfade／tail の各原因候補
・post-window の late publish（gap 未観測域）
Unknown：
・A3 各行の publish 有無・active world generation
・第1ケース task の worker 到達有無（queue／consume／drop の内訳）
```

「rebuild failed ≠ 必ず stale world 測定」の規律を維持する。
ケース間の別 publish があれば active はさらに区別される（現 evidence では relativity のみ）。

## 10. What remains unknown（未観測量の明示）

- per-row sequence（A3／Forward の P1CHAR 行に sequence なし）
- worker 到達有無（REBUILD_TELEMETRY／diagLog／[CONV_STATUS] は writeToLog 系で非出力）
- rebuild queue 深度・pressure・health の直接読値（harness proxy なし）
- post-window 数秒の commit 有無（R1 diag は第1ケース window のみ）

## 11. Gate decision

```text
A（active world まで確定）： NO
B（requested rebuild was not published まで確定）：
  F-vehicle 初ケースの window 限定で YES（post-window gapを明記）。
  A3 行については NO（P2）。
C（stale-world attribution に追加証拠が必要）： YES（Bの限定付き確定と併存）
```

B と C の併存をもって停止する。C は失敗ではない：
「R3-D は不成立、測定値の原因帰属は未確定」として次実験設計へ進む。
次順序（R4指示の固定順序を維持）：測定値有効性・active world確定 → 最小修正方針の
source audit → 修正承認判断 → 修正→build→gate → 新F vehicle → R vehicle →
F/R比較 → P3-1-D。修正には入らない。
