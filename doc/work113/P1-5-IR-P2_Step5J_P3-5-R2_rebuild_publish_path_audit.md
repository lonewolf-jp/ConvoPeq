# P1-5-IR-P2 — Step 5-J / P3-5-R2: Rebuild/Publish Path Audit（source-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R2）
- **種別**: source-only audit。test-only instrumentation 追加なし（R1 の diag 除く）。
  build・実行・R vehicle・P3-1-D なし。
- **目的**: R1-B（30 s seq-flat＋backlog-flat）の停止点を
  admission／pressure／filter／defer／requestRebuild／worker／
  build／commit／publish のどこまで絞れるか確定し、R2-A〜E/U の1つに分類する。
- **観測事実（R1より）**: `before=5 seq=5 backlog=0` が30 s全点で静止。
  中途 advance・backlog>0 なし。backlog WARN なし（waitBacklogZero 通過）。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
source                       P3-5 F vehicle＋R1 diagnostic（revertなし・保持）
tmp/p35_F.exe                f472381550414f51（保持）
tmp/p35_R1.exe               a00d140d479184d4（保持）
production/CMake/JUCE        0 diff
```

## 2. Source baseline = latest ConvoPeq.md

監査対象（いずれも worktree 現行・ConvoPeq.md と一致を確認した範囲）:

```text
AudioEngine.RebuildDispatch.cpp（submitRebuildIntent :151-403・requestRebuild :462-808・worker :860-1422）
AudioEngine.Threading.cpp（pressure :49-60）
AudioEngine.h（sequence :1744-1747・backlog :3644-3647・armCrossfade :4154-4195）
AudioEngine.Commit.cpp（commit :202・:402）
AudioEngine.Processing.PrepareToPlay.cpp（sr/bs設定 :142-143・xfade reset :151-153）
ConvolverProcessor.h（transfer :1269-1288・progressive既定 :88）
RuntimeBuilder.cpp（build :454-509）
BuildErrorPolicy.h（failure分類）
ISRShutdown.cpp（tryAdmit :501-522）
```

## 3. submitRebuildIntent

経路（R1 確定を維持）：main＝message thread のため Structural intent は直接経路。
merge 規則（:201-218）は outstanding 空の第1ケースでは非該当が既定。
3 no-guard setter＋endBulk の intent は必ず発行される（P3-1-B §2-1維持）。

## 4. Admission

- `tryAdmit`（ISRShutdown.cpp:501-522）: `Open` 状態でのみ reservation 成功。
  fresh process（shutdown は run 末尾の `h.stop()` 時のみ）では Open であり、
  **AdmissionClosed は本 window について排除**する。
- reservation overflow（:510）は単発 intent 群では到達不能。

## 5. Pressure

- 入力（Threading.cpp:49-60）: `retirePressureAdmissionStrict_` atomic
  **または** HealthState==Critical。**backlog ではない**。
  よって backlog=0 は pressure 非該当の根拠にならない（指示どおり断定しない）。
- fresh process（retire idle・health nominal が既定）では成立理由が見当たらないが、
  harness 観測手段が存在しないため**完全排除はできない**（残余 U1 として記録）。

## 6. KindFilter

- filter 対象は kind None／Runtime のみ（:301-313）。Structural は通過する。
- 3 setter＋endBulk はいずれも Structural 系（P3-1-B §2-1）につき
  **KindFiltered は排除**する。

## 7. Defer

- MT 直接経路は sr>0・bs>0 を要求（:349-363）。欠落時は DeferredFinalizeAware（:365-374）。
- `prepareToPlay` が両 atomic を設定する（PrepareToPlay.cpp:142-143・h.start 経路）。
  よって sr/bs defer は**排除**する。
- MixedPhaseIntermediate 抑止（:563-586）は progressive upgrade 有効が条件だが
  既定 `enableProgressiveUpgrade=false`（ConvolverProcessor.h:88）のため
  **排除**する（test が変更しない）。

## 8. requestRebuild

- 到達時の処理（:596-808）：snapshot 凍結→重複判定→generation++→queue→
  `rebuildBacklog_=1`（:755）→ `rebuildCV.notify_all()`（:765）。
- 重複抑止 `blockedAsDuplicate`（:659-667）は pending task 存在＋全 snapshot 一致が条件。
  単独では seq-flat を説明しない（pending task 自体の publish があれば seq が動くため。
  下流 failure との複合としての記録に留める）。
- **訂正事項（P3-2／R1 の backlog 解釈の修正）**:
  `getPublicationBacklogCount()` は `runtimePublicationBridge_` の backlog であり
  （AudioEngine.h:3644-3647）、rebuild queue 深度（`rebuildBacklog_`）ではない。
  よって backlog-flat-0 は「worker が消費した」ことの証明にならない。
  証明するのは publication-stage に滞留がないことのみである。

## 9. Worker dispatch

- rebuild thread は `rebuildCV` 待機（:867-889）。wake 時に task 所有権を取得し、
  **`rebuildBacklog_=0` を即時クリアする（:921・build 前）**。
- CoordinatorLoop は harness initialize で起動（h.start コメント）。
  起動時の publish 群（before=5 の根拠）は worker＋commit 経路が生きていた証拠である。
- worker-dead 説：bridge-backlog では否定できない（§8 訂正）。
  ただし始動直後に5 publish 成功しているため、window 内での新規死亡を仮定する必要がある。
  排除も採用もしない（残余 U6）。

## 10. Build / Validation

- worker は build→rebuildIR→`validateWarmup`（:1244-1272）。
- failure 時の silent 継続（いずれも backlog 0・seq flat と両立）:
  warmup-fail retry スケジュール／Exhausted・NoRetry 後 `continue`（:1300-1339）、
  obsolete `continue`（:1257・:1351）、例外 catch（:1410-1418・release は DBG のみ）。
- 成功時は `[CONV_STATUS]`（writeToLog :1363・harness 非出力）→ commit へ（:1408）。

## 11. Commit / Publish

- `enqueuePublicationIntentForRuntimeCommit` → coordinator 消費 → sequence bump
  （Commit.cpp:402）。bridge滞留があれば backlog 非ゼロになるはずであり、
  30 s全点ゼロは **R2-E（commit 後 sequence 未更新・bridge 滞留）を排除**する
  （bridge counter の部分計数の仮定を明記した上での排除）。
- BuildErrorPolicy の failure path（BuildErrorPolicy.h）は backlog／sequence に
  反映されない設計であり（§10 の continue 群）、観測事実と矛盾しない。

## 12. R2 classification

```text
R2-A（admission/pressure/filter 抑止）:
  AdmissionClosed＝排除／KindFiltered＝排除／pressure＝残余U1（unobservable）
  → A単独では不成立
R2-B（defer 未達）: 排除（§7：sr/bs有効・progressive無効）
R2-C（worker 未起動）: 起動時5 publish が反証材料だが window 内死亡は排除不能 → 残余U6
R2-D（build/validation/commit 失敗）: warmup/obsolete/exception の silent 継続が
  観測と完全両立 → 残余U2/U3/U4（分離不能）
R2-E（commit 後 sequence 未更新）: bridge滞留ゼロにより排除
R2-U: ADOPTED（残余集合 U1/U2/U3/U4/U5/U6 を明示した上での U）
  U5＝30 s窓超過の低速build（wake後建築中。backlog語義（§8）により排除不能として残す。
  R1-A棄却（延長根拠なし）とは両立する：「延長すべき証拠がない」≠「低速の可能性ゼロ」）
```

R2-U は許容する（指示どおり無理に分類しない）。
ただし vague な U ではなく、上記排除表つきの U である。

## 13. Evidence gaps

- `REBUILD_TELEMETRY`／`diagLog`／`[CONV_STATUS]`／`[PUBLISH]` はいずれも
  writeToLog 系であり本 vehicle 非出力。U1〜U6 の分離には worker 側可視性が要る。
- harness 観測可能量は sequence＋bridge-backlog のみ（既存 API）。
  rebuild queue 深度・pressure・health・warmup 結果の proxy は存在しない。

## 14. Next gate

```text
P3-5-R2（R2-U・残余 U1-U6 列挙）
   ↓
分岐判断：
 (a) U解消に E3（production計装）を投じるか
 (b) publish依存を迂回する vehicle 再設計（strict第1要件の見直し等）か
   ↓
いずれも本 Step では実施しない。R実行・比較・P3-1-D は保留維持。
```

- 30 s 延長・sleep/post-settle 変更・logger/accessor 追加なし。
- F vehicle source（＋diag）保持・revert なし。
- §7解釈制約維持。H-B 因果主張なし。
