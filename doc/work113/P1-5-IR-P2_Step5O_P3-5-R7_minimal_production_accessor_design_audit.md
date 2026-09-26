# P1-5-IR-P2 — Step 5-O / P3-5-R7: Minimal Production Accessor Design Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R7）
- **種別**: read-only／設計監査。production変更・実装・build・実行なし。
- **目的**: R6-C の production accessibility を最小1点の read-only accessor で
  解決できるか確定する（R7-A/B/C）。API名・型・実装位置は決めない。
- **結論**: **R7-A**（§8）。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
F vehicle＋R1 diagnostic     保持（f4723815…／a00d140d…）
production/CMake/JUCE        0 diff
```

## 2. 案A：messageThreadRcuReader 直接公開（非推奨・問題点の監査）

経路：private reader の public 化 → test側で `RuntimeReadHandle` 構築 →
`getRuntimeWorldFromReadHandle()`。問題点（採用しない理由）：

```text
a. RCUReader（epoch domain 結合物）の露出。domain／epoch の内部実装が
   test-only を超えて外部 API 面になる。
b. RuntimeReadHandle の lifetime 契約が外部へ漏れる。
   将来の利用者が capture 全区間保持（R6 §4 禁止形）を行える余地を増やす。
   epoch pin → retire／reclaim 停滞 → pressure 誘発の経路を開く。
c. 必要最小限の超過。目的は snapshot 値の読取りであり reader 自体は不要。
d. Diagnostic 副作用（observe 系 counter 更新・後方観測時の rollback arm）が
   利用側の持越し方に依存して増幅しうる。
```

**単なる reader accessor の公開は必要最小限ではない**（指示どおり明記）。

## 3. 案B：read-only active snapshot accessor（本命・成立）

概念形状（名・型・位置は未定）：

```text
ownership を返さない（RuntimeReadHandle／RuntimeWorld*／&／shared を出さない）
RCU handle を内部で短時間取得 → 必要 field のみ値コピー → handle 破棄 → 値返却
publish／retire／reclaim／delete／crossfade／rebuild／policy を行わない
sleep／wait／capture／logger を handle lifetime 内に置かない
```

内部実装は既存 `makeRuntimeReadHandle`（Message channel）＋既存 field 読取りの
合成であり、新規 runtime 機構を要しない。
Observer は観測のみ・所有権を持たない（ISR Bridge 原則と整合）。

## 4. Snapshot field 最小性（必要／便宜の分類）

```text
【必要：Identity】generation／worldId／publication.sequenceId
  → 三相関 key（R5 §6）。単独表記禁止・分離記録（R6 §7維持）。
【必要：Routing】convBypassed／eqBypassed／processingOrder
  → H/I・sc 間の弁別子。bypass mirror では不足（committed os 等を欠く）。
【必要：Automation】softClipEnabled／saturationAmount／headroom／makeup／trim
  → softClip・sat は sc 弁別子。gains は staging drift の negative control。
    const 性の仮定ではなく観測で担保するために必要（便宜ではない）。
【必要：DSP projection】oversamplingFactor／irLoaded／irFinalized／structuralHash
  → osFactor は最強弁別子。irLoaded／irFinalized＋structuralHash は
    IR-association proxy（RuntimeBuilder.cpp:240-253 の world-associated 投影）。
    IRファイル名は world 外のため既存 [IR_*] geometry ログと複合する。
【必要：Timing】fadeTimeSec → crossfade 窓計算の入力。
【除外（便宜）】sampleRate 系（dspProjection／timing ともに全条件 48000 固定。
  fade 計算は fadeTimeSec［秒］で閉じる）。baseLatencySamples（同定に不要）。
```

便利追加なし。上記以外は target identity への必要性を source で示せた場合のみ。

## 5. RCU安全性（設計契約）

```text
make handle → copy required fields → destroy handle → return value
```

- handle の外部持出し・区間保持を型で不可能にする（値返却のみ）。
- 非const性：make 側が token manage のため、accessor の const 修飾は
  実装時検証事項とする（不可なら非const observer として明記。audio callback と
  同一 pattern のため RT 安全性の懸念とはしない）。
- audio thread が毎 callback 同一 read を行う設計であり、test 側の数回読取りは
  無視可能な負荷である（per-callback precedent による裏付け）。
- Diagnostic 副作用（observe 系 counter 更新）は audio 既定動作と同一であり、
  許容する。後方観測時の rollback arm は engine 病理時のみ発火する設計であり、
  正常 run では不発（発火時はそれ自体が evidence となる）。

## 6. accessor の副作用監査（全 NO の確認）

```text
publish／retire／reclaim／delete／crossfade変更／rebuild／policy変更： NO
 （read path のみ。commit／retire／dispatch 系を呼ばない）
atomic write： 診断 counter 更新のみ（§5。分岐駆動に使わない Diagnostic 権威）
mutex取得： 原則 NO（RCU token acquire は lock-free。audio precedent により裏付け。
  実装時検証事項として残す）
allocation： NO（固定 POD 値返却。heap 確保なし）
```

案A との対比：案B は利用側が境界を破れない（値しか渡らない）ため、
案A の懸念（b・d）は構造的に解消される。

## 7. test-only 利用可能性（T0–T9 不変）

```text
P1PolyphaseGainCharacterization → AudioEngine → active snapshot accessor
```

- T6／T9 の scoped read に本 accessor を用いる。T0–T9 構造の変更なし。
- 既存 `[P1CHAR]` 行への field 追加は test-log 変更として次gate承認事項
  （新規 prefix・production logger なし）。
- F／R 対称：同一 schema・同一 accessor を両 order で使用する。

## 8. Publish証明との関係（混同禁止の維持）

- sequence／generation／worldId／rebuildRequestGeneration の4者分離を維持する。
  request 系（`requestRebuild(kind)` 再投入：RebuildDispatch.cpp:476-479）と
  build snapshot 系（:678-684 の seal）は別管理であり、accessor は後者の
  world-associated 値を読む。request 側の値で world 同一性を主張しない。
- `seqAfter > seqBefore` のみでの target 断定を禁止する（R6 §7 維持）。
  証拠は `seq delta ＋ active snapshot ＋ requested snapshot` の三点保持とする。

## 9. R7-A/B/C classification

```text
R7-A（Minimal accessor accepted）: ADOPTED
  既存 read-handle 内部利用＋値 snapshot 返却＋ownership／RCU 非露出で
  §4 の field set を取得できる（§3–§7）。
  → 次に test-only accessor 実装設計へ進む（実装自体は次gate承認事項）。
R7-B（further reduction 可能）: REJECTED（§4 で最小化済みのため）
R7-C（過剰露出）: REJECTED（案B は値のみ・露出なしのため）
```

## 10. STOP（実装せず停止）

- production source 変更・accessor 実装・build・F/R 実行・比較・P3-1-D・
  limiter 帰属・stale 帰属はすべて未実施（指示どおり）。
- R6-C の「即実装しない」規律を維持し、本 R7 で設計確定のみ行った。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
