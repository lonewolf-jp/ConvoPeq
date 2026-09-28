# STG-9-D1 Repair Contract Correction（2026-09-28・R2）

> 本書は `P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-AUDIT_20260928.md`（以下「Audit」）の
> I-2 の誤りを記録し、訂正後の契約を確定する。
> **過去文書は改変しない。** Audit 文書自体は非変更のまま残し、本書が訂正の正本となる。

---

## 1. 誤った契約（Audit §4.2 I-2）

Audit は entry の消滅を 3 通り（**再 push** / 成功 consume / drop）とし、
「再 push は member list に残る。counter 不変。**消滅でない**」と書いた。

すなわち Audit の主張は:

```text
retry / re-push は counter 不変
```

## 2. なぜ誤っていたか

Audit は `pending.swap(pendingReclaimHandles_)`（`AudioEngine.Retire.cpp:96`）の意味を
見落としていた。swap の時点で old entry は **member list から除去されている**。
その後の `requestReclaim` の deferred 戻りは reclaimNormal が **新しい `+1`**
（`Coordinator.cpp:705`）を伴い、`push_back` は **新しい entry** を登録する。

すなわち retry は「同一 entry の継続」ではなく:

```text
old pending entry
    ↓  swap で member list から除去
new deferred attempt（新 +1）
    ↓
new pending entry を push
```

という**置換**である。置換される old entry の `+1` を解放しない限り、
**entry 数は不変なのに counter が +1 ずつ増加する**。

Audit の I-2 の証明はこの遷移を数えていなかったため不完全であった。

## 3. T-04 がどう反証したか

T-04 row 0（`{ defer, drain, drain, drain }` = retry のみ）は R1 実装の初手で:

```text
FAIL: T-04 row 0 step 0 1:1 broken (pending=2 counter=825)
```

を出した。停止済み engine の baseline は `pending=1, counter≈830` であり、
R1 の retry は drain pass ごとに +1 を積み上げていた（pre-RC-1 では caller-side pre-check が
`requestReclaim` を呼ばなかったため +1 が発生せず counter=0 のままだった。
RC-1 の pre-check 削除がこの流出を露出させた）。

T-04 の oracle（Owner R2 §5「retry = 新しい +1 としてはいけない」）が
この蓄積を直接検出した。**T-04 がなければ R1 のまま PASS していた**。

## 4. 修正後の契約（R2）

```text
pending entry 1 件は、その residency に対応する
reclaimInFlightCount_ の +1 を 1 個持つ。

retry は「同一 entry の継続」ではなく、

    old pending entry
        ↓
    new deferred attempt
        ↓
    new pending entry

という置換である。

したがって retry では、

    旧 entry の +1 → onReclaimEnd() で解放
    新 deferred の +1 → onReclaimBegin() が生成

となる。

最終的な pending entry 数と counter の対応は 1:1 に戻る。
```

### 4.1 最終 accounting contract（acceptance criterion）

```text
NEW deferred entry:
    reclaimNormal → onReclaimBegin(+1)
    push entry

RETRY:
    old entry is replaced
    new deferred → onReclaimBegin(+1)
    new entry pushed
    old entry → onReclaimEnd(-1)

SUCCESS:
    reclaimNormal → onReclaimEnd(-1)
    entry consumed

TERMINAL DROP:
    entry dropped
    onReclaimEnd(-1)
```

目標: `pending entry ⟷ one outstanding reclaim counter unit` の 1:1。

### 4.2 順序制約

```text
push_back(new entry)
        ↓
onReclaimEnd(old entry ownership)
```

`onReclaimEnd()` を `push_back()` より前に置いてはいけない。
理由: swap window 中に `pendingReclaimHandles_` が空になっている時間帯があり、
その間の drain predicate を壊さないため。

### 4.3 production 差分（R2）

`src/audioengine/AudioEngine.Retire.cpp` の retry branch に **1 行のみ**追加:

```cpp
if (!runtimePublicationBridge_.requestReclaim(handle, dspHandleRuntime_, *m_retireRouter))
{
    std::lock_guard<std::mutex> lock(pendingReclaimHandlesMutex_);
    pendingReclaimHandles_.push_back(
        convo::isr::ReclaimIdentity{ handle, retireEpoch });
    // ★ STG-9-D1 / R2: 置換された old entry の +1 を解放（retry replacement accounting）。
    runtimePublicationBridge_.onReclaimEnd();
}
```

RC-1 の既存修正（pre-check 削除 2 か所 + terminal drop の -1）は維持。
Coordinator・`reclaimNormal`・`reclaimShutdownQuiescent`・`isFullyDrained()` は無変更。

## 5. retry replacement accounting（詳細）

retry 1 回あたり:

```text
counter:
    old +1（swap で除去された E_old が持つ）
    new +1（reclaimNormal の deferred が生成）
    old -1（R2 で追加した onReclaimEnd）
    ----------------
    差分 = +0
```

重要: retry 自体を新しい logical recovery obligation として扱わない。
`ReclaimIdentity` の新規 obligation を生成しない（`retireSequence` の更新は
INV-FIFO-1 secondary の ordering 用であり obligation 生成ではない）。

## 6. T-03 oracle correction

旧 T-03（R1）の前提「停止済み engine は完全 drain 済み = `isFullyDrained()==true`」は撤回する。
実測で停止済み engine には `pendingReclaimHandles_` に **1 件が正当に残留**しており、
pre-RC-1 でも同じである。この 1 件は `pendingReclaimEmpty == false` として
`isFullyDrained()` を false にする正当な判定である。

新 T-03 は baseline-relative:

```text
baseline(P0/C0/D0=isFullyDrained)
    ↓
test operation（defer → retry×2 → terminal resolution）
    ↓
test-induced reclaim activity（delta +1/+1 → retry で維持 → 0/0）
    ↓
terminal resolution
    ↓
baseline へ復帰（pending == P0 && counter == C0 && isFullyDrained == D0）
```

- `P0 != 0` / `C0 != 0` でも FAIL ではない。
- `isFullyDrained() == true` を絶対条件にしない。代わりに test 前後で drain 状態が
  不変であること（test が drain 悪化を持ち込まない）を検証する。
- drain-completion case: terminal resolution 後の追加 drain が fixed point であること
  （accounting 状態が変化しない = drain が終端する）を検証する。

## 7. 新しい accounting invariant（INV-1 / INV-2）

旧 INV-1（絶対値ゼロ）だけでは不十分。以下を test contract とする。

### INV-1（baseline-relative）

```text
pendingReclaimHandles_.empty()（test-induced）
    =>
reclaimInFlightCount_ == baseline counter
```

実装形: scope 終了時に `pending == P0 && counter == C0`。
全ステップで test-induced pending delta と counter delta の 1:1 を検査する。

### INV-2（retry 置換）

retry による置換は `Δcounter == 0` であること。すなわち:

```text
old pending
→ deferred retry
→ new pending
```

で counter が増加し続けないこと。今回の root defect は **INV-2 違反**として明示する。
T-04 は各 drain の前後で counter を比較し、terminal 操作を伴わない drain では
変化がないことを assert する。

## 8. counter の絶対値について

R1 報告の `counter = 822〜844` は、test process が生成した deferred accounting の蓄積
（R1 の retry 欠落による）を含む。修正後に `counter == 0` へ戻ることは要求しない。
必要なのは:

```text
修正前 baseline
        ↓
test operation
        ↓
test-induced accounting delta
        ↓
0
```

すなわち `C_after - C_before == 0` を retry-only path の主要 oracle とする。
T-01〜T-04 はすべてこの baseline-relative oracle で構成する。

## 9. Audit 文書の位置づけ

- Audit（`..._REPAIR-CONTRACT-AUDIT_20260928.md`）は**非変更**のまま残す。
- 本書が I-2 の訂正の正本である。
- Audit の他の確定事項（counter 意味論 §3 / sink 列挙 §4.4 / shutdown §5 /
  `isFullyDrained()` authority §6 / balanced caller §8 / R-3 不要 §9.1 / R-4 分離 §9.2）は
  **有効のまま**であり、R2 は I-2 の一点のみを訂正する。
