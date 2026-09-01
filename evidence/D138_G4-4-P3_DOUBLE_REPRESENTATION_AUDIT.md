# D138 — G-4.4-P3 Double Representation / Delivery Uniqueness Audit (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only audit（P3-0〜P3-2）+ 修復契約固定（P3-3）。**Production source changes: 0. Test source changes: 0.**
**基準:** `ConvoPeq.md Generated: 2026-08-31 07:12:57`（内容一致検証済み: `G-4.4-P2`=15 / `consumeRedriveWake`=9 / `G-4.4-P1`=2、全ソース mtime ≤ 01:23:30 < 生成時刻）。ツリー = P2 完了状態（5 ファイル変更、P2 以降無変更）。
**Build 状態:** 現ツリーは `evidence/g44p2_ctest.log`（Debug 40/40 / Release 40/40、exit 0×2）で実証済み — P2 以降ソース無変更のため再実行不要（G-4.3-T-R 教訓: build 検証は実測 evidence で担保）。

---

## P3-0 — 状態遷移の完全追跡と順序確定

### 「durable slot が空になる時点」の唯一化
`pendingRecoveryAdmission_.state → NoAdmission` は次の 2 経路のみ:
- `settlePendingRecoveryAdmission(false)`（cpp:1246-1247）— Builder の build 成功（RebuildDispatch:1133）/ RECOVERY-6 discard（:1075）。**RebuildThread**。
- `discardPendingRecoveryAdmission`（cpp:1224-1230）— shutdown drain（RebuildDispatch:818）。

`settle(true)`（cpp:1241-1242）は **空にしない**（Building→DurablePending の再武装 — lease 設計）。

### 「delivery: Durable→None」の発生点
- `markTransientFailure`（cpp:1078）— **呼び出し元スレッド**（Builder recovery loop: RebuildDispatch:1006/1033（P1）/1091/1115、Orchestrator trySubmitImpl:311/401）。
- submit defer（cpp:993、CoordinatorLoop）— これは「付与失敗時の None 維持」であり別事象。

### 順序の決着（durable failure 経路、RebuildDispatch:1086-1091）
```
take(O): DurablePending→Building（cpp:1208）
build 失敗
settle(true): Building→DurablePending（cpp:1242）   ← slot は O を保持したまま
markTransientFailure(O): delivery Durable→None（cpp:1078）  ← 住処は有るのに None
```
**2 事象はデカップリングしている**: delivery=None は「durable slot が空」を意味せず、実際 slot は同一 oblId を保持し続ける。これが D136-B の根因。

### durable busy 時の redrive（P3 の中心、cpp:1157-1180 現行）
```
redrive(O): Live ∧ delivery==None（1140-1143）
  → state==NoAdmission のときのみ durable 付着（1158）
  → そうでなければ **保持者（oblId）を見ずに** transport fallback（1174-1177）
```
`O1=Durable ∧ O2=None → redrive(O2) → O2=Transport` は正常（C13）。問題は **busy の保持者が O 自身**のケースのみ（P3-2 反例）。

## P3-1 — delivery writer 全数表（現行行番号）

| # | 場所 | 値 | precondition | thread | container 実体操作 | rollback | 帰属 |
|---|---|---|---|---|---|---|---|
| W1 | h:402 tryInsert | None | 新規/再利用 slot | CoordinatorLoop | table slot 初期化 | n/a | admission |
| W2 | cpp:978 submit | Transport | push 成功 | CoordinatorLoop | queue += intent(O)（977） | push 失敗時は 983 で counter 相殺・991 以降へ | coalesce/NEW 共通後段 |
| W3 | cpp:993 submit | None | durable busy ∧ **異 oblId** | CoordinatorLoop | なし（defer） | n/a | durable fallback |
| W4 | cpp:1008 submit | Durable | durable free ∨ **同 oblId 上書き**（991 ガード通過） | CoordinatorLoop | slot := O（998-1006）+ predicate release（1007） | n/a | durable fallback |
| W5 | cpp:1078 markTransientFailure | None | Live | **caller thread（RebuildThread / Orchestrator 経路）** | なし（durable slot は保持継続し得る） | n/a | retry adjudication |
| W6 | cpp:1169 redrive | Durable | state==NoAdmission | CoordinatorLoop | slot := O（1159-1167）+ predicate release（1168）+ P2 latch（1170） | n/a | redrive |
| W7 | cpp:1177 redrive | Transport | push 成功（**busy 保持者不問**） | CoordinatorLoop | queue += intent(O)（1176）+ P2 latch（1178） | push 失敗 1181 相殺・None 維持 | redrive |

### 「delivery 変更と container 挿入/削除は atomic な意味を持つか」→ **持たない**
1. **順序は常に container 操作 → delivery 書込**（push 977→978、slot 書込 998-1006→1008、1159-1167→1169、push 1176→1177）。逆順（delivery 先行）は存在しない — 単一スレッド視点では「delivery=X なら container 実体あり」が成立する**ように見える**。
2. **しかし pop は delivery を消さない**（popRecoveryRequest cpp:1250-1263 は counter −1 のみ）→ `delivery=Transport ⇏ queue に実在`。「is (or was)」意味論（h:320）で意図的。
3. **W5 が決定的**: settle(true) 済み（slot=O 保持）の後に delivery=None にする — **delivery=None ⇏ 住処なし** の窓を意図的に生成（P-B stranded repair の副作用）。
4. **所有権違反（新規確定事項）**: h:324「CoordinatorLoop-only field (written solely on the producer thread) — non-atomic by design」は **D105-R18 以降偽**。W5 は RebuildThread（Builder loop）から同一 plain フィールドを書く。W6/W7（CoordinatorLoop）と W5（RebuildThread）の並行 read/write は形式上のデータ競合 — **P4（memory-order）へ回す**（P3-4 禁止事項: memory-order 変更）。ただし下記の反例は**この競合を必要としない**（逐次実行で成立）。

## P3-2 — Double representation 反例（現行行番号・逐次実行で成立）→ **欠陥確定**

```text
t0  O = Live, delivery = Durable
    （submit: queue full → cpp:991 通過 → cpp:998-1008 durable 付着。または redrive W6）
t1  Builder take(O): RebuildDispatch:1069 → cpp:1208 state=Building
t2  build 失敗: RebuildDispatch:1081（recoveryResult.runtime==nullptr）
    → settle(true) RebuildDispatch:1086 → cpp:1242: Building→DurablePending（slot は O を保持）
    → markTransientFailure(O) RebuildDispatch:1091 → cpp:1078: delivery=None
    ⟹ 不正中間状態: slot=DurablePending(O) ∧ delivery=None
t3  次 tick redrive(O)（Threading.cpp:270）: Live ✓ delivery=None ✓（cpp:1140-1143）
    cpp:1158: state != NoAdmission → durable 付着スキップ
    cpp:1174-1177: transport push 成功 → delivery=Transport + P2 latch（1178）
t4  ⟹ 同一 obligationId O が durable slot（DurablePending）と transport queue を同時占有
    （P2 wake により Builder は両方消費: pop→build→publish→resolve(Published) L−1、
      take→build→publish→resolve no-op — 有界重複 build/publish。D136-B と同一結末、
      ただし P2 導入後は「確実に消費される」ため窓は自己修復ではなく**二重実行の確定化**）
```
submit 経路の別生成順も同一窓に帰着: t2 窓開放中に同一 {h,target} 再 submit → cpp:897 wasDeferredBefore=true → cpp:902 opportunistic redrive が t3 と同じ付着 → cpp:943-944 早期 return（2 回目の push は起きないが二重住処は既成）。
**反例は成立 → 修復要（P3-3 契約固定へ）。NO-DEFECT ではない。**

## P3-3 — 修復契約（実装は次ターン・承認待ち）

### delivery の意味論（コードから確定）
`delivery` は **「最後に付与された delivery ownership」**（h:320 "is (or was)"、pop 非クリア、C16 の再 push 許容設計）。**「現在 container に存在する」意味ではない。** したがってユーザー提示 INV-P3-3 の `⇔` は現行設計と矛盾する箇所があり、以下に正確化する。

### 固定不変条件
- **INV-P3-1'**: `DurableCount(O) ≤ 1` — 単一スロット + 単一 `recoveryObligationId` フィールドで構造的真（変更不要）。
- **INV-P3-2'（P3 の本命）**: `¬(O ∈ durable slot ∧ O ∈ transport queue)` — **durable+transport 同時存在の排除**。これが D136-B の違反対象。
- **注意（設計上の既存事実）**: `TransportCount(O) ≤ 1` は**現行不変条件ではない**（cpp:961-968 NOTE・C16・fillRecoveryQueue が同一 obligation の複数 transport 表現を意図的に許容）。ここに ≤1 を課すと coalesce/delivery 再設計（P3-4 禁止の「coalescing 新設」）に踏み込むため、**P3 の範囲は durable+transport 排他**に限定することを推奨（判断はユーザー）。
- **INV-P3-3'（意味整合）**: `delivery==Durable ⇒ slot が O を保持`（settle/discard まで）を**常に成立**させる。`delivery==None ⇒ O は durable 住処を持たない`を修復対象の正反対側として保証。`delivery==Transport` は ownership 意味のまま（pop で消えない）変更しない。

### 最小修正方針（次ターン実装候補 — 本ターン実装しない）
**W7 に same-holder 検査を追加**: `redriveDeferredRecovery` で durable が busy の場合、
```
busy 保持者 == 当該 obligationId（state ∈ {DurablePending, Building} ∧ recoveryObligationId==O）
    → transport push しない。s.delivery = Durable へ**再同期**（実住処への enum の修復）
      + redriveWakePending_ = true（住処は消費待ち → wake は既存 P2 機構のまま）
busy 保持者 ≠ 当該 obligationId
    → 従来どおり transport fallback（AC-P3-3 維持・C13 不変）
```
- 生成点の唯一遮断: t3 の transport 付与が消えるため t4 状態が構造的に生成不能。
- submit 経路は cpp:902 opportunistic redrive（同一修復）+ cpp:943-944 早期 return で自動的に保護。
- `markTransientFailure` の意味は変更しない（P3-4 禁止遵守 — 修復は redrive 側に置く）。
- settle 順序は変更しない（lease 設計維持）。durable slot 設計・容量・coalesce・supersession・wake 機構・memory-order 全て無変更。
- P2 テストとの整合: T-P2-5 の「各周期 1 wake」は修復経路も latch を立てるため成立継続。T-P2-1 は slot free 経路（不変）。
- 既知の残存（本修正の対象外・記録）: W5 の cross-thread plain 書込（P4）、Building 中上書き窓（P5）、discard 時 strand（P5 契約化で吸収）。

### 実装時テスト計画（P3 専用 — 次ターン）
T-P3-1（durable→None→redrive 修復で transport 非付与・delivery=Durable 再同期）/ T-P3-2（異 oblId busy→transport fallback 維持）/ T-P3-3（redrive 後 `Durable(O) ∧ Transport(O)` 不成立）/ T-P3-4（repeat redrive で表現増えない）/ T-P3-5（failure→None→redrive 循環で常に ≤1 durable 表現）/ T-P3-6（P1/P2 回帰: transient failure・redrive・wake・K=4）。

## 判定
- **P3-0/P3-1/P3-2: 欠陥確定**（反例は逐次実行で成立 — P4 の競合に依存しない）。
- **P3-3: 修復契約固定済み**（INV-P3-1'/2'/3' + 最小修正方針 W7-same-holder-repair + 実装時テスト計画）。
- **本ターン実装なし（指示遵守）。STOP — 次ターンに契約承認の上で P3-5 実装へ。P4/P5/P6 非着手。**
