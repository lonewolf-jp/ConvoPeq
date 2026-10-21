# STG-11-D18 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D18_REPAIR-CONTRACT-AUDIT_20261001.md`
- Work item: **STG-11-D18-1** — 連続 publish 時の phantom crossfade record と stale completion による DSP resurrection
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用）
- Commit / push: **禁止**

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

---

## 1. Defect statement

`DSPTransition::onPublishCompleted` は fading-slot の claim 成否にかかわらず
crossfade record の登録＋`beginCrossfade`＋ramp 再始動を行う。
claim 失敗時（先行 fade が実行中）、old DSP は即時 retire されるにもかかわらず、
`(old→new)` の record が登録される。この phantom record は audio が実行しない
fade を主張し、後の completion が `endCrossfade` の無条件書き込み
（toHandle→Active、`activeHandle`=toHandle）により retire 済み DSP を
Active に復活させ、かつ `activeHandle` を stale 値で上書きする。
後続 publish は stale handle を oldHandle として使用し、
reclaim 済み object の fading 復活（UAF）に至り得る。

## 2. Exact source location

| 要素 | file:line |
| --- | --- |
| 無条件 register/begin/ramp | `DSPTransition.h:113-135`（claim は `:121` の後） |
| claim 失敗時の即時 retire | 同 `:124-127`（`lifetime.retire(oldDSP)`） |
| front-only 消費＋単一前提 | `AudioEngine.Timer.cpp:1117-1129`（`records.front()`＋`jassert(size==1)`） |
| 無条件 resurrection 書き込み | `ISRDSPHandle.cpp:106-112`（`endCrossfade`） |
| fading-handle の無条件 null | `ISRDSPHandle.h:activate` 内＋`endCrossfade` 内 |
| fire-and-forget 経路の fade gate なし | `RuntimePublishExecutor.h`（fading 検査なし。Orchestrator submit のみ `DeferredFadingActive` を持つ） |

## 3. Reproduction

```text
1. R1 active、R2 publish → fade R1→R2 開始（record id1、slot=R1、handle=R1）
2. fade 完了前に R3 publish（fire-and-forget 経路。fade gate なし）
   → record id2(R2→R3) 登録、claim(R2) 失敗 → R2 即時 retire
3. completion#1：records=[id1,id2] → front id1 → endCrossfade(id1)：
   R1→Retired、R2→Active（復活）、activeHandle=R2（stale。実際は R3）
4. id2 残留。後続 publish は oldHandle=R2（retired）を使用
```

全段 deterministic（timing 依存なし。重ね合わせ自体は fade window 内の
連続 publish で成立し、fire-and-forget 経路に gate がないため構造的に到達可能）。

## 4. Expected behavior

- claim 失敗時は record/begin を作らず、即時 retire のみ（single-record invariant 維持）。
- `endCrossfade` は既に正しい値を上書きしない（idempotent-safe）。
- completion 後：R1 Retired、R2 Retired のまま、R3 Active、`activeHandle`=R3、
  active record なし。

## 5. Actual behavior

- phantom id2 が登録・残留し、`jassert(size==1)` が発火（Debug）／無言残留（Release）。
- `endCrossfade(id1)` が R2 を Active に復活＋`activeHandle`=R2 に上書き。
- 後続 publish の oldHandle が retired R2 になり、claim 成功時は
  reclaim 済み object の fading 復活（UAF）に至り得る。

## 6. Root cause

3 点の無条件操作の組み合わせ：

1. `onPublishCompleted` の register/begin が claim 結果に依存しない。
2. `endCrossfade` の toHandle/activeHandle 書き込みが現状非依存。
3. `activate()` の fading-handle null が in-flight fade の retire 経路を消す
   （B-1/:1310 の observe が null を見る）。

## 7. Affected thread/context

- 全経路 NonRT（CoordinatorLoop／Timer）。RT 構造の変更なし。
- 被害：registry state、active handle、record table（NonRT state）。
  最終被害は audio thread の UAF に波及し得る（reclaim 済み fading 復活時）。

## 8. Data-integrity / DSP impact

- Stale `activeHandle` による後続 publish の誤 oldHandle。
- Retired DSP の Active 復活（registry）による誤 resolve→UAF。
- Phantom record の残留（table leak＋完了シグナル欠落）。

## 9. Invariant violation

- Single-fade invariant（`jassert(size==1)` が設計前提と宣言）。
- `one obligation → one delivery representation` の crossfade 版：
  同一 pair の重複表現（phantom id2）。
- Retire-then-resurrect の禁止（epoch 所有権の単調性）。

## 10. Minimal repair（NonRT のみ。3 点）

**F-A. `DSPHandleRuntime::activate`：`fadingRuntimeDSPHandle_` の null 書き込みを除去。**
in-flight fade の retire 経路（B-1/:1310 の observe）を publish が消さない。
通常 path では既に null のため無変更と等価。caller は
`DSPTransition` の 2 箇所のみ（重ね合わせ以外の意味を持たないことを確認済み）。

**F-B. `DSPTransition::onPublishCompleted`：claim を register/begin より先行。
claim 失敗時は retire のみ行い、register＋begin を skip（ramp.start は維持し
可聴動作を不変に保つ）。**
通常 path の効果順序は同一（register→begin→claim✓→receipt→ramp）。
claim 失敗時は phantom record が生まれない。

**F-C. `DSPHandleRuntime::endCrossfade`：stale 書き込みのガード。**
- toHandle→Active は現在 CrossfadingIn の場合のみ。
- activeHandle=toHandle は現在 ==fromHandle または ==toHandle の場合のみ。
- fading-handle=null は現在 ==fromHandle または既に null の場合のみ。
- fromHandle→Retired は無条件のまま（両到達状態とも Retired が正解のため）。

いずれも通常 path の効果と同一（書き込みが no-op になる条件でのみ skip）。
新規 authority／mutex／atomic／RT 変更なし。

## 11. Regression test

新規 harness サブテストではない。`DSPHandleRuntime` 単体 TU
（`DSPHandleRuntime`＋`CrossfadeAuthorityRuntime` は default 構築可能。
`create(void*)` は dummy pointer で可。engine／DSPCore 不要）：

- T-CFX-A：begin(h1,h2)→retire(h2)→activate(h3)→endCrossfade(id1) →
  h2==Retired のまま、activeHandle==h3、h1==Retired。
- T-CFX-B：onPublishCompleted の claim→register 順序の structural 検証。
- T-CFX-C：activate() が fading handle を消さないこと（begin→activate→handle 維持）。
- Negative control：endCrossfade／activate／onPublishCompleted を旧形に一時復元
  すると各 T が FAIL。

## 12. Negative control

§11 各 T がその役割を果たす。旧 code では resurrection／stale／clobber が再現し
FAIL、修正後は PASS。両方向を実証する。

## 13. D1〜D17 との境界

- D1〜D5（retire/epoch/MMCSS）、D6 NO-GO、D7〜D11（session 値）、
  D12〜D17（監査のみ）に触れない。
- R4/P3（recovery delivery）とは別 authority（crossfade delivery）。混ぜない。
- D14 の generation identity に触れない。

## 14. Risk assessment

- Low-Medium。並行 state machine のため通常 path 等価性をSAS structurally＋
  functionally に検証する（T-CFX＋full CTest）。
- 可聴動作の変更：重ね合わせ時のみ（従来は incoherent）。通常 path は同一。
- 唯一の挙動変更：claim 失敗時の record 非生成、stale 書き込みの抑止、
  activate 時の handle 維持。いずれも incoherent 状態の除去である。
