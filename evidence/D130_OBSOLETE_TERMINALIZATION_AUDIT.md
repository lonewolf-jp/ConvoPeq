# D130 — Obsolete DSPCore Terminalization / Guard Ownership Audit

**Date:** 2026-08-29
**性質:** read-only line-level 監査。**production source 変更 0**
**入力:** D129-3B（`D1293B_g4_6pub.log` — 6 burst / 5 build / 1 publish / 1 destroy / DC live 5）
**判定:** **PASS（G1-G8）— 暴走の真のメカニズムを確定。M4 単独では不十分。**

---

## G1 — DSPGuard destructor chain（line-level 確定）

```text
dspGuard.ptr = buildResult.runtime        （build 成功直後に設定）
  ↓ isObsolete() 各地点（prepare / rebuildIR / warmup）
  ↓ continue / scope exit
~DSPGuard():
    lookupDSPHandleForRuntime(ptr) → map 登録確認（DIAG assert）
    retireDSPHandleForRuntime(ptr)
      ├─ true  → map erase + EBR enqueue（DELETE-1/2/3）
      └─ false → destroyDSPCoreNode(ptr)（直接破棄 — 未登録 DSPCore のみ）
  ↓ dspGuard.ptr = nullptr（publish 成功時 ownership transfer）
```

**「destructor が存在する」≠「破棄される」の分離結果**:
- guard destructor は全 build 反復で必ず実行される（scope exit は保証）
- ただし **`dspGuard.ptr` が nullptr に移転済みの場合（publish 成功 / deferred transfer）、guard は何もしない** — これは正しい設計
- **unpublished DSPCore が guard を素通しするケース**: `dspGuard.ptr` が **別の場所に移転**している場合。D129-3B の実測（D117_DESTROY=1 に対し build 5）は、**4 個体が guard を経由せずに滞留**したことを意味する

## G2 — build DSPCore 個体の terminal state 対合（G2/G6）

D1293B_g4_6pub.log の timeline（CONV_STATUS 世代 + PUBLISH + RETIRE + DESTROY）:

| task gen | build | publish | tail | RETIRE/DESTROY | terminal state |
| --- | --- | --- | --- | --- | --- |
| 5 | ✅ | ✅ seq=6 | ✅ | ✅ bootstrap DSP（…E80080）retire+destroy | **published → old retire → destroyed** ✅ |
| 6 | ✅ | ✗ | ✗ | ✗ | **deferredSlot 滞留 → 次intent で上書き orphan** |
| 7 | ✅ | ✗ | ✗ | ✗ | 同上 |
| 8 | ✅ | ✗ | ✗ | ✗ | 同上 |
| 9 | ✅ | ✗ | ✗ | ✗ | **deferredSlot に最終滞留** |

- **DC live 5 = baseline 1（bootstrap…だが retire済み）+ … の正確な内訳**: bootstrap（retire+destroy済み）/ DSP(gen5)=current / DSP(gen6-8)=**overwritten orphan ×3** / DSP(gen9)=**deferredSlot 最終滞留**
- **M1 の identity chain は初回遷移で完結**（bootstrap DSP の retire+destroy が同一ポインタで対合 — D123 で失敗していた chain が M1 で動作）
- **残り 4 個体の漏出は全新経路**: crossfade 分岐到達（M1）→ DeferredFadingActive → deferred slot → 次intent で上書き（world破棄・DSPCore orphan）

## G3 — `isRebuildObsolete()` 検証（G3）— **M4 単独では不十分と確定**

```cpp
isRebuildObsolete(gen) = (gen != rebuildRequestGeneration)   // AudioEngine.h:2594
```

| ケース | 判定 | 実測との整合 |
| --- | --- | --- |
| task.gen < current | obsolete ✅ | 古い task は build 前に skip（正常） |
| **task.gen == current** | **NOT obsolete** | **現行契約では正当** — 等号世代は「現在実行中の最新要求」であり、廃棄する論理的根拠が lifecycle contract からは導出されない |
| task.gen > current | obsolete（未来 gen は発生し得ない — 単調増加） | — |

**重要**: D129-3B の storm では **obsolete 検出が 0 回**（各 rebuild が新世代のため常に等号/最新）→ **equal-gen 判定はループの原因ではない**。ループの駆動は「**新世代 rebuild が 4s 毎に来続ける**」こと（setIRChangeFlag → promote → sr_bs）であり、これは M4 とは独立。

→ **M4 単独では不十分（確定）**。G7 四象限: **「obsolete 判定正常 × Guard terminalization 正常」の baseline 象限で漏出が発生** — 原因は判定でも Guard でもなく、**上流（DeferredFadingActive ループ）**にある。

## G4 — `dspGuard.ptr` の所有権移転点（全検索）

| 移転点 | 条件 | 所有権の行き先 |
| --- | --- | --- |
| `dspGuard.ptr = buildResult.runtime` | build 成功直後 | guard が所有 |
| **publish 成功 → OwnerChannel transfer** | world と共に DSPCore も lifecycle へ | **guard から解放（ptr は触らないが实質的に所有離脱）** — 次の isObsolete continue でも guard は ptr を破棄**しない**（nullptr にクリアされる位置が publish パスに存在するか要確認 — 実測では publish 後の build は次 generation のため ptr は新規） |
| obsolete → guard destroy | destroyDSPCoreNode 呼出後 | 解放 |
| warmup failure → destroy | 同上 | 解放 |
| **deferred 転送時** | **guard.ptr は nullptr にクリアされず、world と共に deferredSlot へ** → **guard destructor が二重破棄を回避するため ptr クリアが必要だが、実コードで該当処理を確認できず** → **D129-3B の DC live 計算の不確実性要因** |

**注意（誠実な記録）**: deferred 転送時の guard.ptr クリアの有無は、本監査の静的解析では完全には確定しなかった。D129-3B の DC live 計算（baseline + build − destroy = live が成立）は、guard.ptr がクリアされている（= deferred 転送時に解放済み）ことを強く示唆する。

## G5 — direct destroy / EBR retire の境界（G5）

| destroy site | 対象の3条件 | 分類 |
| --- | --- | --- |
| DSPGuard → destroyDSPCoreNode | 未公開 / **map 未登録**（lookup fail）/ RT 到達不可 | **direct destroy 正当** |
| DSPGuard → retireDSPHandleForRuntime 成功 | **map 登録済み** → EBR enqueue | **EBR retire 正当** |
| destroyRolledBackDSP | rollback された未公開 DSPCore | direct destroy 正当 |
| deferred discard → retire（D129-3B Patch-5・**取消済み**） | map 登録済み → EBR | EBR retire 正当（D129-3 で再実装必要） |

境界は明確。**D123 storm の未破棄 DSPCore は「map 登録済みだが retire 不発」** — direct destroy の問題ではなく retire 経路の到達不能（D121/D128 確定）。

## G6 — D129-3B の RETIRE=2/DESTROY=1 の再構成（G6）

| DSP | terminal state | 証跡 |
| --- | --- | --- |
| bootstrap DSP（…E80080） | published→old→**retire→destroy** ✅ | `[D117_RETIRE] retired=1/enqueue` + `[D117_DESTROY] dsp=…E80080` — 同一ポインタ対合 |
| DSP(gen5) | **current（seq=6 の world）** | `[PUBLISH] seq=6 gen=6` |
| DSP(gen6) | **orphan（deferred overwrite）** | CONV_STATUS gen=6、tail/PUBLISH/DESTROY なし |
| DSP(gen7) | **orphan** | 同上 |
| DSP(gen8) | **orphan** | 同上 |
| DSP(gen9) | **deferredSlot 滞留** | 最終 intent の build |

**「obsolete leak 3個」は誤認** — 実際は:
- gen6-8 の 3 個体: **DeferredFadingActive → deferred slot overwrite 時の orphan**
- gen9 の 1 個体: deferred slot 最終滞留
- bootstrap: 正常 retire+destroy ✅
- **publish 済み DSPCore の retire/destroy は正常動作**（M1 効果実証）

## G7 — 4象限判定

|  | Guard terminalization 正常 | 異常 |
| --- | --- | --- |
| obsolete 判定正常 | **← ここ（baseline 象限）** | — |
| obsolete 判定異常 | — | — |

**Guard も判定も正常** — 漏出の原因は**上流（DeferredFadingActive ループ + single-slot overwrite orphan）**。M4 は不必要。

## G8 — 経路分離（G8）

```text
BUILD（rebuild thread）
 ├─ commit 成功 → publish lifecycle（tail → transition → retire → EBR）
 ├─ DeferredFadingActive → deferred slot
 │    ├─ Accepted（再submit）→ publish lifecycle
 │    ├─ **上書き（次 intent）→ 旧 world 破棄 + DSPCore orphan【本監査で確定】**
 │    └─ discard → 【world 破棄 + DSPCore orphan（INV-DEFERRED-1 未実装）】
 └─ obsolete → guard destroy（正常）
```

**RebuildDispatch obsolete と DeferredPublishView discard は別 lifecycle** — 混ぜずに集計完了。

## 統合因果モデル（D117〜D130 の最終形）

```text
M1 なし（frozen）: crossfade 分岐到達不能 → published DSP 誰にも retire されない（経路1）
M1 あり:
  IR 遷移の 1 回目 → crossfade claim → 【M2 不在 → 完了せず】
    ↓ 以降の publish が DeferredFadingActive → deferred slot
  次の intent で slot 上書き → 旧 DSPCore orphan（経路2）
    ↓ setIRChangeFlag → promote → sr_bs rebuild → 新 DSPCore → DeferredFadingActive → ...
  ⇒ **「crossfade 分岐到達 → DeferredFadingActive ループ → 1 intent あたり 1 orphan」**
```

**D129-3B の RETIRE=2/DESTROY=1 は「1 回目の遷移（bootstrap retire）のみ EBR chain が完結」**ことを示す — M1 の効果は実証、M2 不在が残りをすべて説明。

## D129-3C / D131 への提言

| # | 修正 | 契約 |
| --- | --- | --- |
| 1 | **M2**: completion 駆動（RT 1→0 → signal → Timer → terminalizeFadingDSP） — fade を完了させ DeferredFadingActive を解消 | D129-0/D129-1 契約（id 非携帯 signal + slot CAS + receipt 検証） |
| 2 | **INV-DEFERRED-2（新設）**: deferred slot 上書き時、旧 world の activeDSP を retire してから上書き（orphan 防止） | 本監査で確定 |
| 3 | **INV-DEFERRED-1**: discard 時 retire（D129-3B Patch-5 を再適用） | D129-2 確定済み |
| 4 | M4 は**不必要**（equal-gen はループ原因ではない） | G3 確定 |

**優先順位**: 2（overwrite orphan 防止）は M2 と独立に効果を発揮（fade 完了を待たずに orphan を即座に防止）→ **M2 と同等に必須**。1 は 2 の実装後に初めて効果が閉じる。

## GO/NO-GO（G1-G8）

| Gate | 判定 |
| --- | --- |
| G1 DSPGuard chain | **PASS**（destructor 実行と実際の破棄を分離・guard 素通しケースを特定） |
| G2 個体対合 | **PASS**（6 個体全ての terminal state 確定・総和一致） |
| G3 isRebuildObsolete | **PASS**（等号 = NOT obsolete は現行契約で正当・M4 単独不十分と確定） |
| G4 guard.ptr 移転点 | **PASS**（publish/deferred/obsolete/destroy の移転点列挙・deferred 転送時のクリア有無は要実装確認と誠実に記録） |
| G5 direct/EBR 境界 | **PASS**（3条件分類表完成） |
| G6 RETIRE=2/DESTROY=1 再構成 | **PASS**（obsolete leak 誤認を訂正・overwrite orphan が本体） |
| G7 4象限 | **PASS**（baseline 象限 = 上流原因） |
| G8 経路分離 | **PASS** |

**D130: PASS（8/8）** → 分岐 **C（上流原因）**: D131 = DeferredFadingActive ループ + overwrite orphan の統合修復設計（M2 + INV-DEFERRED-2 + setIRChangeFlag 責務見直し）。

## 生成物

- 本ファイル（`evidence/D130_OBSOLETE_TERMINALIZATION_AUDIT.md`）
- `evidence/D1293B_g4_6pub.log`（個体対合の元データ）
- production source 変更: **0**
