# D172-2 — MEM_SNAP Repair Contract / Authority Impact Audit Report

- Date: 2026-09-07
- Task: D172-1 STOP（CONFIRMED LIFETIME HAZARD）を受けた修復契約監査
- Scope: 修復案A preflight / RuntimeWorld→DSP lifetime chain 再証明 / TRK semantics / R2・R3 / 案B formal rejection / 案C necessity
- Type: read-only（production source 0 / test source 0 / CMake 0 / build 0 / CTest 0）
- Evidence: `evidence/D172/D172_2_MEM_SNAP_REPAIR_CONTRACT_AUDIT.md`

## 判定

> ## **D172-2 GO — 案A（RuntimeWorld/read-handle 経由の解決）を採用。implementation contract を固定し、D172-3 implementation へ進行。**

```text
D169-2 CLOSED → D171-1 PASS → D172-1 STOP → D172-2 GO
                                                 │
                                                 ▼
                              D172-3 implementation（contract 固定済み）
```

| 項目 | 判定 |
| --- | --- |
| P1 RuntimeWorld resolution | **PASS** |
| P2 DSP lifetime proof | **PASS**（2 層構成） |
| P3 TRK semantic compatibility | **PASS**（intentional） |
| P4 R2 latency fallback | **PASS** |
| P5 R3 dormant reader | **PASS** |
| P6 Option B rejection | **PASS**（REJECT 固定） |
| P7 Option C necessity | **PASS**（不採用） |
| P8 Repair contract | **GO** |

## 根拠（要約）

1. **P1（案A preflight）**: `resolveActiveRuntimeDSPFromRuntimeWorldOnly()`（AudioEngine.h:3398）は RT path と同一の authoritative resolver であり、timerCallback は既に `:503/685/700` 等で常用。`runtimeReadHandle` は `timerCallback()` 冒頭（Timer.cpp:428・関数トップレベル）で取得され、move は 0 件（全使用が const 参照渡し）、MEM_SNAP block（:1022-1115）はその生存スコープ内。**handle は dereference 完了時点まで有効・その間 MessageThread は EBR reader として active。**
2. **P2（lifetime proof・2 層）**: retire の正式経路は OBSERVE-1（Timer fade 完了 → `submitObserve(handle, currentPublicationEpoch())` → CoordinatorLoop worker `retirePublishedDSP` → `DSPLifetimeManager::retire(D, E_r)` → `enqueueWithRetry(D, &destroyDSPCoreNode, E_r)` → drain で `isOlder(E_r, minReaderEpoch)` 成立時のみ破壊）。Reader は `enterReader` で enter 時の currentEpoch を pin（EpochDomain.h:114-131）し、replacement が enter 後に起きた場合 E_r > E_pin → **read section 中の破壊は遅延**（Layer 1 完全 proof）。Layer 2: `makeRuntimeReadHandle` 内の world observe → enter のμs window（Case 3）は RT/Latency/Publication を含む**全 world reader 共有の既存境界**であり、MEM_SNAP 固有の新規リスクはゼロ。enter-first 逆転は scope 外として記録（未観測・着手理由なし）。
3. **P3（TRK semantics）**: producer 1 箇所のみ・consumer（test assertion / threshold / parser / external tooling）**0 件**。evidence は観測補助のみ（D162-1P は liveCount/DSP_FOOTPRINT が主指標）。現行 TRK は placeholder 破壊後 garbage 値であり意味ある baseline ではない → **semantic change = intentional として許容**（D172-3 でログコメントに意味論変更を明記する契約）。
4. **P4（R2）**: 「world null ∧ slot dangling」は成立しない — world non-null 期間（rebuild 置換後の dangling 期間）は fallback 非到達、release/dtor は同一 critical section 内で slot clear が world clear に先行、rollback（:317→:318）は同一 MessageThread 連続コード。fallback の全 caller（MainWindow UI / AudioEngineProcessor::prepareToPlay — audioEngine.prepareToPlay と同一シーケンス）は MessageThread 系で RT は呼ばない。**現行構造のまま lifetime-safe・変更不要。**
5. **P5（R3）**: production caller 0 件を再確認。dormant 維持とし、復活時は world resolution 経由に統一する旨の契約コメントを D172-3 に含める。
6. **P6（案B formal rejection）**: invariant 5/5 で REJECT — (1) RC-D169-1-2「ownership authority に昇格させない」違反、(2) destroy path の pointer identity 突合は terminal disposition 一本路契約（AudioEngine.h:4332-4348）外、(3) address reuse race（aligned_free 後の再割当で生存 DSP 誤対象化 — D169-1 probe8/9 実測の二重破壊クラスと同型）、(4) D170 handle 一本路への第 2 authority 萌芽、(5) 廃止済み raw-pointer lookup パターンの再導入。
7. **P7（案C）**: 案A で proof 成立のため新規 registry/authority を追加する理由なし → 不採用（A の proof 崩壊時の再設計候補として記録保持）。

## Implementation contract（D172-3 実装境界 — コードは未変更）

1. 変更は **MEM_SNAP block の `getActiveRuntimeDSP()` → `resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)` 1 箇所のみ**。新規 atomic / helper / queue / authority 追加禁止。
2. 禁止: slot writer（W1-W4）変更 / destroy path 変更 / slot clear 追加（案B 相当）/ test source 変更 / CMake 変更。
3. MEM_SNAP コメントに TRK 意味論変更（world current DSP ベース・world 未公開期 0）を明記。`logRuntimeTransitionEvent` に復活時契約コメント。
4. 実装前に方式 α（log 相関 `[D117_DESTROY] dsp=P` × `[MEM_SNAP] TRK≠0`）で baseline evidence を取得（修復前 dangling 状態の記録）。
5. 検証: 既存 CTest 40/40 ×3 config + diagnostic build で MEM_SNAP 出力確認（dangling 解消・TRK が実 active DSP の値を示す）。
6. scope 外記録: makeRuntimeReadHandle enter-first 逆転（全 world reader 影響・別 track）。

## 次工程

- **D172-3 implementation**（contract 準拠・最小差分 1 箇所変更 + baseline evidence 取得 + CTest 回帰）
- doc-only maintenance（inventory STALE 化反映 + buildErrorCount_ trigger 登録）は D172-3 とは別 window で処理
