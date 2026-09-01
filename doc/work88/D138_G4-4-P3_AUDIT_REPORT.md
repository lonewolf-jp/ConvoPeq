# D138 — G-4.4-P3 Double Representation / Delivery Uniqueness Audit（Work Report）

**Status: 欠陥確定 → 修復契約固定済み。実装なし（read-only）。**
**詳細:** `evidence/D138_G4-4-P3_DOUBLE_REPRESENTATION_AUDIT.md`
**基準:** ConvoPeq.md `Generated: 2026-08-31 07:12:57`（P2 後ツリーと内容一致検証済み）/ build 状態 = `evidence/g44p2_ctest.log` 40/40×2（P2 以降ソース無変更）

## P3-0 順序の決着
- durable slot が空になるのは `settle(false)`（cpp:1246）と shutdown `discard`（cpp:1224）のみ。`settle(true)` は空にしない（lease 再武装）。
- durable failure 経路（RebuildDispatch:1086→1091）は **settle(true)（slot=O 保持継続）→ markTransientFailure（delivery=None）** の順序で、**「slot 空」と「delivery None」がデカップリング**する。この窓が根因。
- redrive は durable busy のとき**保持者の oblId を見ずに** transport fallback（cpp:1174-1177）。

## P3-1 delivery writer 全数（W1-W7）+ 2 つの確定事項
1. delivery 変更と container 実体操作は atomic でない（順序は常に container→delivery だが、pop は delivery を消さない＝「is (or was)」意味論、W5 は住処保持中に None を書く）。
2. **新規確定**: h:324「delivery は CoordinatorLoop 単一書込」は D105-R18 以降偽 — `markTransientFailure`（cpp:1078）は RebuildThread（Builder loop :1006/1033/1091/1115）から書く。**形式データ競合は P4 へ回す**（P3-4 禁止遵守）。反例はこの競合を必要としない。

## P3-2 反例（現行行番号・逐次実行で成立）→ **欠陥確定**
```
t0 O=Live, delivery=Durable（slot DurablePending(O)）
t1 take→Building（cpp:1208）
t2 build 失敗→settle(true)（cpp:1242、slot は O 保持）→markTransientFailure（cpp:1078、delivery=None）
t3 次 tick redrive: cpp:1158 busy→スキップ、cpp:1174-1177 transport push→delivery=Transport
t4 同一 oblId が durable slot ∧ transport queue を同時占有（P2 wake 後は両方消費され二重実行が確定化）
```
submit 経路（窓開放中の同一 {h,target} 再 submit → cpp:902 opportunistic redrive → 同一付与）も同じ窓に帰着。**NO-DEFECT 不成立。**

## P3-3 修復契約（固定・実装は次ターン承認後）
- **delivery 意味論の確定**: 「最後に付与された delivery ownership」（h:320 / pop 非クリア / C16 再 push 許容）。「現在 container に存在」ではない → INV-P3-3 の `⇔` は要修正。
- **INV-P3-1'**: DurableCount(O) ≤ 1（構造的真・単一スロット）。
- **INV-P3-2'（本命）**: ¬(O ∈ durable slot ∧ O ∈ transport queue)。
- **明示的範囲決定（要ユーザー承認）**: `TransportCount(O) ≤ 1` は現行設計の不変条件ではない（cpp:961-968・C16・fillRecoveryQueue が複数 transport を意図的許容）。課すと coalescing 再設計（P3-4 禁止）に踏み込むため、**P3 は durable+transport 排他のみ**に限定することを推奨。
- **INV-P3-3'**: delivery==Durable ⇒ slot が O を保持（修復で恢复）。delivery==Transport の ownership 意味は変更しない。
- **最小修正方針**: W7 に same-holder 検査 — durable busy ∧ 保持者==当該 oblId なら transport push せず `delivery=Durable 再同期 + redriveWakePending_=true`（消費待ち wake は P2 機構のまま）。異 oblId は従来 fallback（AC-P3-3・C13 不変）。markTransientFailure 意味・settle 順序・durable 設計・wake・memory-order・coalesce 全て無変更。P2 テスト（T-P2-1/5）は成立継続を確認済み（修復経路も latch を立てるため）。
- **実装時テスト計画**: T-P3-1..6（evidence 参照）。

## STOP
欠陥確定 + 契約固定まで。**P3-5 実装は次ターン（契約承認後）。P4/P5/P6 非着手。**
