# D144 — P4 Selected Contract Implementation Plan / Proof Obligations（Work Report）

**Status: CONDITIONAL / CONTRACT REVISION（I-HS → I-HS2）。Production/Test source changes: 0。**
**詳細:** `evidence/D144_P4_IMPLEMENTATION_PLAN_AND_PROOF.md` / **基準:** ConvoPeq.md `13:27:23`（D143 と同一・ツリー無変更確認）

## 中核の発見: I-HS 単独は Case C を反証できなかった
実コード経路で確定:
```text
t1  Builder take(O): DurablePending→Building（lease 取得）
t2  同一 {h,target} 再 submit（queue full）→ cpp:991 ガードは
    「state!=NoAdmission ∧ oblId==O」を **Building でも通過**させ、
    cpp:998-1008 が Builder の lease 中に payload を上書き
t3  → read/write・write-write 競合 + lease 剥奪の可視化
```
`atomic state + release/acquire` は ordering を与えるが **mutation exclusion を与えない**。よって I-HS のままの実装は不可。

## 修約 I-HS2（最小追加契約・証明済み）
1. `state` を atomic 化し**全遷移を CAS 化**（take=CAS(DurablePending→Building) が lease 取得点）。
2. **payload 書込権限規約**: CL は「NoAdmission を acquire 観測した時だけ」payload を書き、その後 CAS publish。reset は lease 保持者または join 後 discard のみ。
3. **same-oblId overwrite 禁止**: slot が同一 oblId を保持中は submit は何も書かず true を返す（同一 oblId ⇒ 同一 semantic target ⇒ build は現在設定を再読するため意味不変。**唯一の挙動差分**として D145 で固定）。副次効果: D136-C と P5 本命が本修約の定理に縮退。
4. **rearm は Builder 側維持（Option A）**: Orchestrator:413 の実行 thread=RebuildThread=take のスレッド、`Building` は take しか生成しない → 同一スレッド lease 内操作と証明。Option B は同期 lease semantics を壊すため棄却。

## transport / exactly-once / shutdown（確定事項）
- RecoveryFailure は**専用 MPSC ring 64 + fallback 64**（intentQueue_ 共有は backpressure 結合を避け棄却）。未処理上限 = 4×32=128 に構造的一致 → **非 drop が証明可能**（防御的 drop counter + HealthEvent 昇格は前例 cpp:1352 に従う）。
- 6 失敗サイトの 1:1 post をサイト別に証明（duplicate 構造なし、stale/unknown/post-terminal は Live チェック no-op、K=4 意味不変）。
- shutdown は**実測順序**で SAFE: `shutdownCoordinatorLoop()`（CL join）→ `stopRebuildThread()`（Builder join）→ discard（RebuildDispatch:812/818、join 後）。thread join が全先行書込に HB を作り、discard は単一スレッド操作。「通常 concurrent でない」仮定は不使用。
- P3 repair の 2 段読は I-HS2 下で stale combination 消滅（acquire した state と oblId が対になる）。same-holder/fallback/repeat/latch すべて不変。

## 最終判定
```text
Verdict: CONDITIONAL / CONTRACT REVISION
Selected contract: I-HS2（I + atomic state CAS + payload 書込権限規約 + overwrite 禁止 + rearm=Builder）
GO criteria 1-7: I-HS2 で全項目 ✓（evidence の対応表）
NO-GO 項目の残存: なし
→ D145（P4 Implementation）は I-HS2 を対象に進めてよい。
   最初の成果物は修約 3 の挙動差分の既存契約適合証明とすること。
```
テスト仕様（HB/race 4、event semantics 7、retry 7、compatibility + 差分固定）は D144-9 に確定済み（**未実装**）。

**STOP — 実装 0。P5/P6 非着手。D145 の指示を待つ。**
