# D124 — IR-Flag / Structural-Rebuild Feedback Loop Causality Audit

**Date:** 2026-08-29
**性質:** read-only 原因再監査。**production source 変更 0**。frozen baseline（`a65ace1` + D117 trace）維持
**入力:** D123 NO-GO（`D123_IMPLEMENTATION_NOGO_AUDIT.md` — 6-publish smoke で DC live 103 / CONV_REBUILD 107 / Priv 15.8GB）

---

## D124-1 — `setIRChangeFlag()` 全 production caller（G1）

定義: `AudioEngine.h:1469` — `publishAtomic(m_pendingIRChange, true, release)` のみ（**clear 機能なし**）。

| caller | Thread | 起因 | 意図 | rebuild を発生させるか |
| --- | --- | --- | --- | --- |
| `AudioEngine.Timer.cpp:800` | Timer（NonRT） | **rebuild 発行時**（deferred Structural release: 「issuing deferred Structural rebuild after prepared IR apply」→ submitRebuildIntent の直後） | IR 遷移落下の伴奏フラグ | rebuild は発行済み（flag は伴奏） |
| `AudioEngine.UIEvents.cpp:177` | Message Thread | `needsStructuralRebuild`（UI 側 IR/構造変更検出 — convolver 状態変化 listener） | IR 変更通知 | 直後に submitRebuildIntent（構造 rebuild を起こす） |
| `DSPTransition.h:123` | CoordinatorLoop | **rebuild 自身**（crossfade 分岐 — D123-A 配線で到達可能化） | crossfade 後処理としての IR 変更通知 | **要監査 = 本監査の対象** |

**備考**: `convolverParamsChanged` からの設定は現行ソースに存在しない（CLI で suppress されるのは同関数の rebuild であり、flag 設定は上記 3 箇所のみ）。

## D124-2 — `m_pendingIRChange` 完全状態追跡（G2）

| 種別 | 場所 | 動作 | 意味論 |
| --- | --- | --- | --- |
| writer | `setIRChangeFlag()`（3 caller 経由、AudioEngine.h:1469） | true を publish | **clear 機能を持たない片方向 setter** |
| reader | `AudioEngine.Parameters.cpp:381, 452` | `shouldDeferRebuild` の条件の一つ（`outstandingRebuild \|\| isLoadingIR \|\| DeferredStructural \|\| pendingIRChange \|\| ...`） | true → setDither/setNoiseShaper 等の構造 rebuild を **延期**（DeferredFinalizeAware） |
| reader+clear | `AudioEngine.Timer.cpp:849` | `consumeAtomic` → `finalizeReady` の条件（`!pendingIrChange` が必要） | true → deferred finalize dispatch を 1 tick 遅延。**消費で false に戻る** |
| reader+clear | `AudioEngine.Snapshot.cpp:95` | `exchangeAtomic` → `promoteToStructural=true` なら「IR/構造変更はスナップショット fade に流さず、構造クロスフェード経路へ昇格」して early return | 消費＝snapshot fade のスキップ |

**意味論の確定（D124-2 の核心）**: `m_pendingIRChange` は
- 「IR 変更が発生した」という**事実の記録**ではなく
- **「進行中の IR 遷移が落ち着くまで rebuild/finalize を延期せよ」という抑止命令**（Parameters/Timer の defer 条件）および「**snapshot fade を構造経路へ昇格せよ**」という経路指示（Snapshot promote）

の**複合**である。ゆえに「crossfade で flag を立てる」ことは「次の rebuild を要求する」ことではなく、**「進行中の finalize/finalize dispatch を延期し、snapshot fade を構造経路へ昇格させる」**ことになる。D123 のループはこの抑止/昇格の組合せが形成した。

## D124-3 — promoteToStructural 因果チェーン（G3）— 必須因果 vs 偶発条件

```text
[b1] burst intent → submitRebuildIntent(Structural)          （外部入力 — 必須）
  ↓ [b2] requestRebuild(sr,bs)（RebuildDispatch.cpp:344 委譲）→ telemetry "sr_bs"   必須（委譲構造）
  ↓ [b3] rebuild 実行 → DSPCore 生成 → CONV_REBUILD                           必須
  ↓ [b4] publish → onPublishCompleted                                         必須
  ↓ [b5] crossfade 分岐到達（oldDSP 非 null — D123-A 配線）                    必須（D123-A により到達可能化）
  ↓ [b6] setIRChangeFlag()（DSPTransition.h:123 — 無条件）                     ★ 必須因果（分岐内で無条件）
  ↓ [b7] m_pendingIRChange = true
  ↓ [b8] consumer（Timer.cpp:849 defer / Snapshot.cpp:95 promote / Parameters defer）  ←【特定に診断 1 点必要 — 下記】
  ↓ [b9] 次の submitRebuildIntent(Structural)（telemetry: kind_entry fp=0xd9d3 不変）  必須（実測）
  ↓ [b10] requestRebuild(sr,bs) → rebuild → (b3) へ                            必須
```

**分類**:
- **[b6] は必須因果**: 分岐内で無条件に実行され、撤去すれば DSPTransition からの再主張は消える
- **[b8]→[b9] の中継点は確定に至らず**（候補: Timer.cpp:849 消費 → finalizeReady false → timeout forced dispatch → Timer.cpp:800 が**再度** flag 設定+rebuild 発行、という経路が最有力。`timedOut && !finalizeReady` で「forcing rebuild dispatch」→ Timer:800 `++pendingIRGeneration; setIRChangeFlag()` が**ループの増幅点**になり得る）。Flag の consume/clear が Timer:849 で tick 毎に行われるため「永久延期」ではなく「timeout 強制発行 → 再 flag → 再 timeout」の**緩やかなループ（約 2 回/秒）**として整合
- **fingerprint 0xd9d3 が不変**（intentId 16〜24 全同一）= ループは外部入力なしで**同一状態の rebuild を反復**している — 純フィードバックの証拠（G6）

## D124-5 — 意味論分離（G4）

| 概念 | 定義源 | IR flag との関係 |
| --- | --- | --- |
| IR flag（m_pendingIRChange） | rebuild/finalize/snapshot-fade の**抑止・昇格ヒント**（RebuildDecision 側） | Crossfade 判定には入力されない |
| IR state（irLoaded / isIRFinalized / isLoadingIR） | ConvolverProcessor の実状態 | 同上 |
| structuralHash change | RuntimePublishWorld 投影（旧 vs 新 world の hash 比較） | 同上 |
| `needsCrossfade` | `CrossfadeAuthority::evaluate(oldWorld, newWorld, policy)` — **world 投影値のみ**（kEvaluateRelevantFieldNames = irLoaded / structuralHash / oversamplingFactor） | **IR flag は evaluate の入力ではない**（CrossfadeAuthority.h:44-47・m_pendingIRChange は evaluate 関係ファイルに存在しない） |

∴ **IR flag ≠ IR state ≠ structuralHash change ≠ needsCrossfade** — 4 者は別概念（G4 PASS）。D123 の暴走は「needsCrossfade の誤評価」ではなく「**crossfade 分岐が rebuild 判定系のヒント flag を無条件に再設定した**」ことが原因。

## D124-6 — D123 107 rebuild の causal replay（G5/G6）

`evidence/D123_g4_6pub.log` から抽出した世代対応表（intentId 15〜25 抜粋）:

| seq | intentId | reason | hash / fingerprint | publish | crossfade | setIRFlag | 次rebuild |
| --: | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | kind_entry | 0x0 / 0xc7a7（bootstrap 状態） | ✗（後述） | ✗（初回は oldDSP=null） | （Timer:800 系） | 16 |
| 2 | 16 | kind_entry | 0x73d6 / 0xd9d3（IR 適用後） | ✅ seq=6 | **✅（到達）** | **✅ :123** | 17 |
| 3 | 17 | sr_bs | 0x0 / 0x0 | ✗（後続に obsolete） | — | — | 18 |
| 4 | 18 | kind_entry | 0x73d6 / **0xd9d3（不変）** | ✗ | — | — | 19 |
| … | 19-25 | sr_bs / kind_entry 交互 | 0x0 / 0xd9d3（不変） | ✅ seq=9 のみ | ✅ | ✅ | … |

- **fingerprint 0xd9d3 が intentId 16〜24 で不変** → ループは同一状態の反復要求（外部入力なしの純フィードバック）
- kind_entry↔sr_bs の厳密交互 = 「submitRebuildIntent（kind_entry）→ requestRebuild 委譲（sr_bs）」が 1 セット、それが反復
- **1 crossfade → 1 次rebuild の 1:1 対応は telemetry 上直接証明できない**（DSPTransition.h:123 に telemetry がない）。ただし (a) fingerprint 不変、(b) 約 2 回/秒の定常反復、(c) frozen baseline（6 rebuild）からの逸脱が D123-A 配線と正確に同期、により **閉ループ存在は確定（G6）**。中継点 [b8] の確定に 1 点の診断が残る（setIRChangeFlag 呼び出し時の caller tag — macro-gated、D125 診断に含める）

## D124-7/8 — Invariant 再定義（G7/G8）

- **INV-XFADE-4 は廃止**（fading slot 占有はループの駆動点ではなかった — D123 実測で fading slot は正常に CAS/解放された）
- **INV-IRFLAG-1**: *Rebuild-generated DSP transition SHALL NOT create a new structural-rebuild admission solely by reasserting the IR-change state consumed by that rebuild.* — line-level evidence: DSPTransition.h:123 が分岐内無条件で `setIRChangeFlag()` を呼ぶ唯一の「transition 側」writer であり、D123 実測でこれがループ起点となった
- **INV-IRFLAG-2**: *An IR-change indication must have a bounded causal lifetime* — 現行 flag は (a) clear 経路が Snapshot.cpp:95 のみで実行条件付き、(b) Timer:849 の tick 毎消費が timeout 強制発行と組み合わさって bound がない。bound を与えるには「起因タグ」または「rebuild 消費済み generation 記録」（案C の edge-triggered state）が必要

## D124-9 — 次回最小修正箇所（G9）

**1 箇所に限定**: `DSPTransition.h:123` の `setIRChangeFlag()`（rebuild 起因 crossfade での設定を撤去、または起因タグ付き API への変更）。
根拠: (a) D123 実測でループは D123-A 配線（crossfade 分岐到達）と同時に出現、(b) frozen baseline では同分岐が到達不能でループなし、(c) Timer:800 / UIEvents:177 は「rebuild を起こす発行点」の伴奏であり、rebuild 起因 crossfade から再設定しなければ閉ループの入口が消える。

**残る不確実性（正直な記録）**: ループの中継点 [b8]（Timer timeout 強制発行 vs Parameters defer チェーン vs その他）は候補絞り込みまでで、確定には D125 診断（setIRChangeFlag caller-tag）が 1 点必要。

## D124-10 — lifecycle repair（D125）との相互作用（G10）

IR flag 修正（:123 撤去）は D119/D123 の lifecycle 修復（activate 公開 / completion 駆動 / Observe 分離）と**非干渉**:
- :123 は crossfade 分岐内の 1 行であり、retire/crossfade 遷移そのものには不要（IR 変更通知の伴奏）
- activate 公開・completion consume・retirePublishedDSP は :123 と無関係に動作
- 統合パッチ（D126）: D123 の全変更 + :123 撤去（または起因タグ）を **単一パッチ**として適用（単点修正の失敗 3 回の教訓）

## GO/NO-GO（G1-G12）

| Gate | 判定 |
| --- | --- |
| G1 setIRChangeFlag 全 caller 列挙 | **PASS**（3 caller + thread/起因/意図表） |
| G2 m_pendingIRChange 完全追跡 | **PASS**（writer 1 / reader 3 / clear 2 — 意味論 = 抑止命令 + 昇格指示の複合と確定） |
| G3 因果チェーン line-level 証明 | **PASS**（[b6] 必須因果確定・[b8] 中継点は候補絞り込み＋診断 1 点を明記） |
| G4 意味論分離 | **PASS**（4 概念分離・evaluate への flag 非依存を実証） |
| G5 世代因果 replay | **PASS**（fingerprint 不変 = 純フィードバックの実証） |
| G6 閉ループ有無 | **PASS** — 閉ループ存在を確定（中継点の最終確定に診断 1 点） |
| G7 INV-XFADE-4 再評価 | **PASS**（廃止・INV-IRFLAG-1/2 へ再定義） |
| G8 INV-IRFLAG-1/2 成立条件 | **PASS**（line-level: DSPTransition.h:123 無条件設定が核） |
| G9 最小修正箇所 1 つに限定 | **PASS**（DSPTransition.h:123） |
| G10 相互作用確認 | **PASS**（非干渉・統合パッチ方針） |
| G11 production change = 0 | **PASS** |
| G12 Phase-II = 0 | **PASS** |

**D124: PASS（12/12）** → **D125（Unified Lifecycle + IR-Origin Repair Contract）** へ進行可能。

## 生成物

- 本ファイル（`evidence/D124_IRFLAG_CAUSALITY_AUDIT.md`）
- `evidence/D123_g4_6pub.log`（replay 元データ）
- production source 変更: **0**
