# STG-4-1 Pre-Commit Gate — PASS（commit 準備完了・未 commit）

```text
HEAD base : de07ba54301697fc877d005a11b07eac8313c8e8
```

## §1 Authority（最終再生成・一致確認）

```text
ConvoPeq.md SHA-256 : 4FF62E0712F1751E2BEAB5E9183920D64E15ADA4F6CBEE1B3DA3C1DA06DE0FC0
size                : 5,580,623 B
Generated           : 2026-09-28 10:02:13
HEAD                : de07ba54
--check             : FRESH（NEWER_SRC_COUNT 0）
反映確認            : requestGenId 4 件・Stg41StderrLogger 残骸 0 件
```

## §2 Root cause / repair contract（diff 実読）

```text
request start（loadImpulseResponse：trash 後に bump／loadIR：既存 entry bump）
  → request generation fixed（LoaderThread.requestGenerationId）
  → finalize（reqGen 転送）→ applyNewState（commit->requestGenId 刻印）
  → executePendingCommit：isCurrentGeneration 照合
current==request → commit（swap・metadata・publish 従来どおり）
current!=request → 破棄（swap・metadata・publish なし。flag 整理のみ。
  engine は PendingCommit デストラクタが retire。double delete/retire/leak なし。
  ownership/retire mechanism 不変）
stale path の副作用：swap・currentIrFile・publication state 変更なし（早期 return）。
```

## §3 myGen sampling（diff 実読）

```text
LoaderThread.cpp L634：myGen = requestGenerationId（request 開始時固定）。
load 途中 sampling なし。ctor／run／runSynchronously／finalize lambda の全経路で
request-fixed identity を使用。mid-load sampling 残存なし。
```

## §4 Cancellation／guard 責務分離

```text
Cancellation：trash＋exit signal＋checkpoint（最適化）。実測で A-loader が
  checkpoint で早期 abort（A-chain 出力なし・B 維持）。
Currency guard：executePendingCommit の無条件 entry check（correctness）。
  post-checkpoint window 到達時も commit 時に拒否する構造。
  ※ guard 発火そのものの動的観測は message-dispatch 競合のため production hook
     なしでは deterministic に強制不能。構造保証（単一 swap 関口＋無条件 check）で担保。
     これは §8「可能なら」の範囲であり stop 条件に非該当。
```

## §5 PendingCommit ownership／cleanup

```text
stale discard は早期 return のみ。commit unique_ptr のデストラクタが
releaseEngine→retireStereoConvolver（既存経路）。新規 delete/retire なし。
isLoading/isRebuilding の false 整理は Phase 4 と同一（parity）。
```

## §6 Rebuild.cpp scope（propagation のみ）

```text
- incremental LoaderThread ctor：現世代刻印（同経路は finalize/commit guard 非経由。
  job.shouldCancel が currency 担保。動作不変）
- incremental finalize applyNewState：現世代刻印（同上）
- stack LoaderThread（rebuildAllIRsSynchronous）：現世代刻印（同期実行・動作不変）
いずれも rebuild semantics 変更なし。ファイル追加なし。
```

## §7 Call sites（全数確認）

```text
loadImpulseResponse（2 ctor calls：bump 伝搬）／loadIR（既存 bump・変更なし）／
finalize（1 caller：reqGen 転送）／applyNewState（4 callers：全て明示 gen。
  default なし・コンパイラ強制）／commitNewConvolver（callers なし・転送対応）／
LoaderThread ctor（3 sites：全て明示 gen）
未初期化／暗黙 default／旧 semantics 残存：なし。
```

## §8 STG-4-1 regression

```text
修正前：A flight確認→B sync確定（name=B len=18546）→A finalize→engine=A（FAIL）
修正後：同順序で B 維持（name=B len=18546・staleOverwrite=0）→ PASS
cancel-early observed（A-chain なし）。guard-fire は構造保証（§4）。
```

## §9 Regression

```text
STG-4-1 PASS／STG-1 6/6＋StateIO 3/3 PASS／STG-2-A/B/C＋R PASS／
existing loader tests PASS（WORK102／PREV-01／M03 系含む CTest 全件）／
Debug PASS／Release PASS／CTest 42/42 PASS
573→551：suite 変更なし（src file 集合同一＝ modifications のみ・CTest 42 entries 同一・
  全 exe link）。fresh-glob 再構成時の build-graph edge regrouping による count 差。
  実質（suite・binary・結果）は同一。
```

## §10 ISR／boundary

```text
RT path／RuntimeWorld／RuntimeStore／Publication／Crossfade／Retire-Epoch／
Coordinator：unchanged。新規 raw atomic（.load/.store/.exchange）：0 件（diff 実測）。
convolverStateGeneration の再利用。新規 authority なし。
```

## §11 Test 整理

```text
残存：STG-4-1 helpers（pump／wav／poll）＋check＋runner（deterministic FAIL/PASS 構造）。
除去済み：一時 logger（Stg41StderrLogger／Guard）・一時 sanity。残骸 0（authority 実測）。
STG-1/STG-2 checks：無変更。
```

## §12 Diff scope（最終）

```text
M src/convolver/ConvolverProcessor.LoadPipeline.cpp
M src/convolver/ConvolverProcessor.LoaderThread.cpp
M src/convolver/ConvolverProcessor.LoaderThreadInline.h
M src/ConvolverProcessor.h
M src/convolver/ConvolverProcessor.Rebuild.cpp
M src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp
M ConvoPeq.md（再生成・未 commit）
pre-existing（不干渉）：AGENTS.md／headroom-proxy-start.ps1
```

## Stop 条件：14 項目すべて陰性。commit 準備完了。

```text
commit: NOT AUTHORIZED（未実行）
push: NOT AUTHORIZED（未実行）
```
