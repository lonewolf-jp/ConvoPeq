# STG-4-1 Implementation Closure（commit 前停止）

```text
Authority base : de07ba54 / ConvoPeq.md EC8B4F0637F8FDCBCCAEDF46D3BD853F0169AE11ECEBFA0FF18BF7B460583856
STG-1/STG-2    : CLOSED のまま変更なし
```

## Root cause

```text
async load（loadImpulseResponse→LoaderThread→finalize→applyNewState→
executePendingCommit）に currency identity が存在せず、commit が無条件 swap。
sync loadIR は bump するが、async 側がそれを認識しない。
さらに LoaderThread の myGen は load 途中サンプリングのため、新 bump を
自世代と誤認し得る（Owner 指摘の構造）。
```

## Repair contract / Currency identity

```text
request 開始時点で bump した generation を LoaderThread に固定し、
finalize→applyNewState→PendingCommit→executePendingCommit へ伝搬。
commit 時に isCurrentGeneration で照合、不一致は破棄（swap・metadata・publish なし、
flag 整理のみ。engine は PendingCommit デストラクタが retire）。
Cancellation（trash＋exit signal＋checkpoint）は最適化として維持。
Currency guard（commit）が正当性。
```

## Changed production files

```text
M src/convolver/ConvolverProcessor.LoadPipeline.cpp
    loadImpulseResponse：request 時 bump＋LoaderThread へ伝搬
    finalize：requestGen パラメータ追加→applyNewState へ転送
    applyNewState：requestGen パラメータ→commit->requestGenId 刻印
    executePendingCommit：stale-generation discard guard（＋flag 整理）
    commitNewConvolver：転送対応（callers なし）
M src/convolver/ConvolverProcessor.LoaderThread.cpp
    ctor：requestGen 固定／myGen sampling を request-fixed に置換／
    runSynchronously：現世代刻印／finalize lambda へ伝搬
M src/convolver/ConvolverProcessor.LoaderThreadInline.h（member＋ctor decl）
M src/ConvolverProcessor.h（PendingCommit.requestGenId＋3 decl）
M src/convolver/ConvolverProcessor.Rebuild.cpp
    incremental LoaderThread：現世代刻印（stepping 経路・動作不変）
    incremental finalize：現世代刻印（job.shouldCancel が currency 担保・動作不変）
```

## Scope 拡張の理由（§8 報告）

```text
LoadPipeline/LoaderThread だけでは requestGen を finalize/commit へ運べないため、
signature 変更の波及（Inline.h／ConvolverProcessor.h decl／Rebuild.cpp 3 call sites）が
必要最小限で発生。RuntimeWorld／Publication／Crossfade／Retire／RT／atomic 政策は不変。
新規 raw atomic：0。新規 authority／state：なし（既存 convolverStateGeneration を再利用）。
```

## Verification

```text
修正前：A flight確認→B sync確定（name=B len=18546）→A finalize→engine=A（FAIL 確定）
修正後：同順序で B 維持（name=B len=18546・staleOverwrite=0）→ PASS
  ※ A-loader は request-fixed identity により checkpoint で早期 abort（A-chain 出力なし）。
     commit guard は post-checkpoint window の第二防壁（構造保証＋他 swap 経路の監査済み）。
STG-4-1：PASS／STG-1 6/6＋StateIO 3/3：PASS／STG-2-A/B/C＋R：PASS
Debug build：PASS／Release build：PASS（551/551）／CTest：PASS（42/42）
通常 async load（Case A相当）：既存 loader 系 tests 全 PASS（回帰なし）
```

## Authority（再生成・未 commit）

```text
ConvoPeq.md SHA-256 : 47CC8E527C8C369FB08E75DA795A506B4A6AB7D1BB189AAEBF5C62085AF967B6
size                : 5,581,155 B
Generated           : 2026-09-28 09:37:12
HEAD                : de07ba54（commit なし）
```

## Phase gate

```text
commit: NOT AUTHORIZED（未実行）
push: NOT AUTHORIZED（未実行）
```
