# STG-4-1 Test-Only Reproduction — CONFIRMED（修正前 FAIL 確定）

```text
Authority HEAD : de07ba54301697fc877d005a11b07eac8313c8e8
Authority file : ConvoPeq.md EC8B4F0637F8FDCBCCAEDF46D3BD853F0169AE11ECEBFA0FF18BF7B460583856
Phase          : test-only reproduction（production 0 / commit 禁止）
```

## Test 追加内容（test-only）

```text
src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp のみ
  helpers: stg41PumpMessages / stg41SleepPump / stg41WriteDeltaWav（RIFF float32 stereo）
           Stg41StderrLogger＋Stg41LoggerGuard（juce::Logger→stderr 中継・RAII）
  checkSTG41StaleAsyncOverwrite（runner 登録・新 CTest なし）
  ※ 一時切り分け sanity は目的達成後に削除済み。STG-1/STG-2 checks 無変更。
```

## 実測結果（Debug harness・exit 1＝期待される修正前 FAIL）

```text
[STG-4-1] phase1 A requested, inFlight poll=1 recheck=1
[STG-4-1] phase2 post: loading=0 finalized=1 name=stg41_B_short len=18546 err=
[STG-4-1] phase2 B effective=1 name=stg41_B_short len=18546
[STG-4-1] phase3 final name=stg41_A_long len=384000 (B len=18546) staleOverwrite=1
[STG-4-1] FAIL: stale async A overwrote sync B: file=stg41_B_short engine=stg41_A_long len=384000
```

## 判定

```text
A flight: confirmed（isLoading 二重 gate）
B sync completion: confirmed（無 pump 即時観測：name=B・len=18546・finalized=1）
A stale finalize: confirmed（phase3 で name=A・len=384000）
effective engine: expected B / actual A
currentIrFile（irName）: B→A に巻き戻り
deterministic: yes（B 観測が pump 前のため順序証明。false-positive 経路なし）
```

## 先行 checks（全 PASS）

```text
AC/H-02/SR-03/H-01/SR-01/M-04/M-02/M-01・STG-1 6/6・STG-1 StateIO 3/3・
STG-2-A/B/C・STG-2-R：すべて PASS（reproduction は既存契約を破壊しない）
```

## 再現過程の知見（次段階への申送り）

```text
1. A は全長有効サンプル必須（末尾無音は trim で短縮され slow にならない）。
2. B は resample 生存サイズ必須（8x OS 下で 128-sample は outLen=0→fallback→
   sr mismatch 却下。本件とは別の edge。B=4096 で解決）。
3. Phase2 の message pump は A-finalize を割り込ませるため禁止（即時観測）。
4. irName が A/B  discriminator（irLength は geometry 駆動で両者 384000 になり得る）。
5. Phase3 15s window（300×50ms）は A 残処理（実測数秒）に十分。flip で早期 return。
```

## Phase gate

```text
production files changed: NONE
test-only files changed: M src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp
build: Debug PASS（harness exit 1 は期待される修正前 FAIL）
implementation: NOT STARTED
commit: NOT AUTHORIZED
push: NOT AUTHORIZED
STG-1/STG-2: CLOSED のまま変更なし
```
