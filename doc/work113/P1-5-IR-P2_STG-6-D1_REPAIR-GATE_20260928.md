# STG-6-D1 Repair Gate（commit 前停止）

## 1. 修正前 authority

```text
HEAD: 901a6c336bbd9ec83db546028e749cc868f1a26a
ConvoPeq.md: 7FA34F0102CD313ABE990B23D14F8DAB6106D26AE284D7FF972AE56A5B429F7E
size: 5,580,623 B / Generated: 2026-09-28 11:13:58 / --check FRESH
```

## 2. 修正内容

```text
M src/convolver/ConvolverProcessor.LoadPipeline.cpp（+14 のみ）
  - loadIR !exists branch：handleLoadError 追加
  - loadIR no-FFT branch：handleLoadError 追加（convert-null を含む単一報告点）
  - applyComputedIR null／sr-mismatch／no-timeDomainIR：handleLoadError 追加
  - applyComputedIR generation-mismatch：silent のまま（staleness protocol 維持）
  - resampleIR／IRConverter／RuntimeWorld／Publication／Crossfade／Retire／RT：不変
```

## 3. 全 failure branch 監査結果

```text
loadIR: !exists→報告／convert-null→no-FFT branch で単一報告／cache managers
  null（到達不能・同 branch で報告）／apply 内部 4 却下（gen 以外報告）。
handleLoadError：error 設定＋irFinalized=isIRLoaded＋isLoading/isRebuilding=false＋
  coalesced 通知。engine/file/metadata/commit/publish/retire 不変。
  flag クリアは async 契約と同一（parity）。null-timeDomainIR の早期 return は
  「空 commit 確定」から「報告＋旧保持」への是正（開示済み）。
```

## 4. Regression test

```text
checkSTG61LoadFailureReported（既存 harness 内・新 CTest entry なし）：
  baseline B=4096 確定→settle→tiny=128 probe→err 非空・name/len 維持・
  finalized 維持・publication seq 不変を assert。
```

## 5. Counterexample の修正後結果

```text
[STG-6-D1] probe: err=IR apply failed (sample rate mismatch)
  name=stg61_base len=18546 finalized=1 seq=5->5 → PASS
修正前は err 空・無報告（FAIL 相当）。
```

## 6/7. State preservation／error reporting

```text
name/len/finalized 維持・seq 不変。err 非空。
async parity：async は handleLoadError（SR01 が契約証明）。sync も同一関数に統一。
  async 側無変更。
```

## 8/9. Debug／Release

```text
Debug：PASS（harness 全 PASS・exit 0）
Release：PASS（増分 link・全 exe 生成）
```

## 10. CTest

```text
42/42 PASS（entry 数不変。新 test は既存 AudioEngineHarness 内 subtest）。
```

## 11. ISR／authority audit

```text
RT／RuntimeWorld／RuntimeStore／Publication／Crossfade／Retire-Epoch／Coordinator：
  unchanged。新規 raw atomic：0。handleLoadError は message thread 限定経路
  （loadIR／applyComputedIR とも message thread assert 済み）。
```

## 12. Diff scope

```text
M src/convolver/ConvolverProcessor.LoadPipeline.cpp（+14）
M src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp（+134）
M ConvoPeq.md（再生成・未 commit）
pre-existing（不干渉）：AGENTS.md／headroom-proxy-start.ps1
```

## 13. Worktree status

```text
上記のみ。production 追加変更なし。test 追加は STG-6-D1 のみ。
```

## 14. Authority（修正後・未 commit）

```text
ConvoPeq.md SHA-256 : 3339398061662BB383023312C298E8D8094D07F10BD00B350AB6E5980B691C38
size                : 5,586,461 B
Generated           : 2026-09-28 11:24:43
HEAD                : 901a6c33
--check             : FRESH
```

```text
commit: NOT AUTHORIZED（未実行）
push: NOT AUTHORIZED（未実行）
```
