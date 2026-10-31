# STG-6-D1 Post-Commit Gate — PASS

## 1. Commit identity

```text
SHA     : 9afc71deb161b0fd862b21b4f19b4189856cc3b7
message : fix(convolver): report synchronous IR load failures
scope   : 3 files (+297/-1)
  ConvoPeq.md / src/convolver/ConvolverProcessor.LoadPipeline.cpp /
  src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp
```

## 2. Status / scope

```text
residual Ms：AGENTS.md／headroom-proxy-start.ps1 のみ（pre-existing・commit 外・revert なし）。
src/・ConvoPeq.md：残差なし。範囲外 production change：なし。
```

## 3/4. Authority

```text
ConvoPeq.md SHA-256 : 3339398061662BB383023312C298E8D8094D07F10BD00B350AB6E5980B691C38
size                : 5,586,461 B
Generated           : 2026-09-28 11:24:43
HEAD                : 9afc71de
--check             : FRESH（post-commit 再確認済み）
```

## 5-10. Post-commit regression（Debug harness 再実行）

```text
STG-6-D1 PASS（probe: err=IR apply failed (sample rate mismatch)・
  name/len 維持・finalized=1・seq 不変）
null-timeDomainIR 旧 state 保持：コード経路＋全 suite green で維持
  （該当 branch 到達の専用 trigger は存在せず、早期 return＋旧保持を実読確認）
128-sample counterexample：err 非空化を確認（修正前は err 空）
error reporting：非空
publication sequence：不変
STG-1／STG-2／STG-4-1：全 PASS
```

## 11-13. 維持項目（commit は記録のみ・同一 sources／binaries）

```text
Debug：PASS（post-commit 再実行）
Release：PASS（commit 前実行・同一 tree）
CTest：42/42 PASS（commit 前実行・同一 tree・entry 数不変）
```

## 14. Raw atomic／15. 範囲外変更／16. push

```text
新規 raw atomic：0。範囲外 production 変更：なし。push：未実施。
```
