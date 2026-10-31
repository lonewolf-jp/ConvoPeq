# STG-2 Dynamic Reproduction Gate — 結果：既存seamなし（実行不可）

```text
Authority HEAD : 6dc032133f365ed4ad6a525fe8353c54b7124d8f
Authority file : ConvoPeq.md 5,557,334 B
SHA-256        : 3C61996501C6DA27AD85F36998E746301BC6CBF1BBED48F338C11015A508CD7A
Gate           : Dynamic Reproduction（production 0 / test 0 / implementation NOT STARTED を維持）
 working tree  : src/・ConvoPeq.md clean（pre-existing M AGENTS.md / headroom-proxy-start.ps1 のみ残存・不干渉）
```

## 1. 既存seam調査（全数）

### 1.1 harness main の CLI seam

```text
src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp L1085-1229
  --soak / --buzz* / --p1-char / --p1-recovery-origin / --fpm-m0|m1|m2 /
  --odenom / --scenario= / --measurement= / --t1..t4
  → 任意の staging→order 順序を実行する ad-hoc seam なし。
```

### 1.2 staging setter の全 call site（production＋test）

```text
production:
  StateIO.cpp L77                 mode先行復元→staging（clamp 正適用・問題なし）
  DeviceSettings.cpp L1222        device XML（order は XML に無し・set 時 clamp）
  DeviceSettings.cpp L365         UI editor（set 時 clamp）
  ConvolverControlPanel.cpp L54648 trim のみ（ceiling 無関係）
test:
  BassBuzzMeasurement.cpp L2260   order→bypass→staging（set 時 clamp）
  BassBuzzMeasurement.cpp L1057   -8.0（全 mode valid）
  P1PolyphaseGainCharacterization.cpp L417 configureChain：bypass→staging（set 時 clamp）
  ConvolverStateRoundTripTests.cpp L2288 STG-1：-8.0/+3.0/-4.0（全 mode valid・意図的）
```

→ staging（旧 mode で valid）→ order/bypass 遷移 → getter 検証の順序を持つ
   既存 code は1件も存在しない。既存 binary の実行では Case A/B/Persistence の
   観測値を一切得られない（既存 tests は全 PASS のまま＝STG-2 について無言）。

### 1.3 ファイル駆動 seam の不成立

```text
DeviceSettings XML に processingOrder/bypass が無いため、order 遷移を
  ファイルで駆動できない。ValueTree preset の requestLoadState は
  mode 先行＋clamp のため逆に normalize される（defect の証拠側であり再現器にならない）。
```

## 2. Gate 結果

```text
Case A — ProcessingOrder: NOT TESTED（実行 seam なし）
Case B — Bypass:          NOT TESTED（実行 seam なし）
Persistence:              NOT TESTED（実行 seam なし）
Defect status (dynamic):  NOT CONFIRMED — static candidate は維持（前回 Owner 受理）
```

## 3. §7 先行読取（read-only・実装なし）

次 GO 時の判断材料として副作用のみ読取り：

```text
setInputHeadroomDb（Parameters.cpp L223-241）の副作用：
  jlimit clamp → inputHeadroomDb/inputHeadroomGain/m_currentInputHeadroomDb の publish
  → submitRebuildIntent(Structural, Replaceable) 1 件。
  sendChangeMessage なし。他 staging への波及なし。
  値不変時（差分 ≤1e-5）は publish/rebuild なし（冪等）。
→ order/bypass setter からの re-clamp 呼出しは、変化時のみ Replaceable rebuild を
   1 件追加するだけ。副作用は許容範囲と見込まれるが、最終判定は implementation GO 後に行う。
```

## 4. 要求する次 GO（Owner 宛）

```text
STG-1 と同型の pre-repair FAIL を得るには、一時 regression test（STG-2 R1-R7）の
  追加＋build が必要。これは本 gate の禁止範囲のため未実施。
  test 追加の GO があれば、Case A/B/Persistence を実行し、修正前に FAIL を確定できる。
```

## 5. Phase gate

```text
production source: 0
test: 0
build: 0
implementation: NOT STARTED
commit: NOT STARTED
push: NOT AUTHORIZED
STG-1: CLOSED のまま変更なし
```
