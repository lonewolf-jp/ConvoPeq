# STG-2 Test-Only Reproduction — 修正前 FAIL 確定

```text
Authority HEAD : 6dc032133f365ed4ad6a525fe8353c54b7124d8f
Authority file : ConvoPeq.md 3C61996501C6DA27AD85F36998E746301BC6CBF1BBED48F338C11015A508CD7A（不変）
Gate           : test addition + build + pre-repair reproduction のみ
production source: 0（AudioEngine.Parameters.cpp / StateIO.cpp / RuntimeBuilder.cpp 無変更）
```

## Test 追加内容（test-only）

```text
src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp のみ
  checkSTG2OrderStaleHeadroom   (STG-2-A: staging → order 遷移)
  checkSTG2BypassStaleHeadroom  (STG-2-B: staging → bypass 遷移)
  checkSTG2SaveLoadNormalization(STG-2-C: save/load 非冪等)
  runner: A/B/C 全件実行・記録（早期 return なし・新 CTest 登録なし）
STG-1 tests：無変更（意味・期待値とも不変）
```

## Build

```text
方式: build.bat identity gate が cl 表記差異で fail-closed したため、
  fresh configure 済み build dir に対し ninja 直接継続（repo 変更なし・再 configure なし）。
  JUCE header race は増分継続で解消（既知）。
結果: [549/550] link 完了・Debug\AudioEngineHarness.exe 再生成（23:46）
```

## 実測結果（Debug harness・exit 1＝期待通り修正前 FAIL）

```text
[STG-2-A] FAIL: stale headroom above new ceiling: got 0.000000 want <= -6.0
FAIL: checkSTG2OrderStaleHeadroom
[STG-2-B] FAIL: stale headroom above new ceiling: got 0.000000 want <= -6.0
FAIL: checkSTG2BypassStaleHeadroom
[STG-2-C] live/order=ConvThenEQ saved=0.000000 loaded=-6.000000
[STG-2-C] FAIL: save/load non-idempotent: saved 0.000000 loaded -6.000000
FAIL: checkSTG2SaveLoadNormalization
```

## 先行 checks（全 PASS・STG-1 含む）

```text
AC-A1-1/2/3, AC-A2-1/2, H-02 T1-T4, SR-03, H-01, SR-01 全件,
M-04, M-02, M-01, STG-1 bypass 6/6, STG-1 StateIO 3/3：すべて PASS
→ STG-2 tests の追加は既存契約を破壊していない。
```

## Defect 判定

```text
STG-2 = CONFIRMED（動的実測により確定）
  live invariant violation（A/B: 0.0 > -6.0）＋ persistence non-idempotence（C: 0.0 → -6.0）
```

## Phase gate

```text
production source: 0
test: M src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp（STG-2 A/B/C のみ）
build: Debug PASS（harness exit 1 は期待される修正前 FAIL）
implementation: NOT STARTED（Option A 未承認・未実施）
commit: NOT STARTED
push: NOT AUTHORIZED
STG-1: CLOSED のまま変更なし
```

## 次段階への申送り（Owner §6 関連）

```text
setInputHeadroomDb() の副作用（実測・読取り確認済み）:
  clamp → 3 atomic publish → submitRebuildIntent(Structural, Replaceable)。
  値不変時は no-op（冪等）。sendChangeMessage なし。他 staging 波及なし。
懸念の二重 rebuild：order/bypass setter は既に Structural/Replaceable intent を
  1 件 submit しており、re-clamp 呼出しは値変化時のみ +1 件（Replaceable のため coalesce）。
  副作用なし helper（pure clamp 関数の抽出）か、単一 rebuild 統合かの選定は
  implementation GO 後に行う。
```
