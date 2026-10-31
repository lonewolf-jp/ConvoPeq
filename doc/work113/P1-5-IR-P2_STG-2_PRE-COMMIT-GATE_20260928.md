# STG-2 Pre-Commit Gate — PASS（commit 準備完了・未 commit）

```text
Authority base : 6dc03213 / 3C61996501C6DA27AD85F36998E746301BC6CBF1BBED48F338C11015A508CD7A
```

## §1 diff 範囲（3 files のみ）

```text
M src/audioengine/AudioEngine.Parameters.cpp（+41/-4 相当）
M src/audioengine/AudioEngine.h（private 宣言 1 行）
M src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp（STG-2-A/B/C＋STG-2-R＋runner）
M ConvoPeq.md（再生成・未 commit）
```

## §2 repair contract（diff 実読確認）

```text
inputHeadroomCeilingDb / clampInputHeadroomDb：pure（bool/enum/float→float・noexcept・atomic/I-O なし）
revalidateInputHeadroomForCurrentMode：
  ceiling 取得→clamp→変化時のみ 3 atomic publish→submitRebuildIntent なし
3 setters：既存 submit 1 件を維持＋revalidate 1 call を直前に追加（追加 submit 経路なし）
```

## §3 STG-1 契約

```text
- ceiling 内値（-8.0）維持／超過値のみ clamp／default reset なし（R4/R5 実測）
- applyDefaultsForCurrentMode()：定義のみ残存・call 0（復活・再利用なし）
```

## §4 rebuild uniqueness（source-level）

```text
3 setters とも submit 1 / revalidate 1 / revalidate 内 submit 0（python 再集計で確定）
generations 2->2：余分 rebuild なしの動的証拠（一般化しない）
```

## §5 ISR / boundary

```text
変更なし：RT path / RuntimeWorld / RuntimeStore / Publication / Crossfade /
  Retire-Epoch / Coordinator authority / atomic wrapper policy
新規 code の raw .load/.store/.exchange：0（convo:: wrappers のみ）
変更は NonRT parameter/mode state 正規化に限定
```

## §6 test（前回実行・tree 不変のため維持）

```text
STG-2-A/B/C PASS・STG-2-R（R4/R5/R6）PASS・STG-1 6/6＋StateIO 3/3 PASS
Debug PASS・Release 573/573 PASS・CTest 42/42 PASS
```

## §7 authority 再生成

```text
ConvoPeq.md SHA-256 : EC8B4F0637F8FDCBCCAEDF46D3BD853F0169AE11ECEBFA0FF18BF7B460583856
size                : 5,568,326 B
Generated           : 2026-09-28 07:06:17
HEAD                : 6dc03213（commit なし）
sanity：authority 内 applyDefaultsForCurrentMode() call は宣言 1 件（L8675）のみ。
  bypass setter 内 call なし。revalidate 5 件・STG-2 marker 24 件。
production source diff：Parameters.cpp＋AudioEngine.h 宣言 1 行
test diff：ConvolverStateRoundTripTests.cpp のみ
authority diff：ConvoPeq.md 再生成のみ
```

## §8 commit / push

```text
commit: NOT AUTHORIZED（未実行）
push: NOT AUTHORIZED（未実行）
```

## Stop 条件：該当なし（11 項目すべて陰性）
