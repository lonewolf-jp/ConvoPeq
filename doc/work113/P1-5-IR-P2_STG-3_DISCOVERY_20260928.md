# STG-3 Concrete Defect Discovery — NO CONCRETE DEFECT FOUND

```text
Authority HEAD : de07ba54301697fc877d005a11b07eac8313c8e8
Authority file : ConvoPeq.md 5,568,326 B
SHA-256        : EC8B4F0637F8FDCBCCAEDF46D3BD853F0169AE11ECEBFA0FF18BF7B460583856
Generated      : 2026-09-28 07:06:17
Phase          : Discovery only（production 0 / test 0）
worktree       : de07ba54 clean（pre-existing M AGENTS.md / headroom-proxy-start.ps1 のみ・不干渉）
```

## 調査範囲（A〜D）

### A. Parameter / mode transition（staging→mode→derived constraint）

| 対象 | clamp | mode 依存 | 遷移時再検証 | 判定 |
|---|---|---|---|---|
| inputHeadroomDb | -12..ceiling | 有（STG-2 済） | revalidate（3 path） | CLOSED |
| outputMakeupDb | 0..12 固定 | 無 | 不要 | clear |
| convolverInputTrimDb | -12..0 固定 | 無 | 不要 | clear |
| saturationAmount | 0..1 固定 | 無 | 不要 | clear |
| oversamplingFactor | whitelist（0/1/2/4/8） | 無 | 不要 | clear |
| oversamplingType / dither / noiseShaper | enum 範囲 | 無 | 不要 | clear（B-3 guards 維持） |
| filter modes（HC/LC/LPF） | enum | 無 | 下記 D-1 で反証 | clear |
| phase/tail/mixed/IR-length | UI delegate | Convolver 内部 | 既存経路（証拠なし） | observation |

### B. StateIO / persistence（save→load→save）

```text
headroom：STG-2-C/R6 で固定点確認済み。
makeup/trim/saturation/order/bypass/filter/dither/enum：いずれも exact round-trip
  または冪等 clamp。非冪等なし。
復元順序：mode 先行→staging（StateIO L38-77）で正しい。STG-2 revalidate は
  直接 publish の Step1 を経由しないため load 経路に干渉なし。
DeviceSettings：trim を XML に持たないが device 設定と preset の分離設計。
  headroom/makeup の sanitize 範囲は setter 範囲と一致。
```

### C. Rebuild consistency（二重 submit／欠落）

```text
setOversamplingFactor/Type・setDitherBitDepth・setNoiseShaperType の複数 submit は
  defer-aware の既存設計（条件付き第 2 submit）。不変条件違反の証拠なし。
filter setters の submit なしは下記 D-1 により正当。
STG-2 の 3 setter は submit 1＋revalidate（submit 0）で確定済み。
```

### D. RT / ISR boundary

```text
D-1（filter-mode 伝搬仮説→反証）：
  仮説：HC/LC/LPF setter が rebuild なし → DSP が旧 mode のまま。
  反証：AudioThread は毎 callback captureAudioThreadParameterSnapshot で
    filter atomics を live 読取り（AudioEngine.h L4110-4112 / L4175-4177）。
    publish-only setter と整合。不具合なし。
D-2（raw atomic scan）：
  `.load/.store/.exchange` は diagnostics・telemetry・RT-local（relaxed＋NOLINT 根拠付き）
  のみ。convo:: wrapper 政策の concrete 違反なし。件数のみで認定せず。
```

## 追加で確認した STG-2 相互作用

```text
- P1Polyphase configureChain（bulk 内 bypass→staging 順）：staging が後のため最終 state 正当。
- BassBuzz probe（order→bypass→staging 順）：同上。
- begin/endBulkParameterRestore：revalidate は冪等・無害。requestLoadState Step1 は
  直接 publish のため revalidate 非経由（load 意味不変）。
- AutoGain ON/OFF 切替：manual atomics は STG-2 により常時 valid。Builder plan 経路 untouched。
```

## 判定

```text
NO CONCRETE DEFECT FOUND
理由：STG-1/STG-2 型の「setter が確立した不変条件を別 mutator が迂回する」構造は
  残存 staging すべてについて不存在を確認。filter 伝搬仮説は live-read により反証。
  rebuild 多重・欠落・persistence 非冪等・raw atomic 違反のいずれも concrete 証拠なし。
  残るのは observation（UI delegate 経路・device/preset 分離・既存 defer 設計）のみで、
  いずれも実害・契約違反の成立なし。
```

## Phase gate

```text
Production changes: NONE
Test-only changes: NONE（reproduction GO 未発行のため test 追加なし・既存 STG-1/STG-2 不変）
commit: NOT AUTHORIZED
push: NOT AUTHORIZED
STG-1/STG-2: CLOSED のまま変更なし
```
