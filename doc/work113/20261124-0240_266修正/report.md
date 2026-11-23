# CodeQL 自作コード 3 件の解消（#266）

作業日時: 2026-11-24 02:40
対象: main `1712983d`

## 1. 対象

`gh api repos/lonewolf-jp/ConvoPeq/code-scanning/alerts?state=open` で 35 件を確認した。`src/**` と `.github/**` にあるのは 3 件で、残り 32 件は `JUCE/**`（編集禁止）である。

| # | ルール | 深刻度 | 場所 |
| ---: | --- | --- | --- |
| 54 | `cpp/overflowing-snprintf` | high | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp:2443` |
| 55 | `cpp/overflowing-snprintf` | high | 同 2449 |
| 52 | `actions/missing-workflow-permissions` | medium | `.github/workflows/soak-ci.yml:38` |

## 2. 修正

### 2.1 snprintf の戻り値加算（54 / 55）

`std::snprintf` の戻り値は「書いた長さ」ではなく「書こうとした長さ」である。切り詰めが起きると `n` がバッファ長を超えて進み、次の `sizeof(buf) - n` が `size_t` でアンダーフローする。

タップダンプの組み立てを、1 片ずつ作って実長でクランプする形に変えた。

- `appendClamped(buffer, capacity, used, text)` を追加（実長だけを加算し、終端を維持）。
- 各要素は 96 バイトの `piece` に `snprintf` してから連結する。

出力は変えていない。通常のタップ数（16 + 11）では 4096 バイトに収まるため、文字列は従来と同一である。

### 2.2 workflow permissions（52）

`soak-ci.yml` に workflow レベルの最小権限を追加した。他の workflow（`sanitizer-ci.yml` など）と同じ形に揃えている。

```yaml
permissions:
  contents: read
```

`paths-ignore` は追加していない。`JUCE/**` の 32 件は本 Issue の対象外で、検出範囲を変えない方針のため。

## 3. 検証

- `AudioEngineHarness` ターゲット: 129 ステップ、エラー 0。
- 実走: `AudioEngineHarness.exe --buzz-probe=delta` が終了コード 0。タップダンプは従来と同じ書式で出力された（`firstTaps:` 16 個 + `nearPeak:` 11 個）。
- CodeQL（`6722b4e3` の解析）: success。オープンなアラートは **35 → 32**、残りは `JUCE/**` の 32 件だけになった（#54 / #55 / #52 はクローズ）。
- マージコミット `807ddf27` の Sanitizer CI（RelWithDebInfo ASan）: success。

## 4. 参考

`JUCE/**` の 32 件は、編集禁止という方針と、ベンダーコードの検出を減らす目的だけで除外しないという条件（#266 本文）に従い、今回は触っていない。
