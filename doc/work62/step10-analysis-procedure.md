# Step 10a/10e/10d: CallbackArrival 解析手順

## 前提条件
- Step 8a (DiagDrain) 完了 ✅
- ConvoPeq を実機（Voicemeeter ASIO）で動作させ、XRUN を再現
- `diagLog` 出力をファイルに保存（asyncSink 経由で自動出力される）

---

## Step 10a: deltaArrival 解析

### 計測式

| 計測値 | 意味 | 算出方法 |
|--------|------|---------|
| `deltaArrival` | JUCE callback の実到着間隔 | `timestampUs(N) - timestampUs(N-1)` |
| `interval` | processBlockDouble start-to-start | `t0_start(N) - t1_end(N-1)`（既存XRUN計測） |
| `deltaArrival - interval` | ConvoPeq入口〜DSP開始までの差分指標（proxy metric） | 測定区間が一致しないため近似値 |

### 判定ロジック

```
Case A: deltaArrival ≈ 5.33ms（期待値通り）
  → recordCallbackArrival() までは正常
  → 遅延は「recordCallbackArrival 〜 processBlockDouble」の間（ConvoPeq内部）
  → deltaArrival - interval でConvoPeq入口〜DSP開始の差分指標

Case B: deltaArrival ≈ 8〜10ms（XRUN interval と同程度）
  → recordCallbackArrival() より前の区間（OS/JUCE 入口）に遅延
  → ETW 結果と突き合わせて特定

注意: Arrival → processBlockDouble開始 の間には
  JUCE内部・RuntimeScope などを含むため、deltaArrival - interval は
  あくまで proxy metric（差分指標）として扱う。
```

### ログからの抽出例

```
[CB_ARRIVAL] cbIdx=12345 ts=1234567890 expected=5333 cpu=4 tid=1234
[CB_ARRIVAL] cbIdx=12346 ts=1234571234 expected=5333 cpu=4 tid=1234
```

deltaArrival = 1234571234 - 1234567890 = 3344μs = 3.34ms
→ 期待値5.33msより短い → early callback（前回の処理が早く終わった）

---

## Step 10e: callbackIndex 相関分析

### 突き合わせ可能なイベント

全 DiagEvent は `eventIndex`（callbackIndex）を持つ。以下の突き合わせが1つのcallbackIndex で可能:

| イベント | ログタグ | 確認できる相関 |
|---------|---------|--------------|
| CPU_MIG | `[CPU_MIG]` | migration直後だけdeltaArrival悪化？ |
| CallbackArrival | `[CB_ARRIVAL]` | 到着間隔の実測値 |
| DSP_TIMING | `[DSP_TIMING]` | Observe前後で到着間隔変化？ |
| XRUN (CB_HIST) | `[CB_HIST]` | XRUN発生時のdeltaArrival |
| DIAG_DRAIN | `[DIAG_DRAIN]` | 同一tickのDrain状態 |

### 分析例

```
同一 callbackIndex で突き合わせ:
  callbackIndex=12345: deltaArrival=9.1ms, CPU_MIG(cpu=4→15), XRUN#42
  callbackIndex=12346: deltaArrival=5.3ms, (CPU_MIGなし)
  → migration直後だけ到着間隔が悪化 → 相関あり
```

### 手順

1. ログから `[CB_ARRIVAL]` 行を抽出 → `cbIdx` と `ts` で deltaArrival 算出
2. `[CPU_MIG]` 行を抽出 → migration発生callbackIndexを特定
3. 両者を `cbIdx` でJOIN → migration有無でdeltaArrival分布を比較
4. `[CB_HIST]` XRUN行も同様にJOIN → XRUN時のdeltaArrival特性を分析

---

## Step 10d: deltaArrival ヒストグラム

### 目的
deltaArrival の分布パターンから原因を推定

```
パターンA: 5.1ms〜5.4ms に集中 + たまに9ms
  → 通常は正常、テールのみ異常（OS Schedulerの一時的停滞）

パターンB: 7.8ms〜8.2ms に集中
  → 固定的なオフセット遅延（MMCSS/Driver/JUCEの定常的オーバーヘッド）

パターンC: 全域に分散（5〜12ms）
  → システム負荷に依存した変動的遅延
```

### 手順（事後スクリプト）

```bash
# 1. CB_ARRIVAL ログから deltaArrival を計算
grep "\[CB_ARRIVAL\]" convopeq.log | \
  awk '{for(i=1;i<=NF;i++){if($i~/^ts=/){print substr($i,4)}}}' | \
  awk 'NR>1{print $1-prev}{prev=$1}' > delta_arrival_us.txt

# 2. ヒストグラム（100μs刻み）
awk '{bucket=int($1/100); count[bucket]++}END{for(b in count)print b*100, count[b]}' \
  delta_arrival_us.txt | sort -n > histogram_100us.txt

# 3. 統計量
awk '{sum+=$1; if($1>max)max=$1; vals[NR]=$1}END{print "N=" NR, "Avg=" sum/NR, "Max=" max}' \
  delta_arrival_us.txt

# 4. P99/P95
sort -n delta_arrival_us.txt | awk '{vals[NR]=$1}END{n=NR; print "P50=" vals[int(n*0.5)], "P95=" vals[int(n*0.95)], "P99=" vals[int(n*0.99)]}'
```

### ビンサイズ切替

| ビンサイズ | 用途 |
|-----------|------|
| 50μs | 高精細分析（5.3ms ±1ms 範囲） |
| 100μs | 標準（推奨） |
| 250μs | 大まかな分布把握 |
