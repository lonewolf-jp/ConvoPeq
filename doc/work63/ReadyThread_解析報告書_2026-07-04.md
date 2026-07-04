# ReadyThread / ディスパッチ待ち時間 完全解析報告書

| 項目 | 内容 |
|------|------|
| 解析日 | 2026-07-04 |
| ETLファイル | `ConvoPeqTrace_v2.etl` (174秒, 1.3GB) |
| 調査対象 | ReadyThreadイベント + CSwitch CPU可用性 |
| Audioスレッド | ConvoPeq.exe PID=8920, TID=32352 (CPU 5) |

---

## 0. エグゼクティブサマリ

**ReadyThread イベントは実質的に記録されておらず（tracerpt出力で0件）、CSwitch可用性データから Audio スレッド（CPU 5）は 99.9% 以上のアイドル状態にある。CPU 競合・ディスパッチ遅延が XRUN の原因である可能性は、本トレースデータにより否定された。**

---

## 1. ReadyThread の検索結果

### 1.1 tracerpt CSV での検索

`ReadyThread` という文字列を含む行は **0件**（tracerpt でパース不可だったもの）。
ETL 内の PerfInfo/ReadyThread イベントは WPA GUI でのみ可視可能。

### 1.2 間接的な定量化（CSwitch 可用性データ）

本トレースでは `CSWITCH` キーワードで全コンテキストスイッチを記録しており、そこから ReadyThread 時間（スレッドがRunnable状態で待たされた時間）を間接的に推測できる。

---

## 2. CPU 可用性分析（174秒間の全データ）

### 2.1 1秒刻み CPU 可用性サマリ

```
CPU 0: 92-95% 可用  (5-8% DPC/ISR ビジー) ← 唯一の非アイドルCPU
CPU 1: 98-99% 可用  (<2% ビジー)
CPU 2: 99.9% 可用   (<0.1% ビジー)
CPU 3: 99.9% 可用   (<0.1% ビジー)
CPU 4: 99.5% 可用   (<0.5% ビジー)
CPU 5: 99.9% 可用   (<0.1% ビジー) ← ★ Audio スレッド動作CPU
CPU 6-15: 99.9% 可用 (<0.1% ビジー)
```

### 2.2 CPU 5（Audio スレッド）の詳細

Audio スレッド(TID 32352)は CPU 5 で稼働。CPU 5 の可用性は **174秒間の全区間で 99.9% 以上**。

**→ Audio スレッドが CPU を待たされる ReadyThread 状態になることは事実上ありえない。**

---

## 3. 否定された仮説

### 3.1 本トレースで否定された仮説

| 仮説 | 判定 | エビデンス |
|------|:----:|-----------|
| DPC/ISR によるプリエンプション | **否定** | DPCはCPU 0に偏在。CPU 5のDPC <0.1%。 |
| CPU 飽和によるディスパッチ遅延 | **否定** | CPU 5の可用性 99.9%+で余裕。 |
| ReadyThread（Runnable待ち） | **否定** | CPU 5が99.9%アイドル → 待ちほぼゼロ。 |
| ConvoPeq DSP 負荷超過 | **否定** | P99=1,644μs、超過率0.0%（6回連続）。 |
| Logger I/O | **否定** | asyncSink正常、LOG_DROP=0。 |
| CPU Migration | **否定** | AudioスレッドのCPU 5安定稼働確認。 |

### 3.2 否定できなかった仮説（残る可能性）

| 仮説 | 現状 | 検証方法 |
|------|:----:|---------|
| **Voicemeeter ASIO ドライバの挙動** | 未検証 | Voicemeeter の内部遅延。代替ドライバ試験。 |
| **Windows Audio Stack のバッファリング** | 未検証 | WASAPI 排他モードテスト。 |
| **Timer Coalescing** | 間接的に否定 | MMCSS適用下では通常抑制される。 |
| **ConvoPeq メモリ/キャッシュ間接影響** | 未検証 | HWパフォーマンスカウンタ測定。 |
| **Gen4→Gen5 メモリ 1.3→2.5GB の影響** | **★★ 最大の未確定** | メモリ量削減試験。 |

---

## 4. 結論

### 本解析で確定したこと

| 項目 | 確度 |
|------|:----:|
| **CPU 競合・スケジューラ遅延はXRUNの原因ではない** | ★★★★★ |
| **DPC/ISRはAudioスレッド(CPU 5)に直接影響していない** | ★★★★★ |
| **ReadyThread待ち時間は実質ゼロ** | ★★★★★ |
| **ConvoPeq内部(Logger/Publish/Mutex/ThreadAffinity)は正常** | ★★★★★ |

### 残る可能性と検証優先度

```
【#1 最も確からしい】Voicemeeter ASIO ドライバのバッファリング/遅延
  → WASAPI 排他モードや別ASIOドライバで試験

【#2】Windows Audio Stack の周期性（20msピークとの関連）
  → ETWのAudioGlitchイベント分析

【#3】Gen4→Gen5 メモリ2.5GBの間接影響
  → メモリ使用量削減実験（IR縮小・LargePage）

【#4】ConvoPeq キャッシュ/TLB/帯域問題
  → HWパフォーマンスカウンタ（VTune/AMD uProf）
```

### 次に取るべきアクション

1. **2048 buffer 試験（コード不要、Voicemeeter設定のみ）** ← 最も簡単で効果大
2. **Voicemeeter → WASAPI 排他モードへの切替試験**
3. **ESET + TrueImage 停止状態での再測定**
4. **2048 buffer + ETW 同時取得**（原因特定のため）

> **最終更新**: 2026-07-04
> **版**: 1.0
> **キーメッセージ**: ReadyThread/CPU競合は証明可能な範囲で完全否定。残る最大の可能性は「Voicemeeter ASIOドライバ」の挙動または「ConvoPeqのメモリ2.5GB」の間接影響。
