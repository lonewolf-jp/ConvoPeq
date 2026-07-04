# CSwitch WaitTime 抽出結果 — CLI ツールの限界と WPA への誘導

## 実施した抽出方法と結果

| 方法 | 結果 | 理由 |
|------|:----:|------|
| tracerpt → CSV | ❌ CSwitch非デコード | tracerptはCSVでもカーネルMOFイベントをパース不可 |
| xperf dumper (全範囲) | ❌ WaitTime非表示 | dumperは生のMOFバイナリをテキスト化しない |
| xperf dumper (XRUN#1周辺 1秒) | ❌ TID 32352のCSwitchなし | 該当1秒間にAudioスレッドのCSwitchイベントなし |
| WPA GUI | ✅ **唯一の手段** | バイナリMOFを完全デコードしWaitTimeを数値化 |

## 結論

**CSwitch WaitTime（Ready→Running待ち時間）は WPA (Windows Performance Analyzer) GUI でのみ抽出可能です。**

CSwitch イベントはカーネル MOF (Managed Object Format) バイナリとして ETL に記録されており、tracerpt や xperf dumper のテキスト出力では WaitTime フィールドをデコードできません。

## WPA での確認手順

```powershell
wpa doc\work63\ConvoPeqTrace_v2.etl
```

1. 左パネル「Graph Explorer」→ **「Scheduling」→「CPU Usage (Precise)」** をダブルクリック
2. グラフ下のテーブルで **ConvoPeq.exe** を検索・右クリック → **「Filter to Selection」**
3. **TID 32352**（Audioスレッド）を右クリック → 「Filter to Selection」
4. テーブルの列ヘッダを右クリック → **「Select Columns」**
5. 「**Wait Time (us)**」と「**Ready Time (us)**」を追加
6. **XRUN発生時刻との照合**：
   - ConvoPeq.log の `Us` 値を Windows File Time として変換：
   ```
   [datetime]::FromFileTime(89458201460)  # XRUN#1
   [datetime]::FromFileTime(89458303681)  # XRUN#2
   など
   ```
   - WPA 下部のタイムラインを該当時刻にスクロール
   - Ready Time と Wait Time のスパイクを確認

## 参考：今回確認できたこと（CLIのみで）

| 確認項目 | 結果 | 手段 |
|---------|:----:|------|
| CPU 5 可用率 99.9%+ | ✅ | xperf cswitch -avail |
| DPC 567,053件の分布 | ✅ | xperf dpcisr |
| Audio TID 32352 の同定 | ✅ | ConvoPeq.log + ETL照合 |
| ConvoPeq 全スレッドCPU 85.24s | ✅ | xperf cswitch -thread |
| DPC 512μs超は5件のみ | ✅ | xperf dpcisr -bucket |
| **CSwitch WaitTime** | ❌ **WPA GUI要** | CLI全滅 |

## 次の一手

本 ETL (`ConvoPeqTrace_v2.etl`) は完全に有効です。WPA GUI で開けば **CSwitch WaitTime の定量化に加え、XRUN周辺の Ready→Running 遅延の視覚的分析が可能**です。
