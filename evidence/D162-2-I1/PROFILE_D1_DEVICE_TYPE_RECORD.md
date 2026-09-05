# D162-2-I1-D — D-1 Device Type Enumeration Record（read-only probe）

```text
Date:     2026-09-05（probe 実行）
Type:     D-1 device type 実測のみ（cycle 開始なし・production 変更 0 / test 0 / build 0）
Baseline: ConvoPeq.md Generated 2026-09-04 23:38:10（R2 A′ 修復込みの最新再生成版・23:14:29 版の
          内容を含む — source 変更なしの再生成）→ 本監査内で baseline 更新を記録
Binary:   build-diag RelWithDebInfo ConvoPeq.exe（Sep 4 23:09 = R2 build・C profile と同一）
判定:     **D-1 GO（4 条件成立）→ D-2〜D-7 実行条件確定**
```

---

## 1. D-1 実測（probe run・`--cli-device-type PROBE_QUERY` で列挙を強制）

### 1.1 `[CLI_AUDIO_DEV_TYPES]` 全文

```text
[CLI_AUDIO_DEV_TYPES] available=Windows Audio,Windows Audio (Exclusive Mode),Windows Audio (Low Latency Mode),DirectSound,ASIO
```

### 1.2 実際に選択可能な device type（5 種）

| # | type 名 |
| --- | --- |
| 1 | Windows Audio |
| 2 | Windows Audio (Exclusive Mode) |
| 3 | Windows Audio (Low Latency Mode) |
| 4 | DirectSound |
| 5 | ASIO |

### 1.3 probe run の検証

- `--cli-device-type PROBE_QUERY`（存在しない値）を与えた結果:
  `[CLI_AUDIO_DEV_SWITCH] unknown requested=PROBE_QUERY ...` — **存在しない type を仮定しても
  crash せず unknown 判定で clean shutdown**（zone: SHUTDOWN_BEGIN → reset completed →
  LOGGER_DETACH/END 全出現・exit 0x0・dump 0・audio callbacks 1,270 回実行）。
  = device type の実在解決機構（MainWindow.cpp:430-448）の防御動作を実証。

## 2. D profile で使用する device type の決定

**決定: `Windows Audio`（type #1）**

**選択理由**:
1. **既存 A/B/C profile と同一の device 系**（デフォルト = Windows Audio 系）であるため、
   D profile の差分を「device lifecycle の有無」に純粋化できる。
2. Shared mode の標準経路であり、XRUN telemetry / CB_ARRIVAL 等の既存観測 channel が
   B/C と同一条件で比較可能。
3. ASIO / Exclusive Mode は driver 固有の挙動（独自 buffer 設計・device close 時の
   driver 停止処理）を含むため、I1 の「shutdown race 再現試験」としてのノイズになる。
   （ASIO 単独での追加検証は I2 以降の候補。）

**D-2〜D-7 実行条件（確定）**:

```text
--cli-run --cli-log-file D{n}.log --cli-device-type "Windows Audio" --cli-exit-ms 15000
（IR / burst / rebuild なし = device lifecycle を主眼にした最小条件）
run 間隔 2s・6 runs（D-2〜D-7）
```

## 3. D-1 GO 判定（4 条件）

| 条件 | 判定 |
| --- | --- |
| `CLI_AUDIO_DEV_TYPES` 実測済み | ✓（§1.1・5 type 実測） |
| 使用 device type を明示的に固定 | ✓（`Windows Audio`・理由 §2） |
| production/test/build 変更 0 | ✓（probe run のみ・スクリプトは evidence 配下のみ） |
| C profile と同一 binary/baseline | ✓（binary Sep 4 23:09 = R2 build・C と同一。baseline は 23:38:10 再生成版に更新 — source 変更なしの再生成で R2 変更込みを確認: `clearDeferredForShutdown` 11 hits・:38401 に A′ 注記） |

**D-1 = GO。** 次工程: D-2〜D-7（`Windows Audio` 固定 × 6 run）。

## 4. 補足

- probe run 自体も 4-channel 契約で確認（exit 0x0 / dump 0 / zone clean / lifecycle: placeholder
  destroy 経路）。`PROBE_QUERY` unknown 時のフォールバック（default device 継続使用）の
  安全性も実証。
- Baseline 更新の記録: `ConvoPeq.md` は 23:14:29 → 23:38:10 に再生成されたが、
  `clearDeferredForShutdown` 11 hits / A′ 注記（md:38401）で R2 変更込みを確認。
  source mtime（ReleaseResources.cpp 23:02）< R2 binary（23:09）のため source-binary 整合。
