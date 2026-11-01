# D-02 実装報告 — U5 測定構造の改善（案3＋案1）

- 対象: `src/audioengine/RtMeasureRing.h`、`src/tests/AudioEngineHarness/RtMeasureRingTests.cpp`
- 前提: **合格条件（無音 max +50 µs、p99 基準）は不変**。`max_all` の定義・判定方法も不変。
- 制約: U5 の追加測定は**未承認**。本 PR は計測機能の実装・テスト・CI まで。

---

## 1. 実装した内容（案3: 区間の分離）

`BlockScopeTimer` を 3 点計測に変更した。

```
t0: 入口（lfence + rdtsc）          既存
t1: push 直前（rdtscp）             新規
t2: push 直後（rdtscp）             既存（= 従来の終端）
```

| 指標 | 定義 | 用途 |
|---|---|---|
| `t_block` | `t1 − t0` | DSP 本体（診断値） |
| `t_push` | `t2 − t1` | リング push の費用（診断値） |
| `t_total` | `t2 − t0` | **従来と同一の定義**。`max_all` はこれ |
| `max_all` | `max(t_total)` | **判定は従来どおり**（不変） |

- サンプルは `SplitSampleBuffer`（1 件 16 B × 65,536 = 1 MB）に **ペア**で保持する。
  `sampleBuffer()`（t_total）とは別系統で、**同一 dtor 内で push** するため対応が取れる。
- 追加コストは `rdtscp` 1 個（**実測 約 8.5 ns**）。

## 2. 案1（原因帰属）— in-callback API は**採用しない**

Owner 指示「`QueryThreadCycleTime` 等を RT コールバック内で使う場合は RT 安全性とコストを先に検証し、
確認できない場合は ETW 等の外部トレースを優先」に従い、**S-1 の実測**を行った。

| 計測 | median | p99 | 判定 |
|---|---|---|---|
| 空ループ（基準） | 0 ns | 100 ns | — |
| `__rdtscp` | 0 ns | 100 ns | 採用（約 8.5 ns の追加） |
| `QueryThreadCycleTime(NULL)` | **300 ns** | 400 ns | コストは ≤1 µs だが **syscall 経路** |
| `GetThreadTimes` | 300 ns | 400 ns | 同上 |

**判断**: コストは基準内（≤1 µs）だが、**RT コールバック内での安全性を証明できない**ため、
Owner のフォールバック指示に従い **in-callback 呼び出しは行わない**。

代わりに、**外部トレース（ETW / WPR）と突き合わせるための校正点**を prepare 時に 1 回だけ記録する。

- `g_calibTscAtPrepare`（TSC）
- `g_calibQpcNsAtPrepare`（QPC、ns）

`SPLIT` 行に併記するため、TSC → QPC の対応が取れ、ブロック単位の時刻と ETW の
スレッド・スケジューリング イベントを**オフラインで対応付けられる**。
`preemptedBlocks` / `preemptedTicks` は、この対応付けの結果として外部で算出する
（計測器の中に判定値を持たせない）。

## 3. 出力形式（後方互換）

`FINAL` 行は**変更なし**（既存ランナー・検証スクリプトが依存）。
診断行 `SPLIT` を追加する（1 秒 1 行、既存と同じ頻度）。

```
[RTM] FINAL mode=1 n=32003 p99=1245 us max=2071 us dropped=0 writes=34003 ratio=0.0000 overflow=0 warmup_skipped=2000 verdict=OK lower_bound=1
[RTM] SPLIT t_block_p99=1240 us t_block_max=1980 us t_push_p99=0 us t_push_max=2 us t_total_p99=1245 us t_total_max=2071 us n=32003 split_overflow=0 calib_tsc=... calib_qpc_ns=...
```

`SPLIT` は **verdict / max の算出に一切関与しない**（コードレビューで確認可能）。

## 4. 検証結果

| # | 内容 | 結果 |
|---|---|---|
| V-1 | `build-diag` Release ビルド | **成功** |
| V-2 | `RtMeasureRingTests`（新規 `checkSplitInstrumentation` を含む） | **PASS** |
| V-3 | `ctest --test-dir build-diag -C Release -R AudioEngineHarness` | **2 回とも 100% PASS（298 s）** |
| V-4 | 新規テストの内容 | モード 0 で分離サンプル 0／モード 1 で 1 ブロック 1 件／**`t_total == t_block + t_push`（全サンプル一致）**／`push > 0`／校正点が非 0／集計値の単調性 |
| V-5 | 非干渉性（解析） | 追加は `rdtscp` 1 個＝**約 8.5 ns**。ブロック処理 1.25 ms に対し **0.0007%** |
| V-6 | 非干渉性（実測 A/B） | **未実施**（U5 の測定実施が未承認のため） |

### 非干渉性について

- 解析上限: 追加コスト 8.5 ns / ブロック、`t_total` の定義は不変（3 点のうち t0 と t2 は従来と同一）。
- 実測 A/B（同一条件 3 組で `p99` の差 < 1%）は、**測定フェーズの最初の作業**として実施したい
  （測定の実施は Owner の別判断）。

## 5. 変更範囲

| ファイル | 変更 |
|---|---|
| `src/audioengine/RtMeasureRing.h` | `SplitSample` / `SplitSampleBuffer` 追加、`BlockScopeTimer` を 3 点計測へ、`Stats` に診断値、`Aggregator` に分離系列の集計、`SPLIT` 出力、校正点 |
| `src/tests/AudioEngineHarness/RtMeasureRingTests.cpp` | `resetState` に分離バッファ、`checkSplitInstrumentation` 追加と登録 |
| `doc/residual-20261101/D02-MEASUREMENT-DESIGN-20261101.md` | 本報告 |

製品の挙動（既定 OFF・モード 2 の意味）は変更していない。モード 2 でも `rdtscp` が 1 個増えるが、
`SPLIT` の `t_push` は「push が無い」ことを示す 0 近傍の値になる（診断上の意味は保持）。

## 6. 次段階（案2: 環境隔離計画）

Owner 指示に従い、以下を含む計画を作成する（本 PR には含めない）。

- 専用コアと SMT 兄弟スレッドの隔離
- MMCSS Pro Audio の設定
- 測定中に停止する監視アプリ（対象の明示）
- 電源管理条件（高パフォーマンス・コアパーキング）
- 環境の再現性（毎実行の証拠記録）
- 設定変更の復元方法

Defender のスキャン除外は**今回の承認対象外**のため、必要性とセキュリティ影響を別途提示する。
