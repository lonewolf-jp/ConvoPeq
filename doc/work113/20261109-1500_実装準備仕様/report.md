# Part 8 実装準備仕様（受諾条件への対応）

- 作成: 2026-11-09 15:00
- 基準 HEAD: `2aecd73e823067e8cf26c2d4a8fa2e4e6c6d3166`（main、working tree クリーン）
- 位置づけ: Owner 決定「Part 8 最終スコープ」の受諾を踏まえ、実装許可後に迷いなく着手できるよう、コミット分離・テスト仕様・警告設計・告知文案を確定する
- 実行権限: 本決定でも実装・実行は未許可（§7）。本報告は**コード変更なし**

## §1 受諾事項の反映

| Owner 条件 | 本報告での対応 |
|---|---|
| I を F/H と同一 PR・別コミット | §2 の 4 コミット構成（E / F / H / I） |
| I の根本原因は未確定のまま記録 | §7 に「未確定」として明記。回避策と原因解明を分離 |
| block 64〜32768 を正式範囲、超過は保証外 | §4 の警告設計と §3 の境界テスト |
| 警告は NonRT で一度だけ、スレッド安全 | §4 に設計とレビュー項目 |
| MAX_BLOCK_SIZE 524288 の幾何健全性を単体テストで固定 | §3 の `F_GeometryAtMaxBlock` |
| 各条件を個別のテスト名で追跡 | §3 のテスト ID 表 |
| CPU・メモリの測定条件を仕様化 | §3.3 に定義を明記 |
| 試作測定と Production 回帰を同一視しない | §3.4 に区別を明記 |
| E 告知は README / Manual / Release Notes | §5 に文案と配置 |

## §2 コミット計画（PR-2）

### 2.1 4 コミット構成

| # | 対象 | 変更内容 | 単独での回帰範囲（実測で確認） |
|---|---|---|---|
| 1 | E | 既定 ON、テスト期待値、BUILD 表示、`build_identity_gate.py`、文書・告知 | `PolyphaseGainMetrics` 全構成 DC 1.000000000、`PolyphaseGainFidelityTests` 全 PASS |
| 2 | F | Bypass の被覆（L0 被覆の 0.12 秒下限を含む）＋等倍ゲイン＋層の有効化 | block 64〜256 は IR 3.0 秒まで、block 512 は IR 1.152 秒まで完全再構成 |
| 3 | H | 幾何の不変条件（倍率を `l1Part ≤ L0 保持長` に制限） | 上記に加え block 1024〜32768 で IR 1.0 秒が完全再構成、既定モードのテール復帰 |
| 4 | I | 倍率上限 4 | block 512 の 3.0 秒 IR まで含めて全範囲が完全再構成 |

### 2.2 単独で全条件を満たさない理由（実測）

F コミット単独では、block 1024 以上で `l1Part = 8 × block` が L0 保持長（0.12 秒 = 5,760 サンプル）を超え、テール層の先頭が欠落する。実測（Bypass、IR 1.0 秒）は次のとおり。

| block | F 単独 | F + H | F + H + I |
|---|---|---|---|
| 64 / 128 / 256 | 3.33e-16 / 3.89e-16 / 3.33e-16 | 同左 | 同左 |
| 512 | 3.89e-16 | 3.89e-16 | 3.89e-16 |
| 1024 | **2.50e-01** | 3.33e-16 | 3.33e-16 |
| 2048 | **2.50e-01** | 3.05e-16 | 3.05e-16 |
| 4096 | **2.50e-01** | 2.78e-16 | 2.78e-16 |
| 8192 | **2.50e-01** | 2.78e-16 | 2.78e-16 |
| 16384 / 32768 | **2.50e-01** | 2.22e-16 | 2.22e-16 |

IR 3.0 秒では、F + H でも block 512 だけが 2.12e-01（I）で、I のコミットで 3.89e-16 になる。F + H の block 64/128 も実測で 3.33e-16 / 3.89e-16（表の「同左」）である。

- したがって各コミットの回帰は「そのコミットで満たせる範囲」を明示して実施し、全条件の合格は 4 コミット目で判定する
- この段階性は PR の説明に明記する。レビューでは、各コミットの範囲が表のとおりであることを確認する

## §3 Production 回帰テスト仕様

### 3.1 新規テストターゲット

- ターゲット名: `TailLayerIntegrityTests`（`add_test` で ctest に登録）
- 構成: production の `MKLNonUniformConvolver` をリンクし、インパルス入力で Add/Get を回す
- 出力: 各ケースの `PASS/FAIL` と測定値（誤差、エネルギー、CPU、メモリ）を 1 行で出す
- 失敗時は該当 ID と測定値が分かる形式にする

### 3.2 テスト ID と判定

| ID | テスト名 | 判定 | 条件 |
|---|---|---|---|
| F-1 | `F_BypassFullReconstruction` | `maxErrFull ≤ 1e-10` | IR 0.5 / 1.0 / 3.0 秒 × block 64,128,256,512,1024,2048,4096,8192,16384,32768、tailMode=2 |
| F-2 | `F_TailLast100ms` | 出力の末尾 100 ms のエネルギーが IR と ±0.1 dB | 同じ組み合わせ |
| F-3 | `F_NoSilentWindow` | IR の各 8192 サンプル窓で `max|out| ≥ 0.5 × max|ir|` | 同じ組み合わせ |
| F-4 | `F_GeometryInvariants` | `l1Part ≤ l0Len`、層数 ≤ 3、`numPartsIR = ceil(len/partSize)` | 同じ組み合わせ |
| F-5 | `F_GeometryAtMaxBlock` | block 524288 で `l1Part = l0Part ≤ l0Len`、`numPartsIR ≤ kL0PartsHardCap` 相当の健全性 | 幾何のみ（音声処理なし） |
| H-1 | `H_DefaultTailLevel` | 固定区間 0.5〜1.0 秒のエネルギーが 1.18 dB ± 0.1 dB | IR 1.0 秒、block 256〜16384、tailMode=1 |
| H-2 | `H_ContourPreserved` | block 64/128 は ±2.5 dB の許容で、無音・欠落がないこと | IR 1.0 秒、block 64/128 |
| H-3 | `H_PreFixReproduction`（証拠項目） | 修正前コミットで block 1024 = −8.06 dB、block 8192 = 無音になることを記録 | テストではなく PR 添付の証拠。§3.4 参照 |
| H-4 | `H_LongIRTail` | IR 3.0 秒で、全 block にテールエネルギーがあり無音区間がない | block 64〜32768 |
| I-1 | `I_FirstPartitionPresent` | `firstErr = -1`（誤差 1e-6 超の最初の位置がない） | IR 3.0 秒 × block 512、tailMode 1/2 |
| I-2 | `I_MultiplierCap` | `l1Part ≤ 4 × l0Part` かつ `l1Part ≤ l0Len` | 全組み合わせ |

### 3.3 測定条件・集計方法（Owner 条件 4）

| 項目 | 定義 |
|---|---|
| サンプリング周波数 | 48,000 Hz |
| IR 長 | 0.5 / 1.0 / 3.0 秒（末尾に識別マーカー） |
| ブロック長 | 64 / 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 / 16384 / 32768 |
| 入力 | インパルス（1.0 → 以降 0） |
| 誤差 | ピークで遅延を整合したうえでの `max|out − ir|`（倍精度） |
| テール固定区間 | IR の 0.5〜1.0 秒に対応する出力区間のエネルギー（dB） |
| CPU | `Add` + `Get` の総壁時計時間 ÷ 処理した音声秒数。5 回の中央値（初回のウォームアップは除外）。ms / 音声 1 秒で表記 |
| メモリ | `SetImpulse` の直前と直後の `PrivateUsage`（`GetProcessMemoryInfo`）の差。1 インスタンスあたり MB |
| 基準構成 | MSVC Release x64、1 スレッド、同一マシン。CPU は他プロセスの負荷を排除した状態で測る |
| 許容値 | CPU ≤ 10 ms/音声 1 秒、メモリ増分 ≤ 32 MB（実測最大 7.16 ms、21.4 MB は測定結果であり、環境非依存の保証値ではない） |

### 3.4 試作測定との区別（Owner 条件 3）

- 本報告までの数値は、リポジトリ外の試作（`TailTrunc`）による**設計選択の証拠**である
- 受入は `TailLayerIntegrityTests` の結果で判定する。試作の数値を受入証拠として代用しない
- 修正前の再現（H-3）は、コミット 2（F）および 3（H）の状態で新規テストを実行し、H-1/H-4/I-1 が失敗することを記録して示す。修正後の値はコミット 4 で示す

## §4 上限超過の警告設計（Owner 条件 2）

### 4.1 設計

```
// MKLNonUniformConvolver.h
static constexpr int kMaxSupportedBlockSize = 32768;   // 実測で保証する上限
...
// メンバ
std::atomic<bool> m_blockSizeWarned { false };
```

```
// SetImpulse（NonRT）の先頭付近
if (blockSize > kMaxSupportedBlockSize
    && !m_blockSizeWarned.exchange(true, std::memory_order_relaxed))
{
    juce::Logger::writeToLog("MKLNonUniformConvolver: blockSize "
        + juce::String(blockSize) + " exceeds the supported maximum ("
        + juce::String(kMaxSupportedBlockSize) + "). Geometry stays valid; performance is not qualified.");
}
```

- 幾何は従来どおり `l1Part ≤ l0Len` を満たす式で計算するため、上限超でも不正な層構成は生成しない
- 警告は `SetImpulse` の中だけに置く。`Add` / `Get`（RT 経路）にはログも分岐追加もしない
- フラグはインスタンス単位。複数スレッドから `SetImpulse` が呼ばれても `exchange` で 1 回に制限する

### 4.2 実装レビュー条件

1. `kMaxSupportedBlockSize` の比較とログが `Add` / `Get` に存在しないこと（grep で確認）
2. `m_blockSizeWarned` の読み書きが `SetImpulse`（NonRT）のみであること
3. RT 経路に新たな分岐・アロケーション・ロックが入っていないこと
4. 524288 の幾何健全性テスト（F-5）が通ること

## §5 E の告知（最終文案）

### 5.1 README（オーバーサンプリング節の直後）

> **Gain convention change (default).** The input oversampler now has unity round-trip gain. Before this change the gain was 0.75 / 0.5625 / 0.421875 at 2× / 4× / 8× (−2.50 / −5.00 / −7.50 dB). Projects made with earlier versions therefore play 2.5–7.5 dB louder at unchanged settings; preset values are not modified. Check output make-up gain and limiter activity after updating. `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` restores the old behaviour.

### 5.2 Manual（`manual/MAIN_WINDOW_EN.md` の Oversampling 節、`_JP.md` は同内容の和文）

> The oversampler applies the corrected gain convention: a full up/down round trip has unity gain at 2×, 4× and 8×. Versions before this change attenuated the signal by 2.50, 5.00 and 7.50 dB respectively.

### 5.3 Release Notes

和文:

> オーバーサンプリングの利得補正を既定で有効にしました。ラウンドトリップ利得は 1.0 になり、従来は 2×/4×/8× で 0.75 / 0.5625 / 0.421875（−2.50 / −5.00 / −7.50 dB）でした。既存のセッションは同じ設定のまま 2.5〜7.5 dB 大きく再生されます。プリセットの数値は変わりません。更新後は出力メイクアップとリミッタの作動を確認してください。

英文:

> Oversampling gain correction is now enabled by default. The round-trip gain is unity; earlier versions applied 0.75 / 0.5625 / 0.421875 at 2× / 4× / 8× (−2.50 / −5.00 / −7.50 dB). Existing sessions play 2.5–7.5 dB louder at unchanged settings. Preset values are unchanged. Re-check output make-up gain and limiter activity.

- 文案の最終承認は Owner の原文確認後とする（決定 §5）。承認後に上記をそのまま配置する

## §6 実装開始時の手順（許可後）

1. ブランチ作成（PR-1 と PR-2 で分ける）
2. PR-1: A → G の順にコミット。G はローカルで DB あり／なし、ヘッダあり／なしを再確認
3. PR-2: E → F → H → I の順にコミット。各コミットで §2.1 の範囲のテストを実行
4. 各 PR の説明に、受入 ID ごとの結果と、修正前の再現（H-3）を添付
5. 最終確認: `ctest` 54/54、`MTNUPCMeasurement` `[NG]=0`、`--n1-analysis` ALL PASS、CI 監査 PASS

## §7 未確定・未検証

- I の根本原因（block 512 で 13 分割以上になると先頭パーティションが落ちる理由）は未特定。倍率上限 4 は回避策であり、原因解明とは分けて扱う
- block 32768 超の性能・資源は未測定（幾何の健全性のみ担保）
- CI の oneAPI 復元所要時間は未測定（ローカル configure 33.9 秒、生成 2.0 秒、ヘッダ 2 秒）
- 3.0 秒 IR の block 64/128 における層分割差の聴感評価は未実施

## §8 結論

- Owner の追加条件（コミット分離、テスト追跡、測定定義、警告の RT 安全、告知配置）は、本仕様で実装可能な形に確定した
- 各コミットの回帰範囲は実測に基づき明示した。全条件の合格は 4 コミット目で判定する
- 実装・commit・PR・merge は、明示的な実行許可まで行わない
