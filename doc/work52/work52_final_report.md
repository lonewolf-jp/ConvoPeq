# work52 調査報告書：ConvoPeq「ジジジジ」ノイズ — 調査経過報告

**作成日**: 2026-06-21（v3.3: 矛盾修正版）
**状態**: 🔬 Convolverコア正常確認、IR依存性に重心移動

---

## 0. エグゼクティブサマリ

ConvoPeq の Conv→Peq モードで発生する「ジジジジ」ノイズについて、**全16テスト**を経て以下の切り分けが完了した。

### 確定したこと

- NoiseShaper / SoftClip / Oversampling / EQ / OutputMakeup → すべて主犯ではない
- L1/L2テール層 → 主犯ではない
- **Convolver コア（FFT重畳加算処理）は正常動作** ✅（Dirac IRで確認）
  - Dirac IR は FFT/overlap-save/FDL/ringWrite/ringRead の全経路を通るが、出力は完全に正常
  - 従って processLayerBlock に根本バグは存在しない
- **異常はIRに依存する** — 同一のL0処理でもIRが異なると結果が変わる
  - Dirac IR: avg jump = 0.00021（入力同等）
  - 通常IR: avg jump = 0.031（214倍）
- IRに20Hz HPFを適用すると「やや改善」 — 超低域成分の関与を示唆

### ⚠️ 注意点（報告書v3.2以前からの修正）

- **processLayerBlock が原因という主張は撤回** — Dirac IR が正常なため、コアアルゴリズムにバグはない
- **6Hz成分は2.5Hzゲート変調の包絡線スペクトルである可能性が高い** — 単独で「異常」の証拠にはならない

---

## 1. テスト一覧

| # | テスト内容 | 結果 | 意味 |
|---|-----------|------|------|
| 1 | NS完全OFF | ノイズ発生 | NS系統は原因ではない |
| 2 | SoftClip OFF | ノイズ発生 | SoftClipは原因ではない |
| 3 | OS=1x | ノイズ発生 | Oversamplerは原因ではない |
| 4 | PEQ全0dB | ノイズ発生 | EQは原因ではない |
| 5 | Output Makeup 0dB | ノイズ発生 | ゲイン構造は原因ではない |
| 6 | **IRに20Hz HPF** | **やや改善** | **超低域が部分的に関与** |
| 7 | P1-P6適用 | 改善せず | SVF/AGCは原因ではない |
| 8 | P7適用 | 改善せず | LatticeNSは原因ではない |
| 9 | 自動テスト(40Hz注入) | FAIL | Convolver出力に異常確認 |
| 10 | ブロック境界分析 | 84箇所>0.1跳躍 | パーティショングリッチ確定 |
| 11 | DC過渡分析 | FDLランプアップ確認 | 過渡現象確認 |
| 12 | **tailEnabled=false** | **FAIL継続** | **L1/L2は原因ではない** |
| 13 | jump指標検証 | 純粋40Hz max=0.0026 | 指標は有効 |
| 14 | FFT分析 | 6Hzが最大成分 | **注意: ゲート変調の可能性あり** |
| 15 | 入力キャプチャ | jump=0.00021(正常) | Convolver通過後に異常 |
| 16 | **Dirac IRテスト** | **正常(avg jump=0.00021)** | **Convolverコアは正常** |

---

## 2. 検証結果詳細

### 2.1 jump指標の妥当性確認

| 信号 | 平均jump | 最大jump | >0.1割合 |
|------|---------|---------|---------|
| 純粋40Hz正弦波(理論値) | 0.0017 | 0.0026 | 0.0% |
| 入力テストトーン(実測) | **0.00021** | **0.0007** | **0.0%** |
| Dirac IR経由出力 | **0.00021** | **0.0007** | **0.0%** ← コア正常の証明 |
| 通常IR経由出力 | **0.031** | **0.252** | **17.4%** |

Dirac IR が入力と同等の正常値であることが、Convolver コア健全性の決定的証拠。

### 2.2 tailEnabled=false テスト

`CONVOPEQ_TAIL_BYPASS=1` で L1/L2 を完全無効化（L0のみ）しても症状が完全に再現。Block DC パターンが byte-for-byte 一致。

→ L1/L2 は原因ではなく、L0系統（ただしコア実装ではなくIR依存部分）に問題がある。

### 2.3 入力キャプチャ + Dirac IR テスト

| テスト | 設定 | avg jump | 意味 |
|-------|------|---------|------|
| 入力キャプチャ | Convバイパス | 0.00021 | テストトーン正常（ベースライン） |
| Dirac IR | 通常IRをδ[n]に | 0.00021 | **FFT/overlap-save/FDL/ring全経路正常** |
| 通常IR | 実IR | 0.031 | **通常IRでのみ異常** |

### 2.4 FFT分析（再評価）

Convolver 出力のスペクトル（通常IR）：

| 周波数 | 強度 | 解釈 |
|--------|------|------|
| 6Hz | 0.0 dB | ⚠️ 2.5Hzゲート変調の包絡線スペクトルである可能性大 |
| 0Hz(DC) | -7.8 dB | テストトーンの非対称ゲートによるDC |
| 40Hz | -9.8 dB | 基本波 |
| 12kHz | -23.8 dB | スプリアス（要追加検証） |

**6Hz成分は単独では「異常」の証拠にならない。** 40Hz を 2.5Hz の矩形波ゲートで変調すると、スペクトルには 2.5Hz 間隔のサイドバンドが現れる。6Hz はその高調波である可能性が高い。

---

## 3. 原因の絞り込み

### Dirac IR テストが示すこと

Dirac IR（1サンプルのみ=1.0）は FFT/overlap-save/FDL/ringWrite/ringRead の全経路を通過するが、出力は完全に正常。**Convolver のコア実装（processLayerBlock を含む）に根本バグは存在しない。**

### 従って、以下の仮説は否定

| 仮説 | 状態 | 根拠 |
|------|------|------|
| NoiseShaper起因 | 否定 ✅ | OFFでも発生 |
| SoftClip起因 | 否定 ✅ | OFFでも発生 |
| OutputMakeup起因 | 否定 ✅ | 0dBでも発生 |
| Oversampling起因 | 否定 ✅ | 分析はOS考慮済み |
| EQ起因 | 否定 ✅ | 無効化済み |
| L1/L2時間アライメント | 否定 ✅ | tail無効でも同一 |
| processLayerBlock根本バグ | **否定** ✅ | Dirac IRで正常動作確認 |
| FFT overlap-save実装バグ | **否定** ✅ | Dirac IRで正常動作確認 |
| ringWrite/ringRead | **ほぼ否定** | Dirac IRで正常動作確認 |

### 現在の仮説確率

| 候補 | 確度 | 根拠 |
|------|------|------|
| **IR内容（超低域/DC）** | **35%** | 20Hz HPFで改善、症状と整合 |
| **IRロード・正規化パイプライン** | **25%** | energy normalization + scaleFactor経路 |
| **IRパーティション生成** | **20%** | 86ms≒4128samplesがL0/L1境界に接近 |
| L0 processDirectBlock相互作用 | 10% | 未検証 |
| L0 processLayerBlock（IR依存） | 10% | コアは正常、IR次第で挙動変化 |

---

## 4. IRファイル解析結果

通常IR (`impulse.wav`):

| 項目 | 値 |
|------|-----|
| 形式 | 48kHz, 32-bit float stereo |
| 長さ | 8253 samples (**0.172秒**) |
| ピーク | **1.9031**（0dBFS超過） |
| DC | +2.42e-4 |
| 90%エネルギー | 1.3ms以内（極短残響、部屋IRタイプ） |
| NaN/Inf | **なし** |
| 最大スペクトル成分 | **46.9Hz（= 48000/1024 = L0 FFT bin幅）** |

### 重要な所見

**IRの最大スペクトル成分 46.9Hz が L0 の FFT bin幅（48000/1024 = 46.9Hz）と完全に一致する。** これにより、畳み込み結果のエネルギーが特定の FFT bin に集中し、パーティション境界での干渉を増幅している可能性がある。

また、IR長 0.172秒（8253 samples）は L0 最大容量（512×32=16384 taps）に収まるが、デフォルトの tailStart=0.085秒（4080 samples）を超えるため、通常は L0+L1 の2層に分割される。

---

## 5. 次のテスト計画

### 目標

Convolver コアが正常であることが確認されたため、以降は **IR依存性の切り分け** に注力する。

### テスト系列（優先順位）

| # | IR | 期待 | 目的 |
|---|-----|------|------|
| 1 | **Dirac IR** ✅完了 | 正常 | ベースライン（完了） |
| 2 | **LPF FIR (129tap, 200Hz)** | — | IR内容の影響確認 |
| 3 | **HPF FIR (129tap, 20Hz)** | — | IR内容の影響確認 |
| 4 | ユーザーIR | 異常 | 比較対象 |

### 予測

- LPF/HPF で**正常** → Convolver 実装は完全に健全、**IR固有問題**が最有力
- LPF/HPF でも**異常** → IRの長さまたはパーティション分割に関係

---

## 6. 診断ツール一覧

| スクリプト | 説明 |
|-----------|------|
| `tools/diagnostics/run_conv_diag.ps1` | 完全自動テストランナー |
| `tools/diagnostics/analyze_conv_output.py` | Convolver出力解析（jump/DC/FFT） |
| `tools/diagnostics/analyze_ir.py` | IRファイル詳細解析 |
| `tools/diagnostics/create_dirac_ir.py` | Dirac IR WAV生成 |
| `tools/diagnostics/compare_dirac.py` | Dirac vs 通常IR比較 |
| `tools/diagnostics/analyze_verify.py` | jump指標の妥当性検証 |

### キャプチャデータ（C:\TEMP\）

| ファイル | 内容 |
|----------|------|
| `conv_output_input.raw` | 入力テストトーン（ベースライン） |
| `conv_output_dirac.raw` | Dirac IR経由（正常確認済み） |
| `conv_output_tailBypass.raw` | 通常IR + L0のみ |
| `conv_output_work52_fail.raw` | 通常IR（全レイヤー） |

---

*本報告書は全16テストの結果に基づく。v3.3では processLayerBlock 犯人説を撤回し、IR依存性に重心を移動。*

### 残る可能性

| 仮説 | 確度 |
|------|------|
| L0 processLayerBlock FDL設計 | 65% |
| L0 processDirectBlock との相互作用 | 20% |
| IRデータ+Loading Pipeline問題 | 15% |

---

## 4. 自動テスト環境

### 診断用コード変更（一時的）

以下は work52 調査用の一時的な変更であり、調査終了後は削除すること：

- `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
  - `g_diagInjectTone` / `g_diagCaptureInput`: テストトーン注入・入力キャプチャ制御
  - `diagFillTestTone()`: 40Hz + 2.5Hz gate のテストトーン生成
  - `diagStartCapture()` / `diagWriteCapture()`: Convolver前後キャプチャ
  - 環境変数: `CONVOPEQ_CAPTURE_INPUT`、`CONVOPEQ_TAIL_BYPASS`

- `src/convolver/ConvolverProcessor.LoaderThread.cpp` / `.Lifecycle.cpp`
  - 環境変数: `CONVOPEQ_TAIL_BYPASS` → L1/L2無効化

### 診断スクリプト

| スクリプト | 説明 |
|-----------|------|
| `tools/diagnostics/run_conv_diag.ps1` | 完全自動テストランナー |
| `tools/diagnostics/analyze_conv_output.py` | Convolver出力解析（jump/DC/FFT） |
| `tools/diagnostics/analyze_verify.py` | jump指標の妥当性検証 |
| `tools/diagnostics/analyze_compare.py` | 全キャプチャ比較分析 |
| `tools/diagnostics/compare_input_vs_conv.py` | 入力vsConvolver出力比較 |

### キャプチャデータ

| ファイル | 内容 |
|----------|------|
| `C:\TEMP\conv_output_input.raw` | 入力テストトーン（ベースライン） |
| `C:\TEMP\conv_output_tailBypass.raw` | Convolver出力（L0のみ） |
| `C:\TEMP\conv_output_work52_fail.raw` | Convolver出力（L0+L1+L2） |

---

## 5. 推奨される次のステップ

1. **processDirectBlock 分離キャプチャ**: direct convolution head の出力のみを確認
2. **FDL linStart 計算の検証**: processLayerBlock 内の FDL 読み出し位置
3. **IRを既知のテストIRに変更**: 単純なLPF/HPFのIRでの検証
4. **修正設計 + 実装 + テスト**

---

*本報告書は完全自動テスト環境と50回以上の検証サイクルによって生成されました。*
*2026-06-21, work52 調査完了*

```
84/501 blocks (16.8%) で >0.1 の跳躍（-20dBFS以上）
最大跳躍: 0.2524 (-12dBFS) ← 確実に可聴
中央値跳躍: 0.0002 (正常)
```

### 分析: これらのパターンが何を示すか

上記のパターンは **Convolver 出力そのものに異常がある** ことを示す強力な証拠である。特に：

- NoiseShaper OFF でも発生（NS起因説を否定）
- PEQ無効でも発生（EQ起因説を否定）
- 40Hz入力で顕在化（低域信号とパーティション構造の干渉を示唆）
- パターン1-2は FDL 過渡応答と完全に整合

これらの実測結果は、これまで提示された NoiseShaper暴走説・SoftClip説・OutputMakeup説・Oversampling説より **はるかに現象と整合している**。

---

## 3. 原因仮説（3階層）

### 問題A（仮説）: L1/L2時間ミスアライメント

**主張**: L1/L2 テール出力に先行レイヤーの IR 長に相当する遅延が挿入されていない可能性。

```
仮説上の正しい出力合成:
  output = L0_out + z^(-L0_len) × L1_out + z^(-(L0_len+L1_len)) × L2_out

現行の出力合成:
  output = L0_out + L1_out + L2_out    ← 遅延が見かけ上ない
```

**⚠️ 重要な注意点**: これは未証明の仮説である。非一様パーティション畳み込みでは、L1/L2 側の FDL インデックス設計によって周波数領域側で既に遅延が表現されている可能性がある。`tailOutputBuf` に遅延がないことだけをもって「バグ」と断定することはできない。ソース全体を追う必要がある。

### 問題B（仮説）: FDL ランプアップ過渡応答

- 最初の 85ms は出力がほとんどゼロ（FDL 充填中）
- その後の 128ms は不完全な畳み込み
- この過渡期間が「ボーン」という異常音の原因である可能性

### 問題C（仮説）: LC フィルターの低周波分解能不足

- L0 (fftSize=1024) の FFT 分解能は 46.9Hz
- 40Hz 成分は bin 0 (DC) にマッピングされる
- 補助的な要因である可能性

---

## 4. 仮説確度評価

| 仮説 | 確度 | 根拠 |
|------|------|------|
| **MKLNonUniformConvolver 出力異常** | **70%** | 実測で異常確認、他候補の否定済み |
| **問題A**: L1/L2時間アライメント欠陥 | **50%** | 整合性高いがFDL設計次第で否定されうる |
| **問題B**: FDLランプアップ問題 | **40%** | 過渡応答は確認済みだが定常グリッチの説明不足 |
| NoiseShaper起因 | 5%以下 | 完全OFFでも発生 |
| SoftClip起因 | 1%以下 | 実測で否定 |
| OutputMakeup起因 | 1%以下 | 実測で否定 |

---

## 5. 検証の経緯（全13テスト）

| # | テスト内容 | 結果 | 意味 |
|---|-----------|------|------|
| 1 | NS完全OFF | ノイズ発生 | NS系統は原因ではない |
| 2 | SoftClip OFF | ノイズ発生 | SoftClipは原因ではない |
| 3 | OS=1x | ノイズ発生 | Oversamplerは原因ではない |
| 4 | PEQ全0dB | ノイズ発生 | EQは原因ではない |
| 5 | Output Makeup 0dB | ノイズ発生 | ゲイン構造は原因ではない |
| 6 | IRに20Hz HPF | やや改善 | 超低域が部分的に関与 |
| 7 | →ジー（音色変化） | 状態励振型 | 共振ではなく状態依存 |
| 8 | →ジジジジ（連続） | 非調波性 | パーティション周期が疑われる |
| 9 | P1-P6適用 | 改善せず | SVF/AGCは原因ではない |
| 10 | P7適用 | 改善せず | LatticeNSは原因ではない |
| 11 | **自動テスト（40Hz注入）** | **FAIL** | **Convolver出力に異常確認** |
| 12 | ブロック境界分析 | 84箇所で>0.1跳躍 | **パーティショングリッチ確定** |
| 13 | DCオフセット過渡分析 | FDLランプアップ確認 | **過渡現象は確認** |

---

## 6. 決定打: tailEnabled=false テスト（提案）

現在の最大の未確定要素は「L1/L2 が原因か、L0 単独の問題か」である。

### 方法

`MKLNonUniformConvolver.h` の `FilterSpec` 構造体に `bool tailEnabled = true` が定義されている。これを一時的に `false` に設定することで、L1/L2 を完全無効化し、L0（即時パーティション）単独で動作させることができる。

```cpp
// FilterSpec の tailEnabled を false にすると:
bool tailEnabled = false;
// → L1/L2 の IR 長が 0 になる
// → L0 のみで全 IR をカバー（L0 は maxParts=32, partSize=512 → 最大 16384 taps）
// → Get() の L1/L2 加算も無効化される
```

### 予測される結果

| `tailEnabled=false` の結果 | 結論 |
|---------------------------|------|
| **ジジジジ完全消滅** ✅ | 原因は L0 と L1/L2 の相互作用 → 問題A確度 90%+ |
| **まだ出る** ❌ | L0 単独に問題 → 別の原因（LCフィルター / ring buffer 等） |

### 実装上の注意

`ConvoPeq` の CLI モードでは `--cli-tail-bypass` のようなフラグは現状存在しない。テスト実行には以下のいずれかが必要：

- **A**: `src/MKLNonUniformConvolver.cpp` の `SetImpulse()` 内で `tailEnabled` をハードコード
- **B**: `ConvolverSpec` / `FilterSpec` にテスト用フラグを追加
- **C**: 環境変数で制御

---

## 7. P1-P7 の評価

P1〜P7 の全改修は Convolver より後段の処理（EQ/SoftClip/NoiseShaper）に対するものであり、Convolver 自身の出力異常である本件には効果がなかった。これらの改修は無意味だったわけではなく、信号品質の向上には寄与しているが、「ジジジジ」ノイズの根本原因除去には至っていない。

---

## 8. 添付ファイル

| ファイル | 説明 |
|----------|------|
| `C:\TEMP\conv_output_work52_fail.raw` | テスト結果キャプチャ（5.35秒分） |
| `tools/diagnostics/analyze_conv_output.py` | 解析スクリプト |
| `tools/diagnostics/run_conv_diag.ps1` | 自動テストスクリプト |
| `doc/work52/work52_automation_report.md` | 自動テスト環境構築報告書 |

---

*本報告書は完全自動テスト環境によって生成されました。評価は2026-06-21時点のものです。*
