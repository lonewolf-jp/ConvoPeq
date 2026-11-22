# #236 S2（設定読込）の改修（2026-11-22）

Issue #236 の原因分析（`doc/work113/20261122-1652_236原因分析/report.md`）で特定した副因、**bulk parameter restore 中の冗長な rebuild 6 回**を修正した。修正後の S2 は 557.7 ms（E2、20 回の中央値）で、固定ゲート 400 ms は達成していない。残りの約 500 ms は音声デバイスの初回初期化で、アプリ側から制御できない。したがって **Issue は解決とせず、未解決のまま維持する**。

## 1. 改修前の測定結果

同一セッションで、同じビルドに `m_isRestoringState` のガードだけを入れる／入れないの 2 つの実行ファイルを作り、E2 を交互に 20 回ずつ実行した。

| 実行ファイル | S2 中央値 | 最小 | 最大 |
| --- | ---: | ---: | ---: |
| 改修前 | 727.0 ms | 623.0 | 927.5 |
| 改修後 | 550.3 ms | 534.7 | 618.1 |

改修前の内訳（E0/E2 各 10 回の中央値、計測専用ビルド）は次のとおり。

| 区間 | E0 | E2 |
| --- | ---: | ---: |
| S2 全体 | 689.8 | 683.2 |
| `initialise()` | 493.0 | 494.3 |
| 　うち `prepareToPlay` | 52.2 | 54.0 |
| 属性の復元（rebuild 6 回を含む） | 152.8 | 136.1 |
| 最終 rebuild | 24.7 | 42.8 |
| 学習 XML の読込 | 0.1 | 8.3 |

## 2. 原因分析の結論

- S2 の約 500 ms は ASIO デバイスの初回初期化である。同じプロセスでの 2 回目の初期化は約 100 ms で、初回だけが約 450 ms 余分にかかる。WASAPI では 280 ms（CPU 16〜31 ms）、ASIO では 568 ms（CPU 328〜344 ms）で、ASIO だけがプロセス内 CPU を約 300 ms 消費する。サンプルレートには依存しない。
- 属性の復元では、`beginBulkParameterRestore()` から `endBulkParameterRestore()` の間に 6 つの setter が **それぞれ同期 rebuild** を実行していた。`endBulkParameterRestore(true)` が最後に 1 回 rebuild するため、**6 回は不要**である。
- 学習 XML（C-24）の増分は 8 ms で、800 ms の説明にならない。

## 3. 改修対象と変更理由

`src/audioengine/AudioEngine.Parameters.cpp` の 6 か所である。

| setter | 変更 |
| --- | --- |
| `setInputHeadroomDb` | 復元中の `submitRebuildIntent` を抑止 |
| `setOutputMakeupDb` | 同上 |
| `setDitherBitDepth` | 同上（同関数の後段にある `!m_isRestoringState` ガードと同じ扱いに揃えた） |
| `setNoiseShaperType` | 同上 |
| `setOversamplingFactor` | 同上 |
| `setOversamplingType` | 同上 |

いずれも `if (!m_isRestoringState)` を 1 行足すだけで、値の publish と `sendChangeMessage()` は従来どおり即時に実行する。`m_isRestoringState` は `beginBulkParameterRestore` / `endBulkParameterRestore` が Message Thread で設定し、同じ setter の別経路（`AudioEngine.StateIO.cpp`）でも既に同じガードが使われている。したがって新しい状態や排他は導入していない。

bulk restore の外では従来どおり毎回 rebuild するため、UI からのパラメーター変更の挙動は変わらない。

## 4. 改修前後の処理フロー

改修前は、デバイス初期化のあとに次の順で進んでいた。

1. デバイスを初期化する（約 494 ms）。`prepareToPlay` が DSP を 1 回 rebuild する（約 54 ms）。
2. `beginBulkParameterRestore()`。
3. dither を設定 → **rebuild（約 45 ms）**。
4. noise shaper 種類 → **rebuild（約 38 ms）**。
5. Adaptive 係数 30 組 → 1.2 ms。
6. fixed noise log → 0 ms。
7. oversampling 倍率 → **rebuild（約 40 ms）**。
8. input headroom → **rebuild（約 65 ms）**。
9. output makeup → **rebuild（約 50 ms）**。
10. auto gain staging → 0 ms。
11. oversampling 種類 → **rebuild（約 47 ms）**。
12. thread priority、Convolver → 0 ms。
13. `endBulkParameterRestore(true)` → 最後の rebuild（約 43 ms）。

改修後は 3・4・7・8・9・11 の rebuild が無くなり、13 の 1 回だけになる。値の publish は 3〜12 で従来どおり行われるため、最終 rebuild は全パラメーターの最新値を反映する。

## 5. 改修後の S2 時間内訳

計測専用ビルドで E0〜E3 を 20 回ずつ測定した（C-24 の条件と均衡ラテン方格をそのまま使用）。E2 の中央値は次のとおり。

| 区間 | 改修前（E2） | 改修後（E2） |
| --- | ---: | ---: |
| S2 全体 | 683.2 | 557.7 |
| `initialise()` | 494.3 | 502.0 |
| 　うち `prepareToPlay` | 54.0 | 57.8 |
| 属性の復元 | 136.1 | 17.6 |
| 最終 rebuild | 42.8 | 27.9 |
| 学習 XML の読込 | 8.3 | 8.2 |
| XML parse | 0.4 | 0.4 |
| スレッド CPU（S2 全体） | 304.7 | 296.9 |

- 属性の復元は 136.1 ms から 17.6 ms になった（rebuild 6 回の削除）。
- `initialise()` は変わらない。デバイス初期化が床である。
- S2 全体の短縮は 125.5 ms（この計測）。交互 A/B では 176.7 ms だった。差はセッション内の負荷変動による。

## 6. E0〜E3 の測定結果

20 回ずつの中央値・p95（最近順位法、20 回で 19 番目）・IQR / 中央値。

| 区間 | 条件 | 中央値 | p95 | IQR/中央値 | 最小 | 最大 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S2 | E0 | 548.4 | 583.8 | 1.6% | 530.5 | 589.5 |
| S2 | E1 | 548.9 | 565.6 | 2.1% | 540.9 | 597.8 |
| S2 | E2 | 557.7 | 574.6 | 2.6% | 540.4 | 643.1 |
| S2 | E3 | 556.8 | 618.6 | 1.5% | 548.9 | 621.9 |
| S3 | E0 | 417.2 | 444.0 | 10.4% | 336.9 | 475.3 |
| S3 | E1 | 401.0 | 441.5 | 8.7% | 357.2 | 454.1 |
| S3 | E2 | 410.2 | 459.3 | 11.1% | 374.3 | 515.2 |
| S3 | E3 | 415.8 | 442.5 | 9.4% | 371.1 | 453.4 |
| S4 | E0 | 27.6 | 31.7 | 4.3% | 26.7 | 33.9 |
| S4 | E1 | 27.8 | 29.1 | 4.5% | 26.8 | 29.7 |
| S4 | E2 | 34.9 | 36.8 | 5.0% | 32.7 | 38.5 |
| S4 | E3 | 34.4 | 35.8 | 4.1% | 32.3 | 42.4 |

再現性はすべて 30% 以内である。

S1（`MainApplication::initialise` の入口から出口まで）の中央値は E0 1,613.1 ms、E2 1,619.8 ms だった。ただし今回の S1 の起点は `main` ではなく `initialise` の入口であり、C-24 の S1 とは定義が違う。**C-24 の値と直接比較してはいけない**。S3 も同様に、終了処理の入口から `timeEndPeriod(1)` の直後までで、プロセス終了ではない。

## 7. 固定ゲートとの比較

ドリフトは前半 10 回と後半 10 回の中央値の差で見積もり、差の絶対値の 20% を超えたら判定不能とした。

| 指標 | 固定値 | 実測 | 判定 |
| --- | ---: | ---: | --- |
| S2 の絶対値（E2 中央値） | 400 ms 以下 | 557.7 ms | **不合格** |
| S2 の再現性（E2 の IQR / 中央値） | 30% 以内 | 2.6% | 合格 |
| S2 の増分（E2 − E0 中央値） | 300 ms 以下 | +9.3 ms | 判定不能（ドリフト 4.2 ms > 1.9 ms） |
| S4 の絶対値（E2 中央値） | 300 ms 以下 | 34.9 ms | 合格 |
| S4 の増分（E2 − E0 中央値） | 150 ms 以下 | +7.2 ms | 合格（ドリフト 0.2 ms 以下） |
| 再測定間の中央値差 | 10% 以内 | 未実施 | 未検証 |

- 再測定間の差は、C-24 の計画が求める「2 日以上あけた 2 回」を今回実施していないため未検証である。1 回だけの測定で合格とは扱わない。
- S2 の増分はドリフト以下で判定できない。学習 XML の寄与が小さいことは §6 の E0 と E2 の差（9.3 ms）で示されるが、判定規則上は「判定不能」である。

## 8. 回帰の確認

| 項目 | 方法 | 結果 |
| --- | --- | --- |
| 保存済み設定の復元 | フィクスチャの XML で起動し、終了時に書かれる `device_settings.xml` の 12 属性を比較 | 12/12 一致（rate 192000、buffer 1024、dither 32、noiseShaper 2、oversampling 2/type 1、headroom -8.0、makeup 3.0、autoGain 1、MMCSS、log 2000、window 8192） |
| Convolver 状態 | 保存後の XML に `<Convolver>` が残るか | 残る |
| 設定ファイルが無い場合 | `device_settings.xml` を削除して起動 | 終了コード 0、`SHUTDOWN_END` まで到達。終了時に設定ファイルを書かない挙動は改修前と同じ（別項目） |
| XML parse 失敗 | 途中で切れた XML を置いて起動 | 終了コード 0、既定値で 11,387 B の設定を書いた |
| 保存デバイスが使えない場合 | 存在しないデバイス名／種別を指定 | 終了コード 0。JUCE が内部で解決し、エラー文字列が返らないため緩和再試行と fallback の分岐は発火しなかった（未検証） |
| 192 kHz / 32 bit 構成 | 全測定がこの構成 | 80 回とも起動・終了が成功 |
| 音声コールバック | 最初の 5 回のコールバックを記録 | 192 kHz / 1024 で約 5.3 ms 間隔で継続 |
| クラッシュ・デッドロック | 80 回（E0〜E3）＋ 40 回（A/B）＋ 機能確認 4 回 | 終了コード 0、`[FAULT]` 0 |

緩和再試行と fallback の分岐は、この環境では JUCE が先に解決してしまうため発火させられなかった。**検証できていない**と記録する。コードは今回変更していない。

## 9. ビルド・静的ゲート

| 項目 | 結果 |
| --- | --- |
| MSVC Release（計測マークなし） | ビルド成功、エラー 0、警告 10 |
| icx Release（計測マークなし、クリーン） | `build.bat Release clean icx app nopause` でビルド成功（`[SUCCESS] Executable created`、警告 158、エラー 0）。同じ exe を `--cli-run --cli-exit-ms 800` で実行し、終了コード 0、`SHUTDOWN_END`、`[FAULT]` 0 |
| `check-audioengine-lint.ps1` | 合格（LINT-AE-001〜014） |
| `check-list-compliance.ps1` | 合格（警告のみ） |
| CLI 起動・終了 | `--cli-run --cli-exit-ms 800` で終了コード 0、`SHUTDOWN_END`、`[FAULT]` 0 |
| 計測用コードの混入 | マークは計測用ビルドだけ。コミットした差分は `AudioEngine.Parameters.cpp` の 6 か所のみ |

## 10. 未解決事項と制約

- **S2 E2 = 557.7 ms で、固定ゲート 400 ms を満たさない。** 内訳の約 500 ms は ASIO デバイスの初回初期化で、アプリ側から削減できない。ASIO を使う限り、S2 の下限は約 500 ms である。
- デバイスを WASAPI に変えると初期化は 280 ms になるが、これは測定条件の変更であり、ゲートを満たすための手段には使わない。利用者の保存設定を変えない限り ASIO のままである。
- ETW は管理者権限が無く取得できなかった（`wpr` 0xC5585011）。ASIO 初期化の内側（JUCE ラッパーとドライバ）の分離は未実施である。
- 2 日以上あけた再測定を実施していない。
- 初回起動（設定ファイル無し）で設定ファイルが保存されない挙動は、改修前から同じで、本件の対象外である。

## 11. 判定

- 改修は副因（復元の重複 rebuild）を除去し、S2 を 727.0 → 550.3 ms（交互 A/B）、683.2 → 557.7 ms（E0〜E3 の計測）に短縮した。
- ただし主因（ASIO デバイスの初回初期化）は残り、S2 は 400 ms を超える。**#236 は解決としない。**
- 原因はアプリ側のコードではなくデバイス初期化にあるため、解決にはデバイス構成の見直し（ASIO 以外の利用、ドライバ側の改善）か、ゲートの再定義が必要である。どちらも Owner の判断事項である。

## 12. 参照

- Issue #236
- 原因分析: `doc/work113/20261122-1652_236原因分析/report.md`
- 生データ: 同フォルダの `campaign236.csv`（80 行）、`ab236.csv`（40 行）、`breakdown_pre.csv`（20 行）、`breakdown_after_fix.csv`（12 行）、`rate_sweep.csv`、`type_sweep.csv`
- 差分: 同フォルダの `fix_parameters.patch`
- ハーネス: `harness236.py`、`campaign.py`、`ab.py`、`functional.py`
