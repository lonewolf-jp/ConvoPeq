# #236 S2（設定読込）の原因分析（2026-11-22）

Issue #236 の S2（`MainWindow` からの設定読込、約 800 ms）について、区間計測とデバイス種別の切り分けで内訳を求めた。ETW は取得できなかった（理由は §5）。その代わりに、QPC の区間計測、スレッド CPU 時間、デバイス種別・サンプルレート・再初期化の比較で原因を帰属した。

## 1. 結論

- S2 の内訳は、**音声デバイスの初期化が約 494 ms**（このうち `prepareToPlay` が約 54 ms）、**設定値の復元が約 136 ms**（bulk restore 中の冗長な rebuild 6 回）、**最終 rebuild が約 43 ms**、**学習 XML の読込が約 8 ms**、XML parse が 0.4 ms である（修正前、E2、10 回の中央値）。
- デバイス初期化の 500 ms は、**ASIO ドライバの初回初期化**が主体である。同じプロセスで 2 回目の `initialise()` は約 100 ms で終わる。WASAPI では 280 ms（CPU 16〜31 ms）、DirectSound では 345〜378 ms、ASIO では 568 ms（CPU 328〜344 ms）だった。ASIO だけがプロセス内 CPU を約 300 ms 消費する。
- サンプルレートを 48 kHz / 96 kHz / 192 kHz に変えても初期化は 540〜578 ms で変わらない。**フォーマット依存ではない**。
- 分類は **1（デバイス初期化が主因）**、副因が **設定復元の重複 rebuild** である。XML の parse と学習 XML の読込は主因ではない（合計 9 ms 未満）。
- 副因は修正できる（別報告 `20261122-1652_236修正/report.md`）。主因はアプリ側から制御できない部分を含むため、**S2 を 400 ms 以下にすることはこのデバイス構成では達成できない**。残存時間と根拠を §6 に記す。

## 2. 基準と参照範囲

- 基準: `origin/main` = `023917b6bcf187878809afc328114745b33678ff`（作業開始時に `git fetch origin main` で確認）。
- 作業場所: `C:\VSC_Project\fix236`（ブランチ `fix/issue-236-s2-latency`）。保護対象 `C:\VSC_Project\ConvoPeq` は変更していない。
- 参照したソース: `src/DeviceSettings.cpp`（`loadSettings`）、`src/MainWindow.cpp`（呼出しと `~MainWindow`）、`src/MainApplication.cpp`、`src/audioengine/AudioEngine.Parameters.cpp`、`AudioEngine.Processing.PrepareToPlay.cpp`、`AudioEngineProcessor.cpp`、`JUCE/modules/juce_audio_devices/audio_io/juce_AudioDeviceManager.cpp`、`native/juce_ASIO_windows.cpp`、`native/juce_WASAPI_windows.cpp`。
- 既存の測定計画: `doc/work113/20261121-2333_C24計画/report.md`（条件 E0〜E3、20 回、均衡ラテン方格、固定ゲート）。
- 既存の測定結果: `doc/work113/20261122-0618_C24計測/report.md`（C-24。S2 E2 = 799.1 ms）。

## 3. 測定環境

| 項目 | 値 |
| --- | --- |
| OS | Windows 11 x64（ビルド 26300） |
| CPU | 11th Gen Intel Core i7-11700K（8 コア） |
| ビルド | MSVC 14.52.37025、Release（`/O2 /Ob2 /arch:AVX2 /GL`、`/MT`）、Ninja |
| 実行 | `--cli-run --cli-exit-ms 1000`（A/B のみ 1000、区間計測は 2000） |
| 音声デバイス | ASIO「Voicemeeter Virtual ASIO」、192,000 Hz、1024 サンプル、32 bit |
| 端末の状態 | Voicemeeter Potato（`voicemeeterpro_x64`）が常駐。`audiodg` も常駐 |

測定材料は実運用の `%APPDATA%\ConvoPeq` から取得した既存フィクスチャを使った（`doc/work113/20261122-0618_C24計測` と同じ）。

| ファイル | サイズ | SHA-256 先頭 |
| --- | ---: | --- |
| `device_settings.xml` | 11,712 B | `a101915f8ae34a1b` |
| `noise_shaper_learn.xml` | 278,394 B | `556068237729cb42` |
| `learned_state.xml` | 850 B | `af93fe777ff9ef59` |

測定前に `%APPDATA%\ConvoPeq` 全体を ZIP に退避した（1,065,111,510 B）。各回の前に条件どおりのファイル状態へ戻し、サイズと SHA-256 を CSV に記録した。80 回すべてで一致した。

## 4. 計測マーカー

計測専用ビルドにだけ `src/M236Trace.h` を置き、`QueryPerformanceCounter` と `GetThreadTimes`（スレッドの kernel+user 時間）を記録した。マークはメモリ上の配列に積み、プロセス終了時にまとめて 1 回だけファイルへ書く。S2 の処理時間にログ出力を混入させないためである。製品コードには含めず、測定後にソースから外した。

| マーク | 位置 | 意味 |
| --- | --- | --- |
| `s2_begin` / `s2_end` | `MainWindow::loadSettings` の前後 | S2 全体 |
| `noise_begin` / `noise_end` | `DeviceSettings::loadSettings` の `loadNoiseShaperState` | 学習 XML の読込 |
| `bulk_begin` | `beginBulkParameterRestore()` の直前 | bulk restore の開始 |
| `close_begin` / `close_end` | `closeAudioDevice()` | デバイスを閉じる |
| `file_begin` / `file_end` | `getSettingsFile()` と存在確認 | 設定ファイルの取得 |
| `parse_begin` / `parse_end` | `XmlDocument::parse` | XML parse |
| `init_begin` / `init_end` | 保存設定による `initialise()` | デバイス初期化 |
| `relaxed_begin` / `relaxed_end` | `setAudioDeviceSetup(relaxedSetup)` | 緩和設定の再試行（今回の発動なし） |
| `fallback_begin` / `fallback_end` | `initialiseDefaultDevice()`（失敗時） | 既定デバイス（今回の発動なし） |
| `channel_begin` / `channel_end` | `ensureUsableChannelSelection` | チャンネル選択 |
| `restore_begin` / `restore_end` | 設定値の復元ブロック | 属性の読込と反映 |
| `dt_end`、`bd_end`、`ns_end`、`log_end`、`os_end`、`hr_end`、`mk_end`、`ags_end`、`ot_end`、`tp_end` | 各 setter の直後 | 復元の内訳 |
| `adaptive_begin` / `adaptive_end` | Adaptive Noise Shaper の係数復元 | 係数 30 組の反映 |
| `convolver_begin` / `convolver_end` | Convolver 状態の復元 | Convolver |
| `prep_begin` / `prep_end` | `AudioEngineProcessor::prepareToPlay` | エンジンの prepare |
| `aeprep_begin` / `aeprep_end`、`rep_*`、`rb_*` | `AudioEngine::prepareToPlay` | prepare の内訳と rebuild |
| `cb1`〜`cb5` | 最初の 5 回の音声コールバック | コールバック開始時刻 |
| `app_init_start` / `app_init_end` | `MainApplication::initialise` | 起動（参考値） |
| `s4_begin` / `s4_end`、`shutdown_start` / `shutdown_end` | 保存と終了 | S3・S4（参考値） |

E0（学習 XML なし）と E2（学習 XML あり、`learned_state.xml` なし）を交互に実行し、ドリフトを相殺した。10 組 20 回の中央値は次のとおり（ミリ秒）。

| 区間 | E0 | E2 |
| --- | ---: | ---: |
| S2 全体 | 689.8（621.3〜763.7） | 683.2（634.3〜782.9） |
| 学習 XML の読込 | 0.1 | 8.3 |
| `closeAudioDevice` | 0.0 | 0.0 |
| ファイル取得と存在確認 | 0.1 | 0.1 |
| XML parse | 0.5 | 0.4 |
| `initialise()` | 493.0 | 494.3 |
| 　うち `prepareToPlay` | 52.2 | 54.0 |
| 属性の復元 | 152.8 | 136.1 |
| 　うち Adaptive 係数 | 1.2 | 1.3 |
| 　うち Convolver | 0.1 | 0.0 |
| 最終 rebuild（`restore_end`→`s2_end`） | 24.7 | 42.8 |
| スレッド CPU（S2 全体） | 296.9 | 304.7 |

- E0 と E2 の差は学習 XML の 8 ms である。復元は E0 のほうが 17 ms 大きいが、これは同じ処理のばらつきの範囲である。
- S2 = `initialise`（prepare を含む）+ 学習 XML + 復元 + 最終 rebuild + 端数 で閉じる。端数は 10 ms 未満である。
- スレッド CPU は S2 全体の約 45% で、残りは待ちである。

## 5. ETW の取得

`wpr -start CPU -start FileIO` を実行したが、次のエラーで開始できなかった。

```
Failed to enable the policy to profile system performance.
Profile Id: CPU.Verbose.Memory, FileIO.Verbose.Memory
（終了コード 0xC5585011）
```

原因は権限である。

- このセッションは整合性レベル Medium（`S-1-16-8192`）で動いている。`BUILTIN\Administrators` には所属しているが、トークンは昇格していない。
- `Performance Log Users`（`S-1-5-32-559`）には所属していない。カーネル ETW の開始には管理者権限が要る。
- UAC は `ConsentPromptBehaviorAdmin=5`（非 Windows バイナリで同意を求める）で、`cmd.exe` の昇格を試したところ同意画面で停止した。無人で承認できないため、昇格は断念した。

したがって ETL は 1 本も取得できていない。WPA による内訳も無い。原因の帰属は §4 の区間計測と §6 の切り分け実験で行った。ETW が無いぶん、**JUCE の ASIO ラッパーとドライバ内部のどちらが CPU を使っているかは分離できていない**。

## 6. デバイス初期化の切り分け

`initialise()` の中はアプリのコードではないため、条件を変えた比較で構造を推定した。各 2〜3 回の実測である。

### 6.1 デバイス種別

同じ設定ファイルで `deviceType` だけを変えた（48 kHz、512 サンプル）。

| 種別 | `initialise` 実測 | スレッド CPU |
| --- | ---: | ---: |
| ASIO（Voicemeeter Virtual ASIO） | 567.8〜568.3 ms | 328〜344 ms |
| Windows Audio（WASAPI） | 279.8〜294.4 ms | 16〜31 ms |
| DirectSound | 344.5〜377.6 ms | 47〜94 ms |

WASAPI は 280 ms のうち CPU が 16〜31 ms しかない。残りは待ち（`audiodg` との IPC など）である。ASIO は 568 ms のうち 328〜344 ms がプロセス内 CPU で、**ドライバの COM 実装（プロセス内）が CPU を消費している**ことを示す。JUCE の ASIO ラッパーも同じスレッドで動くため、両者の分離はできない。

### 6.2 サンプルレート

ASIO のまま `audioDeviceRate` だけを変えた。

| レート | `initialise` | スレッド CPU |
| --- | ---: | ---: |
| 48,000 Hz | 574.8〜577.8 ms | 296.9 |
| 96,000 Hz | 570.2〜577.0 ms | 281.2〜312.5 |
| 192,000 Hz | 540.0〜551.3 ms | 296.9〜312.5 |

差は無い。192 kHz という条件は原因ではない。

### 6.3 同一プロセス内の 2 回目

`loadSettings` の末尾に 2 回目の `initialise(2, 2, xml, false)` を足した（環境変数で有効化、測定専用）。

| 回 | `initialise` | スレッド CPU |
| --- | ---: | ---: |
| 1 回目 | 549.1〜557.0 ms | 328.1 |
| 2 回目 | 100.1〜106.3 ms | 93.8〜109.4 |

**初回だけが約 450 ms 余分にかかる**。ASIO ドライバの初期化（`ASIOInit` と最初のバッファ生成）が 1 プロセスに 1 回必要な処理で、2 回目は約 100 ms で済む。

### 6.4 既定デバイスを先に開く

`saved XML` の初期化の前に `initialise(2, 2, nullptr, true)` を実行した。

| 処理 | 実測 | スレッド CPU |
| --- | ---: | ---: |
| 先の既定デバイス初期化 | 452.0〜460.2 ms | 0〜62.5 ms |
| 続く ASIO 初期化 | 618.4〜624.4 ms | 296.9〜328.1 ms |

先にデバイスを開いても ASIO の初回コストは減らない。デバイス種別の列挙（`scanDevicesIfNeeded`）は `MainWindow.cpp:326` の `applyAsioBlacklist` が S2 より前に済ませているため、S2 の中には含まれていない。**S2 の 500 ms は ASIO の初回初期化そのもの**である。

### 6.5 初期化の内側

`initialise` の前後と `prepareToPlay` の前後を突き合わせると、次の順で進む。

| 区間 | 実測 |
| --- | ---: |
| `init_begin` → `prep_begin` | 487.8〜497.5 ms |
| `prepareToPlay`（`prep_begin` → `prep_end`） | 47.8〜80.1 ms |
| 最初のコールバック（`cb1`） | `init_end` の直後、約 5 ms 間隔で継続 |

つまり 500 ms のうち約 490 ms は `prepareToPlay` より前、すなわち JUCE の ASIO デバイス生成（`ASIOInit`、チャンネル情報の取得、バッファ生成、`start` と最初のコールバック待ち）である。JUCE のコードは `start` 後に最初のコールバックを 10 ms 間隔で最大 3 秒待つ。待ちの一部はこのループである。

## 7. 原因の帰属

| 要因 | 時間（E2、中央値） | 判定 |
| --- | ---: | --- |
| ASIO デバイスの初回初期化（JUCE + ドライバ） | 約 494（`prepareToPlay` を含む） | 主因。アプリ側から制御できない |
| 　うち `prepareToPlay`（rebuild を含む） | 約 54 | 必要。デバイス開始前に DSP を用意する |
| 設定値の復元（属性 + 冗長 rebuild 6 回） | 約 136 | 副因。修正対象（別報告） |
| 最終 rebuild | 約 43 | 必要（復元値を 1 回で反映する） |
| 学習 XML の読込 | 約 8 | C-24 の対象。主因ではない |
| XML parse | 0.4 | 主因ではない |
| その他（ファイル取得、チャンネル確認、端数） | 約 1 | — |

分類は **1（デバイス初期化が主因）** とし、副因として設定復元の重複処理を記録する。C-24（学習 XML）の増分は 8 ms で、800 ms の説明にならない。これは C-24 の判定（原因コード無し）と矛盾しない。

## 8. 未確認・制約

- ETW が無いため、ASIO 初期化の 490 ms のうち JUCE のラッパーとドライバの内訳は分離できていない。分離には管理者権限で `wpr` を実行できる環境が必要である。
- 初回 450 ms が Voicemeeter 固有か、他の ASIO ドライバでも同じかは、他の ASIO デバイスを接続していないため未確認である。
- 測定は 1 台の端末、MSVC Release の 1 構成である。icx では測定していない。
- 起動（S1）と終了（S3）のマークは参考値で、C-24 の区間定義（`main` の先頭、プロセス終了）と起点が違う。C-24 の S1・S3 と直接比較してはいけない。
- 設定ファイルが存在しない初回起動では、終了時に `device_settings.xml` が書かれなかった。修正前後のバイナリで同じ挙動であり、本件の変更によるものではない（別項目）。

## 9. 参照

- Issue #236
- 計画: `doc/work113/20261121-2333_C24計画/report.md`
- 既存測定: `doc/work113/20261122-0618_C24計測/report.md`
- 修正報告: `doc/work113/20261122-1652_236修正/report.md`
- 生データ: 同フォルダの `breakdown_pre.csv`、`breakdown_after_fix.csv`、`campaign236.csv`、`ab236.csv`、`rate_sweep.csv`、`type_sweep.csv`
- 計測用ヘッダ: 同フォルダの `M236Trace.h`、適用スクリプト `instrument.py`
- ハーネス: `harness236.py`、`campaign.py`、`ab.py`、`type_sweep.py`、`rate_sweep.py`、`functional.py`、`etw_capture.py`
