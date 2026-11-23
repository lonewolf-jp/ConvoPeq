# ConvoPeq 既存バグの調査・修正計画（READ-ONLY）

- 作業日時: 2026-11-13 11:00 JST
- 種別: **調査と計画の策定のみ**。ソース変更・ビルド・テスト・デバッガ・Replay・Git 操作は行っていない。
- 制約: shutdown 設計サイクル（v2-a・U-5 v5・`B_pre = 1000 ms`）の受け入れ状態を維持し、
  shutdown 本体・Worker 停止 API・所有権／破棄順序・`workersStuck_`・U-3 には触れない。

---

## 1. 調査対象と現行ソースの識別情報

| 項目 | 値 |
| --- | --- |
| HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| ブランチ | `main`（upstream `origin/main`） |
| `ConvoPeq.md` | 7,011,745 B / SHA-256 `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` / Generated `2026-11-12 00:48:16` |
| ConvoPeq.md と作業ツリーの対応 | **不一致**。11-13 の変更（本設計サイクルの計測用変更を含む）を反映していないため、該当ソースは working tree から直接読んだ |
| working tree | 変更 37 ／ 未追跡 103 ／ 合計 140 |

調査対象ファイルの同定（調査開始時に取得した SHA-256 先頭 16 桁）:

| ファイル | サイズ | SHA-256 |
| --- | ---: | --- |
| `src/DeviceSettings.cpp` | 57,742 B | `d76203d622cb65a4` |
| `src/NoiseShaperLearner.cpp` | 75,699 B | `8b02b5caf1517d74` |
| `src/audioengine/AudioEngine.Learning.cpp` | 27,064 B | `559e94597813bb13` |
| `src/audioengine/AudioEngine.h` | 328,290 B | `1c2c0a4cc34beea1` |
| `src/NoiseShaperLearningComponent.cpp` | 22,241 B | `1607b7a06193a5fb` |
| `src/NoiseShaperLearningComponent.h` | 2,474 B | `e23f694d031920c8` |
| `src/InputInspection.cpp` | 8,836 B | `3811875f867e2cc0` |
| `src/MainWindow.cpp` | 80,154 B | `ebc4089363219ae4` |

---

## 2. バグ候補一覧

| ID | 概要 | 確度 | 深刻度 | 修正規模 |
| --- | --- | --- | --- | --- |
| BUG-1 | `getAdaptiveNoiseShaperState()` が `stateMutex` を取らずに 504 B の状態をコピーする（データレース） | **高**（コードで確定。実挙動の再現は未実施） | 中 | 2 行 |
| BUG-2 | 設定保存・読込が学習状態ファイル（278 KB）に比例して約 10 秒かかる | **症状は高**（既存の実測）／**原因は未確定** | 高（体感・終了時間） | 段階的（まず計測） |
| BUG-3 | `saveNoiseShaperState()` が XML 全体を 2 回直列化する | 高 | 低 | 数行 |

既知リスト（`doc/work113/20261109-1855_バグ検証` の B-01〜B-16）は、作業ツリー上では対処済みであることを
複数箇所で確認した（B-05 の深さ検査 = `InputInspection.cpp:15`、B-06 の `parseXmlFile` 経由 = `MainWindow.cpp:1823`、
B-14 の `unique_ptr` 保持 = `MainWindow.cpp:450`、B-03 の sanitize = `DeviceSettings.cpp:41-49`）。
**全 16 件の個別再検証は行っていない**（未検証）。

---

## 3. BUG-1 — `getAdaptiveNoiseShaperState()` の無ロック読み

### 3.1 概要と期待／実際

- 期待: 学習状態（mean 9 + covarianceUpperTriangle 45 + bestCoefficients 9 + その他）は、
  1 世代分が一貫したスナップショットとして読める。
- 実際: 書き手は `stateMutex` を取るが、読み手は取らない。**504 B の構造体を無ロックでコピーする**ため、
  書き込みと重なると mean と covariance が別世代になる（torn read）。

### 3.2 ソース上の根拠

```
// src/audioengine/AudioEngine.Learning.cpp
648: [[nodiscard]] bool AudioEngine::getAdaptiveNoiseShaperState(int bankIndex, convo::NoiseShaperLearnerState& outState) const noexcept
649: {
650:     const auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
651:     outState = bank.state;          // ★ ロックなし
652:     return true;
653: }

655: void AudioEngine::setAdaptiveNoiseShaperState(int bankIndex, const convo::NoiseShaperLearnerState& inState) noexcept
656: {
657:     auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
658:     std::lock_guard<std::mutex> lock(bank.stateMutex);
659:     bank.state = inState;
660: }
```

```
// src/audioengine/AudioEngine.h
3072:     struct AdaptiveCoeffBankSlot
3073:     {
3079:         convo::NoiseShaperLearnerState state {};
3080:         std::mutex stateMutex;      // ★ 非 mutable
```

`NoiseShaperLearnerState` は `double mean[9]` / `double covarianceUpperTriangle[45]` / `double bestCoefficients[9]` ほか
（`NoiseShaperLearnerTypes.h:44-49`）で、合計 504 B。1 回のコピーが分割され得る。

### 3.3 実行スレッド（レースの成立条件）

| 経路 | スレッド | 位置 |
| --- | --- | --- |
| 書き込み（モード切替） | **学習 Worker** | `NoiseShaperLearner.cpp:954` → `handleModeSwitch()` → `:341` の `setAdaptiveNoiseShaperState` |
| 書き込み（世代公開） | Message Thread | `NoiseShaperLearner.cpp:1656`（`callAsync` 内） |
| 読み出し（UI） | Message Thread | `NoiseShaperLearningComponent.cpp:392` |
| 読み出し（保存） | Message Thread | `DeviceSettings.cpp:887`（`saveNoiseShaperState`） |
| 読み出し（学習側） | 学習 Worker | `NoiseShaperLearner.cpp:349`、`:1191`、`:1199`（書いた本人と同じスレッドなのでレースなし） |

**Worker の書き込み（:341）と Message Thread の読み出し（:392 / :887）が重なるとレースになる。**
shutdown 時は `releaseResources` が先に learner を止めるため :887 は安全側だが、
**通常運用中に学習ウィンドウを開いていると :392 が毎回読む**ため、そこで成立する。

### 3.4 影響

- 表示・保存に世代混在の状態が入り得る（B-07 で排除したはずの「世代混交」が読み出し側に残っている）。
- C++ のデータレース（未定義動作）。TSan では検出対象。
- 頻度は低い（504 B のコピーが書込みと重なる必要がある）。**実際の torn read は再現していない**。

### 3.5 最小修正

```
// AudioEngine.h:3080
mutable std::mutex stateMutex;          // const getter から lock するため

// AudioEngine.Learning.cpp:650-651
const auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
std::lock_guard<std::mutex> lock(bank.stateMutex);
outState = bank.state;
```

- 変更 2 箇所・2 行。API 契約（`const` 性・戻り値）は変えない。
- getter は RT から呼ばれない（呼出し元は MT と学習 Worker のみ）。ロック追加による RT 影響はない。
- 取得する mutex は 1 種類のみで、ロック順序の逆転は生じない（setter と同じ mutex）。

### 3.6 変更しない範囲

- バンク構造、係数のダブルバッファ（`CoeffSetWriteLockGuard`）、`storeLearnedCoeffsToBank` の CAS 経路。
- `saveNoiseShaperState` / `loadNoiseShaperState` のロジック。

### 3.7 競合と回帰リスク

- `AudioEngine.Learning.cpp` は**既に未コミット変更あり**（`M`）。適用時は既存変更の上に重ねる。
  `AudioEngine.h` は未変更（クリーン）なので `mutable` 追加は衝突しにくい。
- 回帰リスク: 低。getter が const 文脈から呼ばれる箇所（`DeviceSettings::saveNoiseShaperState(const AudioEngine&)`）でも
  `mutable` により成立する。
- 性能: 180 バンク × ロック 1 回。無視できる。

### 3.8 検証計画

| 項目 | 方法 |
| --- | --- |
| レースの再現 | 「別スレッドで `setAdaptiveNoiseShaperState` を連続実行し、MT 側で `getAdaptiveNoiseShaperState` を回して一貫性（mean/cov の世代タグ）を検査する」専用テストを新設 |
| 回帰 | 既存 `ctest`（`AudioEngineHarness`）全件。とくに学習系・保存系 |
| 静的検査 | TSan が使える構成なら TSan。使えなければコードレビュー + 上記ストレス |
| 確認できないこと | 実機での torn read の発生頻度（確率依存） |

---

## 4. BUG-2 — 設定保存・読込の約 10 秒

### 4.1 症状（既存の実測・再掲）

`doc/work113/20261110-1313_終了遅延調査/report.md` の実測:

| 区間 | 通常（278,331 B） | 小（38 B） | 差 |
| --- | ---: | ---: | ---: |
| 起動 | 12.5 s | 1.8 s | **+10.7 s** |
| `saveSettings` | 10.24 s | 0.05 s（別測定 66 ms） | **+10.2 s** |
| 総経過 | 26.9 s | 5.9 s | +21.0 s |

- 保存窓の CPU 時間 +10.39 s、書込み 294 KB。**書込みは 2 ms**で、時間は書込み前に消費されている。
- 同一バイナリ・同一引数で、変えたのは `%APPDATA%\ConvoPeq\noise_shaper_learn.xml` のサイズだけ。

### 4.2 ソース上の候補（読解による見積り）

| 処理 | 位置 | 計算量の見積り |
| --- | --- | --- |
| `parseXmlFile`（既存 XML の再パース） | `DeviceSettings.cpp:855` → `InputInspection.cpp:297` | O(n) 走査（`computeMaxElementDepth` はバイトポインタ走査に最適化済み、`:15-19`）+ JUCE パース。278 KB で**数十 ms 程度** |
| 180 バンクのマージ（`getChildByName` を 180 回） | `DeviceSettings.cpp:864-904` | 最悪 180² の文字列比較 ≈ 3 万回。**数 ms** |
| `root->toString()` によるサイズ判定 | `:906` | 278 KB を 1 回直列化。**数 ms** |
| `writeTo` | `:907` | 直列化 + 書込み。**数 ms** |
| `setAdaptiveNoiseShaperState` 180 回 | `:985` | mutex 180 回。**µs** |
| 起動側の `loadNoiseShaperState` | `:912-992` | パース + 180 回の読出し。**数十 ms** |

**読解の見積りでは 10 秒を説明できない。** したがって「原因を解消する最小修正」を現時点で特定できない。
測定（CPU +10.4 s、ファイルサイズ比例）は確かであり、**症状は実在するが原因関数が未特定**である。

### 4.3 推定原因（優先順・いずれも未証明）

| # | 仮説 | 支持する事実 | 反証・留保 |
| --- | --- | --- | --- |
| H-1 | `parseXmlFile` の内部（`inspectXmlFile` の文字列処理 or JUCE パース）が想定より遅い | save も load も同じ経路を通り、両方が +10 秒 | コード上は O(n)。実測しないと確定できない |
| H-2 | `saveNoiseShaperState` のマージで、要素数に対し超線形な処理が走る | ファイルサイズ比例で悪化 | 180² では 10 秒にならない |
| H-3 | 環境要因（AV のオンフィルタ等） | ファイルサイズ比例 | プロセス CPU +10.4 s と整合しにくい |

### 4.4 修正方針（段階的）

**Step 1（計測・承認が必要）**: `saveNoiseShaperState` の
パース前 / パース後 / バンクループ後 / `toString` 前後 / `writeTo` 前後、および
`loadNoiseShaperState` のパース前後に `steady_clock` の経過を出す。計測後に削除する。
これで H-1/H-2 を切り分ける。

**Step 2（計測結果に応じた候補）**:

| 案 | 内容 | 期待効果 | 変更範囲 |
| --- | --- | --- | --- |
| 2a | 保存時に既存 XML をパースせず、メモリ上の 180 バンクから直接構築する | パース + マージの削除 | `saveNoiseShaperState` のみ。保存形式は不変 |
| 2b | 変更のあったバンクだけ更新する（差分保存） | 書込み量の削減 | 保存形式に版管理が必要（B-15 と関係） |
| 2c | 保存を非同期化する | 終了時間の短縮 | **shutdown 経路に触れるため今回は対象外**（shutdown 本体は未承認） |

- Step 2a が最小で、既存ファイル互換を保てる。ただし H-1（パース自体が遅い）が原因なら、
  2a は「保存側のパース」を消すだけで、読込側の 10 秒は残る。
- したがって **Step 1 の結果を見るまで Step 2 を確定しない**。

### 4.5 変更しない範囲

- `device_settings.xml` の形式、`convolverState` の保存、`saveSettings` の呼出し順。
- shutdown 経路（`MainWindow` の step 6 の位置、`releaseResources` の順序）。

### 4.6 競合と回帰リスク

- `DeviceSettings.cpp` は**既に未コミット変更あり**（`M`）。適用時は既存変更の上に重ねる。
- Step 2a のリスク: 保存内容が「既存ファイルの他バンクを保持する」挙動を失わないこと
  （メモリ上の 180 バンクが常に全て揃っているとは限らない）。実装時は「メモリに無いバンクは
  既存ファイルから引き継ぐ」規則を明示する必要がある。
- 回帰: 保存ファイルの内容一致テスト（正規化比較）が必須。

### 4.7 検証計画

| 項目 | 方法 |
| --- | --- |
| 症状の再現 | 278 KB の `noise_shaper_learn.xml` を用意し、`--cli-run --cli-exit-ms 2000` の総経過と `step 6 → step 7` の差を測る（既存手順） |
| 修正の効果 | 同じ測定で `saveSettings` が 100 ms 未満になること |
| 回帰 | 保存→再起動→読込で 180 バンクの値が一致（XML 正規化比較）、`ctest` 全件 |
| 確認できないこと | 実機の AV 影響下での挙動（Step 1 で分離できれば解消） |

---

## 5. BUG-3 — `saveNoiseShaperState` の二重直列化

- `DeviceSettings.cpp:906` で `root->toString().length()` を評価してサイズ判定し、`:907` の `writeTo` で
  もう一度直列化する。278 KB の木を 2 回文字列化し、1 回分は即捨てられる。
- 修正: 1 回だけ文字列化してサイズを判定し、その文字列を書く（`file.replaceWithText` 等）。
- 効果: 直列化 1 回分（数 ms）。**BUG-2 の主因ではない**（読解見積り）。
- リスク: 極小。書き込み経路が `writeTo` から文字列書込みに変わるため、改行・エンコーディングの一致を確認する。
- 競合: `DeviceSettings.cpp`（未コミット変更あり）。

---

## 6. 修正方針のまとめ（変更対象）

| ID | 変更ファイル | 変更関数 | 行数見込み |
| --- | --- | --- | --- |
| BUG-1 | `src/audioengine/AudioEngine.h`、`src/audioengine/AudioEngine.Learning.cpp` | `AdaptiveCoeffBankSlot::stateMutex`、`getAdaptiveNoiseShaperState` | 2 |
| BUG-2 | `src/DeviceSettings.cpp`（Step 2a を採る場合） | `saveNoiseShaperState` | 20〜40 |
| BUG-3 | `src/DeviceSettings.cpp` | `saveNoiseShaperState` | 3〜5 |
| （計測） | `src/DeviceSettings.cpp` | 上記 2 関数に一時ログ | 10〜15（計測後に削除） |

BUG-1 は他と独立に実装・検証できる。BUG-2 と BUG-3 は同じ関数を触るため、**同時に承認するなら 1 回の変更にまとめる**のが安全。

---

## 7. 回帰リスク（まとめ）

| ID | リスク | 対策 |
| --- | --- | --- |
| BUG-1 | ロック追加による待ち（getter が高頻度で呼ばれる箇所があるか） | 呼出し元は MT と学習 Worker のみ。UI は 5 Hz、保存は終了時 1 回。影響なし |
| BUG-2 | 保存内容の欠落（メモリに無いバンクを捨てる） | 「メモリ優先 + 無いものは既存から引き継ぐ」規則と、正規化比較テスト |
| BUG-3 | 改行・エンコーディングの差 | 同一文字列を書くので差は出ない。テストで XML 正規化比較 |

---

## 8. 検証計画（共通）

| 項目 | 内容 |
| --- | --- |
| ビルド | Release（`build.bat Release`）。BUG-1 は Debug でも可 |
| テスト | `ctest --test-dir build -C Release`（既存の AudioEngineHarness 全件） |
| 専用テスト | BUG-1: set/get の別スレッドストレス（新規）。BUG-2/3: 保存→読込の一致、10 秒の再測 |
| スレッド安全性 | BUG-1 は TSan があれば TSan、無ければストレス + レビュー |
| 実行しない | shutdown の変更・再設計（未承認） |

---

## 9. 未解決事項

| # | 内容 |
| --- | --- |
| U-1 | BUG-2 の原因関数が未特定。Step 1 の計測（承認が必要）で切り分けるまで修正案を確定できない。 |
| U-2 | BUG-1 の torn read はコードで確定したが、実挙動では再現していない（確率依存）。 |
| U-3 | B-01〜B-16 の全数再検証は未実施（対処済みの spot check のみ）。 |
| U-4 | `ConvoPeq.md` が作業ツリーを反映していない（11-12 生成）。コード審査は working tree を正とする。 |

---

## 10. 実装 GO が必要な項目

| # | 項目 | 承認が要る理由 |
| --- | --- | --- |
| A-1 | BUG-1 の修正（`mutable` + getter の lock） | `src/audioengine/` の変更。既存未コミット変更の上に重ねる |
| A-2 | BUG-2 Step 1（一時計測ログ） | `DeviceSettings.cpp` の変更（計測後に削除） |
| A-3 | BUG-2 Step 2a（保存時のパース廃止） | Step 1 の結果を見てからの判断 |
| A-4 | BUG-3（二重直列化の解消） | A-3 と同時に実施するか個別かに分ける |

いずれも shutdown 関連（本文書の対象外）とは独立している。

---

## 11. 停止条件の確認

- ソース変更・ビルド・テスト・デバッガ・Replay・Git 操作を行っていない。
- 既存の未コミット変更（37 件）と未追跡ファイル（103 件）に触れていない。
- 本文書の作成のみを行った。調査対象ソースのハッシュは §1 のとおりで、調査終了時に再取得して一致を確認した（§12）。

---

## 12. working tree の調査前後

| 項目 | 調査開始 | 調査終了 | 判定 |
| --- | --- | --- | --- |
| HEAD | `173e6995…` | `173e6995…` | 一致 |
| 変更（M） | 37 | 37 | 一致 |
| 未追跡（??） | 103 | 104 | **本文書フォルダの追加分のみ** |
| 対象 8 ファイルの SHA-256 | §1 の表 | 全 8 件が同一 | 一致 |

対象ファイルの終了時ハッシュ（先頭 16 桁）:

| ファイル | SHA-256 | 判定 |
| --- | --- | --- |
| `src/DeviceSettings.cpp` | `d76203d622cb65a4` | SAME |
| `src/NoiseShaperLearner.cpp` | `8b02b5caf1517d74` | SAME |
| `src/audioengine/AudioEngine.Learning.cpp` | `559e94597813bb13` | SAME |
| `src/audioengine/AudioEngine.h` | `1c2c0a4cc34beea1` | SAME |
| `src/NoiseShaperLearningComponent.cpp` | `1607b7a06193a5fb` | SAME |
| `src/NoiseShaperLearningComponent.h` | `e23f694d031920c8` | SAME |
| `src/InputInspection.cpp` | `3811875f867e2cc0` | SAME |
| `src/MainWindow.cpp` | `ebc4089363219ae4` | SAME |

未追跡の増分 1 件は本文書フォルダ `doc/work113/20261113-1300_バグ調査/` である。
ソース・テスト・CMake・Git の状態は不変。
