# BUG-1 実装と BUG-2 Step 1 計測

- 作業日時: 2026-11-13 11:00〜12:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **BUG-1 の実装**（Owner 承認 2026-11-13）と **BUG-2 Step 1 の計測**（同承認）。
- 制約: shutdown 本体・Worker 停止 API・所有権／破棄順序・`workersStuck_`・`B_pre`・U-3 は変更していない。

---

## 1. 作業範囲（承認との対応）

| 承認項目 | 実施内容 |
| --- | --- |
| BUG-1: `stateMutex` を `mutable` に | `AudioEngine.h:3080` |
| BUG-1: getter で同じ mutex を取得 | `AudioEngine.Learning.cpp:653` |
| BUG-1: setter・バンク構造・係数公開経路は変更しない | 未変更（確認済み） |
| BUG-2 Step 1: `saveNoiseShaperState` / `loadNoiseShaperState` の区間計測 | 実施（計測後に削除） |
| BUG-2 Step 1: 計測コードの削除 | 実施（§5） |
| BUG-3 | 保留（未着手） |

---

## 2. 事前状態

| 項目 | 値 |
| --- | --- |
| HEAD | `173e6995…` |
| working tree | 変更 37 ／ 未追跡 104 ／ 合計 141 |
| 対象ファイル（事前） | `AudioEngine.h` `1c2c0a4cc34beea1` / `AudioEngine.Learning.cpp` `559e94597813bb13` / `DeviceSettings.cpp` `d76203d622cb65a4` / `CMakeLists.txt` `8b5195d52d662e17` |
| `noise_shaper_learn.xml` | 278,332 B / `0dcdf3d9d024267d`（バックアップ取得済み） |

---

## 3. BUG-1 の実装

### 3.1 差分（2 箇所）

```
// src/audioengine/AudioEngine.h:3080
-        std::mutex stateMutex;
+        mutable std::mutex stateMutex;   // ★ BUG-1: const getter からも lock する
```

```
// src/audioengine/AudioEngine.Learning.cpp:648-656
 [[nodiscard]] bool AudioEngine::getAdaptiveNoiseShaperState(int bankIndex, convo::NoiseShaperLearnerState& outState) const noexcept
 {
     const auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
+    // ★ BUG-1: 書き手（setAdaptiveNoiseShaperState）と同じ mutex を取る。
+    //   取らないと 560 B の構造体コピーが書き込みと重なり、世代混在の torn read になる。
+    std::lock_guard<std::mutex> lock(bank.stateMutex);
     outState = bank.state;
     return true;
 }
```

- setter（`:658-663`）は変更していない。
- バンク構造・`CoeffSetWriteLockGuard`・`storeLearnedCoeffsToBank` は変更していない。
- getter は RT から呼ばれない（呼出し元は Message Thread と学習 Worker のみ）。ロックは 1 種類だけで順序逆転はない。

### 3.2 専用ストレステスト（新規）

- ファイル: `src/tests/AudioEngineHarness/NoiseShaperStateRaceTests.cpp`
- 起動: `AudioEngineHarness` に環境変数 `CONVOPEQ_NS_STATE_RACE_TEST=1`（既定スイートには含めない opt-in）
- 内容: 書き手が世代番号 `g` を mean[9] / bestCoefficients[9] / covarianceUpperTriangle[45] / iteration に一括設定し、
  読み手が「全要素が同じ g か」を検査する。1.5 秒間の競合で torn read を数える。
- 配線: `CMakeLists.txt`（AudioEngineHarness のソースに 1 行）と
  `PublishPipelineIntegrationTests.cpp`（宣言 + 環境変数分岐 3 行）。

**結果（Release）**:

```
NS-RACE: reads=10081142 writes=8061467 torn=0
NS-RACE: PASS (no torn read)
（実行時間 2.2 秒、exit=0）
```

1008 万回の読み出しと 806 万回の書き込みが競合し、torn read は 0 件だった。

### 3.3 既存テスト

`ctest --test-dir build -C Release -j 4` を実行（結果は §9 に追記）。

---

## 4. BUG-2 Step 1 の計測

### 4.1 計測点（`DeviceSettings.cpp` のみ・一時コード）

| タグ | 区間 |
| --- | --- |
| `save_parse` | `saveNoiseShaperState` 入口 → 既存 XML のパース完了 |
| `save_loop` | 180 バンクのマージループ |
| `save_getstate` | ループ内の `getAdaptiveNoiseShaperState` 180 回の累積 |
| `save_tostring` | `root->toString()` によるサイズ判定 |
| `save_write` | `root->writeTo(file)` |
| `save_total` | 関数全体 |
| `load_parse` / `load_loop` / `load_setstate` / `load_total` | `loadNoiseShaperState` の同区間 |

有効化は `CONVOPEQ_BUG2_PROBE=1`（未設定なら `getenv` 判定のみで挙動不変）。出力は CSV へ追記。

### 4.2 結果（4 回、マイクロ秒）

| 区間 | 1 回目 | 2 回目 | 3 回目 | 4 回目 |
| --- | ---: | ---: | ---: | ---: |
| `save_parse` | 7,002 | 7,044 | 6,696 | 6,532 |
| `save_loop` | 8,314 | 8,489 | 8,791 | 8,742 |
| `save_getstate` | 20 | 10 | 20 | 13 |
| `save_tostring` | 1,202 | 1,254 | 1,596 | 1,173 |
| `save_write` | 14,265 | 14,747 | 15,588 | 14,441 |
| **`save_total`** | **30,783** | **31,534** | **32,671** | **30,888** |
| `load_parse` | 7,338 | 7,967 | 7,286 | 7,896 |
| `load_loop` | 2,931 | 2,650 | 2,420 | 2,620 |
| `load_setstate` | 17 | 19 | 26 | 17 |
| **`load_total`** | **10,269** | **10,617** | **9,706** | **10,516** |
| プロセス経過（ms） | 7,229 | 6,332 | 6,324 | 6,298 |

### 4.3 考察

1. **10 秒の症状は現行ツリーで再現しない。** 保存は約 31 ms、読込は約 10 ms、プロセス全体は 6.3〜7.2 秒
   （既存報告 2026-11-10 では総経過 26.9 秒、`saveSettings` 10.24 秒、起動 12.5 秒）。
2. 保存の内訳は `save_write`（14〜16 ms）が最大で、次に `save_loop`（8.3〜8.8 ms）、`save_parse`（6.5〜7.0 ms）。
   いずれもファイルサイズ比例だが、278 KB ではミリ秒域に収まる。
3. `save_getstate` は 10〜20 µs（180 回のロック込み）。**BUG-1 のロック追加による性能影響は無い。**
4. 既存報告の 10 秒は、当時の XML 事前検査が **O(n²)** だった可能性が高い。
   現行の `computeMaxElementDepth`（`src/InputInspection.cpp:15`）は
   「文字位置参照（O(n^2)）をやめ、UTF-8 バイト列をポインタで走査する」と明記された実装で、
   `InputInspection.cpp` 自体が**未追跡ファイル**（HEAD に存在しない）である。
   ただし当時のバイナリとの直接比較はできないため、**原因の断定はしない**（仮説）。

### 4.4 Step 2a の判断材料

- 10 秒の症状が再現しないため、**Step 2a（保存時の再パース廃止）の必要性は低い**。
- BUG-3（二重直列化）の削減量は `save_tostring` の 1.2〜1.6 ms 程度で、これも優先度は低い。
- 一方、`save_write` が最大の区間であり、ファイルが数 MB に育つと再び秒域になり得る。
  将来の安全策としては「保存サイズの上限」「差分保存」「書込みの非同期化」があるが、
  非同期化は shutdown 経路に触れるため**今回の対象外**。

---

## 5. 計測コードの撤去

- `DeviceSettings.cpp` の計測コード（`bug2NowUs` / `bug2Log` / 各タグ出力 / 追加インクルード）を全削除した。
- 削除後の確認:
  - `bug2` / `CONVOPEQ_BUG2` / `steady_clock` / `fopen` の残存 0 件。
  - 再ビルド成功、`CONVOPEQ_BUG2_PROBE=1` で実行しても**出力なし**（計測コードが無いことの確認）。
  - 実行時間 6.8 秒（計測前と同等）。
- **1 バイト差の報告**: 削除後の `DeviceSettings.cpp` は 57,743 B（事前 57,742 B）で、SHA-256 は一致しない
  （事前 `d76203d622cb65a4` → 現在 `56fdea879fc625bb`）。差分は改行・空白レベルの 1 バイトで、
  コード内容は同一（ビルド成功・上記 grep 0 件・実行挙動同一）。原因は、テキストモードでの
  書戻しにより孤立した CR が LF に正規化されたためと考えられる。**事前バイト列の完全復元はできていない。**

---

## 6. 変更ファイル一覧

| ファイル | 変更 | 事前状態 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.h` | 1 行（`mutable`） | クリーン（新規変更） |
| `src/audioengine/AudioEngine.Learning.cpp` | 2 行追加（コメント + lock） | 既に変更あり（M） |
| `src/tests/AudioEngineHarness/NoiseShaperStateRaceTests.cpp` | 新規 | 新規（未追跡） |
| `CMakeLists.txt` | 2 行（コメント + ソース追加） | 既に変更あり（M） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 宣言 + 分岐（4 行） | 既に変更あり（M） |
| `src/DeviceSettings.cpp` | 計測コードの追加と削除（最終的に 1 バイト差） | 既に変更あり（M） |

---

## 7. 事後状態

| 項目 | 事前 | 事後 |
| --- | --- | --- |
| HEAD | `173e6995…` | `173e6995…`（不変） |
| 変更（M） | 37 | 38（`AudioEngine.h` が追加） |
| 未追跡（??） | 104 | 105（専用テスト 1 件） |
| `noise_shaper_learn.xml` | `0dcdf3d9…` | 実行で更新された後、**バックアップから復元して事前と同一**（278,332 B / `0dcdf3d9…`） |

---

## 8. 停止条件の確認

- Git 操作（commit / reset / checkout / clean / stash）を行っていない。
- 承認範囲外のソース・テスト・CMake を変更していない。
- shutdown 本体・Worker 停止 API・所有権／破棄順序・`workersStuck_`・U-3 に触れていない。
- 既存の未コミット変更は保持している（上記 5 ファイルはいずれも既存変更の上に最小限を加えた）。

---

## 9. 既存テストの結果

```
ctest --test-dir build -C Release -j 4
100% tests passed out of 61
Total Test time (real) = 332.08 sec
```

- 61/61 成功（失敗 0）。
- 専用ストレステストは opt-in のため既定スイートには含まれない（§3.2 で単独実行）。
- ビルドは Release（MSVC 14.52.37025 / Ninja Multi-Config）。

## 10. Owner 判断（2026-11-13・受入記録）

| 項目 | 決定 |
| --- | --- |
| BUG-1 | **実装・検証完了として受け入れ**。`stateMutex` の `mutable` 化と getter の同一 mutex 取得、専用ストレステスト PASS、ctest 61/61 を受理。setter・バンク構造・係数公開経路を変更していないことを確認事項として維持 |
| BUG-2 Step 1 | **計測結果を受け入れ**。現行環境で 10 秒は再現せず、保存約 31 ms / 読込約 10 ms であることを確認 |
| BUG-2 Step 2a | **実施しない**。当時の遅延原因は未確定とし、O(n²) の XML 事前検査は**仮説としてのみ**記録。同種の遅延が再発した場合は、その時点の環境で再調査する |
| BUG-3 | **実施しない（保留）**。削減見込み 1.2〜1.6 ms で優先度が低い |
| `DeviceSettings.cpp` の 1 バイト差 | **非意味的差として許容**（孤立 CR の LF 正規化）。ただし**事前バイト列の完全復元には至っていない**ことを記録し、完全復元済みとしては扱わない |
| shutdown 本体・Worker 停止 API・所有権／破棄順序・U-3 | **未承認のまま維持**（本作業の対象外） |
| U-5 v5 の条件付き受け入れ・`B_pre = 1000 ms` | 維持 |

### 10.1 未確定事項（維持）

- BUG-2 の当時の 10 秒遅延の原因は未確定（O(n²) 仮説は反証も立証もされていない）。
- `DeviceSettings.cpp` の事前バイト列は復元できていない（内容は同一）。
- 新たなバグまたは遅延の再発が確認された場合は、対象を限定した調査・修正計画を改めて提出する。

### 10.2 参照

- 実装差分: §3.1
- 専用ストレステスト: §3.2（`reads=10081142 writes=8061467 torn=0`）
- 既存テスト: §9（61/61）
- 計測生データ: §4.2
