# XML 読込上限（8 MiB）と書込ガード（10 MB）の不整合

- 作業日時: 2026-11-13 15:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 判定: **完了**。仕様を確定し、保存側を読込上限へ統一した。

---

## 1. 定義と適用箇所

| 側 | 位置 | 値・単位 |
| --- | --- | --- |
| 読込 | `src/InputInspection.h:23` `kMaxXmlFileBytes = 8 * 1024 * 1024` | 8 MiB（ファイルサイズ＝バイト数） |
| 読込 | `src/InputInspection.cpp:270-272` `inspectXmlFile` | `file.getSize() > limits.maxBytes` で拒否 |
| 書込 | `src/DeviceSettings.cpp:908`（変更前） | `xmlText.length() < 10 * 1024 * 1024`（**文字数**） |

読込は次の 6 経路がすべて既定値（8 MiB）を使う。

| 経路 | 位置 |
| --- | --- |
| 学習状態（保存時マージ） | `DeviceSettings.cpp:855` |
| 学習状態（起動時） | `DeviceSettings.cpp:947` |
| スレッド優先度の事前読込 | `DeviceSettings.cpp:1033` |
| デバイス設定 | `DeviceSettings.cpp:1119` |
| プリセット／セッション | `MainWindow.cpp:1823` |
| 学習スナップショット | `NoiseShaperLearner.cpp:1514`、`learner/LearnedStateMigration.cpp:59` |

書込は 4 経路。ガードがあるのは学習状態だけで、他は無ガードである。

| 書込 | 位置 | ガード |
| --- | --- | --- |
| 学習状態 | `DeviceSettings.cpp:908` | 10 MB（文字数） |
| デバイス設定 | `DeviceSettings.cpp:1085` | 無し |
| プリセット／セッション | `MainWindow.cpp:1810` | 無し |
| 学習スナップショット | `NoiseShaperLearner.cpp:1509` | 無し |

## 2. 上限値の由来

- 8 MiB は B-05 の admission 上限である。コメント（`InputInspection.h:20-22`）に
  「プリセット・セッション・学習状態の実測最大（数百 KiB 規模）に対して十分な余裕」と記録されている。
  パース前に確保と再帰を止めるための**セキュリティ境界**である。
- 10 MB は学習状態の保存側に置かれていた古いガードで、由来の記録は無い。単位も文字数であり、
  ファイルサイズと一致しない。

## 3. 不整合の実害

8 MiB を超え 10 MB 未満の学習状態ファイルは「保存は成功するが、次回起動時に読込が拒否される」。

- `parseXmlFile` は `nullptr` を返す（`InputInspection.cpp:301-302`）。
- `saveNoiseShaperState` は `nullptr` のとき新規ツリーを作る（`DeviceSettings.cpp:857-860`）。
- そのまま 180 バンクを書き戻すため、**旧内容は警告なしに上書きされる**。

実データは 278 KB（`noise_shaper_learn.xml`）なので現実には到達しないが、
「書けたものが読めない」状態を作り得る点が欠陥である。加えて文字数比較は、
非 ASCII を含む文書でバイト数より小さく出るため、上限の意味が読込側とずれる。

## 4. 確定した仕様

**自動保存する状態ファイルは、読込側の上限（`kMaxXmlFileBytes`）を超えるサイズを書かない。
判定は UTF-8 のバイト数（＝ファイルサイズ）で行う。**

- 値は **8 MiB を維持**する。実データが 278 KB であること、8 MiB はパース前の admission 境界であり
  引き上げる利益が無いこと、10 MB を正当化する記録が無いことによる。10 MB への統一は採らない。
- 単一の真実源にするため、判定を `InputInspection.h` の関数へ置き、保存側はその関数だけを呼ぶ。
  これにより将来どちらかだけを変更して再びずれることを防ぐ。

## 5. 変更内容

| ファイル | 変更 |
| --- | --- |
| `src/InputInspection.h` | `isXmlTextWithinLimit(text, limits)` を追加（UTF-8 バイト数で `maxBytes` と比較） |
| `src/DeviceSettings.cpp` | `saveNoiseShaperState` のガードを `xmlText.length() < 10 * 1024 * 1024` から `convo::safeinput::isXmlTextWithinLimit (xmlText)` へ |
| `src/tests/InputInspectionTests.cpp` | X-7（境界: 上限ちょうど＝可、上限+1＝不可）を追加 |

```diff
--- a/src/DeviceSettings.cpp
+++ b/src/DeviceSettings.cpp
@@ -906,5 +906,8 @@
     // ★ BUG-3（案 A）: 直列化は 1 回だけ。結果をそのまま書き、二重直列化を解消する。
     const juce::String xmlText = root->toString();
-    if (xmlText.length() < 10 * 1024 * 1024)
+
+    // ★ XML 上限の対称性: 読込側（kMaxXmlFileBytes）で拒否されるサイズは書かない。
+    //   判定は UTF-8 のバイト数（＝ファイルサイズ）で行う。
+    if (convo::safeinput::isXmlTextWithinLimit (xmlText))
     {
```

サイズとハッシュ:

| ファイル | 変更前 | 変更後 |
| --- | --- | --- |
| `src/DeviceSettings.cpp` | 59,371 B / `92c36da9e2dc1973` | 59,580 B / `f6723d5b84f42ced` |
| `src/InputInspection.h` | 2,155 B / — | 2,812 B / `727b7707c9a1af88` |
| `src/tests/InputInspectionTests.cpp` | 7,432 B / — | 7,796 B / `f4bb2380db082306` |

警告文（`Noise shaper state file too large, skipping save.`）は変更していない。

追跡状態: `src/InputInspection.h` と `src/tests/InputInspectionTests.cpp` は **未追跡**（B-05 の既存未コミット群）、
`src/DeviceSettings.cpp` は追跡済みである。そのため `git diff --stat` に出るのは `DeviceSettings.cpp` だけであり、
`git status` の未追跡一覧に上記 2 ファイルが含まれる。Git 操作は本指示の対象外なので commit はしていない。

## 5.1 影響範囲

`InputInspection.h` を include するのは 8 ファイル（`DeviceSettings.cpp`、`IRConverter.cpp`、
`MainWindow.cpp`、`NoiseShaperLearner.cpp`、`ConvolverProcessor.LoaderThread.cpp`、
`ConvolverProcessor.ResampleAndFallback.cpp`、`LearnedStateMigration.cpp`、`InputInspectionTests.cpp`）。
今回の追加は**新しい inline 関数の追加のみ**で、既存の宣言・既定値・挙動は変えていない。
挙動が変わるのは `DeviceSettings.cpp` の学習状態保存 1 箇所だけである。

## 6. 検証

| # | 検証 | 結果 |
| --- | --- | --- |
| 1 | `InputInspectionTests` の単体実行 | **12 項目 PASS / 失敗 0**（X-1〜X-7 の 8 件、R-1〜R-4 の 4 件）。X-7 で 8,388,608 B は可、8,388,609 B は不可 |
| 2 | `ConvoPeq` Release ビルド | `BUILD_OK`（新規警告なし） |
| 3 | 実アプリの保存 | 入力 `0DCDF3D9D024267D` → 出力 `0DCDF3D9D024267D`、警告なし（278 KB は上限内で従来どおり保存） |
| 4 | `ctest -C Release`（全 61 件） | 下記 §7 |

## 7. 既存テスト

```
ctest --test-dir build -C Release -j 4
100% tests passed out of 61
Total Test time (real) = 307.87 sec
```

61/61 成功。

途中経過の記録: 1 回目の全件実行では `TailLayerIntegrityTests` が 1 件だけ失敗した。同テストを単独で実行すると
2.50 秒で PASS し、2 回目の全件実行でも PASS（2.75 秒）した。このテストは
`RES-F`（`cpuMsPerAudioSec <= kCpuLimitMsPerSec`）という**実時間の閾値**で判定するため、
`-j 4` の並列実行で CPU を奪われると落ちる。今回の変更（`InputInspection.h` の関数追加と
`DeviceSettings.cpp` の保存ガード）はこのテストの経路に触れていない。

## 8. 他ライタを変更しない理由（判断の記録）

- `device_settings.xml`（`DeviceSettings.cpp:1085`）: 内容は音声デバイス設定とコンボリューション状態の
  メタデータで、実測 11,727 B。180 バンクの係数を含めても構造上の上限が 8 MiB に届かない。
  ここへ新しいガードを足すと「8 MiB を超えた日には音声設定ごと保存されない」という**新しい失敗モード**を
  作るため、変更しない。
- プリセット保存（`MainWindow.cpp:1810`）: ユーザーが選んだ保存先への書き出しであり、
  拒否すると保存操作そのものが失敗する。UI 上の扱い（拒否か警告か）は仕様判断が要るため、変更しない。
- 学習スナップショット（`NoiseShaperLearner.cpp:1509`）: 係数 2 配列のみで数百バイト。同上。

## 9. 未実施の検証と理由

| # | 内容 | 理由 |
| --- | --- | --- |
| 1 | 8 MiB 超の学習状態を実際に保存させ、次回起動で拒否されることの実測 | 180 バンクの実データでは 278 KB にしかならず、8 MiB を作るには状態を人工的に膨らませる必要がある。境界は X-7 で固定した |
| 2 | プリセット保存の上限超過時の UI 挙動 | 仕様判断（§8）が未確定のため |

## 10. 残留リスク

- 学習状態が将来 8 MiB を超える設計変更を行った場合、保存は「スキップ」される（警告ログのみ）。
  これは変更前の 10 MB 超過時と同じ挙動であり、新しい失敗モードではない。
- 他 3 経路は無ガードのままである（§8 の理由により意図的に据え置き）。

## 11. 判定

**完了。** 読込上限と書込ガードが単一の真実源（`kMaxXmlFileBytes`、UTF-8 バイト数）で一致した。
