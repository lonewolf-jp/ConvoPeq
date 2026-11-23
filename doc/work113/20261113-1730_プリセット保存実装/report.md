# プリセット保存 UI: 上限判定と失敗通知の実装

- 作業日時: 2026-11-13 17:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **Owner 承認済み仕様の実装**（プリセット／セッション保存経路のみ）。
- 判定: **完了**（UI 境界の 2 分岐は既存手段で検証できず、§6 に理由と代替を示す）。

---

## 1. 変更内容

対象は `src/MainWindow.cpp` の保存分岐 1 箇所だけである。

```diff
--- a/src/MainWindow.cpp
+++ b/src/MainWindow.cpp
@@ -1806,7 +1806,40 @@
         {
             auto state = safeThis->audioEngine.getCurrentState();
+
+            // ★ 保存失敗はユーザーに明示する。原因の詳細はログに残し、
+            //   通知文言は原因ごとに増やさず統一する（保存が完了しなかったことを明示）。
+            const auto notifySaveFailure = [] (const juce::String& detail)
+            {
+                juce::Logger::writeToLog ("[MainWindow] preset XML save rejected: " + detail);
+
+                juce::NativeMessageBox::showAsync (
+                    juce::MessageBoxOptions()
+                        .withIconType (juce::MessageBoxIconType::WarningIcon)
+                        .withTitle ("Preset not saved")
+                        .withMessage ("The preset was not saved. The existing file was left unchanged.")
+                        .withButton ("OK"),
+                    nullptr);
+            };
+
             if (auto xml = state.createXml())
             {
-                xml->writeTo(file);
+                const juce::String xmlText = xml->toString();
+
+                // ★ 上限判定は読込側と同じ関数・同じ単位（UTF-8 バイト数）で行う。
+                //   超過時は writeTo を呼ばない（既存ファイルに触れない）。
+                if (! convo::safeinput::isXmlTextWithinLimit (xmlText))
+                {
+                    notifySaveFailure ("size " + ... + " bytes exceeds the limit of " + ... + " bytes");
+                }
+                else if (! xml->writeTo (file))
+                {
+                    notifySaveFailure ("write failed: " + file.getFullPathName());
+                }
+            }
+            else
+            {
+                notifySaveFailure ("state could not be serialized");
             }
         }
```

| 項目 | 値 |
| --- | --- |
| 変更ファイル | `src/MainWindow.cpp` のみ |
| 差分 | **+34 / -1**（保存分岐の中だけ） |
| サイズ・ハッシュ | 80,154 B / `ebc4089363219ae4` → 81,952 B / `7e85fd8296430e76` |
| 改行 | CRLF を維持（変更後も bare LF 0） |

## 2. 承認仕様との対応

| # | 承認仕様 | 実装 |
| --- | --- | --- |
| 1 | 保存前のサイズ判定は `isXmlTextWithinLimit()` に統一 | 同関数を呼ぶ（`src/InputInspection.h`） |
| 2 | 判定単位は UTF-8 バイト数、上限 8 MiB | 関数内で `getNumBytesAsUTF8()` と `kMaxXmlFileBytes` を比較 |
| 3 | 上限超過時は `writeTo` を呼ばない | 超過分岐は `notifySaveFailure` のみ。`writeTo` は else 側だけ |
| 4 | 上限超過時・保存失敗時に既存ファイルを変更しない | 超過時は未書込み。失敗時は `writeTo` が置換しない（§5） |
| 5 | 失敗内容はログ、ユーザーへは `NativeMessageBox` | `[MainWindow] preset XML save rejected: <detail>` ＋ 警告アイコン |
| 6 | タイトルは `Preset not saved`、文言は原因ごとに増やさない | 3 分岐とも同一タイトル・同一本文（`The preset was not saved. The existing file was left unchanged.`） |

失敗の 3 分岐（サイズ超過・XML 生成失敗・書込み失敗）はすべて同じ通知を通り、原因はログにだけ残る。

## 3. 検証（実測）

### 3.1 正常保存（GUI 自動化）

UIAutomation でアプリの `Save` ボタンを押し、ネイティブ保存ダイアログのファイル名欄へパスを入れて確定させた。

```
submitted: C:\Users\user\AppData\Local\Temp\opencode\preset_ok.xml
FILE: 6364 B sha=295125947E5B3002
head: <?xml version="1.0" encoding="UTF-8"?>
parsed root tag: Preset | children: 2 | attribs: 109
```

プリセットが保存され、妥当な XML（`<Preset>`、属性 109）であることを確認した。

### 3.2 既存ファイルが破壊されないこと

保存先に既存ファイル（7 バイト `KEEP-ME`、読み取り専用）を置いて同じ操作を行った。

```
before: 7 B 579BF28C1920F367
after : 7 B 579BF28C1920F367
content: KEEP-ME
```

ファイルは不変。ただしこの実行では**保存経路に到達していない**（ログに `preset XML save rejected` が無く、
ネイティブダイアログが読み取り専用の上書きを拒否して空の結果を返したため）。
したがって「既存ファイルが壊れない」ことは確認できたが、**書込み失敗分岐の検証ではない**（§6）。

### 3.3 境界値（8 MiB ちょうど／+1）

`InputInspectionTests` の X-7 が同じ関数で境界を固定している（既存テスト）。

```
[PASS] X-7 text exactly at the read limit is writable (8388608)
[PASS] X-7 text one byte above the read limit is not writable (8388609)
[SUMMARY] failures=0
```

### 3.4 ビルド

```
cmake --build build --config Release --target ConvoPeq
=== BUILD_OK ===
```

新規の警告は無い（既存の `MklFftEvaluator.h` の C4390 のみ）。

### 3.5 既存テスト

```
ctest --test-dir build -C Release -j 4
100% tests passed out of 61
Total Test time (real) = 328.47 sec
```

61/61 成功。

## 4. 影響範囲

| 項目 | 内容 |
| --- | --- |
| 挙動の変化 | 8 MiB 超のプリセットは保存されない（従来は保存できたが読込で拒否された）。書込み失敗・生成失敗が UI に出る |
| 他経路 | デバイス設定・学習状態・学習スナップショットの保存は変更していない（Owner 指示） |
| 読込経路 | 変更していない |
| スレッド | すべて Message Thread（`launchAsync` のコールバック）。新しい同期は無い |
| 二重直列化 | サイズ判定のために `toString()` を 1 回行い、`writeTo` がもう 1 回直列化する（プリセットは数 KB・操作は手動のため許容） |

## 5. 既存ファイル保護の根拠

- `juce::XmlElement::writeTo(const File&)` は `TemporaryFile` に書き、`flush` 後に
  `out.getStatus().failed()` を検査し、成功時のみ置換する（`juce_XmlElement.cpp:416-434`）。
  途中で失敗した場合は置換しないため、既存ファイルは切り詰められない。
- 上限超過時は `writeTo` を呼ばないため、ファイルには一切触れない。
- 一時ファイルは失敗時に `TemporaryFile` の破棄で削除される（BUG-3 の実測で残留 0）。

## 6. 未実施の検証と理由

| # | 検証 | 状態 | 理由 |
| --- | --- | --- | --- |
| 1 | 書込み失敗時の通知（UI 境界） | **未実施** | 保存経路は GUI 専用（CLI スイッチ 44 種に保存系は無い）。読み取り専用ターゲットではネイティブダイアログが上書きを拒否して空の結果を返し、書込み不可ディレクトリではダイアログ自体が保存を拒否する。コード経路に到達させる手段が既存には無い |
| 2 | XML 生成失敗時の通知（UI 境界） | **未実施** | 同上（`createXml()` を失敗させる手段が無い） |
| 3 | 上限超過時の通知（UI 境界） | **未実施** | 8 MiB を超える `ValueTree` を作る必要があり、現実的でない。判定は X-7 で固定 |

代替の根拠:

- 3 分岐とも同じ `notifySaveFailure` を通るため、通知の実装は 1 箇所である（分岐ごとの差は無い）。
- 書込み失敗時の既存ファイル保護は、同じ `writeTo(File)` を使う学習状態の保存で
  読み取り専用ファイルに対して実測済み（`doc/work113/20261113-1900_BUG3b実装/report.md` §3.1:
  警告 2 件・原本不変・一時ファイル残留 0）。
- 上限超過時に `writeTo` を呼ばないことは、コード上で分岐が排他であることに依る。

UI 境界まで通す試験が必要な場合は、テストフックの追加が要る（別紙 `20261113-1730_UI境界フック案`）。

## 7. 残留リスク

- 上限超過時に保存を止めるため、8 MiB を超える状態を持つユーザーはプリセットを保存できなくなる
  （通知は出る）。従来は「保存できるが次回読めない」だったので、挙動としては安全側。
- 通知は非同期（`showAsync`）であり、連続して保存操作をすると複数の通知が重なる可能性がある。
  既存の読込拒否（`Preset not loaded`）と同じ実装であり、新しい問題ではない。
- 二重直列化（§4）は数 KB 規模であり、実測でも保存は即時である。

## 8. 判定

**完了。** 承認仕様 6 項目を実装し、正常保存・既存ファイル保護・境界値・ビルド・既存テストを確認した。
UI 境界の 2 分岐（書込み失敗・生成失敗）は既存手段では到達できず、未実施として明記した。
