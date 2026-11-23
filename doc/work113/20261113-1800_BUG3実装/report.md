# BUG-3 実装: 二重直列化の解消（案 A）

- 作業日時: 2026-11-13 13:40 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **Owner 承認済み範囲の実装**。`saveNoiseShaperState` の BUG-3 のみ。
- 判定: **完了**。出力バイト列の一致を実ファイルで確認し、二重直列化は 2 回から 1 回になった。

---

## 1. 何を変えたか

`DeviceSettings.cpp` の `saveNoiseShaperState`（`848-918`）で、サイズ判定のために作っていた
`toString()` の戻り値を使い捨てず、そのままファイルへ書くようにした。直列化は保存 1 回につき 2 回から 1 回になる。

```diff
@@ -904,8 +904,11 @@
-    if (root->toString().length() < 10 * 1024 * 1024)
+    // ★ BUG-3（案 A）: 直列化は 1 回だけ。結果をそのまま書き、二重直列化を解消する。
+    const juce::String xmlText = root->toString();
+    if (xmlText.length() < 10 * 1024 * 1024)
     {
         // ★ W-1: 保存失敗を無言にしない（成功時の挙動は不変）。
-        if (! root->writeTo(file))
+        // lineEndings=nullptr: writeText の改行変換を止め、writeTo と同じバイト列にする。
+        if (! file.replaceWithText(xmlText, false, false, nullptr))
             juce::Logger::writeToLog("Warning: failed to write noise shaper state file: "
                                      + file.getFullPathName());
```

W-1 実装版（`9a76a040caaee6a3`）との差は **+5 行 / -2 行**、この 1 箇所だけである。
10 MB ガードの式（`length() < 10 * 1024 * 1024`）と警告文は変えていない。

### 1.1 承認文の「3 行」に引数を 3 つ足した理由

`File::replaceWithText` の既定引数は `lineEndings = "\r\n"`（`juce_File.h:802-805`）である。
`FileOutputStream::writeText` はこの値を見て**素の LF を CRLF に変換し、単独の CR を捨てる**
（`juce_OutputStream.cpp:211-216, 259-289`）。旧経路の `writeTo(OutputStream&)` は変換しないため、
テキストノードに素の LF が含まれると既定引数のままではバイト列が変わる。

実際にプローブ文書（後述 §3.2）で確認した。既定引数版の出力は `4B14FCF5B52A57E3`（278,419 B）、
`nullptr` 指定版は `BC294B76D1E51E13`（278,417 B）で、差はちょうど素の LF 2 個が CRLF になった 2 バイトである。
したがって `nullptr` は「同じバイト列を書く」という承認条件を満たすための必須指定であり、追加の機能変更ではない。

---

## 2. 基準情報

| 項目 | 値 |
| --- | --- |
| HEAD | `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`） |
| working tree | 実装前 147（変更 38 ／ 未追跡 109） → 実装後 148（変更 38 ／ 未追跡 110）。増分は本報告フォルダ 1 件のみ |
| `ConvoPeq.md` | 7,011,745 B / `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` / Generated `2026-11-12 00:48:16` |
| スナップショットと working tree | **差あり**（W-1 と BUG-3 を未反映）。判断は working tree を直接読んで行った |
| `DeviceSettings.cpp` | 実装前 58,260 B / `9a76a040caaee6a3`（W-1 版） → 実装後 58,566 B / `4b7727b8d6445a18` |
| 保存対象 | `%APPDATA%\ConvoPeq\noise_shaper_learn.xml`（278,332 B / 180 バンク） |

---

## 3. バイト同一性の実測

### 3.1 実ファイルの A/B（通常データ）

`noise_shaper_learn.xml` を既知の状態（`0DCDF3D9D024267D`）へ戻し、`--cli-run --cli-exit-ms 1500` で
各バイナリ 3 回ずつ実行した。書き込みの有無は更新時刻の変化で判定した（入力は `Copy-Item` で復元するため
更新時刻が古いまま残る）。

| バイナリ | 入力 SHA-256 | 出力 SHA-256 | 更新時刻の変化 | 警告 |
| --- | --- | --- | --- | --- |
| W-1 版（`writeTo`、3 回） | `0DCDF3D9D024267D` | `0DCDF3D9D024267D`（3 回とも） | あり（3 回とも） | なし |
| BUG-3 版（`replaceWithText` + `nullptr`、3 回） | `0DCDF3D9D024267D` | `0DCDF3D9D024267D`（3 回とも） | あり（3 回とも） | なし |

6 秒実行（`--cli-exit-ms 6000`、実測 10,304 ms / 9,959 ms）でも BUG-3 版の出力は `0DCDF3D9D024267D` で一致した。
なお 3 回とも入力と出力のハッシュが同じなのは、学習状態が現行ファイルの内容を再現しているためであり、
「書いていない」のではない。更新時刻が毎回変化していることが書き込みの証拠である。

> 計測上の注意: 最初の A/B は `ctest` と並行して実行してしまい、`AudioEngineHarness` が同じ
> `noise_shaper_learn.xml` を書くため出力が `50B6F74369532A8E` に化けた。上表は**他プロセスを止めた状態**で
> 取り直した値である。保存経路は `saveNoiseShaperState` の 1 つだけ（`DeviceSettings.cpp:1021`）で、
> ハーネスも同じ関数を通る。

### 3.2 差が出る条件を狙ったプローブ

現行データはテキストノードを持たないため、上表だけでは「改行変換の差」を検出できない。
そこで基準ファイルに検証用の要素を 1 つ足した入力を作った（`a8e8307cc674243c`、278,411 B）。

```xml
  <ProbeBug3 attr="a&amp;b&lt;c&gt;d&quot;e あ">line1
line2
line3</ProbeBug3>
```

属性値のエスケープ（`&` `<` `>` `"`）、非 ASCII（`あ`）、テキストノード内の素の LF を含む。

| 書き込み経路 | 出力 SHA-256 | サイズ |
| --- | --- | --- |
| W-1 版（`writeTo`） | `BC294B76D1E51E13` | 278,417 B |
| BUG-3 版（`replaceWithText` + `nullptr`） | `BC294B76D1E51E13` | 278,417 B |
| 参考: 既定 `lineEndings` 版 | `4B14FCF5B52A57E3` | 278,419 B |

採用版は旧経路と完全一致した。既定 `lineEndings` 版だけが 2 バイト多く、これが §1.1 の根拠である。
なお属性の `あ` はどちらの経路でも `&#12354;` に、入力末尾の改行は除去され、テキスト内の CRLF は
XML パーサの正規化で LF になっている。これらは書き込み経路ではなくパーサと `toString()` の挙動で、両者で共通である。

### 3.3 保存 → 再読込

`replaceWithText` が書いたファイルを次回起動で読み直し、同じ内容を書き戻せるかを確認した。

| 入力 | 出力 | 判定 |
| --- | --- | --- |
| `BC294B76D1E51E13`（プローブ出力） | `BC294B76D1E51E13` | 一致（冪等） |

プローブ要素・エスケープ・LF を含む文書が往復しても壊れないことを確認した。

---

## 4. W-1 の失敗検出は維持されている

`noise_shaper_learn.xml` と `device_settings.xml` に読み取り専用属性を付けて終了させた。

```
readonly run: exit=0 elapsed=7125ms
HIT: Warning: failed to write noise shaper state file: C:\Users\user\AppData\Roaming\ConvoPeq\noise_shaper_learn.xml
HIT: Warning: failed to write device settings: C:\Users\user\AppData\Roaming\ConvoPeq\device_settings.xml
```

実行前後でファイルの SHA-256 は変わらず（`0DCDF3D9D024267D` / `8DD0C67E03F7664B`）、警告文も W-1 のままである。
通常実行では警告が出ないことも §3.1 の 8 回で確認した。

### 4.1 ただし検出範囲は狭くなった（残存差分）

`File::replaceWithText` は一時ファイルへの書き込み結果を捨て、置換の成否だけを返す（`juce_File.cpp:798-803`）。

```cpp
TemporaryFile tempFile (*this, TemporaryFile::useHiddenFile);
tempFile.getFile().appendText (textToWrite, asUnicode, writeHeaderBytes, lineFeed);  // 戻り値は捨てられる
return tempFile.overwriteTargetFileWithTemporary();
```

旧 `writeTo(File)` は書き込み後に `out.getStatus().failed()` を検査していた（`juce_XmlElement.cpp:421-431`）。
したがって次の差が残る。

| 失敗の種類 | 旧 `writeTo` | 新 `replaceWithText` |
| --- | --- | --- |
| 一時ファイルが開けない（権限・パス） | 検出 | 検出（一時ファイルが存在せず置換が失敗する） |
| 置換失敗（読み取り専用など） | 検出 | 検出（実測済み） |
| 書き込み途中の失敗（ディスク不足など） | 検出（`getStatus().failed()`） | **未検出**（部分内容のまま置換され得る） |

実運用で起きる確率は低いが、W-1 の目的（保存失敗を無言にしない）に対する後退ではある。判断は §8 に置く。

Debug ビルドではもう 1 点差が出る。一時ファイルが作れなかった場合、`overwriteTargetFileWithTemporary` は
`jassertfalse` を踏んでから `false` を返す（`juce_TemporaryFile.cpp:111-118`）。旧経路は同じ状況で
`return false` するだけだったため、Debug で停止させていると新しい経路だけブレークする。Release では
`jassert` は無効なので挙動は変わらない。

---

## 5. ビルドと既存テスト

```
cmake --build build --config Release --target ConvoPeq
=== BUILD_OK ===
```

警告は既存の `MklFftEvaluator.h:326/508` の C4390 のみで、新しい警告は出ていない。

```
ctest --test-dir build -C Release -j 4
100% tests passed out of 61
Total Test time (real) = 328.89 sec
```

61/61 成功（失敗 0）。最終版（`4b7727b8d6445a18`）のバイナリで実行した。
```

---

## 6. 変更しなかった範囲

| 項目 | 状態 |
| --- | --- |
| BUG-2（保存時間・10 秒遅延） | 未着手・未変更 |
| shutdown 本体・Worker 停止 API・所有権／破棄順序・`workersStuck_`・`B_pre` | 未着手・未変更 |
| U-3（W3・W4・W7 の停止要求 API） | 未着手・未変更 |
| 他の保存処理（`saveSettings` の `xml->writeTo`、`loadNoiseShaperState`） | 未変更 |
| 10 MB ガード | 維持（式も型も同じ） |
| W-1 の警告文・失敗時のログ出力 | 維持（呼出し先のみ `replaceWithText` へ変更） |
| テストコード・計測用コード | 追加していない |
| 既存の未コミット変更 | 保持。`DeviceSettings.cpp` の HEAD との差は 20 insertions / 7 deletions で、内訳は既存の B-05（`juce::XmlDocument::parse` → `convo::safeinput::parseXmlFile` の 4 箇所と include）、W-1（+8 / -2）、BUG-3（+5 / -2） |
| Git 操作 | commit / checkout / reset / stash は行っていない |
| 実行で更新された設定ファイル | 作業後にバックアップから復元（`noise_shaper_learn.xml` = `0DCDF3D9D024267D` / 278,332 B、`device_settings.xml` = `8DD0C67E03F7664B` / 11,727 B） |

比較実験のため、既定 `lineEndings` 版（`replaceWithText(xmlText)`）を一時的にビルドして測定し、
直後に採用版（`nullptr` 付き）へ戻して再ビルドした。最終的なソースは `4b7727b8d6445a18` である。

---

## 7. 再現手順

```powershell
$tmp = 'C:\Users\user\AppData\Local\Temp\opencode'
$exe = 'C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe'
# 1) 入力の固定
Copy-Item "$tmp\noise_shaper_learn.backup.xml" "$env:APPDATA\ConvoPeq\noise_shaper_learn.xml" -Force
# 2) 実行（他プロセスを止めてから。ctest と並行させるとハーネスが同じファイルを書く）
& $exe --cli-run --cli-exit-ms 1500
# 3) 出力のハッシュ
(Get-FileHash "$env:APPDATA\ConvoPeq\noise_shaper_learn.xml" -Algorithm SHA256).Hash
```

失敗系は 2) の前に `attrib +R` を 2 ファイルへ適用し、終了後に `attrib -R` で戻す。

---

## 8. 残った判断事項

1. **書き込み途中失敗の未検出（§4.1）** — Owner 承認（2026-11-13）により追加修正を実施済み。
   `TemporaryFile` と `FileOutputStream` で書き込み状態を検査してから置換する実装に置き換えた。
   記録は `doc/work113/20261113-1900_BUG3b実装/report.md`。本文書 §4.1 の表は当時の状態であり、
   現在のコードは「書き込み途中の失敗も検出」に変わっている。
2. **10 MB ガードと読込上限の不整合（既存）。** 書き込み側は 10 MiB まで許すが、読み込み側は
   `kMaxXmlFileBytes = 8 * 1024 * 1024`（`InputInspection.h:23`）で拒否する。8〜10 MiB のファイルは
   書けた次回に読めず、`parseXmlFile` が `nullptr` を返して新規ツリーで上書きされる。BUG-3 が作った問題ではない。
3. **ガードの単位（既存）。** `length()` は文字数で、バイト数ではない。式は変更していないため挙動は同じ。

---

## 9. 未確認事項

| # | 内容 | 状態 |
| --- | --- | --- |
| U-1 | ディスク不足時の部分書き込み → 置換成功の実挙動 | 未実測（コード上の経路から推定） |
| U-2 | 10 MB 超過分岐の実動作 | 未実測（巨大ツリーを作れない。コードレビューのみ） |
| U-3 | 学習状態が毎回同一になる理由 | 未調査（BUG-3 の範囲外。A/B の解釈には影響しない） |
