# BUG-3 調査（read-only）

- 作業日時: 2026-11-13 13:00 JST
- 種別: **read-only 調査**。ソース・テスト・ビルド・実行・Git の変更は行っていない。
- 目的: BUG-3（保存時の二重直列化）の実態、影響範囲、最小修正案、検証方法を、現行ソースに基づいて提示する。
- 境界: BUG-2・W-1・shutdown 関連・U-3 には触れない。本文書は実装承認ではない。

---

## 1. 調査対象と基準情報

| 項目 | 値 |
| --- | --- |
| HEAD | `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`） |
| working tree | 合計 146（変更 38 ／ 未追跡 108）。W-1 実装分を含む |
| `ConvoPeq.md` | 7,011,745 B / SHA-256 `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` / Generated `2026-11-12 00:48:16` |
| スナップショットと working tree | **差あり**（11-13 の変更を未反映）。本文書は working tree を直接読んだ |
| 対象ソース | `src/DeviceSettings.cpp`（58,260 B / `9a76a040caaee6a3`、W-1 反映後） |
| 参照した JUCE | `JUCE/modules/juce_core/xml/juce_XmlElement.cpp`、`.../files/juce_File.cpp`、`.../streams/juce_OutputStream.cpp` |

---

## 2. BUG-3 の現行実装と処理フロー

### 2.1 `saveNoiseShaperState`（`DeviceSettings.cpp:848-915`）

```
848  void DeviceSettings::saveNoiseShaperState(const AudioEngine& engine)
849  {
850      auto file = getNoiseShaperStateFile();
...
855      if (file.existsAsFile())
856          root = convo::safeinput::parseXmlFile(file);     // ① 既存 XML をパース
...
864-904  180 バンクのマージ（getChildByName → deleteAllChildElements → addChildElement）
905
906      if (root->toString().length() < 10 * 1024 * 1024)    // ② 全 XML を直列化（結果は長さだけ使用）
907      {
908          // ★ W-1: 保存失敗を無言にしない（成功時の挙動は不変）。
909          if (! root->writeTo(file))                        // ③ もう一度直列化して書込み
910              juce::Logger::writeToLog("Warning: failed to write noise shaper state file: "
911                                       + file.getFullPathName());
912      }
913      else
914          juce::Logger::writeToLog("Noise shaper state file too large, skipping save.");
915  }
```

### 2.2 `saveSettings`（`DeviceSettings.cpp:1016-1065`）

`saveNoiseShaperState(engine)` を先頭で呼び（`:1012`）、その後 `deviceManager.createStateXml()` に属性を足し、
最後に `xml->writeTo(getSettingsFile())`（`:1061`、W-1 で戻り値検査を追加）で書く。
**この経路に `toString()` は無い**（二重直列化は `saveNoiseShaperState` のみ）。

### 2.3 対象データ

`noise_shaper_learn.xml`（180 バンク）。実測 278,332 B（`%APPDATA%\ConvoPeq\noise_shaper_learn.xml`）。

---

## 3. 二重直列化の発生根拠

### 3.1 `toString()` の実装（JUCE）

```
// juce_XmlElement.cpp:361-366
String XmlElement::toString (const TextFormat& options) const
{
    MemoryOutputStream mem (2048);
    writeTo (mem, options);      // ← ツリー全体を直列化（1 回目）
    return mem.toUTF8();
}
```

### 3.2 `writeTo(const File&)` の実装（JUCE）

```
// juce_XmlElement.cpp:416-434
bool XmlElement::writeTo (const File& destinationFile, const TextFormat& options) const
{
    TemporaryFile tempFile (destinationFile);
    {
        FileOutputStream out (tempFile.getFile());
        if (! out.openedOk()) return false;
        writeTo (out, options);   // ← ツリー全体を直列化（2 回目）
        out.flush();
        if (out.getStatus().failed()) return false;
    }
    return tempFile.overwriteTargetFileWithTemporary();
}
```

### 3.3 結論（確認済み）

- `DeviceSettings.cpp:906` の `root->toString()` は **1 回目の完全直列化**を行い、その戻り値は
  **`.length()` にのみ使われ、その後破棄される**（`:906` 以外に使用箇所は無い）。
- `:909` の `root->writeTo(file)` が **2 回目の完全直列化**を行う。
- したがって、保存 1 回につき同一ツリーを 2 回直列化している。
- `toString()` が必要とされている理由は **10 MB のサイズ上限判定のみ**（`:906` の比較）。

---

## 4. `toString()` と `writeTo` の API 仕様に基づく評価

| 項目 | 事実 | 根拠 |
| --- | --- | --- |
| 出力テキストの同一性 | `toString()` も `writeTo(File)` も同じ `writeTo(OutputStream&, TextFormat)` を通る。既定 `TextFormat`（`newLineChars = "\r\n"`、`addDefaultHeader = true`、`lineWrapLength = 60`）で呼ばれるため、**生成されるバイト列は同一** | `juce_XmlElement.cpp:361-366, 416-434` / `juce_XmlElement.h:208-215` |
| 書込み方式 | `writeTo(File)` は `TemporaryFile` に書いてから `overwriteTargetFileWithTemporary()` で置換（原子的） | `:418, 433` |
| 失敗検出 | 一時ファイルが開けない / 書込み失敗 / 置換失敗で `false` | `:423-433` |
| `File::replaceWithText` | 同じ `TemporaryFile` + 置換方式。`appendText` → `FileOutputStream::writeText(..., lineFeed = nullptr)` | `juce_File.cpp:798-803`, `788-796` |
| `writeText` の改行変換 | `lineFeed == nullptr` のとき**変換しない**（`replaceLineFeedWithWindows/Unix` が両方 false） | `juce_OutputStream.cpp:211-216, 255-305` |
| 結論 | `toString()` の結果を `replaceWithText` で書いても、**現行と同一バイト列**になる（API 仕様から導ける） | 上記 |

> 注意: バイト同一性は API 仕様からの導出であり、実ファイルでの一致確認は未実施（§8）。

---

## 5. 最小修正案と代替案

### 案 A（推奨・最小）: 直列化結果をそのまま書く

```
// DeviceSettings.cpp:906-914
-    if (root->toString().length() < 10 * 1024 * 1024)
+    const juce::String xmlText = root->toString();      // 直列化は 1 回だけ
+    if (xmlText.length() < 10 * 1024 * 1024)
     {
         // ★ W-1: 保存失敗を無言にしない（成功時の挙動は不変）。
-        if (! root->writeTo(file))
+        if (! file.replaceWithText(xmlText))
             juce::Logger::writeToLog("Warning: failed to write noise shaper state file: "
                                      + file.getFullPathName());
     }
     else
         juce::Logger::writeToLog("Noise shaper state file too large, skipping save.");
```

- 直列化回数: **2 → 1**。
- 保存結果: 同一バイト列（§4）。書込み方式も temp + 置換で同じ。
- エラー処理: `replaceWithText` の `false` で従来どおり W-1 の警告が出る（メッセージは不変）。
- 変更行数: 3 行（+1 変数、比較対象の変更、書込み呼出しの変更）。

### 案 B（代替）: MemoryOutputStream で受けてバイト列を書く

```
    juce::MemoryOutputStream mos (2048);
    root->writeTo (mos);
    if ((juce::int64) mos.getDataSize() < 10 * 1024 * 1024)
    {
        if (! file.replaceWithData (mos.getData(), mos.getDataSize()))
            ...
    }
```

- 直列化回数: 1。`toString()` の `toUTF8()` コピー（278 KB）を避けられる。
- 変更行数は案 A より多い（4〜5 行）。出力バイト列は同じ。

### 案 C: ガードを外して `writeTo(file)` のみ

- 直列化回数: 1。ただし **10 MB 上限の保護が消える**（挙動変更）。推奨しない。

### 案 D: 現状維持（不実施）

- 直列化回数: 2。変更なし。Owner の現判断（不実施）に対応。

### 比較

| 案 | 直列化 | 変更行 | 出力バイト | 10 MB ガード | 失敗時ログ（W-1） | リスク |
| --- | ---: | ---: | --- | --- | --- | --- |
| A | 1 | 3 | 同一 | 維持 | 維持（呼出し先が変わる） | 小 |
| B | 1 | 4〜5 | 同一 | 維持 | 維持 | 小 |
| C | 1 | 1 | 同一 | **失う** | 維持 | 中（挙動変更） |
| D | 2 | 0 | 同一 | 維持 | 維持 | なし |

**実測されている削減量は 1 回の直列化分 = 1.2〜1.6 ms**（受理済み BUG-2 Step 1 の `save_tostring`）。
それ以上の効果は主張しない。

---

## 6. リスク・影響範囲

| 項目 | 内容 |
| --- | --- |
| 出力バイト列 | 案 A/B とも同一（§4）。ただし実ファイルでの一致確認を検証項目に入れる |
| 失敗時の挙動 | `replaceWithText` / `replaceWithData` は `false` を返し、W-1 の警告が出る。**W-1 の呼出し先が `writeTo` から変わる**ため、W-1 の実装に依存する（報告のみ・変更しない） |
| 10 MB ガード | 案 A/B とも維持。案 C のみ失う |
| メモリ | 案 A: `toString()` が既に確保している 278 KB の String を保持し続ける（現状も同じ一時確保）。案 B: MemoryOutputStream のバッファのみ |
| BUG-2 への影響 | なし（保存時間が 1.2〜1.6 ms 短くなるだけ） |
| 回帰リスク | 小。保存形式・順序・属性は不変 |
| 競合 | `DeviceSettings.cpp` は未コミット変更あり。案 A は `:906-914` の 3 行のみ |

---

## 7. 修正後に必要となる検証項目

| # | 検証 | 方法 |
| --- | --- | --- |
| 1 | **バイト同一性** | 同一の状態で修正前後の保存ファイルを取得し、SHA-256 を比較（180 バンク・278 KB） |
| 2 | 保存→再起動→読込 | 再起動後に 180 バンクの値が一致（XML 正規化比較） |
| 3 | 10 MB ガード | 巨大ツリーを作れないため、コードレビューで確認（実測は不可） |
| 4 | 失敗時ログ（W-1） | `noise_shaper_learn.xml` を読み取り専用にして終了し、警告が出ること |
| 5 | 既存テスト | `ctest -C Release` 全件（61 件） |
| 6 | 時間 | 保存合計が現行（30.8〜32.7 ms）から 1.2〜1.6 ms 程度短くなること（計測には一時コードが必要で、別承認） |

---

## 8. 未確認事項と追加情報

| # | 内容 | 状態 |
| --- | --- | --- |
| U-1 | 実ファイルでのバイト同一性 | **未確認**（API 仕様からは同一と導ける） |
| U-2 | 修正後の実測短縮量 | 未計測（計測用コードが必要） |
| U-3 | 10 MB ガードの実動作 | 巨大ツリーを作れないため未確認（コード上の分岐のみ） |
| U-4 | `replaceWithText` の書込み途中失敗（ディスク不足等）の検出 | `appendText` が `writeText` の戻り値を返すため検出可能（コード確認済み） |
| U-5 | BUG-2 の 10 秒との関係 | 無関係（保存時間の 1.2〜1.6 ms の話） |

---

## 9. Owner に求める判断事項

| # | 判断 | 選択肢 |
| --- | --- | --- |
| D-1 | BUG-3 を実施するか | 案 A / 案 B / 案 C / 案 D（不実施の維持） |
| D-2 | 実施する場合、W-1 の呼出し先を `writeTo` → `replaceWithText` に変えてよいか | 可 / 不可（不可なら案 B で `replaceWithData`、それも不可なら現状維持） |
| D-3 | バイト同一性の確認をどう担保するか | 実ファイル SHA-256 比較（手作業） / 専用テスト追加（テストコード変更の承認が必要） |
| D-4 | 短縮量 1.2〜1.6 ms を妥当と見るか | 実施する / しない |

**本文書は調査であり、実装承認ではない。** 実装・ビルド・テストは、本報告を確認した Owner の
明示的な承認後に限り実施する。

**追記（2026-11-13）**: 本文書の案 A は Owner 承認を得て実装済みである。実装・検証の記録は
`doc/work113/20261113-1800_BUG3実装/report.md`。実装では `replaceWithText` に `lineEndings = nullptr` を
明示した（§4 の改行変換の差を塞ぐため）。実測したバイト同一性と、`replaceWithText` が書き込み途中の
失敗を検出しない点は同報告 §3・§4.1 にある。
