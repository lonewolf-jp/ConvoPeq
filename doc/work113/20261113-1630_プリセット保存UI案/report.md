# プリセット保存の上限超過時の UI 実装案（提示のみ）

- 作業日時: 2026-11-13 16:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **実装案の提示のみ**。ソース変更・ビルド・テストは行っていない（Owner 指示により実装前に提示）。
- 対象: プリセット／セッションの**保存経路のみ**。

---

## 1. Owner 確定仕様（再掲）

1. 判定は文字数ではなく UTF-8 バイト数で行う。
2. 8 MiB 以下は保存可能、8 MiB を超える場合は保存しない。
3. 拒否時は「保存できなかったこと」と理由を UI に明示する。
4. 警告だけ出して保存を続行する動作にはしない。
5. 既存の保存先ファイルがある場合、内容を破壊・切り詰め・空ファイル化しない。
6. 保存失敗をユーザーが認識できるようにし、成功したかのような表示を行わない。
7. 8 MiB ちょうどと 8 MiB 超過の境界をテストする。

## 2. 現行コード

`src/MainWindow.cpp:1805-1812`（`fileChooser->launchAsync` の保存分岐）。

```cpp
if (isSaving)
{
    auto state = safeThis->audioEngine.getCurrentState();
    if (auto xml = state.createXml())
    {
        xml->writeTo(file);          // ← 戻り値未使用・サイズ検査なし
    }
}
```

問題は 3 点。

| # | 問題 |
| --- | --- |
| 1 | サイズ無制限。8 MiB を超える XML を書くと、次回読込で `parseXmlFile` が拒否する（読込上限 8 MiB） |
| 2 | `writeTo` の戻り値を無視しており、書込み失敗がユーザーに伝わらない |
| 3 | `createXml()` が nullptr の場合も無言（保存されないのに何も出ない） |

## 3. 実装案

### 3.1 判定

既存の `convo::safeinput::isXmlTextWithinLimit()`（`src/InputInspection.h`、BUG-3 の付随作業で追加済み）を使う。
判定は **UTF-8 バイト数**で `kMaxXmlFileBytes`（8 MiB）と比較する。保存経路専用の新しい定数は作らない。

### 3.2 差分イメージ

```cpp
if (isSaving)
{
    auto state = safeThis->audioEngine.getCurrentState();

    auto showSaveError = [safeThis] (const juce::String& why)
    {
        juce::Logger::writeToLog ("[MainWindow] preset XML save rejected: " + why);
        juce::NativeMessageBox::showAsync (
            juce::MessageBoxOptions()
                .withIconType (juce::MessageBoxIconType::WarningIcon)
                .withTitle ("Preset not saved")
                .withMessage (why)
                .withButton ("OK"),
            nullptr);
    };

    if (auto xml = state.createXml())
    {
        const juce::String xmlText = xml->toString();
        const juce::int64 bytes = static_cast<juce::int64> (xmlText.getNumBytesAsUTF8());

        if (! convo::safeinput::isXmlTextWithinLimit (xmlText))
        {
            showSaveError ("The preset is too large to save (" + juce::String (bytes)
                           + " bytes; the limit is " + juce::String (convo::safeinput::kMaxXmlFileBytes)
                           + " bytes). The existing file was left unchanged.");
        }
        else if (! xml->writeTo (file))
        {
            showSaveError ("The preset could not be written to " + file.getFullPathName()
                           + ". The existing file was left unchanged.");
        }
    }
    else
    {
        showSaveError ("The preset could not be created from the current state.");
    }
}
```

### 3.3 UI 文言（既存 UI に合わせる）

| 項目 | 文言 | 根拠 |
| --- | --- | --- |
| タイトル | `Preset not saved` | 既存の読込拒否は `Preset not loaded`（`MainWindow.cpp:1840`） |
| アイコン | `WarningIcon` | 読込拒否と同じ（`:1839`） |
| ボタン | `OK` | 読込拒否と同じ（`:1842`） |
| 本文（サイズ超過） | `The preset is too large to save (N bytes; the limit is 8388608 bytes). The existing file was left unchanged.` | 数値と「既存ファイルは変更していない」を明示 |
| 本文（書込み失敗） | `The preset could not be written to <path>. The existing file was left unchanged.` | 同上 |
| 本文（状態生成失敗） | `The preset could not be created from the current state.` | — |
| ログ | `[MainWindow] preset XML save rejected: <why>` | 読込側 `[MainWindow] preset XML rejected:` と対称 |

表示は `juce::NativeMessageBox::showAsync(..., nullptr)` を使う（既存と同じ・モーダル待ちをしない）。

## 4. 既存ファイルの保護（確認結果）

- `juce::XmlElement::writeTo(const File&)` は `TemporaryFile` に書いてから置換する
  （`JUCE/modules/juce_core/xml/juce_XmlElement.cpp:416-434`）。
  書込み中に失敗した場合は `out.getStatus().failed()` を検査して `false` を返し、置換しない。
  したがって**書込み失敗で既存ファイルが切り詰められることはない**。
- サイズ超過時は `writeTo` を**呼ばない**ため、既存ファイルには一切触れない。
- 一時ファイルは失敗時に `TemporaryFile` の破棄で削除される（BUG-3 で実測: 残留 0）。

## 5. 検証計画

| # | 検証 | 方法 |
| --- | --- | --- |
| 1 | 境界（8 MiB ちょうど＝可、+1＝不可） | `InputInspectionTests` の X-7 が同じ関数で固定済み。保存経路はこの関数を呼ぶだけなので、判定は同一 |
| 2 | サイズ超過時に書かない | コードレビュー＋（可能なら）テスト用フックで巨大 XML を注入して `writeTo` が呼ばれないことを確認 |
| 3 | 既存ファイル不変 | 同上。書込み前に拒否するため、ファイルの mtime とハッシュが変わらないこと |
| 4 | 書込み失敗の通知 | 読み取り専用の保存先を指定して実行し、メッセージが出ること（`attrib +R` で再現可能） |
| 5 | 正常保存 | 既存の保存操作で従来どおり書けること |
| 6 | 既存テスト | `ctest -C Release` 61 件 |

補足: 1 の境界は純関数のテストで固定できるが、**UI まで通した境界試験は 8 MiB の ValueTree を作る必要があり現実的でない**。
UI 経路の確認は 2〜4 の方法（フックまたは読み取り専用ファイル）で行う。
テスト用フック（巨大 XML を注入する環境変数）を追加する場合は、別途承認を得る。

## 6. 影響範囲

| 項目 | 内容 |
| --- | --- |
| 変更ファイル | `src/MainWindow.cpp`（保存分岐 1 箇所）のみ |
| 挙動の変化 | 8 MiB 超のプリセットは保存されなくなる（従来は保存できたが読めなかった）。書込み失敗時にメッセージが出る |
| 他経路 | デバイス設定・学習状態・学習スナップショットの保存には**追加しない**（Owner 指示） |
| スレッド | すべて Message Thread（`launchAsync` のコールバック）。新たな同期は増えない |

## 7. 実装前に確認したい点

1. 文言は英語でよいか（既存 UI が英語のため英語案にした）。
2. 書込み失敗時も同じメッセージ体系で通知してよいか（仕様 6 の解釈）。
3. `createXml()` 失敗時の通知を追加してよいか（従来は無言）。
4. 境界を UI まで通すテストが必要か。必要ならテスト用フックの追加承認が要る。

## 8. 今回行っていないこと

- ソース変更、ビルド、テスト。
- 他経路（デバイス設定・学習状態）への上限追加。
- Git 操作。
