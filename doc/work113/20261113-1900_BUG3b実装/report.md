# BUG-3 追加修正: 書き込み途中失敗の検出回復

- 作業日時: 2026-11-13 14:10 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **Owner 承認済みの追加修正**。対象は `DeviceSettings.cpp` の `saveNoiseShaperState` の保存処理だけ。
- 判定: **完了**。直列化は 1 回のまま、出力バイト列は前版と一致、書き込み途中の失敗検出を回復した。

## 0. 報告サマリ

| 報告項目 | 結果 | 詳細 |
| --- | --- | --- |
| 差分 | `saveNoiseShaperState` の保存ブロックのみ **+23 / -2**。59,371 B / `92c36da9e2dc1973` | §1 |
| バイト同一性 | 正常系 `0DCDF3D9D024267D`、プローブ `BC294B76D1E51E13`（旧 `writeTo` 版・案 A 版と一致） | §2.1・§2.2 |
| 失敗系検証 | 読み取り専用で警告 2 件・原本不変、設定ディレクトリ不可で警告、書き込み失敗は注入 3 種とも原本不変・残留 0 | §3・§4 |
| ビルド・テスト | Release／Debug とも成功、`ctest` 61/61（328.98 秒） | §5 |

---

## 1. 何を変えたか

`File::replaceWithText` をやめ、`juce::TemporaryFile` と `juce::FileOutputStream` で自前に書く。
JUCE のヘッダが示す本来の使い方（`juce_TemporaryFile.h:52-70`）に沿った形である。

```cpp
    const juce::String xmlText = root->toString();          // 直列化は 1 回だけ
    if (xmlText.length() < 10 * 1024 * 1024)                // 10 MB ガードは維持
    {
        juce::TemporaryFile tempFile (file);
        bool saved = false;

        {
            juce::FileOutputStream out (tempFile.getFile());

            if (out.openedOk())
            {
                saved = out.writeText (xmlText, false, false, nullptr);
                out.flush();   // バッファを書き出し、失敗は status に載る
                if (out.getStatus().failed())
                    saved = false;
            }
        }

        if (saved)
            saved = tempFile.overwriteTargetFileWithTemporary();

        // ★ W-1: 保存失敗を無言にしない（成功時の挙動は不変）。
        if (! saved)
            juce::Logger::writeToLog("Warning: failed to write noise shaper state file: "
                                     + file.getFullPathName());
    }
```

前版（`replaceWithText` 版、`4b7727b8d6445a18`）との差は **+23 / -2**、このブロックの中だけである。
ファイル全体は 58,566 B → 59,371 B、SHA-256 は `4b7727b8d6445a18` → `92c36da9e2dc1973`。

### 1.1 検出経路の根拠（推測ではなく JUCE 実装）

| 論点 | 実装 | 根拠 |
| --- | --- | --- |
| 書き込み失敗が status に載るか | `WriteFile` 失敗時に `status = getResultForLastError()` | `juce_Files_windows.cpp:457-466` |
| flush 失敗が status に載るか | `FlushFileBuffers` 失敗時に同じく status 更新 | `juce_Files_windows.cpp:468-473` |
| 途中失敗が戻り値に出るか | `writeText` は `write()` の失敗で `false` を返す | `juce_OutputStream.cpp:210-312` |
| 開けなかった場合 | `openHandle` 失敗で status が失敗になり `openedOk()` が false | `juce_Files_windows.cpp:429-450`、`juce_FileOutputStream.h:93-103` |
| 置換の前提 | 一時ファイルが存在しないと `jassertfalse` のうえ `false` | `juce_TemporaryFile.cpp:100-118` |
| 失敗時の後始末 | 破棄時に一時ファイルを削除（置換失敗時も残らない） | `juce_TemporaryFile.h:66-68, 180-189` |

`out.writeText` の戻り値と `getStatus().failed()` の両方を見るのは、バッファ経由の書き込みで
`write()` が成功しても `flush()` で失敗する場合があるためである。置換は `saved == true` のときだけ呼ぶので、
**不完全な内容が元ファイルを置き換える経路は存在しない**。開けなかった場合も置換を呼ばないため、
`overwriteTargetFileWithTemporary` の `jassertfalse`（Debug）には到達しない。

### 1.2 元ファイルへ書く経路が置換だけであること（現行ソースの実測）

`saveNoiseShaperState`（`848-940`）の中で `file` に触れる箇所を列挙すると次の 6 つだけで、
書き込みは 914 の `TemporaryFile` と 930 の置換しかない。

| 行 | 内容 | 種別 |
| --- | --- | --- |
| 850 | `auto file = getNoiseShaperStateFile();` | 取得 |
| 854 | `file.existsAsFile()` | 読み取り |
| 855 | `convo::safeinput::parseXmlFile(file)` | 読み取り |
| 914 | `juce::TemporaryFile tempFile (file);` | 一時ファイル（同一ディレクトリ） |
| 930 | `tempFile.overwriteTargetFileWithTemporary()` | **置換（929 の `if (saved)` の内側）** |
| 935 | `file.getFullPathName()` | ログ用の文字列 |

`file.writeTo`、`file.replaceWithText`、`file.replaceWithData`、`file.appendText`、`file.createOutputStream`
はいずれも関数内に存在しない（grep で 0 件）。したがって、部分内容が元ファイルへ入る経路は
「929 のガードを通過する」以外に無い。

---

## 2. バイト同一性

### 2.1 通常データ

`noise_shaper_learn.xml` を既知の状態（`0DCDF3D9D024267D`）へ戻し、`--cli-run --cli-exit-ms 1500` で 3 回、
再ビルド後にもう 1 回実行した。

| 実行 | 入力 | 出力 | 更新時刻の変化 | 警告 |
| --- | --- | --- | --- | --- |
| 3 回とも | `0DCDF3D9D024267D` | `0DCDF3D9D024267D` | あり | なし |
| 再ビルド後 1 回 | `0DCDF3D9D024267D` | `0DCDF3D9D024267D` | あり | なし |

BUG-3 案 A の版、W-1 版（`writeTo`）と同じ値であり、置換経路を自前実装に変えてもバイト列は変わっていない。

### 2.2 プローブ（属性エスケープ・非 ASCII・テキストノード内 LF）

`a8e8307cc674243c`（278,411 B）を入力にした結果は次のとおり。

| 書き込み経路 | 出力 SHA-256 | サイズ |
| --- | --- | --- |
| 旧 W-1 版（`XmlElement::writeTo(File)`） | `BC294B76D1E51E13` | 278,417 B |
| BUG-3 案 A 版（`replaceWithText` + `nullptr`） | `BC294B76D1E51E13` | 278,417 B |
| **今回版（`TemporaryFile` + `FileOutputStream`）** | `BC294B76D1E51E13` | 278,417 B |

3 経路が同一バイト列を出した。`writeText(..., nullptr)` の指定により改行変換は入らない。

### 2.3 置換の順序（FileSystemWatcher による実観測）

`%APPDATA%\ConvoPeq` を監視しながら 1 回保存させ、ファイルシステムのイベントを時刻付きで記録した。

```
  5,854 ms  Created  noise_shaper_learn_temp97496eb7.xml          ← 一時ファイル作成
  5,856 ms  Changed  noise_shaper_learn_temp97496eb7.xml          ← 書き込み
  5,857 ms  Changed  noise_shaper_learn_temp97496eb7.xml
  5,857 ms  Created  noise_shaper_learn.xml~RF3f6d53d.TMP         ← 置換（ReplaceFile のバックアップ）
  6,358 ms  Deleted  noise_shaper_learn.xml~RF3f6d53d.TMP
  6,360 ms  Renamed  noise_shaper_learn.xml~RF3f6d53d.TMP         ← 元ファイルを退避
  6,367 ms  Renamed  noise_shaper_learn.xml  (旧 temp97496eb7)    ← 一時ファイルを本体へ昇格
  6,367 ms  Changed  noise_shaper_learn.xml                       ← 本体の内容がここで初めて変わる
  6,372 ms  Deleted  noise_shaper_learn.xml~RF3f6d53d.TMP
```

一時ファイルの書き込み（5,854〜5,857 ms）が終わったあとの 6,367 ms に、初めて本体の内容が変わる。
置換は `ReplaceFile` による名前の差し替えで、途中の状態が本体として見える瞬間は無い。
実行前後のハッシュは `0DCDF3D9D024267D` で一致し、一時ファイルの残留も無かった。

> この観測は**成功経路**の順序を示すもので、書き込みが途中で失敗する場合の分岐を実証するものではない。
> 失敗時の扱いは §3.3 と §1.2 のとおり。

---

## 3. 失敗系

### 3.1 置換失敗（読み取り専用の元ファイル）

`noise_shaper_learn.xml` と `device_settings.xml` に `attrib +R` を付けて終了させた。

```
readonly run: exit=0 elapsed=6790ms
HIT: Warning: failed to write noise shaper state file: C:\Users\user\AppData\Roaming\ConvoPeq\noise_shaper_learn.xml
HIT: Warning: failed to write device settings: C:\Users\user\AppData\Roaming\ConvoPeq\device_settings.xml
after ns=0DCDF3D9D024267D(278332 B) ds=8DD0C67E03F7664B
tmp leftovers: 0
```

元ファイルは不変、警告は W-1 の文面どおり、一時ファイルの残留も 0 件だった。

### 3.2 一時ファイル作成失敗（設定ディレクトリを作れない状態）

`%APPDATA%\ConvoPeq` を一時退避し、同名の**ファイル**を置いてディレクトリ作成を失敗させた。
実行後に退避を戻している（復帰後の項目数 6）。

```
tempfail run: exited=True exitcode=0
HIT: Warning: failed to write noise shaper state file: C:\Users\user\AppData\Roaming\ConvoPeq\noise_shaper_learn.xml
```

こちらも警告のみで exit code 0。クラッシュも無言の失敗も無い。

### 3.3 書き込み失敗時に元ファイルが置換されないこと（障害注入で実証）

ディスク満杯は安全に作れないため、書き込み API の境界で失敗を注入した。WinDbg の `cdb` で
`kernel32!WriteFile` にブレークポイントを置き、3 番目の引数（書き込みバイト数）が 100,000 を超える
呼び出しだけを対象にした。この条件に該当するのは 278 KB の `noise_shaper_learn.xml` の書き込みだけで、
11 KB の `device_settings.xml` やログ出力は影響を受けない。注入は 2 種類行った。

#### 3.3.1 全量失敗（1 バイトも書けない）

関数本体を飛ばして戻り値を 0 にした。

```
0:000> bp kernel32!WriteFile ".if (@r8 > 0n100000) { .echo >>> FORCED_WRITE_FAILURE; r @rip = poi(@rsp); r @rsp = @rsp + 8; r @rax = 0 }; gc"
>>> FORCED_WRITE_FAILURE
```

| 観測項目 | 値 |
| --- | --- |
| 元ファイル | `0DCDF3D9D024267D` / 278,332 B（実行前と同一） |
| 一時ファイルの残留 | 0 件 |
| ログ | `Warning: failed to write noise shaper state file: C:\Users\user\AppData\Roaming\ConvoPeq\noise_shaper_learn.xml` |
| プロセス | デバッガはプロセス終了まで継続（実行 9,635 ms） |

書き込みが失敗すると置換は行われず、元ファイルは完全なまま残り、W-1 の警告が出る。
同じ手順を 2 回実施し、どちらも「強制失敗 1 回・元ファイル不変・警告 1 件・一時ファイル残留 0」で一致した。
この経路は §1.1 の 3 点（`writeInternal` が失敗を返す → `writeText` が false → `saved` が false →
置換しない）が実際に通ることを意味する。

#### 3.3.2 部分書き込み（途中まで書けてから失敗）

`WriteFile` に渡すバイト数を 278,373 から 100,000 に差し替え、実際に 100,000 バイトだけ書かせた。
一時ファイルは不完全な内容（100,000 バイト）で作られ、`write()` は
「書けたバイト数 != 要求バイト数」で false を返す（`juce_FileOutputStream.cpp:114-120`）。

| 観測項目 | 値 |
| --- | --- |
| 強制注入 | 1 回（`>>> PARTIAL_WRITE_INJECTED`） |
| 元ファイル | `0DCDF3D9D024267D` / 278,332 B（実行前と同一） |
| 一時ファイルの残留 | 0 件（不完全な一時ファイルは破棄された） |
| ログ | `Warning: failed to write noise shaper state file: ...\noise_shaper_learn.xml` |

不完全な内容が作られたにもかかわらず、元ファイルは置換されず、警告が出て、書きかけの一時ファイルは
残らなかった。条件 4 が求める性質はこの 2 ケースで確認できた。

**限界**: 残るのは「実ディスク満杯」での確認だけである。同じ `saved` ガードを通るため結果は変わらない
見込みだが、実測はしていない（§3.4）。

### 3.4 実ディスク満杯で閉じるのに何が必要か

全量失敗と部分書き込みは注入で確認できた。実ディスク満杯での確認だけが残っており、それを成立させるには
次のいずれかが要る。いずれも今回の作業境界の外にある。

| 手段 | 必要なもの | 今回できない理由 |
| --- | --- | --- |
| 容量不足を実際に起こす | 満杯寸前のボリューム、または小容量の仮想ディスク | 既存ボリュームを埋めるのは破壊的。VHD の作成・マウントは管理者権限が要る |
| 障害注入（ファイルシステムフィルタ／I/O エラー注入） | 管理者権限と専用ドライバ | この環境では導入できない |
| 注入可能なストリームを使う単体テスト | テストコードの追加と、保存処理のストリーム差し替え | テストコード追加は承認範囲外。保存処理に試験用の入口を足すのも境界外 |

現時点の保証は「置換以外に元ファイルへ書く経路が無いこと」（§1.2）、「書込み失敗が `writeText` の
戻り値と `status` に出ること」（§1.1）、「失敗時に置換されないこと」（§3.3.1）、
「部分書き込みでも置換されないこと」（§3.3.2）の 4 点で成り立っている。残るのは
「実ディスク満杯でも同じになるか」の 1 点だけである。

---

## 4. Debug ビルドでの挙動

Debug 構成（`build/CMakeFiles/ConvoPeq.dir/Debug/src/DeviceSettings.cpp.obj` = 13:48:12、
`ConvoPeq_artefacts/Debug/ConvoPeq.exe` = 13:48:34）で 3 ケースを実行した。
`jassertfalse` に到達するとデバッガ非接続では `__debugbreak` でプロセスが落ちるため、exit code 0 が
「アサートを踏んでいない」ことの観測になる。

| ケース | exit | 結果 |
| --- | --- | --- |
| 正常保存 | 0（6,640 ms） | 出力 `0DCDF3D9D024267D`、警告なし、更新時刻変化あり |
| 置換失敗（元ファイル読み取り専用） | 0（7,303 ms） | 警告 2 件、元ファイル不変、一時ファイル残留 0 |
| 一時ファイル作成失敗（設定ディレクトリを作れない） | 0 | 警告 1 件、アサート無し |
| 書き込み失敗（`WriteFile` 強制失敗を注入） | 0（10,492 ms） | 強制失敗 1 回、元ファイル不変、警告 1 件、残留 0 |

前版（`replaceWithText`）では、一時ファイルが作れない状況で `overwriteTargetFileWithTemporary` の
`jassertfalse`（`juce_TemporaryFile.cpp:111-118`）に到達し得た。今回版は `openedOk()` が false のとき
置換を呼ばないため、その経路自体が無くなっている。上の 4 ケースはいずれも exit 0 で、
Debug でも停止しないことを実測で確認した。

Debug をデバッガ配下で動かすと、この変更とは無関係な既存のアサートが 26 件（4 箇所:
`AudioEngine.h:3342`、`AudioEngine.Threading.cpp:319`、`SafeStateSwapper.h:293`、`text/juce_String.cpp:327`）
出る。書き込み失敗を注入した場合と注入しない対照実験で、件数も箇所も完全に一致した。
保存経路（`DeviceSettings.cpp`、`juce_TemporaryFile.cpp`、`juce_File.cpp`）由来のアサートは 1 件も無い。

---

## 5. Release ビルドと既存テスト

```
cmake --build build --config Release --target ConvoPeq
=== BUILD_OK ===
```

警告は既存の `MklFftEvaluator.h:326/508` の C4390 のみ。

```
ctest --test-dir build -C Release -j 4
100% tests passed out of 61
Total Test time (real) = 328.98 sec
```

61/61 成功。

## 6. 保存 → 再読込

```
reload idempotency: exit=0 in=BC294B76D1E51E13 out=BC294B76D1E51E13 IDENTICAL=True
dir entries: 0 temp leftovers
```

プローブ出力（LF・エスケープ・非 ASCII を含む）を次回起動で読み直し、同じバイト列を書き戻せた。

---

## 7. 変更しなかった範囲

| 項目 | 状態 |
| --- | --- |
| BUG-2 | 未着手・未変更 |
| shutdown 本体・Worker 停止 API・所有権／破棄順序・`workersStuck_`・`B_pre` | 未着手・未変更 |
| U-3（W3・W4・W7 の停止要求 API） | 未着手・未変更 |
| 他の保存処理（`saveSettings` の `xml->writeTo`、`loadNoiseShaperState`） | 未変更 |
| 10 MB ガード | 維持（式も型も同じ） |
| W-1 の警告文とログ出力 | 維持 |
| テストコード・計測用コード | 追加していない |
| 既存の未コミット変更 | 保持 |
| Git 操作 | commit / checkout / reset / stash は行っていない |
| 実行で更新された設定ファイル | バックアップから復元済み |

Debug ビルドは `build/` の既存マルチコンフィグ構成（`Ninja Multi-Config`）を使い、
`--config Debug` で `ConvoPeq` ターゲットだけをビルドした。ただし 1 点副作用があった。
`ConvoPeq_artefacts/JuceLibraryCode/JuceHeader.h` は Debug と Release で共有されており、
Debug ビルド後の次の Release ビルドで同ヘッダが再生成されて Release 側の 96 オブジェクトが再コンパイルされた
（`14:03` に `JuceHeader.h` 更新 → `14:04` に Release 再リンク）。ソースは同じなので結果は変わらないが、
**§2〜§3 の測定は再ビルド後のバイナリで取り直した**（プローブ `BC294B76D1E51E13`、読み取り専用 6,790 ms、
再読込の冪等性、ctest 328.98 秒）。

---

## 8. 未確認事項と残課題

| # | 内容 | 状態 |
| --- | --- | --- |
| U-1 | 実ディスク満杯での書き込み失敗 | **未実施**。全量失敗と部分書き込みは API 境界の障害注入で確認済み（§3.3.1／§3.3.2） |
| U-2 | 10 MB 超過分岐の実動作 | 未実測（巨大ツリーを作れない）。コードレビューのみ |
| U-3 | 10 MB ガードと読込上限 8 MiB の不整合 | **別課題として記録**（`InputInspection.h:23`）。今回は修正していない |

`InputInspection.h:23` の `kMaxXmlFileBytes = 8 * 1024 * 1024` に対し、保存側は 10 MiB まで許す。
8〜10 MiB のファイルは書けた次回に読めず、`parseXmlFile` が `nullptr` を返して新規ツリーで上書きされる。
BUG-3 が作った問題ではなく、Owner 指示により今回は触っていない。
