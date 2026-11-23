# icx 対応のコンパイル互換性修正（最新 main 上での再検証とコミット）

- 実施日時: 2026-11-15 13:30〜14:05 JST
- 作業ブランチ: `fix/icx-default-member-initializers`
- worktree: `C:\VSC_Project\icx-fix`
- 基点: 最新 main `528590c75454266fdd72a240ca8ce96efe2428eb`（PR #213 のマージコミット）
- 種別: コンパイル互換性修正の再検証とコミット（push は未実施）

---

## 1. 最新 main への追従

- 新しい worktree を `528590c7` から作成した。既存の main 作業ツリー（`4d476d5c`、他タスクの未コミット変更 188 件）には触れていない。
- 対象 4 ファイルが `4d476d5c` と `528590c7` で同一であることを `git diff --name-only 4d476d5c 528590c7 -- <4 files>` で確認した（差分 0 件）。したがって、修正内容・初期値・初期化動作は最新 main でもそのまま成立する。
- 4 ファイルを適用した後の差分は **4 ファイル +63 / −35** で、承認された範囲と一致する。無関係な変更は無い。

## 2. 修正内容

icx（Clang 系）は「外側クラスの定義内では、入れ子クラスの既定メンバ初期化子を使えない」という C++ の規則を厳密に適用する。MSVC は緩和するため、MSVC では通っていた。5 箇所で停止する。

| ファイル | 箇所 | 修正 |
| --- | --- | --- |
| `src/core/ReaderMeasureRegistry.h` | `Record` / `Agg` と `static inline` 配列 2 本 | 既定メンバ初期化子を constexpr 既定コンストラクタへ |
| `src/audioengine/ISRRetireLatency.h` | `Classification` と既定引数 `Classification{}` | 同上 |
| `src/core/EpochDomain.h` | `HoldWitness` と `holdWitness_` | 同上 |
| `src/core/ScopedEpochReader.h` | `MeasureRecord` と `measure_` | 同上 |

初期値は従来と同一（`slot = 0xFFFFFFFFu`、他は 0 / false / 空タグ）。C++20 では `std::atomic` のコンストラクタが constexpr なので、配列は定数初期化のまま。4 型はいずれも既定構築のみで、集成体初期化の利用はツリー内に無い。

## 3. 再ビルド結果

### 3.1 icx Release（最新 main）

```
build.bat Release icx app clean nopause
```

| 項目 | 値 |
| --- | --- |
| 結果 | `[138/139] Linking CXX executable` / `[SUCCESS] Executable created successfully.` / `Build Complete` |
| 生成物 | `C:\VSC_Project\icx-fix\build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`（48,674,304 B、13:55:13） |
| バージョン | 0.6.12 |
| コンパイラ | `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` = `icx.exe` |
| 診断 | コンパイルエラー 0 |
| テスト実行ファイル | 0 個（`app` 指定） |

### 3.2 MSVC Release（最新 main、Octave を PATH から除去）

既知の環境問題（GNU Octave の `ld.exe` が PATH で先に来る）を切り分けるため、PATH から Octave / mingw64 / msys を除いて `vcvars64` を呼び、`build.bat Release app nopause` を実行した。

| 項目 | 値 |
| --- | --- |
| 結果 | `[GATE-OK] build identity stamp written (first configure)` / `[SUCCESS]` / `Build Complete` |
| 生成物 | `C:\VSC_Project\icx-fix\build\ConvoPeq_artefacts\Release\ConvoPeq.exe`（47,907,328 B、14:00:09） |
| コンパイラ | `cl.exe`（MSVC 14.52.37025） |
| リンカ | `CMAKE_LINKER` = MSVC の `link.exe`（Octave の `ld.exe` ではない） |
| 診断 | コンパイルエラー 0 |

この結果から、以前の MSVC 構成失敗は PATH とキャッシュ由来であり、今回の 4 ファイル修正が原因ではないことを確認した。

### 3.3 MSVC ヘッダ単体

4 ヘッダを include する単体 TU を `cl /std:c++20 /EHsc /utf-8` でコンパイルし、終了コード 0。

## 4. コミット

| 項目 | 値 |
| --- | --- |
| SHA | **`fc9d3f9a5dccdb8f9f475bbe62066db8b535951a`** |
| 親 | `528590c7`（最新 main） |
| ブランチ | `fix/icx-default-member-initializers` |
| 変更ファイル | `src/core/ReaderMeasureRegistry.h`、`src/audioengine/ISRRetireLatency.h`、`src/core/EpochDomain.h`、`src/core/ScopedEpochReader.h` |
| 差分 | 4 ファイル、+63 / −35 |
| push | **完了**（`origin/fix/icx-default-member-initializers` = `fc9d3f9a`） |
| Pull Request | **#214**（**MERGED**、2026-11-15T06:16:48Z、base `main`） |
| merge commit | **`daf45501020a7471d03445e51eee2061291413ec`**（`origin/main` の head） |

### 4.1 CI とレビュー

| 項目 | 結果 |
| --- | --- |
| チェック | 17 件中 **16 success**、1 skipped（`Windows MSVC / probe` = dispatch 専用の意図的 SKIP）、失敗 0 |
| 必須チェック | `Windows MSVC / Release` = success |
| その他 | ASan（RelWithDebInfo）60.3 分で success、TSan success、CodeQL（Analyze ×4 と集約）success、isr-verify / authority / lint / compliance success |
| `mergeStateStatus` | **CLEAN**（`mergeable` = MERGEABLE） |
| CodeRabbit | インライン指摘 0 件、「No actionable comments were generated in the recent review」、Merge Risk = Minimal、pre-merge チェック 4 件 passed |

### 4.2 マージ後の main CI

`daf45501` を head とする 6 workflow が起動し、**6 件すべて success** だった。

| workflow | 結果 |
| --- | --- |
| Windows MSVC Build | success |
| Sanitizer CI | success |
| ISR Authority Compliance v6.4 | success |
| audioengine-lint | success |
| list-compliance | success |
| Push on main（CodeQL） | success |

## 5. 未実施の検証項目

- テストの実行（`app` 指定のためテストターゲットをビルドしていない。ctest は未実行）
- icx ビルドのテストスイート合格、icx 環境での実機動作
- ConvoPeq 全体の icx 互換性（今回の 4 ファイル以外に未対応箇所が残る可能性は排除していない）
- アプリの起動確認
- マージ

## 6. 補足

- `build.bat` の終了コードは 1 を返すが、これは `clean` 内の `timeout /t 2` が入力リダイレクト不可で失敗した名残で、ビルド自体は成功している（`[SUCCESS]` 行と実行ファイルの生成で確認）。ラッパー全体の終了コードが正常だったとは扱わない。
- 既存の main 作業ツリー（`C:\VSC_Project\ConvoPeq`）にも同じ 4 ファイルの未コミット修正が残っている。今回のコミットは別 worktree のもので、main 作業ツリーは変更していない。
