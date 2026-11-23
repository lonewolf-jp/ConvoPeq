# build.bat による Intel icx Release ビルド（ConvoPeq.exe のみ）

- 実施日時: 2026-11-15 12:50〜13:15 JST
- 対象: `C:\VSC_Project\ConvoPeq`（main `4d476d5c`、未コミット変更あり）
- 目的: プロジェクトルートの `build.bat` を icx モードで使い、Release の `ConvoPeq.exe` だけを生成する

---

## 1. 実行したコマンド

```
build.bat Release icx app clean nopause
```

- `app`: ConvoPeq ターゲットのみをビルドする（テストターゲットはビルドしない）
- `clean`: `build-icx` を作り直す

`clean` が 2 回必要だった理由:

1. 最初の実行は識別ゲートで停止した。`build-icx` のスタンプが旧状態（`compiler_path` がフルパス、`source_revision` が `c94d3767`）で、現在値（`icx`、`4d476d5c+dirty`）と一致しなかった。
2. `clean` 後の 1 回目はソース修正前だったためコンパイルで停止した。2 回目の configure で `-DCMAKE_C_COMPILER=icx` が再適用され、キャッシュ値がフルパスから `icx` に戻ってゲートが再び不一致になった。そのため、ソース修正後にもう一度 `clean` した。

---

## 2. icx で通すために必要だったソース修正

icx（Clang 系）は「外側クラスの定義内では、入れ子クラスの既定メンバ初期化子を使えない」という C++ の規則を厳密に適用する。MSVC はこれを緩和するため、MSVC では通っていた。5 箇所で停止した。

| ファイル | 箇所 | 原因 |
| --- | --- | --- |
| `src/core/ReaderMeasureRegistry.h` | 54, 55 | `static inline Record recs_[...]{}` / `Agg aggs_[...]{}` が入れ子型の既定メンバ初期化子を要求 |
| `src/audioengine/ISRRetireLatency.h` | 1135 | 既定引数 `Classification cls = Classification{}` が外側クラス定義内にあり、`excluded = false` を要求 |
| `src/core/EpochDomain.h` | 818 | `static inline HoldWitness holdWitness_[...]{}` |
| `src/core/ScopedEpochReader.h` | 226 | `static inline MeasureRecord measure_[...]{}` |

修正方針は 1 つに揃えた。既定メンバ初期化子を **constexpr 既定コンストラクタ**に置き換える。初期値は従来と同一で、C++20 では `std::atomic` のコンストラクタが constexpr なので定数初期化のままになる。

- 対象の型は既定構築しかされておらず、集成体初期化（`Classification{a, b, ...}` のような形）の利用は無いことを grep で確認した。
- 変更量: 4 ファイル、+63 / −35。

---

## 3. 結果

| 項目 | 値 |
| --- | --- |
| 実行 | `build.bat Release icx app clean nopause` |
| 所要 | 13:02:06〜13:10:13（8 分 7 秒） |
| 終了表示 | `[140/141] Linking CXX executable ConvoPeq_artefacts\Release\ConvoPeq.exe` / `[SUCCESS] Executable created successfully.` / `Build Complete` |
| 生成物 | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`（48,817,152 B、2026-11-15 13:10:13） |
| バージョン | FileVersion / ProductVersion 0.6.12 |
| コンパイラ | `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` = `C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe` |
| コンパイル数 | 141 ステップ中 135 が CXX のコンパイル。`ninja -t commands ConvoPeq` の 140 行が icx |
| 診断 | コンパイルエラー 0、警告 156 行 |
| テスト | `app` 指定のため `*Tests.exe` は生成していない |

`build.bat` の終了コードは 1 を返すが、これは `clean` 内の `timeout /t 2` が入力リダイレクト不可で失敗した名残で、ビルド自体は成功している（`[SUCCESS]` の行と exe の生成で確認）。

---

## 4. 確認したこと

- `build-icx\CMakeCache.txt` の C / C++ コンパイラが `icx.exe`、`CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`
- `ninja -C build-icx -f build-Release.ninja -t commands ConvoPeq` の 140 行すべてが icx
- 生成物のサイズ・更新時刻・バージョン情報
- `build-icx` 配下の exe は `ConvoPeq.exe` とビルドツール（`juceaide.exe`、CMake のコンパイラ確認用）のみで、テスト実行ファイルは無い
- MSVC でも 4 ヘッダが通ること（`cl /std:c++20 /EHsc` で 4 ヘッダを include する単体 TU をコンパイルし、終了コード 0）

---

## 5. 既存の環境問題（今回の変更とは無関係）

- `build/`（MSVC）の CMake キャッシュは `CMAKE_LINKER` が GNU Octave の `ld.exe` を指しており、再構成に失敗する（AGENTS.md に記載のある既知問題）。MSVC の全体ビルドをやり直すには `build/` を作り直す必要がある。
- `build.bat` の icx モードは 2 回目以降の configure で `compiler_path` が `icx` に戻るため、識別ゲートが再び不一致になる。次回も `clean` が必要。
