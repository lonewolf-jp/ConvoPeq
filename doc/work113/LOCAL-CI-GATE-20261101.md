# ローカル CI ゲート 実行報告（2026-11-01）

## 結論

対象コミット `6e2f1d96` について、GitHub Actions と同等のローカル検証を実行し、必須項目がすべて成功した。これにより開発ゲートは通過とし、GitHub Actions の完了は待たない。TSan は CI と同じく skip となり、CodeQL は非同等のため未実行である。この 2 件は記録に残す。

## 1. 検証結果

| 検証 | コマンド | 結果 |
|---|---|---|
| 静的解析一式 | `.github/scripts/run-all-workflow-checks.ps1` | **成功**（compliance 2、lint 1、authority 4、isr-verifier 18、audit 1） |
| 必須ビルド（Release） | `cmake -S . -B build-ci-local -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCONVOPEQ_REQUIRE_MKL=ON` の後、`ConvoPeq` と `AudioEngineHarness` をビルド | **成功**（configure 0、ConvoPeq 0、Harness 0） |
| ASan 全ターゲット | `cmake --build build-asan-local --config RelWithDebInfo` | **成功**（全 680 ターゲット） |
| ASan テスト | `ctest --test-dir build-asan-local -C RelWithDebInfo --output-on-failure` | **51/51 成功**（`AudioEngineHarness` 355.30 秒） |
| TSan | WSL Ubuntu 26.04 で CI と同じ `-DENABLE_TSAN=ON` | **skip を再現** |
| CodeQL | — | **未実行（非同等）** |

環境は Windows 11 build 26300、Core i7-11700K、MSVC 19.51.36260、CMake 4.4.5、Ninja 1.13.2、oneAPI 2026.2。

## 2. GitHub Actions との区別

`6e2f1d96` の GitHub Actions は、ローカル検証を終えた時点で success 8、skipped 1、in progress 2（`Analyze (c-cpp)` と `RelWithDebInfo ASan`）、failed 0 だった。Owner 決定により、この完了は開発開始の条件としない。

意図的な SKIPPED は `Windows MSVC / probe (intentional failure, dispatch only)` である。dispatch 専用の検出能力確認で、通常の push と PR では走らない。

## 3. WSL で TSan を代替できるか

**skip の再現としては代替できる。本物の TSan 検証にはならない。**

WSL は Ubuntu 26.04.1 LTS で、clang 21.1.8 が入っている。sudo はパスワードなしでは通らないため apt は使えず、`pip install --user --break-system-packages cmake ninja` で CMake 4.4.4 と Ninja 1.13.2 を入れた。CI と同じ構成コマンドを実行すると `Could NOT find PkgConfig` で失敗し、`build.ninja` が生成されない。CI と同じ skip 分岐に入る。

ただし失敗の根因は CI と WSL で異なる可能性がある。WSL は pkg-config 不在で失敗した。CI 側はスクリプトが「MKL/IPP が無い」と出力しているが、実際の cmake エラー本文はログから確認できていない。根因まで揃えるには一度だけ `sudo apt-get install pkg-config libc++-dev libc++abi-dev` が必要になる。

どちらの環境でも、Linux に MKL/IPP が無いため ISR テストのビルドには到達しない。実際のデータ競合を検出するには、CMake に MKL 非依存の ISR テスト構成を用意する設計変更が要る。

## 4. 判明した落とし穴

第一に、`VSCMD_START_DIR` が設定された状態で `vcvars64.bat` を呼ぶと、エラーを出さずに `LIB` が未設定のままになる。その結果リンクが `LNK1104: kernel32.lib` で落ちる。`set "VSCMD_START_DIR="` と `cd /d C:\` を先に実行すれば避けられる。

第二に、前回のテストプロセスが残っていると、次のビルドが `LNK1104: AudioEngineHarness.exe を開けない` で失敗する。ビルド前に `AudioEngineHarness` と `ctest` を終了させる必要がある。

なお Release 構成で `ctest` を実行すると 49 件が Not Run になる。CI の Release ジョブは 2 ターゲットしかビルドせず、テストは ASan ジョブが担うためである。テストまで見るなら全ターゲットをビルドしてから ctest する。

## 5. 次に必要な判断

1. **D-01 の製品 2 行**（`PublicationAdmission::evaluate()` に `|| engine.testFadingRuntimePresent()` を戻す）
2. **測定フェーズの開始**（D-02 の非干渉性 A/B と案2 E-1 の比較）
3. **Debug の assert 違反の修正可否**（`ISRDSPHandle.cpp:529`。製品側の TOCTOU）
4. **間欠 SEGFAULT の調査開始可否**（`ctest` 3 回に 1 回、IR 系テストで発生）
5. **本報告書と関連文書の Git 追跡**（追跡すれば未追跡は 18 に戻る）

U5 は FAIL・既定 OFF、U3/U4 は HOLD、許容値 +50 µs と p99 基準は変更していない。追加測定も実施していない。
