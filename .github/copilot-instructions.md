# コーディング規約 (Coding Standards)

## 1. 使用フレームワーク・ライブラリ
本プロジェクトでは、以下のライブラリの特定バージョンを使用します。

- **JUCE Framework V8.0.12**
    - [公式リポジトリ](https://github.com/juce-framework/JUCE/tree/8.0.12)
    - [APIドキュメント](https://docs.juce.com/master/index.html)
    - **注意事項**: 
        - 必ず最新ドキュメントで当該関数の存在を確認すること。
        - 戻り値の意味、副作用、スレッド安全性、前提条件を完全に理解した上で使用すること。
        - 公式サンプルはVST3等が主であり、本アプリ（スタンドアローン）とは構造が異なる点に注意。
        - 公式サンプルにない過剰な安定化機構は導入しない。

- **r8brain-free-src**
    - [公式リポジトリ](https://github.com/avaneev/r8brain-free-src.git)

- **Intel oneAPI MKL (oneMKL)**
    - [製品ページ](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
    - [Windows用開発者ガイド](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2025-2/overview.html)

- **Windows SDK**
    - [技術ドキュメント](https://learn.microsoft.com/ja-jp/windows/apps/windows-sdk/)

## 2. 編集制限
プロジェクトルート内の以下のディレクトリにあるファイルは**絶対に編集禁止**です。
- `/JUCE` フォルダ
- `/r8brain-free-src` フォルダ

## 3. 禁止事項
- **構造化例外処理 (SEH)**: 絶対に使用しないこと。
- **Audio Thread内でのブロッキング処理**: 
    `getNextAudioBlock()` 等が呼ばれるスレッド内では、待機が発生する可能性のある以下の処理を厳禁とする。
    - **メモリ操作**: `new`, `malloc`, `vector::resize`, `mkl_malloc`, `mkl_free`, `_aligned_malloc`, `vslNewStream`
    - **例外・計算**: `try-catch`, `std::exp()`, `libm` 呼び出しを伴う関数
    - **MKL設定**: `DftiCommitDescriptor`, `mkl_set_interface_layer`
    - **同期・通信**: `mutex lock`, `critical section`, `condition_variable`, `MessageManager` へのアクセス
    - **I/O・リソース**: ファイルI/O, コンソール出力, IRの再ロード, `std::shared_ptr` の使用, MMCSS設定
    - **JUCE特定処理**: `AudioBlock::allocate`, `AudioBlock::copyFrom`, `FFT::performFrequencyOnlyForwardTransform`（事前確保なし）

## 4. メモリ管理とアライメント
- **oneMKL使用箇所のメモリ確保**:
    - `Audio Thread` 以外かつ MKL 使用箇所では、`new`, `std::vector`, `std::make_unique` を使用しない。
    - 代わりに `mkl_malloc` / `mkl_free`, `_aligned_malloc(64)`, `std::pmr` + カスタムアロケータを使用すること。
    - メモリは **64byteアライメント** を必須とする。
    - メモリ確保は `prepareToPlay()` 等のメッセージスレッド（非Audio Thread）でのみ行うこと。
- **リーク対策**: 
    - デストラクタの設置漏れに細心の注意を払い、メモリリークを完全に防止すること。

## 5. 信号処理仕様
- **データ型**: 外部入出力以外のデータ処理はすべて **64bit double** で行うこと（スペアナ演算のみ `float` 可）。
- **デノーマル対策**: 非常に小さな値を扱う際のパフォーマンス低下（デノーマル数）を防ぐ対策を徹底すること。

## 6. 安全性
- メモリ解放の確実な実行と、ポインタ管理の厳格化。
