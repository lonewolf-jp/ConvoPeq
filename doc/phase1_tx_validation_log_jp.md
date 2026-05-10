# Phase 1 TX検証ログ（T5時点）

更新日: 2026-05-10

## 1. 対象

- 設計書: doc/nextgen_runtime_transition_design_jp.md
- 必須TX: TX-02, TX-07, TX-10
- 現在の実装到達点: T5（AudioBlock read-only実行統合）

## 2. 実行環境

- Windows x64
- Visual Studio 2026 (MSVC 18.5.2)
- oneAPI 環境有効
- ビルド確認コマンド:
  - cmd.exe /d /c "call \"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat\" x64 && call \"C:\Program Files (x86)\Intel\oneAPI\setvars.bat\" intel64 && cd /d C:\VSC_Project\ConvoPeq && cmake --build build --config Debug"

## 3. 実施結果サマリ

| TX | 判定 | 種別 | 根拠 |
| --- | --- | --- | --- |
| TX-02 | Waived | ユーザー判断で実機試験スキップ | 本セッションで実装継続を優先 |
| TX-07 | Partial Pass | 静的確認 + Debug Build | releaseResources で pending commit drain -> retire -> release の経路を確認 |
| TX-10 | Waived | ユーザー判断で実機試験スキップ | 本セッションで実装継続を優先 |

## 4. TX-07 根拠

### 4.1 shutdown移行直後の停止処理

- shutdownフラグ立ち上げ: src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:18
- AsyncUpdaterキャンセル: src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:23

### 4.2 pending commit drain

- deferredCommitQueue の回収: src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:81

### 4.3 retire 経路への回収

- staging new/old retire: src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:89, src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:91
- active/fading/queued/pending retire: src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:96, src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:98, src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:100, src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:102, src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:104

### 4.4 queue側の停止整理

- rebuild停止時に runtime command queue clear: src/audioengine/AudioEngine.RebuildDispatch.cpp:302

## 5. TX-08 根拠（参考）

- requestRebuild(kind) の shutdown gate: src/audioengine/AudioEngine.RebuildDispatch.cpp:67
- requestRebuild(sr, bs) の shutdown gate: src/audioengine/AudioEngine.RebuildDispatch.cpp:103

## 6. T5差分の安全性確認

- T5で共通化したヘルパーは AudioEngine.h の inline に限定し、Audio Thread で lock/allocation/delete/logging を追加していない。
- 変更後の Debug Build は成功（ConvoPeq.exe までリンク）。

## 7. 次アクション

1. TX-02 実機手順実施（IR Replace 単発）
2. TX-10 実機手順実施（automation burst 100件）
3. publish件数/retire件数/reclaim遅延の計測値を本ログに追記

## 8. TX-02 実機記録テンプレート（IR Replace 単発）

### 8.1 事前条件

- 実行日時:
- 実行者:
- SR / block:
- IR状態（finalized確認）:
- 入力素材:

### 8.2 実行手順チェック

1. 再生中に IR を 1 回だけ差し替える。
2. Build -> Validate -> Warmup -> Publish -> Crossfade -> Retire の順をログで確認する。
3. 切替中に click/pop がないことを確認する。

### 8.3 観測値

- publish件数:
- retire件数:
- reclaim完了有無:
- フェード中断有無:
- 異常ログ有無:

### 8.4 判定

- 結果: Pass / Fail
- 判定理由:
- 失敗時メモ（再現条件・修正候補）:

## 9. TX-10 実機記録テンプレート（automation burst 100件）

### 9.1 事前条件

- 実行日時:
- 実行者:
- SR / block:
- テスト対象パラメーター:
- 100件投入方法（UI操作 / スクリプト等）:

### 9.2 実行手順チェック

1. 同一窓で 100 件規模の連続更新を投入する。
2. coalescing により rebuild 回数が圧縮されることを確認する。
3. 最終パラメーターが期待値と一致することを確認する。

### 9.3 観測値

- 総更新件数:
- 実際のrebuild件数:
- 抑制件数（総更新 - rebuild）:
- CPUスパイク（主観 / 取得値）:
- 最終値一致:

### 9.4 判定

- 結果: Pass / Fail
- 判定理由:
- 失敗時メモ（再現条件・修正候補）:

## 10. 実行コマンド（再現用）

### 10.1 Debugビルド

- cmd.exe /d /c "call \"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat\" x64 && call \"C:\Program Files (x86)\Intel\oneAPI\setvars.bat\" intel64 && cd /d C:\VSC_Project\ConvoPeq && cmake --build build --config Debug"

### 10.2 既存プロセス停止（必要時）

- taskkill /F /IM "ConvoPeq.exe" >nul 2>&1 || exit 0

### 10.3 実行時ログ観測の観点

- TX-02: IR差し替え直後の build -> validate -> warmup -> publish -> crossfade -> retire が順序崩壊しないこと
- TX-10: 100件更新投入時に最終値一致と rebuild圧縮（coalescing）を確認すること

## 11. このセッションで確定した事実（実測）

- T5差分適用後に Debug Build 成功（ConvoPeq.exe リンク完了）
- releaseResources/shutdown gate/queue clear の静的根拠は本ファイル 4章・5章に記載済み
- TX-02 / TX-10 はユーザー判断で実機試験をスキップ（Waived）

## 12. 追記ルール（運用）

1. 実行ごとに「実行日時」「コミット（または作業時点）」「結果」を追記する。
2. Fail 時は再現条件を最小化し、再現手順を 5 行以内で残す。
3. Pending を Pass に変えるときは、観測値（publish/retire/reclaim または rebuild圧縮）を必ず 1 つ以上記録する。
