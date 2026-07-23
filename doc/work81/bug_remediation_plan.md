# bug.md 改修計画（v6）

**作成日**: 2026-07-23
**改訂日**: 2026-07-23（v6 最終レビュー反映）
**対象**: bug_review.md で「有効」判定 21件 + 「設計上の制約」4件 = **25件（実装工数換算 20件相当）**
**方針**: 優先度（P0〜P3）に分類し、修正難易度・影響範囲を明記
> **注**: 対象 25件のうち、B2/B3 は P3扱い、H8/C3 は同一対策（jassert スレッド検証）として工数集計しているため、工数サマリーでは **20件相当** としている。
**主な改訂点**:
- H12: 呼び出し元例外安全性の確認結果を追記（RuntimeBuilder は try-catch で囲まれている）
- R3: nlohmann/json 導入から `escapeJson()` 自作に変更（プロジェクト方針に適合）
- B5: `if(MSVC)` 削除から条件追加（`IntelLLVM` 含む）に変更
- B2/B3: P1→P3 に降格（実害が限定的）
- D2: 工数を 1-2h→0.5-1日に修正（AVX/MKL/denormal 影響評価が必要）
- C5: `atomic<double>` 化から `cachedTailLength` 方式に変更
- C4: `weak_ptr` 化からコメント補強のみに変更（アーキテクチャ変更不要）
- U3: Logger 追加前に呼び出し元スレッド確認を明記

---

## 優先度定義

| 優先度 | 意味 | 対応期限目安 |
|--------|------|-------------|
| **P0** | クラッシュ / データ破損 / セキュリティ | 速やかに |
| **P1** | 機能不正確 / ビルド不安定 | 次回リリースまでに |
| **P2** | UX 向上 / 防御的コーディング | 余裕があるとき |
| **P3** | 最適化 / 将来的な风险低減 | 長期的改善 |

---

## P0: クラッシュ / データ破損（2件）

### H12: `transferIRStateFrom()` noexcept + bad_alloc → `std::terminate()`

- **ファイル**: `src/ConvolverProcessor.h:1107-1124`
- **問題**: `noexcept` 関数内で `updateIRState()` が `aligned_make_unique` / `std::make_unique` を呼び、`bad_alloc` 発生時に `std::terminate()` になる
- **検証結果**: ✅ 確認済み。呼び出し元は `RuntimeBuilder.cpp:447` のみ。同関数は **`try` ブロック内（行441-）** にあるため、例外を正しく catch できる
- **呼び出し元**: `RuntimeBuilder.cpp:447`（`try { ... }` 内）。`ConvolverProcessor.h:1114` は内部呼び出し
- **影響**: メモリ不足時にアプリケーションが即終了。エラー回復不能
- **修正方針**:
  1. `noexcept` を外す
  2. **呼び出し元の例外安全性を確認**: RuntimeBuilder.cpp は `try-catch` で囲まれているため安全。例外が伝播しても `BuildResult::error = BuildError::...` で回復可能
- **難易度**: Low（1行変更 + 呼び出し元確認済み）
- **影響範囲**: ConvolverProcessor のみ
- **テスト**: Windows 環境での `bad_alloc` 注入テスト（`operator new` 差し替え / テスト用 allocator が最も再現性が高い。`_set_new_handler` は補助的に使用）

### R3: JSON 手組み立てエスケープ不足 → JSON 壊損

- **ファイル**: `src/audioengine/ISREvidenceExporter.cpp:302-305`
- **問題**: `manifest += " \"runId\": \"" + runId + "\""` のような文字列結合で、`runId`/`buildMode`/`proofLevel` に `"`/`\`/改行が含まれると JSON が壊れる
- **検証結果**: ✅ 確認済み。302-305行で `runId`, `buildMode`, `proofLevel` を直接文字列結合
- **影響**: evidence manifest が解析不能、証跡チェーン破損
- **修正方針**:
  - **推奨**: `escapeJson()` 関数を ISREvidenceExporter.cpp 内に実装。nlohmann/json 等の外部ライブラリ導入は不可（プロジェクト方針：依存を極力増やさない）
  - `escapeJson()` の実装内容: `\"→\\\"`, `\→\\\\`, JSON仕様に従い0x00〜0x1Fの制御文字をすべてエスケープ（`\n→\\n`, `\r→\\r`, `\t→\\t`, `\b→\\b`, `\f→\\f`。それ以外は `\\u00XX` 形式）。空文字への削除は不正（文字列そのものが変わるため）
  - 出力形式: shutdown_trace.json, retire_latency_report.json, evidence_manifest.json の3種のみ
- **難易度**: Low（`escapeJson()` 関数20-40行程度 + 呼び出し箇所3-5箇所の変更）
- **影響範囲**: ISREvidenceExporter のみ
- **テスト**: 特殊文字（制御文字含む）を含む runId での生成テスト

---

## P1: 機能不正確 / ビルド不安定（6件）

### U1: 両方バイパス時の表示誤り「Conv → PEQ」

- **ファイル**: `src/DeviceSettings.cpp:660-685`
- **問題**: `convBypassed && eqBypassed` の場合、`else` 分岐で `"Conv -> PEQ"` と表示
- **検証結果**: ✅ 確認済み。行660-685の `updateGainStagingDisplay()` で、両方 true のケースが else に落ちて `"Conv -> PEQ"` を表示する
- **影響**: ユーザーがバイパス状態を誤認
- **修正方針**:
  ```cpp
  if (convBypassed && eqBypassed)
  {
      modeText = "Bypass";
      inputMaxDb = 0.0f;
  }
  else if (convBypassed && !eqBypassed)
  // ... (既存の順序を維持)
  ```
- **難易度**: Low（条件分岐の順序変更のみ）
- **影響範囲**: DeviceSettings のみ
- **テスト**: バイパス状態の組み合わせテスト

### U2: `MessageBoxA` 日本語文字化け

- **ファイル**: `src/CpuFeatureCheck.cpp:84-91`
- **問題**: `MessageBoxA` に UTF-8 narrow string を渡す。Windows の ANSI コードページが UTF-8 でない場合、日本語が文字化け
- **検証結果**: ✅ 確認済み。`::MessageBoxA(nullptr, "ConvoPeq には AVX2...", ...)` で UTF-8 文字列使用
- **影響**: エラーメッセージが読めない
- **修正方針**:
  ```cpp
  ::MessageBoxW(nullptr,
      L"ConvoPeq には AVX2 および FMA 命令に対応した CPU が必要です。\n"
      L"Intel Haswell (2013) 以降、または AMD Excavator (2015) 以降の\n"
      L"CPU が必要です。\n\n"
      L"この CPU ではアプリケーションがクラッシュする可能性があるため、\n"
      L"実行を中断します。",
      L"ConvoPeq - CPU 非対応",
      MB_OK | MB_ICONERROR);
  ```
- **難易度**: Low（文字列リテラル変更のみ）
- **影響範囲**: CpuFeatureCheck のみ
- **テスト**: 日本語ロケール環境での表示確認

### D1: `doubleArrayToString()` 16桁精度不足（IEEE 754要は17桁）

- **ファイル**: `src/DeviceSettings.cpp:812`
- **問題**: `juce::String(arr[i], 16)` は16桁。double 完全往復には17桁必要
- **検証結果**: ✅ 確認済み（実コード、文献とも）。IEEE 754 binary64 の 53-bit significand は 15-17桁の精度。往復保証には 17桁必要
- **影響**: ノイズシェイパー係数が保存/復元で微妙に変わる。学習結果の再現性低下
- **修正方針**: `juce::String(arr[i], 17)` に変更
- **難易度**: Low（1箇所の数値変更）
- **影響範囲**: DeviceSettings のみ（既存セーブファイルとの後方互換性あり。桁増は切り捨てではないため）
- **テスト**: 保存→復元の往復テスト。既存ファイルの読み込み互換性確認

### B5: テスト `/STACK:8388608` と `/GS-` が MSVC 限定

- **ファイル**: `CMakeLists.txt:374-379`
- **問題**: `if(MSVC)` 内に限定。icx でテスト実行時にスタックオーバーフローの可能性
- **検証結果**: ✅ 確認済み。`/GS-` と `/STACK:8388608` が `if(MSVC)` ブロック内にある
- **影響**: icx ビルドでテストがクラッシュする
- **修正方針**:
  ```cmake
  # 変更前
  if(MSVC)
      target_compile_options(BuildInputSemanticContractTests PRIVATE /GS-)
      target_link_options(BuildInputSemanticContractTests PRIVATE "/STACK:8388608")
  endif()
  # 変更後: MSVC と IntelLLVM の両方で適用
  if(MSVC OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      target_compile_options(BuildInputSemanticContractTests PRIVATE /GS-)
      target_link_options(BuildInputSemanticContractTests PRIVATE "/STACK:8388608")
  endif()
  ```
  **注意**: `if(MSVC)` を削除して無条件にするのは危険（clang-cl 互換でもないコンパイラで /GS- が未定義になる可能性）。条件を拡張するのが安全
- **難易度**: Low（条件分岐の拡張）
- **影響範囲**: テストターゲットのみ
- **テスト**: icx ビルドでのテスト実行

### R1: `shutdown_trace.json` が最初から `verified:true`

- **ファイル**: `src/audioengine/ISREvidenceExporter.cpp:276`
- **問題**: ハードコードされた文字列に `"verified":true` と `"sh1_callbackCount":0` 等
- **検証結果**: ✅ 確認済み。行276の shutdown_trace.json 文字列に `"verified":true` がハードコード
- **影響**: 実計測前のテンプレート出力。障害調査で誤った判断
- **修正方針**: 実測値を埋めるか、未計測なら `"verified":false`, `"status": "template"` にする
- **難易度**: Medium（計測ロジックの追加が必要な場合あり）
- **影響範囲**: ISREvidenceExporter のみ

### R2: `retire_latency_report.json` が最初から `withinThreshold:true`

- **ファイル**: `src/audioengine/ISREvidenceExporter.cpp:277`
- **問題**: `"withinThreshold":true` がハードコード
- **検証結果**: ✅ 確認済み。行277の retire_latency_report.json 文字列に `"withinThreshold":true` がハードコード
- **影響**: retire latency が閾値超過していても隠れる
- **修正方針**: R1 と同様に実測値を埋める
- **難易度**: Medium
- **影響範囲**: ISREvidenceExporter のみ

---

## P2: UX 向上 / 防御的コーディング（8件）

### H8: `delayWritePos` データレース（規約依存）

- **ファイル**: `src/ConvolverProcessor.h:885`, `src/convolver/ConvolverProcessor.Runtime.cpp:180`
- **問題**: `delayWritePos` は `int`（非 atomic）。`reset()` は Audio Thread 停止後にのみ呼び出す規約に依存。JUCE の実装によっては `prepareToPlay()` がオーディオスレッド近傍で呼ばれ data race の可能性
- **検証結果**: ✅ 確認済み。フィールドは `int delayWritePos = 0`（非 atomic）
- **修正方針**:
  - デバッグビルドで `jassert(isAudioThreadStopped())` を追加
  - または `reset()` 内で `std::atomic` フラグを検証
- **難易度**: Low〜Medium
- **影響範囲**: ConvolverProcessor のみ

### U3: `getSettingsFile()` createDirectory 失敗無視

- **ファイル**: `src/DeviceSettings.cpp:790-791`
- **問題**: `appDataDir.createDirectory()` の戻り値を確認しない（同一パターンが CacheManager.cpp:108, 421, MixedPhasePersistentCache.cpp:51, 158, NoiseShaperLearner.cpp:1459 にも存在）
- **検証結果**: ✅ 確認済み。呼び出し元はすべて Message Thread（初期化時 / Timer / save/load 関数）。Audio Thread からの到達は確認されず
- **修正方針**:
  ```cpp
  if (!appDataDir.exists())
  {
      auto result = appDataDir.createDirectory();
      if (!result.wasOk())
      {
          // 呼び出し元が Message Thread であることを確認済み。Logger は安全
          juce::Logger::writeToLog("Warning: Could not create settings directory");
      }
  }
  ```
  **重要**: Logger 追加前に呼び出し元が Message Thread であることを確認済み。Audio Thread から到達不可能（全5箇所の呼び出し元を確認）
- **難易度**: Low（5箇所の修正）
- **影響範囲**: DeviceSettings / CacheManager / MixedPhasePersistentCache / NoiseShaperLearner

### U4: `doubleArrayToString()` nullptr 防御

- **ファイル**: `src/DeviceSettings.cpp:808-813`
- **問題**: `arr` の nullptr チェックなし
- **検証結果**: ✅ 確認済み。`const double* arr` が nullptr の場合に UB
- **修正方針**:
  ```cpp
  juce::String doubleArrayToString(const double* arr, int size)
  {
      if (arr == nullptr || size <= 0)
          return {};
      // ... (既存コード)
  }
  ```
- **難易度**: Low（2行追加）
- **影響範囲**: DeviceSettings のみ

### D2: `/fp:fast` 全 Release 適用（msvc/icx 両方）

- **ファイル**: `CMakeLists.txt:902(msvc), 976(icx)`
- **問題**: DSP コアでも `/fp:fast` が適用される
- **検証結果**: ✅ 確認済み。MSVC Release（`/fp:fast`）と icx Release（`/fp:fast`）両方
- **重要**: icx 2026.0 では `/fp:precise + /Qimf-arch-consistency:true` が LLVM ERROR: out of memory を引き起こすため使用不可（CMakeLists.txt:964-965 に注釈済み）
- **修正方針**:
  - 短期: MSVC のみ、DSP コアファイルに `#pragma float_control(precise, on)` を個別追加
  - icx: `#pragma float_control` は icx でも使用可能。ただし `#pragma STDC FENV_ACCESS ON` の併用が必要な場合あり
  - 長期: `/fp:fast` を削除し、個別ファイルの最適化に切り替え
- **難易度**: **High**（AVX/MKL/SIMD/denormal/fast-math 依存を全て確認 + Audio 品質比較が必要）
- **影響範囲**: ビルド設定 + DSP コアファイル
- **注意**: icx で `/fp:precise` をグローバルに設定しないこと（OOM）。プラグマ単位で適用する
- **工数修正**: 0.5-1日（1-2h ではなく、影響評価・品質比較を含む）

### D5: `getTailLengthSeconds()` tail 強度未反映

- **ファイル**: `src/audioengine/AudioEngineProcessor.cpp:22-33`
- **問題**: IR長のみで tail length を返す。oversampling やフィルターによるテール延長を反映していない
- **検証結果**: ✅ 確認済み。`getProperty("irLength")` からの値のみ。oversampling factor（2x程度）やフィルター群の影響は加味されていないが、oversampling による実質的な影響は small（2x 未満）である点に注意
- **修正方針**:
  - 短期: コメントで「IR長のみの概算値」であることを明記
  - 長期: oversampling factor とフィルター特性を反映した有効テール長を計算
- **優先度メモ**: JUCE の `getTailLengthSeconds()` は概算で使われることが多いため、優先度はかなり低い。IR長ベースでも実用上十分妥当
- **難易度**: Medium〜High（DSP チェーンの影響評価が必要）
- **影響範囲**: AudioEngineProcessor のみ

### B4: `add_dependencies` 逆

- **ファイル**: `CMakeLists.txt:1135`
- **問題**: アプリ本体がテストに依存
- **検証結果**: ✅ 確認済み。`add_dependencies(ConvoPeq GainStagingContractTests EQProcessorMaxGainTests)` でアプリ←テストの逆依存
- **修正方針**: 依存を削除する、またはテスト集約ターゲットが存在する場合はそのターゲットに付け替える。`AllTests` ターゲットの存在を前提としない
- **難易度**: Low（1行変更）
- **影響範囲**: ビルド順序のみ

### B6: `/wd4100` `/wd4189` 広範囲

- **ファイル**: `CMakeLists.txt:888-889`
- **問題**: MSVC ターゲット全般（`ConvoPeq` ターゲット）に適用。自前コードの警告も隠す可能性
- **検証結果**: ✅ 確認済み。`ConvoPeq` ターゲットにグローバルに `/wd4100`（未参照パラメータ）と `/wd4189`（未参照ローカル変数）を指定
- **修正方針**:
  - JUCE/r8brain のみ SYSTEM include にし、自前コードは警告を有効にする
  - または `target_compile_options` をターゲット別に分離（JUCE ライブラリターゲットと自前コードターゲットを分割）
- **難易度**: Medium（CMakeLists.txt の構造変更が必要）
- **影響範囲**: ビルド設定

### C5: `getTailLengthSeconds()` ValueTree スレッド安全性

- **ファイル**: `src/audioengine/AudioEngineProcessor.cpp:22-33`
- **問題**: `getConvolverStateTree()` が ValueTree を返すが、スレッド安全性が不明
- **検証結果**: ✅ 確認済み。ValueTree はスレッドセーフでない
- **修正方針**:
  - **推奨**: `cachedTailLength` を Runtime Publish 時に計算・更新し、`getTailLengthSeconds()` はそれを返すだけにする
  - IR変更後 `prepare()` が呼ばれないケースでも更新されるよう、**Runtime Publish 時を Authority にする**（ISR 設計に一致）
  - ValueTree への依存を完全に断つ
  - `cachedTailLength` は `double`（非 atomic）で十分（現在の ConvoPeq の Runtime Publish シーケンスでは同一スレッドで実行されるため。ただし将来の変更に備え、コメントで「現在の ConvoPeq 実装の Runtime Publish シーケンスを前提とする」を添える）
- **難易度**: Low〜Medium（prepare 時の計算ロジック追加）
- **影響範囲**: AudioEngineProcessor のみ

### C3: `scheduleDebounce()` 規約依存（設計上の制約）

- **ファイル**: `src/EQEditProcessor.h:33`, `src/EQEditProcessor.cpp:24`
- **問題**: 「全てのセッターは Message Thread からのみ呼ぶこと」との規約に依存。規約違反が起きると即バグ
- **検証結果**: ✅ 確認済み。`scheduleDebounce()` は Message Thread 呼び出しを前提としているが、アサートによる検証がない
- **修正方針**:
  - デバッグビルドで `jassert(juce::MessageManager::getInstance()->isThisTheMessageThread())` を追加
  - または `scheduleDebounce()` 内でスレッド ID を検証
- **難易度**: Low（1行追加）
- **影響範囲**: EQEditProcessor のみ

---

## P3: 将来的な risk 低減（3件）

### B2: icx Debug/Release runtime 一貫性

- **ファイル**: `CMakeLists.txt:976-977, 1004-1005`
- **問題**: icx Release は `$<$<CONFIG:Release>:/MT>` で明示。icx Debug は明示せずデフォルト依存
- **検証結果**: ✅ 確認済み。icx のデフォルトは `/MT` で一貫しているが、明示性の問題
- **影響**: 実害は限定的。明示性改善のみ
- **修正方針**: icx Debug にも明示的に `/MT` を追加
- **難易度**: Low（CMakeLists.txt に1行追加）
- **影響範囲**: ビルド設定のみ

### B3: PGO が icx で無効化（警告なし）

- **ファイル**: `CMakeLists.txt:825-846`
- **問題**: PGO フラグが `$<$<CXX_COMPILER_ID:MSVC>:...>` で MSVC 限定。icx ユーザーに警告がないまま PGO されない
- **検証結果**: ✅ 確認済み。icx には `/Qprof-gen` / `/Qprof-use` が存在するが未実装
- **影響**: 性能改善の機会損失（バグではない）
- **修正方針**: `message(WARNING ...)` で icx ユーザーに警告
- **難易度**: Low（2行追加）
- **影響範囲**: ビルド設定のみ

### C4: `rcuProvider` 参照寿命（設計上の注意）

- **ファイル**: `src/ConvolverProcessor.h:1073`, `src/convolver/ConvolverProcessor.Lifecycle.cpp:114`
- **問題**: `std::optional<std::reference_wrapper<AudioEngine>> rcuProvider` は所有権を持たない
- **検証結果**: ✅ 確認済み。Lifecycle.cpp:114 に既存コメントあり。ISR Runtime 設計上、AudioEngine→ConvolverProcessor のライフタイムは設計保証
- **修正方針**:
  - **推奨**: コメント補強のみ。`weak_ptr` 化は不要
  - `weak_ptr` 化は shared_ptr 導入・所有権変更・refcount 追加を伴い、Authority Single Source 思想に反する
  - `AudioEngine` が `ConvolverProcessor` より必ず長生きする設計が前提であることをコメントで明文化
- **難易度**: Low（コメント追加のみ）
- **影響範囲**: ConvolverProcessor のみ

---

## 実施スケジュール案（v6）

### フェーズ1（P0）: 速やかに

| # | 項目 | 修正内容 | 工数目安 |
|---|------|----------|----------|
| H12 | noexcept + bad_alloc | `noexcept` 削除（呼び出し元 try-catch 確認済み） | 0.5h |
| R3 | JSON エスケープ | `escapeJson()` 自作（制御文字エスケープ含む10行程度） | 1-2h |

### フェーズ2（P1）: 次回リリースまでに

| # | 項目 | 修正内容 | 工数目安 |
|---|------|----------|----------|
| U1 | バイパス表示 | 条件分岐順序変更 | 0.5h |
| U2 | MessageBoxA | MessageBoxW + L リテラル | 0.5h |
| D1 | double精度 | 16→17桁 | 0.5h |
| B5 | テストスタック | `if(MSVC OR IntelLLVM)` に条件拡張 | 0.5h |
| R1/R2 | ISR evidence | 実測値 or テンプレート明示 | 2-4h |

### フェーズ3（P2）: 余裕があるとき

| # | 項目 | 修正内容 | 工数目安 |
|---|------|----------|----------|
| H8 | delayWritePos | jassert スレッド検証 | 0.5h |
| U3 | createDirectory | 戻り値チェック + Logger（呼び出し元確認済み） | 0.5h |
| U4 | nullptr防御 | ガード追加 | 0.5h |
| C3 | debounce | jassert スレッド検証 | 0.5h |
| C5 | tail length | `cachedTailLength` 方式（ValueTree 依存断ち） | 1-2h |
| B4 | add_dependencies | 依存削除または集約ターゲットへ移動 | 0.5h |
| B6 | 警告抑制 | SYSTEM include 分離 | 1-2h |
| D5 | tail length | 概算コメント or 計算 | 1-2h |
| **D2** | **/fp:fast** | **個別ファイル指定 + AVX/MKL/denormal 影響評価 + Audio 品質比較** | **0.5-1日** |

### フェーズ4（P3）: 長期的改善

| # | 項目 | 修正内容 | 工数目安 |
|---|------|----------|----------|
| B2 | icx runtime | `/MT` 明示化（明示性改善のみ） | 0.5h |
| B3 | PGO icx | 警告メッセージ追加（性能改善のみ） | 0.5h |
| C4 | rcuProvider | コメント補強のみ（`weak_ptr` 化は不要） | 0.5h |

---

## 工数サマリー（v6）

> **注**: 対象は 25件（有効21件＋設計上4件）だが、B2/B3 は P3扱い、H8/C3 は同一対策（jassert スレッド検証）として工数集計しているため、工数サマリーでは 20件相当としている。

| フェーズ | 件数 | 工数目安 |
|----------|------|----------|
| P0（クラッシュ/安全） | 2件 | 1.5-2.5h |
| P1（機能/ビルド） | 6件 | 4-6h |
| P2（UX/防御） | 9件 | 6-10.5h（D2 は 0.5-1日。B6はCMake構造次第で半日かかる可能性あり） |
| P3（長期） | 3件 | 1.5h |
| **合計** | **20件**（H8=C3重複除く） | **13-19h** |

---

## 優先的に対応すべき「安全な修正」（リスク最小、v6）

以下の修正は変更範囲が最小限で、リグレッション risk がほぼゼロ：

1. **H12**: `noexcept` を外す（1行。呼び出し元 try-catch 確認済み）
2. **U1**: 条件分岐の順序変更（5行）
3. **U2**: `MessageBoxA` → `MessageBoxW` + `L` リテラル（文字列変更のみ）
4. **D1**: `16` → `17`（1箇所の数値変更）
5. **U3/U4**: ガード追加（2-3行。呼び出し元スレッド確認済み）
6. **H8/C3**: デバッグビルドでの `jassert` 追加（1行）
7. **B3**: `message(WARNING ...)` 追加（2行）
8. **B4**: 不要な依存を削除（またはテスト集約ターゲットへ移動）（1行）
9. **C4**: コメント追加（1行）

上記 9件は合計工数 2-3h で完了可能。

---

## 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|----------|
| 2026-07-23 | v1 | 初版作成 |
| 2026-07-23 | v2 | レビュー反映: H12呼び出し元確認, R3 escapeJson, B5条件拡張, B2/B3→P3, D2工数修正, C5 cachedTailLength, C4コメント補強, U3スレッド確認 |
| 2026-07-23 | v3 | レビュー反映: R3制御文字エスケープ仕様修正, H12テスト方法Windows対応, B4保守的表現, C5 Runtime Publish時Authority, P2工数見直し |
| 2026-07-23 | v4 | 文書整合性修正: P1件数6件, P3件数3件, D2をP2に統一, B4表現統一, サマリー表修正 |
| 2026-07-23 | v5 | B2/B3をP3章へ完全移動, 25件と20件の関係注記, H12テスト方法詳細化, R3 0x00-0x1F明記, C5 JUCE規約コメント追加 |
| 2026-07-23 | v6 | 最終レビュー反映: escapeJson()工数20-40行に修正, H12テスト手法優先順位整理, C5 ConvoPeq実装前提に表現修正, 件数表現「25件(実装工数換算20件相当)」統一 |
