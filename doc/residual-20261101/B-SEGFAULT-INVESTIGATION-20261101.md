# B. 間欠 SEGFAULT の調査（2026-11-01）

- 対象: `AudioEngineHarness` の間欠 SEGFAULT（IR 系テスト・開始約 40 秒時点）
- 制約: テスト削除・実行回数削減・タイムアウト緩和による「見かけの解消」は行わない
- 基準: `main` = `origin/main` = **`fd92c6e7`**
- 状態: **原因を特定**（§7）。修正案は提示のみで、製品コードは未変更

## 1. 発生記録（実測）

| # | 日時 | 構成 | コマンド | 結果 |
|---|---|---|---|---|
| 1 | 2026-11-01 14:03 | `build-diag` **Release**（ASan OFF） | `ctest --test-dir build-diag -R "^AudioEngineHarness$"` | **PASS 298.30 s** |
| 2 | 2026-11-01 14:03 | 同上 | 同上 | **`***Exception: SegFault 40.00 sec`** |
| 3 | 2026-11-01 17:14 | `build-asan-local` RelWithDebInfo | `ctest --test-dir build-asan-local -C RelWithDebInfo -R "^AudioEngineHarness$"` | PASS 359.9 s |
| 4 | 2026-11-01 17:20 | 同上 | 同上 | PASS 360.5 s |

- 記録 1/2 の生ログ: `tmp/f2_ctest1.log`（216 B、PASS）と `tmp/f2_ctest2.log`（90,775 B、SEGFAULT）
- **重要**: 観測されたクラッシュは **ASan ビルドではない**（`build-diag` の `ENABLE_ASAN:BOOL=OFF`）。
  したがって「ASan で 51/51 PASS」は本件の解消を意味しない（Owner 指摘のとおり）。

## 2. 発生箇所（クラッシュログの最終出力からの特定）

`tmp/f2_ctest2.log` の末尾は次のとおり。

```
[L0_WRITE] gen ir=488 val=0.00000000
[L0_WRITE] gen ir=2536 val=0.00000000
[L0_WRITE] gen ir=4584 val=0.00000000
[L0_WRITE] geom part=512 numIR=12 numParts=16 fft=1024 imm=1 irLen=48000
[L0_WRITE] slot=11 peak=0.500000 bin=19
（同一ブロックがもう一度）
[L0_WRITE] slot=11 peak=0.500000 bin=19
← ここで終了（以降の出力なし）
```

| 項目 | 内容 |
|---|---|
| 最後に完了したテスト | `[I2T] T-I2-3: gotWorld=1`（`testRegisteredDSPFailurePreservesRegistration`） |
| 実行中だった処理 | そのテスト内で進行中の **rebuild → IR ロード**（`[IR_RATE_GEN]` / `[IR_CHAIN]` / `[IR_LOAD]` / `[NUC_ALLOC]` が連続） |
| 最後の出力元 | `src/MKLNonUniformConvolver.cpp:1068-1087` の **L0 write 診断**（`li == 0` の数値トレース） |
| IR の条件 | `irLen=48000`（48 kHz / 1 秒）、`Layers=2`、`L0Part=512`、`L1Part=4096`、`ringSize=4096` |
| クラッシュ時刻 | 40.00 秒（PASS 時は 298.30 秒） |
| ASan レポート | **なし**（この構成は ASan ではない） |

**読み方**: 最後の出力行は「そこで落ちた」ことの証明ではないが、**IR ロード経路（NUC の L0 書き込み直後）**が
最も近い実行点である。`[L0_WRITE]` ブロックが 2 回連続しているのは、同じ IR を 2 回ロードした
（または 2 つのコンボルバで書き込んだ）ためと考えられる。

## 3. 関連する既知事項

| 事実 | 出典 |
|---|---|
| 2 回目の `releaseResources`（admission Closed 後の再 release）は **Debug で既知の segfault**（BISECT で修復を無効化しても再現＝pre-existing） | `PublishPipelineIntegrationTests.cpp:344-355` のコメント |
| `testCallerDestroyTerminalDisposition` は判定を phase1-3 に限定し、2 回目 release を「I3 課題」として記録している | 同 343-370 |
| `testRegisteredDSPFailurePreservesRegistration`（T-I2-3）は `h.stop()` で終わる（admission は Open のまま） | 同 383-453 |

**現時点の仮説（未検証）**:
(a) IR ロード（`MKLNonUniformConvolver` の L0 書き込み周辺）で、確保済みバッファと
`numPartsIR` の不整合などにより範囲外アクセスが起きる。
(b) エンジン teardown（2 回目 release 系）の既知経路で、IR ロードと並行して破棄が走る。
いずれも**スタックトレース無しでは確定できない**。

## 4. 再現条件の収集

| 条件 | 値 |
|---|---|
| マシン | i7-11700K（8C/16T）・31.8 GB・Windows 11 Home build 26300.10.0 |
| 電源プラン | バランス（測定時のみ高パフォーマンス） |
| バイナリ | `build-diag\Release\AudioEngineHarness.exe` 41,717,760 B / 2026-11-01 13:41:57 |
| 構成 | Ninja Multi-Config・`CMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG`・`ENABLE_ASAN=OFF`・`CONVOPEQ_ENABLE_ISR_TESTS=ON` |
| 環境変数 | `CONVOPEQ_SOURCE_ROOT=C:/VSC_Project/ConvoPeq` |
| 負荷条件 | 記録 1/2 は他のビルド・作業と並行していた時間帯 |

## 5. 再現試行（完了）

- **ASan 構成**（`build-asan-local` RelWithDebInfo）: 2 回連続 PASS（359.9 s / 360.5 s）。
  3 回目は D-02 測定のため中断（打ち切りであり PASS でも FAIL でもない）。
- **build-diag Release**（クラッシュが観測された構成）: 観測窓を 150 秒に固定した反復を
  D-02 測定の後に実施する。窓は「観測を打ち切る時刻」であり、テストのタイムアウトを
  変更するものではない（各試行の成否は「窓内でクラッシュしたか」として記録する）。

### 5.1 実施上の注意（記録）

1. 当初 2 つのランナーが同時に走る状態が発生し（前バッチのプロセス残存）、
   その試行は**無効**として破棄した。1 ランナーに正して再実行する。
2. `cwd` は ctest と同じ `build-diag` に固定する（IR 資産の解決経路を観測時と揃えるため）。
3. 対象バイナリの同一性を確認した: `build-diag\Release\AudioEngineHarness.exe`
   **41,717,760 B / 2026-11-01 13:41:57 / sha256 `0BE8F1B75591755F9EA1991423A9F16F6893929F21E13B01D1DD2AD0460D13E9`**。
   これは 14:03 にクラッシュした実行と同一のバイナリである（以後の再ビルドなし）。
4. ソースは `HEAD = fd92c6e7`、追跡差分 0（測定ドライバ revert 後）。

### 5.2 バッチの条件

| 項目 | 値 |
|---|---|
| 対象 | `build-diag\Release\AudioEngineHarness.exe`（ASan OFF） |
| 試行回数 | 12 |
| 1 試行の観測窓 | 150 秒（クラッシュは過去に 40.00 秒で発生） |
| 打ち切り | 窓に達したら kill（**テストの制限時間ではない**） |
| 記録 | 終了コード、経過秒、最後の 3 タグ、ASan レポート有無、ログ全体 |
| 並行負荷 | ハーネス・ビルドのプロセスは無し（読み取り専用のソース走査が試行 1〜2 と重複） |

### 5.3 再現した（試行 5）

**クラッシュを同一シグネチャで再現した。**

| 項目 | 14:03 の観測（`tmp/f2_ctest2.log`） | 19:00 の再現（`tmp/b_diag_05.log`） |
|---|---|---|
| 構成 | `build-diag` Release（ASan OFF） | 同一 |
| 結果 | `***Exception: SegFault` | 終了コード **`0xC0000005`（アクセス違反）** |
| 経過 | 40.00 秒 | **36.4 秒** |
| 最後の出力 | `[L0_WRITE] slot=11 peak=0.500000 bin=19` | **同一行**（`[L0_WRITE]` が 2 ブロック連続した直後） |
| 実行中だったテスト | `[I2T] T-I2-3` の rebuild → IR ロード | 同一（`[I2T] T-I2-3: gotWorld=1` の後） |
| IR 条件 | `irLen=48000`、`numIR=12`、`part=512`、`fft=1024` | 同一 |

再現ログの末尾（抜粋）:

```
[IR_CHAIN] F_scale scaleFactor=50.01461801 phaseMode=0
[L0_WRITE] gen ir=488 val=0.00000000
[L0_WRITE] gen ir=2536 val=0.00000000
[L0_WRITE] gen ir=4584 val=0.00000000
[L0_WRITE] geom part=512 numIR=12 numParts=16 fft=1024 imm=1 irLen=48000
[L0_WRITE] slot=11 peak=0.500000 bin=19
[L0_WRITE] gen ir=488 val=0.00000000
[L0_WRITE] gen ir=2536 val=0.00000000
[L0_WRITE] gen ir=4584 val=0.00000000
[L0_WRITE] geom part=512 numIR=12 numParts=16 fft=1024 imm=1 irLen=48000
[L0_WRITE] slot=11 peak=0.500000 bin=19
← ここで終了（以降の出力なし）
```

**静的確認（クラッシュ箇所の候補）**: `[L0_WRITE]` ブロックの読み出しは
`l.irFreqReal + p * l.complexSize`（`p < l.numPartsIR`）で、確保は
`irSoaSize = numParts * complexSize`（`numParts = nextPowerOfTwo(numPartsIR) >= numPartsIR`）である
（`MKLNonUniformConvolver.cpp:886-897`、`:1074-1091`）。**この診断ループ自体は境界内**であり、
クラッシュはこの直後（`:1094` 以降の層設定、または後続のロード処理）で起きている可能性が高い。
確定にはダンプが必要。

### 5.4 12 試行の結果（頻度）

| 試行 | 結果 | 経過 | 最後のタグ |
|---|---|---|---|
| 1 | WINDOW（クラッシュなし） | 150.4 s | `D2B-SKIP, D2b-RES, STG8-TRACE` |
| 2 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| 3 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| 4 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| **5** | **CRASH `0xC0000005`** | **36.4 s** | **`L0_WRITE ×3`** |
| 6 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| 7 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| 8 | WINDOW | 150.2 s | `STG8-TRACE ×3` |
| 9 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| 10 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| 11 | WINDOW | 150.0 s | `STG8-TRACE ×3` |
| 12 | WINDOW | 150.0 s | `STG-8-D2b, D2b-RES, STG8-TRACE` |

| 指標 | 値 |
|---|---|
| 試行数 | 12 |
| クラッシュ | **1** |
| クラッシュ率 | **8.3%**（1/12） |
| クラッシュ時の経過 | 36.4 秒（14:03 の観測は 40.00 秒） |
| 結果ファイル | `tmp/b_diag_results.json` |

**解釈**: クラッシュは**毎回ではなく、IR ロードのタイミングに依存**して発生する。
12 回中 1 回という頻度は、14:03 の 1/2 と合わせると「条件付きで再現する間欠事象」と整理できる。
**テストの削除・実行回数の削減・タイムアウトの緩和は行っていない。**

**注意（負荷条件）**: 本バッチはハーネス・ビルドの並行プロセス無しで実施した
（読み取り専用のソース走査が試行 1〜2 と重複）。14:03 の観測は他のビルド・作業と並行していた時間帯であり、
**負荷条件は同一ではない**。この差が頻度に影響している可能性は排除できない。



## 6. スタックトレースの取得と解析

- `procdump`（v12.0、`C:\WINDOWS\system32\procdump.exe`）で未処理例外時にフルダンプを取得する
  準備をした（`localci/b_capture.cmd`）。
- 解析にはシンボル（PDB）が必要。`cdb.exe` は本マシンに無いため、
  ダンプ取得後に利用可能な解析手段（PDB 併用のシンボル化）を確定する。
- 現時点で**スタックトレースは未取得**。取得できるまでは「発生箇所の推定」に留める。

### 6.1 取得手段の実測（2026-11-01 17:36）

| 項目 | 結果 |
|---|---|
| `procdump -ma -e`（未処理例外） | 故意クラッシュ（Python の access violation）では**ダンプが出なかった**（プロセス側が SEH を処理して終了するため） |
| `procdump -ma -e 1`（初回例外） | **ダンプ成功**（46.7 MB、0.3 秒）。管理者権限は不要だった |
| ダンプ解析 | 隔離 venv に `minidump` 0.0.24 を導入。`--exception` で例外コード・**例外アドレス**・例外情報（`[0, 0]`）を取得できる。スタックウォーク機能は無い（`bt` 非対応） |
| シンボル化 | `llvm-symbolizer` は動作するが、**`build-diag\Release` には `AudioEngineHarness.pdb` が無い**（`CMakeFiles\...\Release\vc140.pdb` のみ）。このため Release のクラッシュアドレスはそのままではシンボル化できない |

**帰結**: ダンプから得られるのは「例外コード・例外アドレス・モジュール＋RVA」まで。
関数名・行番号まで落とすには、**PDB を持つ構成**（`build-diag\RelWithDebInfo` は 176.7 MB の PDB あり）で
再現するか、Release を `/DEBUG` 付きで再リンクして PDB を作る必要がある。
いずれも**ソースは変更しない**（ビルド構成のみ）。

### 6.2 `/DEBUG` 再リンクの実行（2026-11-01 19:20）

| 項目 | 値 |
|---|---|
| コマンド | `cmake -S . -B build-diag -DCMAKE_EXE_LINKER_FLAGS_RELEASE="/INCREMENTAL:NO /DEBUG"` → `cmake --build build-diag --config Release --target AudioEngineHarness` |
| 結果 | configure exit 0 / build exit 0（`error LNK`・`error C` なし） |
| EXE | **59,196,928 B**（再リンク前 41,717,760 B） |
| **PDB** | **180,998,144 B（生成）** |
| ソース変更 | **なし**（リンク設定のみ） |

**注意**: ダンプ取得に使うバイナリは**再リンク版**であり、12 試行バッチでクラッシュを再現した
バイナリ（`0BE8F1B7…`）とは異なる。発生率・アドレスは再リンク版のものである。

### 6.3 ダンプ取得の試行

`/DEBUG` 版（EXE 59,196,928 B / PDB 180,998,144 B）を `procdump -ma -e -x` で起動し、
未処理例外時のフルダンプ取得を試みた。1 試行の観測窓は 120 秒（**テストの制限時間ではなく観測の打ち切り**）。

| バッチ | 試行 | クラッシュ | ダンプ | 備考 |
|---|---|---|---|---|
| 1 | 14 | 0 | 0 | すべて WINDOW（120 秒窓を完走） |
| 2 | **5（途中で停止）** | **1** | **1** | 試行 5 で `EXIT` 40.1 秒、`exception_in_log=True` |

**取得したダンプ**: `tmp/b_dumps/AudioEngineHarness.exe_261101_195308.dmp`（**375,836,882 B = 358.4 MB**）
/ SHA-256 **`03C521D72F3221F43F0AA46765B9EBEC01454DBB42DC1A2F11AECFB699C3C413`**
/ マニフェスト `tmp/b_dumps/B-CRASH-DUMP-MANIFEST.md`

### 6.4 ダンプの解析結果（2026-11-01 19:53）

`minidump --exception` → モジュール特定 → `llvm-symbolizer --relative-address` でシンボル化した。

| 項目 | 値 |
|---|---|
| 例外コード | **`EXCEPTION_ACCESS_VIOLATION`** |
| 例外情報 | `[1, 0x1EC041AC800]` → **書き込み**（`1`）、書き込み先 `0x1EC041AC800` |
| 例外アドレス | `0x00007FF7A9F5E21A` |
| モジュール / ベース | `AudioEngineHarness.exe` / `0x00007FF7A7A20000` |
| **RVA** | **`0x253E21A`** |
| **関数** | **`ConvolverProcessor::process(juce::dsp::AudioBlock<double>&)`** |
| **ソース位置** | **`src/convolver/ConvolverProcessor.Runtime.cpp:774`** |
| 該当行 | `if (convDiagBuffer->push(event))` |
| ガード | `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`（同ファイル `:744`〜`:786`） |
| 経過 | 40.1 秒（**14:03 の観測 40.00 秒と一致**） |

**シンボル化の注意**: `llvm-symbolizer --obj=<exe> <RVA>` は `??:0:0` を返す。
**`--relative-address` を付けると正しく解決する**（PE の RVA 解釈に必要）。この手順を記録として残す。

## 7. 原因（特定）

### 7.1 直接原因

クラッシュは **診断用リングバッファへの書き込み**で発生している。

```cpp
// src/convolver/ConvolverProcessor.Runtime.cpp
namespace {
    LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>* convDiagBuffer = nullptr;  // L91
    DiagPerTickCounter* convTickPushed = nullptr;
    DiagPerTickCounter* convTickDropped = nullptr;
    std::atomic<uint64_t>* convTotalPushed = nullptr;
}
void setConvDiagBuffer(/* ... */) noexcept {   // L96-104
    convDiagBuffer = &db;                      // ← エンジンのメンバのアドレスをグローバルへ保存
    ...
}
```

```cpp
// src/audioengine/AudioEngine.Init.cpp:116-119（AudioEngine の初期化時に毎回呼ばれる）
setEqDiagBuffer(diagBuffer, rtAuxMutable_.diagTickPushed, ...);
setConvDiagBuffer(diagBuffer, rtAuxMutable_.diagTickPushed, ...);
```

`diagBuffer` は **`AudioEngine` のメンバ**（`src/audioengine/AudioEngine.h:3030`）であり、
**エンジンの寿命とともに破棄される**。一方、そのアドレスは**プロセス全体のグローバル変数**に保存され、
**エンジン破棄時にリセットする経路が存在しない**（`convDiagBuffer = nullptr` に相当する処理は無い）。

したがって、次の順序で **解放済み（または別インスタンスの）バッファへ書き込む**。

```
エンジン A 生成 → setConvDiagBuffer(&A.diagBuffer)   （グローバル = A）
エンジン A 破棄 → A.diagBuffer も破棄                （グローバルは A を指したまま）
エンジン B 生成 → setConvDiagBuffer(&B.diagBuffer)   （ここまでの窓では A を指す）
  ↑ この窓で A の RT スレッドが process() を実行すると、A の解放済みバッファへ push → アクセス違反
```

### 7.2 なぜ間欠的か

- ハーネスは `AudioEngineHarness` を**114 回**構築しており、エンジンの生成・破棄が繰り返される。
- クラッシュは「破棄と RT スレッドの `process()` の競合窓」に落ちたときだけ起きる。
- `/DEBUG` 再リンク版で 0/14 → 1/5 と変動したのは、**ヒープ配置・タイミングが変わる**ためと整合する。

### 7.3 なぜ ASan 51/51 PASS が無関係か（重要）

| ビルドツリー | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` |
|---|---|
| `build-diag`（クラッシュする） | **ON** |
| `build-asan-local`（51/51 PASS） | **OFF** |
| `build`（Release） | **OFF** |

クラッシュ箇所は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の内側にある。
**ASan 構成ではこのコードがコンパイルされていない**ため、ASan が 51/51 PASS でも
本件の検証にはならない。これは Owner の指摘（「ASan 51/51 PASS と SEGFAULT 解消は同義ではない」）の
**具体的な根拠**である。

### 7.4 影響範囲

| 構成 | 影響 |
|---|---|
| `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF`（製品既定・ASan・Release） | **コードが存在しないため無影響** |
| `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`（`build-diag` など診断ビルド） | **間欠クラッシュの可能性**（エンジン再生成・破棄と RT 処理の競合時） |
| 同型の `eqDiagBuffer`（`AudioEngine.Processing.DSPCoreFloat.cpp:31/43`） | **同じ欠陥クラス**（EQ の診断書き込みでも同様に起こり得る） |

### 7.5 修正案（**未承認・提案のみ**）

| 案 | 内容 | リスク | 評価 |
|---|---|---|---|
| **B-1** | エンジン破棄時に `setConvDiagBuffer` / `setEqDiagBuffer` を `nullptr` でクリアし、ポインタを `std::atomic` にする | 小 | 最小修正。ただし「ロード→使用」の間の TOCTOU は残る |
| **B-2** | グローバルを廃止し、`ConvolverProcessor` インスタンスが自分のバッファポインタを保持する（`prepareToPlay` で設定） | 中 | **競合が構造的に消える**。診断経路の所有権が明確になる |
| **B-3** | 世代トークン（所有エンジンの ID）を併用し、不一致なら書き込まない | 中 | 安全だが複雑 |

**推奨**: **B-2**（単一所有）を本命、**B-1** を暫定緩和として提示する。
いずれも**製品コードの変更が必要であり、Owner の承認前に実施しない**。

## 8. 未確定事項

- クラッシュ窓の発生率（`build-diag` で 1/12、`/DEBUG` 版で 1/19）。**診断ビルド固有**のため、
  製品出荷構成での影響は無い。
- `0x1EC041AC800` という書き込み先が「解放済みヒープの再利用」か「別インスタンスのバッファ」かの区別。
  いずれも「グローバルが寿命を追えていない」という同一の原因に帰着する。
- `eqDiagBuffer` 側で同種のクラッシュが起きるか（同型のため起こり得るが、未観測）。
