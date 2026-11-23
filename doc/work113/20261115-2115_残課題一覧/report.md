# 未修正バグ・残課題の全件報告（2026-11-15 21:15 時点）

- 判定基準リビジョン: **`296ecfebd7ac0a646a509cd0076ebbdd91310ff5`**（`main`、PR #216 のマージコミット）
- GitHub 状態: open PR **0 件**、open Issue **1 件**（#215）。main の CI は 6 件すべて success
- ローカル main 作業ツリー: `C:\VSC_Project\ConvoPeq`（`4d476d5c`、未コミット変更 42 ファイル / 未追跡 352 ファイル）
- この報告は「main に取り込まれていない修正」と「そもそも未修正の欠陥」を分けて示す

## 1. 結論

| 区分 | 件数 | 内容 |
| --- | ---: | --- |
| **未修正バグ（main）** | **2** | #215（MSVC の FMA 経路無効）、NoiseShaperLearner の OOM 失敗経路が通らない |
| 仕様の見える化漏れ（バグ扱いしない） | 1 | `tailL1L2Multiplier` の UI 表示と実効値の不一致 |
| **未検証・未計測** | **5** | BUG-029 の数値比較、NUPC 幾何、オーバーサンプリング、qwen 非バグの網羅検証、新規テストの ASan 対象外 |
| 衛生項目（非バグ・低優先度） | 4 | 冗長 zeroupper、atomic ドット呼び出し、TruePeak の防御ガード欠落、キャッシュ重複挿入の可能性 |
| **未マージの作業** | **1 束** | ローカル main 作業ツリーの WIP（GitHub 上の未マージ PR は 0） |
| メモリ記載の再確認で「解消済み」と判明 | 1 | ARCHITECTURE §9 の bypass crossfade 記述 |
| 要再確認（メモリ由来・未検証） | 5 | N1 テール保持、IRState C3、Portable Reader Safety、W1 違反 1 件、CR5 |

## 2. 未修正バグ（main に存在する欠陥）

### 2.1 #215 — MSVC `/arch:AVX2` で decimate の FMA SIMD 経路が無効

- 場所: `src/CustomInputOversampler.cpp:210`（`dotProductDecimateAvx2` の定義）と `:662`（`decimateStage` の convCount 4〜7 経路）。どちらも `#if defined(__AVX2__) && defined(__FMA__)`
- 事実: MSVC は `/arch:AVX2` で `__AVX2__` を定義するが `__FMA__` は定義しない。そのため **MSVC ビルドでは decimate の SIMD 経路が丸ごとコンパイルされず、スカラーループに落ちる**
- 実測（2026-11-15、`dumpbin /disasm`）: MSVC の obj は `decimateStage` の ymm 命令 **0 件**、`-D__FMA__` を強制すると `dotProductDecimateAvx2` 26 件 / `decimateStage` 3 件、icx 2026.2 では 91 件 / 12 件
- 影響: 音響出力の正しさではなく**性能**。MSVC と icx で実行経路が異なる
- 状態: Issue #215 として OPEN。修正未着手（Owner 判断で独立課題）
- 参考: `src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp:124` が「MSVC は `__FMA__` を持たない」ことを既に出力している

### 2.2 `NoiseShaperLearner` の OOM 失敗経路が通らない

- 場所: `src/NoiseShaperLearner.cpp:71-78`（コンストラクタ `NoiseShaperLearner::NoiseShaperLearner` 内）

```cpp
segmentBuffer = AudioSegmentBuffer::create();
if (!segmentBuffer)
{
    DBG_LOG("[NoiseShaperLearner] failed to allocate AudioSegmentBuffer");
    convo::publishAtomic(errorMessage, "Failed to allocate AudioSegmentBuffer", ...);
    convo::publishAtomic(progress.status, Status::Error, ...);
    return;
}
```

- 事実: `AudioSegmentBuffer::create()` は失敗時に `std::bad_alloc` を送出し、**nullptr を返さない**（`AlignedAllocation.h` の `aligned_malloc` が例外を投げる）。したがってこの `if (!segmentBuffer)` は到達せず、OOM 時は `errorMessage` / `Status::Error` に落ちずに例外がコンストラクタから伝播する
- 同じコンストラクタの直前（`:57-66`）は nothrow 版の `makeAlignedArray` を nullptr で判定しており、**同じ関数内で失敗時の意味論が 2 通り**になっている
- 影響: OOM 時に意図した「エラー状態へ遷移して復帰」ではなく例外伝播になる。呼出側が例外を捕捉していなければプロセス終了
- 状態: 未修正。B-1（PR #216）は RAII とリークの修正に限定し、この呼出側の不整合は範囲外とした
- 修正案: `create()` を `try { } catch (const std::bad_alloc&) { return nullptr; }` で包んで契約を nullptr に戻すか、呼出側を `try/catch` にして `errorMessage` へ落とす。どちらかに統一する

## 3. 仕様の見える化漏れ（バグ扱いしない）

### 3.1 `tailL1L2Multiplier` の UI 表示と実効値

- UI スライダーと保存値は 2〜16 の整数を許容するが、実効値は最も近い 2 の冪（2/4/8/16）へ丸められる（例: 6 を選ぶと実効 8）
- 丸め自体は BUG-013 の修正として正しい（L2 の FFT 長が 2 の冪でないと IPP のプランを作れない）
- 残る課題は**表示の正規化**のみ。スライダーを 2/4/8/16 に離散化して実効値と一致させる。PR #216 では 3 コミットに限定するため見送った
- 影響: 表示と実音の不一致（音は正しい）。音響出力の欠陥ではない

## 4. 未検証・未計測

| # | 項目 | 現状 | 必要な作業 |
| ---: | --- | --- | --- |
| 1 | BUG-029（icx `/fp:fast` の数値乖離） | 未計測。icx は `/fp:fast` を残し（`CMakeLists.txt:2196` 付近）、MSVC は `/fp:precise`。`#pragma float_control(precise, on)` は 2 ファイルのみ | icx と MSVC で同一入力の出力を比較し、差が許容範囲か測る |
| 2 | NUPC 幾何（BUG-010〜012, 014） | コード上のガードの存在確認まで。幾何の数値網羅検証は未実施 | パーティション長・層境界・I5 整除条件の全域検証 |
| 3 | オーバーサンプリング（BUG-025, 026） | 同上 | 比・段数・フィルタ長の組合せ検証 |
| 4 | qwen の非バグ 34 件 | 記載理由のスポット検証まで（前回レポート §3.6） | 幾何・数値系の網羅検証 |
| 5 | 新規 3 テストの ASan 対象 | `CONVOPEQ_ASAN_TEST_TARGETS` に未登録（`TailLayerMultiplierTests` / `AudioSegmentBufferTests` / `AllpassDesignerCmaesTests`） | リストへ追加し、ASan でリーク検査を効かせる |

## 5. 衛生項目（非バグ・低優先度）

| # | 項目 | 内容 | 影響 |
| ---: | --- | --- | --- |
| 1 | 冗長な `_mm256_zeroupper()` | `src/CustomInputOversampler.cpp:195`（`dotProductAvx2`）。MSVC 19.52 と icx 2026.2 は AVX/SSE 境界に `vzeroupper` を自動生成するため不要 | なし（1 uop の無駄のみ） |
| 2 | atomic のドット呼び出し | `src/audioengine/AudioEngine.h:884` の `liveCount.fetch_sub(...)`。`convo::fetchSubAtomic` と等価だが規約違反 | なし（診断マクロ内） |
| 3 | `TruePeakDetector::processBlock` の容量ガード欠落 | `numSamples` に対する検査がない。呼出側の `maxSamplesPerBlock` 上限と作業バッファ 8 倍の余裕で現行は到達不能 | なし（防御性の不足） |
| 4 | EQ キャッシュの重複挿入の可能性 | `AudioEngine.Cache.cpp` の lock-free 経路で hash が進むと locked 経路と系列がずれ、同一キーが 2 個できる可能性。係数は同一 | メモリのみ（理論上） |

## 6. 未マージの作業

### 6.1 GitHub 上

- open PR は **0 件**。未マージのブランチも無い（`fix/bugtriage-20261115` は PR #216 でマージ済み）
- open Issue は #215 のみ（#212 は本報告時に、PR #213 のマージで解消済みとしてクローズした）

### 6.2 ローカル main 作業ツリー（`4d476d5c`）の WIP — main には未反映

`C:\VSC_Project\ConvoPeq` に、**`main`（`296ecfeb`）へ取り込まれていない作業**が残っている。別タスクの進行中変更であり、私は内容の帰属を確認していない。

- 変更量: 追跡ファイル 42 件、+10,064 / −1,015。未追跡 **352 ファイル**（`git status` のエントリ数は 149。ディレクトリ単位に集約されるため。内訳は `doc/work113` 337、`src` 12、その他 2）
- 主な変更: `src/MainWindow.cpp`（252 行）、`src/NoiseShaperLearner.cpp`（224 行）、`src/convolver/ConvolverProcessor.LoaderThread.cpp`（182 行）、`src/IRConverter.cpp`（416 行）、`src/CacheManager.cpp`（268 行）、`.github/scripts/isr-run-runtime-evidence.ps1`（433 行）、`AGENTS.md`、`CMakeLists.txt`（+143 行）、`ConvoPeq.md`
- main に存在しない新規ファイル（未追跡）: `src/InputInspection.cpp/.h`、`src/learner/`、`src/convolver/ConvolverStopProbe.h`、`src/convolver/IRTrailingSilence.h`、`.github/scripts/isr-verify-evidence-consistency.ps1`、テスト 6 本（`CacheManagerHeaderValidationTests` / `InputInspectionTests` / `LearnedStateMigrationTests` / `LearnerStateAuthorityTests` / `NoiseShaperStateRaceTests` / `RuntimeParameterGuardTests`）、`resf_raw.json`
- **main との重複**: この WIP は `CMakeLists.txt`、`src/MKLNonUniformConvolver.cpp`、`src/tests/TailLayerIntegrityTests.cpp` の 3 ファイルで **PR #216 と同じファイルを触っている**。統合時に衝突解決が必要
- 注意: 基点が `4d476d5c` なので、統合には `296ecfeb` への rebase かマージが要る
- 補足: プロジェクト記録にある「BUG-2（起動 10 秒遅延、`computeMaxElementDepth` の O(n²)）」の修正は、この未追跡の `src/InputInspection.cpp` にあり、**main には入っていない**。main 側の `DeviceSettings::loadNoiseShaperState` は `computeMaxElementDepth` を持たないため、main が同じ 10 秒遅延を起こすとは限らない（未計測）

## 7. メモリ記載の再確認結果

| 記載 | 再確認結果 |
| --- | --- |
| ARCHITECTURE §9 の bypass crossfade 記述が実装と不整合 | **解消済み（不整合は再現せず）**。`ARCHITECTURE.md:826-828` の「`kBypassCrossfadeTimeSec`（20 ms）で独立にランプ」は、`src/ConvolverProcessor.h:226` の `kBypassCrossfadeTimeSec = 0.020` と `src/convolver/ConvolverProcessor.Runtime.cpp:433` の `activeMixSmoother.resetRT(sampleRate, kBypassCrossfadeTimeSec)` に一致する |
| PR #207 が OPEN | 誤り。PR #207 は 2026-11-14 にマージ済み（`6e099932`） |

## 8. 要再確認（メモリ由来・本報告では未検証）

以下は過去セッションの記録で、**現行 `main` に対して検証していない**。着手前に実装と突き合わせること。

| # | 項目 | 記録の内容 |
| ---: | --- | --- |
| 1 | N1 テール保持 | 未採用（保留）。再検討条件が 3 つあると記録 |
| 2 | IRState の C3 | 未達（deref 読み手 9 箇所に slot 0） |
| 3 | Portable Reader Safety | NOT PROVED（P1 は証明、P2 は未証明） |
| 4 | W1 違反 | `SnapshotCoordinator` に 1 件発見と記録 |
| 5 | CR5 修正 | 未 commit で維持。保証範囲 F/H が未検出と判明 |

## 9. 参考: 今回マージ済み（未修正バグではない）

PR #216（`296ecfeb`）で main に入った修正。

| ID | 内容 | 状態 |
| --- | --- | --- |
| BUG-013 | 非 2 冪 FFT 長の拒否（fail-closed）と L2 テール倍率の 2 冪制約 | 修正済み・マージ済み |
| B-1 | `AudioSegmentBuffer::create()` の例外安全化（OOM 時のリーク解消） | 修正済み・マージ済み |
| B-2 | 到達不能な CMA-ES 早期終了条件の削除（動作変更なし） | 修正済み・マージ済み |
| BUG-028 | decimate 経路の zeroupper 欠落 | **非バグと確定**（両コンパイラが自動生成） |
| B-3 / B-4 / B-5 / B-6 / B-7 / B-8 / B-9 / B-10 | ds の残り 8 件 | 非バグ（B-6 と B-10 は本報告 §5 の衛生項目） |
| BUG-018 / BUG-029 | qwen の 2 件 | 非バグ（BUG-018 は §5、BUG-029 は §4 の未検証） |
| qwen の残り 34 件 | — | 非バグ（スポット検証） |

## 10. 推奨する着手順

1. **§6.2 の WIP の統合**（`296ecfeb` への rebase）。3 ファイルの衝突解決が前提。ここが最大の未処理塊
2. **§2.2** の失敗経路統一（小さい修正、テスト追加が容易）
3. **#215** の条件修正（MSVC では `__AVX2__` のみで FMA 経路を有効とみなす）。逆アセンブルと性能差の測定まで含める
4. **§4 の 5 件**（検証系）。特に BUG-029 の icx/MSVC 比較は数値品質に直結する
5. **§5 の衛生項目** と **§3.1 の UI 正規化**
6. **§8 の再確認**（記録が古い可能性があるため、着手前に実装と突き合わせる）
