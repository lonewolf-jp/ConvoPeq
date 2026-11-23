# CI 失敗 3 件の修正（domain tag commit `173e6995` の後始末）

作業日時: 2026-11-12 08:12〜09:2x
作業場所: 独立 worktree `C:\VSC_Project\ConvoPeq-ci-fix`（branch `ci-fix/173e6995`）
基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（parent `c94d3767`、tree `a1ec450a3ee95b7ba607e170b55ee64da8b726d2`）

## 結論

指摘された 3 件は修正できた。静的検査 3 種類はいずれも 0 件／PASS、proof-scope 検証器の自己テストは 20 ケースすべて期待どおり、Release と ASan の ctest は 57/57 合格、Debug は既知の 2 件（`D8_2_B_2_Tests` のスタック超過、`TailLayerIntegrityTests` の RES-F 予算）だけが従来どおり落ちる。ISR standard tier も完走した。

指摘の 2 点（`reset()` の排他性の証明と、暗黙の既定オーダーを含む比較、proof-scope の A(D1) 検査）にも対応した。commit と push はしていない。差分は `ci-fix.patch`（12 ファイル、+1,705 / −324。新規の自己テスト 2 本を含む）として保存した。

## 作業環境の記録（指示 2）

| 項目 | 値 |
| --- | --- |
| 基準 HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| worktree | `C:\VSC_Project\ConvoPeq-ci-fix`（`git worktree add -b ci-fix/173e6995`、作成時クリーン） |
| 作業前の main 作業ツリー | HEAD `173e6995`、変更 34 件、未追跡 163 件、ステージ 0 件 |
| 作業前の worktree | 変更 0 件、未追跡 0 件（`git status --porcelain -uall` が空） |
| 対象ファイル | 下記 12 件（`.github/scripts` 3、`src/audioengine` 5、`src/core` 4） |
| 保全 | main 作業ツリーへの stage／stash／reset／clean／削除／復元は未実施。`AGENTS.md`、`CMakeLists.txt`、除外 32 ファイル、B-01〜B-13 の未コミット作業は worktree に持ち込んでいない |

## 修正対象ファイル（12 件）

| ファイル | 変更 |
| --- | --- |
| `.github/scripts/isr-verify-proof-scope.ps1` | 関数単位の契約検証へ書き換え（+219 行相当） |
| `.github/scripts/test-isr-verify-proof-scope.ps1` | 新規。検証器の自己テスト 20 ケース |
| `.github/scripts/test-isr-verify-proof-scope-lexer.ps1` | 新規。字句走査の境界テスト 38 項目（範囲検証と変異対照 3 件を含む） |
| `src/audioengine/AtomicAccess.h` | `compareExchangeWeakAtomic` 追加、CAS ヘルパーの desired を変換可能型へ |
| `src/audioengine/ISRDebugRuntime.h` | `mutable` 除去（LINT-AE-011） |
| `src/audioengine/ISRDebugRuntime.cpp` | atomic dot-call 2 件、include 追加 |
| `src/audioengine/ISRRetireLatency.h` | atomic dot-call 218 件、`thread_local` の NOLINT、include 追加 |
| `src/audioengine/WorldLifetimeRecorder.h` | atomic dot-call 30 件、include 追加 |
| `src/core/ReaderMeasureRegistry.h` | atomic dot-call 9 件、include 追加 |
| `src/core/ScopedEpochReader.h` | atomic dot-call 9 件、include 追加 |
| `src/core/EpochDomain.h` / `SnapshotCoordinator.h` | atomic dot-call 各 1 件（include は EpochDomain に既存） |

## LINT-AE-011 の修正（指示 3）

### `mutable HBVerifierRuntime hbVerifier_;`

参照は `ISRDebugRuntime.cpp` の 4 箇所だけで、どちらも非 const メンバが行う。

- `emitFinalViolationReport()`（非 const）: `setTraceRuntime` と `emitViolationReport`
- `emitCIArtifacts()`（非 const）: `setTraceRuntime` と `emitViolationReport`

const メンバからの参照も、const オブジェクト経由のアクセスも無い。`mutable` が必要な設計理由は無かったので除去した。const 性は変更していない。

### `thread_local std::uint32_t assigned`

呼び出し経路は `recordRequestRT()`（RT 経路）と `claim()` から `rangeForCurrentThread()`。初回アクセスが RT スレッドで起こり得る点は事実。RT 安全性は次のとおり確認した。

1. 型は `std::uint32_t` で、初期化子は定数式。TLS ブロックは `.tdata` に置かれ、MSVC の動的初期化用ガードを作らない。
2. MSVC 14.52（/O2 /std:c++20）で同じ構文をコンパイルし、`dumpbin /symbols` で `__Init_thread_header`／`__Init_thread_epoch`／guard 系のシンボルが 0 件、`.tls$` セクションが 4 バイト（uint32 そのもの）、未定義シンボルは `_tls_index` のみであることを確認した（`tls_symbols.txt`、`tls_probe.cpp`）。
3. 初回アクセス時の処理は、番兵値の比較と `convo::fetchAddAtomic(nextRange_, 1u, relaxed)` ＋ 剰余だけ。ロック・動的確保・非決定的処理は無い。
4. 破棄は trivial（整数）。
5. `kRanges = 8`、`nextRange_` の fetch_add と剰余という既存の割当契約は変えていない。

以上から、既存 lint 規則が認める `NOLINT(thread-local)` と `RT-SAFE:` の併記を同一行に置き、根拠をコメントに書いた（`rangeForCurrentThread()`）。方式の変更（非 RT 側での事前登録など）は行っていない。

## atomic 直接 dot-call の修正（指示 4）

### 置換件数（操作種別）

CI の検出は「行」単位で 258 行。実際に置換した呼び出しは 270 件（1 行に複数の呼び出しがある行が 12 行あったため）。

| 元の操作 | 置換先 | 件数 |
| --- | --- | --- |
| `.load(order)` | `convo::consumeAtomic` | 124 |
| `.store(v, order)` | `convo::publishAtomic` | 73 |
| `.store(v)`（既定 seq_cst） | `convo::publishAtomic`（release） | 23 |
| `.fetch_add(v, order)` | `convo::fetchAddAtomic` | 34 |
| `.compare_exchange_strong(e, d, s, f)` | `convo::compareExchangeAtomic` | 11 |
| `.compare_exchange_weak(e, d, s, f)` | `convo::compareExchangeWeakAtomic` | 5 |
| 合計 | | **270** |

ファイル別では `ISRRetireLatency.h` 218、`WorldLifetimeRecorder.h` 30、`ReaderMeasureRegistry.h` 9、`ScopedEpochReader.h` 9、`ISRDebugRuntime.cpp` 2、`EpochDomain.h` 1、`SnapshotCoordinator.h` 1。

`memory_order_relaxed` を同じ行に持つ 104 件は、検査スクリプトの既存ポリシー（relaxed は診断カウンタ用として除外）に従い、そのまま残した。置換対象は CI が指摘した 258 行に限っている。

### 追加したヘルパーと仕様変更

- `convo::compareExchangeWeakAtomic` を追加した。`std::atomic_compare_exchange_weak_explicit` をそのまま使い、spurious failure と失敗時の `expected` 更新を維持する。strong 版への置き換えはしていない。
- 既存の `compareExchangeAtomic` を `template <typename T, typename U>` に広げ、`desired` を `U&&`（`is_convertible_v<U, T>`）で受けるようにした。`std::atomic<std::uint8_t>` に対して `1` のような `int` リテラルを渡す呼び出しが、メンバ関数では暗黙変換されていたため。`expected` は従来どおり `T&` で、失敗時の書き戻しは変わらない。

### メモリオーダーの検証（指示 4.2・再検証版）

置換は「呼び出しの形だけを変え、引数リストはそのまま渡す」方式にした（`X.op (args)` → `convo::helper (X, args)`）。

前回の検証は、明示された `memory_order_*` トークンだけを比較していた。そのため `.store(0)`（既定 seq_cst）が release になっても差分として出ない欠陥があった。今回は次のとおり作り直した（`atomic_order_effective_diff.py`）。

- 各呼び出しを「操作種別＋success order＋failure order」に正規化する。
- 既定値も解決する。メンバ関数は `load`／`store`／`exchange`／`fetch_*` の省略時 seq_cst、CAS の省略時 seq_cst/seq_cst。ヘルパーは `consumeAtomic`=acquire、`publishAtomic`=release、`exchangeAtomic`／`fetch*`=acq_rel、CAS=acq_rel/acquire。
- HEAD と作業ツリーの呼び出し列を突き合わせ、差分を明示的に列挙する。

結果は 7 ファイル・435 操作で、順序の変更は **23 件**。すべて `ISRRetireLatency.h` の `reset()` 内（HEAD の 803〜892 行）にある `store` の seq_cst → release で、これ以外の変更は 0 件だった（`atomic_order_effective_diff.txt`）。23 件が `reset()` の範囲に収まることは機械的に確認している。

weak/strong の別、success/failure の 2 オーダー、`expected` の参照渡し、戻り値、CAS ループの終了条件は変えていない。固定容量スロットの `state`／`object` の公開順序（release store の後に acquire load、CAS で所有権取得）も同じ並びである。

### include と依存方向（指示 4.3）

`AtomicAccess.h` は 6 ファイルで明示的に include した。`src/audioengine` 配下の 3 ファイルは同一ディレクトリの `"AtomicAccess.h"`、`src/core` 配下の 3 ファイルは `"audioengine/AtomicAccess.h"` を使う。core から audioengine のヘルパーへ依存する形は既に `EpochDomain.h`・`RCUReader.h`・`RuntimeStore.h` などで使われている既存の方向で、循環はない。`EpochDomain.h` は JUCE 非依存を保ったまま（`AtomicAccess.h` は `<atomic>` などの標準ヘッダのみ）。

### 唯一の意味変更（seq_cst → release）とその根拠

`.store(v)`（既定 seq_cst）だった 23 件は `publishAtomic`（release）になった。すべて `RetireLatencyRecorder::reset()` の中にある。

`reset()` の呼び出しは全 10 箇所で、本番は `AudioEngine` のコンストラクタ 1 箇所、残りはテストの初期化。コンストラクタ実行時点では、レコーダへ書き込むスレッド（ワーカー、コーディネータ、タイマー、オーディオコールバック）がまだ 1 本も存在しない。テストは逐次実行で、`reset()` は各テストの先頭、スレッド生成はその後である。呼び出し箇所の一覧、ライフサイクル、スレッド生成順、将来の維持条件は `reset_exclusivity.md` にまとめた。

読み手は acquire で、同ファイルの他の更新は release/acq_rel であり、seq_cst 固有の全順序に依存する箇所は無い。プロジェクトの atomic ポリシー自体が `src/**` で seq_cst を禁じており（`check-src-atomic-dotcall.ps1` の forbidden-seq-cst）、この変更はポリシーに合わせる方向でもある。実行時の検出機構は無いため、新しい呼び出し箇所を足す場合は同じ静止状態の条件をレビューで確認する必要がある（`reset()` の直前に契約をコメントとして記載した）。

## ISR Verification（proof-scope）の修正（指示 5）

### 変更内容

旧スクリプトは Release 分岐を文字列で 2 つ要求していた（`if (mode == "Release") { return; }` と `if (mode == "CI") { emitHBTrace(); }`）。これを、関数本体を波括弧対応で切り出して制御フローを見る検査に置き換えた。追加した `-Root` パラメータで任意のツリーを検査できる（テスト用）。

検査項目:

1. `emitCIArtifacts()` に Release の早期 return があること。条件は `mode == "Release"` を含むこと。
2. 強制証跡モード（`isForceEvidenceEmission()`）が実装に存在する場合、Release の早期 return 条件がその否定を含むこと。含まない実装（強制条件を無視する形）は FAIL。
3. `emitCIArtifacts()` が `emitHBTrace()` を呼ぶこと。かつ、その呼び出しが Release 早期 return より後にあること（先に emit してから return する形は FAIL）。
4. `emitHBTrace()` 自身に Release の早期 return があること（force 実装では同じく否定条件を含むこと）。
5. `emitFinalViolationReport()` の**関数本体の中**で `emitViolationReport` を呼ぶこと（A(D1) の維持）。ソース全体で「宣言より後ろに呼び出しがあるか」を見る方式だと、別関数の同名呼び出しで PASS してしまうため、本体に限定した。
6. エクスポータ側の R25 契約（Release=off / Debug=partial / それ以外=full、`isRelease` の早期 return と `minimalEvidence`）は従来どおり。

無関係な `return` の存在だけでは通らない（条件式と位置を見る）。

字句の走査は、コメント・通常文字列・文字リテラル・raw 文字列を 1 回のパスで分類する方式に直した。

- 文字列の中の `//` や `/*` はコメント開始として扱わない（`"https://example.com"` の後ろが消えない）。
- コメントの中の引用符は文字列開始として扱わない。
- raw 文字列 `R"delim( ... )delim"` を 1 つのリテラルとして扱う。
- アポストロフィは、直前と直後の文字がともに 16 進数字で、かつ接頭辞（`L` / `u` / `U` / `u8`）が付いていない場合だけ数値の桁区切りとみなす。`return '}'` や `'0'` のような文字リテラルを桁区切りと誤認しない。
- エスケープは 1 文字のバックスラッシュ（`$backslash = [char]''`）で判定する。`"escaped quote: \" // still inside /* here */"` のように、エスケープされた引用符の後ろに `//` や `/*` があっても 1 つのリテラルとして扱う。
- 行継続は、バックスラッシュ + LF / CRLF / CR のいずれもリテラルの内側として読み飛ばす。CRLF で `\` + CR だけを飛ばして LF で打ち切る、という取りこぼしをしない。
- 条件式の判定はコメントだけを空白化したテキストで行い（`mode == "Release"` を比較するため）、A(D1) のトークン検査は文字列も空白化したテキストで行う。

### 自己テスト（指示 5.3）

`test-isr-verify-proof-scope.ps1` を新設し、検証器を子プロセスとして実際に走らせて判定した。結果は 20/20 期待どおり（`proof_scope_selftest.log`）。

| ケース | 期待 | 実測 |
| --- | --- | --- |
| 旧来の Release 抑制実装 | PASS | exit 0 |
| B-11 の Release 強制証跡対応実装 | PASS | exit 0 |
| Release で無条件に証跡を発行 | FAIL | exit 1 |
| Release の強制条件を無視 | FAIL | exit 1 |
| CI の証跡生成経路を削除 | FAIL | exit 1 |
| `emitHBTrace()` の Release 抑制を削除 | FAIL | exit 1 |
| `//` を含む通常文字列（後続の emit が消えない） | PASS | exit 0 |
| `/*` を含む通常文字列 | PASS | exit 0 |
| `//` と `/*` を含む raw 文字列 | PASS | exit 0 |
| `'}'` を含む文字リテラル | PASS | exit 0 |
| `50'000'000` の桁区切り | PASS | exit 0 |
| 接頭辞付き raw 文字列 `u8R"..."` | PASS | exit 0 |
| エスケープされた引用符＋`//`／`/*` を含む文字列 | PASS | exit 0 |
| CRLF の行継続文字列（内部に `//`） | PASS | exit 0 |
| 文字リテラル内のエスケープ引用符 `'''`（本体内の構造で検出） | PASS | exit 0 |
| Release の早期 return を文字列にだけ書いた実装 | FAIL | exit 1 |
| `emitViolationReport(` を文字列にだけ書いた実装 | FAIL | exit 1 |
| エクスポータ契約を壊した実装 | FAIL | exit 1 |
| A(D1) 呼び出しを本体から削除（後続の別関数には存在） | FAIL | exit 1 |
| 文字列リテラルに波括弧を含む本体 | PASS | exit 0 |

### 走査器の単体確認

検証器の字句走査を関数単位でも確認した。同じ内容を `.github/scripts/test-isr-verify-proof-scope-lexer.ps1` として commit 候補に含め、実行結果を `scanner_probe.txt` に保存した。38 項目すべて PASS。

- `"https://example.com"` の後ろの `emitHBTrace();` が残る（文字列内の `//` をコメントにしない）
- 本物の末尾 `//` コメントは空白化される
- `"/* not a comment */"` の後ろのコードが残る
- raw 文字列 `R"hb(/* ... */ // ...)hb"` の後ろのコードが残る
- 接頭辞付き raw 文字列 `u8R"hb(... )hb"` の後ろのコードが残る
- `'}'` が文字リテラルとして扱われ、波括弧がコードとして数えられない
- `50'000'000` が文字リテラルとして扱われない（コードが変化しない）
- 接頭辞付き `L'}'` が文字リテラルとして扱われる
- コメント内のアポストロフィ（`// don't`）が文字リテラルを開かない
- 文字列内のアポストロフィ（`"it's fine"`）が文字リテラルを開かない
- エスケープされた引用符の後ろの `//` と `/*` がリテラル内に留まり、リテラル範囲の開始・終了位置が閉じ引用符と一致する
- LF の行継続が 1 つのリテラル範囲になり、範囲が両方の行にまたがる
- CRLF の行継続も 1 つのリテラル範囲になり、範囲が `

` を含む
- 行継続したリテラル内の `//` は除去されず、その後ろの本物のコメントだけが除去される
- 文字リテラル `'''` が単一のリテラル範囲になり、開始位置が開き引用符、終了位置が 4 文字後（引用符・バックスラッシュ・引用符・引用符）と一致する
- 変異対照: 通常文字列のエスケープ判定を壊すとエスケープ引用符のケースが失敗し、CRLF の読み飛ばしを 1 文字減らすと CRLF ケースが失敗し、文字リテラル側のエスケープ判定だけを壊すと `'''` のケースが失敗する（境界テストが識別力を持つことの確認）

検証器と自己テストの差分は `proof_scope_diff.txt` と `proof_scope_selftest_diff.txt` に単体で保存した。

## 静的検査の結果（指示 6.1）

| 検査 | 結果 |
| --- | --- |
| `check-audioengine-lint.ps1` | PASS（LINT-AE-001〜014、011 の 2 件は 0） |
| `check-src-atomic-dotcall.ps1` | PASS（検出 0 件。修正前 258） |
| `isr-verify-proof-scope.ps1` | PASS |
| `check-list-compliance.ps1`（参考） | PASS（atomic スキャン含む） |
| `isr-run-tiered-verification.ps1 -Tier standard`（参考） | PASS（完走） |

## ビルド・テスト（指示 6.2）

ビルドは CI と同じ手順（Ninja Multi-Config、`cl`、`CONVOPEQ_REQUIRE_MKL=ON`、JuceHeader 生成のため ConvoPeq を先にビルド）で行った。

| 構成 | ビルド | ctest（今回の修正後に再実行） |
| --- | --- | --- |
| Release | 成功（223 ターゲットを再ビルド） | **57/57 PASS**（うち `AudioEngineHarness` 302 秒） |
| Debug | 成功（223 ターゲットを再ビルド） | 55/57 PASS。失敗は既知の 2 件のみ |
| RelWithDebInfo + ASan（`-DENABLE_ASAN=ON`） | 成功（175 ターゲットを再ビルド） | **57/57 PASS**（うち `AudioEngineHarness` 350 秒、`ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0`） |

レビュー指摘への対応後にすべて取り直した値である（初回の 57/57・55/57 は同じ結果だった）。

Debug の失敗は変更前と同じ 2 件で、今回の修正範囲外として触っていない。

- `D8_2_B_2_Tests`: `0xC00000FD`（スタック超過）。`EpochDomain` をスタックに置く既存構造。
- `TailLayerIntegrityTests`: `RES-F` の CPU・メモリ予算が Debug で超過。

変更前の値は、push 済み commit を Debug で回した実測（同じ 2 件、他は合格）を基準にした。今回の変更で新しい失敗は出ていない。

## CI の状態（指示 6.3）

commit と push が禁止のため、リモート CI の再実行はできていない。ローカルで workflow が呼ぶスクリプトをそのまま実行し、workflow と同じ構成のビルド・テストを回した（上記）。

失敗していた run（参考）:

| workflow | run | 結果（push 時） |
| --- | --- | --- |
| audioengine-lint | 43063319811 | failure（LINT-AE-011 ×2） |
| list-compliance | 43063320048 | failure（atomic dot-call 258） |
| ISR Verification（定期実行） | 43073663375 | failure（proof-scope） |

## 変更差分と証跡

`doc/work113/20261112-0922_CI修正/` に保存した。

- `ci-fix.patch`: commit 候補の全差分（12 ファイル、+1,705 / −324。新規ファイルは `test-isr-verify-proof-scope.ps1` と `test-isr-verify-proof-scope-lexer.ps1`）
- `diffstat.txt` / `diff_numstat.txt`
- `lint_audioengine.log`、`lint_atomic_dotcall.log`、`proof_scope.log`、`proof_scope_selftest.log`、`list_compliance.log`、`isr_tier_standard.log`
- `ctest_release_fast.log`、`ctest_release_aeh.log`、`ctest_debug_fast.log`、`ctest_debug_aeh.log`、`ctest_asan_all.log`
- `configure_release.log`、`configure_asan.log`、`build_tails.txt`
- `atomic_order_effective_diff.py`（比較スクリプト）、`atomic_order_effective_diff.txt`（435 操作・23 件の差分）
- `atomic_order_preservation.txt`（参考: 明示オーダーのみの旧比較。`atomic_order_effective_diff.txt` が正式）
- `reset_exclusivity.md`（`reset()` の呼び出し 10 箇所と排他性）
- `ConvoPeq.md`、`split/`（修正を含む結合版）
- `tls_symbols.txt`、`tls_probe.cpp`、`tls_symbols_full.txt`（TLS ガード無しの証拠）
- `scanner_probe.txt`（`.github/scripts/test-isr-verify-proof-scope-lexer.ps1` の実行結果 38 項目）
- `proof_scope_diff.txt`、`proof_scope_selftest_diff.txt`（検証器と自己テストの差分）
- `mutation_fixture_check.txt`（文字リテラル側のエスケープ判定だけを壊したとき、自己テストのケースが FAIL し、戻すと PASS する記録）

Dropbox の `/ConvoPeq-173e6995-commit/ci-fix-20261112/` にも同じ一式を置いた。

## 修正後のソース結合版（ConvoPeq.md）

指示 4 の指摘に合わせ、今回の修正を含む結合版を作り直した。生成は worktree `C:\VSC_Project\ConvoPeq-ci-fix` で `output_sourcecode_markdown.py` を実行し、`ConvoPeq.md`（7,035,321 B、生成 2026-11-12 10:01:45）と 2 MB 未満の分割 4 件（1,949,901 / 1,949,999 / 1,949,942 / 1,185,479 B、連結はバイト一致）を得た。生成後に worktree の `ConvoPeq.md` は `git checkout` で戻し、修正差分には含めていない。

公開場所は `doc/work113/20261112-0922_CI修正/ConvoPeq.md` と `split/`、Dropbox の `/ConvoPeq-173e6995-commit/ci-fix-20261112/`。生成対象は `src/` 配下のソースで、`.github/scripts` の検証器と自己テストは含まれない（`ci-fix.patch` で配布する）。

main 作業ツリーの `ConvoPeq.md` は、指示（main に触れない）に従って更新していない。承認後に commit と合わせて更新する。

なお、main の `ConvoPeq.md` は `c94d3767` を HEAD とする 2026-11-10 生成版で、`173e6995` も本修正も含まない。今回の判断が差分レビューに基づくのはそのためである。

## 未解決事項

- commit と push は未実施。push 後に remote CI 3 件を再実行して結果を確認する必要がある。
- `Sanitizer CI` のうち TSan ジョブ（Linux Clang）は**未検証**。ASan の成功を TSan の代替証拠にはしない。
- `TailLayerIntegrityTests` の Debug 予算超過と `D8_2_B_2_Tests` のスタック超過は既知のまま（今回の対象外）。
- main 作業ツリーの `AGENTS.md` は「未 push」の記述のまま（push 済みに直すのは別作業）。
- 独立 worktree `C:\VSC_Project\ConvoPeq-ci-fix` と branch `ci-fix/173e6995` は残してある（承認後に commit するため）。

## 指示 9 の遵守

commit、push、merge、release、tag 作成、main への直接変更、既存 dirty worktree の変更、対象外ファイルの変更はいずれも行っていない。main 作業ツリーには本報告フォルダを追加しただけで、既存の変更 34 件と未追跡 163 件には触れていない。追加した 34 ファイルのうち 13 件は `.gitignore` の `*.log` に該当するため、未追跡の計数は 163 から 184 に増えた（21 件分）。増分には結合版 `ConvoPeq.md` と分割 4 件、走査器の単体確認と差分ファイルを含む。
