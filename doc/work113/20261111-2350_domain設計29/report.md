# domain tag 設計 v2.9 — 15-c の条件行列による実測と根拠の整理

Owner レビュー（4 回目）の指摘 1 点への対応（v2.8）に、証跡の補強を加えた版である。設計（B-1）は変えていない。**製品コードは未変更**で、実装は承認済みだが 15-c の方法確認を待って着手する。

## §0 結論

- v2.7 は `jassertfalse` を無条件の `__debugbreak()` として写しており、JUCE の実定義（`JUCE_LOG_CURRENT_ASSERTION` の後、`juce_isRunningUnderDebugger()` が true のときだけブレーク）と異なっていた。v2.8 で **JUCE 8.0.12 の実物ヘッダを include する**形に直し、v2.9 で**条件行列**を実測した。
- 行列は 5 条件（Debug／Release × デバッガ有無 × ガード強制）。**同じ Debug バイナリ系列でガードだけ true にするとブレークする**ため、ブレークを決めるのはビルド構成ではなく JUCE のガードである。Release はデバッガを接続してもブレークしない。5 条件すべて PASS。
- 15-c は **15-c-1（デバッガ非接続: ログのみ）** と **15-c-2（デバッガ接続: ブレークし、デバッガが初回例外をアプリへ渡せば捕捉できる）** に分けた。
- デバッガ設定のうち、`sxd 80000003` の意味は Microsoft Learn の `sx` コマンド資料（**Second chance break (Disabled)**）で確認した。ただし本環境には cdb／WinDbg が無いため、**実測は同条件を再現する最小デバッガで行い、WinDbg／Visual Studio の設定は文献ベース（未実測）**と明記した。

## §1 JUCE の実定義（リポジトリ内 JUCE 8.0.12）

| 場所 | 内容 |
|---|---|
| `JUCE/modules/juce_core/system/juce_PlatformDefs.h:167`（Debug） | `#define jassertfalse JUCE_BLOCK_WITH_FORCED_SEMICOLON (JUCE_LOG_CURRENT_ASSERTION; if (juce::juce_isRunningUnderDebugger()) JUCE_BREAK_IN_DEBUGGER; JUCE_ANALYZER_NORETURN)` |
| 同 `:193`（非 Debug） | `#define jassertfalse JUCE_BLOCK_WITH_FORCED_SEMICOLON (JUCE_LOG_CURRENT_ASSERTION;)` |
| `JUCE/modules/juce_core/native/juce_Threads_windows.cpp:262-265` | `juce_isRunningUnderDebugger() { return IsDebuggerPresent() != FALSE; }` |

## §2 条件行列（v2.9 の実測）

| # | ビルド | デバッガ | `juce_isRunningUnderDebugger()` | ブレーク | ログ | 結果 |
|---|---|---|---|---|---|---|
| A | Debug | なし | false | しない | 出る | PASS |
| B | Debug | あり（最小デバッガ） | true | する（`EXCEPTION_BREAKPOINT` 0x80000003 を `__except` が捕捉） | 出る（ブレーク前） | PASS |
| C | Debug | なし（`DOMAIN_PROBE_FORCE_DEBUGGER=1` でガードだけ true） | true | する | 出る | PASS |
| D | Release | なし | false | しない | 出ない | PASS |
| E | Release | あり | true（ただし `JUCE_DEBUG=0` でブレーク分岐が無い） | しない | 出ない | PASS |

- **A と C の差が本質**である。ビルド構成は同じ Debug で、違うのはガードの戻り値だけ。したがって「Debug なら無条件にブレークする」という v2.7 の前提は成り立たない。
- B と E は同じ最小デバッガから起動している。B はブレークし、E はしない（非 Debug の `jassertfalse` にブレークが無いため）。
- 全条件で `invalidDomainAssignments()` は +1 で、加算は assertion（ブレーク）より前である。

## §3 プローブとデバッガ

`evidence/assert_probe.cpp` は `JUCE/modules/juce_core/system/{juce_TargetPlatform.h,juce_PlatformDefs.h}` をそのまま include し、製品と同じ `jassertfalse` を使う。プローブが自前で与えるのは JUCE 本体をリンクしないための 2 関数だけである。

| 関数 | プローブでの実装 | JUCE との関係 |
|---|---|---|
| `juce::juce_isRunningUnderDebugger()` | `IsDebuggerPresent() != FALSE`（`DOMAIN_PROBE_FORCE_DEBUGGER` のときだけ true） | JUCE の Windows 実装と同一。強制スイッチは C 条件専用 |
| `juce::logAssertion(file, line)` | 呼ばれた事実・回数・行を記録 | 呼ばれる点は同じで、出力先だけ差し替え |

`evidence/mini_debugger.cpp` は 15-c-2 用の最小デバッガで、`DEBUG_ONLY_THIS_PROCESS` で子を起動する（`IsDebuggerPresent()` が true になる）。ローダの初期ブレークは `DBG_CONTINUE`、2 回目以降の `EXCEPTION_BREAKPOINT` は `DBG_EXCEPTION_NOT_HANDLED` でアプリの `__except` へ渡す。`DBG_CONTINUE` のままでは「デバッガが処理した」扱いになり、アプリの SEH へ回らない（最初はこれで失敗した）。

## §4 15-c の合格条件（設計差分 §7.2）

| 副検証 | 合格条件 |
|---|---|
| 15-c-1 | Debug・デバッガ非接続で、`jassertfalse` はログのみ。ブレークせず例外も出ない。`Unknown`、カウンタ +1 |
| 15-c-2 | Debug・デバッガ接続で、`jassertfalse` が `__debugbreak()` を実行し `EXCEPTION_BREAKPOINT` を発生。デバッガが初回例外をアプリへ渡す設定で `__except` が捕捉でき、捕捉後のカウンタが +1 |

デバッガ側の設定は、WinDbg／cdb なら `sxd 80000003` のうえ `g`、Visual Studio なら例外設定で `0x80000003` の中断を外し「Just My Code」を無効にする。`sxd` の意味は Microsoft Learn の `sx` コマンド資料で確認した（Second chance break (Disabled) ＝ 初回例外はデバッガが止めずアプリへ渡る）。この環境には cdb／WinDbg が無いため、実測は最小デバッガで行っている。

## §5 成果物

| 項目 | 内容 |
|---|---|
| 設計差分 | `doc/work113/20261111-1512_domain設計/design_diff.md`（v2.9・53,979 B） |
| 取得経路 | `doc/work113/20261111-1512_domain設計/MANIFEST.md` |
| 証跡 | 本報告の `evidence/`（プローブ・最小デバッガ・5 条件のビルド実行ログ・サイズ一致の確認） |
| コード変更 | なし（`src` に `RetireDomainId` / `sanitizeDomain` は 0 件のまま） |
