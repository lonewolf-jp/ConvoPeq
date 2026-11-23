# domain tag 設計 v2.8 — 15-c を JUCE の実条件に合わせて修正

Owner レビュー（2026-11-11・4 回目）の指摘 1 点を反映した。設計（B-1）は変えていない。**製品コードは未変更**で、実装は承認済みだが 15-c の方法確認を待って着手する。

## §0 結論

- v2.7 のプローブは `jassertfalse` を**無条件の `__debugbreak()`** として写していた。JUCE 8.0.12 の実定義は `JUCE_LOG_CURRENT_ASSERTION` の後に **`juce_isRunningUnderDebugger()` が true のときだけ**ブレークする。条件が違うため、v2.7 の結果は 15-c の根拠にならない。
- v2.8 では **JUCE の実物ヘッダを include して実マクロを使い**、次の 3 条件で実測した。すべて PASS。
- 15-c を **15-c-1（デバッガ非接続: ログのみ・ブレークなし）** と **15-c-2（デバッガ接続: ブレークし、デバッガが初回例外をアプリへ渡せば捕捉できる）** に分けた。15-c-2 のデバッガ設定まで設計差分に書いた。
- v2.7 のプローブは「SEH で捕捉できること」の部分実証として残し、15-c の根拠には使わない。

## §1 JUCE の実定義（リポジトリ内 JUCE 8.0.12）

| 場所 | 内容 |
|---|---|
| `JUCE/modules/juce_core/system/juce_PlatformDefs.h:167` | `#define jassertfalse JUCE_BLOCK_WITH_FORCED_SEMICOLON (JUCE_LOG_CURRENT_ASSERTION; if (juce::juce_isRunningUnderDebugger()) JUCE_BREAK_IN_DEBUGGER; JUCE_ANALYZER_NORETURN)` |
| 同 `:193`（非 Debug） | `#define jassertfalse JUCE_BLOCK_WITH_FORCED_SEMICOLON (JUCE_LOG_CURRENT_ASSERTION;)` |
| `JUCE/modules/juce_core/native/juce_Threads_windows.cpp:262-265` | `juce_isRunningUnderDebugger() { return IsDebuggerPresent() != FALSE; }` |
| 同 `:81-86`（MSVC 系） | `#define JUCE_BREAK_IN_DEBUGGER { __debugbreak(); }` |

したがって Debug でも、**デバッガが接続されていなければブレークしない**。v2.7 のプローブはここが違っていた。

## §2 新しいプローブ（v2.8）

`evidence/assert_probe.cpp` は `JUCE/modules/juce_core/system/{juce_TargetPlatform.h,juce_PlatformDefs.h}` をそのまま include し、製品と同じ `jassertfalse` を使う。プローブが自前で与えているのは、JUCE 本体をリンクしないための 2 関数だけである。

| 関数 | プローブでの実装 | JUCE との関係 |
|---|---|---|
| `juce::juce_isRunningUnderDebugger()` | `IsDebuggerPresent() != FALSE` | JUCE の Windows 実装と同一 |
| `juce::logAssertion(file, line)` | 呼ばれた事実・回数・行を記録 | 記録するだけで、呼ばれる点は同じ |

15-c-2 用に `evidence/mini_debugger.cpp` を用意した。`DEBUG_ONLY_THIS_PROCESS` で子を起動するので `IsDebuggerPresent()` が true になり、ローダの初期ブレークは `DBG_CONTINUE`、2 回目以降の `EXCEPTION_BREAKPOINT` は `DBG_EXCEPTION_NOT_HANDLED` でアプリの `__except` へ渡す。`DBG_CONTINUE` のままでは「デバッガが処理した」扱いになり、アプリの SEH へ回らない（最初はこれで失敗した）。

### 2.1 実行結果（3 条件）

| 条件 | ビルド | 結果 |
|---|---|---|
| A: Debug・デバッガ非接続 | `-O0 -DJUCE_DEBUG=1`（警告なし） | ブレークせず。assertion はログ（count=1, `assert_probe.cpp:74`）。`domainId() == Unknown`、カウンタ +1。`PASS` |
| B: Debug・デバッガ接続 | 同上を `mini_debugger` から起動 | `EXCEPTION_BREAKPOINT`（0x80000003）を発生し、アプリの `__except` が捕捉。捕捉後のカウンタ +1（＝ブレークより前に加算）。assertion はログ済み。`PASS` |
| C: Release・デバッガ非接続 | `-O2 -DNDEBUG -DJUCE_DEBUG=0`（警告なし） | ブレークせず。assertion のログなし（`JUCE_LOG_ASSERTIONS` 無効）。`domainId() == Unknown`、カウンタ +1。`PASS` |

B でデバッガが見たブレークは 2 回（ローダ初期 1 回＋`jassertfalse` 1 回）。2 回目だけをアプリへ渡している。

## §3 設計差分に入れた 15-c の条件

| 副検証 | 合格条件 |
|---|---|
| 15-c-1 | Debug・デバッガ非接続で、`jassertfalse` はログのみ。ブレークせず、例外は出ない。`Unknown`、カウンタ +1 |
| 15-c-2 | Debug・デバッガ接続（`juce_isRunningUnderDebugger()` が true）で、`jassertfalse` が `__debugbreak()` を実行し `EXCEPTION_BREAKPOINT` を発生。デバッガが初回例外をアプリへ渡す設定で `__except` が捕捉でき、捕捉後のカウンタが +1 |

デバッガ側の設定は、WinDbg／cdb なら `sxd 80000003` のうえ `g`、Visual Studio なら「例外設定」で `0x80000003` の中断を外し「Just My Code」を無効にする。既定のままだとデバッガが先に止まり、`__except` へ回らない。

## §4 成果物

| 項目 | 内容 |
|---|---|
| 設計差分 | `doc/work113/20261111-1512_domain設計/design_diff.md`（v2.8・50,811 B） |
| 取得経路 | `doc/work113/20261111-1512_domain設計/MANIFEST.md` |
| 証跡 | 本報告の `evidence/`（JUCE 実ヘッダを使うプローブ・最小デバッガ・3 条件のビルド実行ログ・サイズ一致の確認） |
| コード変更 | なし（`src` に `RetireDomainId` / `sanitizeDomain` は 0 件のまま） |
