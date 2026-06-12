# コーディング規約 (Coding Standards)

## 1. 使用フレームワーク・ライブラリ
本プロジェクトでは、以下のライブラリの特定バージョンを使用します。

- **JUCE Framework V8.0.12**
    - [公式リポジトリ](https://github.com/juce-framework/JUCE/tree/8.0.12)
    - [APIドキュメント](https://docs.juce.com/master/index.html)
    - **注意事項**:
        - 必ずJUCE 8.0.12の公式安定版ドキュメント（上記APIドキュメントURL）で当該関数の存在を確認すること。
        - 戻り値の意味、副作用、スレッド安全性、前提条件を完全に理解した上で使用すること。
        - 公式サンプルはVST3等が主であり、本アプリ（スタンドアローン）とは構造が異なる点に注意。
        - 公式ドキュメントやJUCE 8.0.12サンプルで示されていない追加安定化（例: 多段フォールバック、重複リトライ層、独自ウォッチドッグ）を独自判断で導入しない。

- **r8brain-free-src**
    - [公式リポジトリ](https://github.com/avaneev/r8brain-free-src.git)

- **Intel oneAPI MKL (oneMKL)**
    - [製品ページ](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
    - [Windows用開発者ガイド](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2025-2/overview.html)

- **Windows SDK**
    - [技術ドキュメント](https://learn.microsoft.com/ja-jp/windows/apps/windows-sdk/)

## 2. 編集制限
プロジェクトルート内の以下2ディレクトリ**配下の全ファイル（すべての拡張子）**は、条件なしで編集禁止です。
- `/JUCE` フォルダ
- `/r8brain-free-src` フォルダ

## 3. 禁止事項
> 各カテゴリは独立したチェック項目として適用すること。実装前に該当カテゴリを個別に確認してください。

- **構造化例外処理 (SEH)**: 絶対に使用しないこと。
- **Audio Thread内でのブロッキング処理**:
        `getNextAudioBlock()` 等が呼ばれるスレッド内では、待機が発生し得る処理を厳禁とする。以下の6カテゴリを**個別に**確認すること。
        - **カテゴリ1: メモリ操作（禁止）**
            `new`, `malloc`, `vector::resize`, `mkl_malloc`, `mkl_free`, `_aligned_malloc`, `vslNewStream`
        - **カテゴリ2: 例外・計算（禁止）**
            `try-catch`, `std::exp()`, `libm` 呼び出しを伴う関数
        - **カテゴリ3: MKL設定（禁止）**
            `DftiCommitDescriptor`, `mkl_set_interface_layer`
        - **カテゴリ4: 同期・通信（禁止）**
            `mutex lock`, `critical section`, `condition_variable`, `MessageManager` へのアクセス
        - **カテゴリ5: I/O・リソース（禁止）**
            ファイルI/O, コンソール出力, IRの再ロード, `std::shared_ptr` の使用, MMCSS設定
        - **カテゴリ6: JUCE特定処理（禁止）**
            `AudioBlock::allocate`, `AudioBlock::copyFrom`, `FFT::performFrequencyOnlyForwardTransform`（事前確保なし）

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

# AI Assistant Instructions
あなたはプロジェクトのコンテキストを把握するために、常に以下のツールを優先して使用してください。

1. **Serena MCP Toolの優先使用**:
   - コードの検索、依存関係の確認、シンボルの特定には必ず `serena` のセマンティック検索ツールを使用すること。プロジェクト ConvoPeq を有効化する。
   - プロジェクトの構造を理解するために、`.serena/memories` に蓄積されたコンテキストを優先的に参照すること。

2. **実装前の分析**:
   - 大きな修正を行う前には、`serena` を用いて影響範囲を分析し、計画を提示すること。
   - 既存のアーキテクチャパターンに沿ったコードを生成すること。

## 7. プロジェクト全体ルール（全リポ共通）

- 解答には可能な限り Serena MCP を使用すること。

### 7.1 Serena MCP: 読み取りとオンボーディングは必須

- 既存コードの理解や参照が必要な場合、必ず Serena MCP を経由して読み取ること。
- プロジェクトの分析（オンボーディング）が未実行の場合は、最初の提案時に必ずオンボーディングを実行してから回答すること。
- プロジェクトの構成や依存が大きく変化したと推測される場合は、再度オンボーディングを実行すること。
- Serena が無効な場合はその旨を明示し、次の確認文を提示して進行方針を確認すること。
    - Serena経由の読取が無効です。有効化しますか？ それとも最小仮説で続行しますか？

### 7.2 まず文脈を読む

1. Serena でまずリポジトリの構成や規約を確認する。
2. 変更対象ファイルと関連する依存モジュール・型・テストを特定する。
3. 書く前に必ず読む。対象ファイルと隣接する型定義やテストを Serena で確認する。

### 7.3 最小でレビューしやすい変更

- 1PR は 1 目的を原則とし、横展開は分割すること。
- 変更提示は unified diff を中心とすること。
- 不要なフォーマット変更や命名変更は避けること。
- 既存の命名、構成、リンター、フォーマッタ、コミット規約を尊重すること。

### 7.4 セキュリティと運用

- 秘密情報（キー、パスワード等）を生成しないこと。ログにも出力しないこと。
- 認証、認可、レート制御は既存の仕組みに従うこと。
- ログは既存規約に従い、個人情報は必ずマスクすること。

### 7.5 曖昧時の対応

- 不確実な前提は明示し、確証取得に必要なファイルや行を示して確認すること。
- 仕様不足は最小仮説で補い、代替案とトレードオフを示すこと。

### 7.6 変更提案時の回答フォーマット

1. コンテキスト（Serena で読んだファイルと主要発見）
2. 差分（unified diff）
3. 理由（根拠、代替案、トレードオフ）
4. 検証方法（プロジェクト設定やスクリプトから自動検出したコマンド例）
5. フォローアップ（任意）

### 7.7 絶対ルール

- Serena 経由せずに既存コード理解を前提とした提案をしないこと。
- プロジェクト規約を自動検出し遵守すること。
- 不要な巻き込み変更は行わないこと。

<!-- rtk-instructions v2 -->
# RTK — Token-Optimized CLI

**rtk** is a CLI proxy that filters and compresses command outputs, saving 60-90% tokens.

## Rule

Always prefix shell commands with `rtk`:

```bash
# Instead of:              Use:
git status                 rtk git status
git log -10                rtk git log -10
cargo test                 rtk cargo test
docker ps                  rtk docker ps
kubectl get pods           rtk kubectl pods
```

## Meta commands (use directly)

```bash
rtk gain              # Token savings dashboard
rtk gain --history    # Per-command savings history
rtk discover          # Find missed rtk opportunities
rtk proxy <cmd>       # Run raw (no filtering) but track usage
```
<!-- /rtk-instructions -->

## graphify

For any question about this repo's architecture, structure, components, or how to add/modify/find
code, your first action should be `graphify query "<question>"` when `graphify-out/graph.json`
exists. Use `graphify path "<A>" "<B>"` for relationship questions and `graphify explain "<concept>"`
for focused-concept questions. These return a scoped subgraph, usually much smaller than the full
report or raw grep output.

Triggers: "how do I…", "where is…", "what does … do", "add/modify a <component>",
"explain the architecture", or anything that depends on how files or classes relate.

If `graphify-out/wiki/index.md` exists, use it for broad navigation. Read `graphify-out/GRAPH_REPORT.md`
only for broad architecture review or when query/path/explain do not surface enough context. Only read
source files when (a) modifying/debugging specific code, (b) the graph lacks the needed detail, or
(c) the graph is missing or stale.

Type `/graphify` in Copilot Chat to build or update the graph.

## AiDex — Persistent Code Index (MCP Server)

AiDex は Tree-sitter + SQLite によるコードインデックス MCP サーバー（v2.1.2）。
`.aidex/` が存在する場合、コード検索に Grep/Glob/Read より AiDex を優先すること。

### セッション開始時（必須）
```markdown
1. `aidex_session({ path: "." })` — 外部変更検出、自動再インデックス
2. セッションノートがあれば表示
3. セッション終了時: `aidex_note({ path: ".", note: "...", summary: "..." })`
```

### コード検索 — Grep/Glob の代わりに AiDex を使用
| 代わりに使うもの | AiDex の使用法 |
|---|---|
| `Grep pattern="functionName"` | `aidex_query({ path: ".", term: "functionName" })` |
| `Grep pattern="class.*Name"` | `aidex_query({ path: ".", term: "Name", mode: "contains" })` |
| ファイルを読んで構造把握 | `aidex_signature({ path: ".", file: "src/AudioEngine.h" })` |
| 複数ファイルの構造 | `aidex_signatures({ path: ".", pattern: "src/audioengine/**" })` |
| プロジェクト概要 | `aidex_summary({ path: "." })` + `aidex_tree({ path: "." })` |
| ファイル一覧 | `aidex_files({ path: ".", type: "code" })` |
| 最近の変更 | `aidex_files({ path: ".", modified_since: "30m" })` |

### 利用可能なツール（33種）
- **検索**: `aidex_init`, `aidex_query`, `aidex_search`, `aidex_update`, `aidex_remove`, `aidex_status`
- **シグネチャ**: `aidex_signature`, `aidex_signatures`
- **プロジェクト情報**: `aidex_summary`, `aidex_tree`, `aidex_describe`, `aidex_files`
- **クロスプロジェクト**: `aidex_link`, `aidex_unlink`, `aidex_links`, `aidex_scan`
- **セッション管理**: `aidex_session`, `aidex_note`, `aidex_settings`, `aidex_viewer`
- **タスク管理**: `aidex_task`, `aidex_tasks`
- **グローバル検索**: `aidex_global_init`, `aidex_global_status`, `aidex_global_query`, `aidex_global_signatures`, `aidex_global_refresh`, `aidex_global_guideline`
- **Log Hub**: `aidex_log`
- **スクリーンショット**: `aidex_screenshot`, `aidex_windows`
- **Viewer**: `aidex_viewer`（http://localhost:3333）

### プロジェクトインデックス作成
```markdown
aidex_init({ path: "C:\\VSC_Project\\ConvoPeq" })
```
既に実行済み: 275 files, 38706 items, 3936 methods, 368 types.（`.aidex/` 存在）

### AiDex 使用優先順位（コード検索・理解時）
1. `aidex_query` — 識別子検索（最もトークン効率が良い）
2. `aidex_signature` — ファイル構造確認
3. `aidex_search` — 自然言語検索
4. `aidex_summary` + `aidex_tree` — プロジェクト把握
5. `aidex_session` — セッション開始必須
6. `aidex_note` — 引継ぎノート
