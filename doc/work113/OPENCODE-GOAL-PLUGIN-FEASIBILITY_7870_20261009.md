# opencode-goal-plugin 導入可否調査 — OpenChamber 環境

- 作成: 2026-10-09
- 対象: https://github.com/prevalentWare/opencode-goal-plugin
- 目的: **本環境（OpenChamber）で使用できるか**の調査のみ。**インストールは実施していない。**
- 判定: **サーバ側エントリは使用可能（USEABLE）。契約・エントリ解決・ランタイム互換性的すべて MEASURED で PASS。ただし機能面（V2 API 版数）の互換性は静的には証明できず、TUI 相当機能は OpenChamber では機能しない。**

---

## §0. 結論サマリ

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| **サーバ側エントリのロード** | **USEABLE** | Bun 1.4.2 上で `pkg/server` が第 1 候補で解決し、`{ id, setup }` 契約 PASS（§4 / §5） |
| **OpenCode プラグイン契約** | **PASS** | `default export = { id: "local.goal-mode.server", server, setup }`（§3） |
| **root export 欠如** | **問題なし** | OpenCode は `["server", ""]` の順で候補を試す（§4.1） |
| **config キーの必要性** | **不要** | `plugins` 配列は既存稼働中（§2.2） |
| **engine 適合** | **PASS** | `engines.opencode >= 1.17.1` に対し 2.0.26（§2.1） |
| **TUI サイドバー／パレット** | **N/A（GUI では機能しない）** | OpenChamber は Electron GUI。TUI 用の `./tui` は GUI 経路に載らない（§5.3） |
| **V2 API 版の互換性** | **NOT PROVEN（リスク）** | プラグインは beta-19425 契約に固定、本環境は 2.0.26（§6.1） |
| **副作用（ファイル書き込み）** | **あり** | `~/.local/share/opencode-goal-plugin/goals.json` を作成（§6.3） |
| **auto-continue デフォルト有効** | **要 Owner 判断** | デフォルト `auto_continue: true`（§6.4） |

**総合判定：サーバ機能（`/goal` コマンド・goal ツール・永続 goal state）は導入可能。ただし V2 API 版の互換性は実行時エラーとしてのみ現れる可能性があり、これは静的には証明できない。**

---

## §1. 対象パッケージの正体（MEASURED）

### §1.1 識別情報

```text
package : @prevalentware/opencode-goal-plugin
repo    : https://github.com/prevalentWare/opencode-goal-plugin
latest  : 0.1.59   （npm dist-tags: latest = 0.1.59）
license : MIT
type    : module
tarball: prevalentware-opencode-goal-plugin-0.1.59.tgz  (75,532 B)
```

**GitHub HEAD の `package.json` は version 0.1.1 だが、npm publish 済みは 0.1.59 である。**以下の評価は **publish 済み 0.1.59 の tarball** に対して行った（HEAD の内容ではない）。

### §1.2 提供物

```text
dist/server.js   245,999 B   ← サーバ側（hooks / tools）
src/tui.ts        31,571 B   ← TUI 側（sidebar / palette）
src/i18n.ts       25,911 B
src/state-path.ts     419 B
README.md         27,071 B
package.json       2,473 B
LICENSE            1,070 B
```

### §1.3 exports マップ

```json
{
  "exports": {
    "./server": { "import": "./dist/server.js" },
    "./tui":    { "import": "./src/tui.ts" }
  }
}
```

**`.` root export が無い点に注目する。**§4.1 で、これは OpenCode 側で吸収されることを確認する。

---

## §2. 実行環境の適合性（MEASURED）

### §2.1 OpenCode バージョン

```text
opencode --version  → opencode v2.0.26
opencode2 --version → opencode v2.0.26
real install       : %USERPROFILE%\.bun\install\global\node_modules\@opencode\cli
  package.json version = 2.0.26
  bin = { opencode: ./bin/opencode.exe, opencode2: ./bin/opencode.exe }
  exe size = 212,567,080 B（Bun 単一バイナリ）
```

プラグインの `engines.opencode` との照合:

```json
"engines": { "opencode": ">=0.0.0-beta-0 <0.0.0 || >=1.17.1" }
```

| 判定 | 内容 |
| --- | --- |
| `>=0.0.0-beta-0 <0.0.0` | beta 版のみ。本環境は該当しない |
| `>=1.17.1` | **2.0.26 >= 1.17.1 → 充足** |
| **総合** | **PASS** |

### §2.2 `plugins` config キーは既に稼働中

**新規 config キーを追加する必要はない。**

```text
global  %USERPROFILE%\.config\opencode\opencode.jsonc
        "plugins": [ "opencode-cmd-provider", "@openchamber/opencode-claude@1.3.8" ]

project <worktree>\opencode.json
        "plugins": []
```

サーバーログ（`%USERPROFILE%\.local\share\opencode\log\opencode.log`）で **role=server として実際にロードされている**ことを確認した。

```text
msg="loading plugin" id=opencode-cmd-provider
  entrypoint=file:///C:/Users/user/.cache/opencode/npm/opencode-cmd-provider@latest/1791368417307/
            node_modules/opencode-cmd-provider/dist/index.js   role=server
msg="loading plugin" id=@openchamber/opencode-claude@1.3.8
  entrypoint=file:///C:/Users/user/.cache/opencode/npm/@openchamber/opencode-claude@1.3.8/1791368423220/
            node_modules/@openchamber/opencode-claude/opencode-claude.js   role=server
```

現在ロード中のプラグイン 4 種:

```text
opencode-cmd-provider                              (npm)
@openchamber/opencode-claude@1.3.8                 (npm)
C:\Users\user\.config\openchamber\agent-tool\...    (ローカル JS)
C:\VSC_Project\ConvoPeq\.opencode\plugins\headroom-proxy (ローカル TS)
```

**プラグインロード失敗は 0 件**（`failed to load plugin` / `PluginModule.LoadError` の実エラーは無し。検出された 1 件は PowerShell spawn 行への文字列誤マッチ）。

**→ `plugins`  Mechanism は本環境で実証的に動作している。**

### §2.3 npm キャッシュの構造

```text
C:\Users\user\.cache\opencode\npm\
  @openchamber\  context-mode@latest\  headroom-proxy@latest\
  list@latest\   opencode-cmd-provider@{2.1.9,2.2.0,2.2.1,2.2.2,2.2.4,latest}\  rtk-wsl@latest\
```

**goal プラグインは未インストール**（キャッシュ・global config・project config のいずれにも存在しない）。

---

## §3. OpenCode プラグイン契約の適合性（MEASURED / 決定的）

### §3.1 OpenCode が要求する契約

OpenCode 2.0.26 バイナリ内（`PluginModule` 周辺）に埋め込まれた契約チェック:

```text
embedded strings:
  "Plugin must export a default definition"   x2
  "Plugin entrypoint not found: "             x1
```

### §3.2 goal プラグインの実際の default export

`dist/server.js` の末尾（バンドル实际）:

```javascript
var server_default = {
  id: "local.goal-mode.server",
  server,
  setup: setupV2
};
export {
  server_default as default
};
```

### §3.3 判定

| 契約要件 | goal プラグイン | 判定 |
| --- | --- | --- |
| `default` export が存在する | `export { server_default as default }` | **PASS** |
| `id` が文字列 | `"local.goal-mode.server"` | **PASS** |
| `setup` が関数 | `setup: setupV2`（`typeof === "function"`） | **PASS** |
| オブジェクト形式（V1 の `async (ctx) => {}` 形式でない） | `var server_default = { ... }` | **PASS** |

**→ OpenChamber / OpenCode 2.0.26 のプラグイン契約を充足する。**

**注記：OpenCode 公式ドキュメントの V1 形式 `export const X: Plugin = async (ctx) => ...` は本環境で失敗する（本機の既往実績）。**本プラグインは `export default { id, setup }` のオブジェクト形式であるため、**その既知の失敗モードに当たらない。**

---

## §4. エントリ解決ロジック（MEASURED / 決定的）

### §4.1 OpenCode 2.0.26 の実解決ロジック

バイナリ内に埋め込まれた resolver（抜粋）:

```javascript
function wI(r){                     // r = { name, directory }
  let n = (t) => {
    for (let o of t) {
      let i = r.name ? [r.name, o].filter(Boolean).join("/")
                     : s.resolve(r.directory, o || "index");
      try { return nb(i, r.directory) }
      catch(e) {
        if (!(e instanceof Error) || !("code" in e) ||
            !["ENOENT","ENOTDIR","MODULE_NOT_FOUND","ERR_MODULE_NOT_FOUND",
              "ERR_PACKAGE_PATH_NOT_EXPORTED","ERR_UNSUPPORTED_DIR_IMPORT"]
               .includes(String(e.code)))
          throw e
        // 上記エラーコードは握り潰して次の候補へ進む
      }
    }
    return;   // 全候補失敗 → undefined
  };
  return {
    server: n(["server", ""]),   // ← "pkg/server" を FIRST に試す
    tui:    n(["tui"]),
    rpc:    n(["rpc"])
  };
}
```

ローダ本体:

```javascript
var ZU = c("PluginModule.load")(function*(e,a,i){
  ...
  p = ... wI(_ ?? {directory:e.target}),
  g = p.server;
  if (!g) return yield* new jp({ message: `Plugin entrypoint not found: ${e.target}` });
  yield* Bh({ msg:"loading plugin", id:e.target, entrypoint:g });
  ...
});
```

### §4.2 §1.3 の「root export 欠如」の結論

```text
server 候補順 = [ "pkg/server", "pkg" (root) ]
                 ^^^^^^^^^^^^ FIRST
```

**`./server` export が存在する goal プラグインは、最初の候補で解決する。root export 欠如は問題にならない。**

### §4.3 実測（Node 22 相当の resolver）

```text
@prevalentware/opencode-goal-plugin           -> RESOLVE-ERROR: ERR_PACKAGE_PATH_NOT_EXPORTED
@prevalentware/opencode-goal-plugin/server    -> file:///.../dist/server.js      ← 解決成功
@prevalentware/opencode-goal-plugin/tui       -> file:///.../src/tui.ts          ← 解決成功
```

**Node では root が失敗する。**OpenCode は Bun を使い、`n(["server",""])` の順で試すため、実運用では §4.1 の通り `pkg/server` で解決する（§5 で Bun 上で実証）。

---

## §5. Bun ランタイムでの実証（MEASURED / OpenCode と同一ランタイム）

### §5.1 実行結果

OpenCode 2.0.26 と同一の Bun 上で、OpenCode の `["server", ""]` 順の解決と契約チェックを再現した。

```text
runtime = Bun 1.4.2

=== 1. reproduce OpenCode's entrypoint resolution ===
  candidate "@prevalentware/opencode-goal-plugin/server"
      -> C:\...\node_modules\@prevalentware\opencode-goal-plugin\dist\server.js
  （第 1 候補で成功。root へ fallback しない）

=== 2. import the entrypoint ===
  named exports : default
  default type  : object
  default keys  : id, server, setup
  default.id    : "local.goal-mode.server"
  setup         : function
  effect        : undefined
  server        : function

=== 3. OpenCode PluginModule contract check ===
  requires: default export with id AND (setup | effect)
  RESULT  : PASS
```

### §5.2 判定

**OpenCode 2.0.26 と同じ Bun ランタイム上で、goal プラグインのサーバエントリはロード可能であり、契約チェックも PASS する。**実インストールをしない限额で、ロード成否と契約適合を実証した。

### §5.3 TUI エントリの扱い

```text
@prevalentware/opencode-goal-plugin/tui
  -> .../src/tui.ts   （TypeScript ファイル）
```

**OpenChamber は Electron GUI であり、TUI（端末 UI）をホストしない。**サーバーログでロードされているプラグインエントリは全て `role=server` であり、`wI()` が返す `.server` のみが使用される。したがって：

```text
./tui エントリ : 解決するが、GUI 経路では import されない → 実害なし
sidebar / palette integration : OpenChamber では動けない（別 UI 表面）
```

**TUI 相当の表示機能（goal sidebar、Goal palette エントリ）は OpenChamber では利用できない。**`/goal` コマンドと agent tool はサーバ側なので，利用可能。

---

## §6. 残存リスクと副作用（評価）

### §6.1 V2 API 版のずれ（最重要リスク / NOT PROVEN）

```text
README 記載 : 「V2 開発契約を beta-19425 に固定」
devDeps    : @opencode/{client,plugin,schema,theme} = 0.0.0-beta-19425
本環境     : @opencode/cli = 2.0.26
```

**プラグインは `0.0.0-beta-19425` の V2 API 契約を前提に実装されているが、本環境は `2.0.26` である。**2.0.26 は beta-19425 より大幅に新しい。イベント名（`session.execution.succeeded` / `session.compaction` 等）や ctx API に差異がある可能性があり、**これは実行時エラーとしてのみ顕在化し、静的には証明できない。**

```text
影響範囲の見積り :
  低リスク : goal state の永続化（ファイル I/O ベース）
  中リスク : /goal コマンド登録（command API）
  高リスク : idle continuation（session.idle / execution イベントに依存）
             compaction hook（session.compaction に依存、beta-19425 より新 否则 no-op）
```

### §6.2 peer dependencies（TUI 専用）

```json
"peerDependencies": {
  "@opentui/solid": ">=0.5.8",
  "solid-js": ">=1.9.0"
}
```

**TUI 専用でありサーバ経路では使用しない。**ただし sandbox 実インストールでは **134 packages** が導入された。GUI では実質不要だが、npm peer解決の都合で導入される可能性がある。

### §6.3 ファイル書き込み副作用

```text
state path : $XDG_DATA_HOME/opencode-goal-plugin/goals.json
             (未設定時) ~/.local/share/opencode-goal-plugin/goals.json
```

**このプラグインは `goals.json` を atomically に作成・更新する**（owner-only 権限）。OpenChamber のサーバは同一 user で動作するため写入可能。**repo 外（`~/.local/share/`）であり、repository には影響しない。**

```text
本報告の評価で goals.json を作成していない（未導入のため）。
```

### §6.4 `auto_continue` デフォルト有効

```json
"auto_continue": true,          ← デフォルト有効
"max_auto_turns": 25,
"min_continue_interval_seconds": 3,
"defer_while_tasks_active": true
```

**goal が active の状態では、`session.idle` / execution 完了イベントで自動的に継続 prompt を送信する。**本プロジェクトの Owner ワークフロー（commit / push の一律禁止、Owner GO 必須）と性質が衝突しうるため、**デフォルト有効のまま導入しないことを推奨する。**Owner 判断事項。

---

## §7. 導入する場合の構成（未実施・参考情報）

### §7.1 OpenCode 2 では `plugins` 配列へ追加する

`~/.config/opencode/opencode.jsonc`（global）または `<project>/opencode.json`:

```json
"plugins": [
  "opencode-cmd-provider",
  "@openchamber/opencode-claude@1.3.8",
  "@prevalentware/opencode-goal-plugin"
]
```

**オプションを付ける場合は object 形式:**

```json
"plugins": [
  {
    "package": "@prevalentware/opencode-goal-plugin",
    "options": {
      "auto_continue": false,
      "locale": "ja",
      "restricted_agents": ["plan"]
    }
  }
]
```

### §7.2 TUI 設定は不要

```text
OpenCode 2 は V1 の tui.json を読まない。
本環境は Electron GUI のため TUI 設定も不要。
`~/.config/opencode/cli.json` は本環境には存在しない（GUI では不要）。
```

### §7.3 導入検証手順（案）

```text
1. global config の plugins に追加
2. OpenChamber を完全終了してから再起動（サーバが config を読むため）
3. サーバーログを確認：
   msg="loading plugin" id=@prevalentware/opencode-goal-plugin entrypoint=.../dist/server.js role=server
   ※ "Plugin entrypoint not found" / "PluginModule.LoadError" が無いこと
4. セッション内で /goal を実行し、goal state が作成されることを確認
5. goals.json が ~/.local/share/opencode-goal-plugin/ に生成されることを確認
```

---

## §8. 判定の分離

| 命題 | 判定語 | 根拠 |
| --- | --- | --- |
| npm パッケージの存在と入手可能性 | **MEASURED** | 0.1.59 tarball 取得成功 |
| OpenCode バージョン適合 | **MEASURED / PASS** | 2.0.26 vs `>=1.17.1` |
| `plugins` config mechanism が本環境で動くこと | **MEASURED** | 4 プラグインがロード済み、失敗 0 |
| OpenCode プラグイン契約の充足 | **MEASURED / PASS** | `{ id, setup }` default export |
| エントリ解決（root export 欠如の吸収） | **MEASURED / PASS** | Bun 1.4.2 で `pkg/server` 第1候補成功 |
| サーバエントリのロード可能性 | **DESIGN-VALIDATED → MEASURED** | Bun 上で契約チェック PASS（§5） |
| TUI 機能の GUI 利用可能性 | **NOT APPLICABLE** | OpenChamber は TUI をホストしない |
| V2 API 版の互換性（beta-19425 vs 2.0.26） | **NOT PROVEN** | 実行時エラーとしてのみ顕在化（§6.1） |
| 実際の goal ライフサイクル動作 | **NOT PROVEN** | 未導入のため未観測 |
| 本環境でのインストール有無 | **未インストール** | §2.3 |

---

## §9. 結論

1. **サーバ側エントリは本環境で使用可能である。**OpenCode 2.0.26 と同じ Bun 1.4.2 上で `@prevalentware/opencode-goal-plugin/server` が第 1 候補として解決し、`{ id, setup }` 契約チェックが PASS した（§5）。

2. **OpenCode プラグイン契約に適合する。**`export default { id: "local.goal-mode.server", server, setup }` であり、本環境で失敗する V1 の `async (ctx) => {}` 形式ではない（§3）。

3. **root export 欠如は問題にならない。**OpenCode の resolver は `server: n(["server", ""])` の順で候補を試し、`ERR_PACKAGE_PATH_NOT_EXPORTED` を握り潰す。`./server` が第 1 候補で解決する（§4.1 / §5.1）。

4. **config の追加は不要。**`plugins` 配列は既に global config に存在し、4 プラグインが正常にロードされている（失敗 0 件）。新しい config キーを追加する必要はない（§2.2）。

5. **engine 適合。**`>=1.17.1` に対し 2.0.26（§2.1）。

6. **TUI 相当の機能は OpenChamber では利用できない。**sidebar / palette は TUI 専用であり、GUI 経路に載らない。`/goal` コマンドと agent tool（12 種）はサーバ側なので利用可能（§5.3）。

7. **最大リスクは V2 API 版のずれである。**プラグインは beta-19425 契約に固定されているが本環境は 2.0.26。idle continuation / compaction hook のようなイベント依存部分に差異がある可能性があり、**静的には証明できない（NOT PROVEN）**（§6.1）。

8. **副作用として `~/.local/share/opencode-goal-plugin/goals.json` を作成する。**repo 外であり repository には影響しない（§6.3）。

9. **`auto_continue` がデフォルト `true` である点に注意。**goal active 時に自動継続 prompt が送信される。本プロジェクトの Owner ワークフロー（commit / push の自動禁止）と性質が衝突しうるため、**デフォルト有効のまま導入しないことを推奨する**（§6.4）。

10. **本調査ではインストールを実施していない。**packages の取得と sandbox でのロード検証は `%TEMP%` 内のみで完結しており、OpenCode の config も既存プラグインの状態も変更していない。