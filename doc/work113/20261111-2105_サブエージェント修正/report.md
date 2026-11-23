# サブエージェント欄の整理とプロジェクトルート md の扱い

OpenChamber（バックエンド opencode V2・Windows 11 x64）で、プロジェクトルートのマークダウンを監査するタスクがサブエージェントとして表示されていた件を調査し、表示を解消した記録である。**ルート直下の `*.md` がサブエージェント定義として読まれる経路は存在しなかった。** 実体は 2026-11-08 に作られた子セッション 7 件であり、これらをアーカイブして左ペインから消した。

## §0 結論

- GUI 左ペインの「サブエージェント」は**定義ファイルの一覧ではなく、現在のセッションの子セッション（サブセッション）の一覧**である。並んでいたのは `Audit README against code` など 7 件で、いずれも 2026-11-08 にルート文書を 1 文書 1 サブエージェントで監査したときの子セッションだった。
- **サブエージェント定義として読まれるのは 2 系統だけ**である。ユーザー設定の `~/.config/opencode/{agents,agent,modes,mode}/<name>.md` と、プロジェクトの `<project>/.opencode/{agents,agent,modes,mode}/<name>.md`。ルート直下の md は探索対象ではない（3 通りの実測で確認。§2）。
- 当該 7 件をアーカイブし、左ペインのサブセッション欄は空になった（§4）。アーカイブは OpenCode 側の `time_archived` に書かれるため、`/api/openchamber/sessions/unarchive` で復元できる。
- 再発防止として `AGENTS.md` に「探索元は 2 系統だけ」「ルート md はドキュメントであって定義ではない」「文書監査をサブエージェントへ分割しない」を明記した（§5）。

## §1 症状の特定

OpenChamber のブラウザパネルで `http://127.0.0.1:63621` を開き、「セッションを開く」→ `main-develop` の「サブセッションを展開」で、次の 7 件がサブエージェントとして並んでいた。

```
main-develop
  Third-pass audit README and SOUND sections 1-7
  Deep audit SOUND_PROCESSING part 2
  Deep audit ARCHITECTURE and BUILD_GUIDE part 2
  Audit SOUND_PROCESSING.md against code
  Audit ARCHITECTURE.md against code
  Audit BUILD_GUIDE_WINDOWS.md against code
  Audit README against code
```

いずれも文書名が入っているため、ルートの md がエージェント定義として取り込まれたように見える。実際には次の 2 点が原因で、定義の誤読ではない。

1. 表示主体が「定義」ではなく「子セッション」であること。
2. ルート文書を 1 ファイルずつサブエージェントに読ませる監査を 2026-11-08 に実行し、その子セッションが未アーカイブのまま残っていたこと。

## §2 探索元の実測（3 通り）

| 手段 | 結果 |
|---|---|
| `opencode debug agents` | 7 件（build / compaction / explore / general / plan / summary / title）。md 名のエージェントは無い |
| OpenChamber `GET /api/agent` | 同じ 7 件。GUI が受け取る定義一覧にも md 名は無い |
| プローブ（ルート直下 vs `.opencode/agent/`） | ルートに `zzprobe-root-agent.md`（`description` と `mode: subagent` 付き）を置いても 7 件のまま。`.opencode/agent/zzprobe-singular.md` に置いたときだけ 8 件目として現れた |
| ソース確認（`app.asar`） | `…/web/server/lib/opencode/agents.js` の探索先は `USER_AGENT_DIRS`（`~/.config/opencode/{agents,agent,modes,mode}`）と `<workdir>/.opencode/{agents,agent,modes,mode}` のみ。プロジェクトルートは見ていない |

プローブは検証後に両方削除し、`opencode debug agents` が 7 件へ戻ることと、`.opencode/agent/` ディレクトリが残っていないことを確認した。ルートの md にエージェント用フロントマターが付いていないことも確認済みである（先頭行はすべて `# 見出し` か空行）。

## §3 表示されていた実体

`session_v2` を参照すると、`ses_f19075ee…`（main-develop）の子セッションとして、2026-11-08 17:20〜18:47 に作られた 7 件が未アーカイブで残っていた。

| セッション ID | 作成 | タイトル |
|---|---|---|
| `ses_e45c1792fffep7fm9BHP6PnDsq` | 11-08 17:20 | Audit README against code |
| `ses_e45b91550ffeoAubbUlzelSjDe` | 11-08 17:29 | Audit ARCHITECTURE.md against code |
| `ses_e45b4b56affeu3vc597XtsMUN1` | 11-08 17:34 | Audit BUILD_GUIDE_WINDOWS.md against code |
| `ses_e45b4c8d6ffej8yY75Kaye6gu9` | 11-08 17:34 | Audit SOUND_PROCESSING.md against code |
| `ses_e458abd88ffetVfduK7jUDVsgv` | 11-08 18:20 | Deep audit ARCHITECTURE and BUILD_GUIDE part 2 |
| `ses_e458add4fffee8e1RZjtjLA0lr` | 11-08 18:20 | Deep audit SOUND_PROCESSING part 2 |
| `ses_e4571b0b5ffe2rcLFngd29CLQS` | 11-08 18:47 | Third-pass audit README and SOUND sections 1-7 |

表示条件はフロントエンドの `chat.workStatus.section.subagents` にあり、`child.parentID === 現在のセッション && 未アーカイブ && ワークフロー由来でない && ツール呼び出しに紐づかない` を満たす子セッションを並べる。ログやメッセージの整理でツール呼び出しとの紐づけが外れた子セッションが、ここに残り続ける仕組みである。

## §4 対処 — 7 件のアーカイブ

OpenChamber のアーカイブ API を使った（`body` は `{"ids": [...]}`）。

```
POST http://127.0.0.1:63621/api/openchamber/sessions/archive
{"ids": ["ses_e4571b0b5ffe2rcLFngd29CLQS", ... ]}
→ 200 {"archived":[{"id":"...","archivedAt":1794398425548}],"failedIds":[]}
```

1 件だけ先にアーカイブして左ペインから消えることを確かめ、残り 6 件を同じ API でまとめて処理した。結果は次のとおり。

- 左ペインを再読み込みすると、`main-develop` の下のサブセッションは 1 件も表示されなくなった。
- `session_v2.time_archived` に 2026-11-11 21:00:25 / 21:00:59 が記録された（7 件すべて）。
- アーカイブは OpenCode 側のレコードに書かれるため、TUI など他のクライアントとも状態が一致する。戻すときは `POST /api/openchamber/sessions/unarchive` に同じ `ids` を渡す。

## §5 再発防止（AGENTS.md への追記）

`C:\VSC_Project\ConvoPeq\AGENTS.md` に「サブエージェント定義の探索元 — ルート直下の md は定義ではない（2026-11-11 実測）」を追加した。内容は次の 5 点である。

- 定義として読まれる 2 系統（ユーザー設定とプロジェクトの `.opencode/`）を明記。
- ルート直下の `*.md`（README / ARCHITECTURE / SOUND_PROCESSING / BUILD_GUIDE_WINDOWS / HOW_TO_USE / RELEASE_NOTES / ConvoPeq.md / CLAUDE.md / AGENTS.md）はドキュメントであり、フロントマターを付けても定義にならないこと。
- 実測の再現手順（ルートに置いても増えない／`.opencode/agent/` に置けば増える）と `GET /api/agent` の 7 件。
- 左ペインに出るのは子セッションであること、表示条件、アーカイブで消えること。
- 今後はルート文書の監査をサブエージェントへ分割しないこと。子セッションを使った場合は完了後にアーカイブすること。

## §6 注意と残件

- 今回アーカイブしたのは文書監査の 7 件だけである。同じ main-develop の下には 2026-10-23 の子セッション（`A12 blind …` など 12 件）も未アーカイブで残っているが、左ペインのサブセッション欄には表示されていない（対処前も表示されていなかった）。不要なら同じ API で追加アーカイブできる。
- アーカイブはセッションの削除ではない。内容は残り、`unarchive` で元に戻せる。
- ルート文書をサブエージェントへ読ませる運用自体は禁止していない。1 文書 1 サブエージェントで乱発すると左ペインが埋まる、というのが今回の学びである。
- 本作業は設定・ドキュメントの変更のみで、ソースコードとコミット・マージには触れていない。

## §7 証跡（`evidence/`）

| ファイル | 内容 |
|---|---|
| `agent_discovery_probe.txt` | `opencode debug agents` と `GET /api/agent` の実測、ルート／`.opencode/agent/` のプローブ比較、後片付け確認 |
| `agents_js_excerpt.txt` | OpenChamber の探索コード（`app.asar` から抜粋） |
| `sidebar_before_after.txt` | GUI 左ペインの対処前・1 件アーカイブ後・全件アーカイブ後のテキスト |
| `archive_api_responses.txt` | アーカイブ API のリクエストと応答（HTTP 200） |
| `db_archived_state.txt` | `session_v2.time_archived` の記録（7 件） |
