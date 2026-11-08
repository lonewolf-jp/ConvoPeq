# OpenChamber 更新後の環境変数解除の確認（2026-11-08）

- 目的: OpenCode 2.2.6 のクロスサイト 403 回避に使っていた `OPENCHAMBER_ELECTRON_LOAD_SERVER_UI=1` を、OpenChamber の更新後に外せるかを確認する。
- 判定: **解除してよい**。更新版に修正が入っており、同梱 UI のリクエスト形（パッケージ版 Origin + クロスサイト）で API が 200 を返すことを実測した。
- 実施: 利用者環境変数を削除し、環境変更をブロードキャストした。次回起動で同梱 UI が使われる。

## 1. 更新の確認

| 項目 | 値 |
|---|---|
| 更新前 | `2.6.2-canary.20261107.11`（`main.log` の 2026-11-08 12:51 の更新記録） |
| インストール版 | `2.7.1-canary.20261108.12`（`OpenChamber.exe` のビルド日時 2026-11-08 12:09） |
| 更新チャネル | `desktopUpdateChannel = canary` |
| 同梱 OpenCode | v2.2.6（変更なし。403 の原因側は据え置き） |
| 稼働中プロセス | OpenChamber 4 プロセス（起動 2026-11-08 12:52〜12:53）、opencode v2.2.6 |

## 2. 修正の有無

| 確認 | 結果 |
|---|---|
| リリース v2.7.0 の `packages/web/server/proxy-headers.js` | `BROWSER_FETCH_METADATA_HEADERS`（`sec-fetch-site` / `mode` / `dest` / `user`）を含む |
| リリース v2.6.1 の同ファイル | 含まない（修正前） |
| インストール済み `app.asar` | `filteredRequestHeaders` に `sec-fetch-*` を含み、コメントで「2.2.6 以降の OpenCode は cross-site の `/api` を拒否する」と明記 |
| API プロキシ | `createProxyMiddleware` に `changeOrigin: true`、`pathRewrite` で `/api` を復元。上流へ `sec-fetch-*` を転送しない |
| CORS | 応答に `Access-Control-Allow-Origin: openchamber-ui://app` を付与（`packagedClientOrigins`） |
| 既定の UI | `shouldUsePackagedUi()` は `app.isPackaged`。環境変数が無ければ同梱 UI を使う |

上流の Issue #6503 / #6505 は OPEN のままで、Origin を落とす PR #6508 も未マージである。ただし 403 の判定は `Sec-Fetch-Site` を伴う場合に限られるため、`sec-fetch-*` の除去で解消する。

## 3. 実測（同梱 UI のリクエスト形）

`http://127.0.0.1:63621`（`desktopLocalPort`）に対し、同梱 UI と同じヘッダを付けて GET した。

| パス | ヘッダ | 結果 |
|---|---|---|
| `/api/agent` | `Origin: openchamber-ui://app` + `Sec-Fetch-Site: cross-site` ほか | 200、7,368 バイト |
| `/api/session` | 同上 | 200、39,852 バイト |
| `/api/config` | 同上 | 200、27,638 バイト |
| `/api/plugin` | 同上 | 200、11,875 バイト |
| `/api/agent`（ヘッダなし） | ― | 200、7,368 バイト（同一内容） |

応答には `Access-Control-Allow-Origin: openchamber-ui://app` と `Access-Control-Allow-Credentials: true` が付く。CORS の受理と、上流への転送の両方が通っている。

## 4. 実施した変更

| 対象 | 変更前 | 変更後 |
|---|---|---|
| `HKCU\Environment\OPENCHAMBER_ELECTRON_LOAD_SERVER_UI` | `1` | 値ごと削除 |

- `WM_SETTINGCHANGE`（`Environment`）をブロードキャストし、Explorer の環境ブロックを更新した。
- Explorer 経由で `cmd` を起動して確認したところ、`%OPENCHAMBER_ELECTRON_LOAD_SERVER_UI%` は空（`[]`）だった。次回のショートカット起動には環境変数が渡らない。
- 稼働中のプロセスは起動時の環境を保持するため、この起動では従来どおり 63621 の UI が使われる。次回起動から同梱 UI になる。

## 5. 残る確認と復元手順

- 次回起動で同梱 UI が読み込まれ、`agent.list failed (403)` が出ないことを確認する。このセッションの再起動は作業を中断させるため、実施していない。
- 失敗した場合の復元:

```powershell
[Environment]::SetEnvironmentVariable('OPENCHAMBER_ELECTRON_LOAD_SERVER_UI','1','User')
```

設定後に OpenChamber を起動し直せば元の回避策に戻る。

## 6. 参照

| 資料 | 内容 |
|---|---|
| Issue #6503 / #6505（openchamber/openchamber） | クロスサイト 403 の報告。OPEN |
| PR #6508 | Origin も落とす修正。未マージ |
| リリース v2.7.0 / v2.6.1 | `proxy-headers.js` の差分確認 |
| `app.asar`（2.7.1-canary.20261108.12） | `filteredRequestHeaders`、`packagedClientOrigins`、`shouldUsePackagedUi` |
| `doc/work113/20261107-1510_OC復旧/report.md` | 回避策を導入した経緯 |
