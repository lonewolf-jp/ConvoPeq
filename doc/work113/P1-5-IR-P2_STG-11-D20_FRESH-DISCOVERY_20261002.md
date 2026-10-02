# STG-11-D20 Fresh Discovery — Numeric / Lifecycle / Authority Residual Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D20_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D20** — D19 までの結果を前提とした新規 defect 探索
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-142258).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `b9f2a89d3242d6ce76de905bd75cde58f27dd491`（D19 docs）
  - `Generated: 2026-10-02 23:19:21` / 5,941,824 B
  - SHA-256: `3A51AB3414DBA90005B99540376A93D6AB76AE1A96A754CE9E7050E2E0593045`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery** で開始し、NO-GO 確定のため
  production / test / build ファイルの変更は一切行っていない。

---

## 0. 判定

```
STG-11-D20 — NO-GO（concrete defect 0 件）
```

数値境界（prepare / EQ / convolver / mixed-phase）、lifecycle / authority / queue /
shutdown / RCU / callback / RT / atomic / validator、D7〜D19 の再発、O-1〜O-3 を
現行ソースで確認した。Owner の 7 条件をすべて満たす新規 defect は存在しない。

---

## 1. 追跡結果（新規分）

### 1.1 prepare の sample rate / block size 境界

`prepareToPlay`（`AudioEngine.Processing.PrepareToPlay.cpp:16-`）は
lifecycle CAS、duplicate collapse、rollback を備え、`:127-132` で
sampleRate（`<=0` / 上限超 / 非有限 → 48000）と block size（`<=0` → 既定）を
sanitize する。device 由来の異常値の到達経路は閉じている。

### 1.2 mixed-phase の除算経路

`rmsLinear / rmsMixed`（`MixedPhase.cpp:592-594`）は `>1e-12` 両成立時のみ除算し、
`peak / rms`（`:650`）は silent IR（0/0=NaN）でも比較が偽となり fallback/skip する。
`0.98 / peak`（`:661`）は `peak > 0.99` 条件下のみである。
NaN・ゼロ除算の RT 到達は無い。

### 1.3 lifecycle / authority / queue / shutdown / RCU / RT / atomic / validator

D16〜D19 の結論と同一であり、production 無変更のため再発は無い。
D15-1 mutex の NonRT-only、DSPCore の lock-free、atomic 規約、単一 Bridge は維持されている。

### 1.4 D7〜D19 の再発 / O-1〜O-3

worktree clean（production が `a1ce01b2` 以降 byte-identical）であるため、
全 repair は構造的に維持されている。O-1〜O-3 に昇格条件は無い。

---

## 2. 本 D20 の作業記録

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = b9f2a89d（= origin/main）
```
