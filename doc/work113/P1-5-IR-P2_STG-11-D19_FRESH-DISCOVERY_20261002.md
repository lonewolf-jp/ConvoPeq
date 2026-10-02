# STG-11-D19 Fresh Discovery — Numeric / Lifecycle / Authority Residual Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D19_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D19** — D18 までの結果を前提とした新規 defect 探索
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-142258).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `d90e5795782b89b3746aacb35b534a56de55d816`（D18 docs）
  - `Generated: 2026-10-02 23:19:21` / 5,941,824 B
  - SHA-256: `3A51AB3414DBA90005B99540376A93D6AB76AE1A96A754CE9E7050E2E0593045`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery** で開始し、NO-GO 確定のため
  production / test / build ファイルの変更は一切行っていない。

---

## 0. 判定

```
STG-11-D19 — NO-GO（concrete defect 0 件）
```

数値境界（EQ / convolver / engine / learner の NaN・極値経路）を中心に、
lifecycle / authority / queue / shutdown / RCU / callback / RT / atomic /
validator / 再発 / O-1〜O-3 を現行ソースで確認した。
Owner の 7 条件をすべて満たす新規 defect は存在しない。

---

## 1. 追跡結果（新規分）

### 1.1 EQ band 数値（frequency / gain / Q）の NaN・極値

`setBandFrequency` / `setBandGain` / `setBandQ`
（`EQProcessor.Parameters.cpp:19/39/59`）は band index のみ検査し値を無検証で格納する。
しかし係数経路は多層防御である。

- `validateAndClampParameters`（`EQProcessor.Coefficients.cpp:84-96`）が範囲 clamp。
- 全 5 SVF 計算（`:431-618`）が `isfinite(g) || isfinite(k)` 検査と除算ゼロ保護を持ち、
  NaN（freq NaN→tan NaN、q=0→k=Inf を含む）は bypass 係数に中和される。
- 全 5 biquad 計算（`:168-328`）が alpha 有限性と a0 微小値保護を持つ。

非有限・極値入力は RT 到達前に無害化される。defect 計上しない。

### 1.2 convolver 数値 setter

`setMix`（`:884-894`）、`setSmoothingTime`（`:989-999`）、
`setMixedTransitionStartHz/EndHz`（`:1047-1092`）はいずれも
`jlimit`＋`abs(prev-new) > eps` の 2 段であり、NaN 入力は publish されず no-op になる
（D13 §2.1 と同型の機構）。tail 系は D9 の明示 finite guard 済みである。
defect 計上しない。

### 1.3 shutdown drain / retire / epoch / reclaim

`ReleaseResources.cpp:660-710` の terminal drain は quiescence 確認後の
`drainAllNonRt` であり、residual は Generic 削除鎖へ移譲される。
停止順序違反・drain 漏れの構造は無い。

### 1.4 D7〜D18 の再発 / O-1〜O-3

再発なし。O-1（snapshot 非 lock 読出し）、O-2（unconditional tail）、O-3（dormant 分岐）は
いずれも concrete への昇格条件を確認できないため Observation のまま残す。

---

## 2. 本 D19 の作業記録

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = d90e5795（= origin/main）
```
