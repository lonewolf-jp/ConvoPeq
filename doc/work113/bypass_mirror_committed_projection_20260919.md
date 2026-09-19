# WORK113 Phase 2-2: bypass Active mirror committed projection 同期（2026-09-19）

監査記録。ユーザー承認（2026-09-19 メッセージ「Phase 2-2 の 2 行同期は承認」）に基づく実装・検証結果。

## 1. 実装内容（意味論の固定）

承認された意味論どおり実装:

```text
eqBypassRequested / convBypassRequested  = intent
World.routing                            = committed authority
eqBypassActive / convBypassActive        = committed-state compatibility mirror（authority ではない）
```

| ファイル | 変更 | 性質 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.Commit.cpp` (onRuntimePublishedNonRt 内・`updateMaxMetric(youngest...)` 直後) | `convo::publishAtomic(eqBypassActive, world.routing.eqBypassed, release)` / `convBypassActive` 同様 + 意味論固定コメント | **機能変更（唯一）** |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:189-193` | bootstrap 2 行は残置、「Prepare-time write = bootstrap initialization のみ」コメントを追加 | コメントのみ |
| `src/audioengine/RuntimeBuilder.h:160-165` | 旧「stale mirror」記述を authority/mirror 語彙へ更新（WORK113-16 Phase 1 の記述は維持） | コメントのみ |

- mirror の writer は **prepare（bootstrap 初期化）+ publish（committed projection）の 2 箇所のみ**。mirror を authority と定義する記述はどこにも置いていない。
- Publish 後 immutable 原則: 本同期は `world.routing` からの単方向 copy のみで World mutation を行わない。
- 既知の許容事項（コメントに明記）: publish（RT swap）から本 callback（CoordinatorLoop Non-RT）実行まで mirror は旧値を保持し得る（compatibility mirror としての短時間 eventual consistency）。

## 2. 静的受入ゲート — 全 PASS

```text
[x] production 変更は Commit.cpp の予定箇所のみ（PrepareToPlay / RuntimeBuilder.h はコメントのみ）
[x] RT path 変更 = 0（mirror の reader に RT 経路なし）
[x] World mutation = 0（単方向 copy のみ）
[x] new/delete/allocation = 0
[x] direct atomic API = 0（rg "(eq|conv)BypassActive\.(store|load|...)" → 0 件・publishAtomic 経由のみ）
[x] publish 経路追加 = 0（既存 onRuntimePublishedNonRt 内への mirror 書込のみ）
[x] Active mirror writer = prepare + publish の 2 箇所
    PrepareToPlay.cpp:192-193 / Commit.cpp:422,426（publishAtomic 対象 4 行のみ）
```

- CI 検証スクリプト: `isr-verify-publication-atomicity.ps1` **PASS** / `isr-verify-runtime-world-identity.ps1` **PASS** / `isr-verify-semantic-validity.ps1` **PASS**（既存の deferred WARN のみ・exit 0）
- reader 全数（変更前監査の再確認）: `AudioEngine.Processing.Latency.cpp:124`（host-facing: conv レイテンシ込み/除外判定）・`AudioEngine.h:1373-1374`（public getters）・`AudioEngine.h:3119-3120`（fallback snapshot）・`BassBuzzMeasurement.cpp:1312`（test log のみ）

## 3. ビルド

- **Release build 成功（exit 0・ConvoPeq.exe / AudioEngineHarness.exe 含む全テストターゲット）**。
- 環境メモ（本作業で確定・memory 登録済み）: build.bat(msvc モード) は vcvars を自己注入しない。MKL/IPP は `find_package` に加え一部テストターゲットが `%INCLUDE%` 経由で `mkl.h`/`ipp.h` を解決するため、vcvars 後の INCLUDE prepend が必要。stamp gate（COHERENCE-4）は build.bat が常に `-DCMAKE_CXX_COMPILER=cl` を強行するため、フルパス時代の stamp と必ず衝突する（E-G3-3 既知欠陥の継続）— stamp 削除→再 stamp で回復（物理コンパイラ同一のため clean 不要）。

## 4. 機能検証 — mirror transition 確認

- **default AudioEngineHarness: 全 PASS**（`checkSR03LatencyDelayClash` 等の既存 latency テスト T-SR03-1..6 含む・exit 0）。
- **rigcheck=bare: PASS**（sine50 −6dBFS: ratio 0.8846 / THD −149.3dB — 透過性基準内）。
- **rigcheck=eq: mirror 状態観測 — Phase 2-2 の動的証明**:

```text
[BUZZ] RIGCHECK(eq) state: eqBypassReq=0 eqBypassActive=0 convBypassReq=1 convBypassActive=1 seq=5
```

  prepare（bootstrap: active=false）後に `setConvolverBypassRequested(true)` → publish（seq 進行）→ **mirror が committed world.routing に追従（convBypassActive=1）**。EQ は true→false の反転（ON→OFF 方向）も追従。旧実装（prepare 時のみ sync）では convBypassActive は stale の false のまま残るケースであり、本結果は修正の効能を直接示す。
  - 備考: 同モードの sine50 透過性判定は FAIL（ratio 0.5640 / THD −53.6dB）。これは rig 側の既存不整合（20 バンド +3dB × saturation 0.05 設定に対し identity 判定窓 [0.880,0.897] + THD<−80 を要求）であり、mirror は DSP 経由でないため Phase 2-2 とは無関係（test 側 §6-4）。

## 5. boundary jump 比較（Phase 2-2 後・同一条件 A/B/C）

共通条件: `--buzz-probe=delta --buzz-eq=on --buzz-conv=on --buzz-probe-signal=sine50 --buzz-sr=384000 --buzz-dur=4 --buzz-quiet=10000 --buzz-flip-t=1.0`（4 run のみ flip 種別を変更）。capture: `acc_11317b_{A_eqgain,B_eqbypass,C1_hc,C2_lc}.csv`。

| run | flip | 出力への効果 | max \|x[n]−x[n−1]\|（全区間） |
| --- | --- | --- | --- |
| A | EQ totalGain −3dB (kind 4) | **完全不変**（rebuild/publish 発火せず・DSP 未到達） | 7.9e-5〜8.6e-5 |
| B | EQ bypass on (kind 3) | 不変（flat EQ = identity のため設計上透明） | 同上 |
| C1 | HC mode (kind 1) | 不変（50Hz は帯域内・透明） | 同上 |
| C2 | LC mode (kind 2) | 振幅 0.0679→0.0629（−7.4%）が t≈1.6–1.8s に**滑らかに**着地 | 同上（不連続なし） |

- 7.9e-5 は 50Hz 正弦（amp 0.0960 @384k）の 1 サンプル差分ベースラインと一致。**全カテゴリで境界不連続は観測されなかった。**
- C2 は flip 後に **world publish が発生しないまま**（全 run の [PUBLISH] は capture 前の seq 8 まで）振幅が変化した → HC/LC filter mode は world 経路でなく convolver 側の平滑適用（非 world 経路・crossfade 的）であることが判明。
- **A の結果は T5c 問題の動的再確認**: `setEQTotalGain` は本 flow で rebuild も publish も駆動せず、`applyTotalGainDbNonRt`（world build 時・RuntimeBuilder.cpp:485）は到達しない。→ **現行 build において EQ totalGain 変更は単独では world swap を生まない**ため、「0.0386 が EQ totalGain 実装固有の境界不連続」という仮説は成立しない。
- **総合判定**: prior session の T5c 0.0386 は現行条件下で非再現。現行 build の swap/適用機構に sample 不連続はなく、0.0386 は prior session 固有の条件（当時の build 状態・フラグ・capture 設定）に由来と推定する。本項をもって boundary jump 比較は **確定（CLOSED: 不連続なし・totalGain は swap を駆動しない）**。0.0386 の出典条件を特定する必要が生じた場合は prior session の実行ログの確認が必要（本リポジトリ内に記録なし）。

## 6. test 側計器の欠陥（production 無関係・今後の修正候補）

1. **TransitionMetrics rmsTrMax 常に 0**: ローカル `transientRmsMax` を計算後 `result.` へ代入していない（`TransitionMetrics.h` の computeTransitionMetrics）。awk による同一窓再計算で実値 0.2185 を確認。他指標（jumpPre/jumpTr/rmsPre/dc/amp）は data と完全一致。
2. **`--buzz-eq=` / `--buzz-conv=` は "on"/"off" 期待**: `"1"` を渡すと黙って 0（bypass）に解釈される（fail-silent）。`PROBE_CFG` 行で eq/conv の実効値を必ず確認すること（本作業でも一度踏んだ）。
3. **metrics 窓の sr 仮定**: `computeTransitionMetrics(out, opt.sr, ...)` は capture 実効レート == opt.sr を仮定するが、capture 総サンプル数はデバイス依存（本環境: engine 384kHz・capture 2.0s 指定で ~4.59s 分のサンプル）。opt.sr をデバイス実効レートに合わせても窓ずれが残り得る → flip 位置はデータから実測して確認するのが安全。
4. **rigcheck=eq の判定窓不整合**: 前述 §4 のとおり（設定 vs 判定基準）。

## 7. 保留事項の確定（棚卸し）

| 項目 | 結論 |
| --- | --- |
| 未確認 2 点（`ASSERT_NON_RT_THREAD`=DspNumericPolicy.h:183 / `setEQTotalGain`=AudioEngine.h:1305 で uiEqEditor のみ） | ユーザー提示の直接証跡で close 済み。さらに §5-A で setEQTotalGain 不活性を動的にも確認 |
| ConvoPeq.md 版ずれ（11:45 vs 15:31:30） | 現行リポジトリの ConvoPeq.md は `Generated: 2026-09-19 15:31:30` — ユーザー参照版と一致 → close |
| boundary jump 0.0386（A/B/C 比較） | 本記録 §5 で確定（現行 build: 不連続なし・非再現）→ close |
| Phase 2-2 | 実装完了・全ゲート PASS → closed（commit 待ち） |

## 8. 残置（次アクション候補）

- 本変更の commit（production 3 ファイル + 本記録 + capture CSV 4 点）。
- §6 の test 側計器修正（rmsTrMax 代入漏れ・フラグパース・判定窓）は別 work item 推奨。
- rigcheck=eq の mirror 状態観測行は Phase 2-2 の恒常的な回帰観測点として有用（test 側維持推奨）。

## 9. ユーザー受入判定（2026-09-19 追記）

ユーザー審査の結果、全項目 **PASS / CLOSED** として受理:

- Phase 2-2 実装 / World.routing → mirror 同期 / writer = prepare + committed publish の 2 箇所 / mirror を authority にしない / RT path 0 / World mutation 0 / direct atomic 0 / CI 3 件 / Release build / default AudioEngineHarness / latency T-SR03-1..6 / mirror transition / boundary jump 追加調査 → **全 PASS**
- boundary jump（T5c 0.0386）: **CLOSE** — 現行条件では再現せず、totalGain 単独では world swap が発生しないため「Phase 2-1 実装固有不連続」として残す根拠は消滅
- 一方向関係（Requested → RuntimeBuilder → RuntimeWorld.routing = committed authority → Publish → onRuntimePublishedNonRt → Active mirror = compatibility projection）を設計上の確定として承認（Publish 後 immutable・Build → Validate → Publish・atomic wrapper 経由の不変条件に整合）

**commit**: `3ea3208` `fix(audioengine): project committed RuntimeWorld routing into bypass Active mirrors`（Commit.cpp + RuntimeBuilder.h[Phase 1 込] + PrepareToPlay bootstrap コメント hunk + 本記録）

**残置**: test-instrumentation cleanup は別 work item（→ `test_instrumentation_cleanup_20260919.md`・Phase 2-2 closure には含めない）
