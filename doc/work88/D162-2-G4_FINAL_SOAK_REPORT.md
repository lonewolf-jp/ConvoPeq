# D162-2-G4 Work Report — Final Deterministic Soak（60-gen × 2）

- Work item: D162-2-G4（D162-2-G ステージ 4/4・最終確定用 deterministic soak・**production 変更 0**）
- Date: 2026-09-04
- 基準ソース: ConvoPeq.md `Generated: 2026-09-04 20:28:50` 相当（G2 tree・G3 と同一）
- 判定: **PASS（#1 / #2 両方）→ D162-2-G 完了（G0→G1→G2→G3→G4）**
- 詳細: evidence/D162-2G4_FINAL_SOAK_EVIDENCE.md

## 0. Executive Summary

S3=ON / V-D=ON の最終構成で RWDI 60-gen を 2 回連続実行し、**両 run とも
exit 0x00000000・crash dump 0・Signature A/B/C 0・E-3 0・residual 0・generation 1:1 完全突合・
E-4 会計完全収支・shutdown complete** を達成した。lifecycle closure は決定論的に再現
（収支構造は同一・trigger count のみ負荷揺らぎ）。**#1 / #2 両方 PASS → D162-2-G4 = PASS。**

## 1. 静的 preflight（最終再確認・全 ✓）

- `destroyRolledBackDSP`: V-D caller 0（rollback 経路 Orchestrator.cpp:292 のみ残存）
- S3: shutdown-clear retire が `deferredSlot_.reset()` より前に実行（Orchestrator.cpp:630-634）
- V-D: active/fading とも authority `retire()` のみ・`if (false &&` src 0 件
- `runtimeDSPHandleMap_.erase` は authority retire 内（AudioEngine.h:4367）
- G3 から source 変更なし

## 2. Gate ladder 結果

| Gate | 結果 |
| --- | --- |
| G4-1 Build（Debug/Release/RWDI） | 全 **EXIT=0** |
| G4-2 CTest Debug | **40/40 PASS** |
| G4-2 CTest Release | **39/40** — AudioEngineHarness 0xC0000374 のみ（pre-existing）→ **新規失敗 = 0** →「既知失敗のみ = PASS」条件充足 |
| G4-3 RWDI 60-gen #1 | **exit 0x00000000**・dump 0 |
| G4-4 RWDI 60-gen #2 | **exit 0x00000000**・dump 0 |

## 3. 判定マトリクス（#1 / #2）

| 判定項目 | #1 | #2 |
| --- | --- | --- |
| exit 0 / dump 0 | ✓ / ✓ | ✓ / ✓ |
| Signature A / B / C = 0 | ✓ | ✓ |
| E-3 = 0 / EBR overflow = 0 / EBR final pending = 0 | ✓ | ✓ |
| residual = 0（generation 1:1・連続・gap なし） | ✓（gens 4-64・61 個体） | ✓（gens 3-63・61 個体） |
| direct destroy = 0 / destroy duplication = 0 | ✓ | ✓ |
| stale map = 0（dtor retireByHandle **MISS** 実測） | ✓（handle=31） | ✓（handle=34） |
| E-4 完全収支 | ✓（26,254 = 26,209+35+8+2） | ✓（25,672 = 25,628+36+6+2） |
| shutdown complete / footprint remaining 全件 0 | ✓ | ✓ |
| 新規 XRUN class | なし（15 件・max 2.15ms・Pressure=0・shutdown 窓 0） | なし（12 件・max 1.70ms・同） |

## 4. S3 / V-D 経路分離（指示 §5）

| 項目 | #1 | #2 |
| --- | ---: | ---: |
| S3 CLEAR_SHUTDOWN_DISPOSITION | 8 | 6 |
| S3 closure（E-4d retired=1 → destroy → remaining=0） | **8/8** | **6/6** |
| S3 direct destroy | 0 | 0 |
| S3 standalone retired=1 | **not observed**（E-4d が常に先行・指示 §9 どおり報告） | 同 |
| V-D active target / retired=1 | **1 / 1** | **1 / 1** |
| V-D EBR Success → destroy → remaining=0 | ✓（epoch 232） | ✓（epoch 237） |
| V-D direct destroy / stale map | 0 / 0 | 0 / 0 |

- S3 と V-D の対象は run 内で重複なし（独立個体）。
- #2 の V-D 対象は address reuse 個体（旧 gen=56 / 新 gen=61 が同一アドレス）だったが
  generation 検証 registry が正しく分離 — D162-2-C §10 懸念の継続健全性を追加実証。
- S3 は G1〜G4 累積 14 回の実行時到達 + no-op 安全帰還を実証（standalone retired=1 は
  production 到達不能な異常系であり、指示どおり人工生成しなかった）。

## 5. 判定

**D162-2-G4 = PASS。**

### D162-2-G 総括

| Stage | 判定 | 成果 |
| --- | --- | --- |
| G0 preflight | PASS | switch 特定・V-D-a stale-map latent 発見・V-D-b 設計確定 |
| G1 S3 re-enable | PASS | S3 authority（EBR）化・無処分 1 件解消 |
| G2 V-D re-enable | PASS | direct destroy 廃止 → authority 統一・V-D retired=1 実測 |
| G3 joint reproof | PASS | 両経路同時成立・別個体閉包・変更 0 |
| G4 final soak | PASS | 60-gen × 2 決定論的 closure 再現 |

**D162-2-G（S3/V-D staged re-enable / final validation）完了。** shutdown 時 DSP 破壊は
EBR 単経路（INV-D162-8）で閉じ、INV-D162-1〜9 全契約を維持。V-D の direct destroy は
消滅（rollback 経路の正当 caller のみ残存）。S1-S4 / E-1-E-4 / G1-G4 により
registered DSP の全 terminal disposition が authority に統一された。
