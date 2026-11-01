# Dropbox 配信物の独立検証（2026-11-01）

- 検証方法: `files/list_folder` で列挙し、**全ファイルをダウンロードして SHA-256 を再計算**し、
  ローカルの同一ファイルと突き合わせる（サイズ比較ではない）。
- リポジトリ: `HEAD` = `6bb7c8d5`、`origin/main` と一致
- 配信先ルート: `/ConvoPeq-stage3-20261026`
- ローカル `ConvoPeq.md` は CRLF（6,581,900 B）、配信したのは **LF 正規形**（git blob）。
  正規化すると作業ツリーと一致する（`BYTE-FORM-RECONCILIATION-20261101.md` 参照）。

## 結果

| 項目 | 値 |
|---|---|
| 検証対象ファイル数 | 60 |
| 一致 | 60 |
| 不一致 | 0 |
| 期待外（Dropbox のみに存在） | 452 |

## ファイル別 SHA-256

| # | Dropbox パス | サイズ (B) | SHA-256 | 判定 |
|---|---|---|---|---|
| 1 | `/ConvoPeq-f8aab49b-split/ConvoPeq.part01.md` | 2097115 | `4c1d09212da8aba52f8818c4ad22b8c80b962a8aeed9b5327421a68a4e724b71` | OK |
| 2 | `/ConvoPeq-f8aab49b-split/ConvoPeq.part02.md` | 2097123 | `83fd7edafe36a602e012ba7671bcc88ac3a7e623f7dddf880cf3acabc8417023` | OK |
| 3 | `/ConvoPeq-f8aab49b-split/ConvoPeq.part03.md` | 2097111 | `4340b3ae99d1457317f2d69c0893a0be247920d1d42c5f57b270ca6d101cc2e9` | OK |
| 4 | `/ConvoPeq-f8aab49b-split/ConvoPeq.part04.md` | 145010 | `898036199cdc01cc7712aaf0affdf6452db90a0c7bb60e896d128a9a1b3cfdf0` | OK |
| 5 | `/ConvoPeq-stage3-20261026/ConvoPeq.md` | 6436359 | `f8aab49b49dccd9134ec8a4e777a6ddbfe9ed7c3be7f8a923160049239fa2f08` | OK |
| 6 | `/ConvoPeq-stage3-20261026/ConvoPeq.part01.md` | 2097115 | `4c1d09212da8aba52f8818c4ad22b8c80b962a8aeed9b5327421a68a4e724b71` | OK |
| 7 | `/ConvoPeq-stage3-20261026/ConvoPeq.part02.md` | 2097123 | `83fd7edafe36a602e012ba7671bcc88ac3a7e623f7dddf880cf3acabc8417023` | OK |
| 8 | `/ConvoPeq-stage3-20261026/ConvoPeq.part03.md` | 2097111 | `4340b3ae99d1457317f2d69c0893a0be247920d1d42c5f57b270ca6d101cc2e9` | OK |
| 9 | `/ConvoPeq-stage3-20261026/ConvoPeq.part04.md` | 145010 | `898036199cdc01cc7712aaf0affdf6452db90a0c7bb60e896d128a9a1b3cfdf0` | OK |
| 10 | `/ConvoPeq-stage3-20261026/investigation-20261101/A-ASSERT-RACE-ANALYSIS-20261101.md` | 12905 | `09b45a1f86617300cc0cc5862f88c825a14f71ff4beba0ff9f302174826dc662` | OK |
| 11 | `/ConvoPeq-stage3-20261026/investigation-20261101/A2-A3A-ASSERT-DESIGN-20261101.md` | 11592 | `1e5d730003c69a57b4b479b7d5f796f799836c264ea2144160e1089050195408` | OK |
| 12 | `/ConvoPeq-stage3-20261026/investigation-20261101/A2-A3A-SOURCE-RECONCILIATION-20261101.md` | 12318 | `8425fa70b30f3ccac3c8a938e6aefb2ea2ed4fe6d391dec8700da149a5d44390` | OK |
| 13 | `/ConvoPeq-stage3-20261026/investigation-20261101/B-CRASH-DUMP-MANIFEST.md` | 1089 | `9ab65c7c130254fae42783c0b6428a248218d31dbf34b9cbc933516661ce2cc1` | OK |
| 14 | `/ConvoPeq-stage3-20261026/investigation-20261101/B-SEGFAULT-INVESTIGATION-20261101.md` | 19580 | `2dc63c64e5b6e8571fb12a52dd0b0a6c6778d2e14b3fa74f3800c5b6da5fd2ed` | OK |
| 15 | `/ConvoPeq-stage3-20261026/investigation-20261101/B2-ATTACH-CONTRACT-20261101.md` | 14544 | `81e94208e929fb43e2acfde825c5e3d01fc9affbe1a17ef3520947c338c12dbb` | OK |
| 16 | `/ConvoPeq-stage3-20261026/investigation-20261101/B2-ATTACH-CONTRACT-V2-20261101.md` | 14221 | `cdbc47bbe0c2d1ddf58ccae7a23ca69a949b08bf11f583b290e1adfeee118a31` | OK |
| 17 | `/ConvoPeq-stage3-20261026/investigation-20261101/B2-ATTACH-LIFECYCLE-PLAN-20261101.md` | 12920 | `d9625ece93310c02ef65d1a9b0af306e1233ad770eaa598fc3c2ed2c378073ad` | OK |
| 18 | `/ConvoPeq-stage3-20261026/investigation-20261101/B2-DIAG-LIFETIME-DESIGN-20261101.md` | 15123 | `907afdb0ccecac6143552aed9de8e7b2bfe6a0289c3ea0227ff798bd4b0a00eb` | OK |
| 19 | `/ConvoPeq-stage3-20261026/investigation-20261101/B2-FINAL-VERIFICATION-20261101.md` | 12570 | `b251c9530074eaef0b0c4f3fb997441bf9dfbb1f77dc4911fc182bbf2dbf0f83` | OK |
| 20 | `/ConvoPeq-stage3-20261026/investigation-20261101/B2-IMPL-READINESS-20261101.md` | 17552 | `87e471bb67d0e343968071598f8d44e58440f10cd2e72698aa256914f3250b3d` | OK |
| 21 | `/ConvoPeq-stage3-20261026/investigation-20261101/B2-LIFETIME-PROOF-20261101.md` | 13822 | `59e074306e956f5d50dbe5d522a7ea8efd732210140891c13ea29f8d8bb795f7` | OK |
| 22 | `/ConvoPeq-stage3-20261026/investigation-20261101/BYTE-FORM-RECONCILIATION-20261101.md` | 5588 | `21523cd35df171f3a7e0f1083676aeba59a1c215e9db9ce1292b97d540960e41` | OK |
| 23 | `/ConvoPeq-stage3-20261026/investigation-20261101/D01-IMPACT-ANALYSIS-LATEST-20261101.md` | 5179 | `e71800b9729c1413ec77d84e44a9b97ba4bdf5ed48b65db513ad8df9311b85ef` | OK |
| 24 | `/ConvoPeq-stage3-20261026/investigation-20261101/D02-MEASUREMENT-PROTOCOL-20261101.md` | 6052 | `395b8f86529e633604c08abeba786c81bb13893a3e560b27d626da095b30b814` | OK |
| 25 | `/ConvoPeq-stage3-20261026/investigation-20261101/D02-MEASUREMENT-RESULTS-20261101.md` | 5963 | `75c6123d07a5371c83620c5424a6f69befba4655e69a8fc19a831edd72840396` | OK |
| 26 | `/ConvoPeq-stage3-20261026/investigation-20261101/INDEX-20261101.md` | 4729 | `424fec534346f16a7881baa8d2b12df2a68314eb346cd34a331ff269f06dc22d` | OK |
| 27 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-abc-investigation.md` | 23129 | `8cdb121ebff92f109f3d628f7a9b7b482cd72e6057de5516211f3ac9d61c6f2a` | OK |
| 28 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-attach-contract-v2.md` | 6926 | `029bd9bd17ec46ab493a5cfa2bc1b8cb62c336f6866693477d22b910bc9418a5` | OK |
| 29 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-attach-contract.md` | 7363 | `74fd2134c19f6e0ab451f5e95d5e10cb2f1c6bddf7c11845964eaf16da4fb8e7` | OK |
| 30 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-attach-plan.md` | 6183 | `6a3dfad2f2cd33f550394923b6b412d4938a21dd8e32c3c3a652b774c466dd66` | OK |
| 31 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-design-phase.md` | 8967 | `b65d718f73b996d83b219a6230fb707bd687fc9bc1ea59c0f628d16a848f07fa` | OK |
| 32 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-final-verification.md` | 5302 | `af86667b8b5c24a0c98b6792cbc65bcf4b60f941ecdc185557b2f5f1acf47a08` | OK |
| 33 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-impl-readiness.md` | 8313 | `f0ae12237f0d50d30aa4a86ab31568043685b2adf58578c5066e2aa62636be10` | OK |
| 34 | `/ConvoPeq-stage3-20261026/investigation-20261101/REPORT-20261101-lifetime-proof.md` | 9497 | `3122bde451f7f36a90141e3a251db13301c4ed6c04a82154b6714b669403f5a5` | OK |
| 35 | `/ConvoPeq-stage3-20261026/investigation-20261101/STATUS-20261101.md` | 5597 | `9ec0a488a4a39d07be9a4db71063f1aa5ead08faf177983f140ee3bd76dfe144` | OK |
| 36 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_base_1.env.json` | 722 | `0a5ef9314b825b8e71b116ebdd0c0c3776960b0ebc977a9e6b13a723c52bad29` | OK |
| 37 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_base_1.log` | 48067 | `44fb58ef58b04d00cfd3c47315f63a6bc80645980ffa4b42f5a8a7e2e0f9f7ed` | OK |
| 38 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_base_2.env.json` | 720 | `40fccea4f56184551a0125ae989c41bbd6fc7363e5861b49b42d0ab0da6a179f` | OK |
| 39 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_base_2.log` | 48068 | `09cbaea6cd50cc4846cb543de07bdf36f88bbf57f75b0f9611546008fc58c29c` | OK |
| 40 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_base_3.env.json` | 720 | `6a6ebac6eb23b5f7889153c21180cffb69e1687e4bbb872338d24ba25c90782c` | OK |
| 41 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_base_3.log` | 48067 | `42c302a1d6083325924b7eb934407543c6f47a5783b2004598e35961ddc05962` | OK |
| 42 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_curr_1.env.json` | 682 | `b0bdc1e341929eaf5ae9cb01e63a090a704b6ed9dc8963a711d7d250ee1ddf01` | OK |
| 43 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_curr_1.log` | 120990 | `8dea4bdaef58253112645e5d484d7c4c97dac61c592be840cd3cfc118f47e469` | OK |
| 44 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_curr_2.env.json` | 682 | `852fdc41fd2a0ef0b333ba1276fea9d09deb89498a2a57843be3f8dc4e8e8d54` | OK |
| 45 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_curr_2.log` | 121035 | `2c17f69c32ba68524e51b43d29956123cc0c9b89d453ca026be0e071a6047976` | OK |
| 46 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_curr_3.env.json` | 703 | `631a6ee935f9da1f472c53a4e2f81542a34f15cccfe77a135bbbfc91e908a7c3` | OK |
| 47 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c1_curr_3.log` | 121141 | `3d8e7fe77ae4b12d59c0ce95308d813e44f8b3f012ab7b6b67a2583fee124da2` | OK |
| 48 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_mask4_1.env.json` | 680 | `e3dc991e80153228ad5fa1863cf15b6dc467f491d172b5d7002c18c306c72f9a` | OK |
| 49 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_mask4_1.log` | 121081 | `138291162d174f6b0639179538138acf1d2b5fd29046c15cac3512eae9d6795e` | OK |
| 50 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_mask4_2.env.json` | 679 | `b8e123413a6dc28cb75638ecd9688b4b489b87ce59d78675fcd6509551c6d86a` | OK |
| 51 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_mask4_2.log` | 121179 | `457a86352174b85aa2427b145bd8b44067c311ba749cc4d2a0f1a29d36c509e4` | OK |
| 52 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_mask4_3.env.json` | 679 | `9a7e3e9c41ee594218ed90dc154a0ec9af85038274a9bd9f72ef6d3b2cd92099` | OK |
| 53 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_mask4_3.log` | 120923 | `5ea83ceb284fa823067aa728436506dce2b48623a916a683f5e6d676cea351e1` | OK |
| 54 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_maskC_1.env.json` | 681 | `2e502a6a5929ad0bd98f98623f088a0a6162368d294cd0a68fcc5adebf87f2c0` | OK |
| 55 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_maskC_1.log` | 121180 | `773754110ce79100960bdf4d1ef8231f73919c033087dd77161c15d98b989d90` | OK |
| 56 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_maskC_2.env.json` | 679 | `17f5899f6d2c37fee8a78aa7b96a2e9645527ce9e9658060592d2b801589fdbe` | OK |
| 57 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_maskC_2.log` | 121174 | `2f3253e736d998a009a0cf987acd306b1f18c567e15e91d27aba677d523a14e7` | OK |
| 58 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_maskC_3.env.json` | 679 | `a0c59c23500ed901f1cd737967c78408c2801657eb36134b9573a85a9bbb2f7b` | OK |
| 59 | `/ConvoPeq-stage3-20261026/investigation-20261101/raw/c2_maskC_3.log` | 120923 | `f39837d34040269a4c8264222203843b35d852d2e322abf02338a818ab5e3290` | OK |
| 60 | `/ConvoPeq.md` | 6436359 | `f8aab49b49dccd9134ec8a4e777a6ddbfe9ed7c3be7f8a923160049239fa2f08` | OK |

## 期待外ファイル

配信ルートには過去フェーズの成果物も同居している（本検証の対象外）。
件数のみ記載し、先頭 10 件を例示する。

- `/ConvoPeq-stage3-20261026/A-CLOSURE-READER-SAFETY-20261028.md`
- `/ConvoPeq-stage3-20261026/A2-DESIGN-REVIEW-20261027.md`
- `/ConvoPeq-stage3-20261026/A2-DESIGN-REVISION-20261027.md`
- `/ConvoPeq-stage3-20261026/A2-PROBE-IMPLEMENTATION-20261027.md`
- `/ConvoPeq-stage3-20261026/AGENTS-MD-SYNC-20261028.patch`
- `/ConvoPeq-stage3-20261026/B2-C-STATIC-PROOF-20261027.md`
- `/ConvoPeq-stage3-20261026/B2-CAPTURE-DESIGN-20261028.md`
- `/ConvoPeq-stage3-20261026/B2-STAGE1-FEASIBILITY-20261028.md`
- `/ConvoPeq-stage3-20261026/B2-T1-IMPLEMENTATION-PLAN-20261028.md`
- `/ConvoPeq-stage3-20261026/B2-T1-IMPLEMENTATION-PLAN-V2-20261028.md`
- … 他 442 件

## 判定

**全件一致（内容で検証）**
