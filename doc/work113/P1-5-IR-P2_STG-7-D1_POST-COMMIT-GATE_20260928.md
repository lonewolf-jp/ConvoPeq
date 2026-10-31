# STG-7-D1 Post-Commit Gate（2026-09-28）— CLOSED

## A. Commit identity
- `git log -1 --oneline`: `29dbdca9 fix(audioengine): restore auto gain staging state`
- `git show --stat --oneline HEAD`: 3 files のみ
  (`ConvoPeq.md` +78 / `src/audioengine/AudioEngine.StateIO.cpp` +6 /
  `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` +70)。
- `git status`: commit 対象外の既存無関係差分（`AGENTS.md` / `headroom-proxy-start.ps1`）は
  unstaged のまま残存・非接触。work113 記録は untracked のまま（運用ルール通り非 commit）。

## B. Authority regeneration（commit 後 HEAD から再生成・流用なし）
- SHA-256: `955582FBF3ABBEE532E67EF908DF083D807F34DE550D85EA89A76FA93DB0732C`
- size: 5,589,694 B / Generated: 2026-09-28 14:33:45 / HEAD: `29dbdca9`
- `--check`: FRESH / NEWER_SRC_COUNT: 0
- commit 直後の working-tree 差分は再生成 timestamp の 2 行のみ（想定内、非 commit）。

## C. Regression（commit 後・再確認）
- CTest（`ctest -C Release`）: **42/42 PASS**（Total 94.45s、`AudioEngineHarness` 77.92s 含む）。
- Release harness: STG-7-D1 / STG-1 / STG-2（3/3＋A/B/C＋R）/ STG-4-1 /
  STG-6-D1（`seq=5->5`）全 PASS。
- Debug harness（commit 後再実行）: 同上すべて PASS、
  `AudioEngineHarness: all publish pipeline tests PASS`。`FAIL:` なし。

## D. Worktree
- `AGENTS.md` / `headroom-proxy-start.ps1` の既存変更は混入なし・破壊なし。
- push: NOT AUTHORIZED（継続）。

## E. Verdict
**STG-7-D1 CLOSED。**
