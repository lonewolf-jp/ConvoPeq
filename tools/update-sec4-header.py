# -*- coding: utf-8 -*-
"""
INTEGRATED_BUG_FIX.md の §4 実装順序と最終更新メタ情報を今回の確定結果と整合させる。

- フェーズ 0: 行 7 に 2-2 / 2-5 を追加（今回実施確定に昇格）
- フェーズ 1: 2-5 行を削除（フェーズ 0 へ移動）
- フェーズ 2: 2-2 行を削除（フェーズ 0 へ移動）、R-新規B / R-新規C の説明を更新
- 最終更新メタ情報: 今回の確定内容を追記
"""
import sys

PATH = r'c:\VSC_Project\ConvoPeq\doc\work88\big_bug\INTEGRATED_BUG_FIX.md'


def load_lines():
    with open(PATH, 'rb') as f:
        data = f.read()
    return data.decode('utf-8').split('\n')


def save_lines(lines):
    out = '\n'.join(lines)
    with open(PATH, 'wb') as f:
        f.write(out.encode('utf-8'))


def find_line(lines, prefix, start=0, end=None):
    for i, ln in enumerate(lines[start:end] if end else lines[start:], start):
        if ln.rstrip('\r').startswith(prefix):
            return i
    return -1


def main():
    lines = load_lines()

    sec_start = find_line(lines, '## 4. 実装順序')
    sec_end = find_line(lines, '## 5. 検証計画')
    if sec_start < 0 or sec_end < 0:
        print('[FAIL] §4 range not found')
        sys.exit(1)
    print(f'[INFO] §4 range: {sec_start}..{sec_end}')

    # フェーズ 0 の行 7 を更新（2-2 / 2-5 を追加）
    old7 = None
    for i in range(sec_start, sec_end):
        s = lines[i].rstrip('\r')
        if s.startswith('7. Bug 2-3'):
            old7 = i
            lines[i] = ('7. Bug 2-2（DC 後スクラブ）、2-3（テスト矛盾）、2-4（memcpy）、'
                        '2-5（size クランプ）、2-9（saturating）、2-10（enum 検証）\r')
            print(f'[OK] §4 line 7 updated at {i}')
            break
    if old7 is None:
        print('[FAIL] §4 line 7 not found')

    # フェーズ 1 の 2-5 行を削除
    # フェーズ 2 の 2-2 行を削除
    removed = 0
    for prefix in ['13. Bug 2-5', '18. Bug 2-2']:
        idx = find_line(lines, prefix, sec_start, sec_end)
        if idx >= 0:
            del lines[idx]
            removed += 1
            print(f'[OK] §4 removed: {prefix}')
            # 削除により sec_end がずれるため再計算
            sec_end = find_line(lines, '## 5. 検証計画')
        else:
            print(f'[SKIP] §4 not found: {prefix}')
    print(f'[INFO] §4 removed: {removed}')

    # R-新規B / R-新規C の説明を更新
    rb = find_line(lines, '22. R-新規B', sec_start, sec_end)
    if rb >= 0:
        lines[rb] = '22. R-新規B（incremental rebuild 休眠維持 — reset() の retire 化 + 未定義関数スタブ化）\r'
        print('[OK] §4 R-新規B updated')
    rc = find_line(lines, '23. R-新規C', sec_start, sec_end)
    if rc >= 0:
        lines[rc] = '23. R-新規C（runSynchronously を try/catch + retire 化）\r'
        print('[OK] §4 R-新規C updated')

    # 最終更新メタ情報を更新
    hdr_idx = find_line(lines, '**最終更新**')
    if hdr_idx >= 0:
        lines[hdr_idx] = ('**最終更新**: 2026-08-11（別視点調査・サブエージェント 3 系統で全 36 項目を精緻化。'
                          'P0 バグ 6 件の修正コードをソース検証、設計判断バグの前提を確定（1-7 は RT 違反未顕在化で P0→P2、'
                          '1-9 は到達不能で P0→設計判断、2-6 は MSVC 衝突不可能、2-7 は MSVC spinlock 実装、3-1 は同一スレッドで競合なし）。'
                          '最終別視点調査（2026-08-11）で 13 セクションの実装詳細を確定：'
                          '1-1 は pendingOverride 読取 + jlimit Soft/Soft、1-6 は Result{}/void 戻り（return nullptr 誤り修正）、'
                          '2-2 は UltraHighRateDCBlocker 維持 + DC 後スクラブ、2-3 は check(ubDb>0) 追加、2-4 はチャネル単位 memcpy、'
                          '2-5 はクランプ防衛、2-9 は BlockDouble 行番号修正 + intervalMs 第4サイト、'
                          '3-4 は #if AVX2 直後 jassert、3-5 は #include <atomic> 必須 + accumulator 維持、3-7 は alignas(32) 移動、'
                          'R-新規B は休眠維持（Option 1）、R-新規C は runSynchronously 優先）\r')
        print('[OK] header updated')
    else:
        print('[FAIL] header not found')

    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
