# -*- coding: utf-8 -*-
"""
INTEGRATED_BUG_FIX.md の §1.1 分類表 / §2.1 即時改修表 / §2.2 設計判断表を更新する。

2026-08-11 別視点調査で確定した 4 項目の反映:
- 2-2   → 実施確定（入力側 DC 後スクラブ追加）→ §2.1 追加 / §2.2 除去
- 2-5   → 実施確定（クランプ + コメント明記）→ §2.1 追加 / §2.2 除去
- R-新規C → 実施確定（runSynchronously 優先）→ §2.1 追加 / §2.2 除去
- R-新規B → 方針確定（休眠維持 Option 1）→ §2.2 行を更新（§1.1 は設計判断に残す）

CRLF 改行を維持するため、バイナリモードで読み書きする。
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


def find_line(lines, prefix):
    for i, ln in enumerate(lines):
        if ln.rstrip('\r').startswith(prefix):
            return i
    return -1


def main():
    lines = load_lines()

    # ---------- §1.1 分類表 ----------
    old_immediate = '| **即時改修（実施確定）** | 低リスク・明確な修正方針・§10/別視点調査で確定 | 1-1, 1-2(値初期化), 1-3, 1-4(統合NO-OP), 1-5, 1-6, 1-7(リネーム), 1-8(ストリーミング), 1-10(クリア移動), 2-1, 2-3, 2-4, 2-6(堅牢性), 2-9, 2-10, 3-4, 3-5, 3-6(両面), 3-7, 3-9, 3-10, R-9, R-新規A, R-新規D |'
    new_immediate = '| **即時改修（実施確定）** | 低リスク・明確な修正方針・§10/別視点調査で確定 | 1-1, 1-2(値初期化), 1-3, 1-4(統合NO-OP), 1-5, 1-6, 1-7(リネーム), 1-8(ストリーミング), 1-10(クリア移動), 2-1, 2-2, 2-3, 2-4, 2-5, 2-6(堅牢性), 2-9, 2-10, 3-4, 3-5, 3-6(両面), 3-7, 3-9, 3-10, R-9, R-新規A, R-新規C, R-新規D |'
    old_design = '| **設計判断（要精査）** | 修正方針はあるが設計判断が必要（案 A/B/C 等） | 1-9(最小修正), 2-2, 2-5, 2-7(案A/B/C), 3-1(防御的), R-新規B, R-新規C |'
    new_design = '| **設計判断（方針確定）** | 修正方針を確定済み（休眠維持・保留を含む） | 1-9(最小修正), 2-7(案A/B/C), 3-1(防御的), R-新規B(休眠維持) |'

    replaced = 0
    for i, ln in enumerate(lines):
        s = ln.rstrip('\r')
        if s == old_immediate:
            lines[i] = new_immediate + '\r'
            replaced += 1
        elif s == old_design:
            lines[i] = new_design + '\r'
            replaced += 1
    print(f'[§1.1] replaced rows: {replaced}')

    # ---------- §2.1 即時改修表に 2-2 / 2-5 / R-新規C を挿入 ----------
    insert_after = [
        # (アンカー行プレフィックス, 挿入する行)
        ('| 2-3 | ユニットテストの矛盾条件',
         '| 2-2 | 入力 DC ブロッカー NaN/Inf 非対称 | DSPCoreIO.cpp:231,252,305 | Low | 入力側 DC 後スクラブ追加（§3-12）|'),
        ('| 2-4 | strict-aliasing 違反',
         '| 2-5 | LockFreeRingBuffer::size() データ競合 | LockFreeRingBuffer.h:76 | Medium | クランプ + コメント明記（§3-15）|'),
        ('| R-新規D |',
         '| R-新規C | bad_alloc 時リーク（OOM） | LoaderThread.cpp:367 | P3 | runSynchronously を try/catch + retire 化（§3-34）|'),
    ]
    for anchor, new_row in insert_after:
        idx = find_line(lines, anchor)
        if idx < 0:
            print(f'[FAIL] §2.1 anchor not found: {anchor}')
            sys.exit(1)
        # 二重挿入防止: 既に挿入済みか確認
        dup = find_line(lines, '| ' + new_row.split(' | ')[0].lstrip('| '))
        # 挿入行の先頭トークンで重複チェック
        token = new_row.split('|')[1].strip()
        if find_line(lines, f'| {token} |') >= 0:
            print(f'[SKIP] already present: {token}')
        else:
            lines.insert(idx + 1, new_row + '\r')
            print(f'[OK] §2.1 inserted: {token}')

    # ---------- §2.2 設計判断表の行を更新/削除 ----------
    # R-新規B 行を更新（休眠維持の確定）
    rb_idx = find_line(lines, '| R-新規B |')
    if rb_idx >= 0:
        lines[rb_idx] = ('| R-新規B | Incremental rebuild 未接続 | Rebuild.cpp / ConvolverProcessor.h | P2 | '
                         '**確定（2026-08-11）**: 休眠維持（Option 1）+ reset() の retire 化・未定義関数スタブ化 → §3-33 |\r')
        print('[OK] §2.2 R-新規B updated')
    else:
        print('[FAIL] §2.2 R-新規B not found')

    # §2.2 から確定済み（2-2 / 2-5 / R-新規C）の行を削除
    for prefix in ['| 2-2 |', '| 2-5 |', '| R-新規C |']:
        idx = find_line(lines, prefix)
        if idx >= 0:
            # §2.2 表の範囲（"### 2.2" から "## A." まで）のみを対象
            sec_start = find_line(lines, '### 2.2 設計判断')
            sec_end = find_line(lines, '## A. 現状維持')
            if sec_start <= idx < sec_end:
                del lines[idx]
                print(f'[OK] §2.2 removed: {prefix}')
            else:
                print(f'[SKIP] §2.2 {prefix} is outside 2.2 table (idx={idx})')
        else:
            print(f'[SKIP] §2.2 {prefix} not found (already removed)')

    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
