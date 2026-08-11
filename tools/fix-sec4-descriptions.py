# -*- coding: utf-8 -*-
"""§4 実装順序の 1-7 / 1-9 行の説明を確定内容と整合させる。"""
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


def main():
    lines = load_lines()
    updates = {
        '10. Bug 1-7（ISRRetire RT ロックフリー化）':
            '10. Bug 1-7（ISRRetire リネーム + コメント強化）— ビルド + ISRSoakTests + ctest',
        '16. Bug 1-9（fftSize int64_t 化）':
            '16. Bug 1-9（fftSize 型衛生 — 割当 cast + :778 乗算順序・最小修正）',
    }
    updated = 0
    for i, ln in enumerate(lines):
        s = ln.rstrip('\r')
        for old_prefix, new_s in updates.items():
            if s.startswith(old_prefix):
                lines[i] = new_s + '\r'
                updated += 1
                print(f'[OK] updated: {new_s[:30]}...')
                break
    print(f'[INFO] updated: {updated}')
    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
