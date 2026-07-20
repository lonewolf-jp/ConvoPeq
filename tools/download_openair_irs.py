#!/usr/bin/env python3
"""
OpenAIR Real IR Bulk Downloader
================================
Downloads all available IR ZIP files from OpenAIR (University of York).

Usage:
  python download_openair_irs.py [output_dir]

The script uses webfiles.york.ac.uk direct URLs.
If a URL returns 404, the slug heuristic was wrong and needs manual correction.
"""

import os
import sys
import re
import json
import urllib.request
import zipfile
import wave
from pathlib import Path
from typing import Optional

# ─── IR Registry ────────────────────────────────────────────────────────────
# Based on Browser4 extraction of page_id → title from OpenAIR IR Data page
# Slugs determined by inspecting actual ZIP URLs on each IR's page

IR_REGISTRY = [
    # (title, slug) — slug verified by Browser4 page inspection
    ("1st Baptist Church Nashville", "1st-baptist-nashville"),
    ("Abies Grandis Forest, Wheldrake Wood", "abies-grandis-forest-wheldrake-wood"),
    ("Alcuin College, University of York", "alcuin-college-university-york"),
    ("Arthur Sykes Rymer Auditorium, University of York.", "arthur-sykes-rymer-auditorium-university-york"),
    ("Central Hall, University of York", "central-hall-university-york"),
    ("Clifford's Tower, York", "cliffords-tower-york"),
    ("Creswell Crags", "creswell-crags"),
    ("D/L/028 Hendrix Hall Derwent, University of York", "hendrix-hall-derwent-university-york"),
    ("Elveden Hall (Suffolk England)", "elveden-hall-suffolk-england"),
    ("Falkland Palace Bottle Dungeon", "falkland-palace-bottle-dungeon"),
    ("Falkland Palace Royal Tennis Court", "falkland-palace-royal-tennis-court"),
    ("Forest Scale Model", "forest-scale-model"),
    ("Genesis 6 Studio - Live Room Drum Set Up", "genesis-6-studio-live-room-drum-set-up"),
    ("Gill Heads Mine", "gill-heads-mine"),
    ("Hamilton Mausoleum", "hamilton-mausoleum"),
    ("Heslington Church", "heslington-church"),
    ("Hoffmann Lime Kiln (Langcliffe, UK)", "hoffmann-lime-kiln-langcliffe-uk"),
    ("Holy Trinity Church, Goodramgate, York", "holy-trinity-church-goodramgate-york"),
    ("Innocent Railway Tunnel", "innocent-railway-tunnel"),
    ("Jack Lyons Concert Hall (University of York)", "jack-lyons-concert-hall-university-york"),
    ("Koli National Park - Summer", "koli-national-park-summer"),
    ("Koli National Park - Winter", "koli-national-park-winter"),
    ("Lady Chapel, St Albans Cathedral", "lady-chapel-st-albans-cathedral"),
    ("Maes Howe", "maes-howe"),
    ("Newgrange", "newgrange"),
    ("P/L/001 Physics, University of York", "physics-university-york"),
    ("R1 Nuclear Reactor Hall", "r1-nuclear-reactor-hall"),
    ("Ron Cooke Hub, University of York", "ron-cooke-hub-university-york"),
    ("Saint Lawrence Church Molenbeek-Wersbeek Belgium", "saint-lawrence-church-molenbeek-wersbeek-belgium"),
    ("Slinky IR", "slinky-ir"),
    ("Spokane Woman's Club", "spokane-womans-club"),
    ("Sports Centre, University of York", "sports-centre-university-york"),
    ("Spring Lane Building, University of York", "spring-lane-building-university-york"),
    ("St Andrew's Church", "st-andrews-church"),
    ("St Matthew's Church - Walsall", "st-matthews-church-walsall"),
    ("St. George's Episcopal Church", "st-georges-episcopal-church"),
    ("St. Margaret's Church - National Centre for Early Music", "st-margarets-church-national-centre-early-music"),
    ("St. Margaret's Church (NCEM) 5 Piece Band Spatial Measurements", "st-margarets-church-ncem-5-piece-band-spatial-measurements"),
    ("St. Mary's Abbey Reconstruction", "st-marys-abbey-reconstruction"),
    ("St. Patrick's Church, Patrington", "st-patricks-church-patrington"),
    ("St. Patrick's Church, Patrington - Model", "st-patricks-church-patrington-model"),
    ("St. Paul's Cathedral", "st-pauls-cathedral"),
    ("Stairway, University of York", "stairway-university-york"),
    ("T2 Hangar, Yorkshire Air Museum", "t2-hangar-yorkshire-air-museum"),
    ("Terry's Factory Warehouse", "terrys-factory-warehouse"),
    ("Terry's Typing Room", "terrys-typing-room"),
    ("The Dixon Studio Theatre - University of York", "the-dixon-studio-theatre-university-york"),
    ("The Shrine and Parish Church of All Saints North Street", "shrine-parish-church-all-saints-north-street"),
    ("Theatre@41, York", "theatre-41-york"),
    ("Troller's Gill", "trollers-gill"),
    ("Tvísöngur Sound Sculpture, Iceland (Model)", "tvisongur-sound-sculpture-iceland-model"),
    ("Tyndall Bruce Monument", "tyndall-bruce-monument"),
    ("Usina del Arte Symphony Hall", "usina-del-arte-symphony-hall"),
    ("Virtual Membranes", "virtual-membranes"),
    ("Waveguide Web Example Audio", "waveguide-web-example-audio"),
    ("York Guildhall Council Chamber", "york-guildhall-council-chamber"),
    ("York Minster", "york-minster"),
]

BASE_URL = "https://webfiles.york.ac.uk/OPENAIR/IRs"

def download_zip(slug: str, output_dir: Path) -> Optional[Path]:
    """Download a ZIP file for the given IR slug."""
    url = f"{BASE_URL}/{slug}/{slug}.zip"
    output_path = output_dir / f"{slug}.zip"

    if output_path.exists():
        print(f"  ✓ {slug}: already downloaded")
        return output_path

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            with open(output_path, 'wb') as f:
                f.write(data)
            size_mb = len(data) / (1024 * 1024)
            print(f"  ✓ {slug}: {size_mb:.1f}MB")
            return output_path
    except urllib.error.HTTPError as e:
        print(f"  ✗ {slug}: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  ✗ {slug}: {e}")
        return None

def extract_wavs(zip_path: Path, extract_dir: Path) -> int:
    """Extract WAV files from a ZIP archive."""
    wav_count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.lower().endswith('.wav'):
                    zf.extract(member, extract_dir)
                    wav_count += 1
    except Exception as e:
        print(f"  ⚠ {zip_path.stem}: extraction error: {e}")
    return wav_count

def scan_wavs(directory: Path) -> list:
    """Scan directory for WAV files and return their info."""
    results = []
    for f in sorted(directory.rglob("*.wav")):
        try:
            with wave.open(str(f), 'r') as w:
                duration = w.getnframes() / w.getframerate()
                results.append((f.name, w.getnframes(), duration, w.getnchannels()))
        except:
            results.append((f.name, 0, 0, 0))
    return results

def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sampledata/real_irs")
    zips_dir = output_dir / "zips"
    wavs_dir = output_dir / "wavs"
    zips_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  OpenAIR Real IR Bulk Downloader")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Total IRs in registry: {len(IR_REGISTRY)}")
    print("=" * 70)

    # Step 1: Download all ZIPs
    print(f"\n[1/2] Downloading ZIP files...")
    successful = 0
    failed = 0
    for title, slug in IR_REGISTRY:
        result = download_zip(slug, zips_dir)
        if result:
            successful += 1
        else:
            failed += 1

    print(f"\n  Download complete: {successful} OK, {failed} failed")

    # Step 2: Extract WAVs
    print(f"\n[2/2] Extracting WAV files...")
    total_wavs = 0
    for zip_path in sorted(zips_dir.glob("*.zip")):
        count = extract_wavs(zip_path, wavs_dir / zip_path.stem)
        if count > 0:
            print(f"  ✓ {zip_path.stem}: {count} WAV files")
            total_wavs += count

    # Step 3: Summary
    wav_files = scan_wavs(wavs_dir)
    print(f"\n{'=' * 70}")
    print(f"  Summary:")
    print(f"  ZIPs downloaded: {successful}/{len(IR_REGISTRY)}")
    print(f"  WAV files extracted: {total_wavs}")
    print(f"  Total WAV size on disk: {sum(f.stat().st_size for f in wavs_dir.rglob('*.wav')) / (1024*1024):.0f}MB")
    print()

    if wav_files:
        print(f"  Sample files (first 10):")
        for name, frames, dur, ch in wav_files[:10]:
            print(f"    {name}: {frames}samples, {dur:.1f}s, {ch}ch")
        if len(wav_files) > 10:
            print(f"    ... and {len(wav_files) - 10} more")

    print(f"\n  Failed downloads (may need slug correction):")
    for title, slug in IR_REGISTRY:
        if not (zips_dir / f"{slug}.zip").exists():
            print(f"    {title} -> {slug}")

    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    main()
