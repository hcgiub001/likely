# -*- coding: utf-8 -*-
"""
likely_api.py — fenlight-friendly adapter for the packed "likely/similar" dataset.

================================================================================
HIGH-LEVEL OVERVIEW (read this first)
================================================================================

What does this module do?
  This module lets the Fenlight Kodi addon answer the question:
    "Given a movie or TV show (identified by its TMDB ID), what other
     movies/TV shows are similar to it?"

  It does this by reading a pre-built binary dataset file (dataset.bin) that
  contains similarity data for hundreds of thousands of titles.  The dataset
  is downloaded once from a remote URL and then cached locally on disk.

How is the data stored?
  The dataset uses a compact binary format (magic bytes 'SIML') that packs
  TMDB IDs and media types into integers.  Each "row" in the dataset maps
  one source title to a list of similar titles.

  Packed integer format:  (tmdb_id << 1) | type_bit
    - type_bit = 0 means "movie"
    - type_bit = 1 means "tv"
  So movie ID 4257 becomes (4257 << 1) | 0 = 8514
  And TV show ID 100  becomes (100 << 1)  | 1 = 201

How does loading work?
  The dataset can be loaded in two ways:
    - 'RAM' mode ('air' internally): read the entire file into memory.
      Fastest queries, but uses more RAM.
    - 'mmap' mode: memory-map the file.  The OS loads pages on demand.
      Uses less RAM, still fast for random access.
    - 'auto' mode: the module checks how much RAM is available and picks
      whichever mode is safest.  This is the default.

How does caching work?
  Query results (which are small lists of integers) are cached in Fenlight's
  main_cache with a version token.  To invalidate all cached results cheaply,
  we simply bump the version token — no need to delete individual entries.

How does the code flow?
  1. Caller imports this module and calls ensure_loaded().
  2. ensure_loaded() downloads dataset.bin if missing, then parses it.
  3. Caller queries with query_likely_packed(), query_likely_pairs(), or
     get_likely_for_addon().
  4. The query functions check main_cache first; on miss they do a fast
     binary search in the parsed dataset and cache the result.

================================================================================
PUBLIC API (what callers should use)
================================================================================

  - ensure_loaded(force=False) -> None
      Make sure the dataset is downloaded and parsed.  Safe to call repeatedly.

  - get_setting_mode() -> str
      Returns the user's persisted preference: 'auto', 'RAM', or 'mmap'.

  - set_setting_mode(mode: str) -> bool
      Save the user's preference for loading mode.

  - get_runtime_mode() -> str
      Returns what mode was actually chosen for this session: 'RAM' or 'mmap'.

  - query_likely_packed(tmdb_id, kind) -> List[int]
      Low-level: returns packed integers (id<<1 | typebit).

  - query_likely_pairs(tmdb_id, kind, timing=False) -> dict
      Mid-level: returns {"count": N, "results": [[id, type_bit], ...]}.

  - get_likely_for_addon(tmdb_id, kind, timing=False) -> dict
      High-level: returns {"results": [{"id": .., "media_type": ..}], "total_results": N}.

  - clear_likely_cache() -> None
      Invalidate all cached query results (cheap version-bump).

  - reload_dataset() -> None
      Close current dataset, bump cache, and reload from disk.

  - close_dataset() -> None
      Release all resources (mmap, file handles).

  - dataset_info() -> dict
      Return status dict for debugging / GUI display.

================================================================================
"""

from __future__ import annotations

import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

# =====================================================================
# SECTION 1: Fenlight Imports
# =====================================================================
# These modules come from the Fenlight Kodi addon.  They provide:
#   - main_cache:     a key/value cache (backed by SQLite) for storing
#                     small pieces of data with optional expiration.
#   - get_setting /
#     set_setting:    read/write persistent addon settings (user prefs).
#   - translate_path: converts Kodi "special://" paths to real filesystem
#                     paths (e.g. special://profile -> /home/user/.kodi/).
#   - kodi_dialog:    show Kodi UI dialogs (not used here, but imported
#                     for potential future use).
#   - kodi_log:       write to the Kodi log file for debugging.
#
# If running outside Kodi (e.g. in tests), you must provide compatible
# shims with the same names.
from caches.main_cache import main_cache
from caches.settings_cache import get_setting, set_setting
from modules.kodi_utils import translate_path, kodi_dialog, kodi_log  # noqa: F401


# =====================================================================
# SECTION 2: Standard Library Imports for Dataset Loading
# =====================================================================
# These are all part of Python's standard library — no pip installs needed.

import struct       # Parse binary data (the dataset header)
import mmap         # Memory-map files for efficient random access
import urllib.request  # Download the dataset if it's not cached locally
import shutil       # copyfileobj for streaming download to disk
import tempfile     # Create temp files for atomic writes
import platform     # Detect OS (Linux/Windows/Android) for memory queries
import ctypes       # Call Windows kernel32 API for memory info
import sys          # Check byte order (we only support little-endian)
from array import array       # Compact typed arrays (like C arrays in Python)
from bisect import bisect_left  # Binary search for fast row lookup


# =====================================================================
# SECTION 3: Constants
# =====================================================================

# --- Dataset binary format ---
# The dataset file starts with a fixed-size header.  This struct format
# describes the header layout:
#   4s = 4-byte magic string ('SIML')
#   B  = 1-byte version number
#   B  = 1-byte endianness flag
#   H  = 2-byte flags field
#   I  = 4-byte R (number of source rows / titles)
#   I  = 4-byte E (total number of similarity edges)
#   I  = 4-byte U (number of unique target titles in remap table)
#   B  = 1-byte lengths encoding (0 = 1-byte lengths, 1 = 2-byte lengths)
#   B  = 1-byte remap_index_width (2, 3, or 4 bytes per value index)
#   H  = 2-byte reserved field
#   I  = 4-byte header CRC checksum
_HEADER_STRUCT = "<4s B B H I I I B B H I"
_HEADER_SIZE = struct.calcsize(_HEADER_STRUCT)  # = 28 bytes

# --- Memory heuristics for 'auto' mode ---
# When the user chooses 'auto', we check how much RAM is available:
#   - If at least 300 MB is free AND there's enough room for the dataset
#     plus a safety margin, we load into RAM for fastest queries.
#   - Otherwise we use mmap to conserve memory.
_AUTO_THRESHOLD_DEFAULT = 300 * 1024 * 1024   # 300 MB minimum free RAM
_SAFETY_MARGIN_MIN = 64 * 1024 * 1024         # Always keep at least 64 MB headroom
_SAFETY_MARGIN_FRAC = 0.05                     # Or 5% of total RAM, whichever is larger


# =====================================================================
# SECTION 4: Memory Detection
# =====================================================================
# To decide between RAM and mmap mode automatically, we need to know how
# much memory is available.  Different platforms report this differently,
# so we try multiple methods in priority order:
#
#   1. Kodi InfoLabel  — Best choice when running inside Kodi.  It's a
#      lightweight C++ call that works on every Kodi platform.
#   2. psutil          — Accurate cross-platform library, but it's an
#      optional dependency (not always installed).
#   3. /proc/meminfo   — Linux/Android kernel interface (file-based).
#   4. GlobalMemoryStatusEx — Windows kernel32 API call.
#
# If ALL methods fail, we return (0, 0, "unknown") and the caller falls
# back to mmap mode (the safer default).


def _parse_kodi_memory_mb(label_value: str) -> int:
    """Parse a Kodi InfoLabel memory string into megabytes.

    Kodi typically returns memory values like:
      '1856 MB'   — most common format
      '1856'      — plain number (assumed MB)
      '1.8 GB'    — less common but possible

    Args:
        label_value: Raw string from xbmc.getInfoLabel(), e.g. '1856 MB'.

    Returns:
        Integer number of megabytes, or 0 if parsing fails.

    Examples:
        _parse_kodi_memory_mb('1856 MB')  -> 1856
        _parse_kodi_memory_mb('1.8 GB')   -> 1843  (1.8 * 1024)
        _parse_kodi_memory_mb('')          -> 0
        _parse_kodi_memory_mb('garbage')   -> 0
    """
    try:
        s = label_value.strip().upper()

        # Determine the unit multiplier.
        # Kodi almost always reports in MB, but we handle GB and KB defensively.
        multiplier = 1  # Default: value is already in MB
        if "GB" in s:
            multiplier = 1024          # 1 GB = 1024 MB
            s = s.replace("GB", "")
        elif "MB" in s:
            s = s.replace("MB", "")    # Already in MB, just strip the unit
        elif "KB" in s:
            multiplier = 1.0 / 1024.0  # 1 KB = 1/1024 MB (very unlikely)
            s = s.replace("KB", "")

        s = s.strip()
        return int(float(s) * multiplier)
    except Exception:
        return 0


def _get_mem_via_kodi() -> Optional[Tuple[int, int, str]]:
    """Query available memory using Kodi's built-in InfoLabel system.

    Kodi exposes live system memory statistics that skins use to show RAM
    usage on-screen.  We read these same values programmatically:

      System.Memory(total)     — total physical RAM   (e.g. '3912 MB')
      System.Memory(available) — available RAM         (preferred, more accurate)
      System.Memory(free)      — free RAM              (fallback)
      System.FreeMemory        — older alias for free  (legacy fallback)

    "Available" memory is preferred over "free" because it includes memory
    used for caches/buffers that the OS can reclaim if needed.  "Free" only
    counts truly unused pages, which underestimates what's actually usable.

    These are lightweight C++ calls internally (no file I/O, no subprocess),
    and they work on every platform Kodi runs on (Linux, Windows, Android,
    macOS, iOS, tvOS).

    Returns:
        Tuple of (available_bytes, total_bytes, source_label) or None if
        we're not running inside Kodi or the labels return empty strings.
    """
    try:
        import xbmc  # Kodi's Python bridge — only available inside Kodi
    except ImportError:
        # Not running inside Kodi (e.g. unit-test environment) — skip.
        return None

    try:
        total_str = xbmc.getInfoLabel("System.Memory(total)")

        # Try "available" first (most accurate), then "free", then legacy alias.
        # "available" accounts for reclaimable cache/buffer memory, giving a
        # more realistic picture of how much RAM we can actually use.
        free_str = xbmc.getInfoLabel("System.Memory(available)")
        if not free_str:
            free_str = xbmc.getInfoLabel("System.Memory(free)")
        if not free_str:
            free_str = xbmc.getInfoLabel("System.FreeMemory")

        if not total_str or not free_str:
            return None

        total_mb = _parse_kodi_memory_mb(total_str)
        free_mb = _parse_kodi_memory_mb(free_str)

        if total_mb <= 0 or free_mb <= 0:
            return None

        # Convert MB → bytes to match the common return signature.
        return (free_mb * 1024 * 1024, total_mb * 1024 * 1024, "kodi/InfoLabel")
    except Exception:
        return None


def _get_mem_via_psutil() -> Optional[Tuple[int, int, str]]:
    """Query available memory using the psutil library (if installed).

    psutil is a third-party library that provides cross-platform system
    monitoring.  It's not bundled with Kodi, but some users may have it.

    Returns:
        Tuple of (available_bytes, total_bytes, 'psutil') or None.
    """
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), "psutil"
    except Exception:
        return None


def _get_mem_via_proc_meminfo() -> Optional[Tuple[int, int, str]]:
    """Query available memory by reading /proc/meminfo (Linux/Android only).

    The Linux kernel exposes memory statistics as a virtual text file.
    Key fields:
      MemTotal:     Total physical RAM
      MemAvailable: Best estimate of available RAM (kernel 3.14+)
      MemFree:      Truly free pages (older fallback)
      Cached:       Page cache (reclaimable)
      Buffers:      Buffer cache (reclaimable)

    If MemAvailable is present (modern kernels), we use it directly.
    Otherwise we estimate: available ≈ (free + cached + buffers) * 0.7

    Returns:
        Tuple of (available_bytes, total_bytes, '/proc/meminfo') or None.
    """
    try:
        if not os.path.exists("/proc/meminfo"):
            return None

        # Parse all key:value pairs from /proc/meminfo.
        # Each line looks like: "MemTotal:        3912456 kB"
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]  # Take just the number
                try:
                    info[key] = int(val)  # Values are in kB
                except Exception:
                    pass

        # Prefer MemAvailable (accurate, includes reclaimable memory).
        if "MemAvailable" in info:
            avail = info["MemAvailable"] * 1024  # kB -> bytes
        else:
            # Conservative fallback for older kernels without MemAvailable.
            free = info.get("MemFree", 0)
            cached = info.get("Cached", 0)
            buffers = info.get("Buffers", 0)
            avail = int((free + cached + buffers) * 1024 * 0.7)

        total = info.get("MemTotal", 0) * 1024  # kB -> bytes
        return int(avail), int(total), "/proc/meminfo"
    except Exception:
        return None


def _get_mem_via_windows() -> Optional[Tuple[int, int, str]]:
    """Query available memory using the Windows kernel32 API.

    Uses the GlobalMemoryStatusEx function via ctypes to read physical
    memory statistics without any third-party dependencies.

    Returns:
        Tuple of (available_bytes, total_bytes, 'GlobalMemoryStatusEx') or None.
    """
    try:
        # Define the MEMORYSTATUSEX structure that Windows expects.
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys), int(stat.ullTotalPhys), "GlobalMemoryStatusEx"
        return None
    except Exception:
        return None


def _get_available_memory() -> Tuple[int, int, str]:
    """Detect available system memory using the best available method.

    Tries detection sources in priority order:
      1. Kodi InfoLabel  — preferred; zero-cost inside the running Kodi process
      2. psutil          — accurate cross-platform library (optional install)
      3. /proc/meminfo   — direct kernel interface (Linux / Android only)
      4. GlobalMemoryStatusEx — Win32 API (Windows only)

    Returns:
        Tuple of (available_bytes, total_bytes, source_description).
        Returns (0, 0, 'unknown') if every method fails.
    """
    # 1. Kodi's own InfoLabel system — the most natural choice when running
    #    inside Kodi.  It delegates to the same native code that drives the
    #    skin memory widgets, so it's fast and always matches what the user
    #    sees on screen.
    r = _get_mem_via_kodi()
    if r:
        return r

    # 2. psutil (third-party, may not be installed)
    r = _get_mem_via_psutil()
    if r:
        return r

    # 3–4. OS-specific fallbacks
    sysname = platform.system().lower()
    if sysname in ("linux", "android"):
        r = _get_mem_via_proc_meminfo()
        if r:
            return r
    elif sysname == "windows":
        r = _get_mem_via_windows()
        if r:
            return r

    # All methods failed.  The caller (auto mode logic) will default to mmap.
    return 0, 0, "unknown"


def _compute_safety_margin(total_bytes: int) -> int:
    """Calculate how much RAM headroom to keep when deciding on RAM mode.

    We don't want to load the dataset into RAM if doing so would leave the
    system dangerously low on memory.  The safety margin is:
      max(64 MB, 5% of total RAM)

    For example, on a 2 GB device: max(64MB, 102MB) = 102 MB margin.
    On a 512 MB device: max(64MB, 25MB) = 64 MB margin.
    """
    return max(_SAFETY_MARGIN_MIN, int(total_bytes * _SAFETY_MARGIN_FRAC))


# =====================================================================
# SECTION 5: Packing / Unpacking Helpers
# =====================================================================
# The dataset uses a compact integer representation for (tmdb_id, media_type)
# pairs.  This avoids storing strings and makes comparisons very fast.


def _packed_value(tmdb_id: int, kind: str) -> int:
    """Pack a (tmdb_id, media_type) pair into a single integer.

    Format: (tmdb_id << 1) | type_bit
      - type_bit = 0 for movies
      - type_bit = 1 for TV shows

    Args:
        tmdb_id: The TMDB numeric ID (e.g. 4257).
        kind: Media type string — 'movie' or 'tv' (or anything starting with 'tv').

    Returns:
        Single integer encoding both the ID and the type.

    Examples:
        _packed_value(4257, 'movie')  ->  8514   (4257 << 1 | 0)
        _packed_value(100,  'tv')     ->  201    (100  << 1 | 1)
    """
    return (int(tmdb_id) << 1) | (1 if str(kind).lower().startswith("tv") else 0)


# =====================================================================
# SECTION 6: File Download Helper
# =====================================================================


def _atomic_write_temp(target_path: str, data_stream) -> None:
    """Download/write data to disk safely using atomic replacement.

    Why atomic?  If the download is interrupted (power loss, crash, network
    error), we don't want a half-written dataset.bin on disk.  Instead:
      1. Write to a temporary file in the same directory.
      2. Once writing is complete, atomically rename it to the target path.
      3. If anything goes wrong, clean up the temp file.

    os.replace() is atomic on all modern operating systems — the target
    file either has the old content or the new content, never partial.

    Args:
        target_path: Final destination path for the file.
        data_stream: File-like object to read from (e.g. HTTP response).
    """
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    # Create temp file in the SAME directory as the target.
    # This is important because os.replace() requires source and destination
    # to be on the same filesystem.
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_ds_", suffix=".bin")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(data_stream, out_f)
        # Atomic rename: either fully replaces target or fails entirely.
        os.replace(tmp_path, target_path)
    finally:
        # Clean up temp file if it still exists (means the rename failed).
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =====================================================================
# SECTION 7: Dataset Binary Parser
# =====================================================================
# This function reads the binary dataset file and creates efficient
# "views" (slices) into the data for fast querying.
#
# The binary layout after the 28-byte header is:
#
#   [source_keys]  R × 4 bytes   — sorted packed IDs for binary search
#   [offsets]      R × 4 bytes   — byte offset into values blob per row
#                                  (may be absent if flag bit 0 is set)
#   [lengths]      R × 1 or 2 bytes — number of similar items per row
#   [remap_table]  U × 4 bytes   — maps value indices → packed TMDB IDs
#   [values_blob]  remaining     — indices into remap_table, packed at
#                                  remap_index_width bytes each
#
# Query flow:
#   1. Binary search source_keys for the query's packed ID → row index
#   2. Read offset[row] and length[row] to find the slice in values_blob
#   3. For each value index in the slice, look up remap_table[index]
#      to get the packed TMDB ID of a similar title


def _parse_dataset_file(buf) -> Dict[str, Any]:
    """Parse dataset binary buffer into structured views for querying.

    This function does NOT copy the data (zero-copy where possible).
    It creates memoryview slices that point directly into the buffer,
    which is very memory-efficient.

    Args:
        buf: Either a bytearray (RAM mode) or mmap object (mmap mode).

    Returns:
        Dict containing header fields and a 'views' sub-dict with:
          - source_keys:  memoryview of uint32 — sorted packed source IDs
          - offsets:      memoryview/array of uint32 — per-row offsets
          - lengths:      memoryview of uint8 or uint16 — per-row lengths
          - remap_table:  memoryview of uint32 — index→packed_id mapping
          - values_blob:  memoryview of bytes — raw value indices

    Raises:
        RuntimeError: If running on a big-endian platform (not supported).
        ValueError: If the file is too small or has wrong magic bytes.
    """
    # The dataset is stored in little-endian format.  Big-endian platforms
    # would need byte-swapping, which we don't implement.
    if sys.byteorder != "little":
        raise RuntimeError("Big-endian platform not supported.")

    # Get total buffer size (mmap has .size(), bytearray has len()).
    try:
        size = buf.size()  # mmap object
    except Exception:
        size = len(buf)    # bytearray

    if size < _HEADER_SIZE:
        raise ValueError("Buffer too small")

    # Create a memoryview for zero-copy slicing.
    full_mv = memoryview(buf)

    # --- Unpack the 28-byte header ---
    (
        magic,               # b'SIML' — identifies this as our dataset format
        version,             # Format version number
        endian,              # Endianness flag (always little-endian)
        flags,               # Bit flags (bit 0: offsets omitted if set)
        R,                   # Number of source rows (titles with similar lists)
        E,                   # Total number of edges (sum of all list lengths)
        U,                   # Number of unique target titles in remap table
        lengths_byte,        # 0 = 1-byte lengths, 1 = 2-byte lengths
        remap_index_width,   # Bytes per value index (2, 3, or 4)
        reserved,            # Reserved for future use
        header_crc,          # CRC32 checksum of header (not validated here)
    ) = struct.unpack_from(_HEADER_STRUCT, full_mv, 0)

    if magic != b"SIML":
        raise ValueError("Bad magic (not SIML)")

    # --- Walk through the sections after the header ---
    pos = _HEADER_SIZE

    # SECTION: source_keys — R entries × 4 bytes each (uint32)
    # These are the packed IDs of all source titles, sorted for binary search.
    source_keys_bytes = R * 4
    source_keys_off = pos
    pos += source_keys_bytes

    # SECTION: offsets (optional) — R entries × 4 bytes each (uint32)
    # If flag bit 0 is set, offsets are omitted and must be reconstructed
    # from cumulative lengths.
    offsets_present = not bool(flags & 1)
    offsets_bytes = R * 4 if offsets_present else 0
    offsets_off = pos if offsets_present else None
    pos += offsets_bytes

    # SECTION: lengths — R entries × 1 or 2 bytes each
    # Each length says how many similar titles that source row has.
    if lengths_byte == 0:
        lengths_type = 0           # 1 byte per length (max 255 similar items)
        lengths_bytes = R
    else:
        lengths_type = 1           # 2 bytes per length (max 65535 similar items)
        lengths_bytes = R * 2
    lengths_off = pos
    pos += lengths_bytes

    # SECTION: remap_table — U entries × 4 bytes each (uint32)
    # Maps compact value indices (used in values_blob) to full packed TMDB IDs.
    remap_bytes = U * 4
    remap_off = pos
    pos += remap_bytes

    # SECTION: values_blob — everything remaining
    # Contains indices into remap_table, packed at remap_index_width bytes each.
    values_off = pos
    values_bytes = size - pos

    # --- Create typed memoryview slices ---
    # .cast("I") reinterprets raw bytes as unsigned 32-bit integers.
    source_keys_mv = full_mv[source_keys_off: source_keys_off + source_keys_bytes].cast("I")

    lengths_raw_mv = full_mv[lengths_off: lengths_off + lengths_bytes]
    lengths_mv = lengths_raw_mv.cast("B") if lengths_type == 0 else lengths_raw_mv.cast("H")

    if offsets_present:
        offsets_mv = full_mv[offsets_off: offsets_off + offsets_bytes].cast("I")  # type: ignore[index]
    else:
        # Reconstruct offsets by computing a running sum of lengths.
        # If lengths are [3, 5, 2], offsets become [0, 3, 8].
        offsets_arr = array("I")
        cur = 0
        app = offsets_arr.append
        for length in lengths_mv:
            app(cur)
            cur += int(length)
        offsets_mv = offsets_arr

    remap_mv = full_mv[remap_off: remap_off + remap_bytes].cast("I")
    values_mv = full_mv[values_off: values_off + values_bytes]

    return {
        "magic": magic,
        "version": int(version),
        "endian": int(endian),
        "flags": int(flags),
        "R": int(R),
        "E": int(E),
        "U": int(U),
        "lengths_type": int(lengths_type),
        "remap_index_width": int(remap_index_width),
        "offsets_present": bool(offsets_present),
        "views": {
            "source_keys": source_keys_mv,
            "offsets": offsets_mv,
            "lengths": lengths_mv,
            "remap_table": remap_mv,
            "values_blob": values_mv,
        },
    }


# =====================================================================
# SECTION 8: Dataset Query Engine
# =====================================================================
# The Dataset class wraps the parsed data and provides the main query
# method: query_similar_packed().
#
# Internally it performs:
#   1. Pack the query (tmdb_id, kind) into a single integer.
#   2. Binary search the source_keys array to find the row index.
#   3. Use offset[row] and length[row] to locate the values slice.
#   4. Decode each value index and look up the remap table.
#   5. Return the list of packed result IDs.


class Dataset:
    """Fast in-process query engine for the packed similarity dataset.

    This class holds references to the parsed dataset views and provides
    the query_similar_packed() method for looking up similar titles.

    Lifecycle:
      - Created by load_or_fetch() after parsing the dataset file.
      - Used by query_likely_packed() for lookups.
      - Closed by close_dataset() or reload_dataset() to free resources.

    Note on mmap memory management:
      This class intentionally does NOT store the full parsed dict.
      After __init__ extracts the views it needs into slot attributes,
      the parsed dict (and its memoryview references) can be garbage
      collected.  For mmap mode this is critical: _attempt_ram_copy()
      replaces source_keys, offsets, lengths, and remap with Python
      array copies, so the original memoryviews for those sections
      become unreachable once the parsed dict is freed.  This lets
      the OS evict those mmap pages from physical memory, keeping
      only the values_blob pages mapped on demand.
    """

    __slots__ = (
        "_fileobj",        # Open file handle (only for mmap mode; None for RAM mode)
        "_mmap",           # mmap object (only for mmap mode; None for RAM mode)
        "remap_index_width",  # How many bytes per value index (2, 3, or 4)
        "lengths_type",    # 0 = 1-byte lengths, 1 = 2-byte lengths
        "_source_keys",    # Sorted packed source IDs (for binary search)
        "_offsets",        # Per-row offset into values blob
        "_lengths",        # Per-row count of similar items
        "_remap",          # Index → packed TMDB ID mapping
        "_values_blob",    # Raw value indices (remap_index_width bytes each)
        "_values_u16",     # Cached uint16 cast of values_blob (if width=2)
        "_values_u32",     # Cached uint32 cast of values_blob (if width=4)
        "_ram_copy",       # True if index tables were copied into Python arrays
    )

    def __init__(self, fileobj, mm: Optional[mmap.mmap], parsed: Dict[str, Any], ram_copy: bool = True):
        self._fileobj = fileobj
        self._mmap = mm

        self.remap_index_width = int(parsed["remap_index_width"])
        self.lengths_type = int(parsed["lengths_type"])

        # Extract the views (memoryviews or arrays) from the parsed data.
        views = parsed["views"]
        self._source_keys = views["source_keys"]
        self._offsets = views["offsets"]
        self._lengths = views["lengths"]
        self._remap = views["remap_table"]
        self._values_blob = views["values_blob"]

        # These will be set lazily when needed (typed casts of values_blob).
        self._values_u16 = None
        self._values_u32 = None

        # Optionally copy memoryviews into Python arrays.
        # This is useful because:
        #   - It avoids holding a reference to the mmap (allowing it to close).
        #   - array access can be faster than memoryview on some platforms.
        self._ram_copy = False
        if ram_copy:
            self._attempt_ram_copy()

        # Pre-create typed views of the values blob for the most common widths.
        # This avoids re-casting on every query.
        if self.remap_index_width == 2:
            try:
                self._values_u16 = self._values_blob.cast("H")
            except Exception:
                self._values_u16 = None
        elif self.remap_index_width == 4:
            try:
                self._values_u32 = self._values_blob.cast("I")
            except Exception:
                self._values_u32 = None

    def _attempt_ram_copy(self) -> None:
        """Copy index tables from memoryviews into Python array objects.

        This is a best-effort optimization.  If it fails (e.g. not enough
        memory), we silently fall back to using the original memoryviews.
        """
        try:
            self._source_keys = array("I", self._source_keys)
            self._remap = array("I", self._remap)
            self._offsets = array("I", self._offsets)
            if self.lengths_type == 0:
                self._lengths = array("B", self._lengths)
            else:
                self._lengths = array("H", self._lengths)
            self._ram_copy = True
        except Exception:
            self._ram_copy = False

    def close(self) -> None:
        """Release mmap and file handles.  Safe to call multiple times."""
        try:
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None
        finally:
            if self._fileobj is not None:
                try:
                    self._fileobj.close()
                except Exception:
                    pass
                self._fileobj = None

    def _find_row_index(self, packed_src_key: int) -> int:
        """Find the row index for a packed source key using binary search.

        The source_keys array is sorted, so we use bisect_left for O(log N)
        lookup.

        Args:
            packed_src_key: The packed integer (tmdb_id << 1 | type_bit).

        Returns:
            Row index (>= 0) if found, or -1 if not found.
        """
        sk = self._source_keys
        i = bisect_left(sk, packed_src_key)
        if i != len(sk) and sk[i] == packed_src_key:
            return i
        return -1

    def query_similar_packed(self, tmdb_id: int, kind: str) -> List[int]:
        """Look up similar titles and return them as packed integers.

        This is the core query method.  It:
          1. Packs the query (tmdb_id, kind) into a search key.
          2. Binary searches source_keys to find the row.
          3. Reads the row's offset and length to find its data slice.
          4. Decodes each value index and maps it through remap_table.

        Args:
            tmdb_id: TMDB numeric ID of the source title.
            kind: 'movie' or 'tv'.

        Returns:
            List of packed integers (result_tmdb_id << 1 | type_bit),
            in the order stored in the dataset (typically by relevance).
            Empty list if the title is not found in the dataset.
        """
        packed_src = _packed_value(int(tmdb_id), kind)
        idx = self._find_row_index(packed_src)
        if idx < 0:
            return []

        off = int(self._offsets[idx])      # Starting position in values blob
        length = int(self._lengths[idx])   # Number of similar items

        remap = self._remap
        w = self.remap_index_width

        out: List[int] = []
        append = out.append

        # Decode value indices from values_blob based on the index width.
        # Each value index points into remap_table to get the actual packed ID.
        if w == 2:
            # 2-byte indices (uint16) — most common for datasets with < 65536 unique targets
            v = self._values_u16
            if v is None:
                v = self._values_blob.cast("H")
                self._values_u16 = v
            end = off + length
            for j in range(off, end):
                append(remap[v[j]])
        elif w == 4:
            # 4-byte indices (uint32) — for very large datasets
            v = self._values_u32
            if v is None:
                v = self._values_blob.cast("I")
                self._values_u32 = v
            end = off + length
            for j in range(off, end):
                append(remap[v[j]])
        elif w == 3:
            # 3-byte indices — manual decoding (no native 3-byte integer type)
            # Each index is stored as 3 little-endian bytes: [low, mid, high]
            mv = self._values_blob
            b0 = off * 3  # Byte offset (each index is 3 bytes)
            for _ in range(length):
                ridx = mv[b0] | (mv[b0 + 1] << 8) | (mv[b0 + 2] << 16)
                append(remap[ridx])
                b0 += 3
        else:
            raise ValueError("Unsupported remap_index_width")

        return out


# =====================================================================
# SECTION 9: Dataset Loading Orchestrator
# =====================================================================
# This function ties together downloading, parsing, and mode selection.


def load_or_fetch(
    url: str,
    cache_path: str,
    ram_copy: bool = True,
    mode: str = "auto",
    auto_threshold: int = _AUTO_THRESHOLD_DEFAULT,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Download the dataset if missing, then load and parse it.

    This is the main entry point for getting a ready-to-query Dataset object.

    Flow:
      1. Check if cache_path exists on disk.
         - If not, download from url and write atomically.
      2. Determine loading mode (RAM vs mmap).
         - 'auto': check available memory and decide.
         - 'air': force RAM mode.
         - 'mmap': force mmap mode.
      3. Load the file and parse the binary format.
      4. Create and return a Dataset object.

    Args:
        url: Remote URL to download dataset from (only used if file is missing).
        cache_path: Local filesystem path to store/read dataset.bin.
        ram_copy: If True, copy index tables into Python arrays for speed.
        mode: Loading mode — 'auto', 'air' (RAM), or 'mmap'.
        auto_threshold: Minimum free RAM (bytes) before 'auto' chooses RAM mode.

    Returns:
        Tuple of (Dataset, metadata_dict).
        metadata_dict contains:
          - 'from_cache': bool — True if file was already on disk
          - 'size_bytes': int or None — file size
          - 'mode_chosen': 'air' or 'mmap' — what was actually used
    """
    metadata: Dict[str, Any] = {"from_cache": False, "size_bytes": None, "mode_chosen": None}

    # --- Step 1: Ensure the file exists on disk ---
    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={"User-Agent": "likely_api/1.0"})
        with urllib.request.urlopen(req) as resp:
            _atomic_write_temp(cache_path, resp)
        metadata["from_cache"] = False
    else:
        metadata["from_cache"] = True

    # Get file size for memory decision and metadata.
    try:
        size_bytes = os.path.getsize(cache_path)
    except Exception:
        size_bytes = None
    metadata["size_bytes"] = size_bytes

    # --- Step 2: Decide loading mode ---
    chosen_mode = mode if mode in ("auto", "air", "mmap") else "auto"

    if chosen_mode == "auto":
        avail, total, _src = _get_available_memory()
        safety = _compute_safety_margin(total) if total else _SAFETY_MARGIN_MIN
        dataset_need = (size_bytes or 0) + safety

        # Use RAM if: we could detect memory AND there's enough free AND
        # there's enough after accounting for the dataset + safety margin.
        if avail and avail >= auto_threshold and avail >= dataset_need:
            chosen_mode = "air"
        else:
            chosen_mode = "mmap"

    metadata["mode_chosen"] = chosen_mode

    # --- Step 3: Load and parse ---
    if chosen_mode == "air":
        # RAM mode: read entire file into a bytearray in memory.
        with open(cache_path, "rb") as fh:
            data = bytearray(fh.read())
        parsed = _parse_dataset_file(data)
        ds = Dataset(None, None, parsed, ram_copy=ram_copy)
        # Release the parsed dict immediately.  The Dataset has already
        # extracted everything it needs into its own slot attributes.
        # For RAM mode this is a minor cleanup; the bytearray 'data'
        # stays alive through the memoryview references in ds._values_blob.
        del parsed
        return ds, metadata

    # mmap mode: memory-map the file for on-demand page loading.
    f = open(cache_path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    parsed = _parse_dataset_file(mm)
    ds = Dataset(f, mm, parsed, ram_copy=ram_copy)
    # Release the parsed dict immediately so its memoryview references
    # to the mmap can be garbage collected.  After _attempt_ram_copy(),
    # source_keys / offsets / lengths / remap are Python arrays (heap
    # memory), not memoryviews.  Only ds._values_blob still references
    # the mmap — exactly the section we want to page in on demand.
    # Without this del, the parsed dict would keep memoryviews alive
    # for ALL sections, pinning the entire file in physical memory.
    del parsed
    return ds, metadata


# =====================================================================
# SECTION 10: Configuration Constants
# =====================================================================
# These control where files are stored, what cache keys are used, and
# where to download the dataset from.

# Subdirectory under the addon profile cache for our dataset file.
_DEFAULT_CACHE_SUBDIR = "likely"
_DEFAULT_FILENAME = "dataset.bin"

# Fenlight setting keys (stored in addon settings XML).
_SETTING_KEY_MODE = "fenlight.likely.mode"      # 'auto' | 'RAM' | 'mmap'
_SETTING_KEY_ENABLED = "fenlight.likely.enabled"  # Reserved for future use

# main_cache key for the cache version token (used to cheaply invalidate
# all cached query results without deleting them individually).
_CACHE_VERSION_KEY = "likely:cache_version"

# How long cached query results live (in hours) before they expire.
_DEFAULT_EXPIRATION_HOURS = 24

# Remote URL for the dataset file.  Only used if the local file is missing.
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin"


# =====================================================================
# SECTION 11: Module-Level State
# =====================================================================
# These variables track the currently loaded dataset.  They're protected
# by a reentrant lock for thread safety (Kodi can call from multiple threads).

_lock = threading.RLock()                        # Thread safety for all state changes
_dataset: Optional[Dataset] = None               # The loaded Dataset object (or None)
_dataset_meta: Dict[str, Any] = {}               # Metadata from load_or_fetch()
_dataset_id: Optional[str] = None                # Fingerprint of current dataset file
_runtime_mode: Optional[str] = None              # 'RAM' or 'mmap' (what was chosen)


# =====================================================================
# SECTION 12: Path Helpers
# =====================================================================


def _ensure_cache_dir() -> str:
    """Create and return the directory where dataset.bin is stored.

    Uses Kodi's translate_path to resolve the addon profile directory,
    then creates a 'cache/likely/' subdirectory inside it.

    Returns:
        Absolute filesystem path to the cache directory.
    """
    folder = translate_path("special://profile/addon_data/plugin.video.fenlight/")
    data_folder = os.path.join(folder, "cache", _DEFAULT_CACHE_SUBDIR)
    try:
        os.makedirs(data_folder, exist_ok=True)
    except Exception:
        pass
    return data_folder


def _dataset_cache_path() -> str:
    """Return the full path to dataset.bin on disk."""
    return os.path.join(_ensure_cache_dir(), _DEFAULT_FILENAME)


# =====================================================================
# SECTION 13: Cache Version Management
# =====================================================================
# Instead of deleting cached query results one by one, we use a "version
# token" in the cache key.  When we want to invalidate everything, we
# just change the token — all old keys become orphans that expire naturally.


def _get_cache_version() -> str:
    """Get the current cache version token, creating one if needed.

    The token is a timestamp string stored in main_cache.  It's included
    in every query cache key, so changing it effectively invalidates all
    previously cached results.

    Returns:
        String token (e.g. '1700000000').
    """
    v = main_cache.get(_CACHE_VERSION_KEY)
    if v is None:
        v = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, v, expiration=0)  # Never expires
        except Exception:
            pass
    return str(v)


def clear_likely_cache() -> None:
    """Invalidate all cached query results by bumping the version token.

    This is very cheap — it writes one small value to main_cache.
    All old cached entries become unreachable (wrong version in key)
    and will be cleaned up by the cache's normal expiration mechanism.
    """
    with _lock:
        newv = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, newv, expiration=0)
        except Exception:
            try:
                main_cache.delete(_CACHE_VERSION_KEY)
            except Exception:
                pass


# =====================================================================
# SECTION 14: Dataset Identity
# =====================================================================


def _compute_dataset_id(path: str) -> str:
    """Compute a lightweight fingerprint of the dataset file.

    Uses file modification time and size — fast and deterministic.
    This is included in cache keys so that if the dataset file changes
    (e.g. updated version), old cached results won't be reused.

    Args:
        path: Filesystem path to dataset.bin.

    Returns:
        String like '1700000000:45678901' (mtime:size).
    """
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{int(st.st_size)}"
    except Exception:
        return str(int(time.time()))


# =====================================================================
# SECTION 15: User Settings
# =====================================================================
# The user can choose their preferred loading mode in Fenlight settings.
# These functions read/write that preference.


def get_setting_mode() -> str:
    """Read the user's persisted loading mode preference.

    Returns:
        One of 'auto', 'RAM', or 'mmap'.
    """
    v = get_setting(_SETTING_KEY_MODE, "auto")
    v = (v or "auto").strip()
    if v.lower() == "ram":
        return "RAM"
    if v.lower() == "mmap":
        return "mmap"
    return "auto"


def set_setting_mode(mode: str) -> bool:
    """Save the user's loading mode preference.

    Args:
        mode: 'auto', 'RAM', or 'mmap' (case-insensitive).

    Returns:
        True if saved successfully, False if invalid mode or save failed.
    """
    m = (mode or "auto").strip()
    if m.lower() not in ("auto", "ram", "mmap"):
        return False

    store = "RAM" if m.lower() == "ram" else ("mmap" if m.lower() == "mmap" else "auto")
    try:
        set_setting(_SETTING_KEY_MODE, store)
        return True
    except Exception:
        return False


def get_runtime_mode() -> Optional[str]:
    """Return the loading mode that was actually used for the current session.

    Returns:
        'RAM', 'mmap', or None if no dataset is loaded yet.
    """
    return _runtime_mode


# =====================================================================
# SECTION 16: Dataset Lifecycle (Load / Reload / Close)
# =====================================================================
# These functions manage the module-level dataset state.


def ensure_loaded(url: Optional[str] = None, mode: Optional[str] = None, force: bool = False) -> None:
    """Make sure the dataset is downloaded, parsed, and ready for queries.

    This is safe to call multiple times — if the dataset is already loaded
    and force=False, it returns immediately (no-op).

    Flow:
      1. Acquire the thread lock.
      2. If already loaded and not forcing, return early.
      3. Determine the loading mode:
           - If 'mode' param is given, use that.
           - Otherwise, read the user's persisted setting.
           - Default is 'auto'.
      4. Map user-facing mode names to internal names:
           'auto' → 'auto', 'RAM' → 'air', 'mmap' → 'mmap'
      5. Call load_or_fetch() to download (if needed) and parse.
      6. Store the Dataset object and metadata in module-level state.

    Args:
        url: Override the default dataset download URL.
        mode: Override the loading mode ('auto', 'RAM', 'mmap').
        force: If True, reload even if already loaded.
    """
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    with _lock:
        # Skip if already loaded (unless forcing a reload).
        if _dataset is not None and not force:
            return

        dataset_url = url or DEFAULT_DATASET_URL
        cache_path = _dataset_cache_path()

        # Decide mode: explicit param > persisted setting > default 'auto'.
        mode_setting = get_setting_mode() if mode is None else mode

        # Map user-facing mode names to internal loader mode names.
        # The loader uses 'air' for RAM mode (historical naming).
        loader_mode = "auto" if mode_setting == "auto" else ("air" if mode_setting == "RAM" else "mmap")

        # Download (if needed) and parse the dataset.
        ds, meta = load_or_fetch(dataset_url, cache_path, ram_copy=True, mode=loader_mode)

        # Store in module-level state.
        _dataset = ds
        _dataset_meta = meta or {}
        _dataset_id = _compute_dataset_id(cache_path)

        # Record what mode was actually chosen (for get_runtime_mode()).
        chosen = meta.get("mode_chosen") if isinstance(meta, dict) else None
        if chosen == "air":
            _runtime_mode = "RAM"
        elif chosen == "mmap":
            _runtime_mode = "mmap"
        else:
            _runtime_mode = "RAM" if loader_mode == "air" else "mmap"


def reload_dataset(url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Force a full reload of the dataset.

    This closes the current dataset, bumps the cache version (invalidating
    all cached query results), and loads the dataset fresh from disk.

    Use this after updating the dataset file, or if you suspect corruption.

    Args:
        url: Override the default dataset download URL.
        mode: Override the loading mode ('auto', 'RAM', 'mmap').
    """
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    with _lock:
        # Close existing dataset (release mmap/file handles).
        if _dataset is not None:
            try:
                _dataset.close()
            except Exception:
                pass

        # Reset all module state.
        _dataset = None
        _dataset_meta = {}
        _dataset_id = None
        _runtime_mode = None

        # Invalidate cached query results.
        clear_likely_cache()

        # Load fresh.
        ensure_loaded(url=url, mode=mode, force=True)


def close_dataset() -> None:
    """Close the dataset and release all resources (mmap, file handles).

    After calling this, queries will fail until ensure_loaded() is called again.
    """
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    with _lock:
        if _dataset is not None:
            try:
                _dataset.close()
            except Exception:
                pass
        _dataset = None
        _dataset_meta = {}
        _dataset_id = None
        _runtime_mode = None


# =====================================================================
# SECTION 17: Query Functions
# =====================================================================
# These are the main functions callers use to look up similar titles.
# They go from lowest-level (packed ints) to highest-level (dicts).


def _packed_cache_key(tmdb_id: int, kind: str) -> str:
    """Build a cache key for a query result.

    The key includes:
      - The cache version token (so bumping it invalidates everything)
      - The dataset file fingerprint (so swapping datasets invalidates too)
      - The query parameters (tmdb_id and kind)

    Example key: 'likely:packed:1700000000:1700000000:45678901:4257:movie'
    """
    v = _get_cache_version()
    did = _dataset_id or "nodata"
    return f"likely:packed:{v}:{did}:{tmdb_id}:{kind}"


def query_likely_packed(
    tmdb_id: int,
    kind: str,
    use_cache: bool = True,
    expiration_hours: int = _DEFAULT_EXPIRATION_HOURS,
) -> List[int]:
    """Query the dataset for similar titles, returning packed integers.

    This is the lowest-level public query function.  It returns raw packed
    integers (tmdb_id << 1 | type_bit) which are compact and fast to process.

    Flow:
      1. Ensure the dataset is loaded.
      2. Check main_cache for a cached result.
      3. On cache miss, do a binary search in the dataset.
      4. Cache the result for future calls.
      5. Return the list of packed integers.

    Args:
        tmdb_id: TMDB numeric ID of the source title (e.g. 4257).
        kind: Media type — 'movie' or 'tv'.
        use_cache: If True (default), check/populate the result cache.
        expiration_hours: How long to cache results (default 24 hours).

    Returns:
        List of packed integers, e.g. [8514, 201, 13772, ...].
        Empty list if the title is not in the dataset.
    """
    # Auto-load if not yet loaded (convenience for callers who forget).
    if _dataset is None:
        ensure_loaded()

    # Check cache first.
    key = _packed_cache_key(tmdb_id, kind)
    if use_cache:
        try:
            cached = main_cache.get(key)
            if cached is not None:
                return list(cached)
        except Exception:
            pass

    # Cache miss — do the actual dataset lookup.
    packed = _dataset.query_similar_packed(tmdb_id, kind)  # type: ignore[union-attr]

    # Store in cache for next time.
    try:
        main_cache.set(key, packed, expiration=expiration_hours)
    except Exception:
        pass

    return packed


def query_likely_pairs(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """Query for similar titles, returning a compact pairs format.

    This is a mid-level function that unpacks the results into [id, type_bit]
    pairs, which are easier to work with than raw packed integers.

    Args:
        tmdb_id: TMDB numeric ID of the source title.
        kind: 'movie' or 'tv'.
        timing: If True, also return the query time in milliseconds.

    Returns:
        Dict: {"count": N, "results": [[tmdb_id, type_bit], ...]}
        If timing=True: (dict, total_ms)

    Example:
        query_likely_pairs(4257, 'movie')
        -> {"count": 3, "results": [[1234, 0], [5678, 1], [9012, 0]]}
        # 0 = movie, 1 = tv
    """
    t0 = time.perf_counter()
    packed = query_likely_packed(tmdb_id, kind)
    res = [[(pv >> 1), (pv & 1)] for pv in packed]
    out = {"count": len(res), "results": res}
    total_ms = (time.perf_counter() - t0) * 1000.0
    return (out, total_ms) if timing else out


def get_likely_for_addon(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """Query for similar titles, returning an addon-friendly dict format.

    This is the highest-level function, designed for direct use by addon
    UI code.  Results use named fields ('id', 'media_type') that match
    the TMDB API response format other parts of Fenlight expect.

    Args:
        tmdb_id: TMDB numeric ID of the source title.
        kind: 'movie' or 'tv'.
        timing: If True, also return the query time in milliseconds.

    Returns:
        Dict: {
            "results": [{"id": 1234, "media_type": "movie"}, ...],
            "total_results": N
        }
        If timing=True: (dict, total_ms)

    Example:
        get_likely_for_addon(4257, 'movie')
        -> {
             "results": [
               {"id": 1234, "media_type": "movie"},
               {"id": 5678, "media_type": "tv"},
             ],
             "total_results": 2
           }
    """
    t0 = time.perf_counter()
    pairs = query_likely_packed(tmdb_id, kind)
    results: List[Dict[str, Any]] = []
    for pv in pairs:
        _id = (pv >> 1)
        typebit = (pv & 1)
        results.append({"id": _id, "media_type": "tv" if typebit == 1 else "movie"})

    out = {"results": results, "total_results": len(results)}
    total_ms = (time.perf_counter() - t0) * 1000.0
    return (out, total_ms) if timing else out


# =====================================================================
# SECTION 18: Debugging / Status
# =====================================================================


def dataset_info() -> Dict[str, Any]:
    """Return a status dict describing the currently loaded dataset.

    Useful for debugging, GUI status displays, or health checks.

    Returns:
        Dict with keys:
          - loaded: bool — whether a dataset is currently loaded
          - dataset_id: str or None — file fingerprint (mtime:size)
          - runtime_mode: 'RAM', 'mmap', or None
          - meta: dict — metadata from load_or_fetch()
    """
    return {
        "loaded": _dataset is not None,
        "dataset_id": _dataset_id,
        "runtime_mode": _runtime_mode,
        "meta": _dataset_meta or {},
    }


# =====================================================================
# SECTION 19: Usage Examples
# =====================================================================
#
# Basic usage from another module in the addon:
#
#   from apis import likely_api
#
#   # Step 1: Load the dataset (downloads on first run, then uses cache).
#   likely_api.ensure_loaded()
#
#   # Step 2: Query for movies similar to "The Shawshank Redemption" (TMDB ID 278).
#   result = likely_api.get_likely_for_addon(278, 'movie')
#   print(result)
#   # -> {"results": [{"id": 857, "media_type": "movie"}, ...], "total_results": 42}
#
#   # Step 3 (optional): Check what mode was chosen.
#   print(likely_api.get_runtime_mode())  # -> 'RAM' or 'mmap'
#
#   # Step 4 (optional): Force reload after updating the dataset file.
#   likely_api.reload_dataset()
#
# =====================================================================