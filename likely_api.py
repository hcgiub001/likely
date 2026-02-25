# -*- coding: utf-8 -*-
"""
likely_api.py — Fenlight adapter for the packed similarity dataset.

Resolves "given a TMDB ID + media type, what titles are similar?" by reading
a compact binary file (dataset.bin, magic b'SIML') downloaded once and cached.

Packed ID format: (tmdb_id << 1) | type_bit   (0=movie, 1=tv)

Loading modes: 'RAM' (full read), 'mmap' (on-demand pages), 'auto' (pick by avail mem).

Query results are cached in main_cache with a version token for cheap bulk invalidation.

Public API:
  ensure_loaded / reload_dataset / close_dataset
  get_setting_mode / set_setting_mode / get_runtime_mode
  query_likely_packed / query_likely_pairs / get_likely_for_addon
  clear_likely_cache / dataset_info
"""

from __future__ import annotations

import os
import sys
import time
import struct
import mmap
import threading
import urllib.request
import shutil
import tempfile
import platform
import ctypes
from typing import Any, Dict, List, Optional, Tuple
from array import array
from bisect import bisect_left

from caches.main_cache import main_cache
from caches.settings_cache import get_setting, set_setting
from modules.kodi_utils import translate_path, kodi_dialog, kodi_log  # noqa: F401

# Binary header: magic(4s) ver(B) endian(B) flags(H) R(I) E(I) U(I)
#                lengths_enc(B) remap_width(B) reserved(H) crc(I)
_HEADER_STRUCT = "<4s B B H I I I B B H I"
_HEADER_SIZE = struct.calcsize(_HEADER_STRUCT)

# Auto-mode memory thresholds
_AUTO_THRESHOLD_DEFAULT = 300 * 1024 * 1024
_SAFETY_MARGIN_MIN = 64 * 1024 * 1024
_SAFETY_MARGIN_FRAC = 0.05


# --------------- Memory detection (multi-platform) ---------------

def _parse_kodi_memory_mb(label_value: str) -> int:
    """Parse a Kodi InfoLabel memory string (e.g. '1856 MB') into MB."""
    try:
        s = label_value.strip().upper()
        multiplier = 1
        if "GB" in s:
            multiplier = 1024
            s = s.replace("GB", "")
        elif "MB" in s:
            s = s.replace("MB", "")
        elif "KB" in s:
            multiplier = 1.0 / 1024.0
            s = s.replace("KB", "")
        return int(float(s.strip()) * multiplier)
    except Exception:
        return 0


def _get_mem_via_kodi() -> Optional[Tuple[int, int, str]]:
    """Query memory via Kodi InfoLabels (preferred inside Kodi)."""
    try:
        import xbmc
    except ImportError:
        return None
    try:
        total_str = xbmc.getInfoLabel("System.Memory(total)")
        free_str = (xbmc.getInfoLabel("System.Memory(available)")
                    or xbmc.getInfoLabel("System.Memory(free)")
                    or xbmc.getInfoLabel("System.FreeMemory"))
        if not total_str or not free_str:
            return None
        total_mb = _parse_kodi_memory_mb(total_str)
        free_mb = _parse_kodi_memory_mb(free_str)
        if total_mb <= 0 or free_mb <= 0:
            return None
        return (free_mb * 1024 * 1024, total_mb * 1024 * 1024, "kodi/InfoLabel")
    except Exception:
        return None


def _get_mem_via_psutil() -> Optional[Tuple[int, int, str]]:
    """Query memory via psutil (if installed)."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), "psutil"
    except Exception:
        return None


def _get_mem_via_proc_meminfo() -> Optional[Tuple[int, int, str]]:
    """Query memory from /proc/meminfo (Linux/Android)."""
    try:
        if not os.path.exists("/proc/meminfo"):
            return None
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                try:
                    info[key] = int(val)
                except Exception:
                    pass
        if "MemAvailable" in info:
            avail = info["MemAvailable"] * 1024
        else:
            avail = int((info.get("MemFree", 0) + info.get("Cached", 0)
                         + info.get("Buffers", 0)) * 1024 * 0.7)
        total = info.get("MemTotal", 0) * 1024
        return int(avail), int(total), "/proc/meminfo"
    except Exception:
        return None


def _get_mem_via_windows() -> Optional[Tuple[int, int, str]]:
    """Query memory via Win32 GlobalMemoryStatusEx."""
    try:
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
    """Return (avail_bytes, total_bytes, source). Falls back to (0,0,'unknown')."""
    for fn in (_get_mem_via_kodi, _get_mem_via_psutil):
        r = fn()
        if r:
            return r
    sysname = platform.system().lower()
    if sysname in ("linux", "android"):
        r = _get_mem_via_proc_meminfo()
        if r:
            return r
    elif sysname == "windows":
        r = _get_mem_via_windows()
        if r:
            return r
    return 0, 0, "unknown"


def _compute_safety_margin(total_bytes: int) -> int:
    """RAM headroom: max(64 MB, 5% of total)."""
    return max(_SAFETY_MARGIN_MIN, int(total_bytes * _SAFETY_MARGIN_FRAC))


# --------------- Packing helpers ---------------

def _packed_value(tmdb_id: int, kind: str) -> int:
    """Encode (tmdb_id, media_type) as (tmdb_id << 1) | type_bit."""
    return (int(tmdb_id) << 1) | (1 if str(kind).lower().startswith("tv") else 0)


# --------------- Atomic file write ---------------

def _atomic_write_temp(target_path: str, data_stream) -> None:
    """Stream data_stream to target_path via temp-file + atomic rename."""
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_ds_", suffix=".bin")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(data_stream, out_f)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# --------------- Binary dataset parser ---------------

def _parse_dataset_file(buf) -> Dict[str, Any]:
    """Parse SIML binary buffer into typed views for zero-copy querying.

    Layout after 28-byte header:
      source_keys  R×4B (sorted packed IDs)
      offsets      R×4B (optional, controlled by flags bit 0)
      lengths      R×1B or R×2B
      remap_table  U×4B (value index → packed ID)
      values_blob  remainder (indices into remap, remap_index_width bytes each)
    """
    if sys.byteorder != "little":
        raise RuntimeError("Big-endian platform not supported.")

    try:
        size = buf.size()
    except Exception:
        size = len(buf)

    if size < _HEADER_SIZE:
        raise ValueError("Buffer too small")

    full_mv = memoryview(buf)

    (magic, version, endian, flags, R, E, U,
     lengths_byte, remap_index_width, reserved, header_crc,
     ) = struct.unpack_from(_HEADER_STRUCT, full_mv, 0)

    if magic != b"SIML":
        raise ValueError("Bad magic (not SIML)")

    pos = _HEADER_SIZE

    # Source keys (sorted uint32 for binary search)
    source_keys_bytes = R * 4
    source_keys_off = pos
    pos += source_keys_bytes

    # Offsets (optional — reconstructed from lengths if flag bit 0 set)
    offsets_present = not bool(flags & 1)
    offsets_bytes = R * 4 if offsets_present else 0
    offsets_off = pos if offsets_present else None
    pos += offsets_bytes

    # Lengths (1- or 2-byte per row)
    lengths_type = 0 if lengths_byte == 0 else 1
    lengths_bytes = R if lengths_type == 0 else R * 2
    lengths_off = pos
    pos += lengths_bytes

    # Remap table (uint32)
    remap_bytes = U * 4
    remap_off = pos
    pos += remap_bytes

    # Values blob (remainder)
    values_off = pos
    values_bytes = size - pos

    # Build typed memoryview slices
    source_keys_mv = full_mv[source_keys_off: source_keys_off + source_keys_bytes].cast("I")

    lengths_raw_mv = full_mv[lengths_off: lengths_off + lengths_bytes]
    lengths_mv = lengths_raw_mv.cast("B") if lengths_type == 0 else lengths_raw_mv.cast("H")

    if offsets_present:
        offsets_mv = full_mv[offsets_off: offsets_off + offsets_bytes].cast("I")  # type: ignore[index]
    else:
        # Reconstruct from cumulative lengths
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
        "magic": magic, "version": int(version), "endian": int(endian),
        "flags": int(flags), "R": int(R), "E": int(E), "U": int(U),
        "lengths_type": int(lengths_type),
        "remap_index_width": int(remap_index_width),
        "offsets_present": bool(offsets_present),
        "views": {
            "source_keys": source_keys_mv, "offsets": offsets_mv,
            "lengths": lengths_mv, "remap_table": remap_mv,
            "values_blob": values_mv,
        },
    }


# --------------- Query engine ---------------

class Dataset:
    """In-process query engine over the parsed SIML dataset.

    After init, index tables (source_keys, offsets, lengths, remap) are copied
    into Python arrays so the original mmap pages for those sections can be
    evicted. Only values_blob remains mapped on demand.
    """

    __slots__ = (
        "_fileobj", "_mmap", "remap_index_width", "lengths_type",
        "_source_keys", "_offsets", "_lengths", "_remap",
        "_values_blob", "_values_u16", "_values_u32", "_ram_copy",
    )

    def __init__(self, fileobj, mm: Optional[mmap.mmap], parsed: Dict[str, Any], ram_copy: bool = True):
        self._fileobj = fileobj
        self._mmap = mm
        self.remap_index_width = int(parsed["remap_index_width"])
        self.lengths_type = int(parsed["lengths_type"])

        views = parsed["views"]
        self._source_keys = views["source_keys"]
        self._offsets = views["offsets"]
        self._lengths = views["lengths"]
        self._remap = views["remap_table"]
        self._values_blob = views["values_blob"]
        self._values_u16 = None
        self._values_u32 = None
        self._ram_copy = False

        if ram_copy:
            self._attempt_ram_copy()

        # Pre-cast values blob for the common widths
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
        """Best-effort copy of index tables from memoryview into Python arrays."""
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
        """Release mmap and file handles. Safe to call multiple times."""
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
        """Binary-search source_keys. Returns row index or -1."""
        sk = self._source_keys
        i = bisect_left(sk, packed_src_key)
        if i != len(sk) and sk[i] == packed_src_key:
            return i
        return -1

    def query_similar_packed(self, tmdb_id: int, kind: str) -> List[int]:
        """Look up similar titles; returns list of packed ints. Empty if not found."""
        packed_src = _packed_value(int(tmdb_id), kind)
        idx = self._find_row_index(packed_src)
        if idx < 0:
            return []

        off = int(self._offsets[idx])
        length = int(self._lengths[idx])
        remap = self._remap
        w = self.remap_index_width
        out: List[int] = []
        append = out.append

        if w == 2:
            v = self._values_u16
            if v is None:
                v = self._values_blob.cast("H")
                self._values_u16 = v
            end = off + length
            for j in range(off, end):
                append(remap[v[j]])
        elif w == 4:
            v = self._values_u32
            if v is None:
                v = self._values_blob.cast("I")
                self._values_u32 = v
            end = off + length
            for j in range(off, end):
                append(remap[v[j]])
        elif w == 3:
            # Manual 3-byte little-endian decoding
            mv = self._values_blob
            b0 = off * 3
            for _ in range(length):
                ridx = mv[b0] | (mv[b0 + 1] << 8) | (mv[b0 + 2] << 16)
                append(remap[ridx])
                b0 += 3
        else:
            raise ValueError("Unsupported remap_index_width")

        return out


# --------------- Load orchestrator ---------------

def load_or_fetch(
    url: str,
    cache_path: str,
    ram_copy: bool = True,
    mode: str = "auto",
    auto_threshold: int = _AUTO_THRESHOLD_DEFAULT,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Download dataset if missing, then load in chosen mode. Returns (Dataset, meta)."""
    metadata: Dict[str, Any] = {"from_cache": False, "size_bytes": None, "mode_chosen": None}

    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={"User-Agent": "likely_api/1.0"})
        with urllib.request.urlopen(req) as resp:
            _atomic_write_temp(cache_path, resp)
        metadata["from_cache"] = False
    else:
        metadata["from_cache"] = True

    try:
        size_bytes = os.path.getsize(cache_path)
    except Exception:
        size_bytes = None
    metadata["size_bytes"] = size_bytes

    chosen_mode = mode if mode in ("auto", "air", "mmap") else "auto"

    if chosen_mode == "auto":
        avail, total, _src = _get_available_memory()
        safety = _compute_safety_margin(total) if total else _SAFETY_MARGIN_MIN
        dataset_need = (size_bytes or 0) + safety
        if avail and avail >= auto_threshold and avail >= dataset_need:
            chosen_mode = "air"
        else:
            chosen_mode = "mmap"

    metadata["mode_chosen"] = chosen_mode

    if chosen_mode == "air":
        with open(cache_path, "rb") as fh:
            data = bytearray(fh.read())
        parsed = _parse_dataset_file(data)
        ds = Dataset(None, None, parsed, ram_copy=ram_copy)
        del parsed
        return ds, metadata

    # mmap mode
    f = open(cache_path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    parsed = _parse_dataset_file(mm)
    ds = Dataset(f, mm, parsed, ram_copy=ram_copy)
    del parsed
    return ds, metadata


# --------------- Configuration ---------------

_DEFAULT_CACHE_SUBDIR = "likely"
_DEFAULT_FILENAME = "dataset.bin"
_SETTING_KEY_MODE = "fenlight.likely.mode"
_SETTING_KEY_ENABLED = "fenlight.likely.enabled"
_CACHE_VERSION_KEY = "likely:cache_version"
_DEFAULT_EXPIRATION_HOURS = 24
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin"

# --------------- Module state (thread-safe) ---------------

_lock = threading.RLock()
_dataset: Optional[Dataset] = None
_dataset_meta: Dict[str, Any] = {}
_dataset_id: Optional[str] = None
_runtime_mode: Optional[str] = None


# --------------- Path helpers ---------------

def _ensure_cache_dir() -> str:
    folder = translate_path("special://profile/addon_data/plugin.video.fenlight/")
    data_folder = os.path.join(folder, "cache", _DEFAULT_CACHE_SUBDIR)
    try:
        os.makedirs(data_folder, exist_ok=True)
    except Exception:
        pass
    return data_folder


def _dataset_cache_path() -> str:
    return os.path.join(_ensure_cache_dir(), _DEFAULT_FILENAME)


# --------------- Cache version management ---------------

def _get_cache_version() -> str:
    """Return current version token (timestamp), creating if absent."""
    v = main_cache.get(_CACHE_VERSION_KEY)
    if v is None:
        v = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, v, expiration=0)
        except Exception:
            pass
    return str(v)


def clear_likely_cache() -> None:
    """Invalidate all cached results by bumping the version token."""
    with _lock:
        newv = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, newv, expiration=0)
        except Exception:
            try:
                main_cache.delete(_CACHE_VERSION_KEY)
            except Exception:
                pass


# --------------- Dataset identity ---------------

def _compute_dataset_id(path: str) -> str:
    """Lightweight fingerprint: 'mtime:size'."""
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{int(st.st_size)}"
    except Exception:
        return str(int(time.time()))


# --------------- User settings ---------------

def get_setting_mode() -> str:
    """Read persisted mode preference: 'auto', 'RAM', or 'mmap'."""
    v = get_setting(_SETTING_KEY_MODE, "auto")
    v = (v or "auto").strip()
    if v.lower() == "ram":
        return "RAM"
    if v.lower() == "mmap":
        return "mmap"
    return "auto"


def set_setting_mode(mode: str) -> bool:
    """Persist mode preference. Returns True on success."""
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
    """Return actual mode used this session: 'RAM', 'mmap', or None."""
    return _runtime_mode


# --------------- Lifecycle (load / reload / close) ---------------

def ensure_loaded(url: Optional[str] = None, mode: Optional[str] = None, force: bool = False) -> None:
    """Ensure dataset is downloaded and parsed. No-op if already loaded (unless force)."""
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    with _lock:
        if _dataset is not None and not force:
            return

        dataset_url = url or DEFAULT_DATASET_URL
        cache_path = _dataset_cache_path()
        mode_setting = get_setting_mode() if mode is None else mode

        # Internal loader uses 'air' for RAM mode (historical naming)
        loader_mode = "auto" if mode_setting == "auto" else ("air" if mode_setting == "RAM" else "mmap")

        ds, meta = load_or_fetch(dataset_url, cache_path, ram_copy=True, mode=loader_mode)

        _dataset = ds
        _dataset_meta = meta or {}
        _dataset_id = _compute_dataset_id(cache_path)

        chosen = meta.get("mode_chosen") if isinstance(meta, dict) else None
        if chosen == "air":
            _runtime_mode = "RAM"
        elif chosen == "mmap":
            _runtime_mode = "mmap"
        else:
            _runtime_mode = "RAM" if loader_mode == "air" else "mmap"


def reload_dataset(url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Close current dataset, invalidate cache, and reload from disk."""
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
        clear_likely_cache()
        ensure_loaded(url=url, mode=mode, force=True)


def close_dataset() -> None:
    """Release all resources. Queries will fail until ensure_loaded() is called."""
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


# --------------- Query functions ---------------

def _packed_cache_key(tmdb_id: int, kind: str) -> str:
    v = _get_cache_version()
    did = _dataset_id or "nodata"
    return f"likely:packed:{v}:{did}:{tmdb_id}:{kind}"


def query_likely_packed(
    tmdb_id: int,
    kind: str,
    use_cache: bool = True,
    expiration_hours: int = _DEFAULT_EXPIRATION_HOURS,
) -> List[int]:
    """Return similar titles as packed ints. Auto-loads dataset if needed."""
    if _dataset is None:
        ensure_loaded()

    key = _packed_cache_key(tmdb_id, kind)
    if use_cache:
        try:
            cached = main_cache.get(key)
            if cached is not None:
                return list(cached)
        except Exception:
            pass

    packed = _dataset.query_similar_packed(tmdb_id, kind)  # type: ignore[union-attr]

    try:
        main_cache.set(key, packed, expiration=expiration_hours)
    except Exception:
        pass

    return packed


def query_likely_pairs(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """Return {"count": N, "results": [[id, type_bit], ...]}. Optionally (dict, ms)."""
    t0 = time.perf_counter()
    packed = query_likely_packed(tmdb_id, kind)
    res = [[(pv >> 1), (pv & 1)] for pv in packed]
    out = {"count": len(res), "results": res}
    total_ms = (time.perf_counter() - t0) * 1000.0
    return (out, total_ms) if timing else out


def get_likely_for_addon(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """Return TMDB-style {"results": [{"id":…,"media_type":…}], "total_results": N}."""
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


# --------------- Debug / status ---------------

def dataset_info() -> Dict[str, Any]:
    """Return status dict: loaded, dataset_id, runtime_mode, meta."""
    return {
        "loaded": _dataset is not None,
        "dataset_id": _dataset_id,
        "runtime_mode": _runtime_mode,
        "meta": _dataset_meta or {},
    }