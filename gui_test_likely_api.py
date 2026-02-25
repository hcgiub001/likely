#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui_test_likely_api.py — updated startup/read-settings + robust reload handling.

Place alongside likely_api.py (the single combined file that embeds the dataset
reader/loader).

Run:  python gui_test_likely_api.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict

# Default dataset URL and local cache path used by the loaders
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin"
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".tmdb_similar_cache")
DEFAULT_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "dataset.bin")

# Settings persistence filename (next to this script)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_FILE = os.path.join(_THIS_DIR, ".likely_settings.json")

# ---------------------------------------------------------------------
# Import likely_api with robust stubbing fallback (disk-backed settings)
# ---------------------------------------------------------------------
def import_likely_api_with_stubs() -> Any:
    """
    Import likely_api.py (single combined file).
    If the initial import fails (missing fenlight deps), install lightweight
    stubs for caches.main_cache, caches.settings_cache, and modules.kodi_utils,
    then retry the import.
    """
    try:
        import likely_api  # type: ignore
        return likely_api
    except Exception as exc:
        if not os.path.exists(os.path.join(_THIS_DIR, "likely_api.py")):
            raise RuntimeError(
                "likely_api.py not found in current directory. "
                "Place your likely_api.py here before running this tester."
            ) from exc
        print("Initial import of likely_api failed; installing stubs for testing. Error:", exc)

    # ---- lightweight stubs ------------------------------------------------
    class SimpleCache:
        def __init__(self):
            self._d: Dict[str, Any] = {}
        def get(self, k, default=None):
            return self._d.get(k, default)
        def set(self, k, v, expiration=0):
            self._d[k] = v
        def delete(self, k):
            try:
                del self._d[k]
            except KeyError:
                pass

    def _load_settings_file() -> Dict[str, str]:
        try:
            if os.path.exists(_SETTINGS_FILE):
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_settings_file(d: Dict[str, str]) -> None:
        try:
            with open(_SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(d, fh)
        except Exception:
            pass

    _SETTINGS_STORE: Dict[str, str] = _load_settings_file()

    def get_setting_stub(key: str, default: str = "") -> str:
        return _SETTINGS_STORE.get(key, default)

    def set_setting_stub(key: str, value: str) -> None:
        _SETTINGS_STORE[key] = str(value)
        _save_settings_file(_SETTINGS_STORE)

    def translate_path_stub(path: str) -> str:
        if path.startswith("special://profile"):
            base = os.path.expanduser("~")
            tail = path[len("special://profile"):].lstrip("/\\")
            return os.path.join(base, tail)
        return path

    def kodi_dialog_stub(*args, **kwargs):
        print("kodi_dialog:", args, kwargs)

    def kodi_log_stub(msg: str, level: int = 0):
        print(f"KODI_LOG[{level}]: {msg}")

    import types

    # -- caches package and submodules --
    caches_pkg = types.ModuleType("caches")
    caches_main_mod = types.ModuleType("caches.main_cache")
    caches_main_mod.main_cache = SimpleCache()
    caches_settings_mod = types.ModuleType("caches.settings_cache")
    caches_settings_mod.get_setting = get_setting_stub
    caches_settings_mod.set_setting = set_setting_stub

    # -- modules package and submodules --
    modules_pkg = types.ModuleType("modules")
    kodi_mod = types.ModuleType("modules.kodi_utils")
    kodi_mod.translate_path = translate_path_stub
    kodi_mod.kodi_dialog = kodi_dialog_stub
    kodi_mod.kodi_log = kodi_log_stub

    sys.modules["caches"] = caches_pkg
    sys.modules["caches.main_cache"] = caches_main_mod
    sys.modules["caches.settings_cache"] = caches_settings_mod
    sys.modules["modules"] = modules_pkg
    sys.modules["modules.kodi_utils"] = kodi_mod

    try:
        likely_api = importlib.import_module("likely_api")
        print("Imported likely_api with stubs (testing mode).")
        return likely_api
    except Exception as exc2:
        raise RuntimeError(f"Could not import likely_api even after installing stubs: {exc2}") from exc2


try:
    likely = import_likely_api_with_stubs()
except Exception as e:
    raise SystemExit(f"Failed to import likely_api.py for testing: {e}")

# ---------------------------------------------------------------------
# Helper to read persisted mode (fallback if likely doesn't provide getter)
# ---------------------------------------------------------------------
def read_persisted_mode_from_api_or_stub() -> str:
    # preferred: likely.get_setting_mode()
    try:
        if hasattr(likely, "get_setting_mode"):
            m = likely.get_setting_mode()
            if isinstance(m, str) and m:
                return m
    except Exception:
        pass
    # fallback: try caches.settings_cache.get_setting
    try:
        sc = importlib.import_module("caches.settings_cache")
        getter = getattr(sc, "get_setting", None)
        if callable(getter):
            return getter("fenlight.likely.mode", "auto")
    except Exception:
        pass
    # final fallback
    return "auto"

# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Likely API Tester")
        root.geometry("900x580")

        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="TMDB ID:").grid(row=0, column=0, sticky="w")
        self.id_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.id_var, width=12).grid(row=0, column=1, sticky="w")

        ttk.Label(top, text="Type:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.type_var = tk.StringVar(value="movie")
        ttk.Combobox(top, textvariable=self.type_var, values=("movie", "tv"), width=8, state="readonly").grid(
            row=0, column=3, sticky="w"
        )

        ttk.Label(top, text="Mode:").grid(row=0, column=4, sticky="w", padx=(10, 0))
        # initialize mode_var from persisted setting (so selection is remembered)
        init_mode = read_persisted_mode_from_api_or_stub()
        if init_mode not in ("auto", "RAM", "mmap"):
            init_mode = "auto"
        self.mode_var = tk.StringVar(value=init_mode)
        self.mode_combo = ttk.Combobox(
            top, textvariable=self.mode_var, values=("auto", "RAM", "mmap"), width=10, state="readonly"
        )
        self.mode_combo.grid(row=0, column=5, sticky="w", padx=(4, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_selected)

        self.load_btn = ttk.Button(top, text="Load dataset", command=self.on_load)
        self.load_btn.grid(row=0, column=6, padx=8)

        self.run_btn = ttk.Button(top, text="Run", command=self.on_run)
        self.run_btn.grid(row=0, column=7, padx=8)

        self.reload_btn = ttk.Button(top, text="Reload dataset", command=self.on_reload)
        self.reload_btn.grid(row=0, column=8, padx=8)

        self.clear_btn = ttk.Button(top, text="Clear Cache", command=self.on_clear)
        self.clear_btn.grid(row=0, column=9, padx=8)

        meta_frame = ttk.Frame(root)
        meta_frame.pack(fill="x", padx=8, pady=(6, 0))
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(meta_frame, text="Status:").pack(side="left")
        ttk.Label(meta_frame, textvariable=self.status_var, foreground="blue").pack(side="left", padx=(4, 20))
        self.meta_label = ttk.Label(meta_frame, text="Meta: not loaded", foreground="gray")
        self.meta_label.pack(side="left")

        self.result_box = ScrolledText(root, wrap="none", font=("Courier", 11))
        self.result_box.pack(fill="both", expand=True, padx=8, pady=8)

        self.api = likely
        self.dataset_loaded = False

        threading.Thread(target=self.background_load, daemon=True).start()

    def set_status(self, s: str):
        self.status_var.set(s)
        self.root.update_idletasks()

    def set_meta(self, text: str):
        self.meta_label.config(text=text)
        self.root.update_idletasks()

    def set_result(self, text: str):
        if len(text) > 1_000_000:
            text = text[:1_000_000] + "\n\n[truncated large output]"
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text)
        self.root.update_idletasks()

    def background_load(self):
        try:
            self.set_status("background loading...")
            self._set_buttons_state(load=False, run=False, clear=False)
            self.mode_combo.config(state="disabled")
            mode = self.mode_var.get()
            if hasattr(self.api, "ensure_loaded"):
                try:
                    self.api.ensure_loaded(mode=mode)
                except TypeError:
                    self.api.ensure_loaded(mode)
            self.dataset_loaded = True
            runtime = None
            if hasattr(self.api, "get_runtime_mode"):
                try:
                    runtime = self.api.get_runtime_mode()
                except Exception:
                    runtime = None
            info = f"Loaded. runtime_mode={runtime or 'unknown'} file_bytes={os.path.getsize(DEFAULT_CACHE_PATH) if os.path.exists(DEFAULT_CACHE_PATH) else 'n/a'}"
            self.set_meta(info)
            self.set_result(json.dumps({"loaded": True, "runtime_mode": runtime}, separators=(",", ":"), ensure_ascii=False))
            self.set_status("idle")
        except Exception as e:
            print("background_load error:", e)
            self.set_meta(f"Load error: {e}")
            self.set_result(f"Load error: {e}")
            self.set_status("error")
        finally:
            try:
                self.mode_combo.config(state="readonly")
            except Exception:
                pass
            self._set_buttons_state(load=True, run=True, clear=True)

    def on_mode_selected(self, event=None):
        new_mode = self.mode_var.get()
        persisted = False
        try:
            if hasattr(self.api, "set_setting_mode"):
                try:
                    persisted = bool(self.api.set_setting_mode(new_mode))
                except Exception:
                    persisted = False
            else:
                try:
                    sc = importlib.import_module("caches.settings_cache")
                    if hasattr(sc, "set_setting"):
                        sc.set_setting("fenlight.likely.mode", new_mode)
                        persisted = True
                except Exception:
                    persisted = False
        except Exception:
            persisted = False

        if persisted:
            if messagebox.askyesno("Likely mode changed", f"Mode set to '{new_mode}'. Reload dataset now to apply?"):
                try:
                    if hasattr(self.api, "reload_dataset"):
                        try:
                            self.api.reload_dataset(mode=new_mode)
                        except TypeError:
                            self.api.reload_dataset(new_mode)
                    else:
                        try:
                            self.api.ensure_loaded(mode=new_mode, force=True)
                        except TypeError:
                            self.api.ensure_loaded(new_mode, True)
                    runtime = None
                    if hasattr(self.api, "get_runtime_mode"):
                        try:
                            runtime = self.api.get_runtime_mode()
                        except Exception:
                            runtime = None
                    self.set_meta(f"Reloaded. runtime_mode={runtime or 'unknown'}")
                except Exception as e:
                    print("Reload error in on_mode_selected:", e)
                    messagebox.showerror("Reload failed", f"Reload failed: {e}")
        else:
            messagebox.showwarning("Persist failed", "Could not persist mode selection. Changes will not be remembered on restart.")

    def on_load(self):
        def task():
            try:
                self.set_status("loading...")
                self._set_buttons_state(load=False, run=False, clear=False)
                self.mode_combo.config(state="disabled")
                mode = self.mode_var.get()
                if hasattr(self.api, "ensure_loaded"):
                    try:
                        self.api.ensure_loaded(mode=mode)
                    except TypeError:
                        self.api.ensure_loaded(mode)
                self.dataset_loaded = True
                runtime = None
                if hasattr(self.api, "get_runtime_mode"):
                    try:
                        runtime = self.api.get_runtime_mode()
                    except Exception:
                        runtime = None
                self.set_meta(f"Loaded. runtime_mode={runtime or 'unknown'}")
                self.set_result(json.dumps({"loaded": True, "runtime_mode": runtime}, separators=(",", ":"), ensure_ascii=False))
                self.set_status("idle")
            except Exception as e:
                print("on_load error:", e)
                self.set_meta(f"Load error: {e}")
                self.set_result(f"Load error: {e}")
                self.set_status("error")
            finally:
                try:
                    self.mode_combo.config(state="readonly")
                except Exception:
                    pass
                self._set_buttons_state(load=True, run=True, clear=True)

        threading.Thread(target=task, daemon=True).start()

    def on_reload(self):
        def task():
            try:
                self.set_status("reloading...")
                self._set_buttons_state(load=False, run=False, clear=False)
                try:
                    if hasattr(self.api, "reload_dataset"):
                        try:
                            self.api.reload_dataset(mode=self.mode_var.get())
                        except TypeError:
                            self.api.reload_dataset(self.mode_var.get())
                    else:
                        try:
                            self.api.ensure_loaded(mode=self.mode_var.get(), force=True)
                        except TypeError:
                            self.api.ensure_loaded(self.mode_var.get(), True)
                except Exception as e:
                    raise
                runtime = None
                if hasattr(self.api, "get_runtime_mode"):
                    try:
                        runtime = self.api.get_runtime_mode()
                    except Exception:
                        runtime = None
                self.set_meta(f"Reloaded. runtime_mode={runtime or 'unknown'}")
                self.set_result(json.dumps({"reloaded": True, "runtime_mode": runtime}, separators=(",", ":"), ensure_ascii=False))
                self.set_status("idle")
            except Exception as e:
                print("on_reload error:", e)
                self.set_meta(f"Reload error: {e}")
                self.set_result(f"Reload error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(load=True, run=True, clear=True)

        threading.Thread(target=task, daemon=True).start()

    def on_clear(self):
        try:
            if hasattr(self.api, "clear_likely_cache"):
                try:
                    self.api.clear_likely_cache()
                except Exception:
                    pass
            if hasattr(self.api, "clear_cache"):
                try:
                    self.api.clear_cache()
                except Exception:
                    pass
            try:
                if os.path.exists(DEFAULT_CACHE_PATH):
                    os.remove(DEFAULT_CACHE_PATH)
            except Exception:
                pass
            self.set_meta("Cache cleared.")
            self.set_result(json.dumps({"cache_cleared": True}, separators=(",", ":"), ensure_ascii=False))
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear cache: {e}")

    def on_run(self):
        idtxt = self.id_var.get().strip()
        if not idtxt.isdigit():
            messagebox.showerror("Invalid id", "Please enter a numeric tmdb id.")
            return
        tmdb_id = int(idtxt)
        kind = self.type_var.get()

        def task():
            try:
                self.set_status("running query...")
                self._set_buttons_state(load=False, run=False, clear=False)
                try:
                    if hasattr(self.api, "ensure_loaded"):
                        self.api.ensure_loaded(mode=self.mode_var.get())
                except Exception:
                    pass

                t0 = time.perf_counter()
                if hasattr(self.api, "query_likely_pairs"):
                    out = self.api.query_likely_pairs(tmdb_id, kind)
                elif hasattr(self.api, "query_likely_packed"):
                    packed = self.api.query_likely_packed(tmdb_id, kind)
                    out = {"count": len(packed), "results": [[(pv >> 1), (pv & 1)] for pv in packed]}
                else:
                    out = {"count": 0, "results": []}
                total_ms = (time.perf_counter() - t0) * 1000.0
                compact = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
                self.set_result(compact)
                self.set_meta(f"Found {out.get('count',0)} — total {total_ms:.3f} ms")
                self.set_status("idle")
            except Exception as e:
                print("on_run error:", e)
                self.set_result(f"Error during query: {e}")
                self.set_meta("error")
            finally:
                self._set_buttons_state(load=True, run=True, clear=True)

        threading.Thread(target=task, daemon=True).start()

    def _set_buttons_state(self, load=True, run=True, clear=True):
        try:
            self.load_btn.config(state="normal" if load else "disabled")
            self.run_btn.config(state="normal" if run else "disabled")
            self.clear_btn.config(state="normal" if clear else "disabled")
            self.reload_btn.config(state="normal" if load else "disabled")
            self.root.update_idletasks()
        except Exception:
            pass

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
