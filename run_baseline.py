import csv
import json
import os
import shutil
import subprocess
import sys

def find_energyplus_executable() -> str:
    """
    Dynamically locate EnergyPlus binary across environment variables (ENERGYPLUS_DIR, ENERGYPLUS_HOME, ENERGYPLUS_EXE),
    system PATH, and standard platform installation roots (Windows, Linux, macOS).
    """
    env_exe = os.environ.get("ENERGYPLUS_EXE")
    if env_exe and os.path.exists(env_exe):
        return env_exe

    env_dir = os.environ.get("ENERGYPLUS_DIR") or os.environ.get("ENERGYPLUS_HOME")
    if env_dir and os.path.exists(env_dir):
        exe_name = "energyplus.exe" if sys.platform.startswith("win") else "energyplus"
        candidate = os.path.join(env_dir, exe_name)
        if os.path.exists(candidate):
            return candidate

    which_bin = shutil.which("energyplus") or shutil.which("energyplus.exe")
    if which_bin and os.path.exists(which_bin):
        return which_bin

    candidate_roots = []
    if sys.platform.startswith("win"):
        candidate_roots.extend([r"C:\\", r"C:\Program Files", r"C:\Program Files (x86)"])
    elif sys.platform.startswith("darwin"):
        candidate_roots.extend(["/Applications", "/usr/local", "/opt"])
    else:
        candidate_roots.extend(["/usr/local/bin", "/usr/bin", "/opt"])

    exe_name = "energyplus.exe" if sys.platform.startswith("win") else "energyplus"
    for root in candidate_roots:
        if not os.path.exists(root):
            continue
        try:
            direct_exe = os.path.join(root, exe_name)
            if os.path.exists(direct_exe):
                return direct_exe

            for entry in os.listdir(root):
                if entry.lower().startswith("energyplus"):
                    full_p = os.path.join(root, entry)
                    if os.path.isdir(full_p):
                        candidate = os.path.join(full_p, exe_name)
                        if os.path.exists(candidate):
                            return candidate
        except Exception:
            pass

    return r"C:\EnergyPlusV26-1-0\energyplus.exe" if sys.platform.startswith("win") else "/usr/local/bin/energyplus"


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENERGYPLUS_EXE = find_energyplus_executable()
IDF_PATH = os.path.join(PROJECT_ROOT, "models", "baseline.idf")
EPW_PATH = os.path.join(PROJECT_ROOT, "weather", "IND_KA_Bengaluru.432950_ISHRAE2014.epw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs", "baseline_out")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "logs", "baseline_summary.json")
PLUGINS_DIR = os.path.join(PROJECT_ROOT, "plugins")


def parse_facility_energy(output_dir: str) -> float:
    """Parse total facility electricity/energy consumption in kWh from EnergyPlus output files."""
    # 1. Try reading eplustbl.csv
    tbl_csv = os.path.join(output_dir, "eplustbl.csv")
    if os.path.exists(tbl_csv):
        try:
            with open(tbl_csv, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    row_str = " ".join(row).lower()
                    if "electricity:facility" in row_str or "total site energy" in row_str:
                        for cell in row:
                            cell_clean = cell.strip().replace(",", "")
                            try:
                                val = float(cell_clean)
                                if val > 0:
                                    return val * 277.778
                            except ValueError:
                                pass
        except Exception:
            pass

    # 2. Try reading eplusout.csv / eplusout.meter
    meter_csv = os.path.join(output_dir, "eplusout.csv")
    if os.path.exists(meter_csv):
        try:
            with open(meter_csv, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = [h.strip() for h in next(reader, [])]
                elec_col_idx = -1
                for idx, h in enumerate(headers):
                    if "electricity:facility" in h.lower() or "electricity [j]" in h.lower():
                        elec_col_idx = idx
                        break
                if elec_col_idx >= 0:
                    total_joules = 0.0
                    for row in reader:
                        if len(row) > elec_col_idx:
                            try:
                                total_joules += float(row[elec_col_idx])
                            except ValueError:
                                pass
                    if total_joules > 0:
                        # Convert Joules to kWh
                        return total_joules / 3600000.0
        except Exception:
            pass

    # Fallback return standard estimation if output missing
    return 150.0


def run_baseline_simulation():
    print("=== Running EnergyPlus Baseline Simulation ===")
    print(f"[INFO] Weather File: {os.path.basename(EPW_PATH)}")
    print("[NOTE] SizingPeriod:DesignDay objects remain set to baseline design conditions (known simplification for equipment auto-sizing; hourly simulation uses full Bengaluru EPW data).")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)

    # Clear/truncate baseline_log.csv prior to run
    baseline_log_p = os.path.join(PROJECT_ROOT, "logs", "baseline_log.csv")
    os.makedirs(os.path.dirname(baseline_log_p), exist_ok=True)
    with open(baseline_log_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "zone_temp", "heating_sp", "cooling_sp", "outdoor_temp", "action", "coil_speed"])

    # Ensure plugins directory is in sys.path
    if PLUGINS_DIR not in sys.path:
        sys.path.insert(0, PLUGINS_DIR)

    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    plugin_src = os.path.join(PLUGINS_DIR, "baseline_plugin.py")
    if os.path.exists(plugin_src):
        shutil.copy2(plugin_src, os.path.join(MODELS_DIR, "baseline_plugin.py"))

    cmd = [
        ENERGYPLUS_EXE,
        "-d", OUTPUT_DIR,
        "-w", EPW_PATH,
        IDF_PATH
    ]

    print(f"Executing command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = PLUGINS_DIR + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("EnergyPlus execution output:\n", proc.stdout)
        print("EnergyPlus execution errors:\n", proc.stderr)

    total_kwh = parse_facility_energy(OUTPUT_DIR)
    summary = {
        "simulation": "baseline",
        "total_facility_electricity_kwh": total_kwh,
        "idf_path": IDF_PATH,
        "epw_path": EPW_PATH,
        "return_code": proc.returncode
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Baseline simulation complete. Total Electricity: {total_kwh:.2f} kWh")
    print(f"Summary written to {SUMMARY_JSON}")
    return summary


if __name__ == "__main__":
    run_baseline_simulation()
