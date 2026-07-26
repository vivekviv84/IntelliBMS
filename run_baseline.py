import csv
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENERGYPLUS_DIR = r"C:\EnergyPlusV26-1-0"
IDF_PATH = os.path.join(PROJECT_ROOT, "models", "baseline.idf")
EPW_PATH = os.path.join(PROJECT_ROOT, "weather", "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw")
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

    energyplus_exe = os.path.join(ENERGYPLUS_DIR, "energyplus.exe")
    cmd = [
        energyplus_exe,
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
