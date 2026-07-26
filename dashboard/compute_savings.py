import csv
import json
import os


def parse_facility_energy(output_dir: str) -> float:
    """Parse total facility electricity consumption in kWh from EnergyPlus output files.
    
    Primary source: eplusout.mtr meter file 'Electricity:Facility [J]' RunPeriod line.
    Divides Joules by 3,600,000.0 to obtain kWh with full floating-point precision.
    Fallback: eplustbl.csv row where first non-empty cell is exactly 'Electricity:Facility'.
    """
    # 1. Primary: Try eplusout.mtr
    mtr_file = os.path.join(output_dir, "eplusout.mtr")
    if os.path.exists(mtr_file):
        try:
            with open(mtr_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                run_period_id = None
                for line in lines:
                    if "Electricity:Facility [J]" in line and "!RunPeriod" in line:
                        parts = line.split(",")
                        if len(parts) > 0:
                            run_period_id = parts[0].strip()
                            break
                if run_period_id:
                    for line in reversed(lines):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2 and parts[0] == run_period_id:
                            try:
                                joules = float(parts[1])
                                if joules > 0:
                                    return joules / 3600000.0
                            except ValueError:
                                pass
        except Exception:
            pass

    # 2. Secondary: Try eplusout.csv
    csv_file = os.path.join(output_dir, "eplusout.csv")
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = [h.strip() for h in next(reader, [])]
                elec_col_idx = -1
                for idx, h in enumerate(headers):
                    if h.lower().startswith("electricity:facility"):
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
                        return total_joules / 3600000.0
        except Exception:
            pass

    # 3. Fallback: eplustbl.csv (exact match on Electricity:Facility)
    tbl_csv = os.path.join(output_dir, "eplustbl.csv")
    if os.path.exists(tbl_csv):
        try:
            with open(tbl_csv, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    non_empty = [c.strip() for c in row if c.strip()]
                    if non_empty and non_empty[0].lower() == "electricity:facility":
                        if len(non_empty) >= 2:
                            try:
                                val = float(non_empty[1])
                                if val > 0:
                                    return val * 277.778
                            except ValueError:
                                pass
        except Exception:
            pass

    return 0.0


def parse_peak_demand(output_dir: str) -> tuple:
    """Extract Peak Demand (W) and timestamp from eplustbl.csv row 'Electricity:Facility'."""
    tbl_csv = os.path.join(output_dir, "eplustbl.csv")
    if os.path.exists(tbl_csv):
        try:
            with open(tbl_csv, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    non_empty = [c.strip() for c in row if c.strip()]
                    if non_empty and non_empty[0].lower() == "electricity:facility":
                        if len(non_empty) >= 6:
                            try:
                                peak_w = float(non_empty[4])
                                timestamp = non_empty[5]
                                return peak_w, timestamp
                            except ValueError:
                                pass
        except Exception:
            pass
    return 0.0, "N/A"


def calc_comfort_deviation(csv_filepath: str) -> float:
    """Calculate mean absolute difference between zone_temp and the nearest setpoint across all rows."""
    if not os.path.exists(csv_filepath):
        return 0.0

    total_diff = 0.0
    count = 0

    try:
        with open(csv_filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    zt = float(row["zone_temp"])
                    h_sp = float(row["heating_sp"])
                    c_sp = float(row["cooling_sp"])

                    # If zone temp is within [heating_sp, cooling_sp], deviation from nearest comfort bound is 0.0
                    if h_sp <= zt <= c_sp:
                        diff = 0.0
                    else:
                        diff = min(abs(zt - h_sp), abs(zt - c_sp))

                    total_diff += diff
                    count += 1
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass

    return (total_diff / count) if count > 0 else 0.0


def compute_savings():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    baseline_out_dir = os.path.join(project_root, "logs", "baseline_out")
    ecoloop_out_dir = os.path.join(project_root, "logs", "ecoloop_out")

    baseline_summary_path = os.path.join(project_root, "logs", "baseline_summary.json")
    ecoloop_summary_path = os.path.join(project_root, "logs", "ecoloop_summary.json")
    ai_log_path = os.path.join(project_root, "logs", "decision_log.csv")
    baseline_log_path = os.path.join(project_root, "logs", "baseline_log.csv")
    results_json_path = os.path.join(project_root, "docs", "results.json")

    # 1. Compute baseline electricity in kWh & peak demand
    baseline_kwh = parse_facility_energy(baseline_out_dir)
    baseline_peak_w, baseline_peak_time = parse_peak_demand(baseline_out_dir)

    # 2. Compute AI electricity in kWh & peak demand
    ai_kwh = parse_facility_energy(ecoloop_out_dir)
    ai_peak_w, ai_peak_time = parse_peak_demand(ecoloop_out_dir)

    # Compute percentage energy savings
    if baseline_kwh > 0:
        pct_energy_savings = ((baseline_kwh - ai_kwh) / baseline_kwh) * 100.0
    else:
        pct_energy_savings = 0.0

    # Compute percentage peak demand reduction
    if baseline_peak_w > 0:
        pct_peak_demand_reduction = ((baseline_peak_w - ai_peak_w) / baseline_peak_w) * 100.0
    else:
        pct_peak_demand_reduction = 0.0

    # Compute comfort deviations
    comfort_deviation_ai = calc_comfort_deviation(ai_log_path)
    comfort_deviation_baseline = calc_comfort_deviation(baseline_log_path)

    results = {
        "baseline_energy_kwh": round(baseline_kwh, 2),
        "ai_energy_kwh": round(ai_kwh, 2),
        "pct_energy_savings": round(pct_energy_savings, 2),
        "peak_demand_w_baseline": round(baseline_peak_w, 2),
        "peak_demand_w_ai": round(ai_peak_w, 2),
        "pct_peak_demand_reduction": round(pct_peak_demand_reduction, 2),
        "comfort_deviation_ai": round(comfort_deviation_ai, 3),
        "comfort_deviation_baseline": round(comfort_deviation_baseline, 3)
    }

    # Update summary JSON files with correct kWh values
    if os.path.exists(baseline_summary_path):
        try:
            with open(baseline_summary_path, "r+", encoding="utf-8") as f:
                b_data = json.load(f)
                b_data["total_facility_electricity_kwh"] = round(baseline_kwh, 2)
                b_data["peak_demand_w"] = round(baseline_peak_w, 2)
                b_data["peak_demand_timestamp"] = baseline_peak_time
                f.seek(0)
                json.dump(b_data, f, indent=2)
                f.truncate()
        except Exception:
            pass

    if os.path.exists(ecoloop_summary_path):
        try:
            with open(ecoloop_summary_path, "r+", encoding="utf-8") as f:
                e_data = json.load(f)
                e_data["total_facility_electricity_kwh"] = round(ai_kwh, 2)
                e_data["peak_demand_w"] = round(ai_peak_w, 2)
                e_data["peak_demand_timestamp"] = ai_peak_time
                f.seek(0)
                json.dump(e_data, f, indent=2)
                f.truncate()
        except Exception:
            pass

    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n==================================================")
    print("           ECOLOOP ENERGY SAVINGS SUMMARY          ")
    print("==================================================")
    print(f"  Baseline Facility Electricity : {results['baseline_energy_kwh']:.2f} kWh")
    print(f"  EcoLoop AI Facility Electricity: {results['ai_energy_kwh']:.2f} kWh")
    print(f"  Percentage Energy Savings     : {results['pct_energy_savings']:.2f} %")
    print("--------------------------------------------------")
    print(f"  Baseline Peak Demand          : {results['peak_demand_w_baseline']:.2f} W ({baseline_peak_time})")
    print(f"  EcoLoop AI Peak Demand        : {results['peak_demand_w_ai']:.2f} W ({ai_peak_time})")
    print(f"  Percentage Peak Demand Reduct.: {results['pct_peak_demand_reduction']:.2f} %")
    print("--------------------------------------------------")
    print(f"  Comfort Deviation (Baseline)  : {results['comfort_deviation_baseline']:.3f} °C")
    print(f"  Comfort Deviation (AI)        : {results['comfort_deviation_ai']:.3f} °C")
    print("==================================================")
    print(f"Results saved to {results_json_path}\n")

    return results


if __name__ == "__main__":
    compute_savings()
