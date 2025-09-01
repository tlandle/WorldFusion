#!/usr/bin/env python3
"""
Compute per‑scenario global anchors for OPV2V.

This script scans each scenario directory in an OPV2V dataset (e.g. data/OPV2V/train)
and extracts the lidar_pose entries from every agent’s YAML file.  It then
computes a static anchor as the centre of the bounding box of all x,y,z values.
The results are saved in opv2v_anchors.json in the current directory.

It does not rely on PyYAML loaders, so it won’t choke on numpy tags.
"""

import os
import glob
import re
import json
import sys

def read_lidar_pose(filepath):
    """
    Extract [x,y,z,roll,yaw,pitch] from a YAML by looking for 'lidar_pose:'.
    Handles both inline list and multi‑line forms.

    Returns a list of 6 floats or raises ValueError if not found.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('lidar_pose'):
            # inline form: lidar_pose: [x, y, z, roll, yaw, pitch]
            m = re.search(r'\[(.*?)\]', line)
            if m:
                parts = [float(x.strip()) for x in m.group(1).split(',')]
                if len(parts) == 6:
                    return parts
            # multi-line form:
            vals = []
            for j in range(idx + 1, len(lines)):
                tok = lines[j].strip()
                if tok.startswith('-'):
                    tok = tok[1:].strip()
                tok = tok.rstrip(',')
                try:
                    vals.append(float(tok))
                except ValueError:
                    break
                if len(vals) == 6:
                    return vals
            break
    raise ValueError(f"lidar_pose not found in {filepath}")

def compute_anchor_for_scenario(scenario_path):
    xs, ys, zs = [], [], []
    for cav_id in os.listdir(scenario_path):
        cav_path = os.path.join(scenario_path, cav_id)
        if not os.path.isdir(cav_path):
            continue
        for yml in glob.glob(os.path.join(cav_path, '*.yaml')):
            try:
                pose = read_lidar_pose(yml)
            except Exception:
                continue
            xs.append(pose[0])
            ys.append(pose[1])
            zs.append(pose[2])
    if not xs:
        return [0.0, 0.0, 0.0]
    x_anchor = 0.5 * (max(xs) + min(xs))
    y_anchor = 0.5 * (max(ys) + min(ys))
    z_anchor = 0.5 * (max(zs) + min(zs))
    return [x_anchor, y_anchor, z_anchor]

def main(root_dir):
    anchors = {}
    for scenario_name in sorted(os.listdir(root_dir)):
        scenario_path = os.path.join(root_dir, scenario_name)
        if not os.path.isdir(scenario_path):
            continue
        anchors[scenario_name] = compute_anchor_for_scenario(scenario_path)
        print(f"{scenario_name}: {anchors[scenario_name]}")
    with open('opv2v_anchors.json', 'w') as f:
        json.dump(anchors, f, indent=2)
    print("Wrote anchors to opv2v_anchors.json")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compute_global_anchors.py <path_to_OPV2V/train>")
    else:
        main(sys.argv[1])

