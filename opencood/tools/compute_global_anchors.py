#!/usr/bin/env python3
"""
Compute per-scenario global anchors for OPV2V based on the initial scene centroid.

This script scans each scenario directory in an OPV2V dataset and identifies
the first timestamp. It then reads the lidar_pose of every agent present at
that initial timestamp and computes the anchor as the mean (average) of their
x, y, z positions. This ensures the static anchor is centered on the action
at the beginning of the scenario.

The results are saved in opv2v_anchors.json in the current directory.
"""

import os
import glob
import re
import json
import sys
import numpy as np

def read_lidar_pose(filepath):
    """
    Extract [x,y,z,roll,yaw,pitch] from a YAML by looking for 'lidar_pose:'.
    Handles both inline list and multi-line forms.

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
    """
    MODIFIED: Calculates the anchor as the average position of all vehicles
    at the first timestamp of the scenario.
    """
    cav_paths = [os.path.join(scenario_path, cav_id) for cav_id in os.listdir(scenario_path) if os.path.isdir(os.path.join(scenario_path, cav_id))]
    if not cav_paths:
        return [0.0, 0.0, 0.0]

    # Find the first timestamp by looking at the first agent's sorted yaml files
    try:
        all_yamls = sorted(glob.glob(os.path.join(cav_paths[0], '*.yaml')))
        first_yaml_path = all_yamls[0]
        first_timestamp = os.path.basename(first_yaml_path).replace('.yaml', '')
    except IndexError:
        print(f"Warning: No yaml files found in {cav_paths[0]}, skipping scenario.")
        return [0.0, 0.0, 0.0]

    initial_poses = []
    for cav_path in cav_paths:
        # Construct the path to the yaml for the first timestamp for this agent
        initial_yaml_for_cav = os.path.join(cav_path, f"{first_timestamp}.yaml")
        if os.path.exists(initial_yaml_for_cav):
            try:
                pose = read_lidar_pose(initial_yaml_for_cav)
                initial_poses.append(pose[:3]) # Only need [x, y, z]
            except Exception as e:
                print(f"Warning: Could not read pose from {initial_yaml_for_cav}: {e}")
                continue
    
    if not initial_poses:
        print(f"Warning: No valid initial poses found for scenario {os.path.basename(scenario_path)}")
        return [0.0, 0.0, 0.0]
    
    # Calculate the mean (average) position of all vehicles at the start
    anchor = np.mean(np.array(initial_poses), axis=0)
    return anchor.tolist()

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
    print("\n✅ Wrote new anchors to opv2v_anchors.json")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compute_global_anchors.py <path_to_OPV2V_dataset_root>")
        print("Example: python compute_global_anchors.py data/opv2v/train")
    else:
        main(sys.argv[1])


