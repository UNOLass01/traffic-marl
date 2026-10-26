"""Build the cropped LuST scenario (network, routes, manifest, map figure).

Requires the LuST repository in external/LuSTScenario (see README.md).
All parameters live in configs/scenarios/lust.yaml; the procedure is
documented in src/trafficmarl/scenarios/lust.py and docs/scenarios.md.

Usage: python networks/lust/crop_lust.py
"""

from trafficmarl.scenarios import build_scenario
from trafficmarl.utils.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging("INFO")
    manifest = build_scenario("lust")
    print(f"{len(manifest.tls_ids)} traffic lights -> {manifest.root}")
