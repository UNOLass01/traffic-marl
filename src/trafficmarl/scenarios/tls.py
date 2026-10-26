"""Traffic-light program files shared by all scenarios.

Each built scenario keeps exactly one (static) program per traffic light in its
network and ships an actuated variant as an additional file. Loading that file
makes the actuated program active, which is how the actuated baseline is run.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

ACTUATED_FILE = "tls_actuated.add.xml"
ACTUATED_PROGRAM_ID = "actuated"
# SUMO's netconvert defaults (--tls.min-dur / --tls.max-dur) for actuated phases.
DEFAULT_MIN_DUR = 5
DEFAULT_MAX_DUR = 50


def is_green_phase(state: str) -> bool:
    return "y" not in state.lower() and any(c in "Gg" for c in state)


def to_actuated(tl_logics: Iterable[ET.Element]) -> list[ET.Element]:
    """Copies of ``tl_logics`` as gap-based actuated programs.

    Green phases without a duration range get SUMO's default 5-50 s; without
    it SUMO silently runs an "actuated" program at fixed durations (the case of
    LuST's original actuated programs).
    """
    result = []
    for tl in tl_logics:
        clone = copy.deepcopy(tl)
        clone.set("type", "actuated")
        clone.set("programID", ACTUATED_PROGRAM_ID)
        for phase in clone.findall("phase"):
            if is_green_phase(phase.get("state", "")) and phase.get("minDur") is None:
                phase.set("minDur", str(DEFAULT_MIN_DUR))
                phase.set("maxDur", str(DEFAULT_MAX_DUR))
        result.append(clone)
    return result


def write_additional(tl_logics: Iterable[ET.Element], path: Path) -> None:
    root = ET.Element("additional")
    root.extend(tl_logics)
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def write_actuated_from_net(net_file: Path, out: Path) -> int:
    """Actuated variant of the static programs of ``net_file`` (synthetic scenarios)."""
    logics = ET.parse(net_file).getroot().findall("tlLogic")
    write_additional(to_actuated(logics), out)
    return len(logics)


def split_tls_programs(net_file: Path, actuated_out: Path) -> int:
    """Keep only static programs in ``net_file``; move the others, made actuated, to a file.

    Used for LuST, whose network carries both its static and its original
    (actuated) programs. Connections reference a traffic light by id and link
    index only, so removing alternative programs leaves the network valid.
    Returns the number of programs moved.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    others = [tl for tl in root.findall("tlLogic") if tl.get("type") != "static"]
    for tl in others:
        root.remove(tl)
    tree.write(net_file, encoding="UTF-8", xml_declaration=True)
    write_additional(to_actuated(others), actuated_out)
    return len(others)
