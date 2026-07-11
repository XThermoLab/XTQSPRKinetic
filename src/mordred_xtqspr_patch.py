# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Hamidreza Najafi / XThermo Lab
# Maintainer: Hamidreza Najafi (XThermo Lab)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
XTQSPRKinetic explicit-hydrogen-aware Mordred runtime patch.

Purpose
-------
This module applies a controlled runtime patch to Mordred / MordredCommunity
without manually editing site-packages.

It is intended for radical-rich C/H kinetic reaction networks where the default
Mordred hydrogen handling under-represents hydrogen-sensitive species such as:

    CH4, [CH3], [CH2], [CH], [H], [H][H]

Important
---------
This creates an XTQSPRKinetic modified Mordred descriptor mode. When enabled,
the descriptor set should be reported as:

    XTQSPRKinetic explicit-hydrogen-aware modified Mordred/MordredCommunity
    descriptors

not as fully unmodified standard Mordred.

The patch must be applied before constructing mordred.Calculator(...). For
joblib/loky parallel Mordred calculation, the patch must be applied inside each
worker process before that worker constructs its local Calculator.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

import numpy as np


_PATCH_APPLIED = False
_PATCH_REPORT: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Modules/classes where the original Mordred code has:
#
#     explicit_hydrogens = False
#
# These are the locations that need runtime conversion to True for the
# XTQSPRKinetic explicit-H descriptor mode.
# ---------------------------------------------------------------------------
_MORDRED_EXPLICIT_H_FALSE_MODULES: List[str] = [
    "ABCIndex",
    "AdjacencyMatrix",
    "BCUT",
    "BalabanJ",
    "BaryszMatrix",
    "BertzCT",
    "CarbonTypes",
    "Chi",
    "DetourMatrix",
    "DistanceMatrix",
    "EState",
    "EccentricConnectivityIndex",
    "ExtendedTopochemicalAtom",
    "FragmentComplexity",
    "HydrogenBond",
    "KappaShapeIndex",
    "LogS",
    "MoeType",
    "MolecularDistanceEdge",
    "MolecularId",
    "PathCount",
    "RingCount",
    "RotatableBond",
    "SLogP",
    "TopologicalCharge",
    "TopologicalIndex",
    "VertexAdjacencyInformation",
    "WalkCount",
    "WienerIndex",
    "ZagrebIndex",
]


# ---------------------------------------------------------------------------
# Modules/classes where original Mordred already has:
#
#     explicit_hydrogens = True
#
# These do not need patching. They are kept here only for audit completeness.
# ---------------------------------------------------------------------------
_MORDRED_EXPLICIT_H_ALREADY_TRUE_MODULES: List[str] = [
    "Autocorrelation",
    "GeometricalIndex",
    "Weight",
]


_MORDRED_EXPLICIT_H_ALL_MODULES: List[str] = (
    _MORDRED_EXPLICIT_H_FALSE_MODULES
    + _MORDRED_EXPLICIT_H_ALREADY_TRUE_MODULES
)


def mordred_xtqspr_patch_is_applied() -> bool:
    """Return True if the XTQSPRKinetic Mordred runtime patch has been applied."""
    return bool(_PATCH_APPLIED)


def get_mordred_xtqspr_patch_report() -> Dict[str, Any]:
    """Return a copy of the latest patch report."""
    return dict(_PATCH_REPORT)


def _patch_module_atomic_function_references(
    get_valence_electrons_func,
    get_sigma_electrons_func,
    get_intrinsic_state_func,
) -> int:
    """
    Patch already-imported Mordred modules that may hold direct references to
    functions imported from mordred._atomic_property.

    This is important because some modules may do, for example:

        from mordred._atomic_property import get_intrinsic_state

    If those modules were imported before this runtime patch is applied, updating
    only mordred._atomic_property is not always sufficient. This helper updates
    those module-level references too.
    """
    n_updated = 0

    for mod_name, mod in list(sys.modules.items()):
        if not isinstance(mod_name, str):
            continue
        if not mod_name.startswith("mordred."):
            continue
        if mod is None:
            continue

        try:
            if hasattr(mod, "get_valence_electrons"):
                setattr(mod, "get_valence_electrons", get_valence_electrons_func)
                n_updated += 1

            if hasattr(mod, "get_sigma_electrons"):
                setattr(mod, "get_sigma_electrons", get_sigma_electrons_func)
                n_updated += 1

            if hasattr(mod, "get_intrinsic_state"):
                setattr(mod, "get_intrinsic_state", get_intrinsic_state_func)
                n_updated += 1

        except Exception:
            # This is intentionally non-fatal because not every imported object
            # is safely writable.
            pass

    return int(n_updated)


def _patch_explicit_hydrogen_class_flags(module_names: List[str]) -> Dict[str, Any]:
    """
    Change existing class-level explicit_hydrogens=False attributes to True.

    The function patches only classes that already define the attribute. It does
    not create new explicit_hydrogens attributes in classes where Mordred did not
    define one.
    """
    report: Dict[str, Any] = {
        "modules_checked": 0,
        "classes_seen_with_attribute": 0,
        "classes_false_to_true": 0,
        "classes_already_true": 0,
        "classes_other_value": 0,
        "modules_failed": [],
        "patched_classes": [],
        "already_true_classes": [],
    }

    for module_name in module_names:
        report["modules_checked"] += 1

        try:
            mod = __import__(f"mordred.{module_name}", fromlist=["*"])

            for obj_name in dir(mod):
                obj = getattr(mod, obj_name)

                if not isinstance(obj, type):
                    continue

                if not hasattr(obj, "explicit_hydrogens"):
                    continue

                report["classes_seen_with_attribute"] += 1

                try:
                    current = getattr(obj, "explicit_hydrogens")

                    if current is False:
                        setattr(obj, "explicit_hydrogens", True)
                        report["classes_false_to_true"] += 1
                        report["patched_classes"].append(
                            f"mordred.{module_name}.{obj_name}"
                        )

                    elif current is True:
                        report["classes_already_true"] += 1
                        report["already_true_classes"].append(
                            f"mordred.{module_name}.{obj_name}"
                        )

                    else:
                        report["classes_other_value"] += 1

                except Exception:
                    # Keep patch robust if a descriptor class blocks attribute writes.
                    pass

        except Exception as e:
            report["modules_failed"].append(
                {"module": module_name, "error": str(e)}
            )
            logging.warning(
                "Could not inspect/patch Mordred module %s for explicit hydrogens: %s",
                module_name,
                e,
            )

    return report


def _audit_already_true_modules() -> Dict[str, Any]:
    """
    Audit modules/classes that were already explicit_hydrogens=True in original
    Mordred. This is for documentation/checking only; no patch is required.
    """
    report: Dict[str, Any] = {
        "modules_checked": 0,
        "classes_seen_with_attribute": 0,
        "classes_true": 0,
        "classes_false": 0,
        "modules_failed": [],
        "true_classes": [],
        "false_classes": [],
    }

    for module_name in _MORDRED_EXPLICIT_H_ALREADY_TRUE_MODULES:
        report["modules_checked"] += 1

        try:
            mod = __import__(f"mordred.{module_name}", fromlist=["*"])

            for obj_name in dir(mod):
                obj = getattr(mod, obj_name)

                if not isinstance(obj, type):
                    continue

                if not hasattr(obj, "explicit_hydrogens"):
                    continue

                report["classes_seen_with_attribute"] += 1

                try:
                    current = getattr(obj, "explicit_hydrogens")

                    if current is True:
                        report["classes_true"] += 1
                        report["true_classes"].append(
                            f"mordred.{module_name}.{obj_name}"
                        )

                    elif current is False:
                        report["classes_false"] += 1
                        report["false_classes"].append(
                            f"mordred.{module_name}.{obj_name}"
                        )

                except Exception:
                    pass

        except Exception as e:
            report["modules_failed"].append(
                {"module": module_name, "error": str(e)}
            )
            logging.warning(
                "Could not audit Mordred already-true module %s: %s",
                module_name,
                e,
            )

    return report


def apply_xtqspr_mordred_patch() -> Dict[str, Any]:
    """
    Apply the XTQSPRKinetic explicit-hydrogen-aware Mordred runtime patch.

    Returns
    -------
    dict
        Patch report summarizing atomic-property patching, explicit-H class
        updates, and audit information.

    Notes
    -----
    This function intentionally does NOT call:

        calc.config(explicit_hydrogens=True)

    because validation showed that Mordred's public config pathway did not
    reproduce the manually patched explicit-H descriptor behavior for the target
    radical-rich C/H reaction network.
    """
    global _PATCH_APPLIED
    global _PATCH_REPORT

    if _PATCH_APPLIED:
        return {
            **_PATCH_REPORT,
            "already_applied": True,
        }

    report: Dict[str, Any] = {
        "applied": False,
        "already_applied": False,
        "atomic_property_patched": False,
        "atomic_function_references_updated": 0,
        "explicit_h_false_modules": list(_MORDRED_EXPLICIT_H_FALSE_MODULES),
        "explicit_h_already_true_modules": list(_MORDRED_EXPLICIT_H_ALREADY_TRUE_MODULES),
        "explicit_h_all_modules": list(_MORDRED_EXPLICIT_H_ALL_MODULES),
        "explicit_h_patch_report": None,
        "already_true_audit_report": None,
    }

    # ---------------------------------------------------------------------
    # 1. Patch Mordred atomic-property helper functions.
    # ---------------------------------------------------------------------
    try:
        import mordred._atomic_property as ap

        table = ap._table

        def get_valence_electrons(atom):
            """
            XTQSPRKinetic hydrogen-aware valence helper.

            Relative to original Mordred:
              - H atoms return 1 instead of 0.

            Relative to the earlier broad manual patch:
              - non-H atoms retain the original hydrogen-corrected expression
                instead of being replaced by raw outer valence electron count Zv.

            For denom == 0, return np.nan because the atomic valence quantity is
            mathematically undefined rather than chemically zero.
            """
            N = atom.GetAtomicNum()

            if N == 1:
                return 1

            Zv = table.GetNOuterElecs(N) - atom.GetFormalCharge()
            Z = atom.GetAtomicNum() - atom.GetFormalCharge()
            hi = atom.GetTotalNumHs()
            he = sum(1 for a in atom.GetNeighbors() if a.GetAtomicNum() == 1)
            h = hi + he

            denom = Z - Zv - 1
            if denom == 0:
                return np.nan

            return (Zv - h) / denom

        def get_sigma_electrons(atom):
            """
            Count all sigma neighbors, including explicit hydrogens.

            This is a key C/H-radical-sensitive change relative to the default
            heavy-neighbor-only behavior.
            """
            return sum(1 for a in atom.GetNeighbors())

        def get_intrinsic_state(atom):
            """
            XTQSPRKinetic regularized intrinsic-state helper.

            d == 0 is returned as 0 as a practical regularization for zero-
            connectivity cases in radical/H-only species.
            """
            i = atom.GetAtomicNum()
            d = get_sigma_electrons(atom)
            dv = get_valence_electrons(atom)

            if d == 0:
                return 0

            return ((2.0 / ap.period[i]) ** 2 * dv + 1) / d

        ap.get_valence_electrons = get_valence_electrons
        ap.get_sigma_electrons = get_sigma_electrons
        ap.get_intrinsic_state = get_intrinsic_state

        report["atomic_property_patched"] = True

        # Patch direct references in already-imported Mordred modules.
        report["atomic_function_references_updated"] = (
            _patch_module_atomic_function_references(
                get_valence_electrons,
                get_sigma_electrons,
                get_intrinsic_state,
            )
        )

    except Exception as e:
        logging.warning("Failed to patch mordred._atomic_property: %s", e)
        raise

    # ---------------------------------------------------------------------
    # 2. Patch original explicit_hydrogens=False class attributes to True.
    # ---------------------------------------------------------------------
    explicit_h_patch_report = _patch_explicit_hydrogen_class_flags(
        _MORDRED_EXPLICIT_H_FALSE_MODULES
    )
    report["explicit_h_patch_report"] = explicit_h_patch_report

    # ---------------------------------------------------------------------
    # 3. Audit modules that were already explicit_hydrogens=True.
    # ---------------------------------------------------------------------
    report["already_true_audit_report"] = _audit_already_true_modules()

    _PATCH_APPLIED = True
    report["applied"] = True
    _PATCH_REPORT = dict(report)

    logging.info(
        "Applied XTQSPRKinetic Mordred explicit-H runtime patch: "
        "atomic_property=%s, atomic_refs_updated=%d, false_to_true_classes=%d, "
        "already_true_classes=%d, failed_false_modules=%d",
        report["atomic_property_patched"],
        int(report["atomic_function_references_updated"]),
        int(explicit_h_patch_report.get("classes_false_to_true", 0)),
        int(explicit_h_patch_report.get("classes_already_true", 0)),
        len(explicit_h_patch_report.get("modules_failed", [])),
    )

    return dict(report)