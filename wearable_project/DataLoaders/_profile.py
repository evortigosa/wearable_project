"""
Wearable Data Processing and Modeling project
Root-level empirical profile of one feature, produced by ``AppleHealthFeatureLoader.profile()``.
``DataLoaders.info()`` describes a feature's static contract from the registries.  ``profile()`` describes what
a particular root actually holds for that feature: participants, rows, bytes on disk, curation status, default
inclusion, acquisition-method coverage, unit-resolution coverage, policy currency, and file integrity.  It is
computed from the phase's state database and one ``stat`` per file, so it never parses a feature CSV and is
cheap enough to run before deciding what to load.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd


ACQUISITION_CAVEAT = (
    "Models conditioning on acquisition method are restricted to the classified subset and may face "
    "substantial source-selection bias."
)


@dataclass(frozen=True)
class ProfileReport:
    """
    Empirical profile of one feature at one root.
    ``summary``
        Aggregate figures for the participants in scope, as a plain dictionary.
    ``participants``
        One row per participant in scope, indexed by ``RegistrationCode``.
    """

    summary: dict[str, Any]
    participants: pd.DataFrame
    text: str

    def as_dict(self) -> dict[str, Any]:
        records = self.participants.reset_index().astype(object).where(self.participants.reset_index().notna(), None)
        return {"summary": self.summary, "participants": records.to_dict(orient="records")}

    def __str__(self) -> str:
        return self.text

    def _repr_markdown_(self) -> str:  # pragma: no cover - notebook display hook
        return "```text\n" + self.text + "\n```"


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{value} B"  # pragma: no cover - unreachable


def _counts(items: dict[str, int] | None) -> str:
    if not items:
        return "none"
    return " · ".join(f"{name} {count:,}" for name, count in sorted(items.items(), key=lambda kv: (-kv[1], kv[0])))


def render_profile_text(summary: dict[str, Any]) -> str:
    """Readable rendering of a profile summary; every figure shown is also present in ``summary``."""

    source_labels = {
        "curation_state": "from the curation state database",
        "native_state": "from the native processing state database",
        "filesystem": "from the filesystem only",
    }
    lines = [
        f"{summary['feature']} profile ({summary['phase']}, {source_labels[summary['source']]})",
        f"root: {summary['root']}",
    ]
    if summary.get("source_note"):
        lines.append(f"note: {summary['source_note']}")
    rows = summary.get("rows")
    lines.append(
        f"participants: {summary['participants']:,}    files on disk: {summary['files_on_disk']:,}    "
        f"rows: {'unknown' if rows is None else f'{rows:,}'}    on disk: {format_bytes(summary.get('bytes_on_disk'))}"
    )

    status = summary.get("status_counts")
    if status is not None:
        lines.append("")
        lines.append(f"Curation status    {_counts(status)}")
        inclusion = summary["default_inclusion"]
        text = f"included {inclusion['included']:,} · excluded {inclusion['excluded']:,}"
        if inclusion["unverifiable_files"]:
            text += f" · {inclusion['unverifiable_files']} file(s) without inclusion counts"
        lines.append(f"Default inclusion  {text}")

    acquisition = summary.get("acquisition")
    if acquisition is not None:
        fraction = acquisition["classified_fraction"]
        head = "not available" if fraction is None else f"classified {fraction:.1%} of rows"
        lines.append(f"Acquisition        {head}")
        lines.append(f"                   {_counts(acquisition['counts'])}")
        if fraction is not None and fraction < 1:
            lines.append(f"                   {ACQUISITION_CAVEAT}")

    units = summary.get("unit_resolution")
    if units is not None:
        if units["resolved_fraction"] is None:
            lines.append("Unit resolution    not applicable (the registry fixes a single unit)")
        else:
            lines.append(
                f"Unit resolution    {units['resolved_fraction']:.1%} resolved "
                f"({units['canonical_value_rows']:,} with a canonical value, "
                f"{units['ambiguous_unit_rows']:,} ambiguous)"
            )

    policy = summary.get("policy")
    if policy is not None:
        if policy["installed_fingerprint"] is None:
            lines.append("Policy             no installed policy to compare against")
        elif policy["files_diverged"]:
            lines.append(
                f"Policy             {policy['files_diverged']} file(s) curated under a policy that differs "
                "from the installed registry"
            )
        else:
            lines.append("Policy             every file curated under the installed policy")

    integrity = summary["integrity"]
    problems = [
        (integrity["files_missing_on_disk"], "recorded in state but missing on disk"),
        (integrity["files_without_state"], "on disk without a state record"),
        (integrity["files_size_mismatch"], "whose size differs from the state record"),
    ]
    found = [f"{count} file(s) {label}" for count, label in problems if count]
    if summary["source"] == "filesystem":
        lines.append("Integrity          not checked (no state database was read)")
    else:
        lines.append("Integrity          " + ("; ".join(found) if found else "every file matches its state record"))

    load = summary["load"]
    limit = load["max_rows"]
    limit_text = "no limit" if limit is None else f"limit {limit:,}"
    if load["default_get_data_rows"] is None:
        lines.append(f"Load               get_data() size unknown without state ({limit_text})")
    else:
        verdict = " -- exceeds the limit" if load["exceeds_max_rows"] else ""
        lines.append(
            f"Load               get_data() would return {load['default_get_data_rows']:,} rows "
            f"({limit_text}){verdict}"
        )
    return "\n".join(lines)
