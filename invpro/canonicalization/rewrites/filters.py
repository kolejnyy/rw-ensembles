"""
Filtering used during rewriting (e.g. namespace allowlists for rewrites).
"""

from __future__ import annotations

import re
from typing import List, Optional

from invpro.canonicalization.rewrites.cache import RewriteEntry

# Match Lean namespace prefix: uppercase identifier followed by dot (e.g. RCLike., Real., Int.)
NAMESPACE_PREFIX_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]*)\.", re.ASCII)


def lemma_namespaces_from_tactic(tactic: str) -> list[str]:
    """
    Extract the namespace (first component before the first dot) of each
    *qualified* premise/lemma in an rw tactic. Bare names (e.g. add_comm) have
    no dot and are not included. Handles "← Name" and "Name" forms.
    E.g. "rw [add_comm] at h" -> []; "rw [← Int.natAbs_sq] at h₂" -> ["Int"].
    """
    namespaces: list[str] = []
    if "rw [" not in tactic:
        return namespaces
    start = tactic.find("rw [") + len("rw [")
    end = tactic.find("]", start)
    if end == -1:
        return namespaces
    inner = tactic[start:end]
    for part in inner.split(","):
        name = part.strip()
        if name.startswith("\u2190 "):
            name = name[2:].strip()
        elif name.startswith("\u2190"):
            name = name[1:].strip()
        if not name or "." not in name:
            continue
        prefix = name.split(".", 1)[0]
        namespaces.append(prefix)
    return namespaces


def namespaces_from_premise(premise: str) -> list[str]:
    """
    Extract namespace prefixes from the premise/resulting-type string.
    Matches uppercase identifiers followed by a dot.
    """
    return list(NAMESPACE_PREFIX_RE.findall(premise))


def filter_rewrites_by_namespace(
    rewrites: list[RewriteEntry],
    allowed_namespaces: list[str],
) -> list[RewriteEntry]:
    """
    Keep only rewrites whose qualified namespaces (from tactic and premise) are
    in allowed_namespaces. Checks both the tactic and the premise/resulting-type
    string (tactic may use shortened form while premise contains qualified names).
    """
    allowed = frozenset(allowed_namespaces)
    out: list[RewriteEntry] = []
    for entry in rewrites:
        ns_list = list(
            dict.fromkeys(
                lemma_namespaces_from_tactic(entry.tactic)
                + namespaces_from_premise(entry.premise)
            )
        )
        if all(prefix in allowed for prefix in ns_list):
            out.append(entry)
    return out


def filter_rewrites_by_namespace_blacklist(
    rewrites: list[RewriteEntry],
    namespace_blacklist: list[str],
) -> list[RewriteEntry]:
    """
    Drop rewrites if any extracted namespace appears in ``namespace_blacklist``.
    Namespace extraction checks both tactic and premise/resulting type.
    """
    blocked = frozenset(namespace_blacklist)
    out: list[RewriteEntry] = []
    for entry in rewrites:
        ns_list = list(
            dict.fromkeys(
                lemma_namespaces_from_tactic(entry.tactic)
                + namespaces_from_premise(entry.premise)
            )
        )
        if any(prefix in blocked for prefix in ns_list):
            continue
        out.append(entry)
    return out
