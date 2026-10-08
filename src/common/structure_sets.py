import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StructureSet:
    """One source of protein structures and the folders generated from it."""

    description: str
    # Appended to every folder generated from this set; empty keeps the original paths.
    dir_suffix: str
    # Run tag part, so models and results from different structure sets never share a file.
    tag: str
    # Davis name to PDB file stem, for names the set spells differently.
    file_stems: Mapping[str, str] = field(default_factory=dict)


# Each pair is the same protein under two spellings.
KINASE_DOMAIN_FILE_STEMS = {
    "ABL1": "ABL1-nonphosphorylated",
    "ABL1(E255K)": "ABL1(E255K)-phosphorylated",
    "ABL1(F317I)": "ABL1(F317I)-nonphosphorylated",
    "ABL1(F317I)p": "ABL1(F317I)-phosphorylated",
    "ABL1(F317L)": "ABL1(F317L)-nonphosphorylated",
    "ABL1(F317L)p": "ABL1(F317L)-phosphorylated",
    "ABL1(H396P)": "ABL1(H396P)-nonphosphorylated",
    "ABL1(H396P)p": "ABL1(H396P)-phosphorylated",
    "ABL1(M351T)": "ABL1(M351T)-phosphorylated",
    "ABL1(Q252H)": "ABL1(Q252H)-nonphosphorylated",
    "ABL1(Q252H)p": "ABL1(Q252H)-phosphorylated",
    "ABL1(T315I)": "ABL1(T315I)-nonphosphorylated",
    "ABL1(T315I)p": "ABL1(T315I)-phosphorylated",
    "ABL1(Y253F)": "ABL1(Y253F)-phosphorylated",
    "ABL1p": "ABL1-phosphorylated",
    "EGFR(E746A750del)": "EGFR(E746-A750del)",
    "EGFR(L747E749del)": "EGFR(L747-E749del, A750P)",
    "EGFR(L747S752del)": "EGFR(L747-S752del, P753S)",
    "EGFR(L747T751del)": "EGFR(L747-T751del,Sins)",
    "EGFR(L858RT790M)": "EGFR(L858R,T790M)",
    "EGFR(S752I759del)": "EGFR(S752-I759del)",
    "GCN2(KinDom2S808G)": "GCN2(Kin.Dom.2,S808G)",
    "KIT(V559D-T670I)": "KIT(V559D,T670I)",
    "KIT(V559D-V654A)": "KIT(V559D,V654A)",
    "PFCDPK1(Pfalciparum)": "PFCDPK1(P.falciparum)",
    "PFPK5(Pfalciparum)": "PFPK5(P.falciparum)",
    "PKNB(Mtuberculosis)": "PKNB(M.tuberculosis)",
    "RPS6KA4(KinDom.1-N-terminal)": "RPS6KA4(Kin.Dom.1-N-terminal)",
    "RPS6KA4(KinDom.2-C-terminal)": "RPS6KA4(Kin.Dom.2-C-terminal)",
    "RPS6KA5(KinDom.1-N-terminal)": "RPS6KA5(Kin.Dom.1-N-terminal)",
    "RPS6KA5(KinDom.2-C-terminal)": "RPS6KA5(Kin.Dom.2-C-terminal)",
    "RSK1(KinDom.1-N-terminal)": "RSK1(Kin.Dom.1-N-terminal)",
    "RSK1(KinDom.2-C-terminal)": "RSK1(Kin.Dom.2-C-terminal)",
    "RSK2(KinDom.1-N-terminal)": "RSK2(Kin.Dom.1-N-terminal)",
    "RSK3(KinDom.1-N-terminal)": "RSK3(Kin.Dom.1-N-terminal)",
    "RSK3(KinDom.2-C-terminal)": "RSK3(Kin.Dom.2-C-terminal)",
    "RSK4(KinDom.1-N-terminal)": "RSK4(Kin.Dom.1-N-terminal)",
    "RSK4(KinDom.2-C-terminal)": "RSK4(Kin.Dom.2-C-terminal)",
}

STRUCTURE_SETS = {
    "afdb": StructureSet(
        description="full-length AlphaFold DB v6 chains, one per UniProt accession",
        dir_suffix="",
        tag="",
    ),
    "kinase_domain": StructureSet(
        description="kinase-domain AlphaFold models with mutants modeled, from "
        "receptor-ai/3d-prot-dta prot_3d_for_Davis.tar.gz",
        dir_suffix="_kinase_domain",
        tag="pdb2",
        file_stems=KINASE_DOMAIN_FILE_STEMS,
    ),
}

DEFAULT_STRUCTURE_SET = "afdb"


def add_pdb_set_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pdb-set",
        choices=sorted(STRUCTURE_SETS),
        default=DEFAULT_STRUCTURE_SET,
        help="; ".join(
            f"{name}: {structure_set.description}"
            for name, structure_set in STRUCTURE_SETS.items()
        ),
    )
