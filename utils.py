#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data calculation.
"""

import gzip
from typing import List

# import sys

import pandas as pd
import numpy as np

try:
    from pandarallel import pandarallel

    PARALLEL = True
except ImportError:
    PARALLEL = False


try:
    from tqdm.notebook import tqdm

    tqdm.pandas()
    TQDM = True
except ImportError:
    TQDM = False

from rdkit.Chem import AllChem as Chem
import rdkit.Chem.Descriptors as Desc

# from rdkit.Chem.MolStandardize import rdMolStandardize
# from rdkit.Chem.MolStandardize.validate import Validator
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.MolStandardize.fragment import LargestFragmentChooser
from rdkit.Chem.MolStandardize.standardize import Standardizer
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from rdkit import rdBase

rdBase.DisableLog("rdApp.info")
# rdBase.DisableLog("rdApp.warn")

# molvs_v = Validator()
molvs_s = Standardizer()
molvs_l = LargestFragmentChooser()
molvs_u = Uncharger()
molvs_t = TautomerCanonicalizer(max_tautomers=100)

# params = rdMolStandardize.CleanupParameters()
# params.maxTautomers = 100
# enumerator = rdMolStandardize.TautomerEnumerator()
# uncharger = rdMolStandardize.Uncharger()
# largest = rdMolStandardize.LargestFragmentChooser()
# normal = rdMolStandardize.Normalizer()


def get_value(str_val):
    """convert a string into float or int, if possible."""
    if not str_val:
        return np.nan
    try:
        val = float(str_val)
        if "." not in str_val:
            val = int(val)
    except ValueError:
        val = str_val
    return val


def read_sdf(fn):
    """Create a DataFrame instance from an SD file.
    The input can be a single SD file or a list of files and they can be gzipped (fn ends with `.gz`).
    If a list of files is used, all files need to have the same fields."""

    d = {"Smiles": []}
    ctr = {x: 0 for x in ["In", "Out", "Fail_NoMol"]}
    first_mol = True
    sd_props = set()
    if not isinstance(fn, list):
        fn = [fn]
    for f in fn:
        do_close = True
        if isinstance(f, str):
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "rb")
        else:
            file_obj = f
            do_close = False
        reader = Chem.ForwardSDMolSupplier(file_obj)
        for mol in reader:
            ctr["In"] += 1
            if not mol:
                ctr["Fail_NoMol"] += 1
                continue
            if first_mol:
                first_mol = False
                # Is the SD file name property used?
                name = mol.GetProp("_Name")
                if len(name) > 0:
                    has_name = True
                    d["Name"] = []
                else:
                    has_name = False
                for prop in mol.GetPropNames():
                    sd_props.add(prop)
                    d[prop] = []
            mol_props = set()
            ctr["Out"] += 1
            for prop in mol.GetPropNames():
                if prop in sd_props:
                    mol_props.add(prop)
                    d[prop].append(get_value(mol.GetProp(prop)))
                mol.ClearProp(prop)
            if has_name:
                d["Name"].append(get_value(mol.GetProp("_Name")))
                mol.ClearProp("_Name")

            # append NAN to the missing props that were not in the mol:
            missing_props = sd_props - mol_props
            for prop in missing_props:
                d[prop].append(np.nan)
            d["Smiles"].append(mol_to_smiles(mol))
        if do_close:
            file_obj.close()
    # Make sure, that all columns have the same length.
    # Although, Pandas would also complain, if this was not the case.
    d_keys = list(d.keys())
    if len(d_keys) > 1:
        k_len = len(d[d_keys[0]])
        for k in d_keys[1:]:
            assert k_len == len(d[k]), f"{k_len=} != {len(d[k])}"
    result = pd.DataFrame(d)
    print(ctr)
    return result


def mol_to_smiles(mol, canonical=True):
    """Generate Smiles from mol.

    Returns:
    ========
    The Smiles of the molecule (canonical by default). NANfor failed molecules."""

    if mol is None:
        return np.nan
    try:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
        return smi
    except:
        return np.nan


def drop_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Remove the list of columns from the dataframe.
    Listed columns that are not available in the dataframe are simply ignored."""
    df = df.copy()
    cols_to_remove = set(cols).intersection(set(df.keys()))
    df = df.drop(cols_to_remove, axis=1)
    return df


def standardize_mol(mol, canonicalize_tautomer=False):
    """Standardize the molecule structures.

    Returns:
    ========
    Smiles of the standardized molecule. NAN for failed molecules."""

    if mol is None:
        return np.nan
    mol = molvs_l.choose(mol)
    mol = molvs_u.uncharge(mol)
    mol = molvs_s.standardize(mol)
    if canonicalize_tautomer:
        mol = molvs_t.canonicalize(mol)
    # mol = largest.choose(mol)
    # mol = uncharger.uncharge(mol)
    # mol = normal.normalize(mol)
    # mol = enumerator.Canonicalize(mol)
    return mol_to_smiles(mol)


def apply_to_smiles(
    df: pd.DataFrame, smiles_col: str, new_col: str, func, parallel=False, workers=6
) -> pd.DataFrame:
    """Calculation of chemical properties,
    directly on the Smiles.

    Parameters:
    ===========
    df: Pandas DataFrame
    smiles_col: Name of the Smiles column
    new_col: Name of the new column
    func: function to apply to the mol object.
        If the generation of the intermediary mol object fails, NAN is returned.
    parallel: Set to True when the function should be run in parallel (default: False).
        pandarallel has to be installed for this.
    workers: Number of workers to be used when running in parallel.

    Returns:
    ========
    New DataFrame with the calculated property.
    """

    def _apply(smi):
        if not isinstance(smi, str):
            return np.nan
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.nan
        try:
            result = func(mol)
            return result
        except:
            return np.nan

    df = df.copy()
    if parallel:
        if not PARALLEL:
            print("Parallel option not available. Please install pandarallel.")
            print("Using single core calculation.")
        else:
            pandarallel.initialize(nb_workers=workers, progress_bar=TQDM)
            df[new_col] = df[smiles_col].parallel_apply(_apply)
        return df

    if TQDM:
        df[new_col] = df[smiles_col].progress_apply(_apply)
    else:
        df[new_col] = df[smiles_col].apply(_apply)
    return df


def get_atom_set(mol):
    result = set()
    for at in mol.GetAtoms():
        result.add(at.GetAtomicNum())
    return result


def filter_mols(df: pd.DataFrame, smiles_col, filter) -> pd.DataFrame:
    """Apply different filters to the molecules.
    Parameters:
    ===========
    filter [str or list of strings]: The name of the filter to apply.
        Available filters:
            - Isotope: Keep only non-isotope molecules
            - MedChemAtoms: Keep only molecules with MedChem atoms
            - MinHeavyAtoms: Keep only molecules with 3 or more heacy atoms
            - MaxHeavyAtoms: Keep only molecules with 75 or less heacy atoms

    """
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}

    def has_non_medchem_atoms(mol):
        if len(get_atom_set(mol) - medchem_atoms) > 0:
            return True
        return False

    def has_isotope(mol) -> bool:
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    df = df.copy()
    if isinstance(filter, str):
        filter = [filter]
    calc_ha = False
    cols_to_remove = []
    print(f"Applying filters ({len(filter)})...")
    for filt in filter:
        if filt == "Isotope":
            df = apply_to_smiles(df, smiles_col, "FiltIsotope", has_isotope)
            df = df.query("not FiltIsotope")
            cols_to_remove.append("FiltIsotope")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MedChemAtoms":
            df = apply_to_smiles(
                df, smiles_col, "FiltNonMCAtoms", has_non_medchem_atoms
            )
            df = df.query("not FiltNonMCAtoms")
            cols_to_remove.append("FiltNonMCAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MinHeavyAtoms":
            if not calc_ha:
                df = apply_to_smiles(
                    df, smiles_col, "FiltHeavyAtoms", Desc.HeavyAtomCount
                )
                calc_ha = True
            df = df.query("FiltHeavyAtoms >= 3")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MaxHeavyAtoms":
            if not calc_ha:
                df = apply_to_smiles(
                    df, smiles_col, "FiltHeavyAtoms", Desc.HeavyAtomCount
                )
                calc_ha = True
            df = df.query("FiltHeavyAtoms <= 75")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt}: ", end="")
        else:
            print()
            raise ValueError(f"Unknown filter: {filt}.")
        print(len(df))
    df = drop_cols(df, cols_to_remove)
    return df
