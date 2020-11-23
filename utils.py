#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data calculation.
"""

import gzip
from typing import List, Dict, Callable

# import sys

import pandas as pd
from pandas import DataFrame
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
from rdkit.Chem import Mol
import rdkit.Chem.Descriptors as Desc
import rdkit.Chem.rdMolDescriptors as rdMolDesc
from rdkit.Chem import Crippen

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


def mol_to_smiles(mol: Mol, canonical: bool = True):
    """Generate Smiles from mol.
    Returns:
    ========
    The Smiles of the molecule (canonical by default). NAN for failed molecules."""

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
    df: pd.DataFrame,
    smiles_col: str,
    funcs: Dict[str, Callable],
    parallel=False,
    workers=6,
) -> pd.DataFrame:
    """Calculation of chemical properties,
    directly on the Smiles.
    Parameters:
    ===========
    df: Pandas DataFrame
    smiles_col: Name of the Smiles column
    funcs: A dict of names and functions to apply to the mol object.
        The keys are the names of the generated columns,
        the values are the functions.
        If the generation of the intermediary mol object fails, NAN is returned.
    parallel: Set to True when the function should be run in parallel (default: False).
        pandarallel has to be installed for this.
    workers: Number of workers to be used when running in parallel.
    Returns:
    ========
    New DataFrame with the calculated properties.

    Example:
    ========
    `df` is a DataFrame that contains a "Smiles" column.
    >>> from rdkit.Chem import Descriptors as Desc
    >>> df2 = apply_to_smiles(df, "Smiles", {"MW": Desc.MolWt, "LogP": Desc.MolLogP})
    """

    func_items = funcs.items()
    func_keys = {i: x[0] for i, x in enumerate(func_items)}
    func_vals = [x[1] for x in func_items]

    def _apply(smi):
        if not isinstance(smi, str):
            return np.nan
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.nan
        res = []
        for f in func_vals:
            try:
                r = f(mol)
                res.append(r)
            except:
                r.append(np.nan)
        return pd.Series(res)

    df = df.copy()
    fallback = True
    if parallel:
        if not PARALLEL:
            print("Parallel option not available. Please install pandarallel.")
            print("Using single core calculation.")
        else:
            pandarallel.initialize(nb_workers=workers, progress_bar=TQDM)
            result = df[smiles_col].parallel_apply(_apply)
            fallback = False

    if fallback:
        if TQDM:
            result = df[smiles_col].progress_apply(_apply)
        else:
            result = df[smiles_col].apply(_apply)
    result = result.rename(columns=func_keys)
    df = pd.concat([df, result], axis=1)
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



def read_tsv(input_tsv: str, smiles_col: str = 'Smiles', parse_smiles: bool = False) -> pd.DataFrame:
    """Read a tsv file, optionnally converting smiles into RDKit molecules.

    Parameters:
    ===========
    input_tsv: Input tsv file
    smiles_col: Name of the Smiles column

    Returns:
    ========
    The parsed tsv as Pandas DataFrame.
    """
    df = pd.read_csv(input_tsv, sep='\t')

    if parse_smiles and smiles_col in df.columns:
        df[smiles_col] = df[smiles_col].map(smiles_to_mol)

    return df


def write_tsv(df: pd.DataFrame, output_tsv: str, smiles_col: str = 'Smiles'):
    """Write a tsv file, converting the RDKit molecule column to smiles.
    If the Smiles column contains RDKit Molecules instead of strings, these are converted to Smiles with default parameters.

    Parameters:
    ===========
    input_tsv: Input tsv file
    smiles_col: Name of the Smiles column

    Returns:
    ========
    The parsed tsv as Pandas DataFrame.
    """
    if len(df) > 0 and smiles_col in df.columns:
        probed = df.iloc[0][smiles_col]
        if isinstance(probed, Mol):
            df[smiles_col] = df[smiles_col].map(mol_to_smiles)
    df.to_csv(output_tsv, sep='\t', index=False)



def count_lipinski_violations(molecular_weight: float, slogp: float, num_hbd: int, num_hba: int) -> int:
    """Apply the filters described in reference (Lipinski's rule of 5) and count how many rules
    are violated. If 0, then the compound is strictly drug-like according to Lipinski et al.

    Ref: Lipinski, J Pharmacol Toxicol Methods. 2000 Jul-Aug;44(1):235-49.

    Parameters:
    ===========
    molecular_weight: Molecular weight
    slogp: LogP computed with RDKit
    num_hbd: Number of Hydrogen Donors
    num_hba: Number of Hydrogen Acceptors

    Returns:
    ========
    The number of violations of the Lipinski's rule.
    """
    n = 0
    if molecular_weight < 150 or molecular_weight > 500:
        n += 1
    if slogp > 5:
        n += 1
    if num_hbd > 5:
        n += 1
    if num_hba > 10:
        n += 1
    return n


def count_veber_violations(num_rotatable_bonds: int, tpsa: float) -> int:
    """Apply the filters described in reference (Veber's rule) and count how many rules
    are violated. If 0, then the compound is strictly drug-like according to Veber et al.

    Ref: Veber DF, Johnson SR, Cheng HY, Smith BR, Ward KW, Kopple KD (June 2002).
    "Molecular properties that influence the oral bioavailability of drug candidates".
    J. Med. Chem. 45 (12): 2615–23.

    Parameters:
    ===========
    num_rotatable_bonds: Number of rotatable bonds
    tpsa: Topological Polar Surface Area

    Returns:
    ========
    The number of violations of the Veber's rule.
    """
    n = 0
    if num_rotatable_bonds > 10:
        n += 1
    if tpsa > 140:
        n += 1
    return n



def get_min_ring_size(mol: Mol) -> int:
    """Return the minimum ring size of a molecule. If the molecule is linear, 0 is returned.

    Parameters:
    ===========
    mol: The input molecule

    Returns:
    ========
    The minimal ring size of the input molecule
    """

    ring_sizes = [len(x) for x in mol.GetRingInfo().AtomRings()]
    try:
        return min(ring_sizes)
    except ValueError:
        return 0


def get_max_ring_size(mol: Mol) -> int:
    """Return the maximum ring size of a molecule. If the molecule is linear, 0 is returned.

    Parameters:
    ===========
    mol: The input molecule

    Returns:
    ========
    The maximal ring size of the input molecule
    """
    ring_sizes = [len(x) for x in mol.GetRingInfo().AtomRings()]
    try:
        return max(ring_sizes)
    except ValueError:
        return 0


def compute_descriptors(mol: Mol, descriptors_list: list = None) -> dict:
    """Compute predefined descriptors for a molecule.
    If the parsing of a molcul

    Parameters:
    ===========
    mol: The input molecule
    descriptors_list: A list of descriptors, in case the user wants to compute less than default.

    Returns:
    ========
    A dictionary with computed descriptors with syntax such as descriptor_name: value.
    """
    # predefined descriptors
    descriptors = {
                    # classical molecular descriptors
                    'num_heavy_atoms': lambda x: x.GetNumAtoms(),
                    'molecular_weight': lambda x: round(Desc.ExactMolWt(x), 4),
                    'num_rings': lambda x: rdMolDesc.CalcNumRings(x),
                    'num_rings_arom': lambda x: rdMolDesc.CalcNumAromaticRings(x),
                    'num_rings_ali': lambda x: rdMolDesc.CalcNumAliphaticRings(x),
                    'num_hbd': lambda x: rdMolDesc.CalcNumLipinskiHBD(x),
                    'num_hba': lambda x: rdMolDesc.CalcNumLipinskiHBA(x),
                    'slogp': lambda x: round(Crippen.MolLogP(x), 4),
                    'tpsa': lambda x: round(rdMolDesc.CalcTPSA(x), 4),
                    'num_rotatable_bond': lambda x: rdMolDesc.CalcNumRotatableBonds(x),
                    'num_atom_oxygen': lambda x: len([a for a in x.GetAtoms() if a.GetAtomicNum() == 8]),
                    'num_atom_nitrogen': lambda x: len([a for a in x.GetAtoms() if a.GetAtomicNum() == 7]),
                    # custom molecular descriptors
                    'ring_size_min': get_min_ring_size,
                    'ring_size_max': get_max_ring_size,
                    'frac_sp3': lambda x: rdMolDesc.CalcFractionCSP3(x),
                    }

    # update the list of descriptors to compute with whatever descriptor names are in the prodived list,
    # if the list contains an unknown descriptor, a KeyError will be raised.
    if descriptors_list is not None:
        descriptors = {k: v for k, v in descriptors.items() if k in descriptors_list}

    # parse smiles on the fly
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    # if parsing fails, return dict with missing values
    if mol is None:
        return {k: None for k in descriptors.keys()}

    # run
    try:
        # compute molecular descriptors
        d = {k: v(mol) for k, v in descriptors.items()}
        # annotate subsets
        d['num_lipinski_violations'] = count_lipinski_violations(d['molecular_weight'], d['slogp'], d['num_hbd'], d['num_hba'])
        d['num_veber_violations'] = count_veber_violations(d['num_rotatable_bond'], d['tpsa'])
    except ValueError:
        d = {k: None for k in descriptors.keys()}

    return d



def compute_descriptors_df(df: DataFrame, smiles_col: str) -> DataFrame:
    """Compute descriptors on the smiles column of a DataFrame.
    The Smiles is parsed on the fly only once.


    Parameters:
    ===========
    df: The input DataFrame
    smiles_col: The name of the column with the molecules in smiles format

    Returns:
    ========
    The input dictionary concatenated with the computed descriptors

    """
    return pd.concat([df, df.apply(lambda x: compute_descriptors(x[smiles_col]), axis=1).apply(pd.Series)], axis=1)
