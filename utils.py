#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data calculation.
"""

import gzip
from typing import List, Dict, Set, Callable, Union

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

molvs_s = Standardizer()
molvs_l = LargestFragmentChooser()
molvs_u = Uncharger()
molvs_t = TautomerCanonicalizer(max_tautomers=100)


# from sklearn import decomposition
# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns


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


def read_sdf(fn, merge_prop: str = None, merge_list: Union[List, Set] = None):
    """Create a DataFrame instance from an SD file.
    The input can be a single SD file or a list of files and they can be gzipped (fn ends with `.gz`).
    If a list of files is used, all files need to have the same fields.
    The molecules will be converted to Smiles.

    Parameters:
    ===========
    merge_prop: A property in the SD file on which the file should be merge
        during reading.
    merge_list: A list or set of values on which to merge.
        Only the values of the list are kept.
    """

    d = {"Smiles": []}
    ctr = {x: 0 for x in ["In", "Out", "Fail_NoMol"]}
    if merge_prop is not None:
        ctr["NotMerged"] = 0
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
            if merge_prop is not None:
                # Only keep the record when the `merge_prop` value is in `merge_list`:
                if get_value(mol.GetProp(merge_prop)) not in merge_list:
                    ctr["NotMerged"] += 1
                    continue
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


def mol_to_smiles(mol: Mol, canonical: bool = True) -> str:
    """Generate Smiles from mol.

    Parameters:
    ===========
    mol: the input molecule
    canonical: whether to return the canonical Smiles or not

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


def smiles_to_mol(smiles: str) -> Mol:
    """Generate a RDKit Molecule from a Smiles.

    Parameters:
    ===========
    smiles: the input string

    Returns:
    ========
    The RDKit Molecule. If the Smiles parsing failed, NAN instead.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        return np.nan
    except ValueError:
        return np.nan


def drop_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Remove the list of columns from the dataframe.
    Listed columns that are not available in the dataframe are simply ignored."""
    df = df.copy()
    cols_to_remove = set(cols).intersection(set(df.keys()))
    df = df.drop(cols_to_remove, axis=1)
    return df


def standardize_mol(mol, remove_stereo=False, canonicalize_tautomer=False):
    """Standardize the molecule structures.
    Returns:
    ========
    Smiles of the standardized molecule. NAN for failed molecules."""

    if mol is None:
        return np.nan
    mol = molvs_l.choose(mol)
    mol = molvs_u.uncharge(mol)
    mol = molvs_s.standardize(mol)
    if remove_stereo:
        mol = molvs_s.stereo_parent(mol)
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
            res = [np.nan] * len(func_vals)
            return pd.Series(res)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            res = [np.nan] * len(func_vals)
            return pd.Series(res)
        res = []
        for f in func_vals:
            try:
                r = f(mol)
                res.append(r)
            except:
                res.append(np.nan)
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


def filter_mols(
    df: pd.DataFrame, smiles_col: Union[str, List[str]], filter
) -> pd.DataFrame:
    """Apply different filters to the molecules.

    Parameters:
    ===========
    filter [str or list of strings]: The name of the filter to apply.
        Available filters:
            - Isotopes: Keep only non-isotope molecules
            - MedChemAtoms: Keep only molecules with MedChem atoms
            - MinHeavyAtoms: Keep only molecules with 3 or more heacy atoms
            - MaxHeavyAtoms: Keep only molecules with 75 or less heacy atoms
            - Duplicates: Remove duplicates by InChiKey
    """
    available_filters = {
        "Isotopes",
        "MedChemAtoms",
        "MinHeavyAtoms",
        "MaxHeavyAtoms",
        "Duplicates",
    }
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
    for filt in filter:
        if filt not in available_filters:
            raise ValueError(f"Unknown filter: {filt}")
    calc_ha = False
    cols_to_remove = []
    print(f"Applying filters ({len(filter)})...")
    for filt in filter:
        if filt == "Isotopes":
            df = apply_to_smiles(df, smiles_col, {"FiltIsotopes": has_isotope})
            df = df.query("FiltIsotopes == False")
            cols_to_remove.append("FiltIsotopes")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MedChemAtoms":
            df = apply_to_smiles(
                df, smiles_col, {"FiltNonMCAtoms": has_non_medchem_atoms}
            )
            df = df.query("FiltNonMCAtoms == False")
            cols_to_remove.append("FiltNonMCAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MinHeavyAtoms":
            if not calc_ha:
                df = apply_to_smiles(
                    df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                )
                calc_ha = True
            df = df.query("FiltHeavyAtoms >= 3")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MaxHeavyAtoms":
            if not calc_ha:
                df = apply_to_smiles(
                    df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                )
                calc_ha = True
            df = df.query("FiltHeavyAtoms <= 75")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "Duplicates":
            df = apply_to_smiles(
                df, smiles_col, {"FiltInChiKey": Chem.inchi.MolToInchiKey}
            )
            df = df.drop_duplicates(subset="FiltInChiKey")
            cols_to_remove.append("FiltInChiKey")
            print(f"Applied filter {filt}: ", end="")
        else:
            print()
            raise ValueError(f"Unknown filter: {filt}.")
        print(len(df))
    df = drop_cols(df, cols_to_remove)
    return df


def read_tsv(
    input_tsv: str, smiles_col: str = "Smiles", parse_smiles: bool = False
) -> pd.DataFrame:
    """Read a tsv file, optionnally converting smiles into RDKit molecules.

    Parameters:
    ===========
    input_tsv: Input tsv file
    smiles_col: Name of the Smiles column

    Returns:
    ========
    The parsed tsv as Pandas DataFrame.
    """
    df = pd.read_csv(input_tsv, sep="\t")

    if parse_smiles and smiles_col in df.columns:
        df[smiles_col] = df[smiles_col].map(smiles_to_mol)

    return df


def write_tsv(df: pd.DataFrame, output_tsv: str, smiles_col: str = "Smiles"):
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
    df.to_csv(output_tsv, sep="\t", index=False)


def count_lipinski_violations(
    molecular_weight: float, slogp: float, num_hbd: int, num_hba: int
) -> int:
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
    If the parsing of a molecule fails, then an Nan values are generated for all properties.

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
        "num_heavy_atoms": lambda x: x.GetNumAtoms(),
        "molecular_weight": lambda x: round(Desc.ExactMolWt(x), 4),
        "num_rings": lambda x: rdMolDesc.CalcNumRings(x),
        "num_rings_arom": lambda x: rdMolDesc.CalcNumAromaticRings(x),
        "num_rings_ali": lambda x: rdMolDesc.CalcNumAliphaticRings(x),
        "num_hbd": lambda x: rdMolDesc.CalcNumLipinskiHBD(x),
        "num_hba": lambda x: rdMolDesc.CalcNumLipinskiHBA(x),
        "slogp": lambda x: round(Crippen.MolLogP(x), 4),
        "tpsa": lambda x: round(rdMolDesc.CalcTPSA(x), 4),
        "num_rotatable_bond": lambda x: rdMolDesc.CalcNumRotatableBonds(x),
        "num_atom_oxygen": lambda x: len(
            [a for a in x.GetAtoms() if a.GetAtomicNum() == 8]
        ),
        "num_atom_nitrogen": lambda x: len(
            [a for a in x.GetAtoms() if a.GetAtomicNum() == 7]
        ),
        # custom molecular descriptors
        "ring_size_min": get_min_ring_size,
        "ring_size_max": get_max_ring_size,
        "frac_sp3": lambda x: rdMolDesc.CalcFractionCSP3(x),
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
        d["num_lipinski_violations"] = count_lipinski_violations(
            d["molecular_weight"], d["slogp"], d["num_hbd"], d["num_hba"]
        )
        d["num_veber_violations"] = count_veber_violations(
            d["num_rotatable_bond"], d["tpsa"]
        )
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
    return pd.concat(
        [
            df,
            df.apply(lambda x: compute_descriptors(x[smiles_col]), axis=1).apply(
                pd.Series
            ),
        ],
        axis=1,
    )


def lp(obj, label: str = None, lpad=50, rpad=10):
    """log-printing for different kind of objects"""
    if isinstance(obj, str):
        if label is None:
            label = "String"
        print(f"{label:{lpad}s}: {obj:>{rpad}s}")
        return
    try:
        fval = float(obj)
        if label is None:
            label = "Number"
        if fval == obj:
            print(f"{label:{lpad}s}: {int(obj):{rpad}d}")
        else:
            print(f"{label:{lpad}s}: {obj:{rpad+6}.5f}")
        return
    except (ValueError, TypeError):
        # print("Exception")
        pass

    try:
        shape = obj.shape
        if label is None:
            label = "Shape"
        else:
            label = f"Shape {label}"
        key_str = ""
        try:
            keys = list(obj.keys())
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
        except AttributeError:
            pass
        num_nan_cols = ((~obj.notnull()).sum() > 0).sum()
        has_nan_str = ""
        if num_nan_cols > 0:  # DF has nans
            has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        print(
            f"{label:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        shape = obj.data.shape
        if label is None:
            label = "Shape"
        else:
            label = f"Shape {label}"
        key_str = ""
        try:
            keys = list(obj.data.keys())
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
        except AttributeError:
            pass
        num_nan_cols = ((~obj.data.notnull()).sum() > 0).sum()
        has_nan_str = ""
        if num_nan_cols > 0:  # DF has nans
            has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        print(
            f"{label:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        length = len(obj)
        if label is None:
            label = "Length"
        else:
            label = f"len({label})"
        print(f"{label:{lpad}s}: {length:{rpad}d}")
        return
    except (TypeError, AttributeError):
        pass

    if label is None:
        label = "Object"
    print(f"{label:{lpad}s}: {obj}")


def get_pc_feature_contrib(model: PCA, features: list) -> DataFrame:
    """Get the feature contribution to each Principal Component.

    Parameters:
    ===========
    model: The PCA object
    descriptors_list: The list of feature names that were used for the PCA.

    Returns:
    ========
    A DataFrame with the feature contribution.
    """
    # associate features and pc feature contribution
    ds = []
    for pc in model.components_:
        ds.append(
            {k: np.abs(v) for k, v in zip(features, pc)}
        )  # absolute value of contributions because only the magnitude of the contribution is of interest
    df_feature_contrib = (
        pd.DataFrame(ds, index=[f"PC{i+1}_feature_contrib" for i in range(3)])
        .T.reset_index()
        .rename({"index": "Feature"}, axis=1)
    )

    # compute PC ranks
    for c in df_feature_contrib.columns:
        if not c.endswith("_feature_contrib"):
            continue
        df_feature_contrib = df_feature_contrib.sort_values(
            c, ascending=False
        ).reset_index(drop=True)
        df_feature_contrib[f"{c.split('_')[0]}_rank"] = df_feature_contrib.index + 1

    # clean-up
    return df_feature_contrib.sort_values("Feature").reset_index(drop=True)


def format_pc_feature_contrib(df_feature_contrib: DataFrame) -> DataFrame:
    """
    Parameters:
    ===========
    df_feature_contrib: The DataFrame with PC feature contributions

    Returns:
    ========
    A rearranged DataFrame with the feature contribution, with common column names and each PC as different rows.
    """
    pcs = list(
        set([c.split("_")[0] for c in df_feature_contrib.columns if c.startswith("PC")])
    )
    # init empty DataFrame
    df = pd.DataFrame(None, columns=["PC", "Feature", "Contribution", "Rank"])
    for pc in pcs:
        df_tmp = df_feature_contrib[
            ["Feature", f"{pc}_feature_contrib", f"{pc}_rank"]
        ].rename(
            {f"{pc}_feature_contrib": "Contribution", f"{pc}_rank": "Rank"}, axis=1
        )
        df_tmp["PC"] = pc
        df = pd.concat([df, df_tmp])

    return df.reset_index(drop=True).sort_values(["PC", "Feature"])


def get_pc_var(model):
    feature_contrib = model.explained_variance_ratio_
    pcs = [f"PC{i+1}" for i in range(len(feature_contrib))]
    # generate the variance data
    df_pc_var = pd.DataFrame(
        {
            "var": feature_contrib,
            "PC": pcs,
        }
    )
    df_pc_var["var_perc"] = df_pc_var["var"].map(lambda x: f"{x:.2%}")
    return df_pc_var


def plot_pc_var(model):
    """Plot the explained variance of each principal component."""
    df_pc_var = get_pc_var(model)
    total_var = df_pc_var["var"].sum()

    # generate the variance plot
    # initplot
    # plt.figure(figsize=(12, 9))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    fig.suptitle("Variance Explained by Principal Components", fontsize=30)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)
    x_label = "Principal Components"
    y_label = "% of Total Variance"
    # create the plot
    ax = sns.barplot(x="PC", y="var", data=df_pc_var, color="gray")
    # customize the plot
    ax.set_title(f"Total Variance Explained: {total_var:.2%}", fontsize=20, y=1.02)
    ax.tick_params(labelsize=20)
    ax.set_xlabel(x_label, fontsize=25, labelpad=20)
    ax.set_ylabel(y_label, fontsize=25, labelpad=20)
    ylabels = [f"{x:,.0%}" for x in ax.get_yticks()]
    ax.set_yticklabels(ylabels)
    # add % on the bars
    for a, i in zip(ax.patches, range(len(df_pc_var.index))):
        row = df_pc_var.iloc[i]
        ax.text(
            row.name,
            a.get_height(),
            row["var_perc"],
            color="black",
            ha="center",
            fontdict={"fontsize": 20},
        )

    plt.tight_layout()
    figure = ax.get_figure()
    return figure


def plot_pc_proj(df_pc):
    """Plot the 3 first Principal Components as three 2D projections."""
    fig_size = (32, 12)
    palette = [
        "#2CA02C",
        "#378EBF",
        "#EB5763",
    ]

    sns.set(rc={"figure.figsize": fig_size})
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    fig.suptitle("Principal Component Analysis", fontsize=30)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)
    for i, col_pairs in enumerate([["PC1", "PC2"], ["PC1", "PC3"], ["PC2", "PC3"]]):
        plt.subplot(1, 3, i + 1)
        x_label = col_pairs[0]
        y_label = col_pairs[1]
        ax = sns.scatterplot(
            x=x_label,
            y=y_label,
            data=df_pc,
            hue="dataset",  # color by cluster
            legend=True,
            palette=palette,
            alpha=0.5,
            edgecolor="none",
        )
        ax.set_title(f"{x_label} and {y_label}", fontsize=24, y=1.02)

    plt.tight_layout()
    figure = ax.get_figure()
    figure.subplots_adjust(bottom=0.2)
    return figure


def plot_pc_feature_contrib(df_pc_feature_contrib):
    """Plot the feature contribution to each Principal Component."""
    fig_size = (32, 12)
    sns.set(rc={"figure.figsize": fig_size})
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    sns.set_context("paper", font_scale=2)

    g = sns.catplot(
        data=df_pc_feature_contrib,
        kind="bar",
        x="Feature",
        y="Contribution",
        hue="PC",
        ci="sd",
        palette="gray",
        size=18,
        legend=False,
    )
    for ax in g.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    g.fig.suptitle(
        "Absolute Feature Contribution to Principal Components", fontsize=30
    )  # can also get the figure from plt.gcf()
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    fig = plt.gcf()

    return fig
