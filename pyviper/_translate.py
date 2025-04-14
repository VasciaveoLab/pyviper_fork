### ---------- IMPORT DEPENDENCIES ----------
from tqdm import tqdm
from ._load._load_translate import load_human2mouse
import numpy as np
import pandas as pd
from anndata import AnnData
from warnings import warn

### ---------- EXPORT LIST ----------
__all__ = ['translate_adata_index', '_detect_name_type']

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# -----------------------------------------------------------------------------
# ------------------------------ HELPER FUNCTIONS -----------------------------
# -----------------------------------------------------------------------------
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def _detect_name_type(input_array):
    nrow = len(input_array)
    if nrow == 0: 
        return(None) #redundant: case handled by 0-length for loop
    
    human2mouse = load_human2mouse()

    # Create sets for each column (fast search)
    lookup_sets = {
        "mouse_symbol": set(human2mouse["mouse_symbol"].values),
        "human_symbol": set(human2mouse["human_symbol"].values),
        "mouse_ensembl": set(human2mouse["mouse_ensembl"].values),
        "human_ensembl": set(human2mouse["human_ensembl"].values),
        "mouse_entrez": set(human2mouse["mouse_entrez"].values),
        "human_entrez": set(human2mouse["human_entrez"].values)
    }

    for i in range(nrow):
        gene = str(input_array[i])
        if gene in lookup_sets["mouse_symbol"]:
            return "mouse_symbol"
        elif gene in lookup_sets["human_symbol"]:
            return "human_symbol"
        elif gene in lookup_sets["mouse_ensembl"]:
            return "mouse_ensembl"
        elif gene in lookup_sets["human_ensembl"]:
            return "human_ensembl"
        elif gene in lookup_sets["mouse_entrez"]:
            return "mouse_entrez"
        elif gene in lookup_sets["human_entrez"]:
            return "human_entrez"
        
    return None

def _translate_genes_array(current_gene_names, desired_format):

    if desired_format not in ['human_symbol', 'human_ensembl', 'human_entrez',
                          'mouse_symbol', 'mouse_ensembl', 'mouse_entrez']:
        raise ValueError("Error: desired_format is not one the following:"
                         + "\n\t\t mouse_symbol, mouse_ensembl, mouse_entrez,"
                         + "\n\t\t human_symbol, human_ensembl, human_entrez")

    translate_df = load_human2mouse()
    current_format = _detect_name_type(current_gene_names)
    if current_format is None:
        raise ValueError("Error: could not detect current_format as one of the following:"
                         + "\n\t\t mouse_symbol, mouse_ensembl, mouse_entrez,"
                         + "\n\t\t human_symbol, human_ensembl, human_entrez")

    # If the desired format is the same as the current, just return
    if current_format == desired_format:
        return current_gene_names
    
    # Create a dict with current_format as keys and desired_format as values
    translation_dict = dict(zip(translate_df[current_format], translate_df[desired_format]))

    # Apply translation with array operations
    result = np.array([translation_dict.get(str(gene), np.nan) for gene in current_gene_names])

    # Standardize missing values
    result = _uniform_missing_values(result)

    return result

def _uniform_missing_values(arr):
    arr[pd.isna(arr)] = np.nan
    arr[arr == '-1'] = np.nan
    arr[arr == None] = np.nan
    arr[arr == "nan"] = np.nan
    arr[arr == "NaN"] = np.nan
    return arr

def _keep_first_duplicate_strings(arr):
    arr[pd.isna(arr)] = '-1'
    _, index = np.unique(arr, return_index=True)
    result = np.full_like(arr, fill_value=np.nan, dtype=object)
    result[index] = arr[index]
    result[result == '-1'] = np.nan
    return result

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# -----------------------------------------------------------------------------
# ------------------------------- MAIN FUNCTION -------------------------------
# -----------------------------------------------------------------------------
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


def translate_adata_index(
    adata,
    desired_format,
    eliminate = True,
    remove_dupls = True,
    copy = False
):
    """\
    Take adata.var.index.names, replace them with a translation of desired_format,
    and move the original names to a new column in var. The current name format
    and desired_format of the gene names in adata.var.index should be one of the
    following:
        mouse_symbol, mouse_ensembl, mouse_entrez, human_symbol, human_ensembl, or human_entrez

    Parameters
    ----------
    adata
        Gene expression, protein activity or pathways stored in an anndata object.
    desired_format
        Desired format can be one of six strings: "mouse_symbol", "mouse_ensembl",
        "mouse_entrez", "human_symbol", "human_ensembl", or "human_entrez".
    eliminate : default: True
        Whether to eliminate var rows that don't have a translation. Otherwise,
        None will be left in place in the index.
    remove_dupls : default: True
        Whether to eliminate duplicates. Useful in cases where different Ensembl
        IDs match to the same gene symbol.
    copy : default: False
        Whether to return a translated copy (True) or to instead translate the
        original input (False).

    Returns
    -------
    The translated anndata object.
    """
    # So all translation happen on a new adata that is returned. Otherwise, we
    # edit the original, but eliminate doesn't work if the output isn't taken.
    if copy: adata = adata.copy()
    current_format = _detect_name_type(adata.var.index.values)
    adata.var[current_format] = adata.var.index.values.astype(str)
    adata.var[desired_format] = _translate_genes_array(adata.var[current_format], desired_format)
    if remove_dupls: adata.var[desired_format] = _keep_first_duplicate_strings(adata.var[desired_format].values)
    if eliminate:
        adata._inplace_subset_var(~pd.isna(adata.var[desired_format]))
        # adata._inplace_subset_var(adata.var[desired_format] != "nan")
        # adata._inplace_subset_var(adata.var[desired_format] != "NaN")

    adata.var.set_index(desired_format, inplace=True)

    if copy: return adata


def translate(
    adata,
    desired_format,
    eliminate = True,
    remove_dupls = True,
    copy = False
):
    """\
    Performs translation of an AnnData, np.ndarray, or list object. For AnnData,
    takes adata.var.index.names, replaces them with a translation of desired_format,
    and moves the original names to a new column in var. The current name format
    and desired_format of the gene names in adata.var.index should be one of the
    following:
        mouse_symbol, mouse_ensembl, mouse_entrez, human_symbol, human_ensembl, or human_entrez

    Parameters
    ----------
    adata
        Gene expression, protein activity or pathways stored in an anndata
        object. Alternate inputs: np.ndarray or list.
    desired_format
        Desired format can be one of six strings: "mouse_symbol", "mouse_ensembl",
        "mouse_entrez", "human_symbol", "human_ensembl", or "human_entrez".
    eliminate : default: True
        Whether to eliminate var rows that don't have a translation. Otherwise,
        None will be left in place in the index.
    remove_dupls : default: True
        Whether to eliminate duplicates. Useful in cases where different Ensembl
        IDs match to the same gene symbol.
    copy : default: False
        Whether to return a translated copy (True) or to instead translate the
        original AnnData (False). True when given np.ndarray or list.

    Returns
    -------
    Returns a translated object when copy = True or input is np.ndarray or list. Modifies the original when copy = False and input is AnnData.
    """
    if isinstance(adata, AnnData):
        if copy is True:
            return translate_adata_index(adata, desired_format, eliminate, copy)
        else:
            translate_adata_index(adata, desired_format, eliminate, copy)
    elif isinstance(adata, np.ndarray):
        if copy is False:
            warn("np.ndarray supplied:" +
                 " overriding copy=False and returning a translated copy.")
        original_shape = adata.shape
        adata = adata.flatten()
        current_format = _detect_name_type(adata)
        adata = _translate_genes_array(adata, desired_format)
        # Switch to "-1" so shape isn't affected
        if remove_dupls: adata = _keep_first_duplicate_strings(adata)
        adata = adata.reshape(original_shape)
        if eliminate:
            adata = adata[~pd.isna(adata)]
            # adata = adata[adata != "nan"]
            # adata = adata[adata != "NaN"]
        return adata
    elif isinstance(adata, list):
        if copy is False:
            warn("list supplied to translate:" +
                 " overriding copy=False and returning a translated copy.")
        adata = np.array(adata)
        current_format = _detect_name_type(adata)
        adata = _translate_genes_array(adata, desired_format)
        if remove_dupls: adata = _keep_first_duplicate_strings(adata)
        if eliminate:
            adata = adata[~pd.isna(adata)]
            # adata = adata[adata != "nan"]
            # adata = adata[adata != "NaN"]
        adata = list(adata)
        return adata
