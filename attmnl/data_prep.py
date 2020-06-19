# -*- coding: utf-8 -*-
"""
@author: Zanna Iscenko
credit for the original version of data prep function T. Braithwaite of pylogit

"""
import warnings


import sys
import time
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
import itertools
from collections import OrderedDict
from collections import Iterable
from numbers import Number
from copy import deepcopy
from numba import jit, njit
import numba 







def create_design_matrix(long_form,
                         specification_dict,
                         alt_id_col,
                         names=None,
                         normalise=False):
    """
    Parameters
    ----------
    long_form : pandas dataframe.
        Contains one row for each available alternative, for each observation.
    specification_dict : OrderedDict.
        Keys are a proper subset of the columns in `long_form_df`. Values are
        either a list or a single string, `"all_diff"` or `"all_same"`. If a
        list, the elements should be:
            - single objects that are within the alternative ID column of
              `long_form_df`
            - lists of objects that are within the alternative ID column of
              `long_form_df`. For each single object in the list, a unique
              column will be created (i.e. there will be a unique coefficient
              for that variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification_dict` values, a single column will be created for
              all the alternatives within iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    names : OrderedDict, optional.
        Should have the same keys as `specification_dict`. For each key:
            - if the corresponding value in `specification_dict` is "all_same",
              then there should be a single string as the value in names.
            - if the corresponding value in `specification_dict` is "all_diff",
              then there should be a list of strings as the value in names.
              There should be one string in the value in names for each
              possible alternative.
            - if the corresponding value in `specification_dict` is a list,
              then there should be a list of strings as the value in names.
              There should be one string the value in names per item in the
              value in `specification_dict`.
        Default == None.
    Returns
    -------
    design_matrix, var_names: tuple with two elements.
        First element is the design matrix, a numpy array with some number of
        columns and as many rows as are in `long_form`. Each column corresponds
        to a coefficient to be estimated. The second element is a list of
        strings denoting the names of each coefficient, with one variable name
        per column in the design matrix.
    """
    ##########
    # Check that the arguments meet this functions assumptions.
    # Fail gracefully if the arguments do not meet the function's requirements.
    #########
#    check_argument_type(long_form, specification_dict)
#
#    ensure_alt_id_in_long_form(alt_id_col, long_form)

#    ensure_specification_cols_are_in_dataframe(specification_dict, long_form)

    # Find out what and how many possible alternatives there are
    unique_alternatives = np.sort(long_form[alt_id_col].unique())
    #num_alternatives = len(unique_alternatives)

#    check_type_and_values_of_specification_dict(specification_dict,
#                                                unique_alternatives)

    # Check the user passed dictionary of names if the user passed such a list
#    if names is not None:
#        ensure_object_is_ordered_dict(names, "names")

#        check_keys_and_values_of_name_dictionary(names,
#                                                 specification_dict,
#                                                 num_alternatives)

    ##########
    # Actually create the design matrix
    ##########
    # Create a list of the columns of independent variables
    independent_vars = []
    # Create a list of variable names
    var_names = []
    # Create a list of variable std deviations for normalisation
    var_scale_factor = []

    # Create the columns of the design matrix based on the specification dict.
    for variable in specification_dict:
        
        var_values = long_form[variable].values
        
        if (normalise and variable is not "intercept"): #and  
            #(var_values.dtype==float or var_values.dtype==int)):
              #var_values.dtype!=bool):
              
                scale_factor = var_values.std()
                var_values = (var_values - var_values.mean())/var_values.std()            
        else:
            scale_factor = 1.0
        
        specification = specification_dict[variable]
        
        if specification == "all_same":
            # Create the variable column
            independent_vars.append(var_values)
            # Create the column name
            var_names.append(variable)
            var_scale_factor.append(scale_factor)
            
        elif specification == "all_diff":
            for alt in unique_alternatives:
                # Create the variable column
                independent_vars.append((long_form[alt_id_col] == alt).values *
                                        var_values)  #creates a col for every group and sets data for others to 0
                # create the column name
                var_names.append("{}_{}".format(variable, alt))
                var_scale_factor.append(scale_factor)
        else:
            for group in specification:
                if isinstance(group, list):
                    if (normalise and variable is "intercept"):
                        var = long_form[alt_id_col].isin(group).values * var_values                                     
                        scale_factor = var.std()
                        var = (var - var.mean())/var.std()    
                        independent_vars.append(var)                                     
                    else:                 
                    # Create the variable column                    
                        independent_vars.append(
                                     long_form[alt_id_col].isin(group).values *
                                     var_values)
                    # Create the column name
                    var_names.append("{}_{}".format(variable, str(group)))

                else:  # the group is an integer
                    # Create the variable column
                    new_col_vals = ((long_form[alt_id_col] == group).values *
                                    var_values)
                    independent_vars.append(new_col_vals)
                    # Create the column name
                    var_names.append("{}_{}".format(variable, group))
                    
                var_scale_factor.append(scale_factor)

    # Create the final design matrix
    design_matrix = np.hstack((x[:, None] for x in independent_vars)) #Stack arrays in sequence horizontally (column wise).

    # Use the list of names passed by the user, if the user passed such a list
    if names is not None:
        var_names = []
        for value in names.values():
            if isinstance(value, str):
                var_names.append(value)
            else:
                for inner_name in value:
                    var_names.append(inner_name)
    
    # Create the final set of variable scalers
    var_scale_factor = np.array(var_scale_factor)
    
    return design_matrix, var_names, var_scale_factor

def create_row_to_some_id_col_mapping(id_array):
    """
    Parameters
    ----------
    id_array : 1D ndarray.
        All elements of the array should be ints representing some id related
        to the corresponding row.
    Returns
    -------
    rows_to_ids : 2D scipy sparse array.
        Will map each row of id_array to the unique values of `id_array`. The
        columns of the returned sparse array will correspond to the unique
        values of `id_array`, in the order of appearance for each of these
        unique values.
    """
    # Get the unique ids, in their original order of appearance
 #   original_order_unique_ids = get_original_order_unique_ids(id_array)

    assert isinstance(id_array, np.ndarray)
    assert len(id_array.shape) == 1

    # Get the indices of the unique IDs in their order of appearance
    # Note the [1] is because the np.unique() call will return both the sorted
    # unique IDs and the indices
    original_unique_id_indices =\
        np.sort(np.unique(id_array, return_index=True)[1])
    # Get the unique ids, in their original order of appearance
    original_order_unique_ids = id_array[original_unique_id_indices]

    # Create a matrix with the same number of rows as id_array but a single
    # column for each of the unique IDs. This matrix will associate each row
    # as belonging to a particular observation using a one and using a zero to
    # show non-association.
    rows_to_ids = (id_array[:, None] ==
                   original_order_unique_ids[None, :]).astype(float)
    return rows_to_ids

def create_long_form_mappings(long_form,
                              obs_id_col,
                              alt_id_col,
                              choice_col=None,
                              nest_spec=None,
                              mix_id_col=None,
                              dense=False):
    """
    Parameters
    ----------
    long_form : pandas dataframe.
        Contains one row for each available alternative for each observation.
    obs_id_col : str.
        Denotes the column in `long_form` which contains the choice situation
        observation ID values for each row of `long_form`. Note each value in
        this column must be unique (i.e., individuals with repeat observations
        have unique `obs_id_col` values for each choice situation, and
        `obs_id_col` values are unique across individuals).
    alt_id_col : str.
        Denotes the column in long_form which contains the alternative ID
        values for each row of `long_form`.
    choice_col : str, optional.
        Denotes the column in long_form which contains a one if the alternative
        pertaining to the given row was the observed outcome for the
        observation pertaining to the given row and a zero otherwise.
        Default == None.
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    mix_id_col : str, optional.
        Denotes the column in long_form that contains the identification values
        used to denote the units of observation over which parameters are
        randomly distributed.
    dense : bool, optional.
        Determines whether or not scipy sparse matrices will be returned or
        dense numpy arrays.
    Returns
    -------
    mapping_dict : OrderedDict.
        Keys will be `["rows_to_obs", "rows_to_alts", "chosen_row_to_obs",
        "rows_to_nests"]`. If `choice_col` is None, then the value for
        `chosen_row_to_obs` will be None. Likewise, if `nest_spec` is None,
        then the value for `rows_to_nests` will be None. The value for
        "rows_to_obs" will map the rows of the `long_form` to the unique
        observations (on the columns) in their order of appearance. The value
        for `rows_to_alts` will map the rows of the `long_form` to the unique
        alternatives which are possible in the dataset (on the columns), in
        sorted order--not order of appearance. The value for
        `chosen_row_to_obs`, if not None, will map the rows of the `long_form`
        that contain the chosen alternatives to the specific observations those
        rows are associated with (denoted by the columns). The value of
        `rows_to_nests`, if not None, will map the rows of the `long_form` to
        the nest (denoted by the column) that contains the row's alternative.
        If `dense==True`, the returned values will be dense numpy arrays.
        Otherwise, the returned values will be scipy sparse arrays.
    """
    # Get the id_values from the long_form dataframe
    obs_id_values = long_form[obs_id_col].values
    alt_id_values = long_form[alt_id_col].values

    # Create a matrix with the same number of rows as long_form but a single
    # column for each of the unique IDs. This matrix will associate each row
    # as belonging to a particular observation using a one and using a zero to
    # show non-association.
    rows_to_obs = create_row_to_some_id_col_mapping(obs_id_values)

    # Determine all of the unique alternative IDs
    #all_alternatives = np.sort(np.unique(alt_id_values))

    # Create a matrix with the same number of rows as long_form but a single
    # column for each of the unique alternatives. This matrix will associate
    # each row as belonging to a particular alternative using a one and using
    # a zero to show non-association.
    rows_to_alts = create_row_to_some_id_col_mapping(alt_id_values)
                                         #unique_ids=all_alternatives)

    if choice_col is not None:
        # Create a matrix to associate each row with the same number of
        # rows as long_form but a 1 only if that row corresponds to an
        # alternative that a given observation (denoted by the columns)
        # chose.
        chosen_row_to_obs = rows_to_obs*long_form[choice_col].values[:, None]
        chosen_row_to_obs = chosen_row_to_obs.astype(float)
    else:
        chosen_row_to_obs = None

    rows_to_nests = None

    if mix_id_col is not None:
        # Create a mapping matrix between each row and each 'mixing unit'
        mix_id_array = long_form[mix_id_col].values
        rows_to_mixers = create_row_to_some_id_col_mapping(mix_id_array)

    else:
        rows_to_mixers = None

    # Create the dictionary of mapping matrices that is to be returned
    mapping_dict = OrderedDict()
    mapping_dict["rows_to_obs"] = rows_to_obs
    mapping_dict["rows_to_alts"] = rows_to_alts
    mapping_dict["chosen_row_to_obs"] = chosen_row_to_obs
    mapping_dict["rows_to_nests"] = rows_to_nests
    mapping_dict["rows_to_mixers"] = rows_to_mixers

    return mapping_dict

def get_consset_template(maxopts):

    nsets =  2**maxopts
    C = np.zeros([maxopts,nsets])
    
    for i in np.arange(maxopts):        
        chunk = np.repeat(np.array([0,1]), 2**i)
        line = np.tile(chunk, int(nsets/len(chunk)))
        C[i, :] = line
        
    return C.astype(float)
    

"""
def get_all_conssets_flat(start, end, consider=None): 
    #C = []
    if consider is None:
        consider=np.zeros(end[-1])
    consider=consider.astype(int)
    
    nprod = (end-start) 
    maxprods = nprod.max()
    indmaxprods = nprod.argmax()
    ncons = consider[start[indmaxprods]:end[indmaxprods]].sum()
    maxprods = maxprods - ncons
    
    Ctry = np.empty([end[-1], 2**maxprods])
    Ctry.fill(np.nan)
    
    for i in range(len(start)):
        s = start[i]
        e = end[i] 
        cons_ij = consider[s:e]
        ind = np.squeeze(np.argwhere(cons_ij==1)) # only works for single comp products. 
        nprod_ij = e - s
        if ind.size > 0:
           nprod_ij -= 1

        Ci=[]

        for j in range(nprod_ij+1):
            for x in itertools.combinations( range(nprod_ij), j ):
                Ci.append([ 1.0 if i in x else 0.0 for i in range(nprod_ij)])
                        
        Ci=np.array(Ci, order="C" )
        if ind.size > 0:
            Ci = np.insert(Ci, ind, 1.0, axis=1) 

        #C.append(Ci.T)   # how to insert multiple cols? just another ind?    
        Ctry[s:e, :2**nprod_ij]=Ci.T
        
    #return C, Ctry
    return Ctry
"""
def split_param_att(theta,
                    rows_to_alts=None,
                    design=None,
                    return_all_types=False,
                    *args, **kwargs):
    """
    Parameters
    ----------
    theta : 1D ndarray.
            Parameter array containing attentive and inattentive paras. 
    rows_to_alts : None,
        Not actually used. Included merely for consistency with other models.
    design : None.
        Not actually used. Included merely for consistency with other models.
    return_all_types : bool, optional.
        Determines whether or not a tuple of 4 elements will be returned (with
        one element for the nest, shape, intercept, and index parameters for
        this model). If False, a tuple of 3 elements will be returned, as
        described below.
    Returns
    -------
    tuple.
        `(None, gamma, beta)`. This function is merely for compatibility with
        the other choice model files.
    Note
    ----
    If `return_all_types == True` then the function will return a tuple of four
    objects. In order, these objects will either be None or the arrays
    representing the arrays corresponding to the utility and attn parameters
    """
    
    beta = theta[0:design.shape[1]]
    gamma = theta[design.shape[1]:]    
    
    #if return_all_types:
    #    return None, None, None, beta
    #else:
    return  beta, gamma
    
