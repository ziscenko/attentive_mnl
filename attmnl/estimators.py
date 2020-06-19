import warnings
 
import sys
import time
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
import itertools
from scipy.optimize import minimize, basinhopping
from collections import OrderedDict
from collections import Iterable
from numbers import Number
from copy import deepcopy

### Classes and core routines
    
class MNDC_Model(object):
    """
    Parameters
    ----------
    data : str or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        is has one row per available alternative for each observation. If
        pandas dataframe, the dataframe should be the long format data for the
        choice model.
    alt_id_col : str.
        Should denote the column in data which contains the alternative
        identifiers for each row.
    obs_id_col : str.
        Should denote the column in data which contains the observation
        identifiers for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `data`. Values are either a
        list or a single string, "all_diff" or "all_same". If a list, the
        elements should be:
            - single objects that are in the alternative ID column of `data`
            - lists of objects that are within the alternative ID column of
              `data`. For each single object in the list, a unique column will
              be created (i.e. there will be a unique coefficient for that
              variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification` values, a single column will be created for all
              the alternatives within the iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    intercept_ref_pos : int, optional.
         Valid only when the intercepts being estimated are not part of the
         index. Specifies the alternative in the ordered array of unique
         alternative ids whose intercept or alternative-specific constant is
         not estimated, to ensure model identifiability. Default == None.
    shape_ref_pos : int, optional.
        Specifies the alternative in the ordered array of unique alternative
        ids whose shape parameter is not estimated, to ensure model
        identifiability. Default == None.
    names : OrderedDict, optional.
        Should have the same keys as `specification`. For each key:
            - if the corresponding value in `specification` is "all_same", then
              there should be a single string as the value in names.
            - if the corresponding value in `specification` is "all_diff", then
              there should be a list of strings as the value in names. There
              should be one string in the value in names for each possible
              alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names. There
              should be one string the value in names per item in the value in
              `specification`.
        Default == None.
    intercept_names : list, or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternatives in data, minus 1. Each
        element of the list should be a string--the name of the corresponding
        alternative's intercept term, in sorted order of the possible
        alternative IDs. If None is passed, the resulting names that are shown
        in the estimation results will be
        `["Outside_ASC_{}".format(x) for x in shape_names]`. Default = None.
    shape_names : list, or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternative IDs in data. Each element of
        the list should be a string denoting the name of the corresponding
        shape parameter for the given alternative, in sorted order of the
        possible alternative IDs. Default == None.
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    mixing_id_col : str, or None, optional.
        Should be a column heading in `data`. Should denote the column in
        `data` which contains the identifiers of the units of observation over
        which the coefficients of the model are thought to be randomly
        distributed. If `model_type == "Mixed Logit"`, then `mixing_id_col`
        must be passed. Default == None.
    mixing_vars : list, or None, optional.
        All elements of the list should be strings. Each string should be
        present in the values of `names.values()` and they're associated
        variables should only be index variables (i.e. part of the design
        matrix). If `model_type == "Mixed Logit"`, then `mixing_vars` must be
        passed. Default == None.
    model_type : str, optional.
        Denotes the model type of the choice model being instantiated.
        Default == "".
    """
    def __init__(self,
                 dataframe,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 intercept_ref_pos=None,
                 shape_ref_pos=None,
                 names=None,
                 intercept_names=None,
                 shape_names=None,
                 nest_spec=None,
                 mixing_vars=None,
                 mixing_id_col=None,
                 model_type="",
                 normalise=False):
        
        #drop all checks
        if "intercept" in specification and "intercept" not in dataframe.columns:
            dataframe["intercept"] = 1.0
            
        # Create the design matrix for this model
        design_res = create_design_matrix(dataframe,
                                          specification,
                                          alt_id_col,
                                          names=names,
                                          normalise=normalise)
        
        # Store needed data

        self.data = dataframe
        self.normalise = normalise
        self.name_spec = names
        self.design = design_res[0]
        self.ind_var_names = design_res[1]
        self.util_var_scales = design_res[2]
        self.alt_id_col = alt_id_col
        self.obs_id_col = obs_id_col
        self.choice_col = choice_col
        self.specification = specification
        self.alt_IDs = dataframe[alt_id_col].values
        self.choices = dataframe[choice_col].values
       
        self.model_type = model_type
        self.shape_names = shape_names
        self.intercept_names = intercept_names
        self.shape_ref_position = shape_ref_pos
        #self.intercept_ref_position = intercept_ref_pos
        self.nest_names = (list(nest_spec.keys())
                           if nest_spec is not None else None)
        self.nest_spec = nest_spec
        self.mixing_id_col = mixing_id_col
        self.mixing_vars = mixing_vars
        if mixing_vars is not None:
            #mixing_pos = convert_mixing_names_to_positions(mixing_vars,
                                                           #self.ind_var_names)
            [self.ind_var_names.index(name) for name in mixing_vars]
        else:
            mixing_pos = None
        self.mixing_pos = mixing_pos
        
        dataframe.loc[:,'nr']=np.arange(len(dataframe))
        start = np.array(dataframe.groupby(obs_id_col)['nr'].min())
        end = np.array(dataframe.groupby(obs_id_col)['nr'].max())
        end += 1
        
        self.start = start  #need to write a function for this
        self.end = end
        return None
    
    def get_mappings_for_fit(self, dense=False):
        """
        Parameters
        ----------
        dense : bool, optional.
            Dictates if sparse matrices will be returned or dense numpy arrays.
        Returns
        -------
        mapping_dict : OrderedDict.
            Keys will be `["rows_to_obs", "rows_to_alts", "chosen_row_to_obs",
            "rows_to_nests"]`. The value for `rows_to_obs` will map the rows of
            the `long_form` to the unique observations (on the columns) in
            their order of appearance. The value for `rows_to_alts` will map
            the rows of the `long_form` to the unique alternatives which are
            possible in the dataset (on the columns), in sorted order--not
            order of appearance. The value for `chosen_row_to_obs`, if not
            None, will map the rows of the `long_form` that contain the chosen
            alternatives to the specific observations those rows are associated
            with (denoted by the columns). The value of `rows_to_nests`, if not
            None, will map the rows of the `long_form` to the nest (denoted by
            the column) that contains the row's alternative. If `dense==True`,
            the returned values will be dense numpy arrays. Otherwise, the
            returned values will be scipy sparse arrays.
        """
        #return create_long_form_mappings(self.data,
        #                                 self.obs_id_col,
        #                                 self.alt_id_col,
        #                                 choice_col=self.choice_col,
        #                                 nest_spec=self.nest_spec,
        #                                 mix_id_col=self.mixing_id_col,
        #                                 dense=dense)
        
        return None
    # Note that the function below is a placeholder and template for the
    # function to be placed in each model class.   
    
    def fit_mle(self,
                init_vals,
                print_res=True,
                method="BFGS",
                loss_tol=1e-06,
                gradient_tol=1e-03,
                maxiter=1000,
                ridge=None,
                simple=True,
                use_grad=True,
                random_search=False,
                just_point=False,
                T = 1.0,
                 disp = False,
                 niter = 100,
                 niter_success = None,
                 stepsize = 0.5,
                 seed = 111,
                *args):
        """
        Parameters
        ----------
        init_vals : 1D ndarray.
            The initial values to start the optimizatin process with. There
            should be one value for each utility coefficient, outside intercept
            parameter, shape parameter, and nest parameter being estimated.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string which can be passed to
            scipy.optimize.minimize. Determines the optimization algorithm
            which is used for this problem.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next which is needed to determine
            convergence. Default == 1e-06.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
            Default == 1e-06.
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If an int,
            float or long is passed, then that scalar determines the ridge
            penalty for the optimization. Default == None.
        Returns
        -------
        None. Saves estimation results to the model instance.
        """
       # msg = "This model class' fit_mle method has not been constructed."
       # raise NotImplementedError(msg)
 
    
             
        self.optimization_method = method
        self.random_search = random_search

        # Store the ridge parameter
        self.ridge_param = ridge
        
        self.init_vals = init_vals

        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        #mapping_res = self.get_mappings_for_fit()

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
       
        estimator = LogitTypeEstimator(self,  
                                       #mapping_res,
                                       ridge,
                                       zero_vector,
                                       split_param_att)

        simple = True
                                           #constrained_pos="None")
        # Set the derivative functions for estimation
        #mnl_estimator.set_derivatives()

        # Perform one final check on the length of the initial values
        #mnl_estimator.check_length_of_initial_values(init_vals)
        
        # Get the estimation results
       
        estimation_res =estimate(init_vals,
                                     estimator,
                                     method,
                                     loss_tol,
                                     gradient_tol,
                                     maxiter,
                                     print_res,


                                  gradient=use_grad,
                                  just_point=just_point,
                                  simple = simple, 
                                  random_search=random_search,
                                  T = T,
                                  disp = disp,
                                  niter = niter,
                                  niter_success = niter_success,
                                  stepsize = stepsize,
                                  seed = seed)
       
        if not just_point:
            # Store the estimation results
            
            #if simple:
            #    self.store_fit_results_simple(estimation_res)
            #else:
            self.store_fit_results(estimation_res)

            return None
        else:
            return estimation_res   
             
          
    def print_summaries(self, scale=True):
        """
        #Returns None. Will print the measures of fit and the estimation results
        #for the  model.
        """
        
        if hasattr(self, "fit_summary") and (hasattr(self, "summary_scaled")|
                hasattr(self, "summary")):
            print("\n")
            print(self.fit_summary)
            print("=" * 30)
            
            if scale:
                print("Scaled to original input values")
                print(self.summary_scaled.round(3))     #can add round?      
            else: 
                print("Normalised inputs")
                print(self.summary.round(3))

        else:
            msg = "This {} object has not yet been estimated so there "
            msg_2 = "are no estimation summaries to print."
            raise NotImplementedError(msg.format(self.model_type) + msg_2)

        return None
       
    def store_fit_results_simple(self, results_dict):
        """
        Extracts the basic estimation results (i.e. those that need no further
        calculation or logic applied to them) and stores them on the model
        object.
        Parameters
         ----------
        results_dict : dict.
            The estimation result dictionary that is output from
            scipy.optimize.minimize. In addition to the standard keys which are
            included, it should also contain the following keys:
            `["final_log_likelihood", "chosen_probs", "long_probs",
              "residuals", "ind_chi_squareds", "sucess", "message",
              "rho_squared", "rho_bar_squared", "log_likelihood_null"]`
        Returns
        -------
        None.
        """
        
        all_names = deepcopy(self.ind_var_names)
        all_params = [deepcopy(results_dict["utility_coefs"])]
        
        scales =np.concatenate((self.util_var_scales, 
                                self.attn_var_scales))
        
        
        # Store the model results and values needed for model inference
        # Store the utility coefficients
        self._store_inferential_results(results_dict["utility_coefs"],
                                        index_names=self.ind_var_names,
                                        attribute_name="coefs",
                                        series_name="coefficients")
        
        if results_dict["attn_coefs"] is not None:
            storage_args = [results_dict["attn_coefs"],
                            "attn_var_names",
                            "Attn_{}",
                            all_names,
                            all_params,
                            "attn",
                            "attn_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.attn = None
            
        
        # Store the variance-covariance matrix
        self._store_inferential_results(results_dict["covariance_marix"],
                                        index_names=all_names,
                                        attribute_name="cov",
                                        column_names=all_names)

        # Store ALL of the estimated parameters
        self._store_inferential_results(np.concatenate(all_params, axis=0),
                                        index_names=all_names,
                                        attribute_name="params",
                                        series_name="parameters")

        # Store the standard errors
        self._store_inferential_results(np.sqrt(np.diag(self.cov)),
                                        index_names=all_names,
                                        attribute_name="std_errs",
                                        series_name="std_errs")
        
        
        tvals = self.params / self.std_errs
        
        # Store the p-values
        p_vals = 2 * scipy.stats.norm.sf(np.abs(self.tvalues))
        self._store_inferential_results(p_vals,
                                        index_names=all_names,
                                        attribute_name="pvalues",
                                        series_name="p_values")

        self.results=results_dict
        
        ### SUMMARY
        # Store a summary dataframe of the estimation results

        self.summary = pd.concat((self.params,
                                  self.std_errs,
                                  self.pvalues), 
                        axis=1)

        self.summary_scaled = pd.concat((self.params/scales,
                                         self.std_errs*scales,
                                         self.pvalues), 
                                      axis=1)       
            

        ### Store a "Fit Summary"
        # Record values for the fit_summary and statsmodels table
        
        # Record the number of observations
        self.nobs = self.start.shape[0]
        # This is the number of estimated parameters
        self.df_model = self.params.shape[0]
        # The number of observations minus the number of estimated parameters
        self.df_resid = self.nobs - self.df_model

        self.fit_summary = pd.Series([self.df_model,
                                      self.nobs,
                                      self.results["log_likelihood_null"],
                                      self.results["final_log_likelihood"]],
                                     index=["Number of Parameters",
                                            "Number of Observations",
                                            "Null Log-Likelihood",
                                            "Fitted Log-Likelihood"])
        

        return None 
    
    def _store_inferential_results(self,
                                   value_array,
                                   index_names,
                                   attribute_name,
                                   series_name=None,
                                   column_names=None):
        """
        Store the estimation results that relate to statistical inference, such
        as parameter estimates, standard errors, p-values, etc.
        Parameters
        ----------
        value_array : 1D or 2D ndarray.
            Contains the values that are to be stored on the model instance.
        index_names : list of strings.
            Contains the names that are to be displayed on the 'rows' for each
            value being stored. There should be one element for each value of
            `value_array.`
        series_name : string or None, optional.
            The name of the pandas series being created for `value_array.` This
            kwarg should be None when `value_array` is a 1D ndarray.
        attribute_name : string.
            The attribute name that will be exposed on the model instance and
            related to the passed `value_array.`
        column_names : list of strings, or None, optional.
            Same as `index_names` except that it pertains to the columns of a
            2D ndarray. When `value_array` is a 2D ndarray, There should be one
            element for each column of `value_array.` This kwarg should be None
            otherwise.
        Returns
        -------
        None. Stores a pandas series or dataframe on the model instance.
        """
        if len(value_array.shape) == 1:
            assert series_name is not None
            new_attribute_value = pd.Series(value_array,
                                            index=index_names,
                                            name=series_name)
        elif len(value_array.shape) == 2:
            #assert column_names is not None
            if column_names is None:
                column_names = [str(x) for x in range(value_array.shape[1])]
                
                
            new_attribute_value = pd.DataFrame(value_array,
                                               index=index_names,
                                               columns=column_names)

        setattr(self, attribute_name, new_attribute_value)

        return None
    
    def _store_optional_parameters(self,
                                   optional_params,
                                   name_list_attr,
                                   default_name_str,
                                   all_names,
                                   all_params,
                                   param_attr_name,
                                   series_name):
        """
        Extract the optional parameters from the `results_dict`, save them
        to the model object, and update the list of all parameters and all
        parameter names.
        Parameters
        ----------
        optional_params : 1D ndarray.
            The optional parameters whose values and names should be stored.
        name_list_attr : str.
            The attribute name on the model object where the names of the
            optional estimated parameters will be stored (if they exist).
        default_name_str : str.
            The name string that will be used to create generic names for the
            estimated parameters, in the event that the estimated parameters
            do not have names that were specified by the user. Should contain
            empty curly braces for use with python string formatting.
        all_names : list of strings.
            The current list of the names of the estimated parameters. The
            names of these optional parameters will be added to the beginning
            of this list.
        all_params : list of 1D ndarrays.
            Each array is a set of estimated parameters. The current optional
            parameters will be added to the beginning of this list.
        param_attr_name : str.
            The attribute name that will be used to store the optional
            parameter values on the model object.
        series_name : str.
            The string that will be used as the name of the series that
            contains the optional parameters.
        Returns
        -------
        (all_names, all_params) : tuple.
        """
        # Identify the number of optional parameters
        num_elements = optional_params.shape[0]

        # Get the names of the optional parameters
        parameter_names = getattr(self, name_list_attr)
        if parameter_names is None:
            parameter_names = [default_name_str.format(x) for x in
                               range(1, num_elements + 1)]

        # Store the names of the optional parameters in all_names
        all_names =  list(all_names) +list(parameter_names) 
        # Store the values of the optional parameters in all_params
        all_params.append(optional_params)

        # Store the optional parameters on the model object
        self._store_inferential_results(optional_params,
                                        index_names=parameter_names,
                                        attribute_name=param_attr_name,
                                        series_name=series_name)
        return all_names, all_params

    def predict(self, df, beta = None):
        
        if beta is None:
            beta = self.results.x
                    
        obs_id_col = self.obs_id_col
        alt_id_col = self.alt_id_col
        
        df["nr"]=np.arange(len(df))
        start = np.array(df.groupby(obs_id_col)['nr'].min())
        end = np.array(df.groupby(obs_id_col)['nr'].max())
        end += 1
        
        banklist = self.data.lender.unique()
        spec = self.specification
        if "intercept" in spec.keys():
            if not str(spec["intercept"]):
                prod_ids_by_bank = []    
                for name in banklist:          
                    prod_ids_by_bank.append(
                            df[df.lender==name][alt_id_col].unique().tolist())
                spec["intercept"]=prod_ids_by_bank[1:]
        
        design_res = create_design_matrix(df,
                                          spec,
                                          alt_id_col,
                                          names=self.name_spec,
                                          normalise=self.normalise)
        
        design = design_res[0]
        
        Prob_ij = np.zeros(design.shape[0])
        y_pred = np.zeros(design.shape[0])
        
        exb = calc_exb(design, beta)
           
        for i in np.arange(len(start)):
            # Extract needed variables
            s = start[i]
            e = end[i]
            exb_ij = exb[s: e]
            denom_i = exb_ij.sum()
            prob = exb_ij / denom_i 
            ### Utility probs
            Prob_ij[s:e] = prob
            y_pred[s:e] = (prob == prob.max())
            
            
            probs = {'PY_final':Prob_ij, 'y_pred':y_pred}
            
        return probs
        
    def store_fit_results(self, results_dict):
        """
        Parameters
        ----------
        results_dict : dict.
            The estimation result dictionary that is output from
            scipy.optimize.minimize. In addition to the standard keys which are
            included, it should also contain the following keys:
           `["final_gradient", "final_hessian", "fisher_info",
             "final_log_likelihood", "chosen_probs", "long_probs", "residuals",
             "ind_chi_squareds"]`.
            The "final_gradient", "final_hessian", and "fisher_info" values
            should be the gradient, hessian, and Fisher-Information Matrix of
            the log likelihood, evaluated at the final parameter vector.
        Returns
        -------
        None. Will calculate and store a variety of estimation results and
        inferential statistics as attributes of the model instance.
        """
        needed_result_keys = ["final_log_likelihood",
                      "chosen_probs",
                      "long_probs",
                      "residuals",
                      "ind_chi_squareds",
                      "success",
                      "message",
                      "rho_squared",
                      "rho_bar_squared",
                      "log_likelihood_null",
                      "utility_coefs",
                      "intercept_params",
                      "shape_params",
                      "nest_params",
                      "final_gradient",
                      "final_hessian",
                      "fisher_info"]

        
        # Check to make sure the results_dict has all the needed keys.
        #missing_cols = [x for x in needed_result_keys if x not in results_dict]
        #if missing_cols != []:
        #    msg = "The following keys are missing from results_dict\n{}"
        #    raise ValueError(msg.format(missing_cols))
        #return None

        # Store the basic estimation results that simply need to be transferred
        # from the results_dict to the model instance.
        #self._store_basic_estimation_results(results_dict)
        
        self.log_likelihood = results_dict["final_log_likelihood"]
        self.fitted_probs = results_dict["chosen_probs"]
        self.long_fitted_probs = results_dict["long_probs"]
        self.long_residuals = results_dict["residuals"]
        #self.ind_chi_squareds = results_dict["ind_chi_squareds"]
        self.chi_square = results_dict["ind_chi_squareds"].sum()
        
        # Store the 'estimation success' of the optimization
        self.estimation_success = results_dict["success"]
        self.estimation_message = results_dict["message"]

        # Store the summary measures of the model fit
        self.rho_squared = results_dict["rho_squared"]
        self.rho_bar_squared = results_dict["rho_bar_squared"]

        # Store the initial and null log-likelihoods
        self.null_log_likelihood = results_dict["log_likelihood_null"]
        
        # Account for attributes from the mixed logit model.
        if not hasattr(self, "design_3d"):
            self.design_3d = None



        # Initialize the lists of all parameter names and all parameter values
        # Note we add the new mixing variables to the list of index
        # coefficients after estimation so that we can correctly create the
        # design matrix during the estimation proces. The create_design_3d
        # function relies on the original list of independent variable names.
        
        already_included = any(["Sigma " in x for x in self.ind_var_names])

        if self.mixing_vars is not None and not already_included:
            new_ind_var_names = ["Sigma " + x for x in self.mixing_vars]
            self.ind_var_names += new_ind_var_names        
        
        all_names = deepcopy(self.ind_var_names)
        all_params = [deepcopy(results_dict["utility_coefs"])]

        ##########
        # Figure out whether this model had nest, shape, or intercept
        # parameters and store each of these appropriately
        ##########
        if results_dict["intercept_params"] is not None:
            storage_args = [results_dict["intercept_params"],
                            "intercept_names",
                            "Outside_ASC_{}",
                            all_names,
                            all_params,
                            "intercepts",
                            "intercept_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.intercepts = None

        if results_dict["shape_params"] is not None:
            storage_args = [results_dict["shape_params"],
                            "shape_names",
                            "Shape_{}",
                            all_names,
                            all_params,
                            "shapes",
                            "shape_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.shapes = None

        if results_dict["nest_params"] is not None:
            storage_args = [results_dict["nest_params"],
                            "nest_names",
                            "Nest_Param_{}",
                            all_names,
                            all_params,
                            "nests",
                            "nest_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.nests = None



        # Store the model results and values needed for model inference
        # Store the utility coefficients
        self._store_inferential_results(results_dict["utility_coefs"],
                                        index_names=self.ind_var_names,
                                        attribute_name="coefs",
                                        series_name="coefficients")

        # Store the gradient
        self._store_inferential_results(results_dict["final_gradient"],
                                        index_names=all_names,
                                        attribute_name="gradient",
                                        series_name="gradient")

        # Store the hessian
        self._store_inferential_results(results_dict["final_hessian"],
                                        index_names=all_names,
                                        attribute_name="hessian",
                                        column_names=all_names)

        # Store the variance-covariance matrix
        self._store_inferential_results(results_dict["covariance_marix"],
                                        index_names=all_names,
                                        attribute_name="cov",
                                        column_names=all_names)

        # Store ALL of the estimated parameters
        self._store_inferential_results(np.concatenate(all_params, axis=0),
                                        index_names=all_names,
                                        attribute_name="params",
                                        series_name="parameters")

        # Store the standard errors
        self._store_inferential_results(np.sqrt(np.diag(self.cov)),
                                        index_names=all_names,
                                        attribute_name="standard_errors",
                                        series_name="std_err")

        # Store the t-stats of the estimated parameters
        self.tvalues = self.params / self.standard_errors
        self.tvalues.name = "t_stats"

        # Store the p-values
        p_vals = 2 * scipy.stats.norm.sf(np.abs(self.tvalues))
        self._store_inferential_results(p_vals,
                                        index_names=all_names,
                                        attribute_name="pvalues",
                                        series_name="p_values")

        # Store the fischer information matrix of estimated coefficients
        self._store_inferential_results(results_dict["fisher_info"],
                                        index_names=all_names,
                                        attribute_name="fisher_information",
                                        column_names=all_names)
        
        
        #[    DROPPED ROBUST FOR NOW]
        
        
        
        # Adjust the inferential results to account for parameters that were
        # not actually estimated, i.e. parameters that were constrained.
        constraints = results_dict["constrained_pos"]
        #self._adjust_inferential_results_for_parameter_constraints(constraints)

        ### SUMMARY
        # Store a summary dataframe of the estimation results
        
        # Make sure we have all attributes needed to create the results summary
        needed_attributes_s = ["params",
                             "standard_errors",
                             "tvalues",
                             "pvalues"]
#                             "robust_std_errs",
#                             "robust_t_stats",
#                             "robust_p_vals"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([isinstance(getattr(self, attr), pd.Series)
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)

        self.summary = pd.concat((self.params,
                                  self.standard_errors,
                                  self.tvalues,
                                  self.pvalues),
                                  #self.robust_std_errs,
                                  #self.robust_t_stats,
                                  #self.robust_p_vals), 
                        axis=1)


        # Make sure we have all attributes needed to create the results summary
        needed_attributes_s = ["fitted_probs",
                             "params",
                             "log_likelihood",
                             "standard_errors"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)

        # Record the number of observations
        self.nobs = self.fitted_probs.shape[0]
        # This is the number of estimated parameters
        self.df_model = self.params.shape[0]
        # The number of observations minus the number of estimated parameters
        self.df_resid = self.nobs - self.df_model
        # This is just the log-likelihood. The opaque name is used for
        # conformance with statsmodels
        self.llf = self.log_likelihood
        # This is just a repeat of the standard errors
        self.bse = self.standard_errors
        # These are the penalized measures of fit used for model comparison
        self.aic = compute_aic(self)
        self.bic = compute_bic(self)
        
        
        # Store a "Fit Summary"
        needed_attributes_s = ["df_model",
                             "nobs",
                             "null_log_likelihood",
                             "log_likelihood",
                             "rho_squared",
                             "rho_bar_squared",
                             "estimation_message"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)

        self.fit_summary = pd.Series([self.df_model,
                                      self.nobs,
                                      self.null_log_likelihood,
                                      self.log_likelihood,
                                      self.rho_squared,
                                      self.rho_bar_squared,
                                      self.estimation_message],
                                     index=["Number of Parameters",
                                            "Number of Observations",
                                            "Null Log-Likelihood",
                                            "Fitted Log-Likelihood",
                                            "Rho-Squared",
                                            "Rho-Bar-Squared",
                                            "Estimation Message"])


        return 'Hello world!'


class ATT_MNL(MNDC_Model):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        is has one row per available alternative for each observation. If
        pandas dataframe, the dataframe should be the long format data for the
        choice model.
    alt_id_col :str.
        Should denote the column in data which contains the alternative
        identifiers for each row.
    obs_id_col : str.
        Should denote the column in data which contains the observation
        identifiers for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `data`. Values are either a
        list or a single string, "all_diff" or "all_same". If a list, the
        elements should be:
            - single objects that are in the alternative ID column of `data`
            - lists of objects that are within the alternative ID column of
              `data`. For each single object in the list, a unique column will
              be created (i.e. there will be a unique coefficient for that
              variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification` values, a single column will be created for all
              the alternatives within the iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    names : OrderedDict, optional.
        Should have the same keys as `specification`. For each key:
            - if the corresponding value in `specification` is "all_same", then
              there should be a single string as the value in names.
            - if the corresponding value in `specification` is "all_diff", then
              there should be a list of strings as the value in names. There
              should be one string in the value in names for each possible
              alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names. There
              should be one string the value in names per item in the value in
              `specification`.
        Default == None.
    """
    def __init__(self,
                 dataframe,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 Clist=None,
                 names=None,
                 attn_spec=None,
                 attn_group_col=None,
                 def_consider_col=None,
                 att_names=None,
                 mixing_id_col=None,
                 ndraws=100,
                 seed=333,
                 nropts = None,
                 row_to_attgroup_map = None,
                 model_type="Attentive MNL",
                 normalise=True,
                 faster= True,
                 exact = True,
                 default = False,
                 *args, **kwargs):


        if "intercept_ref_pos" in kwargs:
            if kwargs["intercept_ref_pos"] is not None:
                msg = "The MNL model should have all intercepts in the index."
                raise ValueError(msg)
                

            
        # Carry out the common instantiation process for all choice models
        super(ATT_MNL, self).__init__(dataframe,
                                  alt_id_col,
                                  obs_id_col,
                                  choice_col,
                                  specification,
                                  names=names,
                                  mixing_id_col=None,
                                  model_type="Attentive MNL",
                                  normalise=normalise)
        
        
        self.faster = faster
        self.exact = exact
        self.default = default
        # Store att specific paras
        
        self.attn_names = att_names
        self.attn_spec = attn_spec
        
        if attn_group_col is None:
            attn_group_col=alt_id_col 
                
        if row_to_attgroup_map is None:
            dataframe['grp'] =\
                    dataframe.groupby(obs_id_col)[attn_group_col].transform(
                    lambda x:pd.factorize(x)[0])  
            
            row_to_attgroup_map = \
                create_row_to_some_id_col_mapping(
                        np.array(dataframe.grp))

        if def_consider_col is None:
            self.def_consider = np.zeros(self.design.shape[0])
        else:
            self.def_consider = dataframe[def_consider_col].values
        
        if default:
            temp = dataframe.groupby(obs_id_col)[def_consider_col].sum()
            assert temp.min() > 0
            df_att = dataframe.loc[self.def_consider.astype(bool),:]
            self.groupmap = None
            self.firstob_idx = 0
            
        else:
            df_grouped = dataframe.groupby([obs_id_col, attn_group_col])
            self.groupmap = df_grouped.grouper.group_info[0]
            self.firstob_idx = np.array(df_grouped.cumcount() == 0)
            df_att = dataframe.loc[self.firstob_idx, :]
            
        df_att.loc[:,'nr2']=np.arange(len(df_att))
        startd = np.array(df_att.groupby(obs_id_col)['nr2'].min())
        endd = np.array(df_att.groupby(obs_id_col)['nr2'].max())
        endd += 1
        self.start_att =startd
        self.end_att = endd           
        
        nropts = (endd-startd).astype(int)
   
        attn_design = create_design_matrix(df_att,
                                      attn_spec,
                                      alt_id_col,
                                      names=att_names,
                                      normalise=normalise)
        
        self.attn_group_col = attn_group_col
        self.attn_design = attn_design[0]
        self.attn_var_names = attn_design[1]
        self.attn_var_scales = attn_design[2]
        self.def_consider_col = def_consider_col 
        self.rowmap = row_to_attgroup_map

        if exact:
            self.Clist = Clist
            self.nropts = nropts
                
        else:                
            self.ndraws = ndraws
            self.seed = seed                

        return None

    def fit_mle(self,
                init_vals,
                print_res=True,
                method="BFGS",
                loss_tol=1e-06,
                gradient_tol=1e-03,
                maxiter=1000,
                ridge=None,
                #constrained_pos=None,
                use_grad = True,
                just_point=False,
                simple = False,
                random_search = False,
                 T = 1.0,
                 disp = False,
                 niter = 100,
                 niter_success = None,
                 stepsize = 0.5,
                 seed = 111,
                **kwargs):
        """
        Parameters
        ----------
        init_vals : 1D ndarray.
            The initial values to start the optimization process with. There
            should be one value for each utility coefficient being estimated.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string that can be passed to
            scipy.optimize.minimize. Determines the optimization algorithm that
            is used for this problem. If 'em' is passed, a custom coded EM
            algorithm will be used. Default `== 'newton-cg'`.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next that is needed to determine
            convergence. Default `== 1e-06`.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. Default `== None`.
        constrained_pos : list or None, optional.
            Denotes the positions of the array of estimated parameters that are
            not to change from their initial values. If a list is passed, the
            elements are to be integers where no such integer is greater than
            `init_vals.size.` Default == None.
        just_point : bool, optional.
            Determines whether (True) or not (False) calculations that are non-
            critical for obtaining the maximum likelihood point estimate will
            be performed. If True, this function will return the results
            dictionary from scipy.optimize. Default == False.
        Returns
        -------
        None or dict.
            If `just_point` is False, None is returned and the estimation
            results are saved to the model instance. If `just_point` is True,
            then the results dictionary from scipy.optimize() is returned.
        """
        # Check integrity of passed arguments
        #kwargs_to_be_ignored = ["init_shapes", "init_intercepts", "init_coefs"]
       # if any([x in kwargs for x in kwargs_to_be_ignored]):
       #     msg = "MNL model does not use of any of the following kwargs:\n{}"
      #      msg_2 = "Remove such kwargs and pass a single init_vals argument"
       #     raise ValueError(msg.format(kwargs_to_be_ignored) + msg_2)

        #if ridge is not None:
        #    warnings.warn(_ridge_warning_msg)

        # Store the optimization method
        self.optimization_method = method
        self.random_search = random_search

        # Store the ridge parameter
        self.ridge_param = ridge
        
        self.init_vals = init_vals

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        if self.exact:  
            if self.default:
                mnl_estimator = DEF_MNL_Estimator(self,  
                                           #mapping_res,
                                           zero_vector,
                                           split_param_att)
            else:                
                mnl_estimator = ATT_MNL_Estimator(self,  
                                           #mapping_res,
                                           zero_vector,
                                           split_param_att,
                                           self.faster)
        else:
            mnl_estimator = ATT_MNL_EstimatorSim(self,  
                                               #mapping_res,
                                               zero_vector,
                                               split_param_att)  
            simple = True
                                           #constrained_pos="None")
        # Set the derivative functions for estimation
        #mnl_estimator.set_derivatives()

        # Perform one final check on the length of the initial values
        #mnl_estimator.check_length_of_initial_values(init_vals)
        
        # Get the estimation results
       
        estimation_res = estimate(init_vals,
                                  mnl_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res,
                                  gradient=use_grad,
                                  just_point=just_point,
                                  simple = simple, 
                                  random_search=random_search,
                                  T = T,
                                  disp = disp,
                                  niter = niter,
                                  niter_success = niter_success,
                                  stepsize = stepsize,
                                  seed = seed)
       
        if not just_point:
            # Store the estimation results
            
            #if simple:
            #    self.store_fit_results_simple(estimation_res)
            #else:
            self.store_fit_results(estimation_res)

            return None
        else:
            return estimation_res
    
    def store_fit_results(self, results_dict):
        """
        Parameters
        ----------
        results_dict : dict.
            The estimation result dictionary that is output from
            scipy.optimize.minimize. In addition to the standard keys which are
            included, it should also contain the following keys:
           `["final_gradient", "final_hessian", "fisher_info",
             "final_log_likelihood", "chosen_probs", "long_probs", "residuals",
             "ind_chi_squareds"]`.
            The "final_gradient", "final_hessian", and "fisher_info" values
            should be the gradient, hessian, and Fisher-Information Matrix of
            the log likelihood, evaluated at the final parameter vector.
        Returns
        -------
        None. Will calculate and store a variety of estimation results and
        inferential statistics as attributes of the model instance.
        """
        self.results=results_dict
              
        self.log_likelihood = results_dict["final_log_likelihood"]
        self.fitted_probs = results_dict["chosen_probs"]
        self.long_fitted_probs = results_dict["altchoice_probs"]
        self.attention_probs = results_dict["attn_probs"]
        self.utility_probs = results_dict["util_probs"]
        
        
        self.long_residuals = results_dict["residuals"]
        #self.ind_chi_squareds = results_dict["ind_chi_squareds"]
        #self.chi_square = results_dict["ind_chi_squareds"].sum()
        
        # Store the 'estimation success' of the optimization
        self.estimation_success = results_dict["success"]
        self.estimation_message = results_dict["message"]

        # Store the summary measures of the model fit
        self.rho_squared = results_dict["rho_squared"]
        self.rho_bar_squared = results_dict["rho_bar_squared"]
        

        # Store the initial and null log-likelihoods
        self.null_log_likelihood = results_dict["log_likelihood_null"]
        
        # Account for attributes from the mixed logit model.
        #if not hasattr(self, "design_3d"):
        #    self.design_3d = None


        # Scale paras etc back
        scales =np.concatenate((self.util_var_scales, 
                                self.attn_var_scales))
        #att_scale = self.attn_var_scales
        #u_scale = self.util_var_scales
        
        # Save raw derivatives
        self.derivatives = {'PY_marg_attn':
                                results_dict['PY_marg_attn'],
                            'PY_marg_util':
                                results_dict['PY_marg_util'],
                            'PA_marg':
                                results_dict['PA_marg'],
                            'PU_marg':
                                results_dict['PU_marg']}


        marg_choice= np.concatenate((results_dict['PY_marg_util'].mean(axis=0),
                                    results_dict['PY_marg_attn'].mean(axis=0)))
        
        
        marg_attn = results_dict['PA_marg'].mean(axis=0) 
        marg_util = results_dict['PU_marg'].mean(axis=0) 
        
        # Initialize the lists of all parameter names and all parameter values
        # Note we add the new mixing variables to the list of index
        # coefficients after estimation so that we can correctly create the
        # design matrix during the estimation proces. The create_design_3d
        # function relies on the original list of independent variable names.
        
        #already_included = any(["Sigma " in x for x in self.ind_var_names])

        #if self.mixing_vars is not None and not already_included:
        #    new_ind_var_names = ["Sigma " + x for x in self.mixing_vars]
        #    self.ind_var_names += new_ind_var_names        
        
        all_names = deepcopy(self.ind_var_names)
        all_params = [deepcopy(results_dict["utility_coefs"])]

            
        # Store the model results and values needed for model inference
        # Store the utility coefficients
        self._store_inferential_results(results_dict["utility_coefs"],
                                        index_names=self.ind_var_names,
                                        attribute_name="coefs",
                                        series_name="coefficients")
        
        self._store_inferential_results(marg_util,
                                        index_names=self.ind_var_names,
                                        attribute_name="marginals_utility",
                                        series_name="marginals_utility")
               
        if results_dict["attn_coefs"] is not None:
            storage_args = [results_dict["attn_coefs"],
                            "attn_var_names",
                            "Attn_{}",
                            all_names,
                            all_params,
                            "attn",
                            "attn_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.attn = None
            
        self._store_inferential_results(marg_attn,
                                        index_names=self.attn_var_names,
                                        attribute_name="marginals_attn",
                                        series_name="marginals_attn")            
            
        # Optionals should go after utility coeffs
        """
        if results_dict["intercept_params"] is not None:
            storage_args = [results_dict["intercept_params"],
                            "intercept_names",
                            "Outside_ASC_{}",
                            all_names,
                            all_params,
                            "intercepts",
                            "intercept_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.intercepts = None

        if results_dict["shape_params"] is not None:
            storage_args = [results_dict["shape_params"],
                            "shape_names",
                            "Shape_{}",
                            all_names,
                            all_params,
                            "shapes",
                            "shape_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.shapes = None

        """  

        
        # Store the gradient
        #self._store_inferential_results(results_dict["final_gradient"],
        #                                index_names=all_names,
        #                                attribute_name="gradient",
        #                                series_name="gradient")

        # Store the hessian
        self._store_inferential_results(results_dict["final_hessian"],
                                        index_names=all_names,
                                        attribute_name="hessian",
                                        column_names=all_names)

        # Store the variance-covariance matrices
        self._store_inferential_results(results_dict["covariance_marix"],
                                        index_names=all_names,
                                        attribute_name="cov",
                                        column_names=all_names)
        
        self._store_inferential_results(results_dict["robust_covariance_marix"],
                                        index_names=all_names,
                                        attribute_name="robust_cov",
                                        column_names=all_names)

        # Store ALL of the estimated parameters
        self._store_inferential_results(np.concatenate(all_params, axis=0),
                                        index_names=all_names,
                                        attribute_name="params",
                                        series_name="parameters")
        
        # Store ALL of the marginals for final choice

        self._store_inferential_results(marg_choice,
                                        index_names=all_names,
                                        attribute_name="marginals_choice",
                                        series_name="marginals_choice")
        # Store the standard errors
        self._store_inferential_results(np.sqrt(np.diag(self.cov)),
                                        index_names=all_names,
                                        attribute_name="std_errs",
                                        series_name="std_errs")
        
        self._store_inferential_results(np.sqrt(np.diag(self.robust_cov)),
                                        index_names=all_names,
                                        attribute_name="robust_std_errs",
                                        series_name="robust_std_errs")

        # Store the t-stats of the estimated parameters
        self.tvalues = self.params / self.std_errs
        self.tvalues.name = "t_stats"
        
        
        rtvals = self.params / self.robust_std_errs
        
        # Store the p-values
        p_vals = 2 * scipy.stats.norm.sf(np.abs(self.tvalues))
        self._store_inferential_results(p_vals,
                                        index_names=all_names,
                                        attribute_name="pvalues",
                                        series_name="p_values")

        robust_p_vals = 2 * scipy.stats.norm.sf(np.abs(rtvals))
        self._store_inferential_results(robust_p_vals,
                                        index_names=all_names,
                                        attribute_name="robust_p_vals",
                                        series_name="robust_p_vals")
        ### SUMMARY
        # Store a summary dataframe of the estimation results
        """
        # Make sure we have all attributes needed to create the results summary
        needed_attributes_s = ["params",
                             "std_errs",
                             "tvalues",
                             "pvalues"]
#                             "robust_std_errs",
#                             "robust_t_stats",
#                             "robust_p_vals"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([isinstance(getattr(self, attr), pd.Series)
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)
        """
        self.summary = pd.concat((self.params,
                                  #self.std_errs,
                                  #self.tvalues,
                                  self.pvalues,
                                  #self.robust_std_errs,
                                  #self.robust_t_stats,
                                  self.robust_p_vals), 
                        axis=1)

        self.summary_scaled = pd.concat((self.params/scales,
                                          #self.std_errs*scales,
                                          #self.tvalues,
                                      self.pvalues,
                                      #self.robust_std_errs*scales,
                                      #self.robust_t_stats,
                                      self.robust_p_vals), 
                                      axis=1)       
            
            
        ### Record a marginal effect summary
         
        
        choice = pd.Series(marg_choice,
                           index=all_names,
                           name="dYdx")
        
        util = pd.Series(np.concatenate((marg_util,
                                    np.zeros(len(marg_choice)-len(marg_util)))),
                           index=all_names,
                           name="dUdx")            
        attn = pd.Series(np.concatenate((np.zeros(len(marg_choice)-len(marg_attn)),
                                        marg_attn)),
                           index=all_names,
                           name="dAdx")               
                
        
        self.margins_summary = pd.concat((choice,
                                          util,
                                          attn,
                                          self.pvalues),
                                  #self.robust_std_errs,
                                  #self.robust_t_stats,
                                  #self.robust_p_vals), 
                        axis=1)  
        
        
        
        self.margins_summary_scaled = pd.concat((choice/scales,
                                                 util/scales,
                                                 attn/scales,
                                              self.pvalues),
                                  #self.robust_std_errs,
                                  #self.robust_t_stats,
                                  #self.robust_p_vals), 
                        axis=1)    
        
        
        
        ### Store a "Fit Summary"
        # Record values for the fit_summary and statsmodels table
        
        # Make sure we have all attributes needed to create the results summary
        """
        needed_attributes_s = ["fitted_probs",
                             "params",
                             "log_likelihood",
                             "std_errs"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)
        """
        
        # Record the number of observations
        self.nobs = self.fitted_probs.shape[0]
        # This is the number of estimated parameters
        self.df_model = self.params.shape[0]
        # The number of observations minus the number of estimated parameters
        self.df_resid = self.nobs - self.df_model
        # This is just the log-likelihood. The opaque name is used for
        # conformance with statsmodels
        self.llf = self.log_likelihood
        # This is just a repeat of the standard errors
        self.bse = self.std_errs
        # These are the penalized measures of fit used for model comparison
        self.aic = compute_aic(self)
        self.bic = compute_bic(self)


        """
        needed_attributes_s = ["df_model",
                             "nobs",
                             "null_log_likelihood",
                             "log_likelihood",
                             "rho_squared",
                             "rho_bar_squared",
                             "estimation_message"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)
        """
        
        self.fit_summary = pd.Series([self.df_model,
                                      self.nobs,
                                      self.null_log_likelihood,
                                      self.log_likelihood,
                                      self.rho_squared,
                                      self.rho_bar_squared,
                                      self.aic,
                                      self.estimation_message],
                                     index=["Number of Parameters",
                                            "Number of Observations",
                                            "Null Log-Likelihood",
                                            "Fitted Log-Likelihood",
                                            "Rho-Squared",
                                            "Rho-Bar-Squared",
                                            "AIC",
                                            "Estimation Message"])
        

    def print_marginals(self, scale=True):
        
        if scale:
            if hasattr(self, "margins_summary_scaled"):
                print("\n")
                print("Scaled to original input values")
                print(self.margins_summary_scaled.round(5)*np.array([100, 100, 100, 1]))
        else:
            if hasattr(self, "margins_summary"):            
                print("\n")
                print("Normalised inputs")
                print(self.margins_summary.round(5)*np.array([100, 100, 100, 1]))

        return None
    
    
    def predict(self, df, params=None):
        

        
        if self.default:
            print('Default model version not completed yet')
            return None

        else:

            obs_id_col = self.obs_id_col
            choice_col = self.choice_col
            alt_id_col = self.alt_id_col
            attn_group_col = self.attn_group_col

          
            ## Start and end points
            
            df["nr"]=np.arange(len(df))
            start = np.array(df.groupby(obs_id_col)['nr'].min())
            end = np.array(df.groupby(obs_id_col)['nr'].max())
            end += 1   
    
            df_grouped = df.groupby([obs_id_col, attn_group_col])
            groupmap = df_grouped.grouper.group_info[0]
            firstob_idx = np.array(df_grouped.cumcount() == 0)
            df_att = df.loc[firstob_idx, :]
                
            df_att.loc[:,'nr2']=np.arange(len(df_att))
            starta = np.array(df_att.groupby(obs_id_col)['nr2'].min())
            enda = np.array(df_att.groupby(obs_id_col)['nr2'].max())
            enda += 1
            
            ## Other mappings
            nropts = (enda-starta).astype(int) 
                       
            C_temp = get_consset_template(nropts.max())
            df['grp'] =\
                    df.groupby(obs_id_col)[attn_group_col].transform(
                    lambda x:pd.factorize(x)[0])              
            rowmap = create_row_to_some_id_col_mapping(np.array(df.grp))
             

            ## Design frames
            
            altlist = self.data[attn_group_col].unique()

            attn_design_res = create_design_matrix(df_att,
                                              self.attn_spec,
                                              alt_id_col,
                                              names=None,
                                              normalise=self.normalise)
            attn_design = attn_design_res[0]
            
            spec = self.specification
            if "intercept" in spec.keys():
                if not str(spec["intercept"]):
                    opt_ids_by_alt = []    
                    for name in altlist:          
                        opt_ids_by_alt.append(
                                df[df.lender==name][alt_id_col].unique().tolist())
                    spec["intercept"]=opt_ids_by_alt[1:]
            
            design_res = create_design_matrix(df,
                                              spec,
                                              alt_id_col,
                                              names=self.name_spec,
                                              normalise=self.normalise)
            
            design = design_res[0]
            
            ## Consideration & choice
            
            #def_consider_col = self.def_consider_col
            def_consider_col = None
            if def_consider_col is None:
                    def_consider = np.zeros(attn_design.shape[0])
            else:
                    def_consider = (df[def_consider_col].values)[firstob_idx]

            choice_vector = df[choice_col].values
            
            ## Paras
            if params is None:
                params = self.results.x
                
            betas = params[0:(design.shape[1])]
            gammas = params[(design.shape[1]):] 
   
            ## Run function
            
            args = [betas,
                    gammas,
                    design,
                    attn_design,
                    choice_vector,               
                    def_consider,
                    start,
                    end,
                    starta, 
                    enda,  
                    groupmap]
            kwargs = {'Clist': None,
                      'rowmap': rowmap,
                      'nropt' :  nropts,
                      'C_temp' : C_temp,
                      "faster": True,
                      "exact":True}
         
                                
            return calc_att_predict(*args, **kwargs)[1]
        
        
class ATT_MNL_LC(ATT_MNL):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        is has one row per available alternative for each observation. If
        pandas dataframe, the dataframe should be the long format data for the
        choice model.
    alt_id_col :str.
        Should denote the column in data which contains the alternative
        identifiers for each row.
    obs_id_col : str.
        Should denote the column in data which contains the observation
        identifiers for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `data`. Values are either a
        list or a single string, "all_diff" or "all_same". If a list, the
        elements should be:
            - single objects that are in the alternative ID column of `data`
            - lists of objects that are within the alternative ID column of
              `data`. For each single object in the list, a unique column will
              be created (i.e. there will be a unique coefficient for that
              variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification` values, a single column will be created for all
              the alternatives within the iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    names : OrderedDict, optional.
        Should have the same keys as `specification`. For each key:
            - if the corresponding value in `specification` is "all_same", then
              there should be a single string as the value in names.
            - if the corresponding value in `specification` is "all_diff", then
              there should be a list of strings as the value in names. There
              should be one string in the value in names for each possible
              alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names. There
              should be one string the value in names per item in the value in
              `specification`.
        Default == None.
    """
    def __init__(self,
                 dataframe,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 names,
                 attn_spec,
                 att_names,
                 nclasses,
                 demog_spec,
                 demog_names,
                 attn_group_col=None,
                 def_consider_col=None,
                 nropts = None,
                 row_to_attgroup_map = None,
                 model_type="Attentive MNL",
                 normalise=True,
                 default = False,
                 *args, **kwargs):



        # Carry out the common instantiation process for all choice models
        super(ATT_MNL_LC, self).__init__(dataframe,
                                  alt_id_col,
                                  obs_id_col,
                                  choice_col,
                                  specification,
                                  Clist=None,
                                  names=names,
                                  attn_spec=attn_spec,
                                  attn_group_col=attn_group_col,
                                  def_consider_col=def_consider_col,
                                  att_names=att_names,
                                  mixing_id_col=None,
                                  ndraws=None,
                                  seed=333,
                                  nropts=nropts, 
                                  row_to_attgroup_map=row_to_attgroup_map,                                  
                                  model_type="Attentive MNL",
                                  normalise=normalise, 
                                  faster=True, exact=True, default=default)

        # Store att specific paras

        self.demog_spec = demog_spec
        demog_design = create_design_matrix(dataframe,
                                          demog_spec,
                                          alt_id_col,
                                          names=demog_names,
                                          normalise=normalise)
        self.demog_design = demog_design[0]
        self.demog_var_names = demog_design[1]
        self.demog_var_scales = demog_design[2]
        self.nclasses = nclasses


        return None

    def fit_mle(self,
                init_vals,
                print_res=True,
                method="BFGS",
                loss_tol=1e-06,
                gradient_tol=1e-03,
                maxiter=1000,
                ridge=None,
                #constrained_pos=None,
                use_grad = True,
                just_point=False,
                simple = False,
                random_search = False,
                 T = 1.0,
                 disp = False,
                 niter = 100,
                 niter_success = None,
                 stepsize = 0.5,
                 seed = 111,
                 pygmo = False,
                 ncores = 4,
                 popsize = 10,
                **kwargs):
        """
        Parameters
        ----------
        init_vals : 1D ndarray.
            The initial values to start the optimization process with. There
            should be one value for each utility coefficient being estimated.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string that can be passed to
            scipy.optimize.minimize. Determines the optimization algorithm that
            is used for this problem. If 'em' is passed, a custom coded EM
            algorithm will be used. Default `== 'newton-cg'`.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next that is needed to determine
            convergence. Default `== 1e-06`.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. Default `== None`.
        constrained_pos : list or None, optional.
            Denotes the positions of the array of estimated parameters that are
            not to change from their initial values. If a list is passed, the
            elements are to be integers where no such integer is greater than
            `init_vals.size.` Default == None.
        just_point : bool, optional.
            Determines whether (True) or not (False) calculations that are non-
            critical for obtaining the maximum likelihood point estimate will
            be performed. If True, this function will return the results
            dictionary from scipy.optimize. Default == False.
        Returns
        -------
        None or dict.
            If `just_point` is False, None is returned and the estimation
            results are saved to the model instance. If `just_point` is True,
            then the results dictionary from scipy.optimize() is returned.
        """

        self.optimization_method = method
        self.random_search = random_search

        # Store the ridge parameter
        self.ridge_param = ridge
        
        self.init_vals = init_vals

        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        #mapping_res = self.get_mappings_for_fit()

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
                    
        if self.default:
            mnl_estimator = DEF_MNL_LC_Estimator(self,  
                                       #mapping_res,
                                       zero_vector,
                                       split_param_att)
        else:                
            mnl_estimator = ATT_MNL_LC_Estimator(self,  
                                       #mapping_res,
                                       zero_vector,
                                       split_param_att,
                                       self.faster)

        simple = True
                                           
        
        # Get the estimation results
        if pygmo:
            estimation_res = estimate2(init_vals,
                                     mnl_estimator,
                                     loss_tol,
                                     gradient_tol,
                                     print_res,
                                      just_point=just_point,
                                      simple = simple, 
                                     niter = niter,
                                     popsize =popsize,
                                     stepsize = stepsize,
                                     seed = seed,
                                     classes=True,
                                     ncores = ncores)
        else:    
            estimation_res = estimate(init_vals,
                                  mnl_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res,
                                  gradient=use_grad,
                                  just_point=just_point,
                                  simple = simple, 
                                  random_search=random_search,
                                  T = T,
                                  disp = disp,
                                  niter = niter,
                                  niter_success = niter_success,
                                  stepsize = stepsize,
                                  seed = seed, 
                                  classes=True)
       
        if not just_point:
            # Store the estimation results
            
            #if simple:
            #    self.store_fit_results_simple(estimation_res)
            #else:
            self.store_fit_results(estimation_res)

            return None
        else:
            return estimation_res
        
    def print_marginals(self, scale=True):
    
        if scale:
            if hasattr(self, "margins_summary_scaled"):
                print("\n")
                print("Scaled to original input values")
                print(self.margins_summary_scaled.round(5)*100)
        else:
            if hasattr(self, "margins_summary"):            
                print("\n")
                print("Normalised inputs")
                print(self.margins_summary.round(5)*100)

        return None
    
    def store_fit_results(self, results_dict):
        """
        Parameters
        ----------
        results_dict : dict.
            The estimation result dictionary that is output from
            scipy.optimize.minimize. In addition to the standard keys which are
            included, it should also contain the following keys:
           `["final_gradient", "final_hessian", "fisher_info",
             "final_log_likelihood", "chosen_probs", "long_probs", "residuals",
             "ind_chi_squareds"]`.
            The "final_gradient", "final_hessian", and "fisher_info" values
            should be the gradient, hessian, and Fisher-Information Matrix of
            the log likelihood, evaluated at the final parameter vector.
        Returns
        -------
        None. Will calculate and store a variety of estimation results and
        inferential statistics as attributes of the model instance.
        """
        self.results=results_dict
              
        self.log_likelihood = results_dict["final_log_likelihood"]
        self.fitted_probs = results_dict["chosen_probs"]
        self.long_fitted_probs = results_dict["altchoice_probs"]
        self.attention_probs = results_dict["attn_probs"]
        self.utility_probs = results_dict["util_probs"]
        
        
        self.long_residuals = results_dict["residuals"]
        #self.ind_chi_squareds = results_dict["ind_chi_squareds"]
        #self.chi_square = results_dict["ind_chi_squareds"].sum()
        
        # Store the 'estimation success' of the optimization
        self.estimation_success = results_dict["success"]
        self.estimation_message = results_dict["message"]

        # Store the summary measures of the model fit
        self.rho_squared = results_dict["rho_squared"]
        self.rho_bar_squared = results_dict["rho_bar_squared"]
        

        # Store the initial and null log-likelihoods
        self.null_log_likelihood = results_dict["log_likelihood_null"]
        
        # Account for attributes from the mixed logit model.
        #if not hasattr(self, "design_3d"):
        #    self.design_3d = None


        # Scale paras etc back
        scales_temp =np.concatenate((self.util_var_scales, 
                                  self.attn_var_scales))
        scales = np.concatenate((np.tile(scales_temp, self.nclasses), 
                                self.demog_var_scales))
        
        
        #att_scale = self.attn_var_scales
        #u_scale = self.util_var_scales
        
        # Save raw derivatives
        self.derivatives = {'PY_marg_attn':
                                results_dict['PY_marg_attn'],
                            'PY_marg_util':
                                results_dict['PY_marg_util'],
                            'PA_marg':
                                results_dict['PA_marg'],
                            'PU_marg':
                                results_dict['PU_marg']}


        marg_choice= np.concatenate((results_dict['PY_marg_util'],
                                    results_dict['PY_marg_attn']))
        
       
        marg_attn = results_dict['PA_marg']
        marg_util = results_dict['PU_marg'] 
        
        # Initialize the lists of all parameter names and all parameter values
        # Note we add the new mixing variables to the list of index
        # coefficients after estimation so that we can correctly create the
        # design matrix during the estimation proces. The create_design_3d
        # function relies on the original list of independent variable names.
        
        #already_included = any(["Sigma " in x for x in self.ind_var_names])

        #if self.mixing_vars is not None and not already_included:
        #    new_ind_var_names = ["Sigma " + x for x in self.mixing_vars]
        #    self.ind_var_names += new_ind_var_names        
        
        #all_names = deepcopy(self.ind_var_names)
        #all_params = [deepcopy(results_dict["utility_coefs"])]

            
        # Store the model results and values needed for model inference
        # Store the utility coefficients
        
        all_names=[]
        for c in range(self.nclasses):
            names_b =["C"+str(c)+"_" + x for x in self.ind_var_names]
            names_g =["C"+str(c)+"_" + x for x in self.attn_var_names]
            
            all_names += names_b + names_g
            
        all_names +=  self.demog_var_names
            

        #self._store_inferential_results(results_dict["utility_coefs"],
        #                                index_names=self.ind_var_names,
        #                                attribute_name="coefs",
        #                                series_name="coefficients")
        

        
        self._store_inferential_results(marg_util,
                                        index_names=self.ind_var_names,
                                        attribute_name="marginals_utility",
                                        series_name="marginals_utility")
               

        
        self._store_inferential_results(marg_attn,
                                        index_names=self.attn_var_names,
                                        attribute_name="marginals_attn",
                                        series_name="marginals_attn")            
            
        # Store the hessian
        self._store_inferential_results(results_dict["final_hessian"],
                                        index_names=all_names,
                                        attribute_name="hessian",
                                        column_names=all_names)

        # Store the variance-covariance matrices
        self._store_inferential_results(results_dict["covariance_marix"],
                                        index_names=all_names,
                                        attribute_name="cov",
                                        column_names=all_names)
        
        self._store_inferential_results(results_dict["robust_covariance_marix"],
                                        index_names=all_names,
                                        attribute_name="robust_cov",
                                        column_names=all_names)

        # Store ALL of the estimated parameters
        self._store_inferential_results(results_dict['all_coefs'],
                                        index_names=all_names,
                                        attribute_name="params",
                                        series_name="parameters")
        
        # Store ALL of the marginals for final choice

        self._store_inferential_results(marg_choice,
                                        index_names=self.ind_var_names+self.attn_var_names,
                                        attribute_name="marginals_choice",
                                        series_name="marginals_choice")
        # Store the standard errors
        self._store_inferential_results(np.sqrt(np.diag(self.cov)),
                                        index_names=all_names,
                                        attribute_name="std_errs",
                                        series_name="std_errs")
        
        self._store_inferential_results(np.sqrt(np.diag(self.robust_cov)),
                                        index_names=all_names,
                                        attribute_name="robust_std_errs",
                                        series_name="robust_std_errs")

        # Store the t-stats of the estimated parameters
        self.tvalues = self.params / self.std_errs
        self.tvalues.name = "t_stats"
        
        
        rtvals = self.params / self.robust_std_errs
        
        # Store the p-values
        p_vals = 2 * scipy.stats.norm.sf(np.abs(self.tvalues))
        self._store_inferential_results(p_vals,
                                        index_names=all_names,
                                        attribute_name="pvalues",
                                        series_name="p_values")

        robust_p_vals = 2 * scipy.stats.norm.sf(np.abs(rtvals))
        self._store_inferential_results(robust_p_vals,
                                        index_names=all_names,
                                        attribute_name="robust_p_vals",
                                        series_name="robust_p_vals")
        ### SUMMARY
        # Store a summary dataframe of the estimation results
        """
        # Make sure we have all attributes needed to create the results summary
        needed_attributes_s = ["params",
                             "std_errs",
                             "tvalues",
                             "pvalues"]
#                             "robust_std_errs",
#                             "robust_t_stats",
#                             "robust_p_vals"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([isinstance(getattr(self, attr), pd.Series)
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)
        """
        self.summary = pd.concat((self.params,
                                  #self.std_errs,
                                  #self.tvalues,
                                  self.pvalues,
                                  #self.robust_std_errs,
                                  #self.robust_t_stats,
                                  self.robust_p_vals), 
                        axis=1)

        self.summary_scaled = pd.concat((self.params/scales,
                                          #self.std_errs*scales,
                                          #self.tvalues,
                                      self.pvalues,
                                      #self.robust_std_errs*scales,
                                      #self.robust_t_stats,
                                      self.robust_p_vals), 
                                      axis=1)       
            
            
        ### Record a marginal effect summary
         
    
        #choice = pd.Series(marg_choice,
        choice = pd.DataFrame(marg_choice,
                 #index=all_names,
                 index = self.ind_var_names+ self.attn_var_names,
                 columns=["dYdx_"+str(x) for x in 
                          range(marg_choice.shape[1])])
        
        util = pd.DataFrame(marg_util,
                                    #np.zeros(len(marg_choice)-len(marg_util)))),
                           index=self.ind_var_names,
                           columns=["dUdx_"+str(x) for x in 
                                    range(marg_choice.shape[1])])   
            
        attn = pd.DataFrame(marg_attn,
                           index=self.attn_var_names,
                           columns=["dAdx_"+str(x) for x in 
                                    range(marg_choice.shape[1])])               
                
        
        self.margins_summary = pd.concat((choice,
                                          util,
                                          attn),
                                         #self.pvalues),
                                  #self.robust_std_errs,
                                  #self.robust_t_stats,
                                  #self.robust_p_vals), 
                        axis=1)#, sort=False)  
        
        choice = pd.DataFrame(np.divide( marg_choice, scales_temp.reshape(-1,1)),
                           #index=all_names,
                           index = self.ind_var_names+ self.attn_var_names,
                           columns=["dYdx_"+str(x) for x in 
                                    range(marg_choice.shape[1])])
        
        util = pd.DataFrame(np.divide( marg_util,
                                      self.util_var_scales.reshape(-1,1)),
                                    #np.zeros(len(marg_choice)-len(marg_util)))),
                           index=self.ind_var_names,
                           columns=["dUdx_"+str(x) for x in 
                                    range(marg_choice.shape[1])])   
            
        attn = pd.DataFrame(np.divide( marg_attn, 
                                      self.attn_var_scales.reshape(-1,1)),
                           index=self.attn_var_names,
                           columns=["dAdx_"+str(x) for x in 
                                    range(marg_choice.shape[1])])           
        
        self.margins_summary_scaled = pd.concat((choice,
                                                 util,
                                                 attn),
                                              #self.pvalues),
                                  #self.robust_std_errs,
                                  #self.robust_t_stats,
                                  #self.robust_p_vals), 
                        axis=1)#, sort=False)      
        
        
        
        ### Store a "Fit Summary"
        # Record values for the fit_summary and statsmodels table
        
        # Make sure we have all attributes needed to create the results summary
        """
        needed_attributes_s = ["fitted_probs",
                             "params",
                             "log_likelihood",
                             "std_errs"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)
        """
        
        # Record the number of observations
        self.nobs = self.fitted_probs.shape[0]
        # This is the number of estimated parameters
        self.df_model = self.params.shape[0]
        # The number of observations minus the number of estimated parameters
        self.df_resid = self.nobs - self.df_model
        # This is just the log-likelihood. The opaque name is used for
        # conformance with statsmodels
        self.llf = self.log_likelihood
        # This is just a repeat of the standard errors
        self.bse = self.std_errs
        # These are the penalized measures of fit used for model comparison
        self.aic = compute_aic(self)
        self.bic = compute_bic(self)


        """
        needed_attributes_s = ["df_model",
                             "nobs",
                             "null_log_likelihood",
                             "log_likelihood",
                             "rho_squared",
                             "rho_bar_squared",
                             "estimation_message"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes_s])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes_s])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)
        """
        
        self.fit_summary = pd.Series([self.df_model,
                                      self.nobs,
                                      self.null_log_likelihood,
                                      self.log_likelihood,
                                      self.rho_squared,
                                      self.rho_bar_squared,
                                      self.aic,
                                      self.estimation_message],
                                     index=["Number of Parameters",
                                            "Number of Observations",
                                            "Null Log-Likelihood",
                                            "Fitted Log-Likelihood",
                                            "Rho-Squared",
                                            "Rho-Bar-Squared",
                                            "AIC",
                                            "Estimation Message"])
  

   
        


    
class LogitTypeEstimator(object):#(EstimationObj):
    """
    Generic class for storing pointers to data and methods needed in the
    estimation process.
    Parameters
    ----------
    model_obj : a pylogit.base_multinomial_cm_v2.MNDC_Model instance.
        Should contain the following attributes:
          - alt_IDs
          - choices
          - design
          - intercept_ref_position
          - shape_ref_position
          - utility_transform
    mapping_res : dict.
        Should contain the scipy sparse matrices that map the rows of the long
        format dataframe to various other objects such as the available
        alternatives, the unique observations, etc. The keys that it must have
        are `['rows_to_obs', 'rows_to_alts', 'chosen_row_to_obs']`
    ridge : int, float, long, or None.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. The scalar should be greater than or equal to
            zero..
    zero_vector : 1D ndarray.
        Determines what is viewed as a "null" set of parameters. It is
        explicitly passed because some parameters (e.g. parameters that must be
        greater than zero) have their null values at values other than zero.
    split_params : callable.
        Should take a vector of parameters, `mapping_res['rows_to_alts']`, and
        model_obj.design as arguments. Should return a tuple containing
        separate arrays for the model's shape, outside intercept, and index
        coefficients. For each of these arrays, if this model does not contain
        the particular type of parameter, the callable should place a `None` in
        its place in the tuple.
    constrained_pos : list or None, optional.
        Denotes the positions of the array of estimated parameters that are
        not to change from their initial values. If a list is passed, the
        elements are to be integers where no such integer is greater than
        `num_params` Default == None.
    weights : 1D ndarray or None, optional.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.
    Attributes
    ----------
    Methods
    -------
    """
    def __init__(self,
                 model_obj,
                 #mapping_dict,
                 ridge,
                 zero_vector,
                 split_params,
                 constrained_pos=None,
                 weights=None):

        #kwargs = {"constrained_pos": constrained_pos,
                  #"weights": weights}

        # Store pointers to needed objects
        self.alt_id_vector = model_obj.alt_IDs
        self.choice_vector = model_obj.choices.astype(float)
        self.obs_id_vector = model_obj.data[model_obj.obs_id_col].values
        self.design = model_obj.design.astype(float)
        
        #self.intercept_ref_pos = model_obj.intercept_ref_position
        self.shape_ref_pos = model_obj.shape_ref_position

        # Explicitly store pointers to the mapping matrices
        #self.rows_to_obs = mapping_dict["rows_to_obs"]
        #self.rows_to_alts = mapping_dict["rows_to_alts"]
        #self.chosen_row_to_obs = mapping_dict["chosen_row_to_obs"]
        #self.rows_to_nests = mapping_dict["rows_to_nests"]
        #self.rows_to_mixers = mapping_dict["rows_to_mixers"]

        # Perform necessary checking of ridge parameter here!
        #ensure_ridge_is_scalar_or_none(ridge)
        # Ensure the dataset has contiguity in rows with the same obs_id
       # ensure_contiguity_in_observation_rows(self.obs_id_vector)
        # Ensure the weights are appropriate for model estimation
        #ensure_positivity_and_length_of_weights(weights, model_obj.data)

        # Store the ridge parameter
        self.ridge = ridge

        # Store the constrained parameters
        self.constrained_pos = constrained_pos

        # Store reference to what 'zero vector' is for this model / dataset
        self.zero_vector = zero_vector

        # Store the weights that were passed to the constructor
        if weights is None:
            self.weights = np.ones(self.design.shape[0], dtype=float)
        else:
            weights = weights.astype(float)

        # Store the function that separates the various portions of the
        # parameters being estimated (shape parameters, outside intercepts,
        # utility coefficients)
        self.split_params = split_params

        # Store the function that calculates the transformation of the index
       # self.utility_transform = model_obj.utility_transform

        # Get the block matrix indices for the hessian matrix.
        #self.block_matrix_idxs = create_matrix_block_indices(self.rows_to_obs)

        # Note the following attributes should be set to actual callables that
        # calculate the necessary derivatives in the classes that inherit from
        # EstimationObj
        self.calc_dh_dv = lambda *args: None
        self.calc_dh_d_alpha = lambda *args: None
        self.calc_dh_d_shape = lambda *args: None
        
        
        self.start = model_obj.start
        self.end = model_obj.end
        
        return None

    
    def convenience_split_params(self, params, return_all_types=False):
        """
        Splits parameter vector into shape, intercept, and index parameters.
        Parameters
        ----------
        params : 1D ndarray.
            The array of parameters being estimated or used in calculations.
        return_all_types : bool, optional.
            Determines whether or not a tuple of 4 elements will be returned
            (with one element for the nest, shape, intercept, and index
            parameters for this model). If False, a tuple of 3 elements will
            be returned with one element for the shape, intercept, and index
            parameters.
        Returns
        -------
        tuple. Will have 4 or 3 elements based on `return_all_types`.
        """
        if return_all_types:
            return None, None, None, params
        else:
            return None, None, params
    

    def convenience_calc_probs(self, params, long_probs=True):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        prob_args = [betas,
                     self.design,
                     self.rows_to_obs]#,
                     #self.utility_transform]

        #prob_kwargs = {"return_long_probs": long_probs,
        #               "alt_IDs": self.alt_id_vector,
        #               "rows_to_alts": self.rows_to_alts,
        #               "intercept_params": intercepts,
        #               "shape_params": shapes,
        #               "chosen_row_to_obs": self.chosen_row_to_obs
        #               }
        
        if long_probs:
            prob_results = calc_long_prob(*prob_args)
        else: 
            prob_args.append(self.chosen_row_to_obs)
            prob_results = calc_choice_prob(*prob_args)
        return prob_results
    
    
    def convenience_calc_ll_gradient(self, params, 
                                          use_gradient, use_hessian):
    
        #shapes, intercepts, betas = self.convenience_split_params(params)   
        
        args = [params,
                self.design,
                self.choice_vector,
                self.start,
                self.end]#,
        kwargs = {"use_gradient": use_gradient}
                  
        return calc_ll_gradient_f(*args, **kwargs)


    def convenience_calc_log_likelihood(self, params):
        """
        Calculates the log-likelihood for this model and dataset.
        """      
        log_likelihood, _  =\
            self.convenience_calc_ll_gradient(params, False, False)

        return log_likelihood


    def calc_neg_log_likelihood_and_neg_gradient(self, params, info):
        """
        Calculates and returns the negative of the log-likelihood and the
        negative of the gradient. This function is used as the objective
        function in scipy.optimize.minimize.
        """
        log_likelihood, gradient =\
            self.convenience_calc_ll_gradient(params, True, False)

        neg_log_likelihood = -1 * log_likelihood
        neg_gradient = -1 * gradient

        if self.constrained_pos is not None:
            neg_gradient[self.constrained_pos] = 0
            
                    
        if info['Nfeval']%10 == 0:    
            print(info['Nfeval'], ":", neg_log_likelihood)        
        if info['Nfeval']%50 == 0:
            print(params)
        info['Nfeval'] += 1

        return neg_log_likelihood, neg_gradient

    def calc_neg_hessian(self, params):
        """
        Calculate and return the negative of the hessian for this model and
        dataset.
        """  
        _, _, hess = self.convenience_calc_ll_gradient_hess(params, False, True)
        
        return -1 * hess    
    
class ATT_MNL_Estimator(LogitTypeEstimator):

    def __init__(self,
                 model_obj,
                 #mapping_dict,
                 zero_vector,
                 split_params,
                 faster=True,
                 constrained_pos=None,
                 weights=None):
        ridge=None
        super(ATT_MNL_Estimator, self).__init__(model_obj,
                                             #mapping_dict,
                                             ridge,
                                             zero_vector,
                                             split_params,
                                             constrained_pos=None,
                                             weights=None)
        
        self.attn_design = model_obj.attn_design
        firstob_idx = model_obj.firstob_idx
        self.def_consider = model_obj.def_consider
        self.def_consider_short = model_obj.def_consider[firstob_idx]
        
        self.start = model_obj.start
        self.end = model_obj.end
        self.start_att = model_obj.start_att
        self.end_att = model_obj.end_att
        self.faster=faster

        self.rowmap = model_obj.rowmap
        self.groupmap = model_obj.groupmap
        nropts = model_obj.nropts
        self.nropts = nropts
        
        

       
        #self.in_C = model_obj.Clist

        #if faster:
        if not model_obj.default:
            self.C_temp = get_consset_template(nropts.max())
        #else:
        #    self.C_temp = None
        #    if model_obj.Clist is None:
        #        self.in_C = get_all_conssets(self.start, self.end, 
        #                        self.def_consider)
   
        return None

        
    def convenience_split_params(self, params, return_all_types=False):
          
        beta = params[0:(self.design.shape[1])]
        gamma = params[(self.design.shape[1]):]    
    
    #if return_all_types:
    #    return None, None, None, beta
    #else:
        return  beta, gamma
    
    def convenience_calc_aprobs(self, params, long_probs=True):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        beta, gamma = self.convenience_split_params(params)

        aprobs_ij = calc_aprob_att(gamma, self.attn_design)
        
        return aprobs_ij

    
    def convenience_calc_ll_gradient(self, params, 
                                          use_gradient, use_hessian):
    
        betas, gammas = self.convenience_split_params(params)   
        
        

        kwargs = {"use_gradient": use_gradient,
                  "use_hessian": use_hessian,
                  "weights": self.weights}
        
        #if self.faster:
        args = [betas,
            gammas,
            self.design,
            self.attn_design,
            self.choice_vector, 
            self.C_temp,
            self.nropts,
            self.rowmap,
            self.def_consider_short,   
            #self.firstob_idx,             
            self.start,
            self.end,
            self.start_att,
            self.end_att]#,
                
        res1, res2 = calc_att_ll_gradient_f(*args, 
                                            use_gradient)
        """                        
        else:
            args = [betas,
                gammas,
                self.design,
                self.attn_design,
                self.choice_vector, 
                self.in_C,
                self.def_consider,
                self.start,
                self.end]#,
                    
            res1, res2 = calc_att_ll_gradient(*args, **kwargs) 
        """    
        if use_gradient:
            return res1, res2
        
        else:
            return res1, None
    
        
    def convenience_calc_postest(self, params):
    
        betas, gammas = self.convenience_split_params(params)   
        
        args = [betas,
                gammas,
                self.design,
                self.attn_design,
                self.choice_vector, 
                self.def_consider,
                self.start,
                self.end,
                self.start_att,
                self.end_att]
        kwargs = {"use_gradient": True,
                  "use_hessian": True,
                  "weights": self.weights,
                  'Clist': self.in_C,
                  'rowmap': self.rowmap,
                  'nropt' :  self.nropts,
                  'C_temp' : self.C_temp,
                  "faster":self.faster}
     
        #log_likelihood, gradient , hessian =\
        #    calc_ll_gradient_hessian(*args, **kwargs)
            
        return calc_att_ll_gradient_hess(*args, **kwargs)
    
    def convenience_calc_predict(self, params):
    
        betas, gammas = self.convenience_split_params(params)   
        
        # can transform dataframe into data here if it is passed. For now just basoc
        
        args = [betas,
                gammas,
                self.design,
                self.attn_design,
                self.choice_vector, 
                
                self.def_consider_short,
                self.start,
                self.end,
                self.start_att,
                self.end_att,
                self.groupmap]
        kwargs = {#'Clist': self.in_C,
                  'rowmap': self.rowmap,
                  'nropt' :  self.nropts,
                  'C_temp' : self.C_temp,
                "faster":self.faster,
                  "exact":True}
                
               
                            
        return calc_att_predict(*args, **kwargs)
    

               

    
    def calc_neg_log_likelihood(self, params, info=None):
        """
        Calculates and returns the negative of the log-likelihood and the
        negative of the gradient. This function is used as the objective
        function in scipy.optimize.minimize.
        """
        log_likelihood, _ =\
            self.convenience_calc_ll_gradient(params, False, False)

        neg_log_likelihood = -1 * log_likelihood
        
        if info is not None:
            
            if info['Nfeval']%10 == 0:    
                print(info['Nfeval'], ":", neg_log_likelihood)        
            if info['Nfeval']%50 == 0:
                print(params)
            info['Nfeval'] += 1                

        return neg_log_likelihood
    
    def calc_neg_log_likelihood_and_neg_gradient(self, params, info):
        """
        Calculates and returns the negative of the log-likelihood and the
        negative of the gradient. This function is used as the objective
        function in scipy.optimize.minimize.
        """
        log_likelihood, gradient =\
            self.convenience_calc_ll_gradient(params, True, False)
        
        #### RIDGE HACK
        
        neg_log_likelihood = -1 * log_likelihood #+ 5*np.square(params).sum()
        neg_gradient = -1 * gradient #- 5*2*params

        if self.constrained_pos is not None:
            neg_gradient[self.constrained_pos] = 0
            
            
        if info['Nfeval']%10 == 0:    
            print(info['Nfeval'], ":", neg_log_likelihood)        
        if info['Nfeval']%50 == 0:
            print(params)
        info['Nfeval'] += 1
        
        return neg_log_likelihood, neg_gradient   
    
    
    def fitness(self, params):
        
        log_likelihood, _ =\
            self.convenience_calc_ll_gradient(params, False, False)

        neg_log_likelihood =  -1 * log_likelihood
        
        return [neg_log_likelihood]
    
    def get_name(self):
        return "Attentive MNL"
    
    def get_bounds(self):
        nvar = self.zero_vector.shape[0]
        return ([-5] * nvar, [5] * nvar)
        
#class ATT_MNL_EstimatorSim(LogitTypeEstimator):
    """
    Estimation Object used to enforce uniformity in the estimation process
    across the various logit-type models.
    Parameters
    ----------
    model_obj : a pylogit.base_multinomial_cm_v2.MNDC_Model instance.
        Should contain the following attributes:
          - alt_IDs
          - choices
          - design
          - intercept_ref_position
          - shape_ref_position
          - utility_transform
    mapping_res : dict.
        Should contain the scipy sparse matrices that map the rows of the long
        format dataframe to various other objects such as the available
        alternatives, the unique observations, etc. The keys that it must have
        are `['rows_to_obs', 'rows_to_alts', 'chosen_row_to_obs']`
    ridge : int, float, long, or None.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. The scalar should be greater than or equal to
            zero..
    zero_vector : 1D ndarray.
        Determines what is viewed as a "null" set of parameters. It is
        explicitly passed because some parameters (e.g. parameters that must be
        greater than zero) have their null values at values other than zero.
    split_params : callable.
        Should take a vector of parameters, `mapping_res['rows_to_alts']`, and
        model_obj.design as arguments. Should return a tuple containing
        separate arrays for the model's shape, outside intercept, and index
        coefficients. For each of these arrays, if this model does not contain
        the particular type of parameter, the callable should place a `None` in
        its place in the tuple.
    constrained_pos : list or None, optional.
        Denotes the positions of the array of estimated parameters that are
        not to change from their initial values. If a list is passed, the
        elements are to be integers where no such integer is greater than
        `num_params` Default == None.
    weights : 1D ndarray or None, optional.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.
    """
    """      
        #return None
    def __init__(self,
                 model_obj,
                 #mapping_dict,
                 zero_vector,
                 split_params,
                 constrained_pos=None,
                 weights=None):
        ridge=None
        super(ATT_MNL_EstimatorSim, self).__init__(model_obj,
                                             #mapping_dict,
                                             ridge,
                                             zero_vector,
                                             split_params,
                                             constrained_pos=None,
                                             weights=None)
        
        self.attn_design = model_obj.attn_design
        self.def_consider = model_obj.def_consider
        self.ndraws = model_obj.ndraws
        self.start = model_obj.start
        self.end = model_obj.end
        rtb = model_obj.rowmap
        self.rowmap = rtb
        np.random.seed(model_obj.seed)  
        
        if rtb is None:
            self.sim_draws= np.random.rand(self.design.shape[0]*model_obj.ndraws)
        else:                        
            ni = self.start.shape[0]
            nb = rtb[1]
            sim_bank_draws = np.random.rand(nb * ni , self.ndraws)
            self.sim_draws = (rtb @ sim_bank_draws).T.ravel()
        
        C0_probs, in_C0 = self.convenience_calc_C_probs(model_obj.init_vals)
        C0_probs[C0_probs<=0]= 1e-30
        self.C0_probs =  C0_probs
        self.in_C0 = in_C0

        
    def convenience_split_params(self, params, return_all_types=False):
          
        beta = params[0:(self.design.shape[1])]
        gamma = params[(self.design.shape[1]):]    
    
    #if return_all_types:
    #    return None, None, None, beta
    #else:
        return  beta, gamma
    
    def convenience_calc_aprobs(self, params, long_probs=True):

        beta, gamma = self.convenience_split_params(params)

        aprobs_ij = calc_aprob_att(gamma, self.attn_design)
        
        return aprobs_ij
    
    def convenience_calc_C_probs(self, params):  
        

        beta, gamma = self.convenience_split_params(params)   
        ndraws = self.ndraws
        nobs = self.design.shape[0]
        startv = self.start
        endv = self.end
        sim_draws = self.sim_draws
        ni = len(startv)
        
        #Get Consideration probs      
        aprobs_ij = calc_aprob_att(gamma, self.attn_design)
        aprobs_ij[self.def_consider==1] = 0.9999999

        log_aprob_c = np.log(aprobs_ij)
        log_aprob_nc = np.log(1-aprobs_ij)
    
        ### Consideration set probs 
        inCvec = np.zeros(nobs*ndraws)
        for loop in np.arange(ndraws):
            st = loop * nobs
            en = (loop+1) * nobs
            inCvec[st:en] = (aprobs_ij>=sim_draws[st:en]) #*1.0
            #print(st, en, len(inCvec[st:en])) 
        inCmat = (inCvec.reshape([ndraws, nobs])).T
        C_probs = np.zeros([ni, ndraws])
        
        for i in range(ni):
        # Extract needed variables
            s = startv[i]
            e = endv[i]
            log_aprob_ci = getrowsvec(log_aprob_c, s, e)
            log_aprob_nci = getrowsvec(log_aprob_nc, s, e)               
            in_C_ijr =getrows(inCmat, s, e)
        
            ### Consideration set probs 
                 
            C_prob_temp = log_aprob_ci.T @ in_C_ijr + log_aprob_nci.T @ (1-in_C_ijr) 
            C_probs[i, :] =  np.exp(C_prob_temp)
            
        return C_probs, inCmat
        
    
    def convenience_calc_ll_gradient(self, params, 
                                          use_gradient, use_hessian):
    
        betas, gammas = self.convenience_split_params(params)   
        
        
        args = [betas,
                gammas,
                self.design,
                self.attn_design,
                self.choice_vector,                 
                self.sim_draws,
                self.ndraws,
                self.in_C0,
                self.C0_probs,
                self.def_consider,
                self.start,
                self.end]#,
        kwargs = {"use_gradient": use_gradient,
                  #"use_hessian": use_hessian,
                  "weights": self.weights}
        
        res1, res2 = calc_att_ll_gradient_sim(*args, **kwargs)   
        if use_gradient:
            return res1, res2
        
        else:
            return res1, None
        
    def convenience_calc_predict(self, params):
    
        betas, gammas = self.convenience_split_params(params)   
        
        # can transform dataframe into data here if it is passed. For now just basoc
        
        args = [betas,
                gammas,
                self.design,
                self.attn_design,
                self.choice_vector, 
                None,
                self.def_consider,
                self.start,
                self.end]
        kwargs = {"faster":False,
                  "exact": False}
            
        return calc_att_predict(*args, **kwargs)
                           
                             
    def calc_neg_log_likelihood(self, params):

        log_likelihood, _ =\
            self.convenience_calc_ll_gradient(params, False, False)

        neg_log_likelihood = -1 * log_likelihood
        #neg_gradient = -1 * gradient

        return neg_log_likelihood
    
    def calc_neg_log_likelihood_and_neg_gradient(self, params):

        log_likelihood, gradient =\
            self.convenience_calc_ll_gradient(params, True, False)

        neg_log_likelihood = -1 * log_likelihood
        neg_gradient = -1 * gradient

        if self.constrained_pos is not None:
            neg_gradient[self.constrained_pos] = 0
         
        if info['Nfeval']%10 == 0:    
            print(info['Nfeval'], ":", neg_log_likelihood)        
        if info['Nfeval']%50 == 0:
            print(params)
        info['Nfeval'] += 1
        return neg_log_likelihood, neg_gradient     
    
"""

  

class ATT_MNL_LC_Estimator(ATT_MNL_Estimator):

    def __init__(self,
                 model_obj,
                 #mapping_dict,
                 zero_vector,
                 split_params,
                 constrained_pos=None,
                 weights=None):
        ridge=None
        super(ATT_MNL_LC_Estimator, self).__init__(model_obj,
                                             #mapping_dict,
                                             
                                             zero_vector,
                                             split_params,
                                             faster=True,
                                             constrained_pos=None,
                                             weights=None)
        
        self.demog_design = model_obj.demog_design
        self.nclasses = model_obj.nclasses
        
        
        return None

    def convenience_calc_ll_gradient(self, params, 
                                          use_gradient, use_hessian):
           
        
        
        args = [params,
                self.design,
                self.attn_design,
                self.choice_vector, 
                self.demog_design,
                self.nclasses,
                self.C_temp,
                self.nropts,
                self.rowmap,
                
                self.def_consider_short,                
                self.start,
                self.end,
                self.start_att,
                self.end_att]#,
                    
        res1, res2 = calc_att_ll_gradient_lc(*args, 
                                                use_gradient)
        if use_gradient:
            return res1, res2
        
        else:
            return res1, None

        
    def convenience_calc_predict(self, params):
    
        nbetas = self.design.shape[1]
        ngammas = self.attn_design.shape[1]  #? len gammas?
        nvars = nbetas + ngammas
        ndeltas = self.demog_design.shape[1]
        deltas = params[-ndeltas : (len(params)+1)]
        nrows = self.end.max()
        nclasses = self.nclasses
        PA =PU = np.zeros((nrows, nclasses))
        derivs = {'dPU_long': np.zeros((nbetas, nclasses)),
                'dPA_long': np.zeros((ngammas, nclasses)),                 
              'dPY_u': np.zeros((nbetas, nclasses)), 
              'dPY_a': np.zeros((ngammas, nclasses))} 
                
        sc = calc_aprob_att(deltas, self.demog_design)
        Prob_ij=np.zeros(nrows)  
        
        for c in range(nclasses):
            param = params[c*nvars:(c+1)*nvars]
            beta = param[0:nbetas]
            gamma = param[nbetas:nvars]        
        
            args = [beta,
                    gamma,
                    self.design,
                    self.attn_design,
                    self.choice_vector,                 
                    self.def_consider_short,
                    self.start,
                    self.end,
                self.start_att,
                self.end_att,
                self.groupmap]
            kwargs = {#'Clist': self.in_C,
                     'rowmap': self.rowmap,
                     'nropt' :  self.nropts,
                     'C_temp' : self.C_temp,
                     "faster": True,
                     "exact":True}
           
            derivs_c, probs_c = calc_att_predict(*args, **kwargs)
            
            PA[:, c] = probs_c['PA']
            PU[:, c] = probs_c['PU']
            
            for x in derivs.keys():
          
                derivs[x][:,c] = derivs_c[x].mean(0)
                
                
            if c == 0:            
                share_ic = sc
            else:
                share_ic = 1-sc
      
            Prob_ij += share_ic*probs_c["PY_final"]
        
        probs={"PClass0":sc, 'PU': PU, 'PA': PA, 'PY_final': Prob_ij}
        
        return derivs, probs
                           
                             
    def fitness(self, params):
        
        log_likelihood, _ =\
            self.convenience_calc_ll_gradient(params, False, False)

        neg_log_likelihood =  -1 * log_likelihood
        
        return [neg_log_likelihood]
    
    def get_name(self):
        return "Attentive MNL - Latent Class"
    
    def get_bounds(self):
        nvar = self.zero_vector.shape[0]
        return ([-5] * nvar, [5] * nvar)
  

class DEF_MNL_Estimator(ATT_MNL_Estimator):

    def __init__(self,
                 model_obj,
                 #mapping_dict,
                 zero_vector,
                 split_params,
                 constrained_pos=None,
                 weights=None):
        ridge=None
        super(DEF_MNL_Estimator, self).__init__(model_obj,
                                             #mapping_dict,
                                             
                                             zero_vector,
                                             split_params,
                                             faster=True,
                                             constrained_pos=None,
                                             weights=None)
        
        #self.demog_design = model_obj.demog_design
        #self.nclasses = model_obj.nclasses
        
        defaults = model_obj.def_consider
        self.in_C = np.concatenate((defaults.reshape(-1,1),
                                    np.ones((len(defaults),1))), axis=1)
        

                
        return None
        

           
    def convenience_calc_ll_gradient(self, params, 
                                          use_gradient, use_hessian):
        
        betas, gammas = self.convenience_split_params(params)       

        args =[betas,
                gammas,
                self.design,
                self.attn_design,
                self.choice_vector, 
                self.in_C,                
                self.def_consider.astype(bool),                
                self.start,
                self.end,
                self.start_att,
                self.end_att]#,
                    
        res1, res2 = calc_def_ll_gradient(*args, 
                                                use_gradient)
        if use_gradient:
            return res1, res2
        
        else:
            return res1, None

   
    def convenience_calc_predict(self, params):
    
        beta, gamma = self.convenience_split_params(params)   
        
        design = self.design
        att_design = self.attn_design
        #self.choice_vector, 
        in_C = self.in_C                
        def_vec = self.def_consider                
        startv = self.start
        endv = self.end
        startd =  self.start_att
        endd = self.end_att
        dim = design.shape
        nbetas = dim[1]
        ngammas = att_design.shape[1]    
        
        exb = calc_exb(design, beta)
        
        ### Consideration probs  
        Mu_ij = calc_aprob_att(gamma, att_design)

         
        PU_long = np.zeros(dim[0])
        PA_long = np.zeros(endd.shape[0]) 
        PY_final =np.zeros(dim[0])
        y_pred =np.zeros(dim[0])
        
        dPA_long =    np.zeros(att_design.shape)
        dPU_long = np.zeros(dim)  
        dPY_u = np.zeros(dim)
        dPY_a = np.zeros((dim[0], ngammas))   
        dPY_a2 = np.zeros(att_design.shape)    

        mult = np.array([1.0, -1.0]).reshape(-1, 1)
        for i in np.arange(len(startv)):
            # Extract needed variables
            s = startv[i]
            e = endv[i]
            sd = startd[i]
            ed = endd[i]

            exb_ij = getrowsvec(exb, s, e)
            in_C_ijr = getrows(in_C, s, e)
            def_vec_ij = def_vec[s:e].astype(bool)              

            exb_ij = getrowsvec(exb, s, e)
            in_C_ijr = getrows(in_C, s, e)
              

            ### utility probs
                
            denom_ir = (exb_ij.T @ in_C_ijr).ravel()
            denom_ir[np.isnan(denom_ir)]=1e-30
            denom_ir[denom_ir==0]=1e-30
            denom_vec_ir = denom_ir.reshape((1,-1))
            
            ### Utility probs
            uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 
            PU = exb_ij/ exb_ij.sum()
            
            probs_def_ij = subset1d(uprobs_ijr, 0)        
            probs_def_ij_short = probs_def_ij[def_vec_ij]  
        
            
            ### Search probs              
            mu_ij = Mu_ij[sd:ed]
            mu_i = (probs_def_ij_short @ mu_ij)

            mu_ij_long = np.zeros(probs_def_ij.shape)
            mu_ij_long[def_vec_ij] = mu_ij
        
            C_prob_ir = np.array([mu_i, (1-mu_i)]).reshape(1,-1)

            ### Overall probs
            Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()   

            ### Partial derivs (own!)
            dPU_aux =  PU * (1-PU)
            
            dMdz_aux = probs_def_ij_short @ (mu_ij * (1-mu_ij))
            
            def_vec_ij_r = def_vec_ij.reshape(-1,1)
            
            dPY_u_aux1 = (probs_def_ij_short *(1-probs_def_ij_short)) @ mu_ij
            dPY_u_aux2 = dPY_u_aux1 *((uprobs_ijr @ mult)*def_vec_ij_r)
            
            dPY_u_aux3 =  (uprobs_ijr * (1- uprobs_ijr)) @ C_prob_ir.T
            
            dPY_u_aux = dPY_u_aux2 + dPY_u_aux3 # is the effect through mu important here?
            
            dPY_a_aux = ((uprobs_ijr @ mult)*dMdz_aux) * def_vec_ij_r
            dPY_a_aux2 = ((uprobs_ijr @ mult)*dMdz_aux)[def_vec_ij]
            
            PA_long[i] =  (1- mu_i)
            PU_long[s:e] = PU.ravel()            
            PY_final[s:e] = Prob_ij
            y_pred[s:e] = Prob_ij == Prob_ij.max()
            
            dPA_long[sd:ed, :] = -1* np.tile(dMdz_aux, [1, ngammas]) * gamma.reshape(1, -1)    
            dPY_a[s:e, :] = np.tile(dPY_a_aux, [1, ngammas]) * gamma.reshape(1, -1)
            dPY_a2[sd:ed, :] = np.tile(dPY_a_aux2, [1, ngammas]) * gamma.reshape(1, -1)
            dPY_u[s:e, :] = np.tile(dPY_u_aux, [1, nbetas]) * beta.reshape(1, -1)
            dPU_long[s:e, :] = np.tile(dPU_aux, [1, nbetas]) * beta.reshape(1, -1)    
        
        
        probs = {'PA': PA_long }
        probs['PY_final']=PY_final
        probs['y_pred']=y_pred
        probs['PU'] = PU_long
        derivs = {'dPA_long':dPA_long, 'dPU_long': dPU_long,
                  'dPY_u':dPY_u, 'dPY_a_alt': dPY_a, 'dPY_a': dPY_a2} 

                            
        return derivs, probs
        
class DEF_MNL_LC_Estimator(DEF_MNL_Estimator):

    def __init__(self,
                 model_obj,
                 #mapping_dict,
                 zero_vector,
                 split_params,
                 constrained_pos=None,
                 weights=None):
        ridge=None
        super(DEF_MNL_LC_Estimator, self).__init__(model_obj,
                                             #mapping_dict,
                                             
                                             zero_vector,
                                             split_params,
                                             
                                             constrained_pos=None,
                                             weights=None)
        
        self.demog_design = model_obj.demog_design
        self.nclasses = model_obj.nclasses
        
        return None

    def convenience_calc_ll_gradient(self, params, 
                                          use_gradient, use_hessian):
           
        
        
        args = [params,
                self.design,
                self.attn_design,
                self.choice_vector, 
                self.demog_design,
                self.nclasses,
                self.in_C,                
                self.def_consider.astype(bool),                
                self.start,
                self.end,
                self.start_att,
                self.end_att]
                    
        res1, res2 = calc_def_ll_gradient_lc(*args, 
                                                use_gradient)
        if use_gradient:
            return res1, res2
        
        else:
            return res1, None

   
        
    def convenience_calc_predict(self, params):
    
        nbetas = self.design.shape[1]
        ngammas = self.attn_design.shape[1]  #? len gammas?
        nvars = nbetas + ngammas
        ndeltas = self.demog_design.shape[1]
        deltas = params[-ndeltas : (len(params)+1)]
        nrows = self.end.max()
        nclasses = self.nclasses
        PA =PU = np.zeros((nrows, nclasses))
        derivs = {'dPU_long': np.zeros((nbetas, nclasses)),
                'dPA_long': np.zeros((ngammas, nclasses)),                 
              'dPY_u': np.zeros((nbetas, nclasses)), 
              'dPY_a': np.zeros((ngammas, nclasses))} 
                
        sc = calc_aprob_att(deltas, self.demog_design)
        Prob_ij=np.zeros(nrows)  
        """
        for c in range(nclasses):
            param = params[c*nvars:(c+1)*nvars]
            beta = param[0:nbetas]
            gamma = param[nbetas:nvars]        
        
            args = [beta,
                    gamma,
                    self.design,
                    self.attn_design,
                    self.choice_vector,                 
                    self.def_consider,
                    self.start,
                    self.end]
            kwargs = {'Clist': self.in_C,
                     'rowmap': self.rowmap,
                     'nropt' :  self.nropts,
                     'C_temp' : self.C_temp,
                     "faster": True,
                     "exact":True}
           
            derivs_c, probs_c = calc_att_predict(*args, **kwargs)
            
            PA[:, c] = probs_c['PA']
            PU[:, c] = probs_c['PU']
            
            for x in derivs.keys():
          
                derivs[x][:,c] = derivs_c[x].mean(0)
                
                
            if c == 0:            
                share_ic = sc
            else:
                share_ic = 1-sc
      
            Prob_ij += share_ic*probs_c["PY_final"]
            """
        probs={"PClass0":sc, 'PU': PU, 'PA': PA, 'PY_final': Prob_ij}
        
        return derivs, probs
                           