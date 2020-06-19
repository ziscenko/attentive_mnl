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
from numba import jit, njit#, int64, float64
import numba                       

def estimate2(init_values,
             estimator,
             loss_tol,
             gradient_tol,
             print_results,
             just_point=False,
             simple=False,
             niter = 500,
             popsize = 10,
             stepsize = 0.5,
             seed = 111,
             classes=False,
             ncores = 4,
             **kwargs):
    """
    Estimate the given choice model that is defined by `estimator`.
    Parameters
    ----------
    init_vals : 1D ndarray.
        Should contain the initial values to start the optimization process
        with.
    estimator : an instance of the EstimationObj class.
    method : str, optional.
        Should be a valid string for scipy.optimize.minimize. Determines
        the optimization algorithm that is used for this problem.
        Default `== 'bfgs'`.
    loss_tol : float, optional.
        Determines the tolerance on the difference in objective function
        values from one iteration to the next that is needed to determine
        convergence. Default `== 1e-06`.
    gradient_tol : float, optional.
        Determines the tolerance on the difference in gradient values from
        one iteration to the next which is needed to determine convergence.
        Default `== 1e-06`.
    maxiter : int, optional.
        Determines the maximum number of iterations used by the optimizer.
        Default `== 1000`.
    print_res : bool, optional.
        Determines whether the timing and initial and final log likelihood
        results will be printed as they they are determined.
        Default `== True`.
    use_hessian : bool, optional.
        Determines whether the `calc_neg_hessian` method of the `estimator`
        object will be used as the hessian function during the estimation. This
        kwarg is used since some models (such as the Mixed Logit and Nested
        Logit) use a rather crude (i.e. the BHHH) approximation to the Fisher
        Information Matrix, and users may prefer to not use this approximation
        for the hessian during estimation.
    just_point : bool, optional.
        Determines whether or not calculations that are non-critical for
        obtaining the maximum likelihood point estimate will be performed.
        Default == False.
    Return
    ------
    results : dict.
        The dictionary of estimation results that is returned by
        scipy.optimize.minimize. It will also have (at minimum) the following

    """
    if not just_point:
        # Perform preliminary calculations
        log_likelihood_at_zero =\
            -1*estimator.calc_neg_log_likelihood(estimator.zero_vector)
            #estimator.convenience_calc_log_likelihood(estimator.zero_vector)
              
       # initial_log_likelihood =\
       #     -1*estimator.calc_neg_log_likelihood(init_values)

        if print_results:
            # Print the log-likelihood at zero
            null_msg = "Log-likelihood at zero: {:,.4f}"
            print(null_msg.format(log_likelihood_at_zero))

           # # Print the log-likelihood at the starting values
           # init_msg = "Initial Log-likelihood: {:,.4f}"
           #  print(init_msg.format(initial_log_likelihood))
           #  sys.stdout.flush()

    # Get the hessian fucntion for this estimation process
   # hess_func = estimator.calc_neg_hessian if use_hessian else None

    # Estimate the actual parameters of the model

    start_time = time.time()    
    prob = pg.problem(estimator)
    algo = pg.algorithm(pg.sea(gen = niter))#, ftol = loss_tol))
    arch = pg.archipelago(n=ncores, algo = algo, 
                          prob = prob, 
                          pop_size= 1,#popsize, 
                          seed = seed)
    arch.evolve()
    arch.wait()
    best = np.vstack(arch.get_champions_f()).argmin()
    params = arch.get_champions_x()[best]
    results_dict={}
    results_dict['x']= params
    results_dict['fun'] =  arch.get_champions_f()[best][0]
    results_dict['n_generations'] =  niter
    results_dict['population_size'] = popsize
    results_dict['n_islands'] = ncores
        #xnes gets to the minimum with pop =20 and 300 generations on basic. 
        
    if not just_point:
        if print_results:
            # Stop timing the estimation process and report the timing results
            end_time = time.time()
            elapsed_sec = (end_time - start_time)
            elapsed_min = elapsed_sec / 60.0
            if elapsed_min > 1.0:
                msg = "Estimation Time for Point Estimation: {:.2f} minutes."
                print(msg.format(elapsed_min))
            else:
                msg = "Estimation Time for Point Estimation: {:.2f} seconds."
                print(msg.format(elapsed_sec))
            print("Final log-likelihood: {:,.4f}".format(-1 * results_dict["fun"]))
            sys.stdout.flush()

        # Store the log-likelihood at zero
        results_dict["log_likelihood_null"] = log_likelihood_at_zero

        # Store the final log-likelihood
        final_log_likelihood = -1 * results_dict["fun"]
        results_dict["final_log_likelihood"] = final_log_likelihood
    
        # Get the final array of estimated parameters
        final_params = results_dict["x"]
        
        if not classes:
            # Add the estimated parameters to the results dictionary
            split_res = estimator.convenience_split_params(final_params)
            results_dict["utility_coefs"] = split_res[0]
            results_dict["attn_coefs"] = split_res[1]
        else:
            #result_dict["demog_coefs"] = split_res[2]
            results_dict["all_coefs"] =final_params 
        
        results_dict["simple_res"] = simple
        
        #Calculate derivatives and fitted probs
        
        derivs, probs = estimator.convenience_calc_predict(final_params)
                   
        # Get the probability of the chosen alternative and long_form probabilities
        results_dict["chosen_probs"]=\
            probs['PY_final'][estimator.choice_vector.astype('bool')]
        
        results_dict["altchoice_probs"] = probs['PY_final']

        results_dict["attn_probs"] = probs['PA']     
        results_dict['util_probs'] = probs['PU']
        
        #####
        # Calculate the residuals and individual chi-square values
        #####
        # Calculate the residual vector
        #if len(long_probs.shape) == 1:
        residuals = estimator.choice_vector - probs['PY_final']
        #else:
        #    residuals = estimator.choice_vector[:, None] - long_probs
        results_dict["residuals"] = residuals
    
        # Calculate the observation specific chi-squared components
        #args = [residuals, probs['PY_final'], estimator.rows_to_obs]
        #results_dict["ind_chi_squareds"] = calc_individual_chi_squares(*args)
    
        # Calculate and store the rho-squared and rho-bar-squared
        log_likelihood_null = results_dict["log_likelihood_null"]
        rho_results = calc_rho_and_rho_bar_squared(final_log_likelihood,
                                                   log_likelihood_null,
                                                   final_params.shape[0])
        results_dict["rho_squared"] = rho_results[0]
        results_dict["rho_bar_squared"] = rho_results[1]
    
        #####
        # Save the gradient, hessian, and BHHH approximation to the fisher
        # info matrix
        #####
        if simple:
            _, jac = estimator.convenience_calc_ll_gradient(final_params, 
                                                   True, 
                                                   False)
            results_dict["final_gradient"] = jac
            nvars = final_params.shape[0]
            cov = np.ones((nvars, nvars))
            results_dict["final_hessian"] = cov #-1* np.linalg.inv(cov)
            results_dict["robust_covariance_marix"] = cov
            results_dict["covariance_marix"] = cov
            results_dict["success"]= True
            results_dict["message"] = "N/A"
            
        else:
            start_time = time.time()
            # Get postestimation basics
            _ , g, gl, H = estimator.convenience_calc_postest(final_params)
            end_time = time.time()
            
            elapsed_min = (end_time - start_time)/ 60.0
            msg = "Estimation Time for Hessian: {:.2f} minutes."
            print(msg.format(elapsed_min))          
            sys.stdout.flush()
            
            results_dict["final_gradient"] = g
            results_dict["final_hessian"] = H
            cov = np.linalg.inv(-1*H)
            results_dict["covariance_marix"] = cov
            #results_dict["gradlong"] = gl
            robust_cov = cov@gl.T@gl@cov  # technically need inv(-H) here but works?
            results_dict["robust_covariance_marix"] = robust_cov
        
        #####
        # Store basic marginal effects
        #####

        results_dict['PY_marg_attn'] = derivs['dPY_a']
        results_dict['PY_marg_util'] = derivs['dPY_u']
        results_dict['PA_marg'] = derivs['dPA_long']
        results_dict['PU_marg'] = derivs['dPU_long']
    
        results_dict['probs'] = probs
    return results_dict                                
                           

def estimate(init_values,
             estimator,
             method,
             loss_tol,
             gradient_tol,
             maxiter,
             print_results,
             #use_hessian=True,
             gradient=True,
             just_point=False,
             simple=False,
             random_search = False,
             T = 1.0,
             disp = False,
             niter = 100,
             niter_success = None,
             stepsize = 0.5,
             seed = 111,
             classes=False,
             **kwargs):
    """
    Estimate the given choice model that is defined by `estimator`.
    Parameters
    ----------
    init_vals : 1D ndarray.
        Should contain the initial values to start the optimization process
        with.
    estimator : an instance of the EstimationObj class.
    method : str, optional.
        Should be a valid string for scipy.optimize.minimize. Determines
        the optimization algorithm that is used for this problem.
        Default `== 'bfgs'`.
    loss_tol : float, optional.
        Determines the tolerance on the difference in objective function
        values from one iteration to the next that is needed to determine
        convergence. Default `== 1e-06`.
    gradient_tol : float, optional.
        Determines the tolerance on the difference in gradient values from
        one iteration to the next which is needed to determine convergence.
        Default `== 1e-06`.
    maxiter : int, optional.
        Determines the maximum number of iterations used by the optimizer.
        Default `== 1000`.
    print_res : bool, optional.
        Determines whether the timing and initial and final log likelihood
        results will be printed as they they are determined.
        Default `== True`.
    use_hessian : bool, optional.
        Determines whether the `calc_neg_hessian` method of the `estimator`
        object will be used as the hessian function during the estimation. This
        kwarg is used since some models (such as the Mixed Logit and Nested
        Logit) use a rather crude (i.e. the BHHH) approximation to the Fisher
        Information Matrix, and users may prefer to not use this approximation
        for the hessian during estimation.
    just_point : bool, optional.
        Determines whether or not calculations that are non-critical for
        obtaining the maximum likelihood point estimate will be performed.
        Default == False.
    Return
    ------
    results : dict.
        The dictionary of estimation results that is returned by
        scipy.optimize.minimize. It will also have (at minimum) the following

    """
    if not just_point:
        # Perform preliminary calculations
        log_likelihood_at_zero =\
            -1*estimator.calc_neg_log_likelihood(estimator.zero_vector)
            #estimator.convenience_calc_log_likelihood(estimator.zero_vector)
              
        initial_log_likelihood =\
            -1*estimator.calc_neg_log_likelihood(init_values)

        if print_results:
            # Print the log-likelihood at zero
            null_msg = "Log-likelihood at zero: {:,.4f}"
            print(null_msg.format(log_likelihood_at_zero))

            # Print the log-likelihood at the starting values
            init_msg = "Initial Log-likelihood: {:,.4f}"
            print(init_msg.format(initial_log_likelihood))
            sys.stdout.flush()

    # Get the hessian fucntion for this estimation process
   # hess_func = estimator.calc_neg_hessian if use_hessian else None

    # Estimate the actual parameters of the model
    start_time = time.time()

    #https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function
    #get callback

    #results_dict = minimize(estimator.calc_neg_log_likelihood_and_neg_gradient,
    
    if gradient:
        if random_search:
            results_dict = basinhopping(estimator.calc_neg_log_likelihood_and_neg_gradient,
                           init_values,
                           T = T,
                           disp = disp,
                           niter = niter,
                           niter_success = niter_success,
                           seed = seed,
                           stepsize = stepsize,                       
                           minimizer_kwargs={'method':method,
                                             'jac': True,
                                             'tol':loss_tol, 
                                             'args':({'Nfeval':0},),
                                             'options':{'gtol': gradient_tol,
                                                        "maxiter": maxiter}},                     
                                             **kwargs)
    
        else:
            results_dict = minimize(estimator.calc_neg_log_likelihood_and_neg_gradient,
                           init_values,
                           method=method,
                           jac= True,
                           #hess=hess_func,
                           tol=loss_tol,
                           args=({'Nfeval':0},),
                           options={'gtol': gradient_tol,
                                    "maxiter": maxiter},
                           **kwargs)

    else:
        if random_search:
            results_dict = basinhopping(estimator.calc_neg_log_likelihood,
                           init_values,
                           T = T,
                           disp = disp,
                           niter = niter,
                           niter_success = niter_success,
                           seed = seed,
                           stepsize = stepsize,                       
                           minimizer_kwargs={'method':method,
                                             'jac': False,
                                             'tol':loss_tol, 
                                             'args':({'Nfeval':0},),
                                             'options':{'gtol': gradient_tol,
                                                        "maxiter": maxiter}},                     
                                             **kwargs)
    
        else:
            results_dict = minimize(estimator.calc_neg_log_likelihood,
                           init_values,
                           method=method,
                           jac= False,
                           #hess=hess_func,
                           tol=loss_tol,
                           args=({'Nfeval':0},),
                           options={'gtol': gradient_tol,
                                    "maxiter": maxiter},
                           **kwargs)


    if not just_point:
        if print_results:
            # Stop timing the estimation process and report the timing results
            end_time = time.time()
            elapsed_sec = (end_time - start_time)
            elapsed_min = elapsed_sec / 60.0
            if elapsed_min > 1.0:
                msg = "Estimation Time for Point Estimation: {:.2f} minutes."
                print(msg.format(elapsed_min))
            else:
                msg = "Estimation Time for Point Estimation: {:.2f} seconds."
                print(msg.format(elapsed_sec))
            print("Final log-likelihood: {:,.4f}".format(-1 * results_dict["fun"]))
            sys.stdout.flush()

        # Store the log-likelihood at zero
        results_dict["log_likelihood_null"] = log_likelihood_at_zero

        # Store the final log-likelihood
        final_log_likelihood = -1 * results_dict["fun"]
        results_dict["final_log_likelihood"] = final_log_likelihood
    
        # Get the final array of estimated parameters
        final_params = results_dict["x"]
        
        if not classes:
            # Add the estimated parameters to the results dictionary
            split_res = estimator.convenience_split_params(final_params)
            results_dict["utility_coefs"] = split_res[0]
            results_dict["attn_coefs"] = split_res[1]
        else:
            #result_dict["demog_coefs"] = split_res[2]
            results_dict["all_coefs"] =final_params 
        
        results_dict["simple_res"] = simple
        
        #Calculate derivatives and fitted probs
        
        derivs, probs = estimator.convenience_calc_predict(final_params)
                   
        # Get the probability of the chosen alternative and long_form probabilities
        results_dict["chosen_probs"]=\
            probs['PY_final'][estimator.choice_vector.astype('bool')]
        
        results_dict["altchoice_probs"] = probs['PY_final']

        results_dict["attn_probs"] = probs['PA']     
        results_dict['util_probs'] = probs['PU']
        
        #####
        # Calculate the residuals and individual chi-square values
        #####
        # Calculate the residual vector
        #if len(long_probs.shape) == 1:
        residuals = estimator.choice_vector - probs['PY_final']
        #else:
        #    residuals = estimator.choice_vector[:, None] - long_probs
        results_dict["residuals"] = residuals
    
        # Calculate the observation specific chi-squared components
        #args = [residuals, probs['PY_final'], estimator.rows_to_obs]
        #results_dict["ind_chi_squareds"] = calc_individual_chi_squares(*args)
    
        # Calculate and store the rho-squared and rho-bar-squared
        log_likelihood_null = results_dict["log_likelihood_null"]
        rho_results = calc_rho_and_rho_bar_squared(final_log_likelihood,
                                                   log_likelihood_null,
                                                   final_params.shape[0])
        results_dict["rho_squared"] = rho_results[0]
        results_dict["rho_bar_squared"] = rho_results[1]
    
        #####
        # Save the gradient, hessian, and BHHH approximation to the fisher
        # info matrix
        #####
        if simple:
            if random_search:
                results_dict2 = results_dict["lowest_optimization_result"]
                results_dict["final_gradient"] = results_dict2["jac"]
                cov = results_dict2["hess_inv"]
                results_dict["final_hessian"] = -1* np.linalg.inv(cov)
                results_dict["robust_covariance_marix"] = cov
                results_dict["covariance_marix"] = cov
                results_dict["success"]= True
                results_dict["message"] = "N/A"
            else:
                results_dict["final_gradient"] = results_dict["jac"]
                cov = results_dict["hess_inv"]
                results_dict["final_hessian"] = -1* np.linalg.inv(cov)
                results_dict["robust_covariance_marix"] = cov
                results_dict["covariance_marix"] = cov
                #if
            
        else:
            start_time = time.time()
            # Get postestimation basics
            _ , g, gl, H = estimator.convenience_calc_postest(final_params)
            end_time = time.time()
            
            elapsed_min = (end_time - start_time)/ 60.0
            msg = "Estimation Time for Hessian: {:.2f} minutes."
            print(msg.format(elapsed_min))          
            sys.stdout.flush()
            
            results_dict["final_gradient"] = g
            results_dict["final_hessian"] = H
            cov = np.linalg.inv(-1*H)
            results_dict["covariance_marix"] = cov
            #results_dict["gradlong"] = gl
            robust_cov = cov@gl.T@gl@cov  # technically need inv(-H) here but works?
            results_dict["robust_covariance_marix"] = robust_cov
        
        #####
        # Store basic marginal effects
        #####

        results_dict['PY_marg_attn'] = derivs['dPY_a']
        results_dict['PY_marg_util'] = derivs['dPY_u']
        results_dict['PA_marg'] = derivs['dPA_long']
        results_dict['PU_marg'] = derivs['dPU_long']
    
        results_dict['probs'] = probs
    return results_dict   


### Statistical test calcs (from pylogit)  
def calc_individual_chi_squares(residuals,
                                long_probabilities,
                                rows_to_obs):
    """
    Calculates individual chi-squared values for each choice situation in the
    dataset.
    Parameters
    ----------
    residuals : 1D ndarray.
        The choice vector minus the predicted probability of each alternative
        for each observation.
    long_probabilities : 1D ndarray.
        The probability of each alternative being chosen in each choice
        situation.
    rows_to_obs : 2D scipy sparse array.
        Should map each row of the long format dataferame to the unique
        observations in the dataset.
    Returns
    -------
    ind_chi_squareds : 1D ndarray.
        Will have as many elements as there are columns in `rows_to_obs`. Each
        element will contain the pearson chi-squared value for the given choice
        situation.
    """
    chi_squared_terms = np.square(residuals) / long_probabilities
    return rows_to_obs.T.dot(chi_squared_terms)


def calc_rho_and_rho_bar_squared(final_log_likelihood,
                                 null_log_likelihood,
                                 num_est_parameters):
    """
    Calculates McFadden's rho-squared and rho-bar squared for the given model.
    Parameters
    ----------
    final_log_likelihood : float.
        The final log-likelihood of the model whose rho-squared and rho-bar
        squared are being calculated for.
    null_log_likelihood : float.
        The log-likelihood of the model in question, when all parameters are
        zero or their 'base' values.
    num_est_parameters : int.
        The number of parameters estimated in this model.
    Returns
    -------
    `(rho_squared, rho_bar_squared)` : tuple of floats.
        The rho-squared and rho-bar-squared for the model.
    """
    rho_squared = 1.0 - final_log_likelihood / null_log_likelihood
    rho_bar_squared = 1.0 - ((final_log_likelihood - num_est_parameters) /
                             null_log_likelihood)

    return rho_squared, rho_bar_squared


def compute_aic(model_object):
    """
    Compute the Akaike Information Criteria for an estimated model.
    Parameters
    ----------
    model_object : an MNDC_Model (multinomial discrete choice model) instance.
        The model should have already been estimated.
        `model_object.log_likelihood` should be a number, and
        `model_object.params` should be a pandas Series.
    Returns
    -------
    aic : float.
        The AIC for the estimated model.
    Notes
    -----
    aic = -2 * log_likelihood + 2 * num_estimated_parameters
    References
    ----------
    Akaike, H. (1974). 'A new look at the statistical identification model',
        IEEE Transactions on Automatic Control 19, 6: 716-723.
    """
    assert isinstance(model_object.params, pd.Series)
    assert isinstance(model_object.log_likelihood, Number)

    return -2 * model_object.log_likelihood + 2 * model_object.params.size


def compute_bic(model_object):
    """
    Compute the Bayesian Information Criteria for an estimated model.
    Parameters
    ----------
    model_object : an MNDC_Model (multinomial discrete choice model) instance.
        The model should have already been estimated.
        `model_object.log_likelihood` and `model_object.nobs` should be a
        number, and `model_object.params` should be a pandas Series.
    Returns
    -------
    bic : float.
        The BIC for the estimated model.
    Notes
    -----
    bic = -2 * log_likelihood + log(num_observations) * num_parameters
    The original BIC was introduced as (-1 / 2) times the formula above.
    However, for model comparison purposes, it does not matter if the
    goodness-of-fit measure is multiplied by a constant across all models being
    compared. Moreover, the formula used above allows for a common scale
    between measures such as the AIC, BIC, DIC, etc.
    References
    ----------
    Schwarz, G. (1978), 'Estimating the dimension of a model', The Annals of
        Statistics 6, 2: 461–464.
    """
    assert isinstance(model_object.params, pd.Series)
    assert isinstance(model_object.log_likelihood, Number)
    assert isinstance(model_object.nobs, Number)

    log_likelihood = model_object.log_likelihood
    num_obs = model_object.nobs
    num_params = model_object.params.size

    return -2 * log_likelihood + np.log(num_obs) * num_params



