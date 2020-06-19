
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

### Workhorse likelihood functions

@jit(nopython=True, nogil=True)#, parallel=True) 
def calc_ll_gradient_f(beta,
                            
                             design,
                             
                             choice_vector,
                             
                             
                             startv, 
                             endv,
                             use_gradient=True):

   
    # Calculate the probability of each individual choosing each available
    # alternative for that individual.   
    nbetas = design.shape[1]
        #nrow = len(choice_vector)        
    
    exb = calc_exb(design, beta)
    
    ll = np.array([0.0])
    grad = np.zeros(nbetas) 
       
    for i in np.arange(len(startv)):
        # Extract needed variables
        s = startv[i]
        e = endv[i]
        y_ij = choice_vector[s:e]
        x_ij = getrows(design, s, e)

        
        exb_ij = exb[s: e]
                       
        denom_i = exb_ij.sum()

       
        ### Utility probs
        Prob_ij = exb_ij / denom_i 
        
        ll_ij = np.log(Prob_ij)
        ll_ij[np.isinf(ll_ij)] = 0
        ll_ij[np.isnan(ll_ij)] = 0
        
        # Calculate the log likelihood
        ll_i = y_ij @  ll_ij
        ll += ll_i
        
        if use_gradient:  #parallelise this? or don't bother as might not use in sim anneal?
            
            Prob_ij[Prob_ij==0] = 1e-30
            uprobs_ijr = Prob_ij.reshape((-1,1))
            
            dLdP = (y_ij / Prob_ij).reshape((-1,1)) # j x 1
            
            dLdP_aux = dLdP.T @ uprobs_ijr # 1 x 1 = y
            
            dPdb_k_aux = (x_ij.T @ uprobs_ijr) # kb x r

            
            dLdb_k_t = Prob_ij.reshape((1, -1)) @ (x_ij * dLdP) -\
                         dLdP_aux @ (dPdb_k_aux ).T #_need 1_ x kb or kb x 1
                         
            dLdb_k = dLdb_k_t.ravel()

            grad_i = dLdb_k
            grad_i[np.isnan(grad_i)]=0 
            grad = grad + grad_i

     
    return ll[0], grad     

@jit(nopython=True, nogil=True)#, parallel=True) 
def calc_att_ll_gradient_f(beta,
                             gamma,
                             design,
                             att_design,
                             choice_vector,
                             C_temp,
                             nropt,
                             rowmap,
                             consider_vec,
                             startv, 
                             endv,
                             starta,
                             enda,
                             use_gradient=True):

   
    # Calculate the probability of each individual choosing each available
    # alternative for that individual.   
    nbetas = design.shape[1]
    ngammas = att_design.shape[1]
        #nrow = len(choice_vector)      
    #maxopt = nropt.max()
    
    exb = calc_exb(design, beta)
    
    ### Consideration probs      
    Aprobs_ij = calc_aprob_att(gamma, att_design)
    Aprobs_ij[consider_vec==1]=0.9999999

    ll = np.array([0.0])
    grad = np.zeros(nbetas+ngammas) 
    
    #att_gr_count_s = 0
    for i in np.arange(len(startv)):
        # Extract needed variables
        s = startv[i]
        e = endv[i]
        sa = starta[i]
        ea = enda[i]
        nropt_i =nropt[i]               
        y_ij = choice_vector[s:e]
        x_ij = getrows(design, s, e)
        exb_ij = getrowsvec(exb, s, e)
        rowmap_ij = getrows(rowmap, s, e)
                
        xz_ij = getrows(att_design, sa, ea)
        aprobs_ij = getrowsvec(Aprobs_ij, sa, ea)  
        
        ncol_i = 2**nropt_i     
        C_i=getcols(C_temp, 0, ncol_i)
        in_C_ijr = rowmap_ij @ C_i 
        
        C_i = getrows(C_i, 0, nropt_i)        
        log_aprob_c = np.log(aprobs_ij)
        log_aprob_nc = np.log(1-aprobs_ij)
        
                
        ### Consideration set probs 
             
        #C_prob_temp = log_aprob_c.T @ in_C_ijr + log_aprob_nc.T @ (1-in_C_ijr) 

        C_prob_temp = log_aprob_c.T @ C_i + log_aprob_nc.T @ (1-C_i) 
        
        C_prob_ir = np.exp(C_prob_temp)
    
        denom_ir = (exb_ij.T @ in_C_ijr).ravel()
        denom_ir[np.isnan(denom_ir)]=1e-30
        denom_ir[denom_ir==0]=1e-30
        denom_vec_ir = denom_ir.reshape((1,-1))
        
        ### Utility probs
        uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 

    
        ### Overall probs
        Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()        
        
        ll_ij = np.log(Prob_ij)
        ll_ij[np.isinf(ll_ij)] = 0
        ll_ij[np.isnan(ll_ij)] = 0
        
        # Calculate the log likelihood
        ll_i = y_ij @  ll_ij
        ll += ll_i
        
        #att_gr_count_s += nropt_i
        
        if use_gradient:  #parallelise this? or don't bother as might not use in sim anneal?
            
            Prob_ij[Prob_ij==0] = 1e-30
            dLdP = (y_ij / Prob_ij).reshape((-1,1)) # j x 1
            
            dLdP_aux = dLdP.T @ uprobs_ijr # 1 x r(ncons)
            
            dPdb_k_aux = (x_ij.T @ uprobs_ijr) # kb x r
            #dPdg_k_aux = (xz_ij.T @ (in_C_ijr - aprobs_ij))#kg x r
            dPdg_k_aux = (xz_ij.T @ (C_i - aprobs_ij))#kg x r
              
            dLdb_k_t = Prob_ij.reshape((1, -1)) @ (x_ij * dLdP) -\
                         dLdP_aux @ (dPdb_k_aux * C_prob_ir).T #_need 1_ x kb or kb x 1
                         
            dLdb_k = dLdb_k_t.ravel()
            
            dLdg_k = (dLdP_aux @ (C_prob_ir * dPdg_k_aux).T).ravel() # 1 x kb   
            
            grad_i = np.concatenate((dLdb_k, dLdg_k))
            grad_i[np.isnan(grad_i)]=0 
            grad = grad + grad_i

     
    return ll[0], grad     

"""
@jit(nopython=True, nogil=True) 
def calc_att_ll_gradient_f(beta,
                             gamma,
                             design,
                             att_design,
                             choice_vector,
                             C_temp,
                             nropt,
                             rowmap,
                             consider_vec,
                             startv, 
                             endv,
                             starta,
                             enda,
                             use_gradient=True):

   
    # Calculate the probability of each individual choosing each available
    # alternative for that individual.   
    nbetas = design.shape[1]
    ngammas = att_design.shape[1]
        #nrow = len(choice_vector)      
    #maxopt = nropt.max()
    
    exb = calc_exb(design, beta)
    
    ### Consideration probs      
    Aprobs_ij = calc_aprob_att(gamma, att_design)
    Aprobs_ij[consider_vec==1]=0.9999999

    ll = np.array([0.0])
    grad = np.zeros(nbetas+ngammas) 
    
    #att_gr_count_s = 0
    for i in np.arange(len(startv)):
        # Extract needed variables
        s = startv[i]
        e = endv[i]
        sa = starta[i]
        ea = enda[i]
        nropt_i =nropt[i]               
        y_ij = choice_vector[s:e]
        x_ij = getrows(design, s, e)
        exb_ij = getrowsvec(exb, s, e)
        rowmap_ij = getrows(rowmap, s, e)
                
        xz_ij = getrows(att_design, sa, ea)
        aprobs_ij = getrowsvec(Aprobs_ij, sa, ea)  
        
        ncol_i = 2**nropt_i     
        C_i=getcols(C_temp, 0, ncol_i)
        in_C_ijr = rowmap_ij @ C_i 
        
        C_i = getrows(C_i, 0, nropt_i)        
        log_aprob_c = np.log(aprobs_ij)
        log_aprob_nc = np.log(1-aprobs_ij)
        
                
        ### Consideration set probs 
             
        #C_prob_temp = log_aprob_c.T @ in_C_ijr + log_aprob_nc.T @ (1-in_C_ijr) 

        C_prob_temp = log_aprob_c.T @ C_i + log_aprob_nc.T @ (1-C_i) 
        
        C_prob_ir = np.exp(C_prob_temp)
    
        denom_ir = (exb_ij.T @ in_C_ijr).ravel()
        denom_ir[np.isnan(denom_ir)]=1e-30
        denom_ir[denom_ir==0]=1e-30
        denom_vec_ir = denom_ir.reshape((1,-1))
        
        ### Utility probs
        uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 

    
        ### Overall probs
        Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()        
        
        ll_ij = np.log(Prob_ij)
        ll_ij[np.isinf(ll_ij)] = 0
        ll_ij[np.isnan(ll_ij)] = 0
        
        # Calculate the log likelihood
        ll_i = y_ij @  ll_ij
        ll += ll_i
        
        #att_gr_count_s += nropt_i
        
        if use_gradient:  #parallelise this? or don't bother as might not use in sim anneal?
            
            Prob_ij[Prob_ij==0] = 1e-30
            dLdP = (y_ij / Prob_ij).reshape((-1,1)) # j x 1
            
            dLdP_aux = dLdP.T @ uprobs_ijr # 1 x r(ncons)
            
            dPdb_k_aux = (x_ij.T @ uprobs_ijr) # kb x r
            #dPdg_k_aux = (xz_ij.T @ (in_C_ijr - aprobs_ij))#kg x r
            dPdg_k_aux = (xz_ij.T @ (C_i - aprobs_ij))#kg x r
              
            dLdb_k_t = Prob_ij.reshape((1, -1)) @ (x_ij * dLdP) -\
                         dLdP_aux @ (dPdb_k_aux * C_prob_ir).T #_need 1_ x kb or kb x 1
                         
            dLdb_k = dLdb_k_t.ravel()
            
            dLdg_k = (dLdP_aux @ (C_prob_ir * dPdg_k_aux).T).ravel() # 1 x kb   
            
            grad_i = np.concatenate((dLdb_k, dLdg_k))
            grad_i[np.isnan(grad_i)]=0 
            grad = grad + grad_i

     
    return ll[0], grad     
"""
@jit(nopython=True, nogil=True, fastmath=True)#, parallel=True) 
def calc_att_ll_gradient_lc(params,
                             design,
                             att_design,
                             choice_vector,
                             demogs,
                             nclasses,
                             C_temp,
                             nropt,
                             rowmap,
                             consider_vec,
                             startv, 
                             endv,
                             starta, 
                             enda,
                             use_gradient=False):
    
    
    
    nbetas = design.shape[1]
    ngammas = att_design.shape[1]  #? len gammas?
    nvars = nbetas + ngammas
    ndeltas = demogs.shape[1]
    deltas = params[-ndeltas : (len(params)+1)]
    #nrows = design.shape[0]
    ll = np.array([0.0])
    grad = np.zeros(ndeltas+nclasses*(nvars))
       
    for i in range(len(startv)):
        #ll_i = np.array([0.0])
        grad_i = np.zeros(ndeltas+nclasses*(nvars))

        # Extract needed variables
        s = startv[i]
        e = endv[i]
        sa = starta[i]
        ea = enda[i]
        nropt_i = nropt[i] 
        Prob_ij = np.zeros(e-s)
        dPdd_aux = np.zeros(e-s)   
        y_ij = choice_vector[s:e]
        x_ij = getrows(design, s, e)
        xz_ij = getrows(att_design, sa, ea)
        dem_ij = getrows(demogs, s, e)
        rowmap_ij = getrows(rowmap, s, e)        
        ncol_i = 2** nropt_i      
        C_i=getcols(C_temp, 0, ncol_i)
        in_C_ijr = rowmap_ij @ C_i
               
        C_i = getrows(C_i, 0, nropt_i)        

        
        nrows_ij = x_ij.shape[0]
        #dPdb_k =  []
        #dPdg_k = []
        dPdb_k =  np.zeros(nrows_ij*nbetas*nclasses)
        dPdg_k =  np.zeros(nrows_ij*ngammas*nclasses)
        
        #paralelalise this bit?  
        for c in range(nclasses):
            
            
            ### TWO CLASS SHORTHAND 
            sc = calc_aprob_att(deltas, dem_ij)[0]
            if sc==0:
                sc = 0.0000000001
            elif sc==1:
                sc = 0.9999999999
            
            share_ic = (1-c)*sc + c * (1-sc)
            
            if nclasses ==1:
                share_ic = 1
            
            # Standard calcs, within class
            
            param = params[c*nvars:(c+1)*nvars]
            beta = param[0:nbetas]
            gamma = param[nbetas:nvars]
            
            ### Consideration probs  
            aprobs_ijc = calc_aprob_att(gamma, xz_ij).reshape(-1, 1)
            log_aprob_c = np.log(aprobs_ijc)
            log_aprob_nc = np.log(1-aprobs_ijc)
#           aprobs_ij[consider_vec==1]=0.9999999 # assume no default/consideration
            #log_aprob_c, log_aprob_nc = calc_logaprobs(aprobs_ijc, rowmap_ij, 
            #                                       nropt_i, s, e)   
            
            
            exb_ij = calc_exb(x_ij, beta).reshape(-1, 1)
            
            Prob_ijc, dPdb_k_aux, dPdg_k_aux = calc_att_ll_grad_inner_lc(x_ij, 
                           xz_ij, y_ij, aprobs_ijc, exb_ij, 
                           in_C_ijr, C_i, log_aprob_c, log_aprob_nc,
                           use_gradient)


            Prob_ij += share_ic*Prob_ijc
   
            
            if use_gradient:
                dPdd_aux += (1-2*c)*(1-share_ic) * share_ic * Prob_ijc    
                
                
                dPdg_k[(nrows_ij*ngammas)*c:(nrows_ij*ngammas)*(c+1)] =\
                        (share_ic * dPdg_k_aux).ravel()   
                        
                dPdb_k[(nrows_ij*nbetas)*c:(nrows_ij*nbetas)*(c+1)] =\
                            share_ic *(Prob_ijc.reshape((-1, 1)) *\
                            x_ij - dPdb_k_aux).ravel()
       
        ll_ij = np.log(Prob_ij)
        ll_ij[np.isinf(ll_ij)] = 0
        ll_ij[np.isnan(ll_ij)] = 0
        ll_i = y_ij @  ll_ij
        ll += ll_i
     
        if use_gradient:
            dLdP = (y_ij / Prob_ij).reshape((-1,1)) # j x 1
            
            #calculate the derivatives for deltas 
            for i in range(ndeltas):
                dem_col = dem_ij[:,i]
                dPdd_aux2 = dPdd_aux * dem_col
                gradd_ik = dLdP.T@ dPdd_aux2
                grad_i[len(grad_i) - (ndeltas-i)] = gradd_ik[0]           
            
            for c in range(nclasses):
                
                dPdg_kc = dPdg_k[
                            (nrows_ij*ngammas)*c :(nrows_ij*ngammas)*(c+1)
                            ].reshape(nrows_ij,ngammas) 
                dPdb_kc = dPdb_k[
                          (nrows_ij*nbetas)*c :(nrows_ij*nbetas)*(c+1)
                          ].reshape(nrows_ij,nbetas) 
                
                #betas CHECK THIS - SHOULD BE IJCc AND SHOULD HAV ESHARE THERE
                dLdb_k =  (dLdP.T @ dPdb_kc).ravel()   #_need 1_ x kb or kb x 1
                                         
                #gammmas
                dLdg_k = ( dLdP.T @ dPdg_kc).ravel() # 1 x kb   
            
                grad_i[c*nvars:(c+1)*nvars] =  np.concatenate((dLdb_k, dLdg_k))
            
           
            grad = grad + grad_i
  
    return ll[0], grad  


        
@jit(nopython=True, nogil=True, fastmath=True)    
def calc_att_ll_grad_inner_lc(x_ij, xz_ij, y_ij,
                       aprobs_ij, exb_ij, in_C_ijr, C_i, 
                       log_aprob_c, log_aprob_nc,
                       use_gradient):
      
            
    ### Consideration set probs 
 
    C_prob_temp = log_aprob_c.T @ C_i + log_aprob_nc.T @ (1-C_i) 
    C_prob_ir = np.exp(C_prob_temp)

    denom_ir = (exb_ij.T @ in_C_ijr).ravel()
    denom_ir[np.isnan(denom_ir)]=1e-30
    denom_ir[denom_ir==0]=1e-30
    denom_vec_ir = denom_ir.reshape((1,-1))
    
    ### Utility probs
    uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 

    ### Overall probs
    Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()        
    
   
    # Calculate the log likelihood
    #ll_i = y_ij @  ll_ij
    
    Prob_ij[Prob_ij==0] = 1e-30

    ### Gradient
    if use_gradient:# | use_hessian:

        dPdb_k_aux = (x_ij.T @ uprobs_ijr) # kb x r                
        dPdb_k_aux2= (uprobs_ijr @ (dPdb_k_aux * C_prob_ir).T) 

        dPdg_k_aux = (xz_ij.T @ (C_i - aprobs_ij))#kg x r
        dPdg_k_aux2 = (uprobs_ijr @ (C_prob_ir * dPdg_k_aux).T)                

    else:
        dPdb_k_aux2 = np.zeros(x_ij.shape)
        dPdb_k_aux2 = np.zeros(xz_ij.shape)
        #grad_i[np.isnan(grad_i)]=0        


    #return #ll_i, grad_i, Prob_ij             
    return Prob_ij, dPdb_k_aux2, dPdg_k_aux2


### Defaults: need something to deal with 1 vs many defs
@jit(nopython=True, nogil=True)#, parallel=True) 
def calc_def_ll_gradient(beta,
                             gamma,
                             design,
                             att_design,
                             choice_vector,
                             in_C,
                             def_vec,
                             startv, 
                             endv,
                             startd,
                             endd,
                             use_gradient=True):

   
    # Calculate the probability of each individual choosing each available
    # alternative for that individual.   
    nbetas = design.shape[1]
    ngammas = att_design.shape[1]
        #nrow = len(choice_vector)        
    
    exb = calc_exb(design, beta)
    
    ### Consideration probs      
    Mu_ij = calc_aprob_att(gamma, att_design) # not sure what to do with >1 
    #defaults (have def des etc be reduced to means per obs?) Random per obs? mean mu?
    #should use the reduction stuff I developed earlier
    
    #C_probs = concat columnswise [mu , 1-mu]
      
    ll = np.array([0.0])
    grad = np.zeros(nbetas+ngammas) 
    mult = np.array([1.0, -1.0]).reshape(-1, 1)

    for i in np.arange(len(startv)):
        # Extract needed variables
        s = startv[i]
        e = endv[i]
        sd = startd[i]
        ed = endd[i]
        y_ij = choice_vector[s:e]
        x_ij = getrows(design, s, e)
        exb_ij = getrowsvec(exb, s, e)
        in_C_ijr = getrows(in_C, s, e)
        def_vec_ij = def_vec[s:e]              
        xz_ij = getrows(att_design, sd, ed)      

        ### utility probs
            
        denom_ir = (exb_ij.T @ in_C_ijr).ravel()
        denom_ir[np.isnan(denom_ir)]=1e-30
        denom_ir[denom_ir==0]=1e-30
        denom_vec_ir = denom_ir.reshape((1,-1))
        
        ### Utility probs
        uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 
        
        probs_def_ij = subset1d(uprobs_ijr, 0)
        
        probs_def_ij_short = probs_def_ij[def_vec_ij]  
        
        
        ### Search probs 
             
        mu_ij = Mu_ij[sd : ed]
        mu_i = (probs_def_ij_short @ mu_ij)

        mu_ij_long = np.zeros(probs_def_ij.shape)
        mu_ij_long[def_vec_ij] = mu_ij
        
        C_prob_ir = np.array([mu_i, (1-mu_i)]).reshape(1,-1)

        ### Overall probs
        Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()        
        
        ll_ij = np.log(Prob_ij)
        ll_ij[np.isinf(ll_ij)] = 0
        ll_ij[np.isnan(ll_ij)] = 0
        
        # Calculate the log likelihood
        ll_i = y_ij @  ll_ij
        ll += ll_i

        
        if use_gradient:  #parallelise this? or don't bother as might not use in sim anneal?
            
            Prob_ij[Prob_ij==0] = 1e-30
            dLdP = (y_ij / Prob_ij).reshape((-1,1)) # j x 1
                        
            #betas
            dMdb_k = (mu_ij_long * probs_def_ij).reshape(1,-1) @\
                                    (x_ij - probs_def_ij.T@x_ij)  #1 xkb
            
            dLdb_k1 = (dLdP.T@(uprobs_ijr @ mult)).ravel() * dMdb_k 

            dPdb_k_aux = (uprobs_ijr * C_prob_ir) @ (x_ij.T @ uprobs_ijr).T 
                       
            dLdb_k2 = dLdP.T @ (Prob_ij.reshape((-1, 1))*x_ij - dPdb_k_aux) #j x kb
                      
            dLdb_k = (dLdb_k1 + dLdb_k2).ravel()
            
            # gammas
            dMdg_k = (probs_def_ij_short * (mu_ij *\
                                                (1-mu_ij))).T @ xz_ij #1 x kg             
            
            dLdg_k = (dLdP.T@(uprobs_ijr @ mult)).ravel() * dMdg_k  # 1 x kg   
            
            grad_i = np.concatenate((dLdb_k, dLdg_k))
            grad_i[np.isnan(grad_i)]=0 
            grad = grad + grad_i
            

    return ll[0], grad     


@jit(nopython=True, nogil=True, fastmath=True)#, parallel=True) 
def calc_def_ll_gradient_lc(params,
                             design,
                             att_design,
                             choice_vector,
                             demogs,
                             nclasses,
                             in_C,
                             def_vec,
                             startv, 
                             endv,
                             startd,
                             endd,
                             use_gradient=False):
    

    
    nbetas = design.shape[1]
    ngammas = att_design.shape[1]  #? len gammas?
    nvars = nbetas + ngammas
    ndeltas = demogs.shape[1]
    deltas = params[-ndeltas : (len(params)+1)]
    #nrows = design.shape[0]
    ll = np.array([0.0])
    grad = np.zeros(ndeltas+nclasses*(nvars))
       
    mult = np.array([1.0, -1.0]).reshape(-1, 1)
    
    for i in range(len(startv)):
        s = startv[i]
        e = endv[i]
        sd = startd[i]
        ed = endd[i]        
        
        grad_i = np.zeros(ndeltas+nclasses*(nvars))
        Prob_ij = np.zeros(e-s)
        dPdd_aux = np.zeros(e-s)  
        
        y_ij = choice_vector[s:e]
        x_ij = getrows(design, s, e)
        dem_ij = getrows(demogs, s, e)
        in_C_ijr = getrows(in_C, s, e)
        def_vec_ij = def_vec[s:e]              
        xz_ij = getrows(att_design, sd, ed)      

        nrows_ij = x_ij.shape[0]

        dPdb_k =  np.zeros(nrows_ij*nbetas*nclasses)
        dPdg_k =  np.zeros(nrows_ij*ngammas*nclasses)
        
        #paralelalise this bit?  
        for c in range(nclasses):
            
            
            ### TWO CLASS SHORTHAND 
            sc = calc_aprob_att(deltas, dem_ij)[0]
            if sc==0:
                sc = 0.0000000001
            elif sc==1:
                sc = 0.9999999999
            
            share_ic = (1-c)*sc + c * (1-sc)
            
            if nclasses ==1:
                share_ic = 1
            
            # Standard calcs, within class
            
            param = params[c*nvars:(c+1)*nvars]
            beta = param[0:nbetas]
            gamma = param[nbetas:nvars]
            
            ### Consideration probs  
            mu_ij = calc_aprob_att(gamma, xz_ij)

            exb_ij = calc_exb(x_ij, beta).reshape(-1, 1)
            
            Prob_ijc, dPdb_k_aux, dPdg_k_aux = calc_def_ll_grad_inner_lc(x_ij, 
                           xz_ij, y_ij, mu_ij, exb_ij, def_vec_ij, mult,
                           in_C_ijr, use_gradient)


            Prob_ij += share_ic*Prob_ijc
   
            
            if use_gradient:
                dPdd_aux += (1-2*c)*(1-share_ic) * share_ic * Prob_ijc    
                
                
                dPdg_k[(nrows_ij*ngammas)*c:(nrows_ij*ngammas)*(c+1)] =\
                        (share_ic * dPdg_k_aux).ravel()   
                        
                dPdb_k[(nrows_ij*nbetas)*c:(nrows_ij*nbetas)*(c+1)] =\
                            (share_ic *dPdb_k_aux).ravel()
        
        ll_ij = np.log(Prob_ij)
        ll_ij[np.isinf(ll_ij)] = 0
        ll_ij[np.isnan(ll_ij)] = 0
        ll_i = y_ij @  ll_ij
        ll += ll_i
     
        if use_gradient:
            dLdP = (y_ij / Prob_ij).reshape((-1,1)) # j x 1
            #vectorize?
            #calculate the derivatives for deltas 
            for i in range(ndeltas):  # dPdd * dem_ij 
                dem_col = dem_ij[:,i]
                dPdd_aux2 = dPdd_aux * dem_col
                gradd_ik = dLdP.T@ dPdd_aux2
                grad_i[len(grad_i) - (ndeltas-i)] = gradd_ik[0]           
            
            for c in range(nclasses):
                
                dPdg_kc = dPdg_k[
                            (nrows_ij*ngammas)*c :(nrows_ij*ngammas)*(c+1)
                            ].reshape(nrows_ij,ngammas) 
                dPdb_kc = dPdb_k[
                          (nrows_ij*nbetas)*c :(nrows_ij*nbetas)*(c+1)
                          ].reshape(nrows_ij,nbetas) 
                
                #betas 
                dLdb_k =  (dLdP.T @ dPdb_kc).ravel()   #_need 1_ x kb or kb x 1
                                         
                #gammmas
                dLdg_k = ( dLdP.T @ dPdg_kc).ravel() # 1 x kb   
            
                grad_i[c*nvars:(c+1)*nvars] =  np.concatenate((dLdb_k, dLdg_k))
            
           
            grad = grad + grad_i
  
    return ll[0], grad  


        
@jit(nopython=True, nogil=True, fastmath=True)    
def calc_def_ll_grad_inner_lc(x_ij, 
                           xz_ij, y_ij, mu_ij, exb_ij, 
                           def_vec_ij, mult,
                           in_C_ijr, use_gradient):
      
    ### Utility probs  
    denom_ir = (exb_ij.T @ in_C_ijr).ravel()
    denom_ir[np.isnan(denom_ir)]=1e-30
    denom_ir[denom_ir==0]=1e-30
    denom_vec_ir = denom_ir.reshape((1,-1))
    
   
    uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 
            
    probs_def_ij = subset1d(uprobs_ijr, 0)
    
    probs_def_ij_short = probs_def_ij[def_vec_ij]  
        
        
    ### Search probs 
         
    # Need to get uprobs[:, 1] @ mu_long = True Mu . What of derivatives then? Calc.
    # need to find a way to subset to the right set


    mu_i = (probs_def_ij_short @ mu_ij)

    mu_ij_long = np.zeros(probs_def_ij.shape)
    mu_ij_long[def_vec_ij] = mu_ij
      
    ### Consideration set probs 
 
    C_prob_ir = np.array([mu_i, (1-mu_i)]).reshape(1,-1)



    ### Overall probs
    Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()        
    Prob_ij[Prob_ij==0] = 1e-30

    ### Gradient
    if use_gradient:
        
        #betas
        dMdb_k = (mu_ij_long * probs_def_ij).reshape(1,-1) @\
                                (x_ij - probs_def_ij.T@x_ij)  #1 xkb
        
        dPdb_k1 = np.outer((uprobs_ijr @ mult), dMdb_k)
        dPdb_k2 = (uprobs_ijr * C_prob_ir) @ (x_ij.T @ uprobs_ijr).T 
                   
        dPdb_k = dPdb_k1 + Prob_ij.reshape((-1, 1))*x_ij - dPdb_k2 #j x kb
        
        # gammas
        dMdg_k = (probs_def_ij_short * (mu_ij *\
                                            (1-mu_ij))).T @ xz_ij #1 x kg                     
        dPdg_k = np.outer((uprobs_ijr @ mult), dMdg_k)  # j x kg             

    else:
        dPdb_k = np.zeros(x_ij.shape)
        dPdg_k = np.zeros(xz_ij.shape)
          
    return Prob_ij, dPdb_k, dPdg_k



            

            


    """   
### Simulated likelihood CHECK GRADIENT

@jit(nopython=True) 
def calc_att_ll_gradient_sim(beta,
                             gamma,
                             design,
                             att_design,
                             choice_vector,
                             sim_draws,
                             ndraws,
                             in_C0,
                             C0probs,                             
                             consider_vec,
                             startv, 
                             endv,
                             use_gradient=True,
                             weights=None):

    # Calculate the probability of each individual choosing each available
    # alternative for that individual.   
    nobs, nbetas = design.shape
    ngammas = att_design.shape[1]
        #nrow = len(choice_vector)        
    limit =  1e-10
    exb = calc_exb(design, beta)
    
    ### Consideration probs      
    Aprobs_ij = calc_aprob_att(gamma, att_design)
    Aprobs_ij[consider_vec==1]=0.9999999
    
    """ 
   # inCvec = np.zeros(nobs*ndraws)
   # for loop in np.arange(ndraws):
   #    st = loop * nobs
   #    en = (loop+1) * nobs
   #    inCvec[st:en] = (Aprobs_ij>=sim_draws[st:en])*1.0
     
   # inCmat = (inCvec.reshape((ndraws, nobs))).T # cant reshape here
    """
    
    ll = np.array([0.0])
    grad = np.zeros(nbetas+ngammas) 

    for i in range(len(startv)):
        # Extract needed variables
        s = startv[i]
        e = endv[i]
        y_ij = choice_vector[s:e]
        x_ij = getrows(design, s, e)
        xz_ij = getrows(att_design, s, e)
        cons_ij = consider_vec[s:e]
        #aprobs_ij = Aprobs_ij[s:e]
        aprobs_ij = getrowsvec(Aprobs_ij, s, e)
        exb_ij = getrowsvec(exb, s, e)
#       nprod = e-s  
               
        in_C_ijr =getrows(in_C0, s, e)        
        C0probs_ir = getrows(C0probs, i, i+1)        
        #if (in_C_ijr.sum(axis=0)==0).sum()>0:
        #    print(i)
            
        if weights is None:
            weights_ij = np.array([1.0])
        else:
            weights_ij = getrowsvec(weights, s, e).ravel()
            
        log_aprob_c = np.log(aprobs_ij)
        log_aprob_nc = np.log(1-aprobs_ij)
            
        ### Consideration set probs 
             
        C_prob_temp = log_aprob_c.T @ in_C_ijr + log_aprob_nc.T @ (1-in_C_ijr) 
        C_prob_ir = np.exp(C_prob_temp) /C0probs_ir
    
        denom_ir = (exb_ij.T @ in_C_ijr).ravel()
        denom_ir[np.isnan(denom_ir)]=1e-30
        denom_ir[denom_ir==0]=1e-30
        denom_vec_ir = denom_ir.reshape((1,-1))
        
        ### Utility probs
        uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 
    
        ### Overall probs
        Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel() / ndraws   
        
        Prob_ij[np.isnan(Prob_ij)] = 1e-20
        Prob_ij[Prob_ij==0.] = 1e-20
        #ll_ij = np.multiply(weights_ij, np.log(Prob_ij))
        l_i = y_ij @ Prob_ij
        if l_i < limit:
            ll += (l_i - limit)/limit
        
        #ll_ij[np.isinf(ll_ij)] = 0
        #ll_ij[np.isnan(ll_ij)] = 0
        
        # Calculate the log likelihood
        #ll_i = y_ij @  ll_ij
        ll_i = np.log(l_i)
        ll += ll_i
        
        if use_gradient:# | use_hessian:
            
            
            dLdP = (y_ij / Prob_ij * weights_ij).reshape((-1,1)) # j x 1
            
            dLdP_aux = dLdP.T @ uprobs_ijr # 1 x r(ncons)
            
            dPdb_k_aux = (x_ij.T @ uprobs_ijr) # kb x r
            dLdb_k_t = Prob_ij.reshape((1, -1)) @ (x_ij * dLdP) -\
                         dLdP_aux @ (dPdb_k_aux * C_prob_ir).T / ndraws #_need 1_ x kb or kb x 1                         
            dLdb_k = dLdb_k_t.ravel()
            
            dPdg_k_aux = (xz_ij.T @ (in_C_ijr - aprobs_ij))#kg x r
            dLdg_k = (dLdP_aux @ (C_prob_ir * dPdg_k_aux).T).ravel() /ndraws # 1 x kb   
            
            grad_i = np.concatenate((dLdb_k, dLdg_k)) 
            grad_i[np.isnan(grad_i)]=0 
          
            grad = grad + grad_i

    return ll[0], grad  
"""
    
### Auxilliary calculations 
@jit(nopython=True, parallel=True)
def calc_aprob_att(gamma, att_design):
    min_comp_value = 1e-300
    min_exponent_val = -700
    max_exponent_val = 700
    
    temp = att_design @ gamma
    too_small_idx = temp < min_exponent_val
    too_large_idx = temp > max_exponent_val

    temp[too_small_idx] = min_exponent_val
    temp[too_large_idx] = max_exponent_val
    
    temp = np.exp(temp)
    aprobs_ij = temp/(1+temp)  
    aprobs_ij[aprobs_ij == 0] = min_comp_value 
    aprobs_ij[aprobs_ij == 1] = 0.999999999
    
    return aprobs_ij

@jit(nopython=True)
def calc_logaprobs(aprobs_ij, nropt_i, maxopt):
    
    log_aprob_c1 = np.zeros(maxopt)
    log_aprob_c1[:nropt_i] = np.log(aprobs_ij)              
    log_aprob_c1[nropt_i:] = 0
    log_aprob_c = log_aprob_c1.reshape(1, -1)
    
    log_aprob_nc1 = np.zeros(maxopt)
    log_aprob_nc1[:nropt_i] = np.log(1-aprobs_ij)
    log_aprob_nc1[nropt_i:] = 0
    log_aprob_nc = log_aprob_nc1.reshape(1, -1)   
    
    return log_aprob_c, log_aprob_nc
    

@jit(nopython=True, parallel=True)
def calc_exb(design, beta):

    min_exponent_val = -700
    max_exponent_val = 700

    xb = design @ beta
    too_small_idx = xb < min_exponent_val
    too_large_idx = xb > max_exponent_val

    xb[too_small_idx] = min_exponent_val
    xb[too_large_idx] = max_exponent_val
    
    exb = np.exp(xb)
    
    # Calculate the log likelihood
    return exb

@jit(nopython=True)
def calc_xdphi(gamma, xz):
    min_exponent_val = -700
    max_exponent_val = 700
    
    temp = xz @ gamma
    too_small_idx = temp < min_exponent_val
    too_large_idx = temp > max_exponent_val

    temp[too_small_idx] = min_exponent_val
    temp[too_large_idx] = max_exponent_val
    
    temp = np.exp(temp)
    # e^-x / (1 + e^-x)^2   == ex / (1+ex)^2
    logden = temp/((1+temp)**2) 
    xdphi = logden.reshape(xz.shape[0], 1) * xz

    return xdphi    

### Data element extractions for jit
#@jit(float64[:,:](float64[:,:], int64), nopython=True)
@jit(nopython=True)
def subset(x, mycol):
    xcol = np.copy(x[:,mycol])
    xcol = xcol.reshape((xcol.shape[0], -1))
    return xcol 

@jit(nopython=True)
def subset1d(x, mycol):
    xcol = np.copy(x[:,mycol])
    #xcol = xcol.reshape((xcol.shape[0], -1))
    return xcol 

@jit(nopython=True)
def getrows(array, start, end):
    res = array[start:end, :]
    return res
"""
@jit(nopython=True)
def getfromlist(list, ref):
    res = list[ref]
    return res
"""
@jit(nopython=True)
def getrowsvec(array1d, start, end):
    array=array1d.reshape((len(array1d), 1))
    
    res = array[start:end, :]
    return res
"""
@jit(nopython=True)
def getrowsandcols(array, rstart, rend, cstart, cend):
    res = array[rstart:rend, cstart:cend]
    return res
"""

@jit(nopython=True)
def getcols(array, cstart, cend):
    res = array[:, cstart:cend]
    return res
    