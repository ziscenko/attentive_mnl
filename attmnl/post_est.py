### Post-estimation calculations

### Hessian

def calc_att_ll_gradient_hess(beta,
                                 gamma,
                                 design,
                                 att_design,
                                 choice_vector,
                                 consider_vec,
                                 startv, 
                                 endv,
                                 use_gradient=True,
                                 use_hessian=True, 
                                 weights=None,
                                 Clist = None,
                                 rowmap =None,
                                 nropt = None,
                                 C_temp = None,
                                 faster = True):

   
    # Calculate the probability of each individual choosing each available
    # alternative for that individual.   
    nbetas = design.shape[1]
    ngammas = att_design.shape[1]
    npar = nbetas + ngammas
        #nrow = len(choice_vector)        
    
    exb = calc_exb(design, beta)
    
    ### Consideration probs      
    Aprobs_ij = calc_aprob_att(gamma, att_design)
    Aprobs_ij[consider_vec==1]=0.9999999
    
    if use_hessian:
        Xdphi_ij = calc_xdphi(gamma, att_design)
        consider_mat = np.tile(consider_vec.reshape(-1, 1), ngammas)
        Xdphi_ij[consider_mat==1]=0 # NOT for DEFAULT though
    
    ll = np.array([0.0])
    grad = np.zeros(npar)
    grad_long = np.zeros([startv.shape[0],npar]) 
    #H = np.zeros([nbetas+ngammas, nbetas+ngammas])
    Hall = np.zeros(int((npar+1)*npar/2))
    
    for i in range(len(startv)):
        # Extract needed variables
        s = startv[i]
        e = endv[i]
        nropt_i = nropt[i] 
        x_ij = getrows(design, s, e)
        xz_ij = getrows(att_design, s, e)
        y_ij = choice_vector[s:e]
        exb_ij = getrowsvec(exb, s, e)
        
        aprobs_ij = getrowsvec(Aprobs_ij, s, e)        
        rowmap_ij = getrows(rowmap, s, e)
        ncol_i = 2**nropt_i       
        C_i=getcols(C_temp, 0, ncol_i)        
        in_C_ijr = rowmap_ij @ C_i

        log_aprob_c, log_aprob_nc = calc_logaprobs(aprobs_ij, rowmap_ij, 
                                                   nropt_i, s, e)        
            

        if weights is None:
            weights_ij = np.array([1.0])
        else:
            weights_ij = getrowsvec(weights, s, e).ravel()
        #H_i = np.zeros([nbetas+ngammas, nbetas+ngammas])
        
        if use_hessian:
            xdphi_ij = getrows(Xdphi_ij, s, e)
            haux = np.triu(xdphi_ij.T @ xz_ij) 
        else:
            haux =np.zeros([ngammas, ngammas])
            
        ll_i, grad_i, H_i = calc_att_ll_grad_hess_inner(x_ij, xz_ij, y_ij, 
                                                    aprobs_ij, exb_ij, 
                                                    weights_ij, in_C_ijr, C_i,
                                                    log_aprob_c, log_aprob_nc,
                                                    use_gradient, 
                                                    use_hessian,
                                                    haux)
            
        ll += ll_i
        if use_gradient:
            grad = grad + grad_i
            grad_long[i] = grad_i 
        if use_hessian:
            Hall = Hall + H_i
    
    
    H = np.zeros([npar, npar])
    mapped = 0
    for row in np.arange(npar):
        H[row, row:npar] = Hall[mapped:(mapped+npar-row)]
        #print('H[',row,',' ,row, ':',(npar),'] = Hall[',mapped,':',(mapped+npar-row),']'  )
        mapped = mapped +npar - row
        
    H = np.triu(H)+np.triu(H).T - np.eye(ngammas+nbetas)*H 
            
    return ll[0], grad, grad_long, H  
   
@njit(fastmath=True, nogil=True)#(parallel=True)  
def calc_att_ll_grad_hess_inner(x_ij, xz_ij, y_ij,
                           aprobs_ij, exb_ij, weights_ij, in_C_ijr, C_i,
                           log_aprob_c, log_aprob_nc,
                           use_gradient, use_hessian, haux):
    
    
    nbetas = x_ij.shape[1]
    ngammas = xz_ij.shape[1]
    npar = nbetas+ngammas
    nele = int((npar+1)*npar/2)

        
    ### Consideration set probs          
    C_prob_temp = log_aprob_c @ C_i + log_aprob_nc @ (1-C_i) 
    C_prob_ir = np.exp(C_prob_temp)

    denom_ir = (exb_ij.T @ in_C_ijr).ravel()
    denom_ir[np.isnan(denom_ir)]=1e-30
    denom_ir[denom_ir==0]=1e-30
    denom_vec_ir = denom_ir.reshape((1,-1))
    
    ### Utility probs
    uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir 

    ### Overall probs
    Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()        
    
    ll_ij = np.multiply(weights_ij, np.log(Prob_ij))
    ll_ij[np.isinf(ll_ij)] = 0
    ll_ij[np.isnan(ll_ij)] = 0
    
    # Calculate the log likelihood
    ll_i = y_ij @  ll_ij
    
    grad_i=np.zeros(npar)
    H_i = np.zeros(nele)
    ### Gradient
    if use_gradient:# | use_hessian:
 
        Prob_ij[Prob_ij==0] = 1e-30
        dLdP = (y_ij / Prob_ij * weights_ij).reshape((-1,1)) # j x 1
        
        dLdP_aux = dLdP.T @ uprobs_ijr # 1 x r(ncons)
        
        dPdb_k_aux = (x_ij.T @ uprobs_ijr) # kb x r
        dPdg_k_aux = (xz_ij.T @ (in_C_ijr - aprobs_ij))#kg x r
        
        dLdb_k_t = Prob_ij.reshape((1, -1)) @ (x_ij * dLdP) -\
                     dLdP_aux @ (dPdb_k_aux * C_prob_ir).T #_need 1_ x kb or kb x 1
                     
        dLdb_k = dLdb_k_t.ravel()
        
        dLdg_k = (dLdP_aux @ (C_prob_ir * dPdg_k_aux).T).ravel() # 1 x kb   
        
        grad_i = np.concatenate((dLdb_k, dLdg_k))
        grad_i[np.isnan(grad_i)]=0        
        
    ### Hessian        
        if use_hessian:
            hb  = dLdb_k.reshape(-1, 1) @ dLdb_k.reshape(1, -1) # nb x nb
            hg  = dLdg_k.reshape(-1, 1) @ dLdg_k.reshape(1, -1) # ng x ng
            hbg = dLdb_k.reshape(-1, 1) @ dLdg_k.reshape(1, -1) # nb x ng
            
            ele = 0
            # betas
            for k in np.arange(nbetas):
                x_ijk = subset(x_ij, k)
                #H_ik = np.zeros(nbetas+ngammas)
                
                for l in np.arange(k, nbetas):
                    x_ijl = subset(x_ij, l) # j x 1
                    dPdb_l0 = uprobs_ijr*x_ijl -\
                                 uprobs_ijr * (x_ijl.T @ uprobs_ijr ) #j x r
                    dLdb_kl1 = dPdb_l0 * x_ijk # j x r
                    dLdb_kl2 = dPdb_l0 * getrows(dPdb_k_aux, k, k+1) # j x r * 1 x r =jxr
                    dLdb_kl3 = uprobs_ijr * (x_ijk.T @ dPdb_l0) # j x r
                    dLdb_kl0 = dLdb_kl1 - dLdb_kl2 - dLdb_kl3 # j x r
                    dPdb_kl =  (C_prob_ir @ dLdb_kl0.T ) @ dLdP # 1 x 1
                    H_i[ele] =  ((dPdb_kl - hb[k, l]).ravel())[0]
                    ele = ele+1
                    
                #crosses
                dPdb_kl_aux = uprobs_ijr*x_ijk # j x r
                dPdb_kl_aux = dPdb_kl_aux -\
                    (x_ijk.T @ uprobs_ijr)*uprobs_ijr # j x r
                    
                for l in np.arange(ngammas):
                    xz_ijl = subset(xz_ij, l) # jx 1
                    
                    dPdg_lk_aux1 = xz_ijl.T @ in_C_ijr
                    dPdg_lk_aux2 = (xz_ijl * aprobs_ij).sum()
                    dPdg_lk_aux = (dPdg_lk_aux1-dPdg_lk_aux2) * C_prob_ir # 1 x r                      
                    dPdbg_lk = (dPdg_lk_aux @ dPdb_kl_aux.T)@ dLdP
                    H_i[ele] =(dPdbg_lk - hbg[k, l]).ravel()[0]
                    ele = ele+1
                    
                #if k==0:
                    #H_i = H_ik.reshape(1, -1)
                #else:
                    #H_i = np.concatenate((H_i, H_ik.reshape(1, -1)))
            #gammas
            
            for k in np.arange(ngammas):
                #H_ik = np.zeros(nbetas+ngammas)
                for l in np.arange(k, ngammas):
                    dPdg_kl0 = getrows(dPdg_k_aux, l, l+1) *\
                        getrows(dPdg_k_aux, k, k+1) # 1 x r
                    dPdg_kl0 = (dPdg_kl0 - haux[k, l]) * C_prob_ir #1* r
                    dLdg_kl = dPdg_kl0 @ dLdP_aux.T - hg[k, l] # 1x1
                    H_i[ele] = dLdg_kl.ravel()[0]
                    ele = ele+1
                #H_i = np.concatenate((H_i, H_ik.reshape(1, -1)))


    return ll_i, grad_i, H_i




### Predict probabilities and derivatives
    
def calc_att_predict(beta,
                     gamma,
                     design,
                     att_design,
                     choice_vector,
                     consider_vec,
                     startv, 
                     endv, 
                     starta, 
                     enda,  
                     groupmap,
                     Clist = None,
                     rowmap =None,
                     nropt = None,
                     C_temp = None,
                     faster=True, 
                     exact = True):
    
    min_exponent_val = -700
    max_exponent_val = 700
    
    nbetas = design.shape[1]
    ngammas = att_design.shape[1]
    nrow = len(groupmap)        

    exb = calc_exb(design, beta)
    
    ### Consideration probs  
    Aprobs_ij = calc_aprob_att(gamma, att_design)
    Aprobs_ij_long =Aprobs_ij[groupmap]
    probs = {'PA':Aprobs_ij_long}
    
    temp = np.tile(Aprobs_ij_long, [ngammas, 1]).T
    dPA_long = temp *(1-temp) *gamma.reshape(1, -1)     
    Aprobs_ij[consider_vec==1]=0.9999999
   
    PU_long = np.zeros(nrow)
    PY_final =np.zeros(nrow)
    y_pred =np.zeros(nrow)
    
    dPU_long = np.zeros(design.shape)  
    dPY_u = np.zeros(design.shape)
    dPY_a = np.zeros((design.shape[0], ngammas))    
    #dPY_ua = somehow select dims of variables in both
    
    #PY_final =np.zeros(Aprobs_ij.shape)
    if exact:
        #att_gr_count_s = 0
        for i in range(len(startv)):
            # Extract needed variables
            #print(i)
            s = startv[i]
            e = endv[i]
            sa = starta[i]
            ea = enda[i]
            #print(i)
            nropt_i =nropt[i]        
            #att_gr_count_e = att_gr_count_s + nropt_i
             
            rowmap_ij = getrows(rowmap, s, e)
            ncol_i = 2**nropt_i      
            C_i=getcols(C_temp, 0, ncol_i)        
            in_C_ijr = rowmap_ij @ C_i

            C_i = getrows(C_i, 0, nropt_i)        

            exb_ij = getrowsvec(exb, s, e)
            
            aprobs_ij = getrowsvec(Aprobs_ij, sa, ea)  
       
            aprobs_ij_l = getrowsvec(Aprobs_ij_long, s, e) 
            
            PU, Prob_ij, dPU_aux, dPY_a_aux, dPY_u_aux =\
                            calc_att_predict_inner(aprobs_ij, exb_ij, 
                                                   in_C_ijr, C_i, nropt_i, 
                                                   aprobs_ij_l)
            
            
            dPU_long[s:e, :] = np.tile(dPU_aux, [1, nbetas]) * beta.reshape(1, -1)
            
            PU_long[s:e] = PU.ravel()            
            PY_final[s:e] = Prob_ij
            y_pred[s:e] = Prob_ij == Prob_ij.max()
            
            dPY_a[s:e, :] = np.tile(dPY_a_aux, [1, ngammas]) * gamma.reshape(1, -1)
            dPY_u[s:e, :] = np.tile(dPY_u_aux, [1, nbetas]) * beta.reshape(1, -1)
    
   
    probs['PY_final']=PY_final
    probs['y_pred']=y_pred
    probs['PU'] = PU_long
    derivs = {'dPA_long':dPA_long, 'dPU_long': dPU_long,
              'dPY_u':dPY_u, 'dPY_a': dPY_a} 
    
    return derivs, probs

@jit(nopython=True) 
def calc_att_predict_inner(aprobs_ij, exb_ij, in_C_ijr, C_i,
                           nropt_i, aprobs_ij_l):
    
    #log_aprob_c, log_aprob_nc = calc_logaprobs(aprobs_ij, rowmap_ij, 
    #                                               nropt_i, s, e)       
    
    log_aprob_c = np.log(aprobs_ij)
    log_aprob_nc = np.log(1-aprobs_ij)
    
    ### Consideration set probs          
    C_prob_temp = log_aprob_c.T @ C_i + log_aprob_nc.T @ (1-C_i) 
    C_prob_ir = np.exp(C_prob_temp)

    denom_ir = (exb_ij.T @ in_C_ijr).ravel()
    denom_ir[np.isnan(denom_ir)]=1e-30
    denom_ir[denom_ir==0]=1e-30
    denom_vec_ir = denom_ir.reshape((1,-1))
    
    ### Utility probs
    uprobs_ijr = (exb_ij * in_C_ijr) / denom_vec_ir     
    PU = exb_ij/ exb_ij.sum()
    
    #addvec = np.ones(nropt_i).reshape(1, -1)
    ### Partial derivs
    dPU_aux =  PU * (1-PU)
    #dPY_a_aux = uprobs_ijr @ ((addvec @ (C_i - aprobs_ij)) * C_prob_ir).T 
    # no cross derivs. and only salience of that bank. no sum
    dPY_a_aux = (uprobs_ijr * (in_C_ijr - aprobs_ij_l)) @ C_prob_ir.T 
    dPY_u_aux = (uprobs_ijr * (1-uprobs_ijr)) @ C_prob_ir.T 

    ### Overall probs
    Prob_ij = (uprobs_ijr @ C_prob_ir.T ).ravel()       
    
    return PU, Prob_ij, dPU_aux, dPY_a_aux, dPY_u_aux

