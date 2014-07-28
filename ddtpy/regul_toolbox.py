import numpy as np


# This part does ddt_setup_regularization_galaxy_lambda_normalized_3
class RegulGalaxyXY():

    """
    This class is a conversion of these functions from ddt_regul_toolbox.i:
    - ddt_setup_regularization_galaxy_xy_normalized_1
    - ddt_mu_estim_xy_normalized_1
    - _ddt_setup_regul_2d_variable_mu
    - _ddt_eval_regul_2d_variable_mu
    #Comments from DDT
    *   Prepare regularizations for the different components of the model
    *   and along different directions and store the regularizator inside
    *   ddt_model. --> Setting to ddt_model will probably happen in ddt_data.py

    * DOCUMENT _ddt_setup_regul_2d_variable_mu(fn, 
                         which(1), which(2), hyper, var);
    * hyper is the hyper-parameter
    * var is the variance alond the lambda axis
    """
    
    def __init__(self, data_ref, weight_ref, mu, sky=None):
        
        h_mu_estim_q, h_mu_estim = self.mu_estim_xy_normalized_1(data_ref, 
                                       weight_ref, sky=sky)
        weight = h_mu_estim**2
        
        fn = rgl_roughness_l2 # This is defined somewhere in yeti.doc
        # rgl_roughness computes a regularization penalty based on roughness, 
        # l2 means cost function "L2 norm"
    
        # X and Y are the 1st and 2nd dimensions in the galaxy array
        which1, which2 = np.array([-1,-2]) 
        # Not sure about above indices
        hyper = np.zeros(data_ref.shape[0])
        if mu != None: 
            hyper += mu
    
        if not isinstance(which1, int):
            raise ValueError("expecting a scalar integer for WHICH1")
            
        if not isinstance(which2, int):
            raise ValueError("expecting a scalar integer for WHICH2")
            
        self.off1 = np.zeros(4)
        self.off2 = np.zeros(4)
        self.off3 = np.zeros(4)
        self.off4 = np.zeros(4)
        self.off1[which1] =  1
        self.off2[which2] =  1
        self.off3[which1] =  1
        self.off3[which2] =  1
        self.off4[which1] =  -1
        self.off4[which2] =  1
        # takes into account the weight as a multiplicative 
        # factor for each wavelength slice */
        self.fn = fn
        self.hyper = hyper*weight
        self.q = weight
        self.factor = 1.0
        self.mu_estim = h_mu_estim

    def mu_estim_xy_normalized_1(self, data_ref, weight_ref, 
                                 sky=None):
        """Return mu estimate and inverse of spectrum
        This function does a lot of work to compute mu = size of the data 
        divided an estimate of the element-wise variation in the data
        and then just sets mu to one regardless.
        
        Parameters
        ----------
        data_ref : 3 or 4-d array
        weight_ref : 3 or 4-d array
        sky : 1 or 2-d array
              the above variables have an additional dimension if there are 
              multiple final refs
        no_norm : bool
        
        Returns
        -------
        mu_estim : float
        q : 1-d array
        """
        if len(data_ref.shape) < 4:
            d = np.array([data_ref])
            w = np.array([weight_ref])
            s = np.array([sky])
            
        else:
            d = data_ref
            w = weight_ref
            s = sky
        
        if sky is not None:
            # Current understanding of this: get average of sky along lambda 
            # axis for each day, subtract that from sky spectrum for each day
            # Then subtract resulting spectra for each day from data.
            d -= (s - s.mean(axis=1)[:,None])[:,:,None,None]
            
        # Average along x,y, and epoch axes:
        avg_spectrum = d.mean(axis=-1).mean(axis=-1).mean(axis=0)
        assert len(avg_spectrum) == d.shape[1] 
        
        q = 1./avg_spectrum
        
        N_xy = w.size
        var_ref = 1/w
        
        i_bad = w<= 0
        N_xy -= np.sum(i_bad)
        
        """
        # Comments from DDT:    
        This is what is writen in the paper, including the variance terms
        The calculation is made so that the variance of the terms included in 
        computation of d_x and d_y are accounted for
        """
        
        d_x = d[:,:,:,1:] - d[:,:,:,:-1]
        # Terms that play a role in the dif above
        d_var_x = var_ref[:,:,:,:-1] + var_ref[:,:,:,1:] 
        
        i_ok = np.int_(0.5*(i_bad[:,:,:,1:] + i_bad[:,:,:,:-1]))
        # Only elements where none of elements used in dif (d_x) had zero weight
        i_ok =  (i_ok == 0) 
        d_x *= i_ok
        d_var_x *= i_ok
        mu_estim_x = (d_x**2 - d_var_x)*(q**2)[None,:,None,None]
        mu_estim_x = np.sum(mu_estim_x)

        d_y = d[:,:,1:,:] - d[:,:,:-1,:]
         # Terms that play a role in the dif above
        d_var_y = var_ref[:,:,:-1,:] + var_ref[:,:,1:,:]
        
        i_ok = np.int_(0.5*(i_bad[:,:,1:,:] + i_bad[:,:,:-1,:]))
        # Only elements where none of elements used in dif (d_y) had zero weight
        i_ok =  (i_ok == 0) 
        d_y *= i_ok
        d_var_y *= i_ok
        mu_estim_y = (d_y**2 - d_var_y)*(q**2)[None,:,None,None]
        mu_estim_y = np.sum(mu_estim_y)
        
        mu_estim = mu_estim_x + mu_estim_y
        mu_estim = N_xy/mu_estim
        
        if mu_estim < 0:
            print ("<ddt_mu_estim_xy_normalized_1> WARNING: mu_estim was < 0."+
                   "Not enough information in the data => changed it to be 1.")
            mu_estim = 1 
     
        ## There is a "FIXME" in DDT code here:
        print ("<ddt_mu_estim_xy_normalized_1> WARNING: "
                "mu_estim=1 no matter what")
        mu_estim = 1.
        
        return q, mu_estim
        
    # Where do x and grd come from? Not evident in above use.
    def __call__(self, x, grd): 
        """Get regularization based on mu, regularization function
         From "_eval_regul_2d_variable_mu"
         
        Parameters
        ----------
        x: ?
        grd: ? Should be pointer
        
        Returns
        -------
        regul: float?
        """
            
        if self.mu_estim :
            self.hyper *= self.mu_estim
            
        # regularization weight for diagonal terms, 
        # see yeti def of rgl_roughness_l2
        # diagonal regularization is apparently not needed.
        #hyper2 = 0.5*self.hyper
        fn = self.fn
        
        regul = 0.
        dim_gal = x.shape
        if grd is None:
            for i_l in range(dim_gal[0]):
                """
                Comment from DDT:
                * FIXME: in the paper we show that the spatial regularization 
                * should only include 2 terms, not the diagonal ones
                Then some stuff was commented out.
                """
                    
                regul += (fn(self.hyper[i_l], [1,0], x[i_l,:,:]) +
                          fn(self.hyper[i_l], [0,1], x[i_l,:,:]))
                              
        else:    
            for i_l in range(dim_gal[0]):
                grd_temp = np.zeros((dim_gal[0], dim_gal[1],2))
                # FIXME same as above
                regul += (fn(self.hyper[i_l], [1,0], x[i_l,:,:], grd=grd_temp)+
                          fn(self.hyper[i_l], [0,1], x[i_l,:,:], grd=grd_temp))
                              
        return regul
                          
        
        
        
# Following is for lambda normalization:
# This is a translation of ddt_setup_regularization_galaxy_lambda_normalized_3
# The DDT code says this is from eq 33 in the paper, which doesn't exist. 
# Must be spectral parts of eq 26-29
class RegulGalaxyLambda():
    """
    This class is a conversion of these functions from ddt_regul_toolbox.i:
    - ddt_setup_regularization_galaxy_lambda_normalized_3
    - ddt_mu_estim_lambda_normalized_3
    - _ddt_setup_regularization_lambda_normalized_3
    - _ddt_eval_regularization_galaxy_lambda_normalized_3
    """

    def __init__(self, data_ref, weight_ref, mu, sky=None):
    
        if mu is None:
            mu = 0
        
        q, mu_estim = self.mu_estim_lambda_normalized_3(data_ref, weight_ref, 
                                                        sky=sky) 

        self.mu = mu
        self.q = q
        self.mu_estim = mu_estim
    
    def mu_estim_lambda_normalized_3(self, data_ref, weight_ref, sky=None):
        """Return mu estimate and inverse of spectrum
        This function just sets mu_estim to 1 and calculates q.
        
        Parameters
        ----------
        data_ref : 3 or 4-d array
        weight_ref : 3 or 4-d array
        sky : 1 or 2-d array
              the above variables have an additional dimension if there are 
              multiple final refs
        
        Returns
        -------
        mu_estim : float
        q : 1-d array
        """

        if len((data_ref).shape) < 4:
            d = data_ref[None,:,:,:]
            w = weight_ref[None,:,:,:]
            s = sky[None,:]
        else:
            d = data_ref
            w = weight_ref
            s = sky    
        
        if sky is not None:
            d -= (s - np.mean(s, axis=1)[:,None])[:,:,None, None]
            
        avg_spectrum = d.mean(axis=-1).mean(axis=-1).mean(axis=0)

        assert len(avg_spectrum) == d.shape[1]
        
        q = 1./avg_spectrum
        N_xy = w.size
        
        """
        Comment from DDT:
        * This is what is writen in the paper, including the variance terms
        * The calculation is made so that the variance of the terms included in
        * computation of d_x and d_y are accounted for
        """
        # From DDT : FIXME, calculation not implemented yet 
        # (should this be like in xy_norm above?)
        mu_estim = 1.
        print ("<ddt_mu_estim_xy_normalized_3> WARNING: "+
               "mu_estim calculation not implemented, set to 1.")
            
        return q, mu_estim
                
                        
    def __call__(self, x, grd):
        """Get regularization based on mu, regularization function
        Also changes grd.
         
        Parameters
        ----------
        x: ?
        grd: ? Should be pointer
        
        Returns
        -------
        rgl: float?

        Notes:
        -----
        Here q is the parameter called mu^cal in ddt-paper-a     
        """
        
        r = x*self.q[:,None,None]
        r = x[1:,:,:] - x[:-1,:,:] # Why is r reassigned? Which is right?
        rgl = self.mu*self.mu_estim*np.sum(r**2)
        
        # grd gets changed here--need to see where grd comes from
        # to decide how to translate this
        if grd is not None:
            grd_tmp = np.zeros(x.shape)
            grd_tmp[1:,:,:] = r
            grd_tmp[:-1,:,:] -= r
            grd += (2.* self.mu*self.mu_estim) * self.q[:,None, None]*grd_tmp
            del grd_tmp
            
        return rgl
   

def regul_g(ddt, x, grd, debug=None):
    """
    *   Compute regularization for 'g' (galaxy) or 'h' (PSF's).
    *   The returned value ERR is the penalty.
    *   Argument X is the array of parameters
    *   Optional argument GRD is the output gradient, it must have been
    *   already initialized and its contents is incremented by the gradient
    *   of regularization w.r.t. the parameters X.
    """
   
    galaxy = x
    
    gradient =  not (grd is None)
    if gradient:   
        galaxy_grd = np.zeros(galaxy.shape)
  
    galaxy -= ddt.model_galprior
    galaxy_err = (ddt.regul_galaxy_xy(galaxy, galaxy_grd) +
                  ddt.regul_galaxy_lambda(galaxy, galaxy_grd))
  
    # TODO: below was debug option:
    # galaxy_grd is probably something that gets printed to header if debug=1.
    #h_set, ddt_model, galaxy_grd=galaxy_grd
  
    if gradient:
        grd += galaxy_grd
        del galaxy_grd

  
    return galaxy_err;

def rgl_roughness_l2(hyper, offset, arr, grd=None):
    """Regularization penalty based on roughness of arr
    L2 is cost function mu*x^2, "From a Bayesian viewpoint, L2 corresponds to 
    the neg-log likelihood of a Gaussian distribution"
    
    Parameters
    ----------
    hyper : float
    offset : 1-d array
    arr : 2-d array
    grd : 3-d array - isn't used right now.
    
    Returns
    -------
    err : float
    """
    ind_y, ind_x = arr.shape
    d_arr = arr[offset[0]:,offset[1]:] - arr[:ind_y-offset[0],:ind_x-offset[1]]
    cost = hyper * d_arr**2
    
    return np.sum(cost)
    
    