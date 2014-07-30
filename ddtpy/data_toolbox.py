import numpy as np


def sky_guess_all(ddt_data, ddt_weight, nw, nt):
    """guesses sky with lower signal spaxels compatible with variance
    Parameters
    ----------
    ddt_data, ddt_weight : 4d arrays
    nw, nt : int
    Returns
    -------
    sky : 2d array
    Notes
    -----
    sky_cut: number of sigma used for the sky guess, was unused option in DDT
    """

    sky = np.zeros((nt, nw))
    sky_cut = 2.0
    
    for i_t in range(nt):
        data = ddt_data[i_t]
        weight = ddt_weight[i_t]
        
        var = 1./weight
        ind = np.zeros(data.size)
        prev_npts = 0
        niter_max = 10
        niter = 0
        nxny = data.shape[1]*data.shape[2]
        while (ind.size != prev_npts) and (niter < niter_max):
            prev_npts = ind.size
            #print "<ddt_sky_guess> Sky: prev_npts %d" % prev_npts
            I = (data*weight).sum(axis=-1).sum(axis=-1)
            i_Iok = np.where(weight.sum(axis=-1).sum(axis=-1) != 0.0)
            if i_Iok[0].size != 0:
                I[i_Iok] /= weight.sum(axis=-1).sum(axis=-1)[i_Iok]
                sigma = (var.sum(axis=-1).sum(axis=-1)/nxny)**0.5
                ind = np.where(abs(data - I[:,None,None]) > 
                               sky_cut*sigma[:,None,None])
                if ind[0].size != 0:
                    data[ind] = 0.
                    var[ind] = 0.
                    weight[ind] = 0.
                else:
                    #print "<ddt_sky_guess> no more ind"
                    break
            else:
                break
            niter += 1
                
        sky[i_t] = I
        
    return sky
    
    
def fft_shift_phasor(dimlist, offset, apodize=None, half=None):
    """Return a complex phasor suitable to fine shift an array be OFFSET
    
    Parameters
    ----------
    dimlist : 
    offset : 
    apodize : fn
    half : boolean
    Returns
    -------
    """
    
    ndims = len(dimlist)
    flag = 0 + 0j
    if apodize is None:
        #Computes multi-dimensional un-apodized shift phasor one
        #dimension at a time, starting with the last one. 
        for k in range(ndims)[::-1]:
            n = dimlist[k - 1]
            u = np.arange(n/2+1) # positive frequencies 
            # TODO: Below, I think it should be (k+1) > 1, but then you get an
            # array that can't be combined appropriately back in core.H
            if (n > 2 and ((k + 1 > 0) or (not half))):
                # append negative frequencies 
                u = np.hstack([u, np.arange(-((n - 1)/2),0)])
            off = offset[k] % n
            if off:
                phase = ((2*np.pi/n)*off)*u
                phasor = np.cos(phase) - 1j*np.sin(phase)
                if (half and (n > 1) and not (n%2)):
                    # For Hermitian array, deal with Nyquist
                    # frequency along dimension of even length. 
                    nyquist = n/2 - 1
                    phasor[nyquist] = np.cos(np.pi*off)
        
                if flag:
                    z = np.outer(z[None,:], phasor) #np.array([z])*phasor;
                else:                    
                    z = phasor
                    flag = 1 + 0j
        
            else:
                if flag:
                    z = np.outer(np.ones(u.size), z)
                else:
                    z = np.ones(u.size)
            
      
    
    elif callable(apodize):
        # Computes multi-dimensional apodized shift phasor one
        # dimension at a time, starting with the last one. 
        for k in range(ndims)[::-1]:
            n = dimlist[k - 1]
            u = np.arange(0, n/2) # positive frequencies 
            if (n > 2 and (k > 1 or not half)):
                # append negative frequencies */
                u.append(np.arange(-((n - 1)/2), -1))
      
            filter = apodize(u, n)
            off = offset[k] % n
            if off:
                phase = ((2*np.pi/n)*off)*u
                phasor = np.cos(phase) - 1j*np.sin(phase)
                if (half and n > 1 and not (n % 2)):
                    # For Hermitian array, deal with Nyquist
                    # frequency along dimension of even length.
                    nyquist = n/2 
                    phasor[nyquist] = np.cos(np.pi*off)
        
                filter *= phasor
      
            if flag:
                z = np.array([z])*filter
            else:
                z = filter
                flag = 1 + 0j
      
    
    else:
        raise ValueError("APODIZE must be nil or a function")
  
    return z

