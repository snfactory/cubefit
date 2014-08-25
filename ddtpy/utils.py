"""General numeric utilities."""

import numpy as np
    
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
    if len(offset) > ndims:
        raise ValueError("too many offsets")
    elif len(offset) < ndims:
        offset = offset + [0]*(ndims-len(offset))
    print offset
    flag = 0 + 0j
    if apodize is None:
        #Computes multi-dimensional un-apodized shift phasor one
        #dimension at a time, starting with the last one. 
        for k in range(ndims):
            n = dimlist[k]
            print k, n
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
            print z.shape
      
    
    elif callable(apodize):
        # Computes multi-dimensional apodized shift phasor one
        # dimension at a time, starting with the last one. 
        for k in range(ndims):
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

