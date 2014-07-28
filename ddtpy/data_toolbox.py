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