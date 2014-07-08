import numpy as np


def invert_var(var, job=None, ieee_badval=None):

    """
    DDT Comments:
    * DOCUMENT weight = ddt_invert_var(variance, job=)
    * inverts variance so that it does not crash if variance has 0 points
    * and so that weight is 0 where variance is <=0
    * If job = 1 sets where variance is <=0 to -1
    """
    tmp_var = var
    i_bad = np.where(tmp_var<=0)
    
    if i_bad[0].size != 0:
        tmp_var[i_bad] = 1.    
  
    tmp_var = 1./tmp_var
    """
    # I don't understand this ieee thing and it is probably unnecessary.
    if i_bad.size != 0:
        if job :
            if ieee_badval:
                ieee_var = np.zeros_like(tmp_var)
                ieee_var[i_bad] = ieee_badval
                ieee_set(tmp_var, ieee_var) # Can't find ieee_set                       
            else:
                tmp_var[i_bad] = -1.
      
        else:
            if ieee_badval:
                print "<ddt_invert_var> WARNING: job was not set, ieee_badval ignored";
      
            tmp_var[i_bad] = 0.
    """
    return tmp_var
