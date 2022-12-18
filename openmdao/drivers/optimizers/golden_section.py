import numpy as np


def golden_section(fhandle,a,b,xtol=1E-6,n_iter=None,M_bracket=None):
    ## Constants
    golden_ratio     = (np.sqrt(5)-1)/2
    ## Read Input and Specifiy Algorithm settings
    x_bracket       = [a,b]   
    x_bracket_length = np.abs(b-a)
    if not n_iter:
        n_iter = int(np.ceil(np.log(xtol / x_bracket_length) / np.log(golden_ratio)))
    if not M_bracket:
        M_bracket = [fhandle(a) , fhandle(b)]
        n_eval =  2
    else:
        if M_bracket=='omit-boundaries':
            M_bracket = [np.nan ,np.nan]
        n_eval =  0
    ## Loop through steps
    # Generate solution on both fractions to choose initial direction
    x2   = x_bracket[0] + (1-golden_ratio)*(x_bracket[1]-x_bracket[0])
    M2   = fhandle(x2) 
    x3   = x_bracket[0] + golden_ratio*(x_bracket[1]-x_bracket[0])
    M3   = fhandle(x3) 
    n_eval = n_eval + 2
    # Iterate to find best triplet until bracketing distance is lower than xtol
    for ii in range(n_iter):
        if M2>M3: # chose minimum from triplets M2 M3 M4 and compute next M3
            x_bracket[0] = x2
            M_bracket[0]= M2
            triplet  = [x_bracket[0] ,x3 ,x_bracket[1]]
            tripletM = [M_bracket[0] ,M3 ,M_bracket[1]]
            if ii <n_iter-1:
                x2  = x3
                M2  = M3
                x3  = x_bracket[0] + golden_ratio*(x_bracket[1]-x_bracket[0])
                M3  = fhandle(x3) 
                n_eval=n_eval+1
            
        else: # chose minimum from triplets M1 M2 M3 and compute next M2
            x_bracket[1] = x3
            M_bracket[1]= M3
            triplet  = [x_bracket[0] ,x2 ,x_bracket[1]]
            tripletM = [M_bracket[0] ,M2 ,M_bracket[1]]
            if ii < n_iter-1:
                x3  = x2
                M3  = M2
                x2  = x_bracket[0] + (1-golden_ratio)*(x_bracket[1]-x_bracket[0])
                M2  = fhandle(x2) 
                n_eval=n_eval+1
    
    # [fval,indAux] =   min(tripletM)
    fval =   min(tripletM)
    indAux = tripletM.index(fval)
    x = triplet[indAux]

    ## Additonal output
    debug={'n_iter':n_iter,
           'n_eval':n_eval,
           'x_bracket_length':x_bracket[1]-x_bracket[0]} 
    return x , fval , debug