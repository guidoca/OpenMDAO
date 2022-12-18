# from ascensiontoolbox.optimizers.golden_section import golden_section
from openmdao.drivers.optimizers import golden_section
#goldenSectionTest tests goldenSection to verify its correct implementation
xtol = 1e-4

def costFun_convex1D(x):
    # Convex 1 Dimensional cost function with minimum at 2
    return (x - 2)**2
    
# [x,fval,debug] = golden_section(costFun_convex1D,-2,4.5,xtol,M_bracket=[costFun_convex1D(0) ,costFun_convex1D(4)])
[x,fval,debug] = golden_section(costFun_convex1D,-2,4.5,xtol)
print('Function Cost = %0.8e'%(fval))
print('Optimum X     = %0.8e '%(x))
print('Error   X     = %0.8e '%(abs(x-2)))
print('Bracket Distance = %0.8e '%(debug['x_bracket_length']))
print('Number Of Iterations  = %i '%(debug['n_iter']))
print('Number Of Evaluations = %i '%(debug['n_eval']))