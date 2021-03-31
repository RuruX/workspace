#Final Project 3 - Cantilever Plate Bending
from dolfin import *
from mshr import *
from ufl import indices, Jacobian, shape
from petsc4py import PETSc

def m2p(A):
    return as_backend_type(A).mat()

def v2p(v):
    return as_backend_type(v).vec()
    
# Problem parameters:

E = Constant(4.32e8) # Young's modulus
nu = Constant(0.0) # Poisson ratio
h = Constant(0.2) # Shell thickness
length = 20.
width = 2.
rho_g = -100.
f_d = rho_g*h**3
f = Constant((0,0,f_d)) # Body force per unit volume


# Get manifold mesh with geometric dimension 3 and topological dimension 2:

mesh = Mesh()
filename = "plate3.xdmf"
file = XDMFFile(mesh.mpi_comm(),filename)
file.read(mesh)

#     Use the `refine` function on it to split every triangle into four smaller triangles
level = 3
for i in range(0,level):
    mesh = refine(mesh)


# Set up function space, with the first vector element being
# mid-surface displacement, and the second vector element being linearized
# rotation.
cell = mesh.ufl_cell()
VE = VectorElement("Lagrange",cell,1)
WE = MixedElement([VE,VE])
W = FunctionSpace(mesh,WE)
VT = FunctionSpace(mesh,'DG',0)
# Reduced one-point quadrature:
dx_shear = dx(metadata={"quadrature_degree":0})

# Solution function:
w = Function(W)
u_mid,theta = split(w)

# Reference configuration of midsurface:
X_mid = SpatialCoordinate(mesh)

# Normal vector to each element is the third basis vector of the
# local orthonormal basis (indexed from zero for consistency with Python):
E2 = CellNormal(mesh)

# Local in-plane orthogonal basis vectors, with 0-th basis vector along
# 0-th parametric coordinate direction (where Jacobian[i,j] is the partial
# derivatiave of the i-th physical coordinate w.r.t. to j-th parametric
# coordinate):
A0 = as_vector([Jacobian(mesh)[j,0] for j in range(0,3)])
E0 = A0/sqrt(dot(A0,A0))
E1 = cross(E2,E0)

# Matrix for change-of-basis to/from local/global Cartesian coordinates;
# E01[i,j] is the j-th component of the i-th basis vector:
E01 = as_matrix([[E0[i] for i in range(0,3)],
                 [E1[i] for i in range(0,3)]])

# Displacement at through-thickness coordinate xi2:
def u(xi2):
    # Formula (7.1) from http://www2.nsysu.edu.tw/csmlab/fem/dyna3d/theory.pdf
    return u_mid - xi2*cross(E2,theta)

# In-plane gradient components of displacement in the local orthogonal
# coordinate system:
def gradu_local(xi2):
    gradu_global = grad(u(xi2)) # (3x3 matrix, zero along E2 direction)
    i,j,k,l = indices(4)
    return as_tensor(E01[i,k]*gradu_global[k,l]*E01[j,l],(i,j))

# In-plane strain components of local orthogonal coordinate system at
# through-thickness coordinate xi2, in Voigt notation:
def eps(xi2):
    eps_mat = sym(gradu_local(xi2))
    return as_vector([eps_mat[0,0], eps_mat[1,1], 2*eps_mat[0,1]])

# Transverse shear strains in local coordinates at given xi2, as a vector
# such that gamma_2(xi2)[i] = 2*eps[i,2], for i in {0,1}
def gamma_2(xi2):
    dudxi2_global = -cross(E2,theta)
    i,j = indices(2)
    dudxi2_local = as_tensor(dudxi2_global[j]*E01[i,j],(i,))
    gradu2_local = as_tensor(dot(E2,grad(u(xi2)))[j]*E01[i,j],(i,))
    return dudxi2_local + gradu2_local

# Voigt notation material stiffness matrix for plane stress:
D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                 [nu,   1.0,  0.0         ],
                                 [0.0,  0.0,  0.5*(1.0-nu)]])

# Elastic energy per unit volume at through-thickness coordinate xi2:
def energyPerUnitVolume(xi2):
    G = E/(2*(1+nu))
    return 0.5*(dot(eps(xi2),D*eps(xi2))
                + G*inner(gamma_2(xi2),gamma_2(xi2))) # ?!?!?

# Some code copy-pasted from tIGAr and ShNAPr for Gaussian quadrature through
# the thickness of the shell structure:
def getQuadRule(n):
    """
    Return a list of points and a list of weights for integration over the
    interval (-1,1), using ``n`` quadrature points.  
    """
    if(n==1):
        xi = [Constant(0.0),]
        w = [Constant(2.0),]
        return (xi,w)
    if(n==2):
        xi = [Constant(-0.5773502691896257645091488),
              Constant(0.5773502691896257645091488)]
        w = [Constant(1.0),
             Constant(1.0)]
        return (xi,w)
    if(n==3):
        xi = [Constant(-0.77459666924148337703585308),
              Constant(0.0),
              Constant(0.77459666924148337703585308)]
        w = [Constant(0.55555555555555555555555556),
             Constant(0.88888888888888888888888889),
             Constant(0.55555555555555555555555556)]
        return (xi,w)
    if(n==4):
        xi = [Constant(-0.86113631159405257524),
              Constant(-0.33998104358485626481),
              Constant(0.33998104358485626481),
              Constant(0.86113631159405257524)]
        w = [Constant(0.34785484513745385736),
             Constant(0.65214515486254614264),
             Constant(0.65214515486254614264),
             Constant(0.34785484513745385736)]
        return (xi,w)
    
    print("ERROR: invalid number of quadrature points requested.")
    exit()

def getQuadRuleInterval(n,L):
    """
    Returns an ``n``-point quadrature rule for the interval 
    (-``L``/2,``L``/2), consisting of a list of points and list of weights.
    """
    xi_hat, w_hat = getQuadRule(n)
    xi = []
    w = []
    for i in range(0,n):
        xi += [L*xi_hat[i]/2.0,]
        w += [L*w_hat[i]/2.0,]
    return (xi,w)

class ThroughThicknessMeasure:
    """
    Class to represent a local integration through the thickness of a shell.
    The ``__rmul__`` method is overloaded for an instance ``dxi2`` to be
    used like ``volumeIntegral = volumeIntegrand*dxi2*dx``, where
    ``volumeIntegrand`` is a python function taking a single parameter,
    ``xi2``.
    """
    def __init__(self,nPoints,h):
        """
        Integration uses a quadrature rule with ``nPoints`` points, and assumes
        a thickness ``h``.
        """
        self.nPoints = nPoints
        self.h = h
        self.xi2, self.w = getQuadRuleInterval(nPoints,h)

    def __rmul__(self,integrand):
        """
        Given an ``integrand`` that is a Python function taking a single
        ``float`` parameter with a valid range of ``-self.h/2`` to 
        ``self.h/2``, return the (numerical) through-thickness integral.
        """
        integral = 0.0
        for i in range(0,self.nPoints):
            integral += integrand(self.xi2[i])*self.w[i]
        return integral

# Integrate through the thickness numerically to get total energy (where dx
# is a UFL Measure, integrating over midsurface area).
dxi2 = ThroughThicknessMeasure(3,h)
elasticEnergy = energyPerUnitVolume*dxi2*dx_shear

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 20.0)

class Left(SubDomain):
   def inside(self, x, on_boundary):
      return near(x[0], 0.0)

right = Right()
left = Left()
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0) 
right.mark(boundaries, 1) 
left.mark(boundaries, 2) 
ds = Measure('ds',subdomain_data=boundaries)

# Take a Gatueax derivative and add a source term to obtain the
# weak form of the problem:
dw = TestFunction(W)
du_mid,dtheta = split(dw)
n = CellNormal(mesh)
F = derivative(elasticEnergy,w,dw) - inner(f,du_mid)*dx\
    + DOLFIN_EPS*E*h*dot(theta,n)*dot(dtheta,n)*dx
    
# LHS of linearized problem J == -F:
J = derivative(F,w)

# Set up test problem w/ BCs that fix all DoFs to zero for all nodes
# with x[0] negative.
##
#rightChar = conditional(gt(X_mid[0],1-DOLFIN_EPS),1,Constant(0))
#print("objective:", 
#      assemble(rightChar*Constant(0.5)*dot(u_mid,u_mid)*ds))
bcs = [DirichletBC(W,Constant(6*(0,)),boundaries, 2),]
solve(J==-F,w,bcs,solver_parameters={"linear_solver":"mumps"})


#solve(F == 0, w, bcs, J=J, 
#        solver_parameters={'newton_solver': {
#        "maximum_iterations":1, 
#        "error_on_nonconvergence":False,
#        'linear_solver': 'mumps'}})
#        
        
print(assemble(inner(w,w)*dx))
print(assemble(dot(u_mid,u_mid)*dx))
print(assemble(elasticEnergy))
# Output:
u_mid,theta = w.split(True)
u_mid.rename("u","u")
theta.rename("t","t")
File("u-1.pvd") << u_mid
File("t-1.pvd") << theta

print('number of elements:', len(VT.dofmap().dofs()))
# The solution is compared to the Kirchhoff analytical solution::S
Ix = width*h**3/12
print("Euler-Beinoulli Beam theory deflection:", float(f_d*length**4/(8*E*Ix)))
print("Reissner-Mindlin FE deflection:", min(w.sub(0).vector().get_local())) 

