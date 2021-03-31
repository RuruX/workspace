import math
import matplotlib.pyplot as plt
import numpy as np 
from dolfin import *
import ufl
import tIGAr
from tIGAr import *
from tIGAr.BSplines import *
from tIGAr.compatibleSplines import *
from scipy.sparse import csr_matrix
import importlib

class set_IGA(object):

	def __init__(self, num_elements):
		# importlib.reload(tIGAr)
		# print(tIGAr.__file__)
		# print('Run __init__ in set_IGA ---------------')
		self.num_elements = num_elements
		self.L = 1.
		self.p = 2
		self.quad_deg = 2*self.p
		
		self.p_vec = [self.p,self.p]# order
		self.k_vec = [uniformKnots(self.p,0.0,self.L,self.num_elements),uniformKnots(self.p,0.0,self.L,self.num_elements),]# knot should be two for shell
		self.spline_mesh = ExplicitBSplineControlMesh(self.p_vec, self.k_vec)

		self.spline_generator = EqualOrderSpline(1, self.spline_mesh)
		self.set_bc(self.spline_generator)# drishlet bc

		self.spline = ExtractedSpline(self.spline_generator, self.quad_deg)
		self.spline.setSolverOptions(
			# maxIters=100,
			# relativeTolerance=1e-5,
		)
		self.u = Function(self.spline.V)
		self.v = TestFunction(self.spline.V)
		_, self.num_dof = as_backend_type(self.spline.M).mat().getSize()

		self.VT = FunctionSpace(self.spline.mesh, 'DG', 0)
		self.num_var = self.VT.dim()
		self.t = Function(self.VT)

		self.x = self.spline.spatialCoordinates()
		# self.leftChar = 1.0 - self.x[0]
		self.rightChar = conditional(gt(self.x[0],1-DOLFIN_EPS),1,Constant(0))
		self.force = self.rightChar*Constant(-1.)

	def pdeRes(self, u, v, t): # inner then inner, no inner then dot
		# print('Run pdeRes() ----------------------')
		EI = t**3
		# penalty0 = 1e9
		# penalty1 = 1e9
		return inner(EI*self.lap(u), self.lap(v))*self.spline.dx \
			 - inner(self.force, v)*self.spline.ds \
			 # + self.leftChar*(penalty0*self.u*self.v \
			 # + penalty1*inner(grad(self.u), grad(self.v)))*self.spline.ds 

	def compute_derivative(self, u, v, t):
		# print('Run compute_derivative() in set_IGA -------------------')
		dR_du = derivative(self.pdeRes(u, v, t), u)
		dR_du_matrix = self.spline.assembleMatrix(dR_du)

		dR_dt = derivative(self.pdeRes(u, v, t), t)
		dR_dt_matrix_ = as_backend_type(assemble(dR_dt)).mat()

		dR_dt_matrix = PETScMatrix(as_backend_type(self.spline.M).mat().transposeMatMult(dR_dt_matrix_))

		dR_du_sparse = as_backend_type(dR_du_matrix).mat()
		dR_dt_sparse = as_backend_type(dR_dt_matrix).mat()
		
		dR_du_csr = csr_matrix(dR_du_sparse.getValuesCSR()[::-1], shape=dR_du_sparse.size)
		dR_dt_csr = csr_matrix(dR_dt_sparse.getValuesCSR()[::-1], shape=dR_dt_sparse.size)

		return dR_du_csr.tocoo(), dR_dt_csr.tocoo()

	def lap(self, u):
		return self.spline.div(self.spline.grad(u))

	def set_bc(self, spline_generator):
		field = 0
		parametric_direction = 0
		side = 0
		scalar_spline = spline_generator.getScalarSpline(field)
		side_dofs = scalar_spline.getSideDofs(parametric_direction,side,nLayers=2)
		spline_generator.addZeroDofs(field,side_dofs)
		
		field = 0
		parametric_direction = 1
		for side in [0,1]:
			side_dofs = scalar_spline.getSideDofs(parametric_direction,side,nLayers=2)
			spline_generator.addZeroDofs(field,side_dofs)

	def array2petsc(self, array):
		size = len(array)
		A = PETSc.Mat().create()
		A.setSizes([1, size])
		A.setType('aij')
		A.setUp()
		A.setValues([0], [range(size)], array)
		A.assemble()
		return A.convert("dense")

	def iga2feDoFs(self, array):# iga to fe tranformation
		petsc_iga = self.array2petsc(array)
		petsc_fe = as_backend_type(self.spline.M).mat().matMult(petsc_iga.transpose()).transpose()

		return petsc_fe.getDenseArray().flatten('C')



if __name__ == '__main__':

	iga_test = set_IGA(num_elements=5)
	pde_res = assemble(iga_test.pdeRes(iga_test.u, iga_test.v, iga_test.t)).get_local()
	iga_test.compute_derivative(iga_test.u, iga_test.v, iga_test.t)
















