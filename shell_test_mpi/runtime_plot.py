#import numpy as np
#from matplotlib import pyplot as plt
#
#
#N = 8
#num_pc = np.arange(N)+1
#width = 0.8
#from matplotlib import cm
#cmap = cm.get_cmap('Blues')
#colors = cmap(np.linspace(0.1,1,4))
#
#fig1 = plt.figure()
#t1 = (40.587,24.961,17.673,15.530,13.744,11.415,11.048,10.521)
#p1 = plt.bar(num_pc, t1, width/2, color=colors[1])
#
#plt.ylabel('Time (s)')
#plt.xlabel('Number of processors')
#plt.title('Time for solve with MUMPS (DOFs=800k)')
#
#fig2 = plt.figure()
#t1 = (181.028,111.392,78.724,77.094,60.612,65.482,57.794,59.092)
#p1 = plt.bar(num_pc, t1, width/2, color=colors[1])
#
#plt.ylabel('Time (s)')
#plt.xlabel('Number of processors')
#plt.title('Time for solve with MUMPS (DOFs=2.3M)')
#plt.show()

from dolfin import *
# Define mesh and subdomain:
mesh = UnitSquareMesh(10,10)
d = mesh.topology().dim()
filmx = CompiledSubDomain("x[1] > 1.0-x[0]-DOLFIN_EPS")

# Refinement using the `refine()` function and a Boolean `MeshFunction`:
r_markers = MeshFunction("bool", mesh, d, False)
filmx.mark(r_markers, True)
refinedMesh = refine(mesh,r_markers)

# Transfering a non-negative integer-valued (`size_t`) `MeshFunction` to the
# refined mesh using the `adapt()` function:
meshFunctionToAdapt = MeshFunction("size_t", mesh, d, 0)
filmx.mark(meshFunctionToAdapt,1)
adaptedMeshFunction = adapt(meshFunctionToAdapt,refinedMesh)

# Plot results:
from matplotlib import pyplot as plt
#plot(adaptedMeshFunction)
plot(adaptedMeshFunction.mesh()) # (Adapted function is on refined mesh)
plt.show()
