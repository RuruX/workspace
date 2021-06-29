"""
The "cable_robot_utils" module
------------------------------
contains functions that can be used perform simulation
of cable driven robot.
"""
from dolfin import *
import os
import numpy as np
from petsc4py import PETSc
from scipy.stats import mode

DOLFIN_FUNCTION = function.function.Function
DOLFIN_VECTOR = cpp.la.Vector
DOLFIN_MATRIX = cpp.la.Matrix
DOLFIN_PETSCVECTOR = cpp.la.PETScVector
DOLFIN_PETSCMATRIX = cpp.la.PETScMatrix
PETSC4PY_VECTOR = PETSc.Vec
PETSC4PY_MATRIX = PETSc.Mat

def v2p(v):
    """
    Convert "dolfin.cpp.la.PETScVector" to 
    "petsc4py.PETSc.Vec".
    """
    return as_backend_type(v).vec()

def m2p(A):
    """
    Convert "dolfin.cpp.la.PETScMatrix" to 
    "petsc4py.PETSc.Mat".
    """
    return as_backend_type(A).mat()

def arg2v(x):
    """
    Convert dolfin Function or dolfin Vector to PETSc.Vec.
    """
    if isinstance(x, DOLFIN_FUNCTION):
        x_PETSc = func2p(x)
    elif isinstance(x, DOLFIN_PETSCVECTOR):
        x_PETSc = v2p(x)
    elif isinstance(x, DOLFIN_VECTOR):
        x_PETSc = v2p(x)
    elif isinstance(x, PETSC4PY_VECTOR):
        x_PETSc = x
    else:
        raise TypeError("Type " + str(type(x)) + " is not supported yet.")
    return x_PETSc

def arg2m(A):
    """
    Convert dolfin Matrix to PETSc.Mat.
    """
    if isinstance(A, DOLFIN_PETSCMATRIX):
        A_PETSc = m2p(A)
    elif isinstance(A, DOLFIN_MATRIX):
        A_PETSc = m2p(A)
    elif isinstance(A, PETSC4PY_MATRIX):
        A_PETSc = A
    else:
        raise TypeError("Type " + str(type(A)) + " is not supported yet.")
    return A_PETSc

def zero_petsc_vec(num_el, comm=MPI.comm_world):
    """
    Create zero PETSc vector of size ``num_el``.

    Parameters
    ----------
    num_el : int
    vec_type : str, optional
        For petsc4py.PETSc.Vec types, see petsc4py.PETSc.Vec.Type.
    comm : MPI communicator

    Returns
    -------
    v : petsc4py.PETSc.Vec
    """
    v = PETSc.Vec().create(comm)
#    v.createSeq(num_el, comm=comm)
    v.setSizes(num_el)
    v.setUp()
    v.assemble()
    return v

def zero_petsc_mat(row, col, comm=MPI.comm_world):
    """
    Create zeros PETSc matrix with shape (``row``, ``col``).

    Parameters
    ----------
    row : int
    col : int
    mat_type : str, optional
        For petsc4py.PETSc.Mat types, see petsc4py.PETSc.Mat.Type
    comm : MPI communicator

    Returns
    -------
    A : petsc4py.PETSc.Mat
    """
    A = PETSc.Mat(comm)
    A.createAIJ([row, col], comm=comm)
    A.setUp()
    A.assemble()
    return A
    
def convert_to_dense(A):
    A_petsc = arg2m(A)
    A_petsc.assemble()
    A_dense = A_petsc.convert("dense")
    return A_dense.getDenseArray()


def A_x(A, x):
    """
    Compute b = A*x.

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec

    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """
    A_PETSc = arg2m(A)
    x_PETSc = arg2v(x)
    b_PETSc = zero_petsc_vec(A_PETSc.size[0], comm=A_PETSc.getComm())
    A_PETSc.mult(x_PETSc, b_PETSc)
    return b_PETSc

def A_x_b(A, x, b):
    """
    Compute "Ax = b".

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    b : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    """
    return m2p(A).mult(v2p(x), v2p(b))

def AT_x(A, x):
    """
    Compute b = A^T*x.

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec

    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """
    A_PETSc = arg2m(A)
    x_PETSc = arg2v(x)
    row, col = A_PETSc.getSizes()
#    print(row, col)
    b_PETSc = zero_petsc_vec(col, comm=A_PETSc.getComm())
    A_PETSc.multTranspose(x_PETSc, b_PETSc)
    return b_PETSc
    
#    row, col = m2p(A).getSizes()
#    y = PETSc.Vec().create()
#    y.setSizes(col)
#    y.setUp()
#    m2p(A).multTranspose(v2p(R.vector()),y)
#    y.assemble()
#    return y.getArray()

def AT_x_b(A, x, b):
    """
    Compute b = A^T*x.

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    b : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    """
    arg2m(A).multTranspose(arg2v(x), arg2v(b))

def AT_R_B(A, R, B):
    """
    Compute "A^T*R*B". A,R and B are "petsc4py.PETSc.Mat".

    Parameters
    -----------
    A : petsc4py.PETSc.Mat
    R : petsc4py.PETSc.Mat
    B : petsc4py.PETSc.Mat

    Returns
    -------
    ATRB : petsc4py.PETSc.Mat
    """
    ATRB = A.transposeMatMult(R).matMult(B)
    return ATRB

def generate_interpolated_data(data, num_pts):
    """
    Given initial data ``data`` and specify the number of points 
    ``num_pts``, return the nearly evenly interpolated data.

    Parameters
    ----------
    data : ndarray
    num_pts : int

    Returns
    -------
    interp_data : ndarray
    """
    if data.ndim == 1:
        data = np.array([data]).transpose()
    rows, cols = data.shape

    if rows > num_pts:
        print("Number of points to interpolate {} is smaller than the number "
              "of given points {}, removing points from data to match the "
              "number of points.".format(num_pts, rows))
        num_remove = rows - num_pts
        remove_ind = np.linspace(1, rows-2, num_remove, dtype=int)
        interp_data = np.delete(data, remove_ind, axis=0)
    
    elif rows == num_pts:
        interp_data = data

    else:
        num_insert = num_pts - rows
        num_interval = rows - 1
        interp_data = np.zeros((num_pts, cols))

        num1 = round(num_insert/num_interval)
        num_insert_element = np.ones(num_interval).astype(int)*int(num1)
        round_num = int(num1*num_interval)
        diff = int(round_num - num_insert)

        if diff > 0:
            for i in range(abs(int(diff))):
                num_insert_element[i] -= 1
        elif diff < 0:
            for i in range(abs(int(diff))):
                num_insert_element[i] += 1

        num_pts_element = num_insert_element + 1

        for i in range(num_interval):
            for j in range(cols):
                if i == num_interval-1:
                    interp_data[np.sum(num_pts_element[0:i]):num_pts, j] \
                    = np.linspace(data[i,j], data[i+1,j], 
                                  num_pts_element[i]+1)[0:]
                else:
                    interp_data[np.sum(num_pts_element[0:i]):np.sum(\
                        num_pts_element[0:i+1]), j] = np.linspace(data[i,j], \
                        data[i+1,j], num_pts_element[i]+1)[0:-1]

    return interp_data

def generate_mortar_mesh(pts=None, num_el=None, data=None, 
                         comm=MPI.comm_world):
    """
    Create topologically 1D, geometrically 1, 2 or 3D mortar mesh with 
    a single row of elements connecting them using given data points.

    Parameters
    ----------
    pts : ndarray or None, optional 
        Locations of nodes of mortar mesh
    num_el : int or None, optional 
        number of elements of mortar mesh
    data : ndarray or None, optional 
        Locations of nodes of mortar mesh. If ``data`` is not given, 
        ``pts`` and ``num_el`` are required.
    comm : mpi4py.MPI.Intarcomm, optional

    Returns
    -------
    mesh : dolfin Mesh
    """
    if data is not None:
        data = data
    else:
        data = generate_interpolated_data(pts, num_el+1)

    MESH_FILE_NAME = generateMeshXMLFileName(comm)

    if MPI.rank(comm) == 0:

        if data.ndim == 1:
            data = np.array([data]).transpose()
        rows, cols = data.shape

        dim = cols
        nverts = rows
        nel = nverts - 1

        fs = '<?xml version="1.0" encoding="UTF-8"?>' + "\n"
        fs += '<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">' + "\n"
        fs += '<mesh celltype="interval" dim="' + str(dim) + '">' + "\n"

        fs += '<vertices size="' + str(nverts) + '">' + "\n"
        if dim == 1:
            for i in range(nverts):
                x0 = repr(data[i,0])
                fs += '<vertex index="' + str(i) + '" x="' + x0 + '"/>' + "\n"
            fs += '</vertices>' + "\n"

        elif dim == 2:
            for i in range(nverts):
                x0 = repr(data[i,0])
                y0 = repr(data[i,1])
                fs += '<vertex index="' + str(i) + '" x="' + x0 \
                   + '" y="' + y0 + '"/>' + "\n"
            fs += '</vertices>' + "\n"

        elif dim == 3:
            for i in range(nverts):
                x0 = repr(data[i,0])
                y0 = repr(data[i,1])
                z0 = repr(data[i,2])
                fs += '<vertex index="' + str(i) + '" x="' + x0 \
                   + '" y="' + y0 + '" z="' + z0 +'"/>' + "\n"
            fs += '</vertices>' + "\n"

        else:
            raise ValueError("Unsupported parametric"
                " dimension: {}".format(dim))

        fs += '<cells size="' + str(nel) + '">' + "\n"
        for i in range(nel):
            v0 = str(i)
            v1 = str(i+1)
            fs += '<interval index="' + str(i) + '" v0="' + v0 + '" v1="' \
                + v1 + '"/>' + "\n"

        fs += '</cells></mesh></dolfin>'

        f = open(MESH_FILE_NAME,'w')
        f.write(fs)
        f.close()
        
    MPI.barrier(comm)    
    mesh = Mesh(comm, MESH_FILE_NAME)

    if MPI.rank(comm) == 0:
        os.remove(MESH_FILE_NAME)

    return mesh
    
def getNonzeroEntities(M, node):
    nodes, inds = M.getRowIJ()
    ind = inds[nodes[node]:nodes[node+1]]
    return ind
    
# helper function to generate an identity permutation IS
# given an ownership range
def generateIdentityPermutation(ownRange, comm):

    """
    Returns a PETSc index set corresponding to the ownership range.
    """
    iStart = ownRange[0]
    iEnd = ownRange[1]
    localSize = iEnd - iStart
    iArray = np.zeros(localSize,dtype='int32')
    for i in np.arange(0,localSize):
        iArray[i] = i+iStart
    retval = PETSc.IS(comm)
    retval.createGeneral(iArray,comm=comm)
    return retval
    
# override default behavior to order unknowns according to what task's
# FE mesh they overlap.  this will (hopefully) reduce communication
# cost in the matrix--matrix multiplies
def generatePermutation(V, M, totalDofs):
    """V: Foreground Function Space"""
    """
    Generates a permutation of the IGA degrees of freedom that tries to
    ensure overlap of their parallel partitioning with that of the FE
    degrees of freedom, which are partitioned automatically based on the
    FE mesh.
    """
    worldcomm = MPI.comm_world
    rank = MPI.rank(worldcomm)
    mesh = V.mesh()
    func = Function(V)
    Istart, Iend = as_backend_type(func.vector()).vec().getOwnershipRange()
    nLocalNodes = Iend - Istart

    totalDofs = totalDofs
    DEFAULT_PREALLOC = 500
    MPETSc = PETSc.Mat(comm=worldcomm)
    MPETSc.createAIJ([[nLocalNodes,None],[None,totalDofs]],comm=worldcomm)
    MPETSc.setPreallocationNNZ([DEFAULT_PREALLOC,
                                DEFAULT_PREALLOC])
    MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    MPETSc.setUp()

    offset = 0
    x_nodes = V.tabulate_dof_coordinates()\
                    .reshape((-1,mesh.geometry().dim()))

    dofs = V.dofmap().dofs()
    for I in np.arange(0,len(dofs)):
        x = x_nodes[dofs[I]-Istart]
        matRow = dofs[I]
        print('matRow',matRow)
        nodesAndEvals = getNonzeroEntities(M, matRow)
#        nodesAndEvals = self.getNodesAndEvals(x,field)

        cols = np.array(nodesAndEvals,dtype='int32')[:] + offset
        rows = np.array([matRow,],dtype='int32')
        values = np.full((1,len(nodesAndEvals)),rank+1)
        MPETSc.setValues(rows,cols,values,addv=PETSc.InsertMode.INSERT)
        
#        for i in range(0,len(nodesAndEvals)):
#            MPETSc[matRow,nodesAndEvals[i]+offset]\
#                = rank+1 # need to avoid losing zeros...


    MPETSc.assemblyBegin()
    MPETSc.assemblyEnd()
#    print(convert_to_dense(MPETSc))
    MT = MPETSc.transpose(PETSc.Mat(comm=worldcomm))
    Istart, Iend = MT.getOwnershipRange()
    nLocal = Iend - Istart
    partitionInts = np.zeros(nLocal,dtype='int32')
    for i in np.arange(Istart,Iend):
        rowValues = MT.getRow(i)[0]
        # isolate nonzero entries
        rowValues = np.extract(rowValues>0,rowValues)
        iLocal = i - Istart
        modeValues = mode(rowValues)[0]
        if(len(modeValues) > 0):
            partitionInts[iLocal] = int(mode(rowValues).mode[0]-0.5)
        else:
            partitionInts[iLocal] = 0 # necessary?
    partitionIS = PETSc.IS(comm=worldcomm)
    partitionIS.createGeneral(partitionInts,comm=worldcomm)

    # kludgy, non-scalable solution:
    
    # all-gather the partition indices and apply argsort to their
    # underlying arrays
    bigIndices = np.argsort(partitionIS.allGather().getIndices())\
                 .astype('int32')

    # note: index set sort method only sorts locally on each processor

    # note: output of argsort is what we want for MatPermute(); it
    # maps from indices in the sorted array, to indices in the original
    # unsorted array.
    
    # use slices [Istart:Iend] of the result from argsort to create
    # a new IS that can be used as a petsc ordering
    retval = PETSc.IS(comm=worldcomm)
    retval.createGeneral(bigIndices[Istart:Iend],comm=worldcomm)
    
    return retval
    

def applyPermutation(M, permutation):
    """
    Permutes the order of the IGA degrees of freedom, so that their
    parallel partitioning better aligns with that of the FE degrees
    of freedom, which is generated by standard mesh-partitioning
    approaches in FEniCS.
    """
    worldcomm = MPI.comm_world
    rank = MPI.rank(worldcomm)
    if(MPI.size(worldcomm) > 1):
#        print(rank, M.mat().getSizes())
#        print(rank, permutation.getSizes())
        permutation = permutation
        newM = M.mat().permute\
                 (generateIdentityPermutation\
                  (M.mat().getOwnershipRange(),worldcomm),\
                  permutation)
        M = PETScMatrix(newM)

        # fix list of zero DOFs
#        self.permutationAO = PETSc.AO(self.comm)
#        self.permutationAO\
#            .createBasic(self.permutation,\
#                         generateIdentityPermutation\
#                         (self.M.mat().getOwnershipRangeColumn(),self.comm))
#        zeroDofIS = PETSc.IS(self.comm)
#        zeroDofIS.createGeneral(array(self.zeroDofs,dtype=INDEX_TYPE))
#        self.zeroDofs = self.permutationAO.app2petsc\
#                        (zeroDofIS).getIndices()
    return M
        

