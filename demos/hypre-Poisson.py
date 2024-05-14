# mpirun -np 4 python3 hypre-Poisson.py

from ngsolve import *
from mpi4py import MPI
from time import time
import ngs_hypre
from ngsolve.krylovspace import CGSolver



ts = time()
comm = MPI.COMM_WORLD
ngmesh = unit_cube.GenerateMesh(maxh=0.13, comm=comm)

comm.Barrier()
printonce ("have coarse mesh, t = ", time()-ts)

for l in range(4):
    ngmesh.Refine()
mesh = Mesh(ngmesh)

comm.Barrier()
printonce ("have mesh, t = ", time()-ts)

SetNumThreads(1)
with TaskManager(pajetrace=10**8):
    fes = H1(mesh, order=1, dirichlet=".*")
    printonce ("ndof = ", fes.ndofglobal)
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(1*v*dx).Assemble()

    comm.Barrier()
    printonce ("have matrix, t = ", time()-ts)

    pre = Preconditioner(a, "hypre")
    pre.Update()

    # print ("type pre = ", type(pre))

    comm.Barrier()
    printonce ("have pre   , t = ", time()-ts)

    gfu = GridFunction(fes)

    inv = CGSolver(a.mat, pre, printrates=comm.rank==0)
    gfu.vec.data = inv * f.vec

    comm.Barrier()
    printonce ("have sol   , t = ", time()-ts)


printonce("Done")

