# mpirun -np 4 python3 hypre-Poisson-p2.py

from ngsolve import *
from mpi4py import MPI
from time import time
import ngs_hypre
from ngsolve.krylovspace import CGSolver



ts = time()
comm = MPI.COMM_WORLD
ngmesh = unit_cube.GenerateMesh(maxh=0.1, comm=comm)

comm.Barrier()
printonce ("have coarse mesh, t = ", time()-ts)

for l in range(0):
    ngmesh.Refine()
mesh = Mesh(ngmesh)

comm.Barrier()
printonce ("have mesh, t = ", time()-ts)

SetNumThreads(1)
with TaskManager(pajetrace=10**8):
    fes = H1(mesh, order=2, dirichlet=".*")
    printonce ("ndof = ", fes.ndofglobal)
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(1*v*dx).Assemble()

    comm.Barrier()
    printonce ("have matrix, t = ", time()-ts)

    gfu1 = GridFunction(fes)
    gfu1.Set(1)
    
    # pre = Preconditioner(a, "hypre", flags = { "nullspace" : gfu1.vec, "someval" : 3.4 })
    pre = ngs_hypre.HyprePreconditioner(a.mat, fes.FreeDofs(), gfu1.vecs)

    # print ("type pre = ", type(pre))

    comm.Barrier()
    printonce ("have pre   , t = ", time()-ts)

    gfu = GridFunction(fes)

    inv = CGSolver(a.mat, pre, printrates=comm.rank==0)
    gfu.vec.data = inv * f.vec

    comm.Barrier()
    printonce ("have sol   , t = ", time()-ts)


printonce("Done")

