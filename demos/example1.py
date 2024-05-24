# mpirun -np 4 python3 example1.py

from mpi4py import MPI
from ngsolve import *

import ngs_hypre
from ngsolve.krylovspace import CGSolver



comm = MPI.COMM_WORLD
ngmesh = unit_square.GenerateMesh(maxh=0.1, comm=comm)

for l in range(3):
    ngmesh.Refine()
mesh = Mesh(ngmesh)

comm.Barrier()
printonce ("have mesh")

fes = H1(mesh, order=1, dirichlet=".*")
u,v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(1*v*dx).Assemble()

pre = Preconditioner(a, "hypre")
pre.Update()

gfu = GridFunction(fes)

inv = CGSolver(a.mat, pre, printrates=comm.rank==0)
gfu.vec.data = inv * f.vec

printonce("Done")

