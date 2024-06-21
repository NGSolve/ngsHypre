# ngsHypre

This project provides an interface to the MPI-parallel solver package [Hypre](https://hypre.readthedocs.io/en/latest/)

**NGS-User:** You are installing regular releases of NGSolve 

    pip3 install git+https://github.com/NGSolve/ngsHypre.git
    mpirun -np 4  python3 -m ngs_hypre.demos.example1

**NGS-Developer:** You are using pre-releases, or compile NGSolve yourself:

    python -m pip install scikit-build-core pybind11_stubgen toml
    pip3 install --no-build-isolation git+https://github.com/NGSolve/ngsHypre.git
    mpirun -np 4  python3 -m ngs_hypre.demos.example1
    
