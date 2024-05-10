#include <comp.hpp>
#include <python_comp.hpp>

#include "hypre_precond.hpp"



PYBIND11_MODULE(ngs_hypre, m) {
  cout << "Loading ngsHypre preconditioner" << endl;
}
