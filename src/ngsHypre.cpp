#include <comp.hpp>
#include <python_comp.hpp>

#include "hypre_precond.hpp"



PYBIND11_MODULE(ngsHypre, m) {
  cout << "Loading ngsHypre preconditioner" << endl;
}
