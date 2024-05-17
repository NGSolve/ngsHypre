#include <comp.hpp>
#include <python_comp.hpp>

#include "hypre_precond.hpp"



PYBIND11_MODULE(ngs_hypre, m) {
  cout << IM(1) << "Loading ngsHypre preconditioner" << endl;
  using namespace ngcomp;

  auto prec_hypre = py::class_<HyprePreconditioner, shared_ptr<HyprePreconditioner>, Preconditioner>
    (m,"HyprePreconditioner");
  prec_hypre
    .def(py::init([prec_hypre](shared_ptr<BilinearForm> bfa, py::kwargs kwargs)
    {
      auto flags = CreateFlagsFromKwArgs(kwargs); // , prec_hypre);
      auto hyprepre = make_shared<HyprePreconditioner>(bfa,flags);
      // if(lo_precond.has_value())
      //mgpre->SetCoarsePreconditioner(lo_precond.value());
      return hyprepre;
    }), py::arg("bf")
      )

    .def(py::init([prec_hypre](shared_ptr<BaseMatrix>  mat, shared_ptr<BitArray> freedofs,
                               shared_ptr<MultiVector> nullspace)
    {
      auto hyprepre = make_shared<HyprePreconditioner>(*mat.get(), freedofs, nullspace);
      return hyprepre;
    }), py::arg("mat"), py::arg("freedofs"), py::arg("nullspace")
         )


    
    .def("SetNullSpace", [](shared_ptr<HyprePreconditioner> pre, shared_ptr<MultiVector> mv)
    {
      pre -> SetNullSpace (mv);
    })


    
    ;
}
