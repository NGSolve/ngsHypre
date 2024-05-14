/*********************************************************************/
/* File:   hypre_precond.cpp                                         */
/* Author: Martin Huber, Joachim Schoeberl                           */
/* Date:   July 2012                                                 */
/*********************************************************************/



#include <solve.hpp>
#include "hypre_precond.hpp"
#include <core/ng_mpi_native.hpp>

namespace ngcomp
{



  /*
  HyprePreconditioner :: HyprePreconditioner (const PDE & pde, const Flags & aflags, const string & aname)  
    : Preconditioner (&pde, aflags)
  {
    bfa = pde.GetBilinearForm (flags.GetStringFlag ("bilinearform", NULL));
  }
  */
  

  HyprePreconditioner :: HyprePreconditioner (const BaseMatrix & matrix, const shared_ptr<BitArray> afreedofs)
    : Preconditioner(shared_ptr<BilinearForm>(nullptr), Flags("not_register_for_auto_update"))
  {
    freedofs = afreedofs; 
    Setup (matrix);
  }
  
  HyprePreconditioner :: HyprePreconditioner (shared_ptr<BilinearForm> abfa, const Flags & aflags,
					      const string aname)
    : Preconditioner(abfa, aflags, aname)
  {
    bfa = abfa;
  }


  HyprePreconditioner :: ~HyprePreconditioner ()
  {
    ;
  }
  
  
  
  void HyprePreconditioner :: Update()
  {
    freedofs = bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal());
    Setup (bfa->GetMatrix());
  }

  void HyprePreconditioner :: FinalizeLevel (const BaseMatrix * mat)
  {
    freedofs = bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal());
    Setup (*mat);
  }
  


  void HyprePreconditioner :: Setup (const BaseMatrix & matrix)
  {
    cout << IM(1) << "Setup Hypre preconditioner" << endl;
    static Timer t("hypre setup");
    RegionTimer reg(t);


    const ParallelMatrix & pmat = (dynamic_cast<const ParallelMatrix&> (matrix));
    const SparseMatrix<double> & mat = dynamic_cast<const SparseMatrix<double>&>(*pmat.GetMatrix());
    if (dynamic_cast< const SparseMatrixSymmetric<double> *> (&mat))
      throw Exception ("Please use fully stored sparse matrix for hypre (bf -nonsymmetric)");

    pardofs = pmat.GetParallelDofs ();
    NgMPI_Comm comm = pardofs->GetCommunicator();
    int ndof = pardofs->GetNDofLocal();

    global_nums.SetSize(ndof);
    global_nums = -1;
    int num_master_dofs = 0;
    for (int i = 0; i < ndof; i++)
      if (pardofs->IsMasterDof(i) && (!freedofs || freedofs->Test(i)))
	global_nums[i] = num_master_dofs++;
    
    Array<int> first_master_dof(comm.Size());
    comm.AllGather (num_master_dofs, first_master_dof);

    int num_glob_dofs = 0;
    for (int i = 0; i < first_master_dof.Size(); i++)
      {
	int cur = first_master_dof[i];
	first_master_dof[i] = num_glob_dofs;
	num_glob_dofs += cur;
      }
    first_master_dof.Append(num_glob_dofs);

    int rank = comm.Rank();
    for (int i = 0; i < ndof; i++)
      if (global_nums[i] != -1)
	global_nums[i] += first_master_dof[rank];

    pardofs->ScatterDofData (global_nums);

    used_global.SetSize0();
    used_local.SetSize0();
    used_my_global.SetSize0();
    used_my_local.SetSize0();

    for(auto i : Range(global_nums.Size()))
      if (global_nums[i] != -1)
	{
	  used_global.Append(global_nums[i]);
          used_local.Append(i);
          if (pardofs->IsMasterDof(i))
            {
              used_my_global.Append(global_nums[i]);
              used_my_local.Append(i);
            }
        }

    cout << IM(3) << "num glob dofs = " << num_glob_dofs << endl;
	
    // range of my master dofs ...
    ilower = first_master_dof[rank];
    iupper = first_master_dof[rank+1]-1;
   
    HYPRE_IJMatrixCreate(NG_MPI_Native(comm), ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
   
    Array<int> cols_global;
    Array<double> values_global;

    for (int i = 0; i < mat.Height(); i++)
      {
	int row = global_nums[i];
	if (row == -1) continue;

	FlatArray<int> cols = mat.GetRowIndices(i);
	FlatVector<double> values = mat.GetRowValues(i);

        cols_global.SetSize0();
        values_global.SetSize0();

	for (int j = 0; j < cols.Size(); j++)
	  if (global_nums[cols[j]] != -1)
	    {
	      cols_global.Append (global_nums[cols[j]]);
	      values_global.Append (values[j]);
	    }

	int size = cols_global.Size();
	HYPRE_IJMatrixAddToValues(A, 1, &size, &row, cols_global.Data(), values_global.Data());
      }


    
    HYPRE_IJMatrixAssemble(A);
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    // HYPRE_IJMatrixPrint(A, "IJ.out.A");

    HYPRE_BoomerAMGCreate(&precond);
 
    HYPRE_ParVector par_b = NULL;
    HYPRE_ParVector par_x = NULL;

    HYPRE_BoomerAMGSetPrintLevel(precond, 0);  /* print solve info + parameters */
    HYPRE_BoomerAMGSetCoarsenType(precond, 10); /* Falgout coarsening */
    HYPRE_BoomerAMGSetRelaxType(precond, 6);  // 3 GS, 6 .. sym GS 
    HYPRE_BoomerAMGSetStrongThreshold(precond, 0.5);
    HYPRE_BoomerAMGSetInterpType(precond,6);
    HYPRE_BoomerAMGSetPMaxElmts(precond,4);
    HYPRE_BoomerAMGSetAggNumLevels(precond,1);
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);   /* Sweeeps on each level */
    HYPRE_BoomerAMGSetMaxLevels(precond, 20);  /* maximum number of levels */
    HYPRE_BoomerAMGSetTol(precond, 0.0);      /* conv. tolerance */
    HYPRE_BoomerAMGSetMaxIter(precond,1);


    // with HYPRE_BoomerAMGSetInterpVectors and HYPRE_BoomerAMGSetInterpVecVariant. T
    
    cout << IM(2) << "Call BoomerAMGSetup" << endl;
    HYPRE_BoomerAMGSetup (precond, parcsr_A, par_b, par_x);
  }



  void HyprePreconditioner :: Mult (const BaseVector & f, BaseVector & u) const
  {
    static Timer t("hypre mult");
    RegionTimer reg(t);
    NgMPI_Comm comm = pardofs->GetCommunicator();

    f.Distribute();
    
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;

    HYPRE_IJVectorCreate(NG_MPI_Native(comm), ilower, iupper, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);

    HYPRE_IJVectorCreate(NG_MPI_Native(comm), ilower, iupper, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);
  
    FlatVector<double> ff =f.FVDouble();
    FlatVector<double> fu =u.FVDouble();

    
    Vector<double> free_f(used_local.Size());
    f.GetIndirect (used_local, free_f);

    Array<int> setzi(iupper+1-ilower);
    for (size_t i = 0; i < iupper+1-ilower; i++)
      setzi[i]=ilower+i;
    
    Vector<double> zeros(iupper+1-ilower);
    zeros = 0.0;

    HYPRE_IJVectorSetValues(b, setzi.Size(), &setzi[0], &zeros[0]);
    HYPRE_IJVectorAddToValues(b, used_global.Size(), used_global.Data(), free_f.Data());
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    HYPRE_IJVectorSetValues(x, setzi.Size(), &setzi[0], &zeros[0]);
    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void **) &par_x);
   

    HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);

    Vector<double> hu(iupper-ilower+1);

    HYPRE_IJVectorGetValues (x, used_my_global.Size(), used_my_global.Data(), hu.Data());

    u = 0.0;
    u.SetParallelStatus(DISTRIBUTED);
    u.SetIndirect(used_my_local, hu);
    
    HYPRE_IJVectorDestroy(x);
    HYPRE_IJVectorDestroy(b);
    
    u.Cumulate();
  }


  static RegisterPreconditioner<HyprePreconditioner> init_hyprepre ("hypre");
}
