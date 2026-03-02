#include "LILS.hpp"

//      We want to create a linear implicic solve for linear problems. 
//      For this problem we will use the form:
//
//                             M du/dt = - K u 
//
//      where M is the mass matrix and K is the stiffness matrix. This 
//      solve can be approximated by a backward difference becoming:
//
//                         (M+dt*K)v^n+1 = M*v^n
//
//      We will take the mass and stiffness matricies from the caller 
//      and use a Conjugate Gradient solver with a Gauss-Seidel
//      preconditioner to solve the system.
LinearImplicitLinearSolve::LinearImplicitLinearSolve(mfem::SparseMatrix &M,
                                                     mfem::SparseMatrix &K,
                                                     mfem::real_t dt)
   : M_(M),               // Mass matrix
     K_(K),               // Stiffness matrix
     dt_(dt),             // Time step size
     T_(nullptr),         // System matrix (M + dt*K)
     rhs_(M.Height()),    // Right-hand side vector
     lin_solver_()        // Linear solver
{
   BuildSystemMatrix();
   ConfigureLinearSolver();
}

void LinearImplicitLinearSolve::SetTimeStep(const mfem::real_t dt)
{
   dt_ = dt;
   BuildSystemMatrix();
   ConfigureLinearSolver();
}

mfem::real_t LinearImplicitLinearSolve::GetTimeStep() const
{
   return dt_;
}

void LinearImplicitLinearSolve::Step(mfem::Vector &u_current,
                                     mfem::Vector &u_next)
{
   M_.Mult(u_current, rhs_);
   lin_solver_.Mult(rhs_, u_next);

   mfem::Vector Au(rhs_.Size());
   T_->Mult(u_next, Au);
   Au -= rhs_;
   const mfem::real_t rhs_n = rhs_.Norml2();
   const mfem::real_t rel_res = Au.Norml2() / (rhs_n > 0.0 ? rhs_n : 1.0);
   MFEM_VERIFY(rel_res == rel_res, "Linear solve residual is NaN.");
}

void LinearImplicitLinearSolve::Step(mfem::Vector &u_current,
                                     const mfem::Vector &source,
                                     mfem::Vector &u_next)
{
   M_.Mult(u_current, rhs_);
   rhs_.Add(dt_, source);
   lin_solver_.Mult(rhs_, u_next);


   mfem::Vector Au(rhs_.Size());
   T_->Mult(u_next, Au);
   Au -= rhs_;
   const mfem::real_t rhs_n = rhs_.Norml2();
   const mfem::real_t rel_res = Au.Norml2() / (rhs_n > 0.0 ? rhs_n : 1.0);
   MFEM_VERIFY(rel_res == rel_res, "Linear solve residual is NaN.");
}

void LinearImplicitLinearSolve::BuildSystemMatrix()
{
   T_.reset(mfem::Add(1.0, M_, dt_, K_));
}

void LinearImplicitLinearSolve::ConfigureLinearSolver()
{
   auto new_prec = std::make_unique<mfem::DSmoother>(*T_);
   lin_solver_.SetOperator(*T_);
   lin_solver_.SetPreconditioner(*new_prec);
   A_prec_ = std::move(new_prec);
   lin_solver_.SetRelTol(1e-8);
   lin_solver_.SetAbsTol(0.0);
   lin_solver_.SetMaxIter(1000);
   lin_solver_.SetPrintLevel(0);
}

void LinearImplicitLinearSolve::UpdateStiffness(mfem::SparseMatrix &K)
{
   (void)K;
   BuildSystemMatrix();

   auto new_prec = std::make_unique<mfem::DSmoother>(*T_);
   lin_solver_.SetOperator(*T_);
   lin_solver_.SetPreconditioner(*new_prec);
   A_prec_ = std::move(new_prec);
}
