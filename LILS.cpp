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
     A_(),                // System matrix (M + dt*K)
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
}

void LinearImplicitLinearSolve::Step(mfem::Vector &u_current,
                                     const mfem::Vector &source,
                                     mfem::Vector &u_next)
{
   M_.Mult(u_current, rhs_);
   rhs_.Add(dt_, source);
   lin_solver_.Mult(rhs_, u_next);
}

void LinearImplicitLinearSolve::BuildSystemMatrix()
{
   A_ = M_;
   A_.Add(dt_, K_);
}

void LinearImplicitLinearSolve::ConfigureLinearSolver()
{
   A_prec_ = std::make_unique<mfem::GSSmoother>(A_);
   lin_solver_.SetOperator(A_);
   lin_solver_.SetPreconditioner(*A_prec_);
   lin_solver_.SetRelTol(1e-12);
   lin_solver_.SetAbsTol(0.0);
   lin_solver_.SetMaxIter(200);
   lin_solver_.SetPrintLevel(0);
}

void LinearImplicitLinearSolve::UpdateStiffness(mfem::SparseMatrix &K)
{
   K_ = K;
   BuildSystemMatrix();

   A_prec_ = std::make_unique<mfem::GSSmoother>(A_);
   lin_solver_.SetOperator(A_);
   lin_solver_.SetPreconditioner(*A_prec_);
}