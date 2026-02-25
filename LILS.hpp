#ifndef LILS_HPP
#define LILS_HPP

#include <memory>
#include "mfem.hpp"

class LinearImplicitLinearSolve
{
public:
   LinearImplicitLinearSolve(mfem::SparseMatrix &M,
                             mfem::SparseMatrix &K,
                             mfem::real_t dt);

   void SetTimeStep(mfem::real_t dt);
   mfem::real_t GetTimeStep() const;

   // Solve (M + dt*K) u_next = M u_current.
   void Step(mfem::Vector &u_current, mfem::Vector &u_next);
   // Solve (M + dt*K) u_next = M u_current + dt*source.
   void Step(mfem::Vector &u_current,
             const mfem::Vector &source,
             mfem::Vector &u_next);
   void UpdateStiffness(mfem::SparseMatrix &K);
   // Update the stiffness matrix in the time loop
private:
   void BuildSystemMatrix();
   void ConfigureLinearSolver();

   mfem::SparseMatrix &M_;
   mfem::SparseMatrix &K_;
   mfem::real_t dt_;

   mfem::SparseMatrix A_;
   mfem::Vector rhs_;

   mfem::CGSolver lin_solver_;
   std::unique_ptr<mfem::GSSmoother> A_prec_;
};

#endif
