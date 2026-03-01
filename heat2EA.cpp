#include "mfem.hpp"
#include "LILS.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{

    // MFEM Hardcoded options
    const char *mesh_file = "../data/ref-square.mesh";
    int order = 2;
    bool static_cond = false;
    const char *device_config = "cpu";
    bool visualization = true;
    bool algebraic_ceed = false;

    Device device(device_config);
    device.Print();


    //   1. Read Mesh File, take dimension, and print number of attributes.
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    cout << "Number of Attributes: " << mesh.bdr_attributes.Size() << flush;
    

    //   2. Refine the mesh for resolution. We choose 'ref_levels' to be the
    //      largest number that gives a final mesh with no more than 5,000
    //      elements.
    {
    int ref_levels =
        (int)floor(log(5000./mesh.GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh.UniformRefinement();
        }
    }


    //   3. Define the finite element space on the mesh. 
    FiniteElementCollection *fec;
    FiniteElementSpace *fespace;
    fec = new H1_FECollection(order, dim);
    fespace = new FiniteElementSpace(&mesh, fec);
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;


    //   4. Determine the list of essensial boundary dofs.
    //      Tell MFEM that the outer edge of the outer edge
    //      of the mesh is essensial. Atribute 1 is bottom, 
    //      then 2 is right, 3 is top, and 4 is left. 
    Array<int> ess_tdof_list, left_bdr(mesh.bdr_attributes.Max());
    cout << mesh.bdr_attributes.Max() << flush;
    left_bdr = 0;
    left_bdr[4-1] = 1;
    fespace->GetEssentialTrueDofs(left_bdr, ess_tdof_list);
    

    //   5. Set the coefficents used later in assembly 
    real_t alpha_val = 1.0;
    real_t q_flux_val = 373.15;
    real_t zero = 0.0;
    real_t fixed_temp = 273.15;
    ConstantCoefficient alphaCoef(alpha_val); 
    ConstantCoefficient qFluxCoef(q_flux_val);
    ConstantCoefficient zeroCoef(zero);
    ConstantCoefficient fixedTempCoef(fixed_temp);


    //   6. Define the Grid Function & project Dirichlet Boundary
    //      coefficient
    GridFunction x(fespace);
    x = 0.0;
    x.ProjectBdrCoefficient(fixedTempCoef, ess_tdof_list);


    //   7. Set up the Bileniear Form m(.) and a(.). Here the 
    //      mass integratorsets up the integral over space of 
    //      (u * v), while the diffusion integrator sets up the
    //      integral over space of (alpha*Nabla_u dot Nabla_v).
    BilinearForm m(fespace);
    m.AddDomainIntegrator(new MassIntegrator);
    BilinearForm a(fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator(alphaCoef));


    //   8. Set up linear form b(.) & Neumann Boundary, this 
    //      means that there is no forcing term (exept right)  
    //      and everything enters through that right boundary.
    //      This sets up the surface integral over the boundary
    //      of (-alpha Nabla_u dot n).
    LinearForm b(fespace);
    b = 0.0;
    Array<int> right_bdr(mesh.bdr_attributes.Max());
    right_bdr = 0;
    right_bdr[2-1] = 1;
    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(qFluxCoef),
                            right_bdr);
    

    //   9. Assemble the mass matrix M, diffusion matrix K
    //      and the forcing vector B. Then Form constrained 
    //      linear system for the operator 'a' and rhs 'b'
    m.Assemble();
    a.Assemble();
    b.Assemble();
    SparseMatrix M;
    SparseMatrix K;
    Vector B(b.Size());
    m.FormSystemMatrix(ess_tdof_list, M);
    a.FormSystemMatrix(ess_tdof_list, K);

    OperatorPtr A;
    Vector X;
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
    

    //  12. Solve the system using LinearImplicitLinearSolve class
    //      The python parent file should specify the time step size.
    real_t dt = atof(argv[1]);
    real_t t = 0.0;
    real_t t_final = 1.0;
    LinearImplicitLinearSolve LILS(M, K, dt);
    Vector u = X;
    Vector u_next(u.Size());
    while (t <= t_final)
    {
        LILS.Step(u, B, u_next);
        u = u_next;
        t += dt;
        cout << "Time: " << t << endl;
    }
    X = u;
    x = X;


    //  13. Get L2 Norms and Energy Norms. Compute the L2 norm by: 
    //      sqrt(x^T M x) and energy norm: sqrt(x^T K x).
    Vector tmp(x.Size());
    Vector Mx(x.Size());
    Vector Kx(x.Size());
    M.Mult(x, Mx);
    K.Mult(x, Kx);
    real_t energy_norm = sqrt(x * Kx);


    //      Norms of RHS B and of Mx, Kx
    real_t B_norm = B.Norml2();
    real_t Mx_norm = Mx.Norml2();
    real_t Kx_norm = Kx.Norml2();

    //      Residual r = Kx - B + M(u^n+1 - u^n)/dt
    Vector du(x.Size());
    du = u_next;
    du -= u;           // u_next - u
    
    Vector Mdu(x.Size());
    M.Mult(du, Mdu);   // M * (u_next - u)
    Mdu /= dt;          // divide by dt

    Vector r(x.Size());
    r += Kx;
    r -= B;
    r += Mdu;
    r /= dt; 
    real_t r_l2 = r.Norml2();
    

//    cout << "B L2: " << B_norm << ", Mx L2: " << Mx_norm << ", Kx L2: " << Kx_norm << endl;
    cout << "Residual L2: " << r_l2 << endl;
    string results = "L2_Results";
    filesystem::create_directory(results);
    ofstream outfile(results + "/L2outputFile.txt");
    outfile << r_l2 << endl;
    outfile.close();

    delete fespace;
    delete fec;
 
    return 0;
}
