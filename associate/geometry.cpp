#include "geometry.h" 
#include <ceres/loss_function.h> 
#include "colorterminal.h" 

Eigen::Vector3d NViewDLT(
    const std::vector<Camera>          &cams, 
    const std::vector<Eigen::Vector2d> &xs
)
{
    int N_view = cams.size();
    Eigen::MatrixXd A(N_view * 3, 4); 
    for(int i=0;i<cams.size();i++)
    {
        auto P = cams[i].P_g.cast<double>(); 
        auto x = xs[i]; 
        A.row(3*i) =   -P.row(1) +  P.row(2) * x(1); 
        A.row(3*i+1) = P.row(0) - x(0) * P.row(2); 
        A.row(3*i+2) = -x(1) * P.row(0) + x(0) * P.row(1);  
    }
    Eigen::Matrix4d ATA; 
    // ATA = A.transpose() * A; 
    ATA = A.adjoint() * A;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigenSolver(ATA); 
    // std::cout << "Eigen values: " << eigenSolver.eigenvalues() << std::endl; 
    // std::cout << "Eigen vectors: " << std::endl << eigenSolver.eigenvectors() << std::endl;
    Eigen::Matrix4d eigenvec = eigenSolver.eigenvectors(); 
    Eigen::Vector4d v = eigenvec.col(0); 
    Eigen::Vector3d point = v.block<3,1>(0, 0) / v(3); 
    return point; 
}

using ceres::AutoDiffCostFunction; 
using ceres::NumericDiffCostFunction; 
using ceres::CostFunction; 
using ceres::Problem; 
using ceres::Solver; 
using ceres::Solve; 
using ceres::LossFunction;
using ceres::LossFunctionWrapper; 
using ceres::HuberLoss; 

// Energy term: reprojection. By AN Liang, 20181001
struct E_reproj_euc
{
    E_reproj_euc(const Eigen::Matrix4d _P, 
                 const Eigen::Vector2d _u, 
                 const double _conf)
    {
        P    = _P; 
        u    = _u; 
        conf = _conf; 
    }

    template<typename T>
    bool operator()(const T* const X, T* residuals)const 
    { 
        T u_est[2];
        u_est[0] = T(P(0,0))*X[0] + T(P(0,1))*X[1] + T(P(0,2))*X[2] + T(P(0,3));  
        u_est[1] = T(P(1,0))*X[0] + T(P(1,1))*X[1] + T(P(1,2))*X[2] + T(P(1,3)); 
        T d      = T(P(2,0))*X[0] + T(P(2,1))*X[1] + T(P(2,2))*X[2] + T(P(2,3)); 
        u_est[0] = u_est[0] / d;
        u_est[1] = u_est[1] / d;
        residuals[0] = (u_est[0] - T(u.x()) ) * conf ;
        residuals[1] = (u_est[1] - T(u.y()) ) * conf ;  
        return true;
    }
private: 
    Eigen::Matrix4d P; 
    Eigen::Vector2d u; 
    double conf; 
};  

void Joint3DSolver::SetParam(
    const std::vector<Camera>            &_cams, 
    const std::vector<Eigen::Vector3d>   &_points
)
{
    cams.clear(); 
    u.clear();
    conf.clear(); 
    cams = _cams;
    for(int i = 0;i < cams.size(); i++)
    {
        u.push_back( _points[i].block<2,1>(0,0) );
        conf.push_back( _points[i](2) ); 
    }
}

void Joint3DSolver::SetParam(
    const std::vector<Camera>          &_cams, 
    const std::vector<Eigen::Vector2d> &_points, 
    const std::vector<double>          &_confidences
)
{
    cams.clear(); 
    u.clear(); 
    conf.clear(); 
    cams = _cams; 
    u    = _points; 
    conf = _confidences; 
}

void Joint3DSolver::SetInit( Eigen::Vector3d _X_init)
{
    X = _X_init; 
    status_init_set = true; 
}

void Joint3DSolver::Solve3D()
{
    Problem problem;
    // ceres::HuberLoss* huber_loss = new ceres::HuberLoss(1.0); 
    // ceres::CauchyLoss* loss_function = new ceres::CauchyLoss(0.0001); 
    if(!status_init_set) X = Eigen::Vector3d::Zero(); // default zero init 
    for(int i = 0; i < u.size(); i++)
    {
        E_reproj_euc *term_reproj = new E_reproj_euc(cams[i].P_g, u[i], conf[i]); 
        CostFunction *cost_func = 
            new AutoDiffCostFunction<E_reproj_euc, 2, 3>(term_reproj); 
        problem.AddResidualBlock(cost_func, NULL, X.data()); 
    }
    Solver::Options options;
    options.minimizer_progress_to_stdout = verbose;
    options.minimizer_type = ceres::TRUST_REGION;
    options.max_num_iterations = 200; 
    Solver::Summary summary;
    Solve(options, &problem, &summary);
}

double getEpipolarDistL2L(const Camera& cam1, const Camera& cam2, const Vec2& u1, const Vec2& u2)
{
    Vec3 u1_homo = ToHomogeneous(u1); 
    Vec3 u2_homo = ToHomogeneous(u2); 
    return getEpipolarDist(cam1, cam2, u1_homo, u2_homo);
}

double getEpipolarDistL2L(const Camera& cam1, const Camera& cam2, const Vec3& _u1, const Vec3& _u2)
{
    Vec3 u1 = _u1; 
    Vec3 u2 = _u2;
    u1(2) = 1; 
    u2(2) = 1; 
    Vec3 a = cam1.inv_K * u1; // x2 - x1 
    Vec3 b = cam2.inv_K * u2; // x4 - x3
    Mat3 Rrel = cam1.GetRelR(cam2); 
    Vec3 Trel = cam1.GetRelT(cam2); 
    Vec3 b_in_cam1 = Rrel * b; 
    Vec3 c = Trel; 

    return L2LDist(a, b_in_cam1, c);
}

double getEpipolarDist(const Vec3& p1, const Mat3& F, const Vec3& p2)
{
    Vec3 l = F.transpose() * p1; 
    double sqrt_ab = l.segment<2>(0).norm(); 
    return fabs(l.dot(p2) / sqrt_ab);
}

double getEpipolarDist(const Camera& cam1, const Camera& cam2, const Vec3& _u1, const Vec3& _u2)
{
    Mat3 F1 = cam1.GetFundamental(cam2); 
    Mat3 F2 = cam2.GetFundamental(cam1); 
    Vec3 u1 = _u1; u1(2) = 1; 
    Vec3 u2 = _u2; u2(2) = 1; 
    double dist1 = getEpipolarDist(u1, F1, u2); 
    double dist2 = getEpipolarDist(u2, F2, u1); 
    return (dist1 + dist2) / 2; 
}

void project(const Camera& cam, const vector<Eigen::Vector3d> &points3d, vector<Eigen::Vector3d> &points2d)
{
    points2d.resize(points3d.size()); 
    for(int i = 0; i < points3d.size(); i++)
    {
        Vec3 p = points3d[i];
        Vec3 p_proj = cam.K * (cam.R * p + cam.T); 
        p_proj = p_proj / p_proj(2); 
        points2d[i] = p_proj; 
    }
}

void project(const Camera& cam, Eigen::Vector3d p3d, Eigen::Vector3d &p2d)
{
    p2d = cam.K * (cam.R * p3d + cam.T);
    p2d = p2d / p2d(2); 
}

Eigen::Vector3d project(const Camera& cam, Eigen::Vector3d p3d)
{
    Vec3 p2d = cam.K * (cam.R * p3d + cam.T);
    p2d = p2d / p2d(2); 
    return p2d; 
}

Eigen::Vector3d triangulate_ceres(const std::vector<Camera> cams, const std::vector<Eigen::Vector3d> joints2d)
{
    Joint3DSolver solver; 
    solver.SetParam(cams, joints2d); 
    solver.SetVerbose(false); 
    solver.Solve3D(); 
    Eigen::Vector3d X = solver.GetX(); 
    return X; 
}