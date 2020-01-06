#include "smal_2dsolver.h" 
#include <Eigen/SVD> 
#include "../utils/colorterminal.h"
#include <fstream>
#include "../utils/math_utils.h"
#include <Eigen/Eigen>
#include "../associate/skel.h"

#define DEBUG_SOLVER

SMAL_2DSOLVER::SMAL_2DSOLVER(std::string folder) : SMAL(folder)
{
    m_topo = getSkelTopoByType("UNIV"); 
    m_poseToOptimize = {
        0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23
    }; 
}

void SMAL_2DSOLVER::globalAlign() // procrustes analysis, rigid align (R,t), and transform Y
{
    m_weights.resize(Y.cols()); 
    m_weightsEigen = Eigen::VectorXd::Zero(3*Y.cols()); 
    // compute scale
    std::vector<double> target_bone_lens; 
    std::vector<double> source_bone_lens; 
    for(int bid = 0; bid < m_topo.bones.size(); bid++)
    {
        int sid = m_topo.bones[bid](0);
        int eid = m_topo.bones[bid](1);
        if(Y.col(sid).norm()==0 || Y.col(eid).norm()==0) continue; 
        double target_len = (Y.col(sid) - Y.col(eid)).norm(); 
        std::vector<int> ids = {sid, eid}; 
        std::vector<Eigen::Vector3d> points;
        for(int k = 0; k < 2; k++)
        {
            if(m_mapper[ids[k]].first < 0) continue; 
            if(m_mapper[ids[k]].first == 0) 
            {
                Eigen::Vector3d X = m_jointsFinal.col(m_mapper[ids[k]].second);
                points.push_back(X); 
            }
            else 
            {
                Eigen::Vector3d X = m_verticesFinal.col(m_mapper[ids[k]].second); 
                points.push_back(X); 
            }
        }
        double source_len = (points[0] - points[1]).norm(); 
        target_bone_lens.push_back(target_len); 
        source_bone_lens.push_back(source_len); 
    }
    double a = 0; 
    double b = 0; 
    for(int i = 0; i < target_bone_lens.size(); i++)
    {
        a+=target_bone_lens[i] * source_bone_lens[i]; 
        b+=source_bone_lens[i] * source_bone_lens[i];
    }
    double alpha = a / b; 
    m_scale = alpha; 
    m_jointsOrigin *= alpha; 
    m_verticesOrigin *= alpha; 
    m_shapeBlendV *= alpha; 
    m_shapeBlendJ *= alpha; 
#ifdef DEBUG_SOLVER
    std::cout << BLUE_TEXT("scale: ") << alpha << std::endl; 
#endif 
    UpdateVertices(); 

    // compute translation 
    Eigen::Vector3d barycenter_target = Y.col(18); 
    Eigen::Vector3d barycenter_source = m_jointsOrigin.col(25); 
    m_translation = barycenter_target - barycenter_source; 

    // compute rotation 
    Eigen::MatrixXd A,B;
    int nonzero = 0;
    for(int i = 0; i < Y.cols(); i++) {
        if(Y.col(i).norm() > 0) {
            nonzero++; 
            m_weights[i] = 1; 
            m_weightsEigen.segment<3>(3*i) = Eigen::Vector3d::Ones(); 
        }
        else m_weights[i] = 0;
    }
    A.resize(3,nonzero); B.resize(3,nonzero); 
    int k = 0; 
    for(int i = 0; i < Y.cols(); i++)
    {
        if(m_mapper[i].first < 0) continue; 
        if(Y.col(i).norm() > 0) 
        {
            A.col(k) = Y.col(i);
            if(m_mapper[i].first == 0) B.col(k) = m_jointsFinal.col(m_mapper[i].second); 
            else B.col(k) = m_verticesFinal.col(m_mapper[i].second); 
            k++; 
        }
    }
    Eigen::MatrixXd V_target = A.colwise() - barycenter_target; 
    Eigen::MatrixXd V_source = B.colwise() - barycenter_source; 
    Eigen::Matrix3d S = V_source * V_target.transpose(); 
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV); 
#ifdef DEBUG_SOLVER
    std::cout << BLUE_TEXT("svd singular values: ") << svd.singularValues().transpose() << std::endl; 
#endif 
    Eigen::MatrixXd U = svd.matrixU(); 
    Eigen::MatrixXd V = svd.matrixV(); 
    Eigen::Matrix3d R = V * U.transpose(); 
    Eigen::AngleAxisd ax(R); 
    m_poseParam.segment<3>(0) = ax.axis() * ax.angle(); 
#ifdef DEBUG_SOLVER
    std::cout << BLUE_TEXT("m_translation: ") << m_translation.transpose() << std::endl; 
#endif 
    UpdateVertices(); 
}

void SMAL_2DSOLVER::optimizePose(const int maxIterTime, const double updateTolerance)
{
#ifdef DEBUG_SOLVER
    std::cout << GREEN_TEXT("solving pose ... ") << std::endl; 
#endif 
    int M = m_poseToOptimize.size(); 
    int N = Y.cols(); 
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
#ifdef DEBUG_SOLVER
        std::cout << "iter time: " << iterTime << std::endl; 
#endif 
        UpdateVertices(); 
        CalcPoseJacobi();
        Eigen::VectorXd theta(3+3*M); 
        theta.segment<3>(0) = m_translation; 
        for(int i = 0; i < M; i++)
        {
            int jIdx = m_poseToOptimize[i];
            theta.segment<3>(3+3*i) = m_poseParam.segment<3>(3*jIdx); 
        }
        std::cout << "ok" << std::endl; 
        // solve
        Eigen::MatrixXd H1 = Eigen::MatrixXd::Zero(3+3*M, 3+3*M); // data term 
        Eigen::VectorXd b1 = Eigen::VectorXd::Zero(3+3*M);  // data term 
        for(int k = 0; k < m_source.view_ids.size(); k++)
        {
            Eigen::MatrixXd H_view;
            Eigen::VectorXd b_view;
            Calc2DJacobi(k, H_view, b_view); 
            std::cout << "ok " << k << std::endl; 
            H1 += H_view; 
            b1 += b_view;  
        }
        double lambda = 0.001; 
        double w1 = 1; 
        double w_reg = 0.01; 
        Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3+3*M, 3+3*M);
        Eigen::MatrixXd H_reg = DTD;  // reg term 
        Eigen::VectorXd b_reg = - theta; // reg term 

        Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda; 
        Eigen::VectorXd b = b1 * w1 + b_reg * w_reg; 
        
        Eigen::VectorXd delta = H.ldlt().solve(b);

        // update 
        m_translation += delta.segment<3>(0); 
        for(int i = 0; i < M; i++)
        {
            int jIdx = m_poseToOptimize[i];
            m_poseParam.segment<3>(3*jIdx) += delta.segment<3>(3+3*i); 
        }
#ifdef DEBUG_SOLVER
        std::cout << "delta.norm() : " << delta.norm() << std::endl; 
#endif 
        if(delta.norm() < updateTolerance) break; 
	}
}




void SMAL_2DSOLVER::CalcPoseJacobi()
{
/*
AN Liang. 20191227 comments. This function was written by ZHANG Yuxiang. 
y: 3*N vector(N is joint num) 
x: 3+3*N vector(freedom variable, global t, and r for each joint (lie algebra)
dy/dx^T: Jacobi matrix J[3*N, 3+3*N]
below function update this matrix in colwise manner, in articulated tree structure. 
*/
    int N = Y.cols(); 
    int M = m_poseToOptimize.size(); 
    m_JacobiPose = Eigen::MatrixXd::Zero(3*N, 3+3*M); 

	// calculate delta rodrigues
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3d& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
		rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiD(pose);
	}

	//dJ_dtheta
	Eigen::MatrixXd m_jointJacobiPose = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	for (int jointDerivativeId = 0; jointDerivativeId < m_jointNum; jointDerivativeId++)
	{
		// update translation term
		m_jointJacobiPose.block<3, 3>(jointDerivativeId * 3, 0).setIdentity();

		// update poseParam term
		for (int axisDerivativeId = 0; axisDerivativeId < 3; axisDerivativeId++)
		{
			std::vector<std::pair<bool, Eigen::Matrix4d>> globalAffineDerivative(m_jointNum, std::make_pair(false, Eigen::Matrix4d::Zero()));
			globalAffineDerivative[jointDerivativeId].first = true;
			auto& affine = globalAffineDerivative[jointDerivativeId].second;
			affine.block<3, 3>(0, 0) = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jointDerivativeId + axisDerivativeId));
			affine = jointDerivativeId == 0 ? affine : (m_globalAffine.block<4, 4>(0, 4 * m_parent(jointDerivativeId)) * affine);

			for (int jointId = jointDerivativeId + 1; jointId < m_jointNum; jointId++)
			{
				if (globalAffineDerivative[m_parent(jointId)].first)
				{
					globalAffineDerivative[jointId].first = true;
					globalAffineDerivative[jointId].second = globalAffineDerivative[m_parent(jointId)].second * m_singleAffine.block<4, 4>(0, 4 * jointId);
					// update jacobi for pose
					m_jointJacobiPose.block<3, 1>(jointId * 3, 3 + jointDerivativeId * 3 + axisDerivativeId) = globalAffineDerivative[jointId].second.block<3, 1>(0, 3);
				}
			}
		}
	}
    for(int i = 0; i < N; i++)
    {
        if(m_mapper[i].first != 0) continue; 
        int jIdx = m_mapper[i].second; 
        m_JacobiPose.block<3,3>(3*i,0) = m_jointJacobiPose.block<3,3>(3*jIdx, 0); 
        for(int k = 0; k < M; k++)
        {
            int thetaIdx = m_poseToOptimize[k]; 
            m_JacobiPose.block<3,3>(3*i, 3+3*k) = m_jointJacobiPose.block<3,3>(3*jIdx,3+3*thetaIdx); 
        }
    }

	// this version was deduced by AN Liang, 20191231
	// assume that you have computed Pose Jacobi
    // O(n^2)
    Eigen::Matrix<double, -1,-1, Eigen::ColMajor> RP(9*m_jointNum, 3); 
    Eigen::Matrix<double, -1,-1, Eigen::ColMajor> LP(3*m_jointNum, 3*m_jointNum); 
    RP.setZero();  
    LP.setZero(); 
    for(int jIdx = 0; jIdx < m_jointNum; jIdx++)
    {
        for(int aIdx = 0; aIdx < 3; aIdx++)
        {
            Eigen::Matrix3d dR = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jIdx + aIdx));
            if(jIdx > 0) 
            {
                dR = m_globalAffine.block<3,3>(0, 4*m_parent(jIdx)) * dR; 
            }
            RP.block<3,3>(9*jIdx+3*aIdx, 0) = dR; 
        }
        LP.block<3,3>(3*jIdx, 3*jIdx) = Eigen::Matrix3d::Identity(); 
        for(int child = jIdx+1; child < m_jointNum; child++)
        {
            int father = m_parent(child); 
            LP.block<3,3>(3*child, 3*jIdx) = LP.block<3,3>(3*father,3*jIdx) * m_singleAffine.block<3,3>(0, 4*child); 
        }
    }

    Eigen::MatrixXd m_vertJacobiPose = Eigen::MatrixXd::Zero(3, 3+3*m_jointNum); 
    for(int i=0; i<N; i++)
    {
        if(m_mapper[i].first!=1) continue; 
        int vIdx = m_mapper[i].second; 
        Eigen::Vector3d v0 = m_verticesShaped.col(vIdx); 
        m_vertJacobiPose.setZero(); 
        for(int jIdx=0; jIdx<m_jointNum; jIdx++)
        {
            if(m_lbsweights(jIdx, vIdx) < 0.00001) continue;
            Eigen::Vector3d j0 = m_jointsShaped.col(jIdx);  
            m_vertJacobiPose += ( m_lbsweights(jIdx, vIdx) * m_jointJacobiPose.middleRows(3*jIdx, 3) );
            for(int pIdx=jIdx; pIdx>-1;pIdx=m_parent(pIdx)) // poseParamIdx to derive
            {
                Eigen::Matrix3d T = Eigen::Matrix3d::Zero(); 
                for(int axis_id=0;axis_id<3;axis_id++)
                {
                    Eigen::Matrix3d dR1 = RP.block<3,3>(9*pIdx+3*axis_id,0) * LP.block<3,3>(3*jIdx, 3*pIdx); 
                    T.col(axis_id) = dR1 * (v0 - j0) * m_lbsweights(jIdx,vIdx); 
                }
                m_vertJacobiPose.block<3,3>(0,3+3*pIdx) += T; 
            }
        }
        m_JacobiPose.block<3,3>(3*i,0) = m_vertJacobiPose.block<3,3>(0,0); 
        for(int k = 0; k < M; k++)
        {
            int thetaIdx = m_poseToOptimize[k];
            m_JacobiPose.block<3,3>(3*i, 3+3*k) = m_vertJacobiPose.block<3,3>(0, 3+3*thetaIdx); 
        }
    }
}


Eigen::MatrixXd SMAL_2DSOLVER::getRegressedSkel()
{
    int N = Y.cols(); 
    Eigen::MatrixXd skel = Eigen::MatrixXd::Zero(3,N); 
    for(int i = 0; i < N; i++)
    {
        if(m_mapper[i].first < 0) continue; 
        if(m_mapper[i].first == 0) 
        {
            int jIdx = m_mapper[i].second; 
            skel.col(i) = m_jointsFinal.col(jIdx);
        }
        else if(m_mapper[i].first==1)
        {
            int vIdx = m_mapper[i].second; 
            skel.col(i) = m_verticesFinal.col(vIdx); 
        }
        else {
            std::cout << RED_TEXT("fatal error: in SMAL_2DSOLVER::getRegressedSkel.") << std::endl;
            exit(-1);
        }
    }
    return skel; 
}

void SMAL_2DSOLVER::Calc2DJacobi(
    int k, 
    Eigen::MatrixXd& H,
    Eigen::VectorXd& b
)
{
    std::cout << m_source.view_ids.size() << std::endl; 
    std::cout << "k; " << k<< std::endl; 
    int view = m_source.view_ids[k]; 
    Camera cam = m_cameras[view]; 
    DetInstance& det = m_source.dets[k]; 
    Eigen::Matrix3d R = cam.R;
    Eigen::Matrix3d K = cam.K; 
    Eigen::Vector3d T = cam.T; 
    
    int N = Y.cols(); 
    int M = m_poseToOptimize.size(); 
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2*N, 3+3*M); 
    Eigen::VectorXd r = Eigen::VectorXd::Zero(2*N); 
    Eigen::MatrixXd skel = getRegressedSkel(); 
    for(int i = 0; i < N; i++)
    {
        Eigen::Vector3d x_local = K * (R * skel.col(i) + T);
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2,3); 
        D(0,0) = 1/x_local(2); 
        D(1,1) = 1/x_local(2); 
        D(0,2) = -x_local(0) / (x_local(2) * x_local(2)); 
        D(1,2) = -x_local(1) / (x_local(2) * x_local(2)); 
        J.middleRows(2*i,2) = D * K * R * m_JacobiPose.middleRows(3*i, 3);
        Eigen::Vector2d u;
        u(0) = x_local(0) / x_local(2); 
        u(1) = x_local(1) / x_local(2); 
        r.segment<2>(2*i) = m_weights[i] * (u - det.keypoints[i].segment<2>(0) ); 
        if(det.keypoints[i](2) < m_topo.kpt_conf_thresh[i]) r.segment<2>(2*i) = Eigen::Vector2d::Zero();  
    }
    b = - J.transpose() * r; 
    H = J.transpose() * J; 
}