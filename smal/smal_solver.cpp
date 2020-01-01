#include "smal_solver.h" 
#include <Eigen/SVD> 
#include "../utils/colorterminal.h"
#include <fstream>
#include "../utils/math_utils.h"
#include <Eigen/Eigen>

SMAL_SOLVER::SMAL_SOLVER(std::string folder) : SMAL(folder)
{
    
}

// WARNINIG: deprecated now! need to update 
// AN Liang, 20191227
void SMAL_SOLVER::globalAlign()
{
    // // coarse align: global T
    // m_translation = Y.col(0) - jointsOrigin.col(0); 

    // // coarse align: global R 
    // Eigen::Matrix3d originAxis;
	// Eigen::Matrix3d targetAxis;

	// originAxis.col(0) = (jointsOrigin.col(2) - jointsOrigin.col(1)).normalized();
	// originAxis.col(1) = (jointsOrigin.col(3) - jointsOrigin.col(1)).normalized();
	// originAxis.col(2) = (originAxis.col(0).cross(originAxis.col(1))).normalized();
	// originAxis.col(1) = (originAxis.col(2).cross(originAxis.col(0))).normalized();
	
	// targetAxis.col(0) = (jointsTarget.col(2) - jointsTarget.col(1)).normalized();
	// targetAxis.col(1) = (jointsTarget.col(3) - jointsTarget.col(1)).normalized();
	// targetAxis.col(2) = (targetAxis.col(0).cross(targetAxis.col(1))).normalized();
	// targetAxis.col(1) = (targetAxis.col(2).cross(targetAxis.col(0))).normalized();

    // Eigen::Matrix3d rotationMat = targetAxis*(originAxis.inverse());
	// Eigen::AngleAxisd angleAxis(rotationMat);
	// m_poseParam.head(3) = angleAxis.axis() * angleAxis.angle();
}

void SMAL_SOLVER::globalAlignByVertices() // procrustes analysis, rigid align (R,t), and transform Y
{
    UpdateVertices(); 
    Eigen::Vector3d barycenter_target = Y.rowwise().mean(); 
    Eigen::Vector3d barycenter_source = m_verticesFinal.rowwise().mean(); 
    std::cout << "barycenter Y : " << barycenter_target.transpose() << std::endl; 
    std::cout << "barycenter   : " << barycenter_source.transpose() << std::endl; 
    m_translation = barycenter_target - barycenter_source; 
    Eigen::MatrixXd V_target = Y.colwise() - barycenter_target; 
    Eigen::MatrixXd V_source = m_verticesFinal.colwise() - barycenter_source; 
    Eigen::Matrix3d S = V_source * V_target.transpose(); 
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV); 
    std::cout << svd.singularValues() << std::endl; 
    Eigen::MatrixXd U = svd.matrixU(); 
    Eigen::MatrixXd V = svd.matrixV(); 
    Eigen::Matrix3d R = V * U.transpose(); 
    Eigen::AngleAxisd ax(R); 
    m_poseParam.segment<3>(0) = ax.axis() * ax.angle(); 
    // Y = (R.transpose() * V_target).colwise() + J_source.col(0); 

    std::cout << BLUE_TEXT("m_translation: ") << m_translation.transpose() << std::endl; 
}

// toy function: optimize shape without pose. 
void SMAL_SOLVER::OptimizeShape(const int maxIterTime, const double terminal)
{
    std::cout << GREEN_TEXT("solving shape ... ") << std::endl; 
    Eigen::VectorXd V_target = Eigen::Map<Eigen::VectorXd>(Y.data(), 3 * m_vertexNum);
    int iter = 0; 
    for(; iter < maxIterTime; iter++)
    {
        UpdateVertices(); 
        CalcShapeJacobi(); 
        Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd>(m_verticesFinal.data(), 3*m_vertexNum) - V_target; 
        Eigen::MatrixXd H1 = m_vertJacobiShape.transpose() * m_vertJacobiShape; 
        Eigen::VectorXd b1 = - m_vertJacobiShape.transpose() * r; 
        Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(m_shapeNum, m_shapeNum);  // Leveberg Marquart
        Eigen::MatrixXd H_reg = DTD;
        Eigen::VectorXd b_reg = -m_shapeParam; 
        double lambda = 0.0;  
        double w1 = 10000; 
        double w_reg = 0.0; 
        Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda; 
        Eigen::VectorXd b = b1 * w1 + b_reg * w_reg; 
        
        Eigen::VectorXd delta = H.ldlt().solve(b); 
        m_shapeParam = m_shapeParam + delta; 
        std::cout << "residual     : " << r.norm() << std::endl; 
        std::cout << "delta.norm() : " << delta.norm() << std::endl; 
        if(delta.norm() < terminal) break;
    }
    std::cout << "iter times: " << iter << std::endl; 
}


void SMAL_SOLVER::OptimizePose(const int maxIterTime, const double updateTolerance, bool is_numeric)
{
    std::cout << GREEN_TEXT("solving pose ... ") << std::endl; 
    Eigen::VectorXd V_target = Eigen::Map<Eigen::VectorXd>(Y.data(), 3*m_vertexNum); 
    Eigen::MatrixXd J; 
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
        std::cout << "iter time: " << iterTime << std::endl; 
        UpdateVertices(); 
        if(is_numeric) 
        {
            CalcPoseJacobiNumeric(); 
            J = m_vertJacobiPoseNumeric; 
        }
		else 
        {
            CalcPoseJacobi();
            J = m_vertJacobiPose; 
        }
        
        Eigen::VectorXd theta(3+3*m_jointNum); 
        theta.segment<3>(0) = m_translation; 
        theta.segment<99>(3) = m_poseParam; 
        Eigen::VectorXd f = Eigen::Map<Eigen::VectorXd>(m_verticesFinal.data(), 3*m_vertexNum); 
        Eigen::VectorXd r = f - V_target;
    
        Eigen::MatrixXd H1 = J.transpose() * J;
        double lambda = 0.001; 
        double w1 = 1; 
        double w_reg = 0.00001; 
        Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3+3*m_jointNum, 3+3*m_jointNum);
        Eigen::VectorXd b1 = - J.transpose() * r; 
        Eigen::MatrixXd H_reg = DTD; 
        Eigen::VectorXd b_reg = - theta; 

        Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda; 
        Eigen::VectorXd b = b1 * w1 + b_reg * w_reg; 
        
        Eigen::VectorXd delta = H.ldlt().solve(b); 
        m_translation += delta.segment<3>(0); 
        m_poseParam += delta.segment<99>(3); 

        std::cout << "residual     : " << r.norm() << std::endl; 
        std::cout << "delta.norm() : " << delta.norm() << std::endl; 
        if(delta.norm() < updateTolerance) break; 

	}
}




void SMAL_SOLVER::CalcPoseJacobi()
{
/*
AN Liang. 20191227 comments. This function was written by ZHANG Yuxiang. 
y: 3*N vector(N is joint num) 
x: 3+3*N vector(freedom variable, global t, and r for each joint (lie algebra)
dy/dx^T: Jacobi matrix J[3*N, 3+3*N]
below function update this matrix in colwise manner, in articulated tree structure. 
*/
	// calculate delta rodrigues
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3d& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
		rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiD(pose);
	}

	//dJ_dtheta
	m_jointJacobiPose = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
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

	// this version was deduced by AN Liang, 20191231
	// assume that you have computed Pose Jacobi
	m_vertJacobiPose = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum);
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
#pragma omp parallel for
    for(int vIdx=0; vIdx<m_vertexNum; vIdx++)
    {
        Eigen::Vector3d v0 = m_verticesShaped.col(vIdx); 
        for(int jIdx=0; jIdx<m_jointNum; jIdx++)
        {
            if(m_lbsweights(jIdx, vIdx) < 0.00001) continue;
            Eigen::Vector3d j0 = m_jointsShaped.col(jIdx);  
            m_vertJacobiPose.middleRows(3*vIdx, 3) += ( m_lbsweights(jIdx, vIdx) * m_jointJacobiPose.middleRows(3*jIdx, 3) );
            for(int pIdx=jIdx; pIdx>-1;pIdx=m_parent(pIdx)) // poseParamIdx to derive
            {
                Eigen::Matrix3d T = Eigen::Matrix3d::Zero(); 
                for(int axis_id=0;axis_id<3;axis_id++)
                {
                    Eigen::Matrix3d dR1 = RP.block<3,3>(9*pIdx+3*axis_id,0) * LP.block<3,3>(3*jIdx, 3*pIdx); 
                    T.col(axis_id) = dR1 * (v0 - j0) * m_lbsweights(jIdx,vIdx); 
                }
                m_vertJacobiPose.block<3,3>(3*vIdx,3+3*pIdx) += T; 
            }
        }
    }
	
// 	// calculate vertex jacobi, deduced by ZHANG Yuxiang. Use twist approximation at end joints. 
// 	m_vertJacobiPose = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum);
// #pragma omp parallel for
// 	for (int vIdx = 0; vIdx < m_vertexNum; vIdx++) 
// 	{
// 		for (int jIdx = 0; jIdx < m_jointNum; jIdx++) {
// 			if (m_lbsweights(jIdx, vIdx) < 0.00001)
// 				continue;
// 			m_vertJacobiPose.middleRows(3 * vIdx, 3) += m_lbsweights(jIdx, vIdx) * m_jointJacobiPose.middleRows(3 * jIdx, 3);
// 			const Eigen::Matrix3d skew = - GetSkewMatrix(m_verticesFinal.col(vIdx) - m_jointsFinal.col(jIdx));
// 			for (int _prtIdx = jIdx; _prtIdx != -1; _prtIdx = m_parent(_prtIdx) )
// 				m_vertJacobiPose.block<3, 3>(3 * vIdx, 3 + 3 * _prtIdx) += m_lbsweights(jIdx, vIdx) * skew * m_globalAffine.block<3, 3>(0, 4 * _prtIdx);
// 		}
// 	}
}

void SMAL_SOLVER::CalcPoseJacobiNumeric()
{
    m_jointJacobiPoseNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
    m_vertJacobiPoseNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum); 

    UpdateVertices(); 
    Eigen::MatrixXd previousJ = m_jointsFinal; 
    Eigen::MatrixXd previousV = m_verticesFinal; 
    Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(3*m_jointNum); 
    double alpha = 0.000001; 
    double inv_alpha = 1.0/alpha; 
    for(int i = 0; i < 3; i++)
    {
        Eigen::Vector3d delta_t = Eigen::Vector3d::Zero(); 
        delta_t(i) = alpha; 
        m_translation += delta_t;
        UpdateVertices(); 
        Eigen::MatrixXd delta_j = m_jointsFinal - previousJ; 
        m_jointJacobiPoseNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_j.data(), 3*m_jointNum) * inv_alpha; 
        Eigen::MatrixXd delta_v = m_verticesFinal - previousV; 
        m_vertJacobiPoseNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_v.data(), 3*m_vertexNum) * inv_alpha; 
        m_translation -= delta_t; 
    }
    for(int i = 0; i < 3*m_jointNum; i++) 
    {
        delta_x.setZero(); 
        delta_x(i) = alpha;
        m_poseParam += delta_x; 
        UpdateVertices(); 
        Eigen::MatrixXd delta_j = m_jointsFinal - previousJ; 
        m_jointJacobiPoseNumeric.col(3+i) = Eigen::Map<Eigen::VectorXd>(delta_j.data(), 3*m_jointNum) * inv_alpha; 
        Eigen::MatrixXd delta_v = m_verticesFinal - previousV; 
        m_vertJacobiPoseNumeric.col(3+i) = Eigen::Map<Eigen::VectorXd>(delta_v.data(), 3*m_vertexNum) * inv_alpha; 
        m_poseParam -= delta_x; 
    }
} 

void SMAL_SOLVER::CalcShapeJacobiNumeric()
{
    m_jointJacobiShapeNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3*m_jointNum, m_shapeNum); 
    m_vertJacobiShapeNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3*m_vertexNum, m_shapeNum); 
    UpdateVertices(); 
    Eigen::MatrixXd previousJ = m_jointsFinal; 
    Eigen::MatrixXd previousV = m_verticesFinal; 
    Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(m_shapeNum); 
    double alpha = 0.0001; 
    double inv_alpha = 1.0 / alpha; 
    for(int i = 0; i < m_shapeNum; i++)
    {
        delta_x.setZero(); 
        delta_x(i) = alpha; 
        m_shapeParam += delta_x; 
        UpdateVertices(); 
        Eigen::MatrixXd delta_j = m_jointsFinal - previousJ; 
        m_jointJacobiShapeNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_j.data(), 3*m_jointNum) * inv_alpha; 
        Eigen::MatrixXd delta_v = m_verticesFinal - previousV; 
        m_vertJacobiShapeNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_v.data(), 3*m_vertexNum) * inv_alpha; 
        m_shapeParam -= delta_x; 
    }
}

void SMAL_SOLVER::CalcShapeJacobi()
{
    // compute d_joint_d_beta
    m_jointJacobiShape.resize(3*m_jointNum, m_shapeNum); 
    m_jointJacobiShape.setZero(); 
    for(int jIdx = 0; jIdx < m_jointNum; jIdx++)
    {
        if(jIdx==0)
            m_jointJacobiShape.middleRows(3*jIdx, 3) += m_shapeBlendJ.middleRows(3*jIdx,3);
        else 
        {
            int pIdx = m_parent(jIdx); 
            m_jointJacobiShape.middleRows(3*jIdx,3) = m_jointJacobiShape.middleRows(3*pIdx, 3) + m_globalAffine.block<3,3>(0,4*pIdx) 
                                                      * ( m_shapeBlendJ.middleRows(3*jIdx,3) - m_shapeBlendJ.middleRows(3*pIdx,3) ); 
        }
    }

    // compute d_v_d_beta
    m_vertJacobiShape.resize(3*m_vertexNum, m_shapeNum);
    m_vertJacobiShape.setZero(); 
    for(int vIdx = 0; vIdx < m_vertexNum; vIdx++)
    {
        for(int jIdx=0; jIdx < m_jointNum; jIdx++)
        {
            if(m_lbsweights(jIdx, vIdx) < 0.00001) continue; 
            m_vertJacobiShape.middleRows(3*vIdx, 3) += ( ( m_jointJacobiShape.middleRows(3*jIdx,3) 
                                                    + m_globalAffine.block<3,3>(0, 4*jIdx) * (m_shapeBlendV.middleRows(3*vIdx,3) - m_shapeBlendJ.middleRows(3*jIdx,3))
            ) * m_lbsweights(jIdx, vIdx) ) ; 
        }
    }
}

void SMAL_SOLVER::debug()
{
    UpdateVertices(); 
    CalcPoseJacobi(); 
    CalcPoseJacobiNumeric(); 
    Eigen::MatrixXd C = (m_jointJacobiPose - m_jointJacobiPoseNumeric);
    Eigen::MatrixXd D = (m_vertJacobiPose - m_vertJacobiPoseNumeric);
    std::cout << BLUE_TEXT("D") << std::endl; 
    // std::cout << D.middleRows(0, 3).middleCols(3,3) << std::endl; 
    std::cout << D.middleRows(0, 3).transpose() << std::endl; 

    std::cout << "analytic: "<< std::endl; 
    // std::cout << m_vertJacobiPose.middleRows(0,3).middleCols(3,3) << std::endl; 
    std::cout << m_vertJacobiPose.middleRows(0,3).transpose() << std::endl; 

    std::cout << "numeric: " << std::endl; 
    // std::cout << m_vertJacobiPoseNumeric.middleRows(0,3).middleCols(3,3) << std::endl; 
    std::cout << m_vertJacobiPoseNumeric.middleRows(0,3).transpose() << std::endl; 

    std::cout << "max diff: " << D.array().abs().matrix().maxCoeff() << std::endl; 

    // CalcShapeJacobi(); 
    // CalcShapeJacobiNumeric(); 
    // Eigen::MatrixXd A = m_jointJacobiShape - m_jointJacobiShapeNumeric; 
    // Eigen::MatrixXd B = m_vertJacobiShape - m_vertJacobiShapeNumeric;
    // std::cout << RED_TEXT("A") << std::endl; 
    // std::cout << A.middleRows(0, 6).transpose() << std::endl; 
    // std::cout << GREEN_TEXT("B") << std::endl; 
    // std::cout << B.middleRows(0,6).transpose() << std::endl; 
}