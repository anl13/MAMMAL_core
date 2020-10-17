#include "NanoRenderer.h"
#include "../utils/math_utils.h"

void ArcballCanvas::SetExtrinsic(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _up, const Eigen::Vector3f& _center)
{
	pos = _pos;
	up = _up;
	center = _center;

	front = (pos - center).normalized();
	right = (front.cross(up)).normalized();
	up = (right.cross(front)).normalized();

	m_viewRT = LookAt(pos, center, up);
}

// ATTENTION: transform coordinate system to opencv image system
void ArcballCanvas::SetExtrinsic(const Eigen::Matrix3f& _R, const Eigen::Vector3f& _T)
{
	R = _R;
	T = _T;
	front = -R.row(2).transpose();
	up = -R.row(1).transpose();
	pos = -R.transpose() * T;

	center = pos - 1.0f*front;
	SetExtrinsic(pos, up, center);
}

void ArcballCanvas::UpdateViewport()
{
	Matrix4f T_nano(0);
	for (int r = 0; r < 4; ++r)
	{
		for (int c = 0; c < 4; ++c)
		{
			T_nano.m[r][c] = m_viewRT(c, r);
		}
	}

	for (auto& it : m_render_objects)
	{
		it->SetView(T_nano);
	}
}

Eigen::Matrix4f ArcballCanvas::RotateAroundCenterAxis(const float& _theta, const Eigen::Vector3f& _w, const Eigen::Vector3f& _rot_center)
{
	const float theta = _theta * 0.1;
	const Eigen::Vector3f rot_center = _rot_center;
	
	Eigen::Vector3f w = _w;
	Eigen::Matrix4f T;
	T.setIdentity();

	if (std::fabsf(theta) < FLT_EPSILON) return T;
	if (w.isZero()) return T;

	w = w.normalized();


	Eigen::Vector3f v = rot_center.cross(w);
	std::cout << v << std::endl;


	Eigen::AngleAxisf psi(theta, w);
	Eigen::Vector3f rho = theta * v;

	T.topLeftCorner(3, 3) = psi.matrix();

	Eigen::Matrix3f wwT;
	for (int r = 0; r < 3; ++r)
	{
		for (int c = 0; c < 3; ++c)
		{
			wwT(r, c) = w(r) * w(c);
		}
	}

	Eigen::Matrix3f w_hat; w_hat.setZero();
	w_hat(0, 1) = -w(2);
	w_hat(0, 2) = w(1);
	w_hat(1, 2) = -w(0);
	w_hat(1, 0) = w(2);
	w_hat(2, 0) = -w(1);
	w_hat(2, 1) = w(0);
	T.topRightCorner(3, 1) = (sinf(theta) / theta * Eigen::Matrix3f::Identity() + (1 - sinf(theta) / theta) * wwT + (1 - cosf(theta)) / theta * w_hat) * rho;

	return T;
}

Eigen::Vector3f ArcballCanvas::GetArcballCoord
(
	const Eigen::Vector2f& planeCoord,
	const Eigen::Vector3f& front,
	const Eigen::Vector3f& up,
	const Eigen::Vector3f& right
	)
{
	// Attention: planeCoord should between [-1, 1]
	float x = planeCoord.x() / 1;
	float y = planeCoord.y() / 1;
	float z = 0;
	float r = x * x + y * y;
	if (r > 1)
	{
		x = x / r;
		y = y / r;
		z = 0;
	}
	else
	{
		z = sqrtf(1 - powf(x, 2) - powf(y, 2));
	}

	return (right * x + up * y + front * z);
};

Eigen::Vector3f ArcballCanvas::Project2Arcball(Eigen::Vector2f& p)
{
	if (p.norm() < 1.f) {
		return Eigen::Vector3f(p[0], p[1], std::sqrtf(1 - p.squaredNorm()));
	}
	else {
		return Eigen::Vector3f(p.normalized()[0], p.normalized()[1], 0.f);
	}
}

void ArcballCanvas::RotateViewport(const Vector2i &p, const Vector2i &rel)
{

#if 1
	// 2020/10/17
	// this is wired, p.x + rel.x is ok, but p.x is not ok. 
	// I trully cant understand now. 
	Eigen::Vector2f nowPos = Eigen::Vector2f(p.x() + rel.x(), p.y() + rel.y());
	const Eigen::Vector3f camCenter = center; 
	const Eigen::Vector3f camPos = pos; 
	const Eigen::Vector3f camUp = up;
	const Eigen::Vector3f camRight = right; 
	const Eigen::Vector3f camFront = front; 
	Eigen::Vector2f _nowPos = nowPos; 
	_nowPos(0) /= size().x();
	_nowPos(1) /= size().y(); 
	const Eigen::Vector3f nowArcCoord = GetArcballCoord(
		_nowPos * 2 - Eigen::Vector2f::Ones(), camFront, camUp, camRight);
	Eigen::Vector2f _before_pos;
	_before_pos(0) = p.x(); 
	_before_pos(1) = p.y(); 
	_before_pos(0) /= size().x();
	_before_pos(1) /= size().y();
	const Eigen::Vector3f beforeArcCoord = GetArcballCoord(
		_before_pos * 2 - Eigen::Vector2f::Ones(), camFront, camUp, camRight);

	float sensitivity = 2;
	const float theta = acos(beforeArcCoord.dot(nowArcCoord));
	const Eigen::Vector3f rotationAxis = theta < FLT_EPSILON ? Eigen::Vector3f(0.0f, 0.0f, 1.0f) : (beforeArcCoord.cross(nowArcCoord)).normalized();

	const Eigen::Vector3f nowCamPos = Eigen::AngleAxisf(sensitivity * theta, rotationAxis) * (camPos - camCenter) + camCenter;
	SetExtrinsic(nowCamPos, camUp, camCenter); 

#endif 

#if 0
	const int canvas_width = size()[0];
	const int canvas_height = size()[1];
	//const float pixel_radius = std::min(canvas_height, canvas_width) / 2;
	const float pixel_radius = std::max(canvas_height, canvas_width) / 2;

	Vector2i pe = p + rel;
	Eigen::Vector2f pBegin((p[0] - canvas_width / 2) / pixel_radius, (p[1] - canvas_height / 2) / pixel_radius);
	Eigen::Vector2f pEnd((pe[0] - canvas_width / 2) / pixel_radius, (pe[1] - canvas_height / 2) / pixel_radius);

	Eigen::Vector3f vBegin = Project2Arcball(pBegin);
	Eigen::Vector3f vEnd = Project2Arcball(pEnd);
	
	const float theta = std::acosf(vBegin.dot(vEnd)); 
	if (isnan(theta)) return;
	Eigen::Vector3f w = (vBegin.cross(vEnd)).normalized();
	Eigen::Matrix4f deltaT = RotateAroundCenterAxis(theta, w, m_rot_center);

	m_viewRT = deltaT * m_viewRT; 
	UpdateViewport();
#endif 
}


void ArcballCanvas::TranslateViewport(const Vector2i &p, const Vector2i &rel)
{
	Eigen::Vector2f nowPos = Eigen::Vector2f(p.x() + rel.x(), p.y() + rel.y());
	const Eigen::Vector3f camCenter = center;
	const Eigen::Vector3f camPos = pos;
	const Eigen::Vector3f camUp = up;
	const Eigen::Vector3f camRight = right;
	const Eigen::Vector3f camFront = front;
	Eigen::Vector2f _nowPos = nowPos;
	_nowPos(0) /= size().x();
	_nowPos(1) /= size().y();
	const Eigen::Vector3f nowArcCoord = GetArcballCoord(
		_nowPos * 2 - Eigen::Vector2f::Ones(), camFront, camUp, camRight);
	Eigen::Vector2f _before_pos;
	_before_pos(0) = p.x();
	_before_pos(1) = p.y();
	_before_pos(0) /= size().x();
	_before_pos(1) /= size().y();
	const Eigen::Vector3f beforeArcCoord = GetArcballCoord(
		_before_pos * 2 - Eigen::Vector2f::Ones(), camFront, camUp, camRight);

	const float distance = (camPos - camCenter).norm();
	Eigen::Vector3f nowCamcenter = camCenter + distance * (nowArcCoord - beforeArcCoord);
	SetExtrinsic(camPos, camUp, nowCamcenter);
#if 0
	float speed = 1e-3f;
	float x = rel[0] * speed;
	float y = rel[1] * speed;
	Eigen::Matrix4f deltaT; deltaT.setIdentity();
	deltaT.topRightCorner(3, 1) = Eigen::Vector3f(x, y, 0);
	m_viewRT = deltaT * m_viewRT;
	UpdateViewport();
#endif 
}

void ArcballCanvas::ZoomViewport(const Vector2i &p, const Vector2f &rel)
{
#if 0
	const float speed = 1e-1f;
	float shift = rel[1] * speed;
	Eigen::Matrix4f deltaT; deltaT.setIdentity();
	deltaT(2, 3) += shift;
	m_viewRT = deltaT * m_viewRT;
	UpdateViewport();
#endif 

	float sensitivity = 0.2f;

	const Eigen::Vector3f newPos = pos - sensitivity * float(rel.y()) * front;
	if ((newPos - center).dot(pos - center) > 0.0f)
	{
		SetExtrinsic(newPos, up, center);
	}
}


bool ArcballCanvas::mouse_drag_event(const Vector2i &p, const Vector2i &rel, int button, int modifiers)
{

	if (button == GLFW_MOUSE_BUTTON_LEFT + 1) 
	{ // left button rotate 
		RotateViewport(p, rel);
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT + 1) {
		TranslateViewport(p, rel);
	}
	return true;
}
