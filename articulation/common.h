#pragma once

enum BODY_PART
{
	NOT_BODY = 0,
	MAIN_BODY,
	HEAD,
	L_EAR,
	R_EAR,
	L_F_LEG,
	R_F_LEG,
	L_B_LEG,
	R_B_LEG,
	TAIL,
	OTHERS
};

struct CorrPair
{
	CorrPair() {
		target = -1;
		type = 0;
		index = 0;
		weight = 0;
	}
	int target;
	int type;
	int index;
	float weight;
};

