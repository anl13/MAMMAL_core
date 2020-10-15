#include "shader_file.h"

const std::string vs_vertex_color(
	// Vertex shader
	R"(#version 330
        uniform mat4 mvp;
        in vec4 positions;
        in vec4 colors;
        out vec4 frag_color;
        void main() {
            frag_color = colors;
            gl_Position = mvp * vec4(positions.xyz, 1.0);
        })"
);

const std::string fs_vertex_color(
	// Fragment shader
	R"(#version 330
        out vec4 color;
        in vec4 frag_color;
        void main() {
            color = frag_color;
        })"
);


const std::string vs_uniform_color(
	// Vertex shader
	R"(#version 330
        uniform mat4 mvp;
		uniform vec4 color; 
        in vec4 positions;
        out vec4 frag_color;
        void main() {
            frag_color = color;
            gl_Position = mvp * vec4(positions.xyz, 1.0);
        })"
);

const std::string fs_uniform_color(
	// Fragment shader
	R"(#version 330
        out vec4 color;
        in vec4 frag_color;
        void main() {
            color = vec4(frag_color.xyz, 1.0);
        })"
);


const std::string vs_texture_object(
	// Vertex shader
	R"(#version 330
        uniform mat4 mvp;
        in vec4 positions;
		in vec2 texture_coords;
		out vec2 texCoords;
        void main() {
			texCoords = texture_coords;
            gl_Position = mvp * vec4(positions.xyz, 1.0);
        })"
);


const std::string fs_texture_object(
	// Fragment shader
	R"(#version 330

		uniform sampler2D tex0;
		in vec2 texCoords; 
		out vec4 frag_color;
        void main() {
            frag_color = texture(tex0, texCoords);
        })"
);


const std::string vs_vertex_position(
	// Vertex shader
	R"(#version 330
        uniform mat4 mvp;
        uniform mat4 view;
        in vec4 positions;
		out vec4 positions_out;
        void main() {
			vec4 v = mvp * vec4(positions.xyz, 1.0);
			positions_out = view * vec4(positions.xyz, 1.0); 
            gl_Position = v;
        })"
);

const std::string fs_vertex_position(
	// Fragment shader
	R"(#version 330
		in vec4 positions_out;
		layout(location = 0) out vec4 frag_color;
        void main() {
            frag_color = positions_out;
        })"
);


const std::string vs_vertex_normal(
	// Vertex shader
	R"(#version 330
        uniform mat4 mvp;
        in vec4 positions;
		in vec4 normals;
		out vec4 normals_out;
        void main() {
			normals_out = vec4(normals.xyz, 1.0);
            gl_Position = mvp * vec4(positions.xyz, 1.0);
        })"
);

const std::string fs_vertex_normal(
	// Fragment shader
	R"(#version 330
		in vec4 normals_out;
		layout(location = 0) out vec4 frag_color;
        void main() {
            frag_color = normals_out;
        })"
);


/************************************************************************/
/* shaders for geometry rendering                                       */
/************************************************************************/
const std::string vs_phong_geometry(
	R"(
	#version 450 core
    uniform mat4 mvp;
	uniform mat4 model;
	uniform mat4 view;
	in vec4 positions;
	in vec4 normals;

	out VS_OUT
	{
		vec3 v;
		vec3 fn;
		vec3 bn;
	} vs_out;

	void main()
	{
		/*calculate vertex coordinates in camera frame*/
		mat4 T = view * model;
		vec4 v_cam = T * vec4(positions.xyz, 1.0);
		vs_out.v = v_cam.xyz;
	
		/*calculating front and back normal directions*/
		mat3 R = mat3(view) * mat3(model);
		vec3 front_normal = normalize(R * normals.xyz);
		vec3 back_normal = -front_normal;
		vs_out.fn = front_normal;
		vs_out.bn = back_normal;
	
		gl_Position = mvp * vec4(positions.xyz, 1.0);
	}
	)"
);

const std::string fs_phong_geometry(
	R"(
	#version 450 core

	struct LightAttrib
	{
		vec3 la;
		vec3 ld;
		vec3 ls;
		vec3 ldir;
	};
	
	struct MaterialAttrib
	{
		vec3 ma;
		vec3 md;
		vec3 ms;
		float ss;
	};
	
	in VS_OUT
	{
		vec3 v;
		vec3 fn;
		vec3 bn;
	} fs_in;
	
	out vec4 frag_color;
	
	void main()
	{
		/* init lighting, front material and back material */
		LightAttrib light = LightAttrib(
			vec3(0.3, 0.3, 0.3),
			vec3(0.7, 0.7, 0.7),
			vec3(1.0, 1.0, 1.0),
			vec3(0.0, 0.0, 1.0)
		);
		
		MaterialAttrib fmat = MaterialAttrib(
			vec3(0.63, 0.64, 0.85),
			vec3(0.63, 0.64, 0.85),
			vec3(0.12, 0.12, 0.12),
			10.0
		);

		MaterialAttrib bmat = MaterialAttrib(
			vec3(0.85, 0.85, 0.85),
			vec3(0.85, 0.85, 0.85),
			vec3(0.6, 0.6, 0.6),
			100.0
		);
		
		/*Calculate light, view, front-facing and back-facing normals*/
		vec3 ldir = normalize(light.ldir);
		vec3 fn = normalize(fs_in.fn);
		vec3 bn = normalize(fs_in.bn);
		vec3 vdir = normalize(-fs_in.v);
		vec3 frdir = reflect(-ldir, fn);
		vec3 brdir = reflect(-ldir, bn);
		
		/*discard this fragment if normal is NAN*/
		if (any(isnan(fn)) || any(isnan(bn))) discard;
		
		/*render double faces*/
		if (gl_FrontFacing) {
			/*calculate radiance*/
			vec3 ka = light.la * fmat.ma;
			vec3 kd = light.ld * fmat.md;
			vec3 ks = light.ls * fmat.ms;
		
			/*calculate Phong lighting of front-facing fragment*/
			vec3 fca = ka;
			vec3 fcd = kd * max(dot(fn, ldir), 0.0);
			vec3 fcs = ks * pow(max(dot(vdir, frdir), 0.0), fmat.ss);
		
			vec3 fc = clamp(fca + fcd + fcs, 0.0, 1.0);
			frag_color = vec4(fc, 1.0);
		}
		else {
			/*calculate radiance*/
			vec3 ka = light.la * bmat.ma;
			vec3 kd = light.ld * bmat.md;
			vec3 ks = light.ls * bmat.ms;
		
			/*calculate Phong lighting of back-facing fragment*/
			vec3 bca = ka;
			vec3 bcd = kd * max(dot(bn, ldir), 0.0);
			vec3 bcs = ks * pow(max(dot(vdir, brdir), 0.0), bmat.ss);
		
			vec3 bc = clamp(bca + bcd + bcs, 0.0, 1.0);
			frag_color = vec4(bc, 1.0);
		}
	}
	)"
);


// AN Liang 20200515:blinn phong shader for different color objects 
const std::string vs_phong_color(
	R"(
	#version 450 core
    uniform mat4 mvp;
	uniform mat4 model;
	uniform mat4 view;
    uniform vec4 incolor; 
	in vec4 positions;
	in vec4 normals;
	out vec4 material_color; 

	out VS_OUT
	{
		vec3 v;
		vec3 fn;
		vec3 bn;
	} vs_out;

	void main()
	{
		/*calculate vertex coordinates in camera frame*/
        material_color = incolor;
		mat4 T = view * model;
		vec4 v_cam = T * vec4(positions.xyz, 1.0);
		vs_out.v = v_cam.xyz;
	
		/*calculating front and back normal directions*/
		mat3 R = mat3(view) * mat3(model);
		vec3 front_normal = normalize(R * normals.xyz);
		vec3 back_normal = -front_normal;
		vs_out.fn = front_normal;
		vs_out.bn = back_normal;
	
		gl_Position = mvp * vec4(positions.xyz, 1.0);
	}
	)"
);
const std::string fs_phong_color(
	R"(
	#version 450 core

	struct LightAttrib
	{
		vec3 la;
		vec3 ld;
		vec3 ls;
		vec3 ldir;
	};
	
	struct MaterialAttrib
	{
		vec3 ma;
		vec3 md;
		vec3 ms;
		float ss;
	};
	
	in VS_OUT
	{
		vec3 v;
		vec3 fn;
		vec3 bn;
	} fs_in;
	in vec4 material_color; 
	
	out vec4 frag_color;
	
	void main()
	{
		/* init lighting, front material and back material */
		LightAttrib light = LightAttrib(
			vec3(0.3, 0.3, 0.3),
			vec3(0.7, 0.7, 0.7),
			vec3(1.0, 1.0, 1.0),
			vec3(0.0, 0.0, 1.0)
		);
		
		MaterialAttrib fmat = MaterialAttrib(
			vec3(0.63, 0.64, 0.85),
			vec3(0.63, 0.64, 0.85),
			vec3(0.12, 0.12, 0.12),
			10.0
		);

		MaterialAttrib bmat = MaterialAttrib(
			vec3(0.85, 0.85, 0.85),
			vec3(0.85, 0.85, 0.85),
			vec3(0.6, 0.6, 0.6),
			100.0
		);
		
		/*Calculate light, view, front-facing and back-facing normals*/
		vec3 ldir = normalize(light.ldir);
		vec3 fn = normalize(fs_in.fn);
		vec3 bn = normalize(fs_in.bn);
		vec3 vdir = normalize(-fs_in.v);
		vec3 frdir = reflect(-ldir, fn);
		vec3 brdir = reflect(-ldir, bn);
		
		/*discard this fragment if normal is NAN*/
		if (any(isnan(fn)) || any(isnan(bn))) discard;
		
		/*render double faces*/
		if (gl_FrontFacing) {
			/*calculate radiance*/
			//vec3 ka = light.la * fmat.ma;
			//vec3 kd = light.ld * fmat.md;
			//vec3 ks = light.ls * fmat.ms;
			vec3 ka = light.la * vec3(material_color);
			vec3 kd = light.ld * vec3(material_color);
			vec3 ks = light.ls * vec3(material_color);
		
			/*calculate Phong lighting of front-facing fragment*/
			vec3 fca = ka;
			vec3 fcd = kd * max(dot(fn, ldir), 0.0);
			vec3 fcs = ks * pow(max(dot(vdir, frdir), 0.0), fmat.ss);
		
			vec3 fc = clamp(fca + fcd + fcs, 0.0, 1.0);
			frag_color = vec4(fc, 1.0);
		}
		else {
			/*calculate radiance*/
			vec3 ka = light.la * bmat.ma;
			vec3 kd = light.ld * bmat.md;
			vec3 ks = light.ls * bmat.ms;
		
			/*calculate Phong lighting of back-facing fragment*/
			vec3 bca = ka;
			vec3 bcd = kd * max(dot(bn, ldir), 0.0);
			vec3 bcs = ks * pow(max(dot(vdir, brdir), 0.0), bmat.ss);
		
			vec3 bc = clamp(bca + bcd + bcs, 0.0, 1.0);
			frag_color = vec4(bc, 1.0);
		}
	}
	)"
);

// AN Liang 20200516:blinn phong shader for different color vertex 
const std::string vs_phong_vertex_color(
	R"(
	#version 450 core
    uniform mat4 mvp;
	uniform mat4 model;
	uniform mat4 view;
    in vec4 incolor; 
	in vec4 positions;
	in vec4 normals;
	out vec4 material_color; 

	out VS_OUT
	{
		vec3 v;
		vec3 fn;
		vec3 bn;
	} vs_out;

	void main()
	{
		/*calculate vertex coordinates in camera frame*/
        material_color = incolor;
		mat4 T = view * model;
		vec4 v_cam = T * vec4(positions.xyz, 1.0);
		vs_out.v = v_cam.xyz;
	
		/*calculating front and back normal directions*/
		mat3 R = mat3(view) * mat3(model);
		vec3 front_normal = normalize(R * normals.xyz);
		vec3 back_normal = -front_normal;
		vs_out.fn = front_normal;
		vs_out.bn = back_normal;
	
		gl_Position = mvp * vec4(positions.xyz, 1.0);
	}
	)"
);
const std::string fs_phong_vertex_color(
	R"(
	#version 450 core

	struct LightAttrib
	{
		vec3 la;
		vec3 ld;
		vec3 ls;
		vec3 ldir;
	};
	
	struct MaterialAttrib
	{
		vec3 ma;
		vec3 md;
		vec3 ms;
		float ss;
	};
	
	in VS_OUT
	{
		vec3 v;
		vec3 fn;
		vec3 bn;
	} fs_in;
	in vec4 material_color; 
	
	out vec4 frag_color;
	
	void main()
	{
		/* init lighting, front material and back material */
		LightAttrib light = LightAttrib(
			vec3(0.3, 0.3, 0.3),
			vec3(0.7, 0.7, 0.7),
			vec3(1.0, 1.0, 1.0),
			vec3(0.0, 0.0, 1.0)
		);
		
		MaterialAttrib fmat = MaterialAttrib(
			vec3(0.63, 0.64, 0.85),
			vec3(0.63, 0.64, 0.85),
			vec3(0.12, 0.12, 0.12),
			10.0
		);

		MaterialAttrib bmat = MaterialAttrib(
			vec3(0.85, 0.85, 0.85),
			vec3(0.85, 0.85, 0.85),
			vec3(0.6, 0.6, 0.6),
			100.0
		);
		
		/*Calculate light, view, front-facing and back-facing normals*/
		vec3 ldir = normalize(light.ldir);
		vec3 fn = normalize(fs_in.fn);
		vec3 bn = normalize(fs_in.bn);
		vec3 vdir = normalize(-fs_in.v);
		vec3 frdir = reflect(-ldir, fn);
		vec3 brdir = reflect(-ldir, bn);
		
		/*discard this fragment if normal is NAN*/
		if (any(isnan(fn)) || any(isnan(bn))) discard;
		
		/*render double faces*/
		if (gl_FrontFacing) {
			/*calculate radiance*/
			//vec3 ka = light.la * fmat.ma;
			//vec3 kd = light.ld * fmat.md;
			//vec3 ks = light.ls * fmat.ms;
			vec3 ka = light.la * vec3(material_color);
			vec3 kd = light.ld * vec3(material_color);
			vec3 ks = light.ls * vec3(material_color);
		
			/*calculate Phong lighting of front-facing fragment*/
			vec3 fca = ka;
			vec3 fcd = kd * max(dot(fn, ldir), 0.0);
			vec3 fcs = ks * pow(max(dot(vdir, frdir), 0.0), fmat.ss);
		
			vec3 fc = clamp(fca + fcd + fcs, 0.0, 1.0);
			frag_color = vec4(fc, 1.0);
		}
		else {
			/*calculate radiance*/
			//vec3 ka = light.la * bmat.ma;
			//vec3 kd = light.ld * bmat.md;
			//vec3 ks = light.ls * bmat.ms;
			vec3 ka = light.la * vec3(material_color);
			vec3 kd = light.ld * vec3(material_color);
			vec3 ks = light.ls * vec3(material_color);
		
			/*calculate Phong lighting of back-facing fragment*/
			vec3 bca = ka;
			vec3 bcd = kd * max(dot(bn, ldir), 0.0);
			vec3 bcs = ks * pow(max(dot(vdir, brdir), 0.0), bmat.ss);
		
			vec3 bc = clamp(bca + bcd + bcs, 0.0, 1.0);
			frag_color = vec4(bc, 1.0);
		}
	}
	)"
);