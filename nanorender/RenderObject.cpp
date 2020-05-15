#include "RenderObject.h"
#include <vector_functions.hpp>


void RenderObject::InitBuffers(const int& vnum)
{
	int buf_num = 1;
	std::vector<float4> vertices(vnum, make_float4(0, 0, 0, 1));
	set_buffer("positions", VariableType::Float32, { vertices.size(), 4 }, vertices.data());
	m_vertex_buffers.emplace(std::make_pair("positions", pcl::gpu::DeviceArray<float4>()));

	if (m_buffers.find("normals") != m_buffers.end())
	{
		++buf_num;
		std::vector<float4> normals(vnum, make_float4(0, 0, 0, 0));
		set_buffer("normals", VariableType::Float32, { normals.size(), 4 }, normals.data());
		m_vertex_buffers.emplace(std::make_pair("normals", pcl::gpu::DeviceArray<float4>()));
	}

	if (m_buffers.find("colors") != m_buffers.end())
	{
		++buf_num;
		std::vector<float4> colors(vnum, make_float4(1, 1, 1, 1));
		set_buffer("colors", VariableType::Float32, { colors.size(), 4 }, colors.data());
		m_vertex_buffers.emplace(std::make_pair("colors", pcl::gpu::DeviceArray<float4>()));
	}
	m_cuda_gl_resources.resize(buf_num);

	int tex_num = 0;
	for (int i = 0; i < GL_MAX_TEXTURE_IMAGE_UNITS; ++i)
	{
		const std::string tex_name_in_shader = "tex" + std::to_string(i);
		if (m_buffers.find(tex_name_in_shader) != m_buffers.end())
		{
			++tex_num;
			m_cuda_surfaces.emplace(std::make_pair(tex_name_in_shader, CudaSurfaceObject()));
		}
	}
	m_cuda_surf_resources.resize(tex_num);
}


void RenderObject::InitCudaMapping()
{
	m_mapped = true;
	///////////////////////////////////////
	// init cuda mapping for vertex buffers
	int idx = 0;
	for (auto& it : m_vertex_buffers)
	{
		//std::cout << it.first << std::endl;
		cudaGraphicsResource_t cuda_gl_resource;
		GLuint buffer_id = (GLuint)((uintptr_t)m_buffers[it.first].buffer);
		cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_gl_resource, buffer_id, cudaGraphicsRegisterFlagsWriteDiscard)); // write only

		void* dptr;
		size_t buf_size;
		cudaSafeCall(cudaGraphicsMapResources(1, &cuda_gl_resource));
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buf_size, cuda_gl_resource));
		cudaSafeCall(cudaGraphicsUnmapResources(1, &cuda_gl_resource));

		it.second = pcl::gpu::DeviceArray<float4>((float4*)dptr, buf_size / sizeof(float4));
		m_cuda_gl_resources[idx++] = cuda_gl_resource;
	}

	/////////////////////////////////
	// init cuda mapping for textures

	// initialize cudaResourceDesc for mapped cuda texture
	cudaResourceDesc m_cuda_res_desc;
	memset(&m_cuda_res_desc, 0, sizeof(cudaResourceDesc));
	m_cuda_res_desc.resType = cudaResourceTypeArray;

	idx = 0;
	for (auto& it : m_cuda_surfaces)
	{
		cudaGraphicsResource_t cuda_surf_resource;
		GLuint texture_handle = (GLuint)((uintptr_t)m_buffers[it.first].buffer);
		cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_surf_resource, texture_handle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
		cudaSafeCall(cudaGraphicsMapResources(1, &cuda_surf_resource));
		cudaArray_t array;
		cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&array, cuda_surf_resource, 0, 0));
		m_cuda_res_desc.res.array.array = array;
		cudaSurfaceObject_t mapped_cuda_surf;
		cudaSafeCall(cudaCreateSurfaceObject(&mapped_cuda_surf, &m_cuda_res_desc));
		cudaSafeCall(cudaGraphicsUnmapResources(1, &cuda_surf_resource));

		cudaGetChannelDesc(&it.second.channel_desc, array);
		it.second.cuarray = array;
		it.second.surf = mapped_cuda_surf;
		it.second.width = m_buffers[it.first].shape[0];
		it.second.height = m_buffers[it.first].shape[1];
		m_cuda_surf_resources[idx++] = cuda_surf_resource;
	}
}


std::map<std::string, pcl::gpu::DeviceArray<float4>> RenderObject::MapBuffers()
{
	if (!m_mapped) InitCudaMapping();
	cudaSafeCall(cudaGraphicsMapResources(m_cuda_gl_resources.size(), &m_cuda_gl_resources[0]));
	return m_vertex_buffers;
}


void RenderObject::UnmapBuffers(const int valid_vnum /*= -1*/)
{
	cudaSafeCall(cudaGraphicsUnmapResources(m_cuda_gl_resources.size(), &m_cuda_gl_resources[0]));
	if (valid_vnum != -1) // update the shape of vertex buffers
	{
		m_buffers["positions"].shape[0] = valid_vnum;
		if (m_buffers.find("normals") != m_buffers.end())
			m_buffers["normals"].shape[0] = valid_vnum;
		if (m_buffers.find("colors") != m_buffers.end())
			m_buffers["colors"].shape[0] = valid_vnum;
	}
}

std::map<std::string, CudaSurfaceObject> RenderObject::MapTextures()
{
	if (!m_mapped) InitCudaMapping();
	cudaSafeCall(cudaGraphicsMapResources(m_cuda_surf_resources.size(), &m_cuda_surf_resources[0]));
	return m_cuda_surfaces;
}

void RenderObject::UnmapTextures()
{
	cudaSafeCall(cudaGraphicsUnmapResources(m_cuda_surf_resources.size(), &m_cuda_surf_resources[0]));
}

void RenderObject::SetTexture(const std::string& tex_name, const CudaSurfaceObject& src)
{
	MapTextures();

	CudaSurfaceObject& tar = m_cuda_surfaces[tex_name];
	if (tar.width != src.width || tar.height != src.height)
	{
		throw std::runtime_error("RenderObject::SetTexture: texture size not match!!!\n");
	}

	if (tar.channel_desc.x != src.channel_desc.x
		|| tar.channel_desc.y != src.channel_desc.y
		|| tar.channel_desc.z != src.channel_desc.z
		|| tar.channel_desc.w != src.channel_desc.w
		|| tar.channel_desc.f != src.channel_desc.f)
	{
		throw std::runtime_error("RenderObject::SetTexture: texture channel description not match!!!\n");
	}

	int elem_bytes = (src.channel_desc.x + src.channel_desc.y + src.channel_desc.z + src.channel_desc.w) / 8;
	cudaSafeCall(cudaMemcpy2DArrayToArray(tar.cuarray, 0, 0, src.cuarray, 0, 0, src.width * elem_bytes, src.height, cudaMemcpyDeviceToDevice));
	
	UnmapTextures();
}

void RenderObject::SetBuffer(const std::string& buf_name, const pcl::gpu::DeviceArray<float4>& vertex_buffer)
{
	const auto& it = m_vertex_buffers.find(buf_name);
	if (it == m_vertex_buffers.end()){
		throw std::runtime_error("RenderObject " + m_name + ": Cannot find " + buf_name + " buffer!!!\n");
	}

	MapBuffers();
	if (it->second.size() < vertex_buffer.size())
	{
		UnmapBuffers();
		InitBuffers(vertex_buffer.size() * 2);
		MapBuffers();
	}
	cudaSafeCall(cudaMemcpy(it->second.ptr(), vertex_buffer.ptr(), vertex_buffer.size() * sizeof(float4), cudaMemcpyDeviceToDevice));
	UnmapBuffers(vertex_buffer.size());
}