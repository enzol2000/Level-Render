// minimalistic code to draw a single triangle, this is not part of the API.

#include "shaderc/shaderc.h" // needed for compiling shaders at runtime
#ifdef _WIN32 // must use MT platform DLL libraries on windows
#pragma comment(lib, "shaderc_combined.lib") 
#endif

#define UP GW::MATH::GVECTORF {{{0,1,0,0}}} // to use on GATEWARE lookat function
#define PI G2D_PI 
#define MAX_SUBMESH_PER_DRAW 1024

#pragma region Vert Shader



// Simple Vertex Shader
const char* vertexShaderSource = R"(
// an ultra simple hlsl vertex shader



#pragma pack_matrix( row_major )



[[vk::push_constant]] 
cbuffer
	{
		uint model_ID;
		uint mesh_ID;
	};

struct OBJ_ATTRIBUTES
	{
		float3    Kd;		 // diffuse reflectivity
		float	   d;		 // dissolve (transparency)
		float3    Ks;		 // specular reflectivity
		float     Ns;		 // specular exponent
		float3    Ka;		 // ambient reflectivity
		float     sharpness; // local reflection map sharpness
		float3    Tf;		 // transmission filter
		float     Ni;		 // optical density (index of refraction)
		float3    Ke;		 // emissive reflectivity
		uint      illum;	 // illumination model
	};

struct SHADER_MODEL_DATA{

		//global share data

		float4 SunDiretion;
		float4 SunColor;
		matrix viewMatrix;
		matrix projectionMatrix;

		// per submesh transform and material data

		matrix matricies[1024];
		OBJ_ATTRIBUTES materials[1024];
};

struct OBJ_VERT_IN
{
	float3 pos : POSITION; // Left-handed +Z forward coordinate w not provided, assumed to be 1.
	float3 uvw : TEXCOORD; // D3D/Vulkan style top left 0,0 coordinate.
	float3 nrm : NORMAL; // Provided direct from obj file, may or may not be normalized.
};

struct RASTERIZER_OUT
{
	float4 posH : SV_POSITION; // Homogeneous projection space
	float3 nrmW : NORMAL; // normal in World Space (for lighting)
	float3 posW : WORLD; // position in World Space (for lighting)
};

StructuredBuffer<SHADER_MODEL_DATA> SceneData;

RASTERIZER_OUT main(OBJ_VERT_IN inputVertex)
{
	matrix world = SceneData[0].matricies[model_ID];

	RASTERIZER_OUT vout;

	vout.posW =  mul(float4(inputVertex.pos, 1.0f), world).xyz;
	vout.posH = float4(vout.posW , 1.0f);
	vout.nrmW =  mul(float4(inputVertex.nrm, 0.0f), world).xyz;
	matrix camera = mul(SceneData[0].viewMatrix, SceneData[0].projectionMatrix);
	vout.posH = mul(vout.posH, camera);

	return vout;


	//return float4(inputVertex.pos.x, inputVertex.pos.y, inputVertex.pos.z, 1);
}
)";
#pragma endregion


#pragma region Pixel Shader
// Simple Pixel Shader
const char* pixelShaderSource = R"(
// an ultra simple hlsl pixel shader

#pragma pack_matrix( row_major )


struct RASTERIZER_OUT
{
	float4 posH : SV_POSITION; // Homogeneous projection space
	float3 nrmW : NORMAL; // normal in World Space (for lighting)
	float3 posW : WORLD; // position in World Space (for lighting)
};


struct OBJ_ATTRIBUTES
	{
		float3    Kd;		 // diffuse reflectivity
		float	   d;		 // dissolve (transparency)
		float3    Ks;		 // specular reflectivity
		float     Ns;		 // specular exponent
		float3    Ka;		 // ambient reflectivity
		float     sharpness; // local reflection map sharpness
		float3    Tf;		 // transmission filter
		float     Ni;		 // optical density (index of refraction)
		float3    Ke;		 // emissive reflectivity
		uint      illum;	 // illumination model
	};

struct SHADER_MODEL_DATA{

		//global share data

		float4 SunDiretion;
		float4 SunColor;
		matrix viewMatrix;
		matrix projectionMatrix;

		// per submesh transform and material data

		matrix matricies[1024];
		OBJ_ATTRIBUTES materials[1024];
};

StructuredBuffer<SHADER_MODEL_DATA> SceneData;

[[vk::push_constant]] 
cbuffer
	{
		uint model_ID;
		uint mesh_ID;
	};

float4 main(RASTERIZER_OUT pin) : SV_TARGET 
{	
	
	float3 Normal = normalize(pin.nrmW);
	float3 Worldpos = pin.posW;
	float4 LightDir = SceneData[0].SunDiretion;
	float4 LightColor = SceneData[0].SunColor;
	float3 SpecularIntensity = SceneData[0].materials[mesh_ID].Ks;
	float SpecularExp = SceneData[0].materials[mesh_ID].Ns;
	float3 CameraPos = float3(SceneData[0].viewMatrix[3][0], SceneData[0].viewMatrix[3][1], SceneData[0].viewMatrix[3][2]);
	//float3 AmbientReflect = SceneData[0].materials[mesh_ID].Ka;



//           specular

	float3 ViewDir = normalize(CameraPos - Worldpos);

	float3 HalfVec = normalize((-LightDir) + ViewDir);

	float3 Intensity = max(pow(saturate( dot(Normal, HalfVec)), SpecularExp) , 0);

	float4 Reflect = float4(LightColor * Intensity * SpecularIntensity , 0);


// Light Ratio

	float4 Ambient = float4(0.25f, 0.25f, 0.35, 1);

    float4 LightRatio = saturate(dot(-LightDir, Normal));
	
	LightRatio = saturate(LightRatio + Ambient);

// SurfaceColor


	float4 SurfaceColor = float4(SceneData[0].materials[mesh_ID].Kd, SceneData[0].materials[mesh_ID].d);

// calc all

	float4 pout = (LightRatio * LightColor * SurfaceColor) + Reflect;

	float brightness = 0.05;

	pout.rgb += brightness;
	
	pout.w = 1;

	return pout;
}
)";
#pragma endregion

#include "Definitions.h"
#include "h2bParser.h" 
#include <fstream>
//#include <numeric>

struct Timer {
	std::chrono::time_point<std::chrono::steady_clock> start, end;
	std::chrono::duration<float> duration;

	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}
	void Start() {
		start = std::chrono::high_resolution_clock::now();
	}
	float GetDuration() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		return duration.count();
	}
};


struct Shader_Data {

	//global share data
	GW::MATH::GVECTORF SunDiretion, SunColor;
	GW::MATH::GMATRIXF viewMatrix, projectionMatrix;

	// per submesh transform and material data

	GW::MATH::GMATRIXF matricies[MAX_SUBMESH_PER_DRAW];
	OBJ_ATTRIBUTES materials[MAX_SUBMESH_PER_DRAW];
};

static OBJ_VEC3 VecToObjVec(H2B::VECTOR vec) {
	OBJ_VEC3 fvec;
	fvec.x = vec.x;
	fvec.y = vec.y;
	fvec.z = vec.z;
	return fvec;
}


static OBJ_ATTRIBUTES AttriToObjAttri(H2B::MATERIAL material) {
	OBJ_ATTRIBUTES Omaterial;
	Omaterial.d = material.attrib.d;
	Omaterial.illum = material.attrib.illum;
	Omaterial.Ka = VecToObjVec(material.attrib.Ka);
	Omaterial.Kd = VecToObjVec(material.attrib.Kd);
	Omaterial.Ke = VecToObjVec(material.attrib.Ke);
	Omaterial.Ks = VecToObjVec(material.attrib.Ks);
	Omaterial.Ni = material.attrib.Ni;
	Omaterial.Ns = material.attrib.Ns;
	Omaterial.sharpness = material.attrib.sharpness;

	return Omaterial;
}

#define MAX_MODEL_NAME_LENGTH 128

class Model {

	UINT ID;
	char Name[MAX_MODEL_NAME_LENGTH];

public:


	GW::MATH::GMATRIXF matrix;

	OBJ_ATTRIBUTES materials[MAX_SUBMESH_PER_DRAW];
	std::vector<H2B::MESH> meshes;

	UINT MaterialOffset = 0;

	//Shader_Data uniforms;
	H2B::Parser parser;
	VkBuffer vertexHandle = nullptr;
	VkBuffer indexHandle = nullptr;
	VkDeviceMemory vertexData = nullptr;
	VkDeviceMemory indexData = nullptr;

	UINT GetID() {
		return ID;
	}
	void SetID(UINT id) {
		ID = id;
	}
	char* GetName() {
		return Name;
	}
	void SetName(char* name) {
		strcpy(Name, name);
	}

	Model(const char* path) {
		parser = {};
		if (parser.Parse(path))
		{
			for (size_t i = 0; i < parser.materialCount; i++)
			{
				materials[i] = AttriToObjAttri(parser.materials[i]);

			}
			meshes = parser.meshes;
		}
	}

};

struct CBuffer {
	UINT model_ID;
	UINT mesh_ID;
};

class LevelData {
public:
	std::vector<Model> models;
	UINT modelCount = 0;
	GW::MATH::GMATRIXF CameraMatrix;
	GW::MATH::GMATRIXF LightMatrix;

	bool AddModel(Model Nmodel) {

		if (modelCount > 0)
		{
			Model PrevModel = models[modelCount - 1];

			Nmodel.meshes[0].materialIndex = PrevModel.meshes[PrevModel.parser.meshCount - 1].materialIndex + 1;

			for (size_t i = 1; i < Nmodel.meshes.size(); i++)
			{
				Nmodel.meshes[i].materialIndex = Nmodel.meshes[i - 1].materialIndex + 1;
			}
		}

		Nmodel.SetID(models.size());
		models.push_back(Nmodel);
		++modelCount;
		return true;
	}



};

// Creation, Rendering & Cleanup
class Renderer
{

	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GVulkanSurface vlk;
	GW::CORE::GEventReceiver shutdown;

	// what we need at a minimum to draw a triangle
	VkDevice device = nullptr;


	VkDescriptorSetLayout LayoutDescriptor = nullptr;
	VkDescriptorPool PoolDescriptor = nullptr;
	std::vector<VkDescriptorSet> SetDescriptor;

	std::vector<VkDeviceMemory>buffersData;
	std::vector<VkBuffer>buffersHandle;

	VkShaderModule vertexShader = nullptr;
	VkShaderModule pixelShader = nullptr;
	// pipeline settings for drawing (also required)
	VkPipeline pipeline = nullptr;
	VkPipelineLayout pipelineLayout = nullptr;
	GW::MATH::GMatrix math;

	Timer timer;
	LevelData level;
	//LevelDataa level2;
	//Level_Data dt;

	Shader_Data PixelShaderData;
	GW::INPUT::GInput InputKeyboard;
	GW::INPUT::GController  InputController;

	bool togglewalk = true;
	bool crouch = false;
	bool running = false;
	float CPressed = 0;
	float WalkPressed = 0;
	float F1Pressed = 0;


public:

	void ArrayToFloat(char* srcar, char* inarr, int size, size_t& offset) {

		for (size_t i = 0; offset < size; offset++, ++i)
		{
			if (srcar[offset] == ',' || srcar[offset] == ')')
			{
				++offset;
				break;
			}
			inarr[i] = srcar[offset];
		}
	}

	GW::MATH::GVECTORF ReadVec(char* srcar, int size) {

		GW::MATH::GVECTORF vec;

		char xfloat[100];
		char yfloat[100];
		char zfloat[100];
		char wfloat[100];
		for (size_t i = 0; i < size; i++)
		{
			if (srcar[i] == '(')
			{
				++i;
				ArrayToFloat(srcar, xfloat, size, i);
				ArrayToFloat(srcar, yfloat, size, i);
				ArrayToFloat(srcar, zfloat, size, i);
				ArrayToFloat(srcar, wfloat, size, i);
			}
		}
		vec.x = atof(xfloat);
		vec.y = atof(yfloat);
		vec.z = atof(zfloat);
		vec.w = atof(wfloat);
		return vec;
	}

	GW::MATH::GMATRIXF GetMatrixFromFile(std::ifstream& file) {

		GW::MATH::GMATRIXF matrix;

		char MatrixData[1024];
		file.getline(MatrixData, 1024, '\n');
		matrix.row1 = ReadVec(MatrixData, 1024);

		file.getline(MatrixData, 1024, '\n');
		matrix.row2 = ReadVec(MatrixData, 1024);

		file.getline(MatrixData, 1024, '\n');
		matrix.row3 = ReadVec(MatrixData, 1024);

		file.getline(MatrixData, 1024, '\n');
		matrix.row4 = ReadVec(MatrixData, 1024);


		return matrix;


	}

	// if ruuning .exe converter .obj to .h in cmd if there is a triangulate warning to modeling to triangulater in blander 
	

	void LoadLvl(char* lvlName) {

		std::ifstream file;

		char* levelPath = "../Render/level/";

		int pathsize = strlen(levelPath);
		int namesize = strlen(lvlName);

		std::vector<char>d;

		for (size_t i = 0; i < pathsize; i++)
		{
			d.push_back(levelPath[i]);
		}

		for (size_t i = 0; i < namesize; i++)
		{
			d.push_back(lvlName[i]);
		}
		d.push_back('\0');
		levelPath = d.data();
		file.open(levelPath);

		// get gamelevel txt

		char line[1024];

		if (file.is_open())
		{
			while (!file.eof())
			{
				file.getline(line, 1024, '\n');

				if (std::strcmp(line, "Light") == 0)
				{
					level.LightMatrix = GetMatrixFromFile(file);
				}

				if (std::strcmp(line, "Camera") == 0)
				{
					level.CameraMatrix = GetMatrixFromFile(file);
				}


				if (std::strcmp(line, "MESH") == 0)
				{
					char h2bNamePath[1024];
					file.getline(h2bNamePath, 1024, '\n');

					char h2bName[1024];
					std::copy(std::begin(h2bNamePath), std::end(h2bNamePath), std::begin(h2bName));


					for (size_t i = 0; i < 1024; i++)
					{
						if (h2bNamePath[i] == '\0' || h2bNamePath[i] == '.')
						{
							h2bNamePath[i] = '.';
							h2bNamePath[++i] = 'h';
							h2bNamePath[++i] = '2';
							h2bNamePath[++i] = 'b';
							h2bNamePath[++i] = '\0';
							break;
						}
					}
					char PathToh2b[1024] = "../Render/OBJS/h2b/";

					for (size_t i = 19, j = 0; i < 1024; i++, j++)
					{
						PathToh2b[i] = h2bNamePath[j];
					}

					Model model(PathToh2b);
					model.SetName(h2bName);

					model.matrix = GetMatrixFromFile(file);

					level.AddModel(model);
				}

			}
		}
		InicializeLevelBuffers();

	}


	void LoadNewLvl(char* lvlName) {
		CleanUp();

		Renderer render(win, vlk, lvlName);


		buffersData = render.buffersData;
		buffersHandle = render.buffersHandle;
		
		level = render.level;

		togglewalk = render.togglewalk;
		WalkPressed = render.WalkPressed;
		F1Pressed = render.F1Pressed;
		crouch = render.crouch;
		CPressed = render.CPressed;
		running = render.running;

		device =render.device;

		math = render.math;

		pipeline = render.pipeline;
		pipelineLayout = render.pipelineLayout;

		vlk = render.vlk;
		win = render.win;

		InputController = render.InputController;
		InputKeyboard = render.InputKeyboard;


		vertexShader = render.vertexShader;
		pixelShader = render.pixelShader;

		PixelShaderData = render.PixelShaderData;

		PoolDescriptor = render.PoolDescriptor;
		SetDescriptor = render.SetDescriptor;
		LayoutDescriptor = render.LayoutDescriptor;

		timer = render.timer;
	}

	void ClearLevel() {

		vkDeviceWaitIdle(device);
		for (size_t i = 0; i < level.modelCount; i++)
		{
			vkFreeMemory(device, level.models[i].vertexData, nullptr);
			vkFreeMemory(device, level.models[i].indexData, nullptr);

			vkDestroyBuffer(device, level.models[i].vertexHandle, nullptr);
			vkDestroyBuffer(device, level.models[i].indexHandle, nullptr);
		}
	}

	void InicializeLevelBuffers() {

		VkPhysicalDevice physicalDevice = nullptr;
		vlk.GetDevice((void**)&device);
		vlk.GetPhysicalDevice((void**)&physicalDevice);

		for (size_t i = 0; i < level.modelCount; i++)
		{
			PixelShaderData.matricies[i] = level.models[i].matrix;

			for (size_t j = 0; j < level.models[i].parser.materialCount; j++)
			{
				PixelShaderData.materials[level.models[i].meshes[j].materialIndex] = level.models[i].materials[j];
			}

			H2B::VERTEX* vertices = level.models[i].parser.vertices.data();

			int vertsize = sizeof(H2B::VERTEX) * level.models[i].parser.vertexCount;
			//int b = sizeof(vertices);


			GvkHelper::create_buffer(physicalDevice, device, vertsize,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &level.models[i].vertexHandle, &level.models[i].vertexData);
			GvkHelper::write_to_buffer(device, level.models[i].vertexData, vertices, vertsize);
			// Transfer triangle data to the vertex buffer. (staging would be prefered here)



			unsigned* indices = level.models[i].parser.indices.data();
			int indsize = sizeof(unsigned) * level.models[i].parser.indexCount;


			GvkHelper::create_buffer(physicalDevice, device, indsize,
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &level.models[i].indexHandle, &level.models[i].indexData);
			GvkHelper::write_to_buffer(device, level.models[i].indexData, indices, indsize);
		}
	}


	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GVulkanSurface _vlk, char * levelName)
	{
		win = _win;
		vlk = _vlk;
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);

		/***************** GEOMETRY INTIALIZATION ******************/
		// Grab the device & physical device so we can allocate some stuff
		VkPhysicalDevice physicalDevice = nullptr;
		vlk.GetDevice((void**)&device);
		vlk.GetPhysicalDevice((void**)&physicalDevice);


		LoadLvl(levelName);

		//constants
		GW::MATH::GMATRIXF view = { 0 };
		GW::MATH::GMATRIXF Projection = { 0 };

		//math.IdentityF(World);
		math.IdentityF(view);
		math.IdentityF(Projection);

		GW::MATH::GVECTORF CameraPos = { 0.035f, 0.46f, -0.658f, 1.0f };
		GW::MATH::GVECTORF LookAt = { 0.2f, 0.46f, 0.25f, 1.0f };


		GW::MATH::GVECTORF LightDirection = { -1.0f, -1.0f, 2.0f, 1.0f };
		GW::MATH::GVECTORF LightColor = { 0.9f, 0.9f, 1.0f, 1.0f };

		math.TranslateLocalF(view, CameraPos, view);

		float YFOV = 65 * (PI / 180);
		float Ratio = 0.0f;

		vlk.GetAspectRatio(Ratio);

		math.ProjectionVulkanLHF(YFOV, Ratio, 0.1f, 100.0f, Projection);






		math.LookAtLHF(CameraPos, LookAt, UP, view);


		PixelShaderData.SunColor = LightColor;
		PixelShaderData.SunDiretion = LightDirection;
		PixelShaderData.viewMatrix = view;
		PixelShaderData.projectionMatrix = Projection;


		unsigned int MaxFrames;

		vlk.GetSwapchainImageCount(MaxFrames);
		buffersData.resize(MaxFrames);
		buffersHandle.resize(MaxFrames);

		for (size_t i = 0; i < 2; i++)
		{
			GvkHelper::create_buffer(physicalDevice, device, sizeof(Shader_Data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffersHandle[i], &buffersData[i]);
			GvkHelper::write_to_buffer(device, buffersData[i], &PixelShaderData, sizeof(Shader_Data));
		}


		/***************** SHADER INTIALIZATION ******************/
		// Intialize runtime shader compiler HLSL -> SPIRV
		shaderc_compiler_t compiler = shaderc_compiler_initialize();
		shaderc_compile_options_t options = shaderc_compile_options_initialize();
		shaderc_compile_options_set_source_language(options, shaderc_source_language_hlsl);
		shaderc_compile_options_set_invert_y(options, false); 
#ifndef NDEBUG
		shaderc_compile_options_set_generate_debug_info(options);
#endif
		// Create Vertex Shader
		shaderc_compilation_result_t result = shaderc_compile_into_spv( // compile
			compiler, vertexShaderSource, strlen(vertexShaderSource),
			shaderc_vertex_shader, "main.vert", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Vertex Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &vertexShader);
		shaderc_result_release(result); // done
		// Create Pixel Shader
		result = shaderc_compile_into_spv( // compile
			compiler, pixelShaderSource, strlen(pixelShaderSource),
			shaderc_fragment_shader, "main.frag", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Pixel Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &pixelShader);
		shaderc_result_release(result); // done
		// Free runtime shader compiler resources
		shaderc_compile_options_release(options);
		shaderc_compiler_release(compiler);

		/***************** PIPELINE INTIALIZATION ******************/
		// Create Pipeline & Layout (Thanks Tiny!)
		VkRenderPass renderPass;
		vlk.GetRenderPass((void**)&renderPass);
		VkPipelineShaderStageCreateInfo stage_create_info[2] = {};
		// Create Stage Info for Vertex Shader
		stage_create_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stage_create_info[0].module = vertexShader;
		stage_create_info[0].pName = "main";
		// Create Stage Info for Fragment Shader
		stage_create_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stage_create_info[1].module = pixelShader;
		stage_create_info[1].pName = "main";
		// Assembly State
		VkPipelineInputAssemblyStateCreateInfo assembly_create_info = {};
		assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assembly_create_info.primitiveRestartEnable = false;

		// Vertex Input State
		VkVertexInputBindingDescription vertex_binding_description = {};
		vertex_binding_description.binding = 0;
		vertex_binding_description.stride = sizeof(OBJ_VERT);
		vertex_binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		VkVertexInputAttributeDescription vertex_attribute_description[3] = {
			{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(OBJ_VERT, pos) },
			{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(OBJ_VERT, uvw) },
			{ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(OBJ_VERT, nrm) }  //uv, normal, etc....
		};

		VkPipelineVertexInputStateCreateInfo input_vertex_info = {};
		input_vertex_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		input_vertex_info.vertexBindingDescriptionCount = 1;
		input_vertex_info.pVertexBindingDescriptions = &vertex_binding_description;
		input_vertex_info.vertexAttributeDescriptionCount = 3;
		input_vertex_info.pVertexAttributeDescriptions = vertex_attribute_description;
		// Viewport State (we still need to set this up even though we will overwrite the values)
		VkViewport viewport = {
			0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
		};
		VkRect2D scissor = { {0, 0}, {width, height} };
		VkPipelineViewportStateCreateInfo viewport_create_info = {};
		viewport_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_create_info.viewportCount = 1;
		viewport_create_info.pViewports = &viewport;
		viewport_create_info.scissorCount = 1;
		viewport_create_info.pScissors = &scissor;
		// Rasterizer State
		VkPipelineRasterizationStateCreateInfo rasterization_create_info = {};
		rasterization_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterization_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterization_create_info.lineWidth = 1.0f;
		rasterization_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterization_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterization_create_info.depthClampEnable = VK_FALSE;
		rasterization_create_info.depthBiasEnable = VK_FALSE;
		rasterization_create_info.depthBiasClamp = 0.0f;
		rasterization_create_info.depthBiasConstantFactor = 0.0f;
		rasterization_create_info.depthBiasSlopeFactor = 0.0f;
		// Multisampling State
		VkPipelineMultisampleStateCreateInfo multisample_create_info = {};
		multisample_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_create_info.sampleShadingEnable = VK_FALSE;
		multisample_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_create_info.minSampleShading = 1.0f;
		multisample_create_info.pSampleMask = VK_NULL_HANDLE;
		multisample_create_info.alphaToCoverageEnable = VK_FALSE;
		multisample_create_info.alphaToOneEnable = VK_FALSE;
		// Depth-Stencil State
		VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info = {};
		depth_stencil_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil_create_info.depthTestEnable = VK_TRUE;
		depth_stencil_create_info.depthWriteEnable = VK_TRUE;
		depth_stencil_create_info.depthCompareOp = VK_COMPARE_OP_LESS;
		depth_stencil_create_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_create_info.minDepthBounds = 0.0f;
		depth_stencil_create_info.maxDepthBounds = 1.0f;
		depth_stencil_create_info.stencilTestEnable = VK_FALSE;
		// Color Blending Attachment & State
		VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
		color_blend_attachment_state.colorWriteMask = 0xF;
		color_blend_attachment_state.blendEnable = VK_FALSE;
		color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR;
		color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
		color_blend_attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
		color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		color_blend_attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;
		VkPipelineColorBlendStateCreateInfo color_blend_create_info = {};
		color_blend_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend_create_info.logicOpEnable = VK_FALSE;
		color_blend_create_info.logicOp = VK_LOGIC_OP_COPY;
		color_blend_create_info.attachmentCount = 1;
		color_blend_create_info.pAttachments = &color_blend_attachment_state;
		color_blend_create_info.blendConstants[0] = 0.0f;
		color_blend_create_info.blendConstants[1] = 0.0f;
		color_blend_create_info.blendConstants[2] = 0.0f;
		color_blend_create_info.blendConstants[3] = 0.0f;
		// Dynamic State 
		VkDynamicState dynamic_state[2] = {
			// By setting these we do not need to re-create the pipeline on Resize
			VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamic_create_info = {};
		dynamic_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_create_info.dynamicStateCount = 2;
		dynamic_create_info.pDynamicStates = dynamic_state;


		VkPushConstantRange Crange;

		Crange.offset = 0;
		Crange.size = sizeof(unsigned int) * 2;
		Crange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;


#pragma region CreateSetLayout


		VkDescriptorSetLayoutBinding layout_binding = {};
		layout_binding.binding = 0;
		layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layout_binding.descriptorCount = 1;
		layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo Layout_Create_Info = {};
		Layout_Create_Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		Layout_Create_Info.bindingCount = 1;
		Layout_Create_Info.pBindings = &layout_binding;
		vkCreateDescriptorSetLayout(device, &Layout_Create_Info, nullptr, &LayoutDescriptor);


#pragma endregion

#pragma region CreatePool

		VkDescriptorPoolSize Pool_Size = {};
		Pool_Size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		Pool_Size.descriptorCount = MaxFrames;

		VkDescriptorPoolCreateInfo Pool_Create_Info = {};
		Pool_Create_Info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
		Pool_Create_Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		Pool_Create_Info.poolSizeCount = 1;
		Pool_Create_Info.pPoolSizes = &Pool_Size;
		Pool_Create_Info.maxSets = MaxFrames;

		vkCreateDescriptorPool(device, &Pool_Create_Info, nullptr, &PoolDescriptor);


#pragma endregion

#pragma region SetAllocate

		VkDescriptorSetAllocateInfo Set_Allocate_Info = {};
		Set_Allocate_Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		Set_Allocate_Info.descriptorPool = PoolDescriptor;
		Set_Allocate_Info.pSetLayouts = &LayoutDescriptor;
		Set_Allocate_Info.descriptorSetCount = 1;
		SetDescriptor.resize(MaxFrames);

#pragma endregion


#pragma region WriteDescriptorSet

		VkWriteDescriptorSet WriteSet = {};
		WriteSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteSet.dstBinding = 0;
		WriteSet.dstArrayElement = 0;
		WriteSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteSet.descriptorCount = 1;
		for (size_t i = 0; i < SetDescriptor.size(); i++)
		{
			vkAllocateDescriptorSets(device, &Set_Allocate_Info, &SetDescriptor[i]);

			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = buffersHandle[i];
			bufferInfo.offset = 0;
			bufferInfo.range = VK_WHOLE_SIZE;
			WriteSet.dstSet = SetDescriptor[i];
			WriteSet.pBufferInfo = &bufferInfo;

			vkUpdateDescriptorSets(device, 1, &WriteSet, 0, nullptr);
		}
		//  get int and floats form line fscanf
#pragma endregion

		// Descriptor pipeline layout
		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts = &LayoutDescriptor;

		pipeline_layout_create_info.pushConstantRangeCount = 1;
		pipeline_layout_create_info.pPushConstantRanges = &Crange;
		vkCreatePipelineLayout(device, &pipeline_layout_create_info,
			nullptr, &pipelineLayout);
		// Pipeline State... (FINALLY) 
		VkGraphicsPipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_create_info.stageCount = 2;
		pipeline_create_info.pStages = stage_create_info;
		pipeline_create_info.pInputAssemblyState = &assembly_create_info;
		pipeline_create_info.pVertexInputState = &input_vertex_info;
		pipeline_create_info.pViewportState = &viewport_create_info;
		pipeline_create_info.pRasterizationState = &rasterization_create_info;
		pipeline_create_info.pMultisampleState = &multisample_create_info;
		pipeline_create_info.pDepthStencilState = &depth_stencil_create_info;
		pipeline_create_info.pColorBlendState = &color_blend_create_info;
		pipeline_create_info.pDynamicState = &dynamic_create_info;
		pipeline_create_info.layout = pipelineLayout;
		pipeline_create_info.renderPass = renderPass;
		pipeline_create_info.subpass = 0;
		pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
		vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
			&pipeline_create_info, nullptr, &pipeline);

		//start math
		//starting Gateware math
		math.Create();
		
		/***************** CLEANUP / SHUTDOWN ******************/
		// GVulkanSurface will inform us when to release any allocated resources
		shutdown.Create(vlk, [&]() {
			if (+shutdown.Find(GW::GRAPHICS::GVulkanSurface::Events::RELEASE_RESOURCES, true)) {
				CleanUp(); // unlike D3D we must be careful about destroy timing
			}
			});
	}


	void UpdateCamera() {
		InputKeyboard.Create(win);
		InputController.Create();
		GW::MATH::GMATRIXF Camera = { 0, };

		math.InverseF(PixelShaderData.viewMatrix, Camera);

		float Space;
		float LeftShift;
		float LeftCtrl;
		float Wkey;
		float Skey;
		float Akey;
		float Dkey;
		float Ekey;
		float Ckey;
		float F1key;

		float RIGHT_TRIGGER = 0;
		float LEFT_TRIGGER = 0;

		float L_STICK_X_AXIS = 0;
		float L_STICK_Y_AXIS = 0;

		float R_STICK_Y_AXIS = 0;
		float R_STICK_X_AXIS = 0;

		GW::MATH::GVECTORF TranslateCamera = { 0, };
		GW::MATH::GVECTORF TranslateCameraOnY = { 0, };

		float Camera_Speed = 0.52f;
		

		InputKeyboard.GetState(G_KEY_SPACE, Space);
		InputKeyboard.GetState(G_KEY_LEFTSHIFT, LeftShift);
		InputKeyboard.GetState(G_KEY_LEFTCONTROL, LeftCtrl);
		InputKeyboard.GetState(G_KEY_W, Wkey);
		InputKeyboard.GetState(G_KEY_S, Skey);
		InputKeyboard.GetState(G_KEY_A, Akey);
		InputKeyboard.GetState(G_KEY_D, Dkey);
		InputKeyboard.GetState(G_KEY_E, Ekey);
		InputKeyboard.GetState(G_KEY_C, Ckey);
		InputKeyboard.GetState(G_KEY_F1, F1key);

		//GW::I::GXboxController
		//virtual void Release()

		if (F1key == 1 && F1Pressed != 1)
		{
			LoadNewLvl("Gamelevel2.txt");
		}

		// toggle crouch 

		if (LeftCtrl == 1)
		{
			Camera_Speed += 0.3f;
		}

		if (Ckey == 1 && CPressed != 1)
		{
			if (crouch)
			{
				crouch = false;
				Camera.row4 = { Camera.row4.x, Camera.row4.y + 0.20f, Camera.row4.z, Camera.row4.w };
			}
			else
			{
				crouch = true;
				Camera.row4 = { Camera.row4.x, Camera.row4.y - 0.20f, Camera.row4.z, Camera.row4.w };
			}
		}

		// toggle between walkmode and spectator mode 
		if (Ekey == 1 && WalkPressed != 1)
		{
			if (togglewalk)
			{
				Camera.row4 = { 0.2f, 0.46f, 0.0f, Camera.row4.w };
				togglewalk = false;
				crouch = false;
			}
			else
			{
				Camera.row4 = { 0.2f, 0.46f, 0.0f, Camera.row4.w };
				togglewalk = true;
			}
		}

		CPressed = Ckey;
		WalkPressed = Ekey;
		F1Pressed = F1key;

		//G_CODE_MAPPING_GENERAL
		InputController.GetState(G_CODE_MAPPING_GENERAL, G_LEFT_TRIGGER_AXIS, LEFT_TRIGGER);
		InputController.GetState(G_CODE_MAPPING_GENERAL, G_RIGHT_TRIGGER_AXIS, RIGHT_TRIGGER);

		InputController.GetState(G_CODE_MAPPING_GENERAL, G_LX_AXIS, L_STICK_X_AXIS);
		InputController.GetState(G_CODE_MAPPING_GENERAL, G_LY_AXIS, L_STICK_Y_AXIS);

		InputController.GetState(G_CODE_MAPPING_GENERAL, G_RY_AXIS, R_STICK_Y_AXIS);
		InputController.GetState(G_CODE_MAPPING_GENERAL, G_RX_AXIS, R_STICK_X_AXIS);

		float ChangeX = Dkey - Akey + L_STICK_X_AXIS;
		float ChangeY = Space - LeftShift + RIGHT_TRIGGER - LEFT_TRIGGER;

		float ChangeZ = Wkey - Skey + L_STICK_Y_AXIS;

		float PerFrameSpeed = Camera_Speed * timer.GetDuration();
		TranslateCameraOnY.y = ChangeY * PerFrameSpeed;
		TranslateCamera.x = ChangeX * PerFrameSpeed;
		TranslateCamera.z = ChangeZ * PerFrameSpeed;


		math.TranslateLocalF(Camera, TranslateCamera, Camera);
		math.TranslateGlobalF(Camera, TranslateCameraOnY, Camera);


		// rotate camera on X rotaion by the mouse

		float MouseYDelta;
		float MouseXDelta;
		GW::GReturn result = InputKeyboard.GetMouseDelta(MouseXDelta, MouseYDelta);
		float FOV = 65 * (PI / 180);
		const float Thumb_Speed = PI * timer.GetDuration();

		UINT Height;
		UINT Width;

		win.GetHeight(Height);
		win.GetWidth(Width);

		float Total_Pitch = 0;


		GW::MATH::GQUATERNIONF q;
		math.GetRotationF(Camera, q);

		if (result == GW::GReturn::SUCCESS && result != GW::GReturn::REDUNDANT)
		{
			if (q.x < -0.6f)
			{
				if (MouseYDelta < 0)
				{
					Total_Pitch = FOV * MouseYDelta / Height + R_STICK_Y_AXIS * Thumb_Speed;
				}
			}
			else if (q.x > 0.6f)
			{
				if (0 < MouseYDelta)
				{
					Total_Pitch = FOV * MouseYDelta / Height + R_STICK_Y_AXIS * Thumb_Speed;
				}
			}
			else
			{
				Total_Pitch = FOV * MouseYDelta / Height + R_STICK_Y_AXIS * Thumb_Speed;
			}
		}


		math.RotateXLocalF(Camera, Total_Pitch, Camera);



		float ratio;
		vlk.GetAspectRatio(ratio);
		float Total_Yaw;
		if (result == GW::GReturn::SUCCESS && result != GW::GReturn::REDUNDANT)
		{
			Total_Yaw = FOV * ratio * MouseXDelta / Width + R_STICK_X_AXIS * Thumb_Speed;
		}
		else
		{
			Total_Yaw = 0;
		}

		math.RotateYGlobalF(Camera, Total_Yaw, Camera);

		math.InverseF(Camera, PixelShaderData.viewMatrix);

		timer.Start();

	}


	void Render()
	{

		unsigned int currentBuffer;
		vlk.GetSwapchainCurrentImage(currentBuffer);


		// grab the current Vulkan commandBuffer
		VkCommandBuffer commandBuffer;
		vlk.GetCommandBuffer(currentBuffer, (void**)&commandBuffer);
		// what is the current client area dimensions?
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		// setup the pipeline's dynamic settings
		VkViewport viewport = {
			0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
		};
		VkRect2D scissor = { {0, 0}, {width, height} };
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &SetDescriptor[currentBuffer], 0, nullptr);
		// now we can draw 
		//const float RotSpeed = 10.0f;
		//float Rot = timer.GetDuration() * RotSpeed;
		//math.RotateYLocalF(PixelShaderData.matricies[2], Rot, PixelShaderData.matricies[2]);
		GvkHelper::write_to_buffer(device, buffersData[currentBuffer], &PixelShaderData, sizeof(Shader_Data));


		for (size_t i = 0; i < level.modelCount; i++)
		{
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &level.models[i].vertexHandle, offsets);
			vkCmdBindIndexBuffer(commandBuffer, level.models[i].indexHandle, *offsets, VK_INDEX_TYPE_UINT32);

			CBuffer cbuffer;
			cbuffer.model_ID = i;

			for (size_t j = 0; j < level.models[i].parser.meshCount; j++)
			{
				cbuffer.mesh_ID = level.models[i].meshes[j].materialIndex;
				UINT IndexCount = level.models[i].meshes[j].drawInfo.indexCount;
				UINT offset = level.models[i].meshes[j].drawInfo.indexOffset;


				vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CBuffer), &cbuffer);
				vkCmdDrawIndexed(commandBuffer, IndexCount, 1, offset, 0, 0);

			}
		}

		timer.Start();
	}

private:
	void CleanUp()
	{
		// wait till everything has completed
		vkDeviceWaitIdle(device);
		// Release allocated buffers, shaders & pipeline

		for (size_t i = 0; i < buffersHandle.size(); i++)
		{
			vkDestroyBuffer(device, buffersHandle[i], nullptr);
			vkFreeMemory(device, buffersData[i], nullptr);
		}

		// clean up handles and data for each model in level
		ClearLevel();

		vkDestroyShaderModule(device, vertexShader, nullptr);
		vkDestroyShaderModule(device, pixelShader, nullptr);

		vkDestroyDescriptorPool(device, PoolDescriptor, nullptr);
		vkDestroyDescriptorSetLayout(device, LayoutDescriptor, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
	}
};
