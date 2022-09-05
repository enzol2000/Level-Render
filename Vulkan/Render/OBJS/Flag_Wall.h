// "Flag_Wall.h" generated by "Obj2Header.exe" [Version 1.9d] Author: L.Norri Fullsail University.
// Data is converted to left-handed coordinate system and UV data is V flipped for Direct3D/Vulkan.
// The companion file "Flag_Wall.h2b" is a binary dump of this format(-padding) for more flexibility.
// Loading *.h2b: read version, sizes, handle strings(max len 260) by reading until null-terminated.
/***********************/
/*  Generator version  */
/***********************/
#ifndef _Flag_Wall_version_
const char Flag_Wall_version[4] = { '0','1','9','d' };
#define _Flag_Wall_version_
#endif
/************************************************/
/*  This section contains the model's size data */
/************************************************/
#ifndef _Flag_Wall_vertexcount_
const unsigned Flag_Wall_vertexcount = 202;
#define _Flag_Wall_vertexcount_
#endif
#ifndef _Flag_Wall_indexcount_
const unsigned Flag_Wall_indexcount = 612;
#define _Flag_Wall_indexcount_
#endif
#ifndef _Flag_Wall_materialcount_
const unsigned Flag_Wall_materialcount = 1; // can be used for batched draws
#define _Flag_Wall_materialcount_
#endif
#ifndef _Flag_Wall_meshcount_
const unsigned Flag_Wall_meshcount = 1;
#define _Flag_Wall_meshcount_
#endif
/************************************************/
/*  This section contains the raw data to load  */
/************************************************/
#ifndef __OBJ_VEC3__
typedef struct _OBJ_VEC3_
{
	float x,y,z; // 3D Coordinate.
}OBJ_VEC3;
#define __OBJ_VEC3__
#endif
#ifndef __OBJ_VERT__
typedef struct _OBJ_VERT_
{
	OBJ_VEC3 pos; // Left-handed +Z forward coordinate w not provided, assumed to be 1.
	OBJ_VEC3 uvw; // D3D/Vulkan style top left 0,0 coordinate.
	OBJ_VEC3 nrm; // Provided direct from obj file, may or may not be normalized.
}OBJ_VERT;
#define __OBJ_VERT__
#endif
#ifndef _Flag_Wall_vertices_
// Raw Vertex Data follows: Positions, Texture Coordinates and Normals.
const OBJ_VERT Flag_Wall_vertices[202] =
{
	{	{ -0.127718f, 0.816885f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.158067f, 0.810784f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.046600f, -0.000600f, -0.998900f }	},
	{	{ -0.183597f, 0.988900f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.059300f, -0.000200f, -0.998200f }	},
	{	{ -0.135066f, 0.999897f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.288073f, 0.816471f, -0.181631f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.059800f, -0.001100f, -0.998200f }	},
	{	{ -0.285094f, 1.021706f, -0.182066f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.067300f, 0.006600f, -0.997700f }	},
	{	{ -0.271971f, 1.971537f, -0.184274f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.442786f, 1.971537f, -0.178947f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.461733f, 1.971537f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.254408f, 1.971537f, 0.112633f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.217093f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.147411f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.278934f, 1.194680f, -0.179626f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.095900f, -0.002400f, -0.995400f }	},
	{	{ -0.348916f, 1.215000f, -0.173986f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.015800f, 0.000700f, -0.999900f }	},
	{	{ -0.450575f, 1.487243f, -0.178947f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.030900f, -0.011800f, -0.999500f }	},
	{	{ -0.300678f, 1.500884f, -0.183608f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.079000f, -0.012500f, -0.996800f }	},
	{	{ -0.179147f, 1.171900f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.125400f, -0.000400f, -0.992100f }	},
	{	{ -0.205369f, 1.509169f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.119400f, -0.011300f, -0.992800f }	},
	{	{ 0.000000f, 1.237725f, -0.203144f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.002500f, -1.000000f }	},
	{	{ -0.081732f, 1.221732f, -0.203072f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.001200f, 0.002200f, -1.000000f }	},
	{	{ -0.073543f, 1.456440f, -0.202240f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.003600f, 0.001700f, -1.000000f }	},
	{	{ 0.000000f, 1.396044f, -0.202643f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.002800f, -1.000000f }	},
	{	{ -0.137150f, 1.676102f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.015500f, -0.004000f, -0.999900f }	},
	{	{ -0.221166f, 1.668251f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.014200f, -0.003600f, -0.999900f }	},
	{	{ -0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.029400f, -0.007500f, -0.999500f }	},
	{	{ -0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.029400f, -0.007500f, -0.999500f }	},
	{	{ -0.146120f, 1.197559f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.179147f, 1.171900f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.205369f, 1.509169f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.141110f, 1.466400f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.270600f, 0.962700f, 0.007700f }	},
	{	{ -0.271971f, 1.971537f, -0.184274f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.270600f, 0.962700f, 0.007700f }	},
	{	{ -0.254408f, 1.971537f, 0.112633f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.270600f, 0.962700f, 0.007700f }	},
	{	{ -0.217093f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.270600f, 0.962700f, 0.007700f }	},
	{	{ 0.000000f, 0.995624f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.000600f, -1.000000f }	},
	{	{ -0.072544f, 1.006148f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000300f, 0.000700f, -1.000000f }	},
	{	{ -0.221166f, 1.668251f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.129200f, 0.000200f, -0.991600f }	},
	{	{ -0.293907f, 1.661930f, -0.184851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.081800f, -0.000400f, -0.996600f }	},
	{	{ -0.271971f, 1.971537f, -0.184274f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.095900f, 0.004900f, -0.995400f }	},
	{	{ -0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.144400f, 0.007100f, -0.989500f }	},
	{	{ -0.066120f, 0.820950f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.000000f, 0.828637f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.000000f, 0.588097f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.079885f, 0.597837f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.169709f, 0.590184f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.049000f, 0.002700f, -0.998800f }	},
	{	{ -0.301469f, 0.596348f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.057900f, 0.002600f, -0.998300f }	},
	{	{ -0.433275f, 0.828748f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.018900f, -0.000700f, -0.999800f }	},
	{	{ -0.433275f, 0.604516f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.017100f, -0.000300f, -0.999900f }	},
	{	{ -0.433275f, 1.086335f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.001300f, 0.020000f, -0.999800f }	},
	{	{ -0.131758f, 0.589036f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.082634f, 1.676852f, -0.202018f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.147400f, -0.010600f, -0.989000f }	},
	{	{ -0.137150f, 1.676102f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.147800f, -0.011000f, -0.989000f }	},
	{	{ -0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.161400f, -0.021000f, -0.986700f }	},
	{	{ -0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.161400f, -0.021000f, -0.986700f }	},
	{	{ -0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.448800f, 0.893600f, 0.006000f }	},
	{	{ -0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.448800f, 0.893600f, 0.006000f }	},
	{	{ -0.147411f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.448800f, 0.893600f, 0.006000f }	},
	{	{ -0.135459f, 1.978261f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.448800f, 0.893600f, 0.006000f }	},
	{	{ -0.066120f, 0.820950f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.158900f, 0.000100f, -0.987300f }	},
	{	{ -0.127718f, 0.816885f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.159500f, 0.000700f, -0.987200f }	},
	{	{ -0.135066f, 0.999897f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.148400f, -0.006000f, -0.988900f }	},
	{	{ -0.072544f, 1.006148f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.148200f, -0.006000f, -0.988900f }	},
	{	{ 0.000000f, 1.971537f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.999900f, -0.010200f }	},
	{	{ -0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.026600f, 0.999600f, -0.010200f }	},
	{	{ -0.135459f, 1.978261f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.026600f, 0.999600f, -0.010200f }	},
	{	{ 0.000000f, 1.971537f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.999900f, -0.010200f }	},
	{	{ -0.081732f, 1.221732f, -0.203072f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.137500f, 0.000700f, -0.990500f }	},
	{	{ -0.146120f, 1.197559f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.140200f, -0.001800f, -0.990100f }	},
	{	{ -0.141110f, 1.466400f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.133700f, 0.001800f, -0.991000f }	},
	{	{ -0.073543f, 1.456440f, -0.202240f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.133600f, 0.002300f, -0.991000f }	},
	{	{ 0.000000f, 1.687770f, -0.202314f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.001900f, -1.000000f }	},
	{	{ -0.082634f, 1.676852f, -0.202018f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.002800f, -0.001600f, -1.000000f }	},
	{	{ -0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.001300f, -0.004400f, -1.000000f }	},
	{	{ 0.000000f, 1.971537f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, -0.004400f, -1.000000f }	},
	{	{ -0.079885f, 0.597837f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.171900f, 0.007900f, -0.985100f }	},
	{	{ -0.131758f, 0.589036f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.173400f, 0.008300f, -0.984800f }	},
	{	{ -0.444103f, 1.665839f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.035100f, -0.002300f, -0.999400f }	},
	{	{ -0.441359f, 1.683357f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.035100f, 0.002200f, -0.999400f }	},
	{	{ -0.442786f, 1.971537f, -0.178947f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.035100f, 0.002200f, -0.999400f }	},
	{	{ -0.445850f, 1.195912f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.050900f, 0.000800f, -0.998700f }	},
	{	{ -0.433275f, 1.162486f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.009600f, 0.032200f, -0.999400f }	},
	{	{ -0.169709f, -0.031176f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.047800f, -0.003200f, -0.998900f }	},
	{	{ -0.301469f, -0.025013f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.063600f, -0.003700f, -0.998000f }	},
	{	{ -0.301469f, 0.178496f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.065400f, -0.001600f, -0.997900f }	},
	{	{ -0.195785f, 0.160544f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.058800f, -0.002100f, -0.998300f }	},
	{	{ -0.089474f, 0.395568f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.176200f, 0.005100f, -0.984300f }	},
	{	{ -0.143657f, 0.390617f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.175800f, 0.004400f, -0.984400f }	},
	{	{ -0.196183f, 0.383503f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.059100f, 0.001700f, -0.998200f }	},
	{	{ -0.301469f, 0.401428f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.064400f, 0.001900f, -0.997900f }	},
	{	{ -0.433275f, -0.016845f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.016500f, 0.000000f, -0.999900f }	},
	{	{ -0.433275f, 0.186664f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.016500f, 0.000000f, -0.999900f }	},
	{	{ -0.433275f, 0.409596f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.016500f, 0.000000f, -0.999900f }	},
	{	{ 0.000000f, -0.033264f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.079885f, -0.023524f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.089406f, 0.172552f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.000000f, 0.166880f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.000000f, 0.389892f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.089474f, 0.395568f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.131758f, -0.032324f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.143747f, 0.167591f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ -0.079885f, -0.023524f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.175900f, -0.009600f, -0.984400f }	},
	{	{ -0.131758f, -0.032324f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.175900f, -0.009600f, -0.984400f }	},
	{	{ -0.143747f, 0.167591f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.174700f, -0.005000f, -0.984600f }	},
	{	{ -0.089406f, 0.172552f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.174600f, -0.004600f, -0.984600f }	},
	{	{ -0.143657f, 0.390617f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.183597f, 0.988900f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.059300f, -0.000200f, -0.998200f }	},
	{	{ 0.158067f, 0.810784f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.046600f, -0.000600f, -0.998900f }	},
	{	{ 0.127718f, 0.816885f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.135066f, 0.999897f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.285094f, 1.021706f, -0.182066f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.067300f, 0.006600f, -0.997700f }	},
	{	{ 0.288073f, 0.816471f, -0.181631f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.059800f, -0.001100f, -0.998200f }	},
	{	{ 0.461733f, 1.971537f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.442786f, 1.971537f, -0.178947f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.271971f, 1.971537f, -0.184274f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.254408f, 1.971537f, 0.112633f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.217093f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.147411f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.450575f, 1.487243f, -0.178947f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.030900f, -0.011800f, -0.999500f }	},
	{	{ 0.348916f, 1.215000f, -0.173986f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.015800f, 0.000700f, -0.999900f }	},
	{	{ 0.278934f, 1.194680f, -0.179626f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.095900f, -0.002400f, -0.995400f }	},
	{	{ 0.300678f, 1.500884f, -0.183608f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.079000f, -0.012500f, -0.996800f }	},
	{	{ 0.179147f, 1.171900f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.125400f, -0.000400f, -0.992100f }	},
	{	{ 0.205369f, 1.509169f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.119400f, -0.011300f, -0.992800f }	},
	{	{ 0.073543f, 1.456440f, -0.202240f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.003600f, 0.001700f, -1.000000f }	},
	{	{ 0.081732f, 1.221732f, -0.203072f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.001200f, 0.002200f, -1.000000f }	},
	{	{ 0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.029400f, -0.007500f, -0.999500f }	},
	{	{ 0.221166f, 1.668251f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.014200f, -0.003600f, -0.999900f }	},
	{	{ 0.137150f, 1.676102f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.015500f, -0.004000f, -0.999900f }	},
	{	{ 0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.029400f, -0.007500f, -0.999500f }	},
	{	{ 0.205369f, 1.509169f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.179147f, 1.171900f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.146120f, 1.197559f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.141110f, 1.466400f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.254408f, 1.971537f, 0.112633f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.270600f, 0.962700f, 0.007700f }	},
	{	{ 0.271971f, 1.971537f, -0.184274f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.270600f, 0.962700f, 0.007700f }	},
	{	{ 0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.270600f, 0.962700f, 0.007700f }	},
	{	{ 0.217093f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.270600f, 0.962700f, 0.007700f }	},
	{	{ 0.072544f, 1.006148f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000300f, 0.000700f, -1.000000f }	},
	{	{ 0.271971f, 1.971537f, -0.184274f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.095900f, 0.004900f, -0.995400f }	},
	{	{ 0.293907f, 1.661930f, -0.184851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.081800f, -0.000400f, -0.996600f }	},
	{	{ 0.221166f, 1.668251f, -0.193852f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.129200f, 0.000200f, -0.991600f }	},
	{	{ 0.216476f, 1.984662f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.144400f, 0.007100f, -0.989500f }	},
	{	{ 0.066120f, 0.820950f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.079885f, 0.597837f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.169709f, 0.590184f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.049000f, 0.002700f, -0.998800f }	},
	{	{ 0.301469f, 0.596348f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.057900f, 0.002600f, -0.998300f }	},
	{	{ 0.433275f, 0.828748f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.018900f, -0.000700f, -0.999800f }	},
	{	{ 0.433275f, 0.604516f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.017100f, -0.000300f, -0.999900f }	},
	{	{ 0.433275f, 1.086335f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.001300f, 0.020000f, -0.999800f }	},
	{	{ 0.131758f, 0.589036f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.161400f, -0.021000f, -0.986700f }	},
	{	{ 0.137150f, 1.676102f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.147800f, -0.011000f, -0.989000f }	},
	{	{ 0.082634f, 1.676852f, -0.202018f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.147400f, -0.010600f, -0.989000f }	},
	{	{ 0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.161400f, -0.021000f, -0.986700f }	},
	{	{ 0.147411f, 1.984662f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.448800f, 0.893600f, 0.006000f }	},
	{	{ 0.144194f, 1.984662f, -0.198501f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.448800f, 0.893600f, 0.006000f }	},
	{	{ 0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.448800f, 0.893600f, 0.006000f }	},
	{	{ 0.135459f, 1.978261f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.448800f, 0.893600f, 0.006000f }	},
	{	{ 0.135066f, 0.999897f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.148400f, -0.006000f, -0.988900f }	},
	{	{ 0.127718f, 0.816885f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.159500f, 0.000700f, -0.987200f }	},
	{	{ 0.066120f, 0.820950f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.158900f, 0.000100f, -0.987300f }	},
	{	{ 0.072544f, 1.006148f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.148200f, -0.006000f, -0.988900f }	},
	{	{ 0.135459f, 1.978261f, 0.102243f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.026600f, 0.999600f, -0.010200f }	},
	{	{ 0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.026600f, 0.999600f, -0.010200f }	},
	{	{ 0.141110f, 1.466400f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.133700f, 0.001800f, -0.991000f }	},
	{	{ 0.146120f, 1.197559f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.140200f, -0.001800f, -0.990100f }	},
	{	{ 0.081732f, 1.221732f, -0.203072f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.137500f, 0.000700f, -0.990500f }	},
	{	{ 0.073543f, 1.456440f, -0.202240f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.133600f, 0.002300f, -0.991000f }	},
	{	{ 0.117203f, 1.971537f, -0.203401f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.001300f, -0.004400f, -1.000000f }	},
	{	{ 0.082634f, 1.676852f, -0.202018f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.002800f, -0.001600f, -1.000000f }	},
	{	{ 0.079885f, 0.597837f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.171900f, 0.007900f, -0.985100f }	},
	{	{ 0.131758f, 0.589036f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.173400f, 0.008300f, -0.984800f }	},
	{	{ 0.444103f, 1.665839f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.035100f, -0.002300f, -0.999400f }	},
	{	{ 0.442786f, 1.971537f, -0.178947f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.035100f, 0.002200f, -0.999400f }	},
	{	{ 0.441359f, 1.683357f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.035100f, 0.002200f, -0.999400f }	},
	{	{ 0.445850f, 1.195912f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.050900f, 0.000800f, -0.998700f }	},
	{	{ 0.433275f, 1.162486f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.009600f, 0.032200f, -0.999400f }	},
	{	{ 0.301469f, 0.178496f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.065400f, -0.001600f, -0.997900f }	},
	{	{ 0.301469f, -0.025013f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.063600f, -0.003700f, -0.998000f }	},
	{	{ 0.169709f, -0.031176f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.047800f, -0.003200f, -0.998900f }	},
	{	{ 0.195785f, 0.160544f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.058800f, -0.002100f, -0.998300f }	},
	{	{ 0.143657f, 0.390617f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.175800f, 0.004400f, -0.984400f }	},
	{	{ 0.089474f, 0.395568f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.176200f, 0.005100f, -0.984300f }	},
	{	{ 0.301469f, 0.401428f, -0.181121f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.064400f, 0.001900f, -0.997900f }	},
	{	{ 0.196183f, 0.383503f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.059100f, 0.001700f, -0.998200f }	},
	{	{ 0.433275f, 0.186664f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.016500f, 0.000000f, -0.999900f }	},
	{	{ 0.433275f, -0.016845f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.016500f, 0.000000f, -0.999900f }	},
	{	{ 0.433275f, 0.409596f, -0.178946f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.016500f, 0.000000f, -0.999900f }	},
	{	{ 0.089406f, 0.172552f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.079885f, -0.023524f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.089474f, 0.395568f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.131758f, -0.032324f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.143747f, 0.167591f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
	{	{ 0.143747f, 0.167591f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.174700f, -0.005000f, -0.984600f }	},
	{	{ 0.131758f, -0.032324f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.175900f, -0.009600f, -0.984400f }	},
	{	{ 0.079885f, -0.023524f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.175900f, -0.009600f, -0.984400f }	},
	{	{ 0.089406f, 0.172552f, -0.203408f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.174600f, -0.004600f, -0.984600f }	},
	{	{ 0.143657f, 0.390617f, -0.193851f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.000000f, -1.000000f }	},
};
#define _Flag_Wall_vertices_
#endif
#ifndef _Flag_Wall_indices_
// Index Data follows: Sequential by mesh, Every Three Indices == One Triangle.
const unsigned int Flag_Wall_indices[612] =
{
	 0, 1, 2,
	 3, 0, 2,
	 1, 4, 5,
	 2, 1, 5,
	 6, 7, 8,
	 9, 6, 8,
	 10, 11, 12,
	 13, 10, 12,
	 14, 15, 16,
	 17, 14, 16,
	 18, 14, 17,
	 19, 18, 17,
	 20, 21, 22,
	 23, 20, 22,
	 24, 25, 26,
	 27, 24, 26,
	 28, 29, 30,
	 31, 28, 30,
	 32, 33, 34,
	 35, 32, 34,
	 20, 36, 37,
	 21, 20, 37,
	 18, 2, 5,
	 14, 18, 5,
	 38, 39, 40,
	 41, 38, 40,
	 42, 43, 44,
	 45, 42, 44,
	 4, 1, 46,
	 47, 4, 46,
	 48, 4, 47,
	 49, 48, 47,
	 4, 48, 50,
	 5, 4, 50,
	 1, 0, 51,
	 46, 1, 51,
	 28, 3, 2,
	 29, 28, 2,
	 52, 53, 54,
	 55, 52, 54,
	 56, 57, 58,
	 59, 56, 58,
	 60, 61, 62,
	 63, 60, 62,
	 43, 42, 37,
	 36, 43, 37,
	 64, 65, 66,
	 67, 64, 66,
	 68, 69, 70,
	 71, 68, 70,
	 72, 73, 74,
	 75, 72, 74,
	 61, 60, 76,
	 77, 61, 76,
	 68, 63, 62,
	 69, 68, 62,
	 73, 72, 23,
	 22, 73, 23,
	 39, 38, 19,
	 17, 39, 19,
	 78, 39, 17,
	 16, 78, 17,
	 25, 24, 31,
	 30, 25, 31,
	 53, 52, 71,
	 70, 53, 71,
	 39, 79, 80,
	 40, 39, 80,
	 15, 14, 5,
	 15, 81, 16,
	 50, 82, 15,
	 5, 50, 15,
	 83, 84, 85,
	 86, 83, 85,
	 87, 88, 77,
	 76, 87, 77,
	 89, 90, 47,
	 46, 89, 47,
	 84, 91, 92,
	 85, 84, 92,
	 90, 93, 49,
	 47, 90, 49,
	 94, 95, 96,
	 97, 94, 96,
	 98, 99, 45,
	 44, 98, 45,
	 100, 83, 86,
	 101, 100, 86,
	 102, 103, 104,
	 105, 102, 104,
	 93, 90, 85,
	 92, 93, 85,
	 99, 98, 97,
	 96, 99, 97,
	 90, 89, 86,
	 85, 90, 86,
	 88, 87, 105,
	 104, 88, 105,
	 89, 106, 101,
	 86, 89, 101,
	 106, 89, 46,
	 51, 106, 46,
	 107, 108, 109,
	 110, 107, 109,
	 111, 112, 108,
	 107, 111, 108,
	 113, 114, 115,
	 116, 113, 115,
	 117, 118, 119,
	 120, 117, 119,
	 121, 122, 123,
	 124, 121, 123,
	 124, 123, 125,
	 126, 124, 125,
	 127, 128, 20,
	 23, 127, 20,
	 129, 130, 131,
	 132, 129, 131,
	 133, 134, 135,
	 136, 133, 135,
	 137, 138, 139,
	 140, 137, 139,
	 141, 36, 20,
	 128, 141, 20,
	 111, 107, 125,
	 123, 111, 125,
	 142, 143, 144,
	 145, 142, 144,
	 44, 43, 146,
	 147, 44, 146,
	 148, 108, 112,
	 149, 148, 112,
	 149, 112, 150,
	 151, 149, 150,
	 152, 150, 112,
	 111, 152, 112,
	 153, 109, 108,
	 148, 153, 108,
	 107, 110, 135,
	 134, 107, 135,
	 154, 155, 156,
	 157, 154, 156,
	 158, 159, 160,
	 161, 158, 160,
	 162, 163, 164,
	 165, 162, 164,
	 141, 146, 43,
	 36, 141, 43,
	 166, 167, 64,
	 67, 166, 64,
	 168, 169, 170,
	 171, 168, 170,
	 172, 173, 72,
	 75, 172, 72,
	 174, 164, 163,
	 175, 174, 163,
	 162, 165, 170,
	 169, 162, 170,
	 23, 72, 173,
	 127, 23, 173,
	 126, 144, 143,
	 124, 126, 143,
	 124, 143, 176,
	 121, 124, 176,
	 136, 131, 130,
	 133, 136, 130,
	 171, 156, 155,
	 168, 171, 155,
	 177, 178, 143,
	 142, 177, 143,
	 122, 111, 123,
	 122, 121, 179,
	 122, 180, 152,
	 111, 122, 152,
	 181, 182, 183,
	 184, 181, 183,
	 175, 185, 186,
	 174, 175, 186,
	 149, 187, 188,
	 148, 149, 188,
	 189, 190, 182,
	 181, 189, 182,
	 151, 191, 187,
	 149, 151, 187,
	 192, 193, 94,
	 97, 192, 94,
	 147, 194, 98,
	 44, 147, 98,
	 184, 183, 195,
	 196, 184, 195,
	 197, 198, 199,
	 200, 197, 199,
	 181, 187, 191,
	 189, 181, 191,
	 97, 98, 194,
	 192, 97, 194,
	 184, 188, 187,
	 181, 184, 187,
	 200, 186, 185,
	 197, 200, 185,
	 196, 201, 188,
	 184, 196, 188,
	 148, 188, 201,
	 153, 148, 201,
};
#define _Flag_Wall_indices_
#endif
// Part of an OBJ_MATERIAL, the struct is 16 byte aligned so it is GPU register friendly.
#ifndef __OBJ_ATTRIBUTES__
typedef struct _OBJ_ATTRIBUTES_
{
	OBJ_VEC3    Kd; // diffuse reflectivity
	float	    d; // dissolve (transparency) 
	OBJ_VEC3    Ks; // specular reflectivity
	float       Ns; // specular exponent
	OBJ_VEC3    Ka; // ambient reflectivity
	float       sharpness; // local reflection map sharpness
	OBJ_VEC3    Tf; // transmission filter
	float       Ni; // optical density (index of refraction)
	OBJ_VEC3    Ke; // emissive reflectivity
	unsigned    illum; // illumination model
}OBJ_ATTRIBUTES;
#define __OBJ_ATTRIBUTES__
#endif
// This structure also has been made GPU register friendly so it can be transfered directly if desired.
// Note: Total struct size will vary depenedening on CPU processor architecture (string pointers).
#ifndef __OBJ_MATERIAL__
typedef struct _OBJ_MATERIAL_
{
	// The following items are typically used in a pixel/fragment shader, they are packed for GPU registers.
	OBJ_ATTRIBUTES attrib; // Surface shading characteristics & illumination model
	// All items below this line are not needed on the GPU and are generally only used during load time.
	const char* name; // the name of this material
	// If the model's materials contain any specific texture data it will be located below.
	const char* map_Kd; // diffuse texture
	const char* map_Ks; // specular texture
	const char* map_Ka; // ambient texture
	const char* map_Ke; // emissive texture
	const char* map_Ns; // specular exponent texture
	const char* map_d; // transparency texture
	const char* disp; // roughness map (displacement)
	const char* decal; // decal texture (lerps texture & material colors)
	const char* bump; // normal/bumpmap texture
	void* padding[2]; // 16 byte alignment on 32bit or 64bit
}OBJ_MATERIAL;
#define __OBJ_MATERIAL__
#endif
#ifndef _Flag_Wall_materials_
// Material Data follows: pulled from a .mtl file of the same name if present.
const OBJ_MATERIAL Flag_Wall_materials[1] =
{
	{
		{{ 0.210681f, 0.067752f, 0.018038f },
		1.000000f,
		{ 0.500000f, 0.500000f, 0.500000f },
		96.078430f,
		{ 1.000000f, 1.000000f, 1.000000f },
		60.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		1.000000f,
		{ 0.000000f, 0.000000f, 0.000000f },
		2},
		"Flag",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
	},
};
#define _Flag_Wall_materials_
#endif
/************************************************/
/*  This section contains the model's structure */
/************************************************/
#ifndef _Flag_Wall_batches_
// Use this conveinence array to batch render all geometry by like material.
// Each entry corresponds to the same entry in the materials array above.
// The two numbers provided are the IndexCount and the IndexOffset into the indices array.
// If you need more fine grained control(ex: for transformations) use the OBJ_MESH data below.
const unsigned int Flag_Wall_batches[1][2] =
{
	{ 612, 0 },
};
#define _Flag_Wall_batches_
#endif
#ifndef __OBJ_MESH__
typedef struct _OBJ_MESH_
{
	const char* name;
	unsigned    indexCount;
	unsigned    indexOffset;
	unsigned    materialIndex;
}OBJ_MESH;
#define __OBJ_MESH__
#endif
#ifndef _Flag_Wall_meshes_
// Mesh Data follows: Meshes are .obj groups sorted & split by material.
// Meshes are provided in sequential order, sorted by material first and name second.
const OBJ_MESH Flag_Wall_meshes[1] =
{
	{
		"default",
		612,
		0,
		0,
	},
};
#define _Flag_Wall_meshes_
#endif
