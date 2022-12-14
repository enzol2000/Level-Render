// "Bush_Round.h" generated by "Obj2Header.exe" [Version 1.9d] Author: L.Norri Fullsail University.
// Data is converted to left-handed coordinate system and UV data is V flipped for Direct3D/Vulkan.
// The companion file "Bush_Round.h2b" is a binary dump of this format(-padding) for more flexibility.
// Loading *.h2b: read version, sizes, handle strings(max len 260) by reading until null-terminated.
/***********************/
/*  Generator version  */
/***********************/
#ifndef _Bush_Round_version_
const char Bush_Round_version[4] = { '0','1','9','d' };
#define _Bush_Round_version_
#endif
/************************************************/
/*  This section contains the model's size data */
/************************************************/
#ifndef _Bush_Round_vertexcount_
const unsigned Bush_Round_vertexcount = 221;
#define _Bush_Round_vertexcount_
#endif
#ifndef _Bush_Round_indexcount_
const unsigned Bush_Round_indexcount = 936;
#define _Bush_Round_indexcount_
#endif
#ifndef _Bush_Round_materialcount_
const unsigned Bush_Round_materialcount = 1; // can be used for batched draws
#define _Bush_Round_materialcount_
#endif
#ifndef _Bush_Round_meshcount_
const unsigned Bush_Round_meshcount = 1;
#define _Bush_Round_meshcount_
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
#ifndef _Bush_Round_vertices_
// Raw Vertex Data follows: Positions, Texture Coordinates and Normals.
const OBJ_VERT Bush_Round_vertices[221] =
{
	{	{ 0.078075f, 0.148503f, -0.302481f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.304300f, 0.857900f, -0.414100f }	},
	{	{ 0.026944f, 0.054529f, -0.457222f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.212300f, 0.655100f, -0.725100f }	},
	{	{ -0.140261f, -0.012293f, -0.511694f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.134000f, 0.535100f, -0.834100f }	},
	{	{ -0.171581f, 0.154930f, -0.327204f },	{ 0.500000f, 0.500000f, 0.000000f },	{ -0.195700f, 0.865600f, -0.461000f }	},
	{	{ -0.317518f, -0.019011f, -0.449043f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.493900f, 0.504400f, -0.708300f }	},
	{	{ -0.395470f, 0.047405f, -0.291238f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.646400f, 0.654900f, -0.391500f }	},
	{	{ -0.177133f, 0.208200f, -0.082026f },	{ 0.242613f, 0.597317f, 0.000000f },	{ -0.208100f, 0.977700f, 0.028500f }	},
	{	{ -0.344339f, 0.141379f, -0.136497f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.548900f, 0.833200f, -0.067600f }	},
	{	{ 0.000124f, 0.214918f, -0.144676f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.157300f, 0.983900f, -0.084300f }	},
	{	{ -0.160909f, 0.156311f, 0.134669f },	{ 0.000100f, 0.689010f, 0.000000f },	{ -0.197100f, 0.950000f, 0.242200f }	},
	{	{ 0.138855f, 0.167672f, 0.028719f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.267100f, 0.962500f, 0.048600f }	},
	{	{ 0.184213f, -0.103569f, -0.499839f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.337300f, 0.542800f, -0.769200f }	},
	{	{ -0.098555f, -0.216573f, -0.591958f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.096200f, 0.346900f, -0.932900f }	},
	{	{ -0.443676f, 0.043307f, 0.042550f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.634100f, 0.770000f, 0.070000f }	},
	{	{ -0.530146f, -0.115616f, -0.219137f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.794200f, 0.510200f, -0.330000f }	},
	{	{ -0.398320f, -0.227934f, -0.486008f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.564000f, 0.350400f, -0.747800f }	},
	{	{ 0.270681f, 0.055354f, -0.238152f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.501000f, 0.786700f, -0.360700f }	},
	{	{ 0.158568f, 0.348865f, 0.275871f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.337900f, 0.895600f, 0.289500f }	},
	{	{ 0.083338f, 0.398293f, 0.087993f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.211100f, 0.975500f, -0.062300f }	},
	{	{ -0.110300f, 0.390147f, 0.011572f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.149500f, 0.970400f, -0.189600f }	},
	{	{ -0.117858f, 0.374470f, 0.288943f },	{ 0.500000f, 0.500000f, 0.000000f },	{ -0.162300f, 0.936400f, 0.311300f }	},
	{	{ -0.301608f, 0.336243f, 0.073995f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.501500f, 0.860600f, -0.088300f }	},
	{	{ -0.370628f, 0.263563f, 0.256628f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.621500f, 0.740900f, 0.254600f }	},
	{	{ -0.101762f, 0.222280f, 0.520928f },	{ 0.242613f, 0.597317f, 0.000000f },	{ -0.134000f, 0.666100f, 0.733700f }	},
	{	{ -0.295398f, 0.214135f, 0.444507f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.490000f, 0.634600f, 0.597600f }	},
	{	{ 0.089547f, 0.276184f, 0.458505f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.222600f, 0.749400f, 0.623500f }	},
	{	{ -0.072222f, 0.010826f, 0.645929f },	{ 0.000100f, 0.689010f, 0.000000f },	{ -0.111000f, 0.494200f, 0.862200f }	},
	{	{ 0.251307f, 0.101985f, 0.540363f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.339000f, 0.639200f, 0.690300f }	},
	{	{ 0.240804f, 0.308487f, -0.086223f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.324300f, 0.927700f, -0.185000f }	},
	{	{ -0.086659f, 0.294712f, -0.215461f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.132100f, 0.908800f, -0.395800f }	},
	{	{ -0.399687f, -0.002949f, 0.516691f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.570400f, 0.492600f, 0.657200f }	},
	{	{ -0.526914f, 0.080641f, 0.198962f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.775100f, 0.596200f, 0.209500f }	},
	{	{ -0.410189f, 0.203553f, -0.109895f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.585100f, 0.781100f, -0.218100f }	},
	{	{ 0.368030f, 0.224897f, 0.231505f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.532000f, 0.806800f, 0.257000f }	},
	{	{ 0.408170f, 0.232338f, -0.127372f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.742100f, 0.645300f, -0.181400f }	},
	{	{ 0.331686f, 0.220436f, -0.302535f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.593200f, 0.604900f, -0.531300f }	},
	{	{ 0.157660f, 0.267247f, -0.367321f },	{ 0.757387f, 0.402683f, 0.000000f },	{ 0.248100f, 0.714200f, -0.654600f }	},
	{	{ 0.197433f, 0.376222f, -0.139709f },	{ 0.500000f, 0.500000f, 0.000000f },	{ 0.324500f, 0.923500f, -0.204500f }	},
	{	{ -0.006538f, 0.339732f, -0.300550f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.088000f, 0.845100f, -0.527300f }	},
	{	{ -0.056798f, 0.396340f, -0.124644f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.174900f, 0.968700f, -0.176000f }	},
	{	{ 0.193713f, 0.361430f, 0.115305f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.319200f, 0.899900f, 0.297200f }	},
	{	{ 0.019688f, 0.408242f, 0.050518f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.035100f, 0.983100f, 0.179700f }	},
	{	{ 0.357910f, 0.288946f, 0.048534f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.646000f, 0.742800f, 0.175700f }	},
	{	{ 0.157283f, 0.254861f, 0.312885f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.291600f, 0.818400f, 0.495200f }	},
	{	{ 0.434963f, 0.132279f, 0.199966f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.714500f, 0.635100f, 0.293500f }	},
	{	{ 0.390614f, 0.016419f, -0.393739f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.647100f, 0.459000f, -0.608800f }	},
	{	{ 0.096314f, 0.095585f, -0.503301f },	{ 0.999900f, 0.310990f, 0.000000f },	{ 0.194700f, 0.565300f, -0.801500f }	},
	{	{ -0.137018f, 0.334026f, 0.203322f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.154800f, 0.941700f, 0.298600f }	},
	{	{ -0.266363f, 0.313897f, -0.092901f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.381500f, 0.912200f, -0.149500f }	},
	{	{ -0.181368f, 0.218167f, -0.390383f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.222200f, 0.765600f, -0.603700f }	},
	{	{ 0.519960f, 0.036548f, -0.097515f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.867800f, 0.471600f, -0.156800f }	},
	{	{ -0.100371f, 0.196571f, -0.380602f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.086600f, 0.770200f, -0.631900f }	},
	{	{ -0.271255f, 0.130786f, -0.432784f },	{ 0.618652f, 0.237074f, 0.000000f },	{ -0.249200f, 0.623300f, -0.741200f }	},
	{	{ -0.434546f, 0.116016f, -0.335984f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.576300f, 0.610400f, -0.543400f }	},
	{	{ -0.276516f, 0.273482f, -0.214552f },	{ 0.500000f, 0.500000f, 0.000000f },	{ -0.261300f, 0.917200f, -0.300800f }	},
	{	{ -0.502194f, 0.153779f, -0.162059f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.717000f, 0.669900f, -0.192800f }	},
	{	{ -0.417849f, 0.228180f, -0.008428f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.543200f, 0.832900f, 0.106300f }	},
	{	{ -0.083675f, 0.308735f, -0.053045f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.119700f, 0.992700f, 0.017800f }	},
	{	{ -0.246967f, 0.293965f, 0.043754f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.200000f, 0.953800f, 0.224000f }	},
	{	{ -0.016026f, 0.270972f, -0.226970f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.267800f, 0.907300f, -0.324300f }	},
	{	{ 0.076703f, 0.237959f, 0.089775f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.278400f, 0.947600f, 0.157000f }	},
	{	{ 0.191106f, 0.174098f, -0.204356f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.430200f, 0.847100f, -0.312000f }	},
	{	{ -0.240522f, -0.062976f, -0.552414f },	{ 0.725594f, 0.000100f, 0.000000f },	{ -0.229500f, 0.484700f, -0.844000f }	},
	{	{ -0.516667f, -0.087954f, -0.388713f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.669800f, 0.426800f, -0.607600f }	},
	{	{ -0.199442f, 0.212982f, 0.253476f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.166700f, 0.906500f, 0.387800f }	},
	{	{ -0.488432f, 0.101730f, 0.165229f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.624700f, 0.729900f, 0.277600f }	},
	{	{ -0.631069f, -0.024092f, -0.094583f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.826500f, 0.544100f, -0.144200f }	},
	{	{ 0.048467f, 0.048275f, -0.464167f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.233200f, 0.644400f, -0.728200f }	},
	{	{ 0.310814f, 0.260768f, 0.027337f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.388400f, 0.905100f, -0.173100f }	},
	{	{ 0.119094f, 0.289166f, -0.049074f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.039700f, 0.944900f, -0.324800f }	},
	{	{ -0.067522f, 0.282122f, 0.043261f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.297400f, 0.943800f, -0.144200f }	},
	{	{ 0.131898f, 0.303707f, 0.235623f },	{ 0.500000f, 0.500000f, 0.000000f },	{ 0.063700f, 0.977000f, 0.203300f }	},
	{	{ -0.147953f, 0.248017f, 0.232388f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.454700f, 0.868800f, 0.196200f }	},
	{	{ -0.056305f, 0.204199f, 0.414273f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.277100f, 0.802500f, 0.528400f }	},
	{	{ 0.322032f, 0.182845f, 0.398349f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.408800f, 0.763800f, 0.499500f }	},
	{	{ 0.135415f, 0.175802f, 0.490684f },	{ 0.381347f, 0.762926f, 0.000000f },	{ 0.069900f, 0.735100f, 0.674400f }	},
	{	{ 0.402461f, 0.216951f, 0.209221f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.564200f, 0.811300f, 0.153300f }	},
	{	{ 0.446210f, -0.011574f, 0.487743f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.528800f, 0.609400f, 0.590800f }	},
	{	{ 0.582228f, 0.046102f, 0.167903f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.698500f, 0.704300f, 0.126800f }	},
	{	{ 0.103016f, 0.168227f, -0.268910f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.029000f, 0.874900f, -0.483400f }	},
	{	{ -0.212580f, 0.156315f, -0.112759f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.433300f, 0.854600f, -0.286200f }	},
	{	{ 0.130615f, -0.023486f, 0.643894f },	{ 0.274406f, 0.999900f, 0.000000f },	{ 0.067600f, 0.607100f, 0.791800f }	},
	{	{ -0.193609f, 0.024537f, 0.514673f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.405600f, 0.662100f, 0.630100f }	},
	{	{ -0.348597f, 0.098639f, 0.207081f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.601900f, 0.777700f, 0.181500f }	},
	{	{ 0.427239f, 0.120204f, -0.139688f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.501100f, 0.801800f, -0.325500f }	},
	{	{ -0.248882f, 0.237043f, 0.204380f },	{ 0.402683f, 0.242613f, 0.000000f },	{ -0.110700f, 0.981500f, 0.156200f }	},
	{	{ -0.413389f, 0.196781f, 0.144558f },	{ 0.618652f, 0.237074f, 0.000000f },	{ -0.454000f, 0.890900f, 0.014000f }	},
	{	{ -0.540118f, 0.088516f, 0.211498f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.723100f, 0.669200f, 0.171200f }	},
	{	{ -0.376674f, 0.136078f, 0.380136f },	{ 0.500000f, 0.500000f, 0.000000f },	{ -0.377200f, 0.764600f, 0.522600f }	},
	{	{ -0.564819f, -0.020018f, 0.352469f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.779100f, 0.425400f, 0.460400f }	},
	{	{ -0.457057f, -0.060998f, 0.490203f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.548500f, 0.354800f, 0.757200f }	},
	{	{ -0.165820f, 0.087529f, 0.483086f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.063900f, 0.667100f, 0.742200f }	},
	{	{ -0.292549f, -0.020736f, 0.550026f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.194500f, 0.423900f, 0.884600f }	},
	{	{ -0.141120f, 0.196063f, 0.342115f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.130600f, 0.889400f, 0.438200f }	},
	{	{ 0.016816f, -0.020845f, 0.503101f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.253500f, 0.571400f, 0.780500f }	},
	{	{ 0.058588f, 0.162701f, 0.264699f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.294500f, 0.875900f, 0.382100f }	},
	{	{ -0.401855f, 0.163915f, -0.069396f },	{ 0.725594f, 0.000100f, 0.000000f },	{ -0.451600f, 0.877900f, -0.159200f }	},
	{	{ -0.616174f, -0.019175f, 0.043809f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.818700f, 0.574300f, 0.002500f }	},
	{	{ -0.197501f, -0.203935f, 0.616305f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.120500f, 0.281900f, 0.951900f }	},
	{	{ -0.475706f, -0.272023f, 0.515137f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.580700f, 0.146000f, 0.800900f }	},
	{	{ -0.657946f, -0.202721f, 0.282210f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.866600f, 0.283800f, 0.410500f }	},
	{	{ -0.123652f, 0.232003f, 0.031772f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.015600f, 0.999700f, -0.017900f }	},
	{	{ 0.345225f, 0.068367f, -0.370985f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.632800f, 0.561800f, -0.532900f }	},
	{	{ 0.180918f, 0.111334f, -0.459474f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.303100f, 0.631200f, -0.714000f }	},
	{	{ 0.021825f, 0.196342f, -0.395157f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.005000f, 0.814200f, -0.580600f }	},
	{	{ 0.214192f, 0.238316f, -0.232376f },	{ 0.500000f, 0.500000f, 0.000000f },	{ 0.372100f, 0.891600f, -0.258000f }	},
	{	{ -0.045987f, 0.271661f, -0.232670f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.153800f, 0.954000f, -0.257200f }	},
	{	{ 0.033293f, 0.288774f, -0.059189f },	{ 0.597317f, 0.757387f, 0.000000f },	{ 0.017600f, 0.996500f, 0.082000f }	},
	{	{ 0.356691f, 0.160799f, -0.035018f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.655400f, 0.744100f, 0.129700f }	},
	{	{ 0.197600f, 0.245807f, 0.029300f },	{ 0.381347f, 0.762926f, 0.000000f },	{ 0.336700f, 0.902000f, 0.270300f }	},
	{	{ 0.424505f, 0.085480f, -0.197505f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.793700f, 0.579100f, -0.186400f }	},
	{	{ 0.416357f, 0.014206f, 0.128299f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.728600f, 0.620200f, 0.290500f }	},
	{	{ 0.531036f, -0.113169f, -0.146488f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.886800f, 0.436900f, -0.150500f }	},
	{	{ 0.119099f, -0.069446f, -0.589514f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.260800f, 0.503400f, -0.823800f }	},
	{	{ -0.149948f, 0.074313f, -0.480743f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.171100f, 0.715700f, -0.677100f }	},
	{	{ 0.147311f, 0.157965f, 0.237069f },	{ 0.274406f, 0.999900f, 0.000000f },	{ 0.303600f, 0.849000f, 0.432400f }	},
	{	{ -0.130556f, 0.230629f, 0.087422f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.140300f, 0.964100f, 0.225600f }	},
	{	{ -0.264627f, 0.201688f, -0.205956f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.322400f, 0.915400f, -0.240900f }	},
	{	{ 0.396965f, -0.142110f, -0.439867f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.697800f, 0.371900f, -0.612200f }	},
	{	{ 0.134033f, 0.088248f, 0.267892f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.413200f, 0.864000f, 0.287600f }	},
	{	{ 0.074946f, 0.133671f, 0.105759f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.295600f, 0.948500f, -0.113400f }	},
	{	{ -0.087388f, 0.132468f, 0.036732f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.109100f, 0.956300f, -0.271100f }	},
	{	{ -0.100833f, 0.129631f, 0.271012f },	{ 0.500000f, 0.500000f, 0.000000f },	{ -0.141000f, 0.941300f, 0.306800f }	},
	{	{ -0.251243f, 0.091268f, 0.086518f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.515400f, 0.841600f, -0.161500f }	},
	{	{ -0.314390f, 0.029956f, 0.241441f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.653400f, 0.722500f, 0.226000f }	},
	{	{ -0.092968f, -0.014264f, 0.472601f },	{ 0.242613f, 0.597317f, 0.000000f },	{ -0.128700f, 0.644000f, 0.754200f }	},
	{	{ -0.255303f, -0.015467f, 0.403574f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.515100f, 0.612800f, 0.599300f }	},
	{	{ 0.070887f, 0.026936f, 0.422815f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.266600f, 0.717200f, 0.643800f }	},
	{	{ -0.070942f, -0.223522f, 0.586478f },	{ 0.000100f, 0.689010f, 0.000000f },	{ -0.111400f, 0.466000f, 0.877800f }	},
	{	{ 0.206158f, -0.153848f, 0.502284f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.367900f, 0.590900f, 0.718000f }	},
	{	{ 0.213021f, 0.026654f, -0.033901f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.431300f, 0.869700f, -0.240100f }	},
	{	{ -0.061507f, 0.024620f, -0.150635f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.081500f, 0.858900f, -0.505600f }	},
	{	{ -0.345470f, -0.225557f, 0.469745f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.587300f, 0.461500f, 0.664900f }	},
	{	{ -0.445396f, -0.148741f, 0.195555f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.814700f, 0.550000f, 0.183800f }	},
	{	{ -0.338607f, -0.045054f, -0.066441f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.609900f, 0.731900f, -0.304000f }	},
	{	{ 0.312947f, -0.050161f, 0.240288f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.615100f, 0.741800f, 0.267300f }	},
	{	{ 0.480159f, 0.011224f, 0.179201f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.824100f, 0.531700f, 0.195300f }	},
	{	{ 0.439055f, 0.077251f, 0.024180f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.745400f, 0.641400f, -0.181700f }	},
	{	{ 0.297064f, 0.171430f, -0.035193f },	{ 0.757387f, 0.402683f, 0.000000f },	{ 0.445900f, 0.838000f, -0.314600f }	},
	{	{ 0.299393f, 0.184856f, 0.189744f },	{ 0.500000f, 0.500000f, 0.000000f },	{ 0.455300f, 0.858300f, 0.236400f }	},
	{	{ 0.144521f, 0.238811f, 0.021200f },	{ 0.762926f, 0.618653f, 0.000000f },	{ 0.071900f, 0.977400f, -0.198900f }	},
	{	{ 0.075254f, 0.233327f, 0.175105f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.094400f, 0.973200f, 0.209600f }	},
	{	{ 0.258349f, 0.073121f, 0.389499f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.328500f, 0.634900f, 0.699300f }	},
	{	{ 0.116358f, 0.167301f, 0.330126f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.010100f, 0.808300f, 0.588700f }	},
	{	{ 0.410893f, 0.005740f, 0.333106f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.659600f, 0.499400f, 0.561700f }	},
	{	{ 0.196216f, -0.101537f, 0.507953f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.272200f, 0.471900f, 0.838600f }	},
	{	{ 0.454189f, -0.215488f, 0.412585f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.704400f, 0.358000f, 0.612900f }	},
	{	{ 0.501814f, -0.094554f, -0.109850f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.813100f, 0.512400f, -0.276100f }	},
	{	{ 0.261689f, 0.064716f, -0.210258f },	{ 0.999900f, 0.310990f, 0.000000f },	{ 0.418600f, 0.735400f, -0.532900f }	},
	{	{ -0.043908f, 0.057733f, 0.407546f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.160900f, 0.715600f, 0.679700f }	},
	{	{ -0.113423f, 0.169393f, 0.145384f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.341100f, 0.922500f, 0.180500f }	},
	{	{ 0.003719f, 0.178667f, -0.114890f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.068600f, 0.936800f, -0.343000f }	},
	{	{ 0.571329f, -0.206214f, 0.152312f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.915200f, 0.361800f, 0.177600f }	},
	{	{ -0.190288f, -0.000985f, -0.364343f },	{ 0.618652f, 0.237074f, 0.000000f },	{ -0.247100f, 0.519200f, -0.818100f }	},
	{	{ -0.331609f, -0.023295f, -0.290994f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.609500f, 0.481300f, -0.630000f }	},
	{	{ -0.220395f, 0.148064f, -0.184126f },	{ 0.500000f, 0.500000f, 0.000000f },	{ -0.337400f, 0.856800f, -0.389900f }	},
	{	{ -0.055656f, 0.076918f, -0.312984f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.085900f, 0.709700f, -0.699200f }	},
	{	{ -0.401723f, 0.015029f, -0.148898f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.798500f, 0.538800f, -0.268700f }	},
	{	{ -0.346323f, 0.098933f, -0.016804f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.666400f, 0.743600f, 0.054300f }	},
	{	{ -0.070368f, 0.199146f, -0.038794f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.065000f, 0.997900f, -0.002900f }	},
	{	{ -0.211693f, 0.176836f, 0.034555f },	{ 0.381347f, 0.762926f, 0.000000f },	{ -0.312300f, 0.925800f, 0.212900f }	},
	{	{ -0.000255f, 0.160822f, -0.180890f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.259000f, 0.892500f, -0.369400f }	},
	{	{ 0.063530f, 0.140820f, 0.091593f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.248300f, 0.953400f, 0.171500f }	},
	{	{ 0.182099f, 0.076008f, -0.148710f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.445100f, 0.823500f, -0.351700f }	},
	{	{ -0.139268f, -0.197630f, -0.458954f },	{ 0.725594f, 0.000100f, 0.000000f },	{ -0.218800f, 0.376400f, -0.900300f }	},
	{	{ -0.378264f, -0.235359f, -0.334910f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.667500f, 0.289000f, -0.686300f }	},
	{	{ -0.175467f, 0.103090f, 0.215638f },	{ 0.274406f, 0.999900f, 0.000000f },	{ -0.283200f, 0.867100f, 0.409800f }	},
	{	{ -0.403146f, -0.028654f, 0.128783f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.758600f, 0.607200f, 0.236100f }	},
	{	{ -0.496835f, -0.170548f, -0.094606f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.890900f, 0.387000f, -0.238000f }	},
	{	{ 0.088411f, -0.065885f, -0.372099f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.245700f, 0.571900f, -0.782700f }	},
	{	{ 0.182312f, 0.059126f, 0.046982f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.085500f, 0.929600f, -0.358500f }	},
	{	{ 0.013101f, 0.056060f, 0.119463f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.309900f, 0.934500f, -0.175300f }	},
	{	{ 0.179885f, 0.072070f, 0.298328f },	{ 0.500000f, 0.500000f, 0.000000f },	{ 0.085100f, 0.965300f, 0.247100f }	},
	{	{ 0.349074f, 0.023426f, 0.123044f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.458600f, 0.874100f, -0.160300f }	},
	{	{ -0.065996f, 0.020864f, 0.281944f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.506400f, 0.838400f, 0.201600f }	},
	{	{ 0.007714f, -0.029175f, 0.446054f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.317800f, 0.754400f, 0.574300f }	},
	{	{ 0.343687f, -0.061809f, 0.449636f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.425900f, 0.706500f, 0.565200f }	},
	{	{ 0.174477f, -0.064875f, 0.522116f },	{ 0.381347f, 0.762926f, 0.000000f },	{ 0.049200f, 0.672800f, 0.738200f }	},
	{	{ 0.422786f, -0.026613f, 0.287154f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.617800f, 0.760400f, 0.200100f }	},
	{	{ 0.454580f, -0.271021f, 0.532758f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.532600f, 0.540900f, 0.650900f }	},
	{	{ 0.588346f, -0.211500f, 0.257980f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.746100f, 0.638100f, 0.190000f }	},
	{	{ 0.181671f, -0.066503f, -0.148184f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.091300f, 0.837200f, -0.539200f }	},
	{	{ -0.104489f, -0.071688f, -0.025610f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.457000f, 0.817800f, -0.349800f }	},
	{	{ 0.168421f, -0.276206f, 0.655332f },	{ 0.274406f, 0.999900f, 0.000000f },	{ 0.053400f, 0.531400f, 0.845400f }	},
	{	{ -0.113597f, -0.215831f, 0.526701f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.451000f, 0.584500f, 0.674400f }	},
	{	{ -0.238254f, -0.131209f, 0.249168f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.670000f, 0.719900f, 0.181200f }	},
	{	{ 0.463687f, -0.126878f, -0.019553f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.590200f, 0.745000f, -0.310900f }	},
	{	{ 0.203436f, -0.065338f, -0.407902f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.252500f, 0.675900f, -0.692400f }	},
	{	{ 0.053028f, -0.001072f, -0.352111f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.083700f, 0.814000f, -0.574700f }	},
	{	{ 0.204139f, 0.048655f, -0.206243f },	{ 0.500000f, 0.500000f, 0.000000f },	{ 0.307800f, 0.929400f, -0.203800f }	},
	{	{ 0.347498f, -0.097702f, -0.331790f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.593900f, 0.641300f, -0.485800f }	},
	{	{ -0.020916f, 0.055973f, -0.211729f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.283900f, 0.930900f, -0.229800f }	},
	{	{ 0.039077f, 0.069068f, -0.062104f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.107600f, 0.979300f, 0.171600f }	},
	{	{ 0.333548f, -0.027561f, -0.041783f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.592300f, 0.776300f, 0.215500f }	},
	{	{ 0.183139f, 0.036704f, 0.014008f },	{ 0.381347f, 0.762926f, 0.000000f },	{ 0.249800f, 0.891500f, 0.377900f }	},
	{	{ 0.407489f, -0.084606f, -0.182165f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.747000f, 0.655000f, -0.113400f }	},
	{	{ 0.406092f, -0.178279f, 0.086353f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.684500f, 0.634600f, 0.358900f }	},
	{	{ 0.531136f, -0.274749f, -0.151052f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.841500f, 0.530000f, -0.104900f }	},
	{	{ 0.186054f, -0.242164f, -0.532803f },	{ 0.725594f, 0.000100f, 0.000000f },	{ 0.257400f, 0.540500f, -0.801000f }	},
	{	{ -0.068306f, -0.133482f, -0.438453f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.236200f, 0.673300f, -0.700600f }	},
	{	{ 0.151731f, -0.069597f, 0.180703f },	{ 0.274406f, 0.999900f, 0.000000f },	{ 0.229700f, 0.800400f, 0.553700f }	},
	{	{ -0.091899f, -0.014865f, 0.051988f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.286000f, 0.898600f, 0.332700f }	},
	{	{ -0.193353f, -0.037012f, -0.201048f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.473200f, 0.851700f, -0.225200f }	},
	{	{ 0.429683f, -0.296896f, -0.404087f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.676700f, 0.481400f, -0.557000f }	},
	{	{ 0.263586f, 0.150484f, -0.427012f },	{ 0.402683f, 0.242613f, 0.000000f },	{ 0.398000f, 0.637900f, -0.659300f }	},
	{	{ 0.077529f, 0.185161f, -0.456225f },	{ 0.618652f, 0.237074f, 0.000000f },	{ 0.024600f, 0.690900f, -0.722600f }	},
	{	{ -0.057595f, 0.253793f, -0.339159f },	{ 0.757387f, 0.402683f, 0.000000f },	{ -0.235400f, 0.841600f, -0.486100f }	},
	{	{ 0.173689f, 0.301836f, -0.241875f },	{ 0.500000f, 0.500000f, 0.000000f },	{ 0.219400f, 0.930800f, -0.292400f }	},
	{	{ -0.074646f, 0.314614f, -0.158373f },	{ 0.762926f, 0.618653f, 0.000000f },	{ -0.281900f, 0.951600f, -0.122700f }	},
	{	{ 0.054388f, 0.328448f, -0.017545f },	{ 0.597317f, 0.757387f, 0.000000f },	{ -0.014600f, 0.988900f, 0.148200f }	},
	{	{ 0.375571f, 0.225140f, -0.105398f },	{ 0.242613f, 0.597317f, 0.000000f },	{ 0.618800f, 0.785100f, -0.025100f }	},
	{	{ 0.240445f, 0.293772f, 0.011668f },	{ 0.381347f, 0.762926f, 0.000000f },	{ 0.352700f, 0.909600f, 0.219700f }	},
	{	{ 0.392621f, 0.164319f, -0.286184f },	{ 0.237074f, 0.381347f, 0.000000f },	{ 0.659100f, 0.648900f, -0.380100f }	},
	{	{ 0.492211f, 0.075028f, 0.019437f },	{ 0.000100f, 0.689010f, 0.000000f },	{ 0.746300f, 0.658800f, 0.094900f }	},
	{	{ 0.521044f, -0.027830f, -0.286297f },	{ 0.000100f, 0.274406f, 0.000000f },	{ 0.766800f, 0.512800f, -0.386000f }	},
	{	{ -0.011817f, 0.007418f, -0.573859f },	{ 0.725594f, 0.000100f, 0.000000f },	{ -0.043000f, 0.566300f, -0.823100f }	},
	{	{ -0.240333f, 0.123484f, -0.375886f },	{ 0.999900f, 0.310990f, 0.000000f },	{ -0.417600f, 0.735800f, -0.533100f }	},
	{	{ 0.263697f, 0.191093f, 0.217410f },	{ 0.274406f, 0.999900f, 0.000000f },	{ 0.375700f, 0.845500f, 0.379500f }	},
	{	{ -0.050952f, 0.249736f, 0.168007f },	{ 0.689010f, 0.999900f, 0.000000f },	{ -0.116700f, 0.936400f, 0.331000f }	},
	{	{ -0.269165f, 0.226341f, -0.070152f },	{ 0.999900f, 0.725594f, 0.000000f },	{ -0.434100f, 0.899000f, -0.057500f }	},
	{	{ 0.302831f, -0.051225f, -0.524456f },	{ 0.310990f, 0.000100f, 0.000000f },	{ 0.445400f, 0.458200f, -0.769200f }	},
};
#define _Bush_Round_vertices_
#endif
#ifndef _Bush_Round_indices_
// Index Data follows: Sequential by mesh, Every Three Indices == One Triangle.
const unsigned int Bush_Round_indices[936] =
{
	 0, 1, 2,
	 3, 0, 2,
	 3, 2, 4,
	 5, 3, 4,
	 6, 3, 5,
	 7, 6, 5,
	 8, 0, 3,
	 6, 8, 3,
	 9, 10, 8,
	 6, 9, 8,
	 11, 12, 2,
	 1, 11, 2,
	 13, 9, 6,
	 7, 13, 6,
	 14, 13, 7,
	 5, 14, 7,
	 12, 15, 4,
	 2, 12, 4,
	 10, 16, 0,
	 8, 10, 0,
	 15, 14, 5,
	 4, 15, 5,
	 16, 11, 1,
	 0, 16, 1,
	 17, 18, 19,
	 20, 17, 19,
	 20, 19, 21,
	 22, 20, 21,
	 23, 20, 22,
	 24, 23, 22,
	 25, 17, 20,
	 23, 25, 20,
	 26, 27, 25,
	 23, 26, 25,
	 28, 29, 19,
	 18, 28, 19,
	 30, 26, 23,
	 24, 30, 23,
	 31, 30, 24,
	 22, 31, 24,
	 29, 32, 21,
	 19, 29, 21,
	 27, 33, 17,
	 25, 27, 17,
	 32, 31, 22,
	 21, 32, 22,
	 33, 28, 18,
	 17, 33, 18,
	 34, 35, 36,
	 37, 34, 36,
	 37, 36, 38,
	 39, 37, 38,
	 40, 37, 39,
	 41, 40, 39,
	 42, 34, 37,
	 40, 42, 37,
	 43, 44, 42,
	 40, 43, 42,
	 45, 46, 36,
	 35, 45, 36,
	 47, 43, 40,
	 41, 47, 40,
	 48, 47, 41,
	 39, 48, 41,
	 46, 49, 38,
	 36, 46, 38,
	 44, 50, 34,
	 42, 44, 34,
	 49, 48, 39,
	 38, 49, 39,
	 50, 45, 35,
	 34, 50, 35,
	 51, 52, 53,
	 54, 51, 53,
	 54, 53, 55,
	 56, 54, 55,
	 57, 54, 56,
	 58, 57, 56,
	 59, 51, 54,
	 57, 59, 54,
	 60, 61, 59,
	 57, 60, 59,
	 62, 63, 53,
	 52, 62, 53,
	 64, 60, 57,
	 58, 64, 57,
	 65, 64, 58,
	 56, 65, 58,
	 63, 66, 55,
	 53, 63, 55,
	 61, 67, 51,
	 59, 61, 51,
	 66, 65, 56,
	 55, 66, 56,
	 67, 62, 52,
	 51, 67, 52,
	 68, 69, 70,
	 71, 68, 70,
	 71, 70, 72,
	 73, 71, 72,
	 74, 71, 73,
	 75, 74, 73,
	 76, 68, 71,
	 74, 76, 71,
	 77, 78, 76,
	 74, 77, 76,
	 79, 80, 70,
	 69, 79, 70,
	 81, 77, 74,
	 75, 81, 74,
	 82, 81, 75,
	 73, 82, 75,
	 80, 83, 72,
	 70, 80, 72,
	 78, 84, 68,
	 76, 78, 68,
	 83, 82, 73,
	 72, 83, 73,
	 84, 79, 69,
	 68, 84, 69,
	 85, 86, 87,
	 88, 85, 87,
	 88, 87, 89,
	 90, 88, 89,
	 91, 88, 90,
	 92, 91, 90,
	 93, 85, 88,
	 91, 93, 88,
	 94, 95, 93,
	 91, 94, 93,
	 96, 97, 87,
	 86, 96, 87,
	 98, 94, 91,
	 92, 98, 91,
	 99, 98, 92,
	 90, 99, 92,
	 97, 100, 89,
	 87, 97, 89,
	 95, 101, 85,
	 93, 95, 85,
	 100, 99, 90,
	 89, 100, 90,
	 101, 96, 86,
	 85, 101, 86,
	 102, 103, 104,
	 105, 102, 104,
	 105, 104, 106,
	 107, 105, 106,
	 108, 105, 107,
	 109, 108, 107,
	 110, 102, 105,
	 108, 110, 105,
	 111, 112, 110,
	 108, 111, 110,
	 113, 114, 104,
	 103, 113, 104,
	 115, 111, 108,
	 109, 115, 108,
	 116, 115, 109,
	 107, 116, 109,
	 114, 117, 106,
	 104, 114, 106,
	 112, 118, 102,
	 110, 112, 102,
	 117, 116, 107,
	 106, 117, 107,
	 118, 113, 103,
	 102, 118, 103,
	 119, 120, 121,
	 122, 119, 121,
	 122, 121, 123,
	 124, 122, 123,
	 125, 122, 124,
	 126, 125, 124,
	 127, 119, 122,
	 125, 127, 122,
	 128, 129, 127,
	 125, 128, 127,
	 130, 131, 121,
	 120, 130, 121,
	 132, 128, 125,
	 126, 132, 125,
	 133, 132, 126,
	 124, 133, 126,
	 131, 134, 123,
	 121, 131, 123,
	 129, 135, 119,
	 127, 129, 119,
	 134, 133, 124,
	 123, 134, 124,
	 135, 130, 120,
	 119, 135, 120,
	 136, 137, 138,
	 139, 136, 138,
	 139, 138, 140,
	 141, 139, 140,
	 142, 139, 141,
	 143, 142, 141,
	 144, 136, 139,
	 142, 144, 139,
	 145, 146, 144,
	 142, 145, 144,
	 147, 148, 138,
	 137, 147, 138,
	 149, 145, 142,
	 143, 149, 142,
	 150, 149, 143,
	 141, 150, 143,
	 148, 151, 140,
	 138, 148, 140,
	 146, 152, 136,
	 144, 146, 136,
	 151, 150, 141,
	 140, 151, 141,
	 152, 147, 137,
	 136, 152, 137,
	 153, 154, 155,
	 156, 153, 155,
	 155, 154, 157,
	 158, 155, 157,
	 159, 155, 158,
	 160, 159, 158,
	 161, 156, 155,
	 159, 161, 155,
	 162, 163, 161,
	 159, 162, 161,
	 164, 165, 154,
	 153, 164, 154,
	 166, 162, 159,
	 160, 166, 159,
	 167, 166, 160,
	 158, 167, 160,
	 165, 168, 157,
	 154, 165, 157,
	 163, 169, 156,
	 161, 163, 156,
	 168, 167, 158,
	 157, 168, 158,
	 169, 164, 153,
	 156, 169, 153,
	 170, 171, 172,
	 173, 170, 172,
	 172, 171, 174,
	 175, 172, 174,
	 176, 172, 175,
	 177, 176, 175,
	 178, 173, 172,
	 176, 178, 172,
	 179, 180, 178,
	 176, 179, 178,
	 181, 182, 171,
	 170, 181, 171,
	 183, 179, 176,
	 177, 183, 176,
	 184, 183, 177,
	 175, 184, 177,
	 182, 185, 174,
	 171, 182, 174,
	 180, 186, 173,
	 178, 180, 173,
	 185, 184, 175,
	 174, 185, 175,
	 186, 181, 170,
	 173, 186, 170,
	 187, 188, 189,
	 190, 187, 189,
	 189, 188, 191,
	 192, 189, 191,
	 193, 189, 192,
	 194, 193, 192,
	 195, 190, 189,
	 193, 195, 189,
	 196, 197, 195,
	 193, 196, 195,
	 198, 199, 188,
	 187, 198, 188,
	 200, 196, 193,
	 194, 200, 193,
	 201, 200, 194,
	 192, 201, 194,
	 199, 202, 191,
	 188, 199, 191,
	 197, 203, 190,
	 195, 197, 190,
	 202, 201, 192,
	 191, 202, 192,
	 203, 198, 187,
	 190, 203, 187,
	 204, 205, 206,
	 207, 204, 206,
	 207, 206, 208,
	 209, 207, 208,
	 210, 207, 209,
	 211, 210, 209,
	 212, 204, 207,
	 210, 212, 207,
	 213, 214, 212,
	 210, 213, 212,
	 215, 216, 206,
	 205, 215, 206,
	 217, 213, 210,
	 211, 217, 210,
	 218, 217, 211,
	 209, 218, 211,
	 216, 219, 208,
	 206, 216, 208,
	 214, 220, 204,
	 212, 214, 204,
	 219, 218, 209,
	 208, 219, 209,
	 220, 215, 205,
	 204, 220, 205,
};
#define _Bush_Round_indices_
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
#ifndef _Bush_Round_materials_
// Material Data follows: pulled from a .mtl file of the same name if present.
const OBJ_MATERIAL Bush_Round_materials[1] =
{
	{
		{{ 0.640000f, 0.640000f, 0.640000f },
		1.000000f,
		{ 0.500000f, 0.500000f, 0.500000f },
		96.078430f,
		{ 1.000000f, 1.000000f, 1.000000f },
		60.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		1.000000f,
		{ 0.000000f, 0.000000f, 0.000000f },
		2},
		"Texture_Leaves",
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
#define _Bush_Round_materials_
#endif
/************************************************/
/*  This section contains the model's structure */
/************************************************/
#ifndef _Bush_Round_batches_
// Use this conveinence array to batch render all geometry by like material.
// Each entry corresponds to the same entry in the materials array above.
// The two numbers provided are the IndexCount and the IndexOffset into the indices array.
// If you need more fine grained control(ex: for transformations) use the OBJ_MESH data below.
const unsigned int Bush_Round_batches[1][2] =
{
	{ 936, 0 },
};
#define _Bush_Round_batches_
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
#ifndef _Bush_Round_meshes_
// Mesh Data follows: Meshes are .obj groups sorted & split by material.
// Meshes are provided in sequential order, sorted by material first and name second.
const OBJ_MESH Bush_Round_meshes[1] =
{
	{
		"default",
		936,
		0,
		0,
	},
};
#define _Bush_Round_meshes_
#endif
