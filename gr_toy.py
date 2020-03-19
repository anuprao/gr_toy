# -*- coding: utf-8 -*-

import argparse
import datetime
from enum import Enum, unique
import json
import logging
import os
import os.path
import platform
#from queue import Queue
import shutil
import subprocess
import sys
import time
#import threading
#from threading import Thread

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import colorama
from colorama import Fore, Back, Style

import glfw
#import OpenGL.GL as gl

import moderngl
from PIL import Image as PIL_Image

from smc.freeimage import Image as FreeImage
from smc.freeimage.enums import FI_FILTER

import numpy as np
from pyrr import Matrix44, matrix44, Vector4, Vector3
from pyrr import Quaternion, quaternion

from objloader import Obj

from copy import copy, deepcopy

#import pyuv
#import requests

#import cProfile

np.set_printoptions(formatter={'float': '{: 8.3f}'.format})

################################################################################


@unique
class visualMode(Enum):
	MONO = 0
	STEREO = 1
	DEBUG_TEXTURE = 2

@unique
class eyePosition(Enum):
	NONE = 0
	LEFT = 1
	RIGHT = 2

def local(*path):
	return os.path.join(os.path.dirname(__file__), *path)

def copyM44(md, ms):
	md.m11 = ms.m11
	md.m12 = ms.m12
	md.m13 = ms.m13
	md.m14 = ms.m14

	md.m21 = ms.m21
	md.m22 = ms.m22
	md.m23 = ms.m23
	md.m24 = ms.m24

	md.m31 = ms.m31
	md.m32 = ms.m32
	md.m33 = ms.m33
	md.m34 = ms.m34

	md.m41 = ms.m41
	md.m42 = ms.m42
	md.m43 = ms.m43
	md.m44 = ms.m44

class base_shader(object):
	def __init__(self):
		self.program = None

	def associateUniforms(self):
		self.attr_projection_matrix = self.program['projection']
		self.attr_view_matrix = self.program['view']
		self.attr_model_matrix = self.program['model']

	def update_proj_matrix_bytes(self, byte_proj_matrix):
		self.attr_projection_matrix.write(byte_proj_matrix)

	def update_view_matrix_bytes(self, byte_view_matrix):
		self.attr_view_matrix.write(byte_view_matrix)

	def update_model_matrix(self, model_matrix):
		byte_model = model_matrix.astype('f4').tobytes()	
		self.attr_model_matrix.write(byte_model)

class flat_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;

				void main() {
					gl_Position = projection * view * model * vec4(in_vert, 1.0);
				}
			''',
			fragment_shader='''
				#version 330

				uniform vec4 colourObject;

				out vec4 FragColor;

				void main() {
					FragColor = colourObject ;
				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.colour_Object = self.program['colourObject']

class debug_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(	
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec2 in_vert;
				in vec2 in_text;

				out vec2 v_text;

				void main() 
				{
					gl_Position = projection * view * model * vec4(in_vert, 0.0, 1.0);

					v_text = in_text;
				}
			''',
			fragment_shader='''
				#version 330
				
				uniform sampler2D debugMap;

				in vec2 v_text;

				out vec4 FragColor;

				void main() 
				{
					FragColor = vec4(texture(debugMap, v_text).rgb, 1.0);
				}
			''',
		)
	def associateUniforms(self):
		super().associateUniforms()

		self.debug_Map = self.program['debugMap']


class grid_shader(flat_shader):
	def __init__(self, ctx):
		super().__init__(ctx)

class light_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;

				void main() {
					gl_Position = projection * view * model * vec4(in_vert, 1.0);
				}
			''',
			fragment_shader='''
				#version 330

				uniform vec3 colourLight;

				out vec4 FragColor;

				void main() {
					FragColor = vec4(colourLight, 1.0) ;
				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.colour_Light = self.program['colourLight']

class fbo_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(	
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec2 in_vert;
				in vec2 in_text;

				out vec2 v_text;

				void main() 
				{
					gl_Position = projection * view * model * vec4(in_vert, 0.0, 1.0);

					v_text = in_text;
				}
			''',
			fragment_shader='''
				#version 330
				
				uniform sampler2D Texture;

				in vec2 v_text;

				out vec4 FragColor;

				void main() 
				{
					FragColor = vec4(texture(Texture, v_text).rgb, 1.0);
				}
			''',
		)

class regular_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;
				in vec3 in_norm;
				in vec2 in_text;

				out vec3 v_norm;
				out vec2 v_text;

				out vec3 v_worldPos;

				void main() 
				{
					gl_Position = projection * view * model * vec4(in_vert, 1.0);

					v_norm = in_norm;
					v_text = in_text;

					v_worldPos = vec3(model * vec4(in_vert, 1.0));
				}
			''',
			fragment_shader='''
				#version 330

				uniform vec3 colourAmbient;
				uniform float strengthAmbient;

				uniform vec3 posLight;
				uniform vec3 colourLight;

				uniform vec3 posView;

				uniform float specularStrength;

				uniform vec4 colourObject;

				uniform bool UseTexture;
				uniform sampler2D Texture;

				in vec3 v_norm;
				in vec2 v_text;

				in vec3 v_worldPos;

				out vec4 FragColor;

				void main() 
				{
					//

					vec4 ambient = vec4(colourAmbient.rgb * strengthAmbient, 1.0);
					vec4 colourSample = vec4(1.0, 1.0, 1.0, 1.0);

					vec3 norm = normalize(v_norm);
					vec3 lightDir = normalize(posLight - v_worldPos); 

					float diff = max(dot(norm, lightDir), 0.0);
					vec4 diffuse = diff * vec4(colourLight, 1.0);

					//

					vec3 viewDir = normalize(posView - v_worldPos);
					vec3 reflectDir = reflect(-lightDir, norm); 

					float spec = pow(max(dot(viewDir, reflectDir), 0.0), 2);
					vec4 specular = vec4(specularStrength * spec * colourLight, 1.0); 

					if (UseTexture) 
					{
						colourSample = texture(Texture, v_text);
						FragColor = (ambient + diffuse + specular) * colourSample;
					} 
					else 
					{
						FragColor = (ambient + diffuse + specular) * colourObject;
					}
				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.colourAmbient_scene = self.program['colourAmbient']
		self.strengthAmbient_scene = self.program['strengthAmbient']

		self.posLight_scene = self.program['posLight']
		self.colourLight_scene = self.program['colourLight']

		self.posView_scene = self.program['posView']
		self.specularStrength_scene = self.program['specularStrength']

		#? self.color_scene = self.program['Color']
		self.bUseTexture = self.program['UseTexture']

class hdr2cubemap_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;

				out vec3 v_worldPos;

				void main() 
				{
					gl_Position = projection * view * model * vec4(in_vert, 1.0);

					v_worldPos = vec3(model * vec4(in_vert, 1.0));
				}
			''',
			fragment_shader='''
				#version 330

				const vec2 invAtan = vec2(0.1591, 0.3183);
				
				uniform sampler2D hdrMap;

				in vec3 v_worldPos;
				
				out vec4 FragColor;

				vec2 SampleSphericalMap(vec3 v)
				{
					//vec2 uv = vec2(atan(v.z, v.x), asin(v.y));

					vec2 uv = vec2(atan(v.x, v.y), asin(v.z));
					uv *= invAtan;
					uv += 0.5;

					return uv;
				}

				void main()
				{		
					vec2 uv_coord = SampleSphericalMap(normalize(v_worldPos)); // make sure to normalize localPos
					
					vec3 color = texture(hdrMap, uv_coord).rgb;
					//vec3 color = texture(hdrMap, uv_coord).bgr;

					FragColor = vec4(color, 1.0);
				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.hdr_Map = self.program['hdrMap']

class irradiance_cubemap_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;

				out vec3 v_worldPos;

				void main() 
				{
					gl_Position = projection * view * model * vec4(in_vert, 1.0);

					v_worldPos = vec3(model * vec4(in_vert, 1.0));
				}
			''',
			fragment_shader='''
				#version 330

				const float PI = 3.14159265359;
				
				uniform samplerCube envMap;

				in vec3 v_worldPos;
				
				out vec4 FragColor;

				void main()
				{
					vec3 N = normalize(v_worldPos);

					//

					vec3 irradiance = vec3(0.0);  

					vec3 up    = vec3(0.0, 1.0, 0.0);
					vec3 right = cross(up, N);
					up         = cross(N, right);

					float sampleDelta = 0.025;
					float nrSamples = 0.0; 
					for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
					{
						for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
						{
							// spherical to cartesian (in tangent space)
							vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));

							// tangent space to world
							vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 

							irradiance += texture(envMap, sampleVec).rgb * cos(theta) * sin(theta);
							nrSamples++;
						}
					}
					irradiance = PI * irradiance * (1.0 / float(nrSamples));

					//

					FragColor = vec4(irradiance, 1.0);
				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.env_Map = self.program['envMap']

class prefilter_cubemap_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;

				out vec3 v_worldPos;

				void main() 
				{
					gl_Position = projection * view * model * vec4(in_vert, 1.0);

					v_worldPos = vec3(model * vec4(in_vert, 1.0));
				}
			''',
			fragment_shader='''
				#version 330

				const float PI = 3.14159265359;

				uniform samplerCube envMap;

				uniform float roughness;

				in vec3 v_worldPos;
				
				out vec4 FragColor;

				float VanDerCorpus(uint n, uint base)
				{
					float invBase = 1.0 / float(base);
					float denom   = 1.0;
					float result  = 0.0;

					for(uint i = 0u; i < 32u; ++i)
					{
						if(n > 0u)
						{
							denom   = mod(float(n), 2.0);
							result += denom * invBase;
							invBase = invBase / 2.0;
							n       = uint(float(n) / 2.0);
						}
					}

					return result;
				}

				vec2 HammersleyNoBitOps(uint i, uint N)
				{
					return vec2(float(i)/float(N), VanDerCorpus(i, 2u));
				}

				vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
				{
					float a = roughness*roughness;

					float phi = 2.0 * PI * Xi.x;
					float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
					float sinTheta = sqrt(1.0 - cosTheta*cosTheta);

					// from spherical coordinates to cartesian coordinates
					vec3 H;
					H.x = cos(phi) * sinTheta;
					H.y = sin(phi) * sinTheta;
					H.z = cosTheta;

					// from tangent-space vector to world-space sample vector
					vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
					vec3 tangent   = normalize(cross(up, N));
					vec3 bitangent = cross(N, tangent);

					vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
					return normalize(sampleVec);
				}

				void main()
				{
					vec3 N = normalize(v_worldPos);
					vec3 R = N;
					vec3 V = R;

					const uint SAMPLE_COUNT = 1024u;
					float totalWeight = 0.0;   
					vec3 prefilteredColor = vec3(0.0);     
					for(uint i = 0u; i < SAMPLE_COUNT; ++i)
					{
						vec2 Xi = HammersleyNoBitOps(i, SAMPLE_COUNT); //Hammersley(i, SAMPLE_COUNT);
						vec3 H  = ImportanceSampleGGX(Xi, N, roughness);
						vec3 L  = normalize(2.0 * dot(V, H) * H - V);

						float NdotL = max(dot(N, L), 0.0);
						if(NdotL > 0.0)
						{
							prefilteredColor += texture(envMap, L).rgb * NdotL;
							totalWeight      += NdotL;
						}
					}
					prefilteredColor = prefilteredColor / totalWeight;

					//FragColor = vec4(prefilteredColor.r, prefilteredColor.b, 1.0, 1.0);
					FragColor = vec4(prefilteredColor, 1.0);
				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.env_Map = self.program['envMap']

		self.roughness = self.program['roughness']

class brdf_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;
				in vec2 in_text;

				out vec2 v_text;

				void main() 
				{
					gl_Position = projection * view * model * vec4(in_vert, 1.0);

					v_text = in_text;
				}
			''',
			fragment_shader='''
				#version 330
				
				const float PI = 3.14159265359;

				in vec2 v_text;

				out vec4 FragColor;
				
				// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
				// efficient VanDerCorpus calculation.
				float RadicalInverse_VdC(uint bits) 
				{
					bits = (bits << 16u) | (bits >> 16u);
					bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
					bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
					bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
					bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
					return float(bits) * 2.3283064365386963e-10; // / 0x100000000
				}
				
				vec2 Hammersley(uint i, uint N)
				{
					return vec2(float(i)/float(N), RadicalInverse_VdC(i));
				}
				
				vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
				{
					float a = roughness*roughness;

					float phi = 2.0 * PI * Xi.x;
					float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
					float sinTheta = sqrt(1.0 - cosTheta*cosTheta);

					// from spherical coordinates to cartesian coordinates - halfway vector
					vec3 H;
					H.x = cos(phi) * sinTheta;
					H.y = sin(phi) * sinTheta;
					H.z = cosTheta;

					// from tangent-space H vector to world-space sample vector
					vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
					vec3 tangent   = normalize(cross(up, N));
					vec3 bitangent = cross(N, tangent);

					vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
					return normalize(sampleVec);
				}
				
				float GeometrySchlickGGX(float NdotV, float roughness)
				{
					// note that we use a different k for IBL
					float a = roughness;
					float k = (a * a) / 2.0;

					float nom   = NdotV;
					float denom = NdotV * (1.0 - k) + k;

					return nom / denom;
				}
				
				float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
				{
					float NdotV = max(dot(N, V), 0.0);
					float NdotL = max(dot(N, L), 0.0);
					float ggx2 = GeometrySchlickGGX(NdotV, roughness);
					float ggx1 = GeometrySchlickGGX(NdotL, roughness);

					return ggx1 * ggx2;
				}
				
				vec2 IntegrateBRDF(float NdotV, float roughness)
				{
					vec3 V;
					V.x = sqrt(1.0 - NdotV*NdotV);
					V.y = 0.0;
					V.z = NdotV;

					float A = 0.0;
					float B = 0.0; 

					vec3 N = vec3(0.0, 0.0, 1.0);

					const uint SAMPLE_COUNT = 1024u;
					for(uint i = 0u; i < SAMPLE_COUNT; ++i)
					{
						// generates a sample vector that's biased towards the
						// preferred alignment direction (importance sampling).
						vec2 Xi = Hammersley(i, SAMPLE_COUNT);
						vec3 H = ImportanceSampleGGX(Xi, N, roughness);
						vec3 L = normalize(2.0 * dot(V, H) * H - V);

						float NdotL = max(L.z, 0.0);
						float NdotH = max(H.z, 0.0);
						float VdotH = max(dot(V, H), 0.0);

						if(NdotL > 0.0)
						{
							float G = GeometrySmith(N, V, L, roughness);
							float G_Vis = (G * VdotH) / (NdotH * NdotV);
							float Fc = pow(1.0 - VdotH, 5.0);

							A += (1.0 - Fc) * G_Vis;
							B += Fc * G_Vis;
						}
					}

					A /= float(SAMPLE_COUNT);
					B /= float(SAMPLE_COUNT);

					return vec2(A, B);
				}

				void main() 
				{
					vec2 integratedBRDF = IntegrateBRDF(v_text.x, v_text.y);

					FragColor = vec4(integratedBRDF.x, integratedBRDF.y, 0.0, 1.0);

					// For Debugging only
					//FragColor = (0.01 * integratedBRDF)+ (0.99 * vec4(v_text.x, v_text.y, 0.0, 1.0));
				}
			''',
		)

class skybox_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;

				out vec3 v_worldPos;

				void main() 
				{
					mat4 rotView = mat4(mat3(view)); // remove translation from the view matrix

					vec4 clipPos = projection * rotView * model * vec4(in_vert, 1.0);

					gl_Position = clipPos.xyzw; //clipPos.xyww;

					v_worldPos = vec3(model * vec4(in_vert, 1.0));
				}
			''',
			fragment_shader='''
				#version 330
				
				//uniform sampler2D envMap;
				uniform samplerCube envMap;

				uniform samplerCube irrMap;

				uniform samplerCube prefilteredMap;

				in vec3 v_vert;

				in vec3 v_worldPos;
				
				out vec4 FragColor;

				void main()
				{
					//vec3 envColor = texture(envMap, vec2(v_worldPos.x, v_worldPos.z)).rgb;



					vec3 envColor = texture(envMap, v_worldPos).rgb;

					envColor = envColor / (envColor + vec3(1.0));
					envColor = pow(envColor, vec3(1.0/2.2)); 



					vec3 irrColor = texture(irrMap, v_worldPos).rgb;

					//irrColor = irrColor / (irrColor + vec3(1.0));
					//irrColor = pow(irrColor, vec3(1.0/2.2)); 



					vec3 prefilteredColor = texture(prefilteredMap, v_worldPos).rgb;

					//prefilteredColor = prefilteredColor / (prefilteredColor + vec3(1.0));
					//prefilteredColor = pow(prefilteredColor, vec3(1.0/2.2)); 



					//envColor = ( 0.99 * envColor) + ( 0.01 * irrColor);
					//envColor = ( 0.01 * envColor) + ( 0.99 * irrColor);
					//envColor = ( 0.01 * envColor) + ( 0.01 * irrColor) + ( 0.98 * prefilteredColor);

					envColor = ( 0.98 * envColor) + ( 0.01 * irrColor) + ( 0.01 * prefilteredColor);

					FragColor = vec4(envColor, 1.0);
				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.env_Map = self.program['envMap']
		self.irr_Map = self.program['irrMap']
		self.prefiltered_Map = self.program['prefilteredMap']

class pbr_shader(base_shader):
	def __init__(self, ctx):
		super().__init__()

		self.program = ctx.program(
			vertex_shader='''
				#version 330
				
				uniform mat4 projection;
				uniform mat4 view;
				uniform mat4 model;

				in vec3 in_vert;
				in vec3 in_norm;
				in vec2 in_text;

				out vec3 v_vert;
				out vec3 v_norm;
				out vec2 v_text;

				out vec3 v_worldPos;

				void main() 
				{
					vec4 tmp_worldPos = model * vec4(in_vert, 1.0);

					gl_Position = projection * view * tmp_worldPos;

					v_vert = in_vert;
					v_norm = in_norm;
					v_text = in_text;

					v_worldPos = tmp_worldPos.xyz;
				}
			''',
			
			fragment_shader='''
				#version 330

				const float PI = 3.14159265359;

				////

				in vec3 v_vert;
				in vec3 v_norm;
				in vec2 v_text;

				in vec3 v_worldPos;

				////

				uniform vec3 posView;

				////

				// material parameters
				uniform sampler2D albedoMap;
				uniform sampler2D normalMap;
				uniform sampler2D metallicMap;
				uniform sampler2D roughnessMap;
				uniform sampler2D aoMap;

				uniform sampler2D emissiveMap;

				//
				uniform samplerCube irradianceMap;
				uniform samplerCube prefilterMap;
				uniform sampler2D brdfLUT;

				////

				// lights
				uniform vec3 posLights[4];
				uniform vec3 colourLights[4];

				////

				out vec4 FragColor;

				// ----------------------------------------------------------------------------
				// Easy trick to get tangent-normals to world-space to keep PBR code simplified.
				// Don't worry if you don't get what's going on; you generally want to do normal 
				// mapping the usual way for performance anways; I do plan make a note of this 
				// technique somewhere later in the normal mapping tutorial.
				vec3 getNormalFromMap()
				{
					vec3 tangentNormal = texture(normalMap, v_text).xyz * 2.0 - 1.0;

					vec3 Q1  = dFdx(v_worldPos);
					vec3 Q2  = dFdy(v_worldPos);
					vec2 st1 = dFdx(v_text);
					vec2 st2 = dFdy(v_text);

					vec3 N   = normalize(v_norm);
					vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
					vec3 B  = -normalize(cross(N, T));
					mat3 TBN = mat3(T, B, N);

					return normalize(TBN * tangentNormal);
				}

				// ----------------------------------------------------------------------------
				float DistributionGGX(vec3 N, vec3 H, float roughness)
				{
					float a = roughness*roughness;
					float a2 = a*a;
					float NdotH = max(dot(N, H), 0.0);
					float NdotH2 = NdotH*NdotH;

					float nom   = a2;
					float denom = (NdotH2 * (a2 - 1.0) + 1.0);
					denom = PI * denom * denom;

					return nom / denom;
				}

				// ----------------------------------------------------------------------------
				float GeometrySchlickGGX(float NdotV, float roughness)
				{
					float r = (roughness + 1.0);
					float k = (r*r) / 8.0;

					float nom   = NdotV;
					float denom = NdotV * (1.0 - k) + k;

					return nom / denom;
				}

				// ----------------------------------------------------------------------------
				float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
				{
					float NdotV = max(dot(N, V), 0.0);
					float NdotL = max(dot(N, L), 0.0);
					float ggx2 = GeometrySchlickGGX(NdotV, roughness);
					float ggx1 = GeometrySchlickGGX(NdotL, roughness);

					return ggx1 * ggx2;
				}

				// ----------------------------------------------------------------------------
				vec3 fresnelSchlick(float cosTheta, vec3 F0)
				{
					return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
				}

				// ----------------------------------------------------------------------------
				vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
				{
					return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
				}

				// ----------------------------------------------------------------------------
				void main()
				{		
					vec3 albedo     = pow(texture(albedoMap, v_text).rgb, vec3(2.2));
					float ao        = texture(aoMap, v_text).r;
					float roughness = texture(roughnessMap, v_text).g;
					float metallic  = texture(metallicMap, v_text).b;
					vec3 emissive   = texture(emissiveMap, v_text).rgb;

					vec3 N = getNormalFromMap();
					vec3 V = normalize(posView - v_worldPos);
					vec3 R = reflect(-V, N);

					// calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
					// of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
					vec3 F0 = vec3(0.04); 
					F0 = mix(F0, albedo, metallic);

					// reflectance equation
					vec3 Lo = vec3(0.0);
					for(int i = 0; i < 4; ++i) 
					{
						// calculate per-light radiance
						vec3 L = normalize(posLights[i] - v_worldPos);
						vec3 H = normalize(V + L);
						float distance = length(posLights[i] - v_worldPos);
						float attenuation = 1.0 / (distance * distance);
						vec3 radiance = colourLights[i] * attenuation;

						// Cook-Torrance BRDF
						float NDF = DistributionGGX(N, H, roughness);   
						float G   = GeometrySmith(N, V, L, roughness);      
						vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
						
						vec3 nominator    = NDF * G * F; 
						float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
						vec3 specular = nominator / denominator;
						
						// kS is equal to Fresnel
						vec3 kS = F;
						// for energy conservation, the diffuse and specular light can't
						// be above 1.0 (unless the surface emits light); to preserve this
						// relationship the diffuse component (kD) should equal 1.0 - kS.
						vec3 kD = vec3(1.0) - kS;
						// multiply kD by the inverse metalness such that only non-metals 
						// have diffuse lighting, or a linear blend if partly metal (pure metals
						// have no diffuse light).
						kD *= 1.0 - metallic;	  

						// scale light by NdotL
						float NdotL = max(dot(N, L), 0.0);        

						// add to outgoing radiance Lo
						Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
					}   
					
					// ambient lighting (we now use IBL as the ambient term)
					vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

					vec3 kS = F;
					vec3 kD = 1.0 - kS;
					kD *= 1.0 - metallic;	  

					vec3 irradiance = texture(irradianceMap, N).rgb;
					vec3 diffuse = irradiance * albedo;

					// sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
					//const float MAX_REFLECTION_LOD = 4.0;
					//vec3 prefilteredColor = textureLod(prefilterMap, R,  roughness * MAX_REFLECTION_LOD).rgb; 

					vec3 prefilteredColor = texture(prefilterMap, R).rgb;

					vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
					vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

					vec3 ambient = (kD * diffuse + specular) * ao;

					vec3 color = ambient + Lo + emissive;

					// HDR tonemapping
					color = color / (color + vec3(1.0));

					// gamma correct
					color = pow(color, vec3(1.0/2.2)); 

					FragColor = vec4(color , 1.0);

				}
			''',
		)

	def associateUniforms(self):
		super().associateUniforms()

		self.posView_scene = self.program['posView']

		self.albedo_Map = self.program['albedoMap']
		self.normal_Map = self.program['normalMap']
		self.metallic_Map = self.program['metallicMap']
		self.roughness_Map = self.program['roughnessMap']
		self.ao_Map = self.program['aoMap']
		self.emissive_Map = self.program['emissiveMap']

		self.irradiance_Map = self.program['irradianceMap']
		self.prefilter_Map = self.program['prefilterMap']
		self.brdf_LUT = self.program['brdfLUT']

		self.posLights = self.program['posLights']
		self.colourLights = self.program['colourLights']

class base_node(object):
	def __init__(self, vPos, qRot):

		self.vPos = vPos
		self.qRot = qRot

		self.model_matrix = Matrix44.from_translation(vPos)
		q_tmp = quaternion.inverse(qRot)
		self.model_matrix = self.model_matrix * Matrix44.from_quaternion(q_tmp) 

	def moveForward(self, incr):
		move_matrix = Matrix44.from_translation([0.0, 0.0, -incr, 1.0])
		self.model_matrix = self.model_matrix * move_matrix

	def moveBackward(self, incr):
		move_matrix = Matrix44.from_translation([0.0, 0.0, incr, 1.0])
		self.model_matrix = self.model_matrix * move_matrix
		
	def rotateX(self, incr):
		rotate_matrix = Matrix44.from_x_rotation(np.sin(incr))
		self.model_matrix = self.model_matrix * rotate_matrix

	def rotateY(self, incr):
		rotate_matrix = Matrix44.from_y_rotation(np.sin(incr))
		self.model_matrix = self.model_matrix * rotate_matrix

	def rotateZ(self, incr):
		rotate_matrix = Matrix44.from_z_rotation(np.sin(incr))
		self.model_matrix = self.model_matrix * rotate_matrix

	def render(self):
		pass

class renderable_node(base_node):
	def __init__(self, vPos, qRot, oVAO, oShader):

		super().__init__(vPos, qRot)

		self.oVAO = oVAO
		self.oShader = oShader

	def render(self, typeOfPrimitive):
		self.oShader.update_model_matrix(self.model_matrix)
		self.oVAO.render(typeOfPrimitive) 

class grid_node(renderable_node):
	def __init__(self, vPos, qRot, ctx, oShader, colour_grid):

		byte_grid_points = self.grid(20, 20).astype('f4').tobytes()
		vbo_grid = ctx.buffer(byte_grid_points)
		oVAO = ctx.simple_vertex_array(oShader.program, vbo_grid, 'in_vert')

		self.colour_grid = colour_grid

		super().__init__(vPos, qRot, oVAO, oShader)

	def render(self):
		self.oShader.colour_Object.value = self.colour_grid

		super().render(moderngl.LINES)

	def grid(self, size, steps):

		u = np.repeat(np.linspace(-size/2, size/2, steps+1), 2)
		#print("u : ", u)
		v = np.tile([-size/2, size/2], steps+1)
		#print("v : ", v)
		w = np.zeros((steps+1) * 2)
		return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])

class light_node(renderable_node):
	def __init__(self, vPos, qRot, oVAO, oShader, colour_Light):
		super().__init__(vPos, qRot, oVAO, oShader)

		self.colour_Light = colour_Light

	def render(self):
		self.oShader.colour_Light.value = self.colour_Light

		super().render(moderngl.TRIANGLES)

class object_node(renderable_node):
	def __init__(self, vPos, qRot, oVAO, oShader):
		super().__init__(vPos, qRot, oVAO, oShader)

class debug_object_node(object_node):
	def __init__(self, vPos, qRot, oVAO, oShader):
		super().__init__(vPos, qRot, oVAO, oShader)

	def render(self, typeOfPrimitive=moderngl.TRIANGLE_STRIP):
		super().render(typeOfPrimitive)

class typical_object_node(object_node):
	def __init__(self, vPos, qRot, oVAO, oShader, oTexture):
		super().__init__(vPos, qRot, oVAO, oShader)

		self.oTexture = oTexture

	def render(self, typeOfPrimitive=moderngl.TRIANGLES):
		self.oTexture.use()

		super().render(typeOfPrimitive)

class skybox_object_node(object_node):
	def __init__(self, vPos, qRot, oVAO, oShader, oTexEnvMap, oTexIrrMap, oTexPreFilteredMap):
		super().__init__(vPos, qRot, oVAO, oShader)

		self.oTexEnvMap = oTexEnvMap
		self.oTexIrrMap = oTexIrrMap
		self.oTexPreFilteredMap = oTexPreFilteredMap

	def render(self, typeOfPrimitive=moderngl.TRIANGLES):

		nTexLoc = 0

		self.oTexEnvMap.use(nTexLoc)
		self.oShader.env_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexIrrMap.use(nTexLoc)
		self.oShader.irr_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexPreFilteredMap.use(nTexLoc)
		self.oShader.prefiltered_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		super().render(typeOfPrimitive)

class fbo_node(typical_object_node):

	def __init__(self, vPos, qRot, ctx, oShader, nBufWidth, nBufHeight , nWidth, nHeight):

		self.ctx = ctx

		self.texSize = (nBufWidth, nBufHeight)
		textureFbo = self.ctx.texture(self.texSize, 3)
		textureFbo.repeat_x = False
		textureFbo.repeat_y = False
		self.depthAttachment = self.ctx.depth_renderbuffer(self.texSize)
		self.oFBO = self.ctx.framebuffer(textureFbo, self.depthAttachment)
		
		vertices_quad = np.array([
			0.0, 0.0,			0.0,1.0,
			0.0, nHeight,		0.0,0.0,
			nWidth, 0.0,		1.0,1.0,
			nWidth, nHeight,	1.0,0.0
		])
		vbo_quad = self.ctx.buffer(vertices_quad.astype('f4').tobytes())
		vao_quad = self.ctx.simple_vertex_array(oShader.program, vbo_quad, 'in_vert', 'in_text')

		super().__init__(vPos, qRot, vao_quad, oShader, textureFbo)

	def render(self):
		super().render(moderngl.TRIANGLE_STRIP)

class cubemap_preprocess_node(renderable_node):
	def __init__(self, vPos, qRot, ctx, oVAO, oShader, nWidth, nHeight):
		
		super().__init__(vPos, qRot, oVAO, oShader)

		# pbr: setup cubemap to render to and attach to framebuffer
		
		self.nWidth = nWidth
		self.nHeight = nHeight

		self.texSize = (nWidth, nHeight)
		#self.textureFbo_Cube = ctx.texture_cube(self.texSize, 3)

		self.ctx = ctx

		# pbr: set up projection and view matrices for capturing data onto the 6 cubemap face directions

		self.p_env = Matrix44()
		self.proj_env_mono = Matrix44.perspective_projection(90.0, 1.0, 0.1, 100.0)

		#

		self.arrFace = []

		#

		self.textureFbo_Up = ctx.texture(self.texSize, 3)
		self.textureFbo_Up.repeat_x = False
		self.textureFbo_Up.repeat_y = False
		self.depthAttachment_Up = ctx.depth_renderbuffer(self.texSize)
		self.oFBO_Up = ctx.framebuffer(self.textureFbo_Up, self.depthAttachment_Up)

		qUp = Quaternion([1.0, 0.0, 0.0, 0.0]) 
		tmpRot = Quaternion.from_z_rotation(0.0)
		qUp = qUp * tmpRot
		self.oCameraUp = camera_node(	Vector4([0.0, 0.0, 0.0, 1.0]), qUp)
		self.view_Up = matrix44.inverse(self.oCameraUp.model_matrix)
		self.byte_view_matrix_Up = self.view_Up.astype('f4').tobytes()

		self.arrFace.append((self.oFBO_Up,self.byte_view_matrix_Up))

		#

		self.textureFbo_Down = ctx.texture(self.texSize, 3)
		self.textureFbo_Down.repeat_x = False
		self.textureFbo_Down.repeat_y = False
		self.depthAttachment_Down = ctx.depth_renderbuffer(self.texSize)
		self.oFBO_Down = ctx.framebuffer(self.textureFbo_Down, self.depthAttachment_Down)

		qDown = Quaternion([0.0, 0.0, 0.0, 1.0])
		tmpRot = Quaternion.from_z_rotation(np.pi)
		qDown = qDown * tmpRot
		self.oCameraDown = camera_node(	Vector4([0.0, 0.0, 0.0, 1.0]), qDown)
		self.view_Down = matrix44.inverse(self.oCameraDown.model_matrix)
		self.byte_view_matrix_Down = self.view_Down.astype('f4').tobytes()

		self.arrFace.append((self.oFBO_Down,self.byte_view_matrix_Down))

		#

		self.textureFbo_Front = ctx.texture(self.texSize, 3)
		self.textureFbo_Front.repeat_x = False
		self.textureFbo_Front.repeat_y = False
		self.depthAttachment_Front = ctx.depth_renderbuffer(self.texSize)
		self.oFBO_Front = ctx.framebuffer(self.textureFbo_Front, self.depthAttachment_Front)

		qFront = Quaternion(([0.707, 0.0, 0.0, 0.707]))
		tmpRot = Quaternion.from_z_rotation(0.0)
		qFront = qFront * tmpRot
		self.oCameraFront = camera_node( Vector4([0.0, 0.0, 0.0, 1.0]), qFront)
		self.view_Front = matrix44.inverse(self.oCameraFront.model_matrix)
		self.byte_view_matrix_Front = self.view_Front.astype('f4').tobytes()

		self.arrFace.append((self.oFBO_Front,self.byte_view_matrix_Front))

		#

		self.textureFbo_Back = ctx.texture(self.texSize, 3)
		self.textureFbo_Back.repeat_x = False
		self.textureFbo_Back.repeat_y = False
		self.depthAttachment_Back = ctx.depth_renderbuffer(self.texSize)
		self.oFBO_Back = ctx.framebuffer(self.textureFbo_Back, self.depthAttachment_Back)

		qBack = Quaternion([0.0, 0.707, 0.707, 0.0])
		tmpRot = Quaternion.from_z_rotation(np.pi)
		qBack = qBack * tmpRot
		self.oCameraBack = camera_node(	Vector4([0.0, 0.0, 0.0, 1.0]), qBack)
		self.view_Back = matrix44.inverse(self.oCameraBack.model_matrix)
		self.byte_view_matrix_Back = self.view_Back.astype('f4').tobytes()

		self.arrFace.append((self.oFBO_Back,self.byte_view_matrix_Back))

		#

		self.textureFbo_Left = ctx.texture(self.texSize, 3)
		self.textureFbo_Left.repeat_x = False
		self.textureFbo_Left.repeat_y = False
		self.depthAttachment_Left= ctx.depth_renderbuffer(self.texSize)
		self.oFBO_Left = ctx.framebuffer(self.textureFbo_Left, self.depthAttachment_Left)

		qLeft = Quaternion([0.5, 0.5, 0.5, 0.5]) 
		tmpRot = Quaternion.from_z_rotation(np.pi/2.0)
		qLeft = qLeft * tmpRot
		self.oCameraLeft = camera_node(	Vector4([0.0, 0.0, 0.0, 1.0]), qLeft)
		self.view_Left = matrix44.inverse(self.oCameraLeft.model_matrix)
		self.byte_view_matrix_Left = self.view_Left.astype('f4').tobytes()

		self.arrFace.append((self.oFBO_Left,self.byte_view_matrix_Left))

		#

		self.textureFbo_Right = ctx.texture(self.texSize, 3)
		self.textureFbo_Right.repeat_x = False
		self.textureFbo_Right.repeat_y = False
		self.depthAttachment_Right = ctx.depth_renderbuffer(self.texSize)
		self.oFBO_Right = ctx.framebuffer(self.textureFbo_Right, self.depthAttachment_Right)

		qRight = Quaternion([-0.5, 0.5, 0.5, -0.5]) 
		tmpRot = Quaternion.from_z_rotation(-np.pi/2.0)
		qRight = qRight * tmpRot
		self.oCameraRight = camera_node( Vector4([0.0, 0.0, 0.0, 1.0]), qRight)
		self.view_Right = matrix44.inverse(self.oCameraRight.model_matrix)
		self.byte_view_matrix_Right = self.view_Right.astype('f4').tobytes()

		self.arrFace.append((self.oFBO_Right,self.byte_view_matrix_Right))

		##

		self.rootObjects_env = {}

	def render(self):

		# pbr: convert HDR equirectangular environment map to cubemap equivalent

		for sampleFBO, sample_byte_view_matrix in self.arrFace:

			sampleFBO.use()
			sampleFBO.clear(0.4, 0.4, 0.4, 1.0)

			self.ctx.enable(moderngl.BLEND)
			self.ctx.enable(moderngl.DEPTH_TEST)
			#self.ctx.enable(moderngl.MULTISAMPLE)
			
			copyM44(self.p_env, self.proj_env_mono)
			byte_proj_matrix = self.p_env.astype('f4').tobytes()

			# Render the environment 

			self.oShader.update_proj_matrix_bytes(byte_proj_matrix)
			self.oShader.update_view_matrix_bytes(sample_byte_view_matrix)
			
			self.associateTextures()

			super().render(moderngl.TRIANGLES) 

		self.ctx.screen.use()

	def associateTextures(self):
		pass

class hdr2cubemap_preprocess_node(cubemap_preprocess_node):
	def __init__(self, vPos, qRot, ctx, oVAO, oShader, oTexHdrMap, nWidth, nHeight):

		self.oTexHdrMap = oTexHdrMap

		super().__init__(vPos, qRot, ctx, oVAO, oShader, nWidth, nHeight)

	def associateTextures(self):
		self.oShader.hdr_Map.value = 0
		self.oTexHdrMap.use(0)

class envmap_preprocess_node(cubemap_preprocess_node):
	def __init__(self, vPos, qRot, ctx, oVAO, oShader, oTexEnvMap, nWidth, nHeight):

		self.oTexEnvMap = oTexEnvMap

		super().__init__(vPos, qRot, ctx, oVAO, oShader, nWidth, nHeight)

	def associateTextures(self):
		self.oShader.env_Map.value = 0
		self.oTexEnvMap.use(0)

class irradiancemap_preprocess_node(envmap_preprocess_node):
	def __init__(self, vPos, qRot, ctx, oVAO, oShader, oTexEnvMap, nWidth, nHeight):
		super().__init__(vPos, qRot, ctx, oVAO, oShader, oTexEnvMap, nWidth, nHeight)

class prefiltermap_preprocess_node(envmap_preprocess_node):
	def __init__(self, vPos, qRot, ctx, oVAO, oShader, oTexEnvMap, nWidth, nHeight):
		super().__init__(vPos, qRot, ctx, oVAO, oShader, oTexEnvMap, nWidth, nHeight)

class brdf_preprocess_node(typical_object_node):

	def __init__(self, vPos, qRot, ctx, oShader, nWidth, nHeight):

		left = -1.0
		right = 1.0
		top = 1.0
		bottom = -1.0
		near = 1.0
		far = 1000.0

		self.proj_brdf = Matrix44.orthogonal_projection(left, right, bottom, top, near, far)

		oCamera = camera_node(	Vector4([0.0, 0.0, 10.0, 1.0]), 
								Quaternion([0.0, 0.0, 0.0, 1.0]) )

		self.view_brdf = matrix44.inverse(oCamera.model_matrix)

		#

		self.byte_proj_brdf_matrix = self.proj_brdf.astype('f4').tobytes()
		self.byte_view_brdf_matrix = self.view_brdf.astype('f4').tobytes()

		#

		self.ctx = ctx

		self.texSize = (nWidth, nHeight)
		textureFbo = self.ctx.texture(self.texSize, 3)
		textureFbo.repeat_x = False
		textureFbo.repeat_y = False
		self.depthAttachment = self.ctx.depth_renderbuffer(self.texSize)
		self.oFBO = self.ctx.framebuffer(textureFbo, self.depthAttachment)
		
		vertices_quad = np.array([
			-1.0, 1.0,  0.0,			0.0,1.0,
			-1.0, -1.0,  0.0,			0.0,0.0,
			 1.0, 1.0,  0.0,			1.0,1.0,
			 1.0, -1.0,  0.0,			1.0,0.0
		])
		vbo_quad = self.ctx.buffer(vertices_quad.astype('f4').tobytes())
		vao_quad = self.ctx.simple_vertex_array(oShader.program, vbo_quad, 'in_vert', 'in_text')

		super().__init__(vPos, qRot, vao_quad, oShader, textureFbo)

	def render(self):

		self.oFBO.use()
		self.oFBO.clear(0.4, 0.4, 0.4, 1.0)

		self.ctx.enable(moderngl.BLEND)
		self.ctx.enable(moderngl.DEPTH_TEST)
		#self.ctx.enable(moderngl.MULTISAMPLE)

		# Render the environment 

		self.oShader.update_proj_matrix_bytes(self.byte_proj_brdf_matrix)
		self.oShader.update_view_matrix_bytes(self.byte_view_brdf_matrix)
		
		super().render(moderngl.TRIANGLE_STRIP)

		self.ctx.screen.use()

class pbr_object_node(object_node):
	def __init__(	self, vPos, qRot, oVAO, oShader, 
					oTexAlbedoMap, oTexNormalMap, oTexMetallicMap, oTexRoughnessMap, oTexAoMap, oTexEmissiveMap,
					oTexIrradianceMap, oTexPrefilterMap, oTexBrdfLUT):
		super().__init__(vPos, qRot, oVAO, oShader)

		self.oTexAlbedoMap = oTexAlbedoMap
		self.oTexNormalMap = oTexNormalMap
		self.oTexMetallicMap = oTexMetallicMap
		self.oTexRoughnessMap = oTexRoughnessMap
		self.oTexAoMap = oTexAoMap
		self.oTexEmissiveMap = oTexEmissiveMap

		self.oTexIrradianceMap = oTexIrradianceMap
		self.oTexPrefilterMap = oTexPrefilterMap
		self.oTexBrdfLUT = oTexBrdfLUT

	def render(self):

		nTexLoc = 0
		
		self.oTexAlbedoMap.use(nTexLoc)
		self.oShader.albedo_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexNormalMap.use(nTexLoc)
		self.oShader.normal_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexMetallicMap.use(nTexLoc)
		self.oShader.metallic_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexRoughnessMap.use(nTexLoc)
		self.oShader.roughness_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexAoMap.use(nTexLoc)
		self.oShader.ao_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexEmissiveMap.use(nTexLoc)
		self.oShader.emissive_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexIrradianceMap.use(nTexLoc)
		self.oShader.irradiance_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexPrefilterMap.use(nTexLoc)
		self.oShader.prefilter_Map.value = nTexLoc
		nTexLoc = nTexLoc + 1

		self.oTexBrdfLUT.use(nTexLoc)
		self.oShader.brdf_LUT.value = nTexLoc
		nTexLoc = nTexLoc + 1

		super().render(moderngl.TRIANGLES)

class camera_node(base_node):
	def __init__(self, vPos, qRot):
		super().__init__(vPos, qRot)

class xrApp(object):
	def __init__(	self, 
					nWidth=1280, nHeight=720, 
					nOffsetX=0, nOffsetY=0, 
					ipd_l_t=(1.0, 0.0, 0.0), ipd_r_t=(-1.0, 0.0, 0.0), 
					bForceRedraw=False):

		self.nWidth = nWidth
		self.nHeight = nHeight
		
		self.nOffsetX = nOffsetX
		self.nOffsetY = nOffsetY

		self.ipd_l_t = ipd_l_t
		self.ipd_r_t = ipd_r_t

		self.bVsync = True

		self.window_name = "prototype"
		self.window = None

		self.bRedraw = False
		self.bForceRedraw = bForceRedraw

		'''
		self.impl = None
		'''

		self.ctx = None

		self.eVisualMode = visualMode.MONO
		#self.eVisualMode = visualMode.STEREO

		self.currSelection = None

		#self.time = 0
		#self.start_time = time.clock()
		
	def toggle_visual_mode(self):
		if visualMode.MONO == self.eVisualMode:
			self.eVisualMode = visualMode.STEREO
		elif visualMode.STEREO == self.eVisualMode:
			self.eVisualMode = visualMode.DEBUG_TEXTURE
		elif visualMode.DEBUG_TEXTURE == self.eVisualMode:
			self.eVisualMode = visualMode.MONO

		self.bRedraw = True

	def key_event_callback(self, window, key, scancode, action, mods):

		# The well-known standard key for quick exit
		if key == glfw.KEY_ESCAPE:
			glfw.set_window_should_close(self.window, True)
			return

		####

		# Toggle visual mode
		if glfw.KEY_G == key and glfw.RELEASE == action :
			self.toggle_visual_mode()

		if glfw.KEY_W == key and glfw.RELEASE == action :
			self.oCameraDefault.moveForward(0.2)

			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True

		if glfw.KEY_S == key and glfw.RELEASE == action :
			self.oCameraDefault.moveBackward(0.2)

			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True

		####

		if glfw.KEY_J == key and glfw.RELEASE == action :
			self.oCameraDefault.rotateY(-0.05)

			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True

		if glfw.KEY_L == key and glfw.RELEASE == action :
			self.oCameraDefault.rotateY(0.05)

			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True

		if glfw.KEY_I == key and glfw.RELEASE == action :
			self.oCameraDefault.rotateX(-0.05)

			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True

		if glfw.KEY_K == key and glfw.RELEASE == action :
			self.oCameraDefault.rotateX(0.05)

			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True

		if glfw.KEY_U == key and glfw.RELEASE == action :
			self.oCameraDefault.rotateZ(-0.05)

			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True

		if glfw.KEY_O == key and glfw.RELEASE == action :
			self.oCameraDefault.rotateZ(0.05)
			
			self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
			self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

			self.bRedraw = True	
		####

		if glfw.KEY_LEFT == key and glfw.RELEASE == action :
			self.currSelection.rotateZ(-0.05)
			self.bRedraw = True

		if glfw.KEY_RIGHT == key and glfw.RELEASE == action :
			self.currSelection.rotateZ(0.05)
			self.bRedraw = True

		if glfw.KEY_UP == key and glfw.RELEASE == action :
			self.currSelection.rotateX(-0.05)
			self.bRedraw = True

		if glfw.KEY_DOWN == key and glfw.RELEASE == action :
			self.currSelection.rotateX(0.05)
			self.bRedraw = True

		'''
		# Toggle pause time
		if key == self.keys.SPACE and action == self.keys.ACTION_PRESS:
			self.timer.toggle_pause()

		# Camera movement
		# Right
		if key == self.keys.D:
			if action == self.keys.ACTION_PRESS:
				self.sys_camera.move_right(True)
			elif action == self.keys.ACTION_RELEASE:
				self.sys_camera.move_right(False)
		# Left
		elif key == self.keys.A:
			if action == self.keys.ACTION_PRESS:
				self.sys_camera.move_left(True)
			elif action == self.keys.ACTION_RELEASE:
				self.sys_camera.move_left(False)
		# Forward
		elif key == self.keys.W:
			if action == self.keys.ACTION_PRESS:
				self.sys_camera.move_forward(True)
			if action == self.keys.ACTION_RELEASE:
				self.sys_camera.move_forward(False)
		# Backwards
		elif key == self.keys.S:
			if action == self.keys.ACTION_PRESS:
				self.sys_camera.move_backward(True)
			if action == self.keys.ACTION_RELEASE:
				self.sys_camera.move_backward(False)
		
		# UP
		elif key == self.keys.Q:
			if action == self.keys.ACTION_PRESS:
				self.sys_camera.move_down(True)
			if action == self.keys.ACTION_RELEASE:
				self.sys_camera.move_down(False)

		# Down
		elif key == self.keys.E:
			if action == self.keys.ACTION_PRESS:
				self.sys_camera.move_up(True)
			if action == self.keys.ACTION_RELEASE:
				self.sys_camera.move_up(False)
		'''
	
	def start(self):

		if not glfw.init():
			print("Could not initialize OpenGL context")
			exit(1)

		glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
		glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
		glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
		glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

		#glfw.window_hint(glfw.RESIZABLE, self.resizable)
		glfw.window_hint(glfw.DOUBLEBUFFER, True)
		glfw.window_hint(glfw.DEPTH_BITS, 24)
		glfw.window_hint(glfw.SAMPLES, 2)

		glfw.window_hint(glfw.DECORATED, False)

		monitor = None
		
		'''
		if self.fullscreen:
			# Use the primary monitors current resolution
			monitor = glfw.get_primary_monitor()
			self.width, self.height = mode.size.width, mode.size.height
			mode = glfw.get_video_mode(monitor)
		'''

		self.window = glfw.create_window(self.nWidth, self.nHeight, self.window_name, monitor, None)
		
		if not self.window:
			glfw.terminate()
			raise ValueError("Failed to create window")
		
		'''
		if not self.cursor:
			glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
		'''

		glfw.set_window_pos(self.window, self.nOffsetX, self.nOffsetY)

		# Get the actual buffer size of the window
		# This is important for some displays like Apple's Retina as reported window sizes are virtual
		self.buffer_width, self.buffer_height = glfw.get_framebuffer_size(self.window)
		#print("Frame buffer size:", self.buffer_width, self.buffer_height)
		#print("Actual window size:", glfw.get_window_size(self.window))

		glfw.make_context_current(self.window)

		# The number of screen updates to wait from the time glfwSwapBuffers
		# was called before swapping the buffers and returning
		if self.bVsync:
			glfw.swap_interval(1)

		glfw.set_key_callback(self.window, self.key_event_callback)
		#glfw.set_cursor_pos_callback(self.window, self.mouse_event_callback)
		#glfw.set_window_size_callback(self.window, self.window_resize_callback)

		# Create mederngl context from existing context
		self.ctx = moderngl.create_context(require=330)
		self.fbo = self.ctx.screen
		
		self.rootObjects_pbr = {}
		self.lights = {}
		self.dictVAOs = {}

		self.prepareShaders()
		self.prepareFBOs(self.buffer_width, self.buffer_height)
		self.prepareScene()

		self.prepareDebug()

		self.prepareEnv()
		self.loadOBJs()

	def prepareShaders(self):

		self.oFlatShader = flat_shader(self.ctx)
		self.oFlatShader.associateUniforms()

		self.oDebugShader = debug_shader(self.ctx)
		self.oDebugShader.associateUniforms()

		self.oFboShader = fbo_shader(self.ctx)
		self.oFboShader.associateUniforms()

		self.oGridShader = grid_shader(self.ctx)
		self.oGridShader.associateUniforms()

		self.oLightShader = light_shader(self.ctx)
		self.oLightShader.associateUniforms()

		self.oRegularShader = regular_shader(self.ctx)
		self.oRegularShader.associateUniforms()

		self.oPbrShader = pbr_shader(self.ctx)
		self.oPbrShader.associateUniforms()

		self.oHdr2CubemapShader = hdr2cubemap_shader(self.ctx)
		self.oHdr2CubemapShader.associateUniforms()

		self.oIrradianceCubemapShader = irradiance_cubemap_shader(self.ctx)
		self.oIrradianceCubemapShader.associateUniforms()

		self.oPreFilterCubemapShader = prefilter_cubemap_shader(self.ctx)
		self.oPreFilterCubemapShader.associateUniforms()

		self.oBrdfShader = brdf_shader(self.ctx)
		self.oBrdfShader.associateUniforms()

		self.oSkyboxShader = skybox_shader(self.ctx)
		self.oSkyboxShader.associateUniforms()
	
	def prepareFBOs(self, buffer_width, buffer_height):

		# * Ortho scene for FBO Setup

		left = 0 #-float(self.width)
		right = float(self.nWidth)
		top = 0 #float(self.height) 
		bottom = -float(self.nHeight) 
		near = 1.0
		far = 1000.0
		self.projFBO = Matrix44.orthogonal_projection(left, right, bottom, top, near, far)

		#

		self.oCameraFBO = camera_node(	Vector4([0.0, 0.0, -10.0, 1.0]), 
										Quaternion([1.0, 0.0, 0.0, 0.0]) )

		self.viewFBO = matrix44.inverse(self.oCameraFBO.model_matrix)

		#

		self.ipd_l = Matrix44.from_translation(Vector4([ self.ipd_l_t[0], self.ipd_l_t[1], self.ipd_l_t[2], 1.0]))
		self.ipd_r = Matrix44.from_translation(Vector4([ self.ipd_r_t[0], self.ipd_r_t[1], self.ipd_r_t[2], 1.0]))

		#

		qLeftFBO = quaternion.inverse([0.0, 0.0, 0.0, 1.0])
		self.oLeftFBO = fbo_node(	Vector4([0.0, 0.0, 0.0, 1.0]), qLeftFBO, 
									self.ctx, self.oFboShader, 
									buffer_width//2, buffer_height,
									self.nWidth//2, self.nHeight)

		qRightFBO = quaternion.inverse([0.0, 0.0, 0.0, 1.0])
		self.oRightFBO = fbo_node(	Vector4([self.nWidth/2, 0.0, 0.0, 1.0]), qRightFBO, 
									self.ctx, self.oFboShader, 
									buffer_width//2, buffer_height,
									self.nWidth//2, self.nHeight)

	def prepareScene(self):

		# * Persp setup for Scene

		self.p_scene = Matrix44()
		self.proj_scene_mono = Matrix44.perspective_projection(45.0, self.nWidth / self.nHeight, 0.1, 1000.0)
		self.proj_scene_stereo = Matrix44.perspective_projection(45.0, (self.nWidth/2) / self.nHeight, 0.1, 1000.0)

		#

		self.oCameraDefault = camera_node(	Vector4([3.0, 3.0, 3.0, 1.0]), 
											Quaternion([0.174, 0.419, 0.823,  0.341]) )
		
		self.view_sample = matrix44.inverse(self.oCameraDefault.model_matrix)
		self.byte_view_matrix = self.view_sample.astype('f4').tobytes()

		#

		self.m_grid = matrix44.create_identity()

		self.model_refaxis = Matrix44.from_translation(Vector4([0.0, 0.0, 0.0, 1.0]))

		self.colour_ambient = (1.0, 1.0, 1.0)
		self.strength_ambient = 0.25

		self.pos_View = (0.0, 0.0, 0.0)
		self.strength_specular = 0.5

		#

		self.oGrid = grid_node(		Vector4([0.0, 0.0, 0.0, 1.0]), 
									Quaternion([0.000, 0.0, 0.0, 1.0]),
									self.ctx, self.oGridShader,
									(0.32, 0.32, 0.32, 1.0))

	def prepareDebug(self):

		# * Ortho scene for Debug Setup

		#left = -float(self.nWidth)/2
		#right = float(self.nWidth)/2
		#top = float(self.nHeight)/2
		#bottom = -float(self.nHeight)/2 

		left = 0 #-float(self.width)
		right = float(self.nWidth)
		top = 0 #float(self.height) 
		bottom = -float(self.nHeight) 

		#left = 0 + float(self.nWidth)/2 #-float(self.width)
		#right = float(self.nWidth) + float(self.nWidth)/2
		#top = 0 - float(self.nHeight)/2 #float(self.height) 
		#bottom = -float(self.nHeight) - float(self.nHeight)/2

		near = 1.0
		far = 1000.0

		self.projDebug = Matrix44.orthogonal_projection(left, right, bottom, top, near, far)

		#

		self.oCameraDebug = camera_node(	Vector4([0.0, 0.0, -10.0, 1.0]), 
										Quaternion([1.0, 0.0, 0.0, 0.0]) )

		self.viewDebug = matrix44.inverse(self.oCameraDebug.model_matrix)

		#
		
		name = 'debugplane'

		vertices_quad = np.array([
			0.0, 0.0,					0.0,1.0,
			0.0, self.nHeight,			0.0,0.0,
			self.nWidth, 0.0,			1.0,1.0,
			self.nWidth, self.nHeight,	1.0,0.0
		])
		vbo = self.ctx.buffer(vertices_quad.astype('f4').tobytes())
		vao = self.ctx.simple_vertex_array(self.oDebugShader.program, vbo, 'in_vert', 'in_text')

		'''
		obj = Obj.open(local('assets', '%s.obj' % name))
		vbo = self.ctx.buffer(obj.pack('vx vy vz tx ty'))
		vao = self.ctx.simple_vertex_array(self.oDebugShader.program, vbo, 'in_vert', 'in_text')
		'''

		self.dictVAOs[name] = vao

		self.oDebugPlane = debug_object_node(	Vector4([0.0, 0.0, 0.0, 1.0]), 
												Quaternion([0.0, 0.0, 0.0, 1.0]),
												self.dictVAOs[name], self.oDebugShader)

	def prepareEnv(self):
		
		img_HDRMap = FreeImage(local('assets', 'pedestrian_overpass_2k.hdr'))
		oTexHDRMap = self.ctx.texture(img_HDRMap.size, 3, img_HDRMap.getRaw(), dtype='f4')

		#img_HDRMap = FreeImage(local('assets', 'ref_equirectangular_map.png'))
		#oTexHDRMap = self.ctx.texture(img_HDRMap.size, 4, img_HDRMap.getRaw())

		#

		name = 'preprocess_hdr2cubemap'
		obj = Obj.open(local('assets', 'skybox.obj'))
		vbo = self.ctx.buffer(obj.pack('vx vy vz'))
		vao = self.ctx.simple_vertex_array(self.oHdr2CubemapShader.program, vbo, 'in_vert')
		self.dictVAOs[name] = vao

		self.oHdrCubeMap = hdr2cubemap_preprocess_node(	Vector4([0.0, 0.0, 0.0, 1.0]),
														Quaternion([0.0, 0.0, 0.0, 1.0]),
														self.ctx, 
														self.dictVAOs[name], 
														self.oHdr2CubemapShader, oTexHDRMap,
														1024, 1024
														#512, 512
														)

		# Run pre-processing step to convert HDR to cubemap

		self.oHdrCubeMap.render()

		#oTexSkyBox_Env = self.ctx.texture_cube((512, 512), 3)
		oTexSkyBox_Env = self.ctx.texture_cube((1024, 1024), 3)

		oTexSkyBox_Env.write(0, self.oHdrCubeMap.textureFbo_Right.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_X 
		oTexSkyBox_Env.write(1, self.oHdrCubeMap.textureFbo_Left.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_X 
		oTexSkyBox_Env.write(2, self.oHdrCubeMap.textureFbo_Front.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_Y 
		oTexSkyBox_Env.write(3, self.oHdrCubeMap.textureFbo_Back.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 
		oTexSkyBox_Env.write(4, self.oHdrCubeMap.textureFbo_Up.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_Z 
		oTexSkyBox_Env.write(5, self.oHdrCubeMap.textureFbo_Down.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 

		#

		name = 'preprocess_irrcubemap'
		obj = Obj.open(local('assets', 'skybox.obj'))
		vbo = self.ctx.buffer(obj.pack('vx vy vz'))
		vao = self.ctx.simple_vertex_array(self.oIrradianceCubemapShader.program, vbo, 'in_vert')
		self.dictVAOs[name] = vao

		self.oIrradianceCubeMap = irradiancemap_preprocess_node(	Vector4([0.0, 0.0, 0.0, 1.0]),
																	Quaternion([0.0, 0.0, 0.0, 1.0]),
																	self.ctx, 
																	self.dictVAOs[name], 
																	self.oIrradianceCubemapShader, oTexSkyBox_Env,
																	#1024, 1024
																	#512, 512
																	32, 32
																	)

		# Run pre-processing step to convert HDR to cubemap

		self.oIrradianceCubeMap.render()

		self.oTexSkyBox_Irradiance = self.ctx.texture_cube((32, 32), 3)
		#self.oTexSkyBox_Irradiance = self.ctx.texture_cube((512, 512), 3)
		#self.oTexSkyBox_Irradiance = self.ctx.texture_cube((1024, 1024), 3)

		self.oTexSkyBox_Irradiance.write(0, self.oIrradianceCubeMap.textureFbo_Right.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_X 
		self.oTexSkyBox_Irradiance.write(1, self.oIrradianceCubeMap.textureFbo_Left.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_X 
		self.oTexSkyBox_Irradiance.write(2, self.oIrradianceCubeMap.textureFbo_Front.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_Y 
		self.oTexSkyBox_Irradiance.write(3, self.oIrradianceCubeMap.textureFbo_Back.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 
		self.oTexSkyBox_Irradiance.write(4, self.oIrradianceCubeMap.textureFbo_Up.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_Z 
		self.oTexSkyBox_Irradiance.write(5, self.oIrradianceCubeMap.textureFbo_Down.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 

		#
		
		name = 'preprocess_prefilteredcubemap'
		obj = Obj.open(local('assets', 'skybox.obj'))
		vbo = self.ctx.buffer(obj.pack('vx vy vz'))
		vao = self.ctx.simple_vertex_array(self.oPreFilterCubemapShader.program, vbo, 'in_vert')
		self.dictVAOs[name] = vao

		self.oPreFilterCubeMap = prefiltermap_preprocess_node(	Vector4([0.0, 0.0, 0.0, 1.0]),
																Quaternion([0.0, 0.0, 0.0, 1.0]),
																self.ctx, 
																self.dictVAOs[name], 
																self.oPreFilterCubemapShader, oTexSkyBox_Env,
																#1024, 1024
																#512, 512
																128, 128
																#32, 32
																)

		# Run pre-processing step to convert HDR to cubemap

		self.oPreFilterCubemapShader.roughness.value = 0.1 #0.5

		self.oPreFilterCubeMap.render()

		#self.oTexSkyBox_PreFilter = self.ctx.texture_cube((32, 32), 3)
		self.oTexSkyBox_PreFilter = self.ctx.texture_cube((128, 128), 3)
		#self.oTexSkyBox_PreFilter = self.ctx.texture_cube((512, 512), 3)
		#self.oTexSkyBox_PreFilter = self.ctx.texture_cube((1024, 1024), 3)

		self.oTexSkyBox_PreFilter.write(0, self.oPreFilterCubeMap.textureFbo_Right.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_X 
		self.oTexSkyBox_PreFilter.write(1, self.oPreFilterCubeMap.textureFbo_Left.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_X 
		self.oTexSkyBox_PreFilter.write(2, self.oPreFilterCubeMap.textureFbo_Front.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_Y 
		self.oTexSkyBox_PreFilter.write(3, self.oPreFilterCubeMap.textureFbo_Back.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 
		self.oTexSkyBox_PreFilter.write(4, self.oPreFilterCubeMap.textureFbo_Up.read()) # GL_TEXTURE_CUBE_MAP_POSITIVE_Z 
		self.oTexSkyBox_PreFilter.write(5, self.oPreFilterCubeMap.textureFbo_Down.read()) # GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 

		#

		# TODO : run a quasi monte-carlo simulation on the environment lighting to create a prefilter (cube)map.

		#

		# TODO : generate a 2D LUT from the BRDF equations used.

		qSampleBrdf = quaternion.inverse([0.0, 0.0, 0.0, 1.0])
		self.oSampleBrdf = brdf_preprocess_node(	Vector4([0.0, 0.0, 0.0, 1.0]), qSampleBrdf, 
													self.ctx, self.oBrdfShader, 
													512, 512)

		self.oSampleBrdf.render()
		
		#

		name = 'skybox'
		obj = Obj.open(local('assets', 'skybox.obj'))
		vbo = self.ctx.buffer(obj.pack('vx vy vz'))
		vao = self.ctx.simple_vertex_array(self.oSkyboxShader.program, vbo, 'in_vert')
		self.dictVAOs[name] = vao

		self.oSkybox = skybox_object_node(	Vector4([0.0, 0.0, 0.0, 1.0]), 
											Quaternion([0.000, 0.0, 0.0, 1.0]),
											self.dictVAOs[name], 
											self.oSkyboxShader, 
											oTexSkyBox_Env,
											self.oTexSkyBox_Irradiance,
											self.oTexSkyBox_PreFilter)

	def loadOBJs(self):

		self.posLights_scene = np.array([[-5.0, -5.0, 1.0], [-5.0, 5.0, 1.0], [5.0, -5.0, 1.0], [5.0, 5.0, 1.0]])
		self.byte_posLights = self.posLights_scene.astype('f4').tobytes()

		self.colourLights_scene = np.array([[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
		self.byte_colourLights= self.colourLights_scene.astype('f4').tobytes()

		img_refaxis = PIL_Image.open(local('assets', 'refaxis.png')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		self.texture_refaxis = self.ctx.texture(img_refaxis.size, 3, img_refaxis.tobytes())
		self.texture_refaxis.build_mipmaps()

		img_jeep = PIL_Image.open(local('assets', 'jeep.png')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		self.texture_jeep = self.ctx.texture(img_jeep.size, 3, img_jeep.tobytes())
		self.texture_jeep.build_mipmaps()

		##

		for name in ['light1']:
			obj = Obj.open(local('assets', '%s.obj' % name))
			vbo = self.ctx.buffer(obj.pack('vx vy vz'))
			vao = self.ctx.simple_vertex_array(self.oLightShader.program, vbo, 'in_vert')
			self.dictVAOs[name] = vao

		#

		numLights = len(self.posLights_scene)
		for i in range(0, numLights):
			strLightSample = 'light' + str(i)
			oLightSample = light_node(	Vector4([self.posLights_scene[i][0], self.posLights_scene[i][1], self.posLights_scene[i][2], 1.0]), 
										Quaternion([0.0, 0.0, 0.0, 1.0]),
										self.dictVAOs['light1'], #strLightSample],
										self.oLightShader,
										colour_Light = (self.colourLights_scene[i][0], self.colourLights_scene[i][1], self.colourLights_scene[i][2]) )

			self.lights[strLightSample] = oLightSample

		##

		'''
		for name in ['refaxis', 'jeep']:
			obj = Obj.open(local('assets', '%s.obj' % name))
			vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
			vao = self.ctx.simple_vertex_array(self.oRegularShader.program, vbo, 'in_vert', 'in_norm', 'in_text')
			self.dictVAOs[name] = vao

		#

		self.oRefAxis = typical_object_node(	Vector4([0.0, 0.0, 0.0, 1.0]), 
										Quaternion([0.0, 0.0, 0.0, 1.0]),
										self.dictVAOs['refaxis'], self.oRegularShader, 
										self.texture_refaxis )

		self.rootObjects_typical['refaxis'] = self.oRefAxis

		self.oJeep = typical_object_node(	Vector4([-2.0, 1.0, 0.0, 1.0]), 
									Quaternion([0.0, 0.0, 0.383,  0.924]),
									self.dictVAOs['jeep'], self.oRegularShader,
									self.texture_jeep )

		self.rootObjects_typical['jeep'] = self.oJeep
		'''

		##
		
		img_AlbedoMap = PIL_Image.open(local('assets', 'Default_albedo.jpg')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		oTexAlbedoMap = self.ctx.texture(img_AlbedoMap.size, 3, img_AlbedoMap.tobytes())
		oTexAlbedoMap.build_mipmaps()
		
		img_NormalMap = PIL_Image.open(local('assets', 'Default_normal.jpg')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		oTexNormalMap = self.ctx.texture(img_NormalMap.size, 3, img_NormalMap.tobytes())
		oTexNormalMap.build_mipmaps()

		img_MetallicMap = PIL_Image.open(local('assets', 'Default_metalRoughness.jpg')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		oTexMetallicMap = self.ctx.texture(img_MetallicMap.size, 3, img_MetallicMap.tobytes())
		oTexMetallicMap.build_mipmaps()

		img_RoughnessMap = PIL_Image.open(local('assets', 'Default_metalRoughness.jpg')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		oTexRoughnessMap = self.ctx.texture(img_RoughnessMap.size, 3, img_RoughnessMap.tobytes())
		oTexRoughnessMap.build_mipmaps()

		img_AoMap = PIL_Image.open(local('assets', 'Default_AO.jpg')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		oTexAoMap = self.ctx.texture(img_AoMap.size, 3, img_AoMap.tobytes())
		oTexAoMap.build_mipmaps()

		img_EmissiveMap = PIL_Image.open(local('assets', 'Default_Emissive.jpg')).transpose(PIL_Image.FLIP_TOP_BOTTOM).convert('RGB')
		oTexEmissiveMap = self.ctx.texture(img_EmissiveMap.size, 3, img_EmissiveMap.tobytes())
		oTexEmissiveMap.build_mipmaps()

		for name in ['damagedhelmet']:
			obj = Obj.open(local('assets', '%s.obj' % name))
			vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
			vao = self.ctx.simple_vertex_array(self.oPbrShader.program, vbo, 'in_vert', 'in_norm', 'in_text')
			self.dictVAOs[name] = vao

		#

		self.oDamagedHelmet = pbr_object_node(	Vector4([0.0, 0.0, 0.0, 1.0]), 
												Quaternion([0.000, 0.0, 0.0, 1.0]),
												self.dictVAOs['damagedhelmet'], 
												self.oPbrShader, 
												oTexAlbedoMap, 
												oTexNormalMap, 
												oTexMetallicMap, 
												oTexRoughnessMap, 
												oTexAoMap,
												oTexEmissiveMap,
												self.oTexSkyBox_Irradiance,
												self.oTexSkyBox_PreFilter,
												self.oSampleBrdf.oTexture
												)

		self.rootObjects_pbr['damagedhelmet'] = self.oDamagedHelmet

		self.currSelection = self.oDamagedHelmet

		#

		'''
		for objName, obj in self.rootObjects_pbr.items():
			print(obj)
		'''					

	def renderSceneFromEye(self, fbo_curr, eEyePos):
		fbo_curr.use()
		fbo_curr.clear(0.2, 0.2, 0.2, 1.0)

		self.ctx.enable(moderngl.BLEND)
		self.ctx.enable(moderngl.DEPTH_TEST)
		#self.ctx.enable(moderngl.MULTISAMPLE)
		
		if visualMode.STEREO == self.eVisualMode:
			copyM44(self.p_scene, self.proj_scene_stereo)

			if eyePosition.LEFT == eEyePos:
				self.p_scene = self.p_scene * self.ipd_l

			elif eyePosition.RIGHT == eEyePos:
				self.p_scene = self.p_scene * self.ipd_r

		else:
			copyM44(self.p_scene, self.proj_scene_mono)

		'''
		time_sample = 0.0 #time.clock() - self.start_time
		rotate = Matrix44.from_z_rotation(np.sin(time_sample*10.0) * 0.5 ) #+ 0.2)
		self.proj_scene = self.proj_scene * rotate
		'''

		self.byte_proj_matrix = self.p_scene.astype('f4').tobytes()

		# Grid

		self.oGridShader.update_proj_matrix_bytes(self.byte_proj_matrix)
		self.oGridShader.update_view_matrix_bytes(self.byte_view_matrix)

		self.oGrid.render()

		# Scene
		
		'''

		self.oRegularShader.update_proj_matrix_bytes(self.byte_proj_matrix)
		self.oRegularShader.update_view_matrix_bytes(self.byte_view_matrix)
		
		self.oRegularShader.colourAmbient_scene.value = self.colour_ambient
		self.oRegularShader.strengthAmbient_scene.value = self.strength_ambient

		self.oRegularShader.posLight_scene.value = (self.oLight1.vPos.x, self.oLight1.vPos.y, self.oLight1.vPos.z)
		self.oRegularShader.colourLight_scene.value = self.oLight1.colour_Light

		self.oRegularShader.posView_scene.value = (	self.oCameraDefault.model_matrix.m41, 
													self.oCameraDefault.model_matrix.m42, 
													self.oCameraDefault.model_matrix.m43)

		self.oRegularShader.specularStrength_scene.value = self.strength_specular

		#? self.oRegularShader.bUseTexture.value = False
		self.oRegularShader.bUseTexture.value = True

		for objName, obj in self.rootObjects_pbr.items():
			obj.render()

		
		'''
		#

		self.oSkyboxShader.update_proj_matrix_bytes(self.byte_proj_matrix)
		self.oSkyboxShader.update_view_matrix_bytes(self.byte_view_matrix)

		self.oSkybox.render()

		#

		self.oPbrShader.update_proj_matrix_bytes(self.byte_proj_matrix)
		self.oPbrShader.update_view_matrix_bytes(self.byte_view_matrix)

		self.oPbrShader.posView_scene.value = (	self.oCameraDefault.model_matrix.m41, 
												self.oCameraDefault.model_matrix.m42, 
												self.oCameraDefault.model_matrix.m43)
												
		self.oPbrShader.posLights.write(self.byte_posLights)
		self.oPbrShader.colourLights.write(self.byte_colourLights)

		for objName, obj in self.rootObjects_pbr.items():
			obj.render()
		
		#
		
		self.oLightShader.update_proj_matrix_bytes(self.byte_proj_matrix)
		self.oLightShader.update_view_matrix_bytes(self.byte_view_matrix)

		for lightName, oLight in self.lights.items():
			oLight.render()

	def renderDebugTexture(self, fbo_curr, oTexDebug):
		fbo_curr.use()
		fbo_curr.clear(0.1, 0.4, 0.5, 1.0)
		
		self.ctx.enable(moderngl.BLEND)
		self.ctx.enable(moderngl.DEPTH_TEST)
		#self.ctx.enable(moderngl.MULTISAMPLE)

		byte_projDebug_matrix = self.projDebug.astype('f4').tobytes()
		byte_viewDebug_matrix = self.viewDebug.astype('f4').tobytes()

		self.oDebugShader.update_proj_matrix_bytes(byte_projDebug_matrix)
		self.oDebugShader.update_view_matrix_bytes(byte_viewDebug_matrix)

		oTexDebug.use(0)
		self.oDebugShader.debug_Map.value = 0

		self.oDebugPlane.render()

	def prerenderUI_2D(self):
		''' 
		# TODO: 
		self.impl.process_inputs()

		imgui.new_frame()

		if imgui.begin_main_menu_bar():
			if imgui.begin_menu("File", True):

				clicked_quit, selected_quit = imgui.menu_item(
					"Quit", 'Cmd+Q', False, True
				)

				if clicked_quit:
					exit(1)

				imgui.end_menu()
			imgui.end_main_menu_bar()

		#imgui.show_test_window()

		#imgui.begin("Custom window", True)
		#imgui.text("Bar")
		#imgui.text_colored("Eggs", 0.2, 1., 0.)
		#imgui.end()
		'''
		pass

	def renderUI_2D(self):
		'''
		# TODO: 
		imgui.render()
		'''
		pass

	def mainloop(self):

		FPS = float(60)
		render_interval = 1/FPS

		# Force first render
		self.bRedraw = True

		while not glfw.window_should_close(self.window):
			
			glfw.wait_events_timeout(render_interval)

			# TODO: self.prerenderUI_2D()

			if True == self.bRedraw:

				if visualMode.MONO == self.eVisualMode:
					self.renderSceneFromEye(self.ctx.screen, eyePosition.NONE)

				elif visualMode.STEREO == self.eVisualMode:
					self.renderSceneFromEye(self.oLeftFBO.oFBO, eyePosition.LEFT)
					self.renderSceneFromEye(self.oRightFBO.oFBO, eyePosition.RIGHT)

					self.ctx.screen.use()
					self.ctx.clear(0.16, 0.16, 0.16, 1.0)

					self.ctx.enable(moderngl.BLEND)
					self.ctx.enable(moderngl.DEPTH_TEST)

					byte_projFBO_matrix = self.projFBO.astype('f4').tobytes()
					byte_viewFBO_matrix = self.viewFBO.astype('f4').tobytes()

					self.oFboShader.update_proj_matrix_bytes(byte_projFBO_matrix)
					self.oFboShader.update_view_matrix_bytes(byte_viewFBO_matrix)

					self.oLeftFBO.render()
					self.oRightFBO.render()

				elif visualMode.DEBUG_TEXTURE == self.eVisualMode:
					#self.renderDebugTexture(self.ctx.screen, self.oHdrCubeMap.textureFbo_Up)
					#self.renderDebugTexture(self.ctx.screen, self.oHdrCubeMap.textureFbo_Down)
					#self.renderDebugTexture(self.ctx.screen, self.oHdrCubeMap.textureFbo_Left)
					#self.renderDebugTexture(self.ctx.screen, self.oHdrCubeMap.textureFbo_Front)
					#self.renderDebugTexture(self.ctx.screen, self.oHdrCubeMap.textureFbo_Right)
					#self.renderDebugTexture(self.ctx.screen, self.oHdrCubeMap.textureFbo_Back)

					#self.renderDebugTexture(self.ctx.screen, self.oIrradianceCubeMap.textureFbo_Up)
					#self.renderDebugTexture(self.ctx.screen, self.oIrradianceCubeMap.textureFbo_Down)
					#self.renderDebugTexture(self.ctx.screen, self.oIrradianceCubeMap.textureFbo_Left)
					#self.renderDebugTexture(self.ctx.screen, self.oIrradianceCubeMap.textureFbo_Front)
					#self.renderDebugTexture(self.ctx.screen, self.oIrradianceCubeMap.textureFbo_Right)
					#self.renderDebugTexture(self.ctx.screen, self.oIrradianceCubeMap.textureFbo_Back)

					
					#self.renderDebugTexture(self.ctx.screen, self.oPreFilterCubeMap.textureFbo_Up)
					#self.renderDebugTexture(self.ctx.screen, self.oPreFilterCubeMap.textureFbo_Down)
					#self.renderDebugTexture(self.ctx.screen, self.oPreFilterCubeMap.textureFbo_Left)
					#self.renderDebugTexture(self.ctx.screen, self.oPreFilterCubeMap.textureFbo_Front)
					#self.renderDebugTexture(self.ctx.screen, self.oPreFilterCubeMap.textureFbo_Right)
					#self.renderDebugTexture(self.ctx.screen, self.oPreFilterCubeMap.textureFbo_Back)

					self.renderDebugTexture(self.ctx.screen, self.oSampleBrdf.oTexture)

				# TODO: self.renderUI_2D()
			
				glfw.swap_buffers(self.window)

				#? self.bRedraw = False

				# For profiling
				self.bRedraw = self.bForceRedraw

	def stop(self):
		'''
		# TODO: 
		self.impl.shutdown()
		imgui.shutdown()
		'''
		glfw.terminate()

def main():
	
	#######################

	bForceRedraw = False
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--width', help='Width', nargs='?', const=1, type=int, default=1280)
	parser.add_argument('--height', help='Height', nargs='?', const=1, type=int, default=720)

	parser.add_argument('--offsetx', help='OffsetX', nargs='?', const=1, type=int, default=40)
	parser.add_argument('--offsety', help='OffsetY', nargs='?', const=1, type=int, default=40)

	parser.add_argument('--ipd_l_x', help='IPD L X', nargs='?', const=1, type=float, default=0.5)
	parser.add_argument('--ipd_l_y', help='IPD L Y', nargs='?', const=1, type=float, default=0.0)
	parser.add_argument('--ipd_l_z', help='IPD L Z', nargs='?', const=1, type=float, default=0.0)

	parser.add_argument('--ipd_r_x', help='IPD R X', nargs='?', const=1, type=float, default=-0.5)
	parser.add_argument('--ipd_r_y', help='IPD R Y', nargs='?', const=1, type=float, default=0.0)
	parser.add_argument('--ipd_r_z', help='IPD R Z', nargs='?', const=1, type=float, default=0.0)

	parser.add_argument('--force_redraw', help='Force redrawing (enable for profiling)', action='store_true')

	args = parser.parse_args()
	
	nWidth = args.width
	nHeight = args.height

	nOffsetX = args.offsetx
	nOffsetY = args.offsety

	ipd_l_t = (args.ipd_l_x, args.ipd_l_y, args.ipd_l_z)
	ipd_r_t = (args.ipd_r_x, args.ipd_r_y, args.ipd_r_z)

	bForceRedraw = args.force_redraw

	#######################
	
	'''
	pr = cProfile.Profile()   
	pr.enable()
	'''

	oMainApp = xrApp(	nWidth=nWidth, nHeight=nHeight, 
						nOffsetX=nOffsetX, nOffsetY=nOffsetY, 
						ipd_l_t=ipd_l_t, ipd_r_t=ipd_r_t,
						bForceRedraw = bForceRedraw
					)

	oMainApp.start()
	oMainApp.mainloop()
	oMainApp.stop()

	'''
	pr.disable()
	pr.print_stats()
	pr.dump_stats("profiler.prof")
	'''
	
################################################################################

if __name__ == "__main__":
	main()
