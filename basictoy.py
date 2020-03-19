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
from PIL import Image

import numpy as np
from pyrr import Matrix44, Vector4

from objloader import Obj


#import pyuv
#import requests

#import cProfile

################################################################################


@unique
class visualMode(Enum):
	MONO = 0
	STEREO = 1

@unique
class eyePosition(Enum):
	NONE = 0
	LEFT = 1
	RIGHT = 2

def local(*path):
	return os.path.join(os.path.dirname(__file__), *path)

class xrApp(object):
	def __init__(self):
		self.width = 1280
		self.height = 720
		self.window_name = "prototype"

		self.bVsync = True

		self.window = None

		'''
		self.impl = None
		'''

		self.ctx = None

		self.eVisualMode = visualMode.MONO
		#self.eVisualMode = visualMode.STEREO

		self.time = 0
		self.start_time = time.clock()

	def grid(self, size, steps):
		u = np.repeat(np.linspace(0, size, steps+1), 2)
		#print("u : ", u)
		v = np.tile([0, size], steps+1)
		#print("v : ", v)
		w = np.zeros((steps+1) * 2)
		return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])
		
	def toggle_visual_mode(self):
		if visualMode.MONO == self.eVisualMode:
			self.eVisualMode = visualMode.STEREO
		elif visualMode.STEREO == self.eVisualMode:
			self.eVisualMode = visualMode.MONO

	def key_event_callback(self, window, key, scancode, action, mods):

		# The well-known standard key for quick exit
		if key == glfw.KEY_ESCAPE:
			glfw.set_window_should_close(self.window, True)
			return

		# Toggle visual mode
		if glfw.KEY_G == key and glfw.RELEASE == action :
			self.toggle_visual_mode()

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
		#glfw.window_hint(glfw.SAMPLES, 1)

		glfw.window_hint(glfw.DECORATED, False)

		monitor = None
		
		'''
		if self.fullscreen:
			# Use the primary monitors current resolution
			monitor = glfw.get_primary_monitor()
			self.width, self.height = mode.size.width, mode.size.height
			mode = glfw.get_video_mode(monitor)
		'''

		self.window = glfw.create_window(self.width, self.height, self.window_name, monitor, None)
		
		if not self.window:
			glfw.terminate()
			raise ValueError("Failed to create window")
		
		'''
		if not self.cursor:
			glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
		'''

		glfw.set_window_pos(self.window, 40, 40)

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

		#self.ctx.viewport = self.window.viewport
		#self.set_default_viewport()
		#print("Wow ! GL_SHADING_LANGUAGE_VERSION :", gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION))

		'''
		self.proj = Matrix44.perspective_projection(45.0, float(self.width)/float(self.height), 0.1, 1000.0)
		self.lookat = Matrix44.look_at(
			(0.0, 0.0, 10.0),
			(0.0, 0.0, 0.0),
			(0.0, -1.0, 0.0),
		)
		'''
		
		left = 0 #-float(self.width)
		right = float(self.width)
		top = 0 #float(self.height) 
		bottom = -float(self.height) 
		near = 1.0
		far = 1000.0
		self.proj = Matrix44.orthogonal_projection(left, right, bottom, top, near, far)
		self.lookat = Matrix44.look_at(
			(0.0, 0.0, -10.0),
			(0.0, 0.0, 0.0),
			(0.0, -1.0, 0.0),
		)

		self.prog = self.ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 MVP;

				//in vec3 in_vert;
				in vec2 in_vert;
				in vec2 in_text;

				out vec2 v_text;

				void main() 
				{
					//gl_Position = MVP * vec4(in_vert, 1.0);
					gl_Position = MVP * vec4(in_vert, 0.0, 1.0);
					v_text = in_text;
				}
			''',
			fragment_shader='''
				#version 330
				
				uniform sampler2D Texture;

				in vec2 v_text;

				out vec4 f_color;

				void main() 
				{
					//f_color = vec4(0.3, 0.5, 1.0, 1.0);
					f_color = vec4(texture(Texture, v_text).rgb, 1.0);
				}
			''',
		)
		
		self.mvp = self.prog['MVP']

		img_left = Image.open(local('assets', 'l.png')).convert('RGB')
		self.texture_left = self.ctx.texture(img_left.size, 3, img_left.tobytes())
		#self.texture_left.build_mipmaps()

		#img_right = Image.open(local('assets', 'r.png')).convert('RGB')
		#self.texture_right = self.ctx.texture(img_right.size, 3, img_right.tobytes())
		#self.texture_right.build_mipmaps()

		texSize = (1024,1024)

		self.texture_left = self.ctx.texture((1024,1024), 3)
		depth_attachment_left = self.ctx.depth_renderbuffer(texSize)
		self.fbo_left = self.ctx.framebuffer(self.texture_left, depth_attachment_left)

		self.texture_right = self.ctx.texture((1024,1024), 3)
		depth_attachment_right = self.ctx.depth_renderbuffer(texSize)
		self.fbo_right = self.ctx.framebuffer(self.texture_right, depth_attachment_right)

		vertices_left_quad = np.array([
			0.0, 0.0,					0.0,1.0,
			0.0, self.height,			0.0,0.0,
			self.width/2, 0.0,			1.0,1.0,
			self.width/2, self.height,	1.0,0.0
		])
		self.vbo_left_quad = self.ctx.buffer(vertices_left_quad.astype('f4').tobytes())
		self.vao_left_quad = self.ctx.simple_vertex_array(self.prog, self.vbo_left_quad, 'in_vert', 'in_text')
		
		vertices_right_quad = np.array([
			self.width/2, 0.0,			0.0,1.0,
			self.width/2, self.height,	0.0,0.0,
			self.width, 0.0,			1.0,1.0,
			self.width, self.height, 	1.0,0.0
		])
		self.vbo_right_quad = self.ctx.buffer(vertices_right_quad.astype('f4').tobytes())
		self.vao_right_quad = self.ctx.simple_vertex_array(self.prog, self.vbo_right_quad, 'in_vert', 'in_text')

		#####

		# Persp scene

		self.proj_sample = Matrix44.perspective_projection(45.0, self.width / self.height, 0.1, 1000.0)

		self.proj_sample_stereo = Matrix44.perspective_projection(45.0, (self.width/2) / self.height, 0.1, 1000.0)

		self.lookat_sample = Matrix44.look_at(
			(50.0, 20.0, 30.0),
			(0.0, 0.0, 10.0),
			(0.0, 0.0, 1.0),
		)
		
		self.prog_sample = self.ctx.program(
			vertex_shader='''
				#version 330

				uniform mat4 MVP;

				in vec3 in_vert;
				in vec3 in_norm;
				in vec2 in_text;

				out vec3 v_vert;
				out vec3 v_norm;
				out vec2 v_text;

				void main() {
					gl_Position = MVP * vec4(in_vert, 1.0);
					v_vert = in_vert;
					v_norm = in_norm;
					v_text = in_text;
				}
			''',
			fragment_shader='''
				#version 330

				uniform vec3 Light;
				uniform vec3 Color;
				uniform bool UseTexture;
				uniform sampler2D Texture;

				in vec3 v_vert;
				in vec3 v_norm;
				in vec2 v_text;

				out vec4 f_color;

				void main() {
					float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
					if (UseTexture) {
						f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
					} else {
						f_color = vec4(Color * lum, 1.0);
					}
				}
			''',
		)
		
		self.objects = {}

		for name in ['ground', 'grass', 'billboard', 'billboard-holder', 'billboard-image']:
			obj = Obj.open(local('assets', 'scene-1-%s.obj' % name))
			vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
			vao = self.ctx.simple_vertex_array(self.prog_sample, vbo, 'in_vert', 'in_norm', 'in_text')
			self.objects[name] = vao

		img = Image.open(local('assets', 'infographic-1.jpg')).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
		self.texture_sample = self.ctx.texture(img.size, 3, img.tobytes())
		self.texture_sample.build_mipmaps()

		self.mvp_sample = self.prog_sample['MVP']
		self.bUseTexture_sample = self.prog_sample['UseTexture']
		self.light_sample = self.prog_sample['Light']
		self.color_sample = self.prog_sample['Color']

	def renderSceneFromEye(self, fbo_curr, eEyePos):
		fbo_curr.use()

		fbo_curr.clear(1.0, 1.0, 1.0, 1.0)

		'''
		if self.fbo_left == fbo_curr:
			fbo_curr.clear(0.32, 0.0, 0.0, 1.0)
		else:
			fbo_curr.clear(0.0, 0.0, 0.32, 1.0)
		'''

		ipd_translate = None
		if eyePosition.LEFT == eEyePos:
			ipd_translate = Matrix44.from_translation(Vector4([ 0.0, 5.0, 0.0, 1.0]))
		elif eyePosition.RIGHT == eEyePos:
			ipd_translate = Matrix44.from_translation(Vector4([ 0.0, -5.0, 0.0, 1.0]))
		else:
			ipd_translate = Matrix44.from_translation(Vector4([ 0.0, 0.0, 0.0, 1.0]))

		self.ctx.enable(moderngl.DEPTH_TEST)

		'''
		self.mvp_sample.write((self.proj_sample * self.lookat_sample).astype('f4').tobytes())

		self.texture_sample.use()
		self.vao_sample.render(moderngl.TRIANGLE_STRIP)
		'''

		time_sample = time.clock() - self.start_time
		rotate = Matrix44.from_z_rotation(np.sin(time_sample*10.0) * 0.5 + 0.2)

		self.bUseTexture_sample.value = False

		self.light_sample.value = (67.69, -8.14, 52.49)

		if visualMode.STEREO == self.eVisualMode:
			self.mvp_sample.write((self.proj_sample_stereo * self.lookat_sample * ipd_translate * rotate).astype('f4').tobytes())

		else:
			self.mvp_sample.write((self.proj_sample * self.lookat_sample * ipd_translate * rotate).astype('f4').tobytes())

		#self.mvp_sample.write((self.proj_sample * self.lookat_sample * ipd_translate).astype('f4').tobytes())
		#self.mvp_sample.write((self.proj_sample * self.lookat_sample).astype('f4').tobytes())
		#self.mvp_sample.write((self.proj_sample * ipd_translate * rotate).astype('f4').tobytes())

		self.color_sample.value = (0.67, 0.49, 0.29)
		self.objects['ground'].render()

		self.color_sample.value = (0.46, 0.67, 0.29)
		self.objects['grass'].render()

		self.color_sample.value = (1.0, 1.0, 1.0)
		self.objects['billboard'].render()

		self.color_sample.value = (0.2, 0.2, 0.2)
		self.objects['billboard-holder'].render()

		self.bUseTexture_sample.value = True
		self.texture_sample.use()

		self.objects['billboard-image'].render()

	def prerenderUI_2D(self):
		'''
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
		imgui.render()
		'''
		pass

	def mainloop(self):

		FPS = float(60)
		render_interval = 1/FPS

		while not glfw.window_should_close(self.window):
			
			#glfw.poll_events()
			glfw.wait_events_timeout(render_interval)

			#self.prerenderUI_2D()
			
			if visualMode.STEREO == self.eVisualMode:
				self.renderSceneFromEye(self.fbo_left, eyePosition.LEFT)
				self.renderSceneFromEye(self.fbo_right, eyePosition.RIGHT)

				self.ctx.screen.use()
				self.ctx.clear(0.16, 0.16, 0.16, 1.0)
				self.ctx.enable(moderngl.DEPTH_TEST)

				#self.vao.render()

				self.mvp.write((self.proj * self.lookat).astype('f4').tobytes())
				
				self.texture_left.use()
				self.vao_left_quad.render(moderngl.TRIANGLE_STRIP)
				
				self.texture_right.use()
				self.vao_right_quad.render(moderngl.TRIANGLE_STRIP)

			else:
				self.renderSceneFromEye(self.ctx.screen, eyePosition.NONE)

			#self.renderUI_2D()
			
			glfw.swap_buffers(self.window)

	def stop(self):
		'''
		self.impl.shutdown()
		imgui.shutdown()
		'''
		glfw.terminate()

def main():
	
	#######################

	#bLocalUI = False
	
	#parser = argparse.ArgumentParser()
	#parser.add_argument('--localui', help='Show the local UI for this module', action='store_true')
	#args = parser.parse_args()
	
	#bLocalUI = args.localui
	
	#######################
		
	#pr = cProfile.Profile()
	#pr.enable()
	
	oMainApp = xrApp()
	oMainApp.start()
	oMainApp.mainloop()
	oMainApp.stop()
	
	#pr.disable()
	#pr.print_stats()
	#pr.dump_stats("profiler.prof")
	
################################################################################

if __name__ == "__main__":
	main()
