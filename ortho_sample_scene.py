		'''
		#####

		# Ortho scene

		left = 0 #-float(self.width)
		right = float(self.width)/2
		top = 0 #float(self.height) 
		bottom = -float(self.height)
		near = 1.0
		far = 1000.0
		self.proj_sample = Matrix44.orthogonal_projection(left, right, bottom, top, near, far)
		self.lookat_sample = Matrix44.look_at(
			(0.0, 0.0, -10.0),
			(0.0, 0.0, 0.0),
			(0.0, -1.0, 0.0),
		)

		self.prog_sample = self.ctx.program(
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
		
		self.mvp_sample = self.prog['MVP']

		img_sample = Image.open(local('assets', 'sample.png')).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
		self.texture_sample = self.ctx.texture(img_sample.size, 3, img_sample.tobytes())
		self.texture_sample.build_mipmaps()

		vertices_sample = np.array([
			100.0, 100.0,								0.0,0.0,
			100.0, self.height - 100.0,					0.0,1.0,
			self.width/2 - 100.0, 100.0,				1.0,0.0,
			self.width/2 - 100.0, self.height - 100.0, 	1.0,1.0
		])
		self.vbo_sample = self.ctx.buffer(vertices_sample.astype('f4').tobytes())
		self.vao_sample = self.ctx.simple_vertex_array(self.prog, self.vbo_sample, 'in_vert', 'in_text')
		'''