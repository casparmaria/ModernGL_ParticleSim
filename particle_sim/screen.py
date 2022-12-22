import numpy as np
import glm
import moderngl as mgl
import sys
from pathlib import Path
from time import time


class Screen:
    def __init__(self, app):
        # inherit app functionalities
        self.app = app
        self.ctx = app.ctx
        # vertex buffer data
        self.vbo = self.get_vbo()
        # init shader program with vertex and fragment shaders
        self.shader_program = self.get_shader_program('default')
        # vertex array object
        self.vao = self.get_vao()
        self.model_matrix = self.get_model_matrix()
        # read and init compute shaders
        self.cs_dim = self.get_compute_shader('dim')
        self.cs_blur = self.get_compute_shader('blur')
        self.cs_frame = self.get_compute_shader('frame')
        # init buffers storage objects to store particles
        print('Sending buffer data to GPU...',end='\r')
        start = time()
        self.particle_buf_read, self.particle_buf_write = self.assign_particle_buffers()
        # init texture objects storing current frame
        self.frame_buf_read, self.frame_buf_write = self.assign_frame_buffer_textures()
        end = time()
        print(f'Sent all buffer data to GPU in {end-start:.1f} sec.')
        # used to dim every nth screen
        self._dim_iterator = 0
        # matrices to shader
        self.shader_program['m_projection'].write(
            glm.ortho(0, self.app.WINDOW_SIZE[0],self.app.WINDOW_SIZE[1], 0, 0, 100))
        self.shader_program['m_view'].write(glm.mat4())
        self.shader_program['m_model'].write(self.model_matrix)
        print('Alright, we\'re done setting up. Run!')

    
    def assign_particle_buffers(self):
        # writes particle array to buffers
        compute_data = self.app.particleBag.particle_arr
        buffer_a = self.ctx.buffer(compute_data)
        buffer_b = self.ctx.buffer(compute_data)
        return (buffer_a, buffer_b)

    def assign_frame_buffer_textures(self):
        # init read textures in compute shaders
        self.cs_blur['readTex'] = 1
        self.cs_dim['readTex'] = 1
        self.cs_frame['readTex'] = 1
        # init read texture
        frame_buf_read = self.ctx.texture(
            (self.app.WINDOW_SIZE[0], self.app.WINDOW_SIZE[1]), 4)
        frame_buf_read.filter = mgl.NEAREST, mgl.NEAREST
        # init write textures in compute shaders
        self.cs_blur['writeTex'] = 2
        self.cs_dim['writeTex'] = 2
        self.cs_frame['writeTex'] = 2
        # init write texture
        frame_buf_write = self.ctx.texture(
            (self.app.WINDOW_SIZE[0], self.app.WINDOW_SIZE[1]), 4)
        frame_buf_write.filter = mgl.NEAREST, mgl.NEAREST
        return (frame_buf_read, frame_buf_write)

    def get_compute_shader(self, shader_name):
        # load shaders from file
        with open(f'{self.app.working_dir}shaders/{shader_name}.compute') as file:
            # write program constants dynamically to compute shaders
            compute_shader_str = file.read()\
                .replace("%NUM_PARTICLES%", str(self.app.NUM_PARTICLES))\
                .replace("%SCREEN_W%", str(self.app.WINDOW_SIZE[0]))\
                .replace("%SCREEN_H%", str(self.app.WINDOW_SIZE[1]))\
                .replace("%SENSOR_OFFSET_ANGLE%", str(round(self.app.SENSOR_OFFSET_ANGLE, 4)))\
                .replace("%SENSOR_DISTANCE%", str(round(self.app.SENSOR_DISTANCE, 4)))\
                .replace("%TURNING_ANGLE%", str(round(self.app.TURNING_ANGLE, 4)))\
                .replace("%PARTICLE_VELO%", str(round(self.app.PARTICLE_VELO, 4)))\
                .replace("%RANDOM_WANDERING%", str(round(self.app.RANDOM_WANDERING, 4)))\
                .replace("%DIMMING_FACTOR%", str(round(self.app.DIMMING_FACTOR, 4)))\
                .replace("%BLUR_STRENGTH%", str(round(self.app.BLUR_STRENGTH, 4)))\
                .replace("%BORDER_OFFSET%", str(round(self.app.BORDER_OFFSET, 4)))\
                
        compute = self.ctx.compute_shader(compute_shader_str)
        return compute

    def dim_screen(self):
        w, h = self.app.WINDOW_SIZE
        gw, gh = 16, 16
        nx, ny = int(w/gw), int(h/gh)
        # run with right work group dimensions for screen
        # shader will subtract brightness from read frame to write frame
        self.cs_dim.run(nx, ny)
    
    def blurr_screen(self):
        w, h = self.app.WINDOW_SIZE
        gw, gh = 16, 16
        nx, ny = int(w/gw), int(h/gh)
        # run with right workgroup dimensions for screen
        # shader will blurrs current read frame to write frame
        self.cs_blur.run(nx, ny)

    def update(self):
        # bind textures to GPU locations
        self.frame_buf_read.bind_to_image(1, read=True, write=False)
        self.frame_buf_write.bind_to_image(2, read=False, write=True)
        # every n-th iteration, dimms the current frame
        if self._dim_iterator == self.app.DIM_FRAME_THRESHOLD:
            self.dim_screen()
            # after dimming, read and write need to be swapped for cs_blurr to work
            self.frame_buf_read, self.frame_buf_write = self.frame_buf_write, self.frame_buf_read
            self.frame_buf_read.bind_to_image(1, read=True, write=False)
            self.frame_buf_write.bind_to_image(2, read=False, write=True)
            self._dim_iterator = 0
        self._dim_iterator += 1
        # blurrs the current frame
        self.blurr_screen()
        # binds particle storage buffers to storage locations in GPU (flip from previous loc)
        self.particle_buf_read.bind_to_storage_buffer(1)
        self.particle_buf_write.bind_to_storage_buffer(2)
        # nr work groups because of three storage items in struct Particle: x, y, rad
        self.cs_frame.run(group_x=32, group_y=32)
        # make texture available to fragment shader
        self.frame_buf_write.use()
        # swap read & write buffer for next iteration
        self.particle_buf_read, self.particle_buf_write = self.particle_buf_write, self.particle_buf_read
        self.frame_buf_read, self.frame_buf_write = self.frame_buf_write, self.frame_buf_read

    def get_model_matrix(self):
        # scale model matrix to screen dimensions
        model_matrix = glm.scale(
            glm.mat4(), 
            glm.vec3(self.app.WINDOW_SIZE[0], self.app.WINDOW_SIZE[1], 0))
        return model_matrix

    def render(self):
        # run the compute shaders
        self.update()
        # render to screen
        self.vao.render()

    def destroy(self):
        # release GPU storage
        self.vbo.release()
        self.shader_program.release()
        self.vao.release()
        self.cs_dim.release()
        self.cs_blur.release()
        self.cs_frame.release()
        self.particle_buf_read.release()
        self.particle_buf_write.release()

    def get_vao(self):
        # make vertex array object
        vao = self.ctx.vertex_array(
            self.shader_program, [(self.vbo, '3f 2f', 'in_position', 'in_texcoord')])
        return vao

    def get_vertex_data(self):
        # vertex coords
        vertices = np.array([(0.0, 1.0, 0.0),
                             (1.0, 0.0, 0.0),
                             (0.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0),
                             (1.0, 1.0, 0.0),
                             (1.0, 0.0, 0.0)], dtype='f4')
        # texture coords
        texture_coords = np.array([(0.0, 0.0),
                                   (1.0, 1.0),
                                   (0.0, 1.0),
                                   (0.0, 0.0),
                                   (1.0, 0.0),
                                   (1.0, 1.0)], dtype='f4')
        # stack data to right dimensions
        vertex_data = np.hstack([vertices, texture_coords]).flatten()
        return vertex_data

    def get_vbo(self):
        vertex_data = self.get_vertex_data()
        # init vertex buffer from vertex data
        vbo = self.ctx.buffer(vertex_data)
        return vbo

    def get_shader_program(self, shader_name):
        # open shader files
        with open(f'{self.app.working_dir}shaders/{shader_name}.vert') as file:
            vertex_shader = file.read()
        with open(f'{self.app.working_dir}shaders/{shader_name}.frag') as file:
            fragment_shader = file.read()
        # init context program with frag & vert shaders
        program = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program
