import pygame as pg
import moderngl as mgl
import sys
from screen import *
import numpy as np
from particles import ParticleBag


class Engine:
    def __init__(self, window_size=(1024, 512)):
        
        pg.init()
        self.working_dir = self.get_working_dir()
        self.WINDOW_SIZE = window_size

        # CONSTANTS
        self.NUM_PARTICLES = 2**10#2**27 # 2**27 = 134217728 = max; otherwise np breaks on init
        self.PARTICLE_VELO = 0.5
        self.SENSOR_OFFSET_ANGLE = np.pi/16
        self.SENSOR_DISTANCE = 30
        self.TURNING_ANGLE = np.pi/30
        self.BLUR_STRENGTH = 0.1
        self.RANDOM_WANDERING = 0.075
        self.BORDER_OFFSET = 10
        # subtract brightness every nth frame
        self.DIM_FRAME_THRESHOLD = 2
        self.DIMMING_FACTOR = 0.002 # min: 0.002

        # gl attributes
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 4)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(
            pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.WINDOW_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)

        # make coursor invisible
        pg.mouse.set_visible(False)

        # create context
        self.ctx = mgl.create_context()
        self.ctx.enable(flags=mgl.CULL_FACE)

        # init helper functions
        self.clock = pg.time.Clock()
        self.particleBag = ParticleBag(self)
        self.delta_time = 0
        self.screen = Screen(self)
        self.time = 0

    def get_working_dir(self) -> str:
        return str(Path(__file__).parent.absolute()) + '/'

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and (event.key == pg.K_ESCAPE or event.key == pg.K_q)):
                self.quit()

    def quit(self):
        # release GPU resources
        self.screen.destroy()
        # quit pg
        pg.quit()
        sys.exit()

    def get_time(self):
        # current time
        self.time = pg.time.get_ticks() * 0.001

    def render(self):
        # black background
        self.ctx.clear(color=(0, 0, 0))
        # render screen
        self.screen.render()
        # flip the frame
        pg.display.flip()

    def run(self):
        while True:
            self.get_time()
            # check if close was pressed
            self.check_events()
            # render to screen
            self.render()
            # print frames to console
            self.delta_time = self.clock.tick(60)
            print(round(self.clock.get_fps()), end='\r')


if __name__ == '__main__':
    app = Engine()
    app.run()
