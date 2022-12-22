import numpy as np
from time import time

class ParticleBag:
    def __init__(self, app):
        self.app = app
        self.particle_arr = self.init_particles()

    # returns [[x,y,rad], ...] * particle number
    def init_particles(self):
        print('Engine started. Initializing Particles...',end='\r')
        start = time()
        # random x positions
        xs = np.random.random_sample(
            self.app.NUM_PARTICLES) * (self.app.WINDOW_SIZE[0]-1)
        # random y positions
        ys = np.random.random_sample(
            self.app.NUM_PARTICLES) * (self.app.WINDOW_SIZE[1]-1)
        # random randians
        rads = np.random.random_sample(self.app.NUM_PARTICLES) * 2 * np.pi
        end = time()
        print(f'{int(self.app.NUM_PARTICLES)} particles initialized in {end-start:.1f} sec.')
        return np.dstack([xs, ys, rads]).astype('f4').flatten()