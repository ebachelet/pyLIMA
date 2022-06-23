from tqdm import tqdm
import time as python_time
import numpy as np
import sys
from multiprocessing import Manager
import multiprocessing
import numpy as np
import os
import scipy.linalg
from pydream.parameters import FlatParam, SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
import scipy.stats as ss

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions



class DEMCfit(MLfit):

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DE_population_size=10, max_iteration=10000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method='polyfit')

        self.DE_population = Manager().list() # to be recognize by all process during parallelization
        self.DE_population_size = DE_population_size #Times number of dimensions!
        self.fit_boundaries = []
        self.max_iteration = max_iteration
        self.fit_time = 0 #s

    def fit_type(self):
        return "Differential Evolution Markov Chain"

    def objective_function(self, fit_process_parameters):

        likelihood = 0
        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescale_photometry:

                rescaling_photometry_parameters = 10**(fit_process_parameters[self.rescale_photometry_parameters_index])

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=rescaling_photometry_parameters)
            else:

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=None)

            photometric_likelihood = 0.5*(np.sum(residus ** 2 + np.log(2*np.pi*errflux**2)))

            likelihood += photometric_likelihood

        if self.model.astrometry:

            if self.rescale_astrometry:


                rescaling_astrometry_parameters = 10**(fit_process_parameters[self.rescale_astrometry_parameters_index])

                residus, errors = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                      pyLIMA_parameters,
                                                                                                      norm=True,
                                                                                                      rescaling_astrometry_parameters= rescaling_astrometry_parameters)

            else:

                residus, errors= pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                   pyLIMA_parameters,
                                                                                                   norm=True,
                                                                                                   rescaling_astrometry_parameters=None)

            astrometric_likelihood = 0.5*(np.sum(residus ** 2 + np.log(2*np.pi*errors**2)))

            likelihood += astrometric_likelihood

        return likelihood

    def new_individual(self, ind1, pop, var_pop):

        parent1 = pop[ind1]

        number_of_parents = np.random.randint(1, 3) * 2

        indexes = np.random.choice(len(pop), number_of_parents, replace=False)
        #crossover = np.random.uniform(0.0,1.0,len(parent1[:-1]))
        #crossover = self.crossover
        crossover_index = np.random.choice(len(self.crossover), 1, p=self.prob_crossover)
        crossover = self.crossover[crossover_index]
        #crossover = np.random.uniform(0.0, 1.0, len(parent1[:-1]))

        mutate = np.random.uniform(0, 1, len(parent1[:-1])) < crossover

        if np.all(mutate == False):

            rand = np.random.randint(0, len(mutate))
            mutate[rand] = True

        #mutation = np.random.uniform(-2, 2, len(parent1[:-1]))
        eps1 = 10**-3
        eps2 = 10**-7

        mutation = np.random.uniform(1-eps1, 1+eps1, len(parent1[:-1]))
        #mutation = np.random.uniform(0,2, len(parent1[:-1]))
        shifts = np.random.normal(0, eps2, len(parent1[:-1]))#*self.scale

        gamma = 2.38/(2*len(indexes[::2])*len(parent1[:-1][mutate]))**0.5#*self.scale

        jumping_modes = np.random.randint(0, 5)

        if jumping_modes == 4:

            #mutation = np.ones(len(parent1[:-1]))
            gamma = 1

        mutation *= gamma

        progress1 = np.sum([pop[i] for i in indexes[::2]], axis=0)
        progress2 = np.sum([pop[i] for i in indexes[1::2]], axis=0)
        progress = (progress1[:-1] - progress2[:-1]) * mutation

        #print(gamma,progress)

        child = np.copy(parent1)

        child[:-1][mutate] += progress[mutate]
        child[:-1] += shifts

        jump = np.zeros(len(self.crossover))
        nid = np.zeros(len(self.crossover))
        nid[crossover_index] += 1

        for ind, param in enumerate(self.fit_parameters.keys()):

            if (child[ind] < self.fit_parameters[param][1][0]) | (child[ind] > self.fit_parameters[param][1][1]):

                #progress[ind] = (pop[indexes[0]][ind]-parent1[ind])/2
                #child[ind] = parent1[ind]+progress[ind]
                child[ind] = parent1[ind]
                progress[ind] = 0
                #return parent1, 0, jump, nid

        objective = self.objective_function(child[:-1])


        casino = np.random.uniform(0, 1)
        probability = np.exp((-objective + parent1[-1]))

        if probability > casino:

            child[-1] = objective
            jump[crossover_index] += np.sum(progress[mutate]**2/var_pop[:-1][mutate])

            return child, 1, jump, nid

        else:

            return parent1, 0, jump, nid

    def fit(self, computational_pool=None):

        starting_time = python_time.time()
        #self.scale = np.abs([self.fit_parameters[i][1][0] for i in self.fit_parameters.keys()])
        self.scale = np.ones(len(self.fit_parameters.keys()))

        n_crossover = len(self.fit_parameters.keys())
        n_crossover = int(n_crossover/3)
        self.crossover = np.arange(1,n_crossover+1)/n_crossover
        self.prob_crossover = np.ones(n_crossover)/n_crossover


        initial_population = []
        import scipy.stats as ss
        sampler = ss.qmc.LatinHypercube(d=len(self.fit_parameters))

        for i in range(int(self.DE_population_size*len(self.fit_parameters))):

            individual = sampler.random(n=1)[0]

            for ind,j in enumerate(self.fit_parameters.keys()):

                individual[ind] = individual[ind]*(self.fit_parameters[j][1][1]-self.fit_parameters[j][1][0])+self.fit_parameters[j][1][0]

            individual = np.array(individual)
            objective = self.objective_function(individual)
            individual = np.r_[individual,objective]
            initial_population.append(individual)


       #import pdb;
       # pdb.set_trace()
        initial_population = np.array(initial_population)

        all_population = []
        all_population.append(initial_population)

        loop_population = np.copy(initial_population)

        Jumps = np.ones(len(self.crossover))

        N_id = np.ones(len(self.crossover))

        for loop in tqdm(range(self.max_iteration)):

            #if loop == 0:
            #    self.archive = loop_population
            #else:
            #    self.archive = np.array(all_population).reshape((loop+1)*len(loop_population),len(initial_population[0]))

            indexes = [(i,loop_population,np.var(loop_population,axis=0)) for i in range(len(loop_population))]

            if computational_pool is not None:

                new_step = np.array(computational_pool.starmap(self.new_individual, indexes))

                loop_population = np.vstack(new_step[:,0])
                acceptance = new_step[:, 1]
                jumps = new_step[:,2]
                n_id = new_step[:,3]

            else:
                var_pop = np.var(loop_population, axis=0)
                loop_population = []
                acceptance = []
                jumps = []
                n_id = []

                for j,ind in enumerate(indexes):

                    new_step = self.new_individual(ind[0], ind[1],var_pop)
                    loop_population.append(new_step[0])
                    acceptance.append(new_step[1])
                    jumps.append(new_step[2])
                    n_id.append(new_step[3])

                loop_population = np.array(loop_population)
                jumps = np.array(jumps)
                n_id = np.array(n_id)

            if loop<0.1*self.max_iteration:

                jumps = np.sum(jumps,axis=0)
                mask = jumps == 0
                jumps[mask] = np.max(jumps)
                Jumps += jumps

                n_id = np.sum(n_id, axis=0)
                mask = n_id == 0
                n_id[mask] = 1
                N_id += n_id

                pCR = Jumps/N_id
                self.prob_crossover = pCR/np.sum(pCR)
                #import pdb;
                #pdb.set_trace()
            all_population.append(loop_population)
            accepted = np.sum(acceptance)/len(loop_population)
            print(accepted,np.min(loop_population[:,-1]))



            #if accepted<0.10:

            #   self.scale /= 2

            #if accepted>0.40:

            #    self.scale *= 2


        self.DE_population = np.array(all_population)

        computation_time = python_time.time() - starting_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.fit_time = computation_time