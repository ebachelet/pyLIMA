from tqdm import tqdm
import time as python_time
import numpy as np
import sys
from multiprocessing import Manager
import multiprocessing


from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions



class DEMCfit(MLfit):

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DE_population_size=10, max_iteration=10000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters, rescale_photometry, rescale_astrometry, telescopes_fluxes_method)

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

        #self.DE_population.append(fit_process_parameters.tolist() + [likelihood])

        return likelihood

    def new_individual(self, ind1, pop):

        ind2, ind3 = np.random.choice(len(pop), 2, replace=False, p=pop[:, -1]/pop[:, -1].sum())

        #number_of_parents = np.random.randint(1, int(len(pop))/4)*2
        #indexes = np.random.choice(len(pop), number_of_parents, replace=False)

        parent1 = pop[ind1]
        parent2 = pop[ind2]
        parent3 = pop[ind3]

        crossover = np.random.uniform(0.0, 1.0)

        mutate = np.random.uniform(0, 1, len(parent1[:-1])) < crossover

        #if np.all(mutate == False):

        #    rand = np.random.randint(0, len(mutate))
        #    mutate[rand] = True

        #mutation = np.random.uniform(-2, 2, len(parent1[:-1]))
        eps1 = 10**-4
        eps2 = 10**-8

        mutation = np.random.uniform(1-eps1, 1+eps1, len(parent1[:-1]))
        gamma = 2.38/(2*2*len(parent1[:-1]))**0.5

        #mutation = np.random.uniform(-2, 2, len(parent1[:-1]))

        jumping_modes = np.random.randint(0, 5)

        if jumping_modes == 4:

            #mutation = np.ones(len(parent1[:-1]))
            gamma = 1

        mutation *= gamma
        
        shifts = np.random.normal(0, eps2, len(parent1[:-1]))*self.scale

        progress = (parent2[:-1] - parent3[:-1]) * mutation
        #progress1 = np.sum([pop[i] for i in indexes[::2]],axis=0)
        #progress2 = np.sum([pop[i] for i in indexes[1::2]],axis=0)
        #progress = (progress1[:-1]-progress2[:-1])*mutation

        child = np.copy(parent1)

        child[:-1][mutate] += progress[mutate]
        child[:-1] += shifts

        for ind, param in enumerate(self.fit_parameters.keys()):

            if (child[ind] < self.fit_parameters[param][1][0]) | (child[ind] > self.fit_parameters[param][1][1]):

                return parent1,0

        objective = self.objective_function(child[:-1])

        casino = np.random.uniform(0, 1)

        if np.exp(-objective + parent1[-1]) > casino:

            child[-1] = objective

            return child,1

        else:

            return parent1,0

    def fit(self, computational_pool=None):

        starting_time = python_time.time()
        self.scale = np.abs([self.fit_parameters[i][1][0] for i in self.fit_parameters.keys()])

        initial_population = []

        for i in range(int(self.DE_population_size*len(self.fit_parameters))):

            individual = []
            for j in self.fit_parameters.keys():

                individual.append(np.random.uniform(self.fit_parameters[j][1][0],self.fit_parameters[j][1][1]))

            individual = np.array(individual)
            objective = self.objective_function(individual)
            individual = np.r_[individual,objective]
            initial_population.append(individual)

        initial_population = np.array(initial_population)

        all_population = []
        all_population.append(initial_population)

        loop_population = np.copy(initial_population)

        for loop in tqdm(range(self.max_iteration)):

            indexes = [(i, loop_population) for i in range(len(loop_population))]

            if computational_pool is not None:

                new_step = np.array(computational_pool.starmap(self.new_individual, indexes))

                loop_population = np.vstack(new_step[:,0])
                acceptance = new_step[:, 1]

            else:

                loop_population = []
                acceptance = []

                for j,ind in enumerate(indexes):

                    new_step = self.new_individual(ind[0], ind[1])
                    loop_population.append(new_step[0])
                    acceptance.append(new_step[1])

                loop_population = np.array(loop_population)

            all_population.append(loop_population)
            print(np.sum(acceptance)/len(loop_population))

        self.DE_population = np.array(all_population)

        computation_time = python_time.time() - starting_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.fit_time = computation_time