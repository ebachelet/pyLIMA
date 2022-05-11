from tqdm import tqdm
import time as python_time
import numpy as np
import sys
from multiprocessing import Manager



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

    def gelman_rubin(self, chain):
        ssq = np.var(chain, axis=1, ddof=1)
        W = np.mean(ssq, axis=0)
        θb = np.mean(chain, axis=1)
        θbb = np.mean(θb, axis=0)
        m = chain.shape[0]
        n = chain.shape[1]
        B = n / (m - 1) * np.sum((θbb - θb) ** 2, axis=0)
        var_θ = (n - 1) / n * W + 1 / n * B
        GR= np.sqrt(var_θ / W)
        return GR

    def new_individual(self, ind1, pop):

        crossover = np.random.uniform(0.1, 1.0)

        ind2, ind3 = np.random.randint(0, len(pop), 2)
        parent1 = pop[ind1]
        parent2 = pop[ind2]
        parent3 = pop[ind3]

        mutate = np.random.uniform(0, 1, len(parent1[:-1])) < crossover
        mutation = np.random.uniform(-2, 2, len(parent1[:-1]))

        shifts = np.random.normal(0,10**-3,len(parent1[:-1]))

        progress = (parent2[:-1] - parent3[:-1]) * mutation
        child = np.copy(parent1)

        child[:-1][mutate] += progress[mutate]
        child[:-1] += shifts

        objective = 0

        for ind, param in enumerate(self.fit_parameters.keys()):

            if (child[ind] < self.fit_parameters[param][1][0]) | (child[ind] > self.fit_parameters[param][1][1]):
                objective = np.inf
                break

        if np.isinf(objective):

            pass

        else:

            objective = self.objective_function(child[:-1])

        casino = np.random.uniform(0, 1)

        if np.exp(-objective + parent1[-1]) > casino:
        #if objective<parent1[-1]:

            child[-1] = objective

            return child

        else:

            return parent1

    def ptform(self, u):

        x = []
        for ind,key in enumerate(self.fit_parameters):

            bound = self.fit_parameters[key][1]
            x.append(u[ind]*(bound[1]-bound[0])+bound[0])

        return np.array(x)


    def objective_function2(self, fit_process_parameters):

        like = self.objective_function(fit_process_parameters)

        return -like

    def fit(self, computational_pool=None):

        starting_time = python_time.time()

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

            loop_population = np.array(computational_pool.starmap(self.new_individual, indexes))

            all_population.append(loop_population)

        self.DE_population = np.array(all_population)

        computation_time = python_time.time() - starting_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.fit_time = computation_time