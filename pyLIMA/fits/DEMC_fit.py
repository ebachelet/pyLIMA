from tqdm import tqdm
import time as python_time
import sys
import numpy as np

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions



class DEMCfit(MLfit):

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DEMC_population_size=10, max_iteration=10000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method=telescopes_fluxes_method)

        self.population = [] # to be recognize by all process during parallelization
        self.DEMC_population_size = DEMC_population_size #Times number of dimensions!
        self.max_iteration = max_iteration

    def fit_type(self):
        return "Differential Evolution Markov Chain"

    def objective_function(self, fit_process_parameters):

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        likelihood = 0

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

                rescaling_astrometry_parameters = 10 ** (
                fit_process_parameters[self.rescale_astrometry_parameters_index])

                residuals = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                pyLIMA_parameters,
                                                                                                norm=True,
                                                                                                rescaling_astrometry_parameters=rescaling_astrometry_parameters)

            else:

                residuals = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                pyLIMA_parameters,
                                                                                                norm=True,
                                                                                                rescaling_astrometry_parameters=None)

            residus = np.r_[residuals[:, 0], residuals[:, 2]]  # res_ra,res_dec
            errors = np.r_[residuals[:, 1], residuals[:, 3]]  # err_res_ra,err_res_dec

            astrometric_likelihood = 0.5*(np.sum(residus ** 2 + np.log(2*np.pi*errors**2)))

            likelihood += astrometric_likelihood

        return likelihood

    def new_individual(self, ind1, pop, var_pop):

        parent1 = pop[ind1]

        number_of_parents = np.random.randint(1, 3) * 2

        indexes = np.random.choice(len(pop), number_of_parents, replace=False)

        #crossover_index = np.random.choice(len(self.crossover), 1, p=self.prob_crossover)
        #crossover = self.crossover[crossover_index]
        crossover = np.random.uniform(0.0, 1.0, len(parent1[:-1]))

        mutate = np.random.uniform(0, 1, len(parent1[:-1])) < crossover

        if np.all(mutate == False):

            rand = np.random.randint(0, len(mutate))
            mutate[rand] = True

        #mutation = np.random.uniform(-2, 2, len(parent1[:-1]))
        eps1 = 10**-3
        eps2 = 10**-7

        mutation = np.random.uniform(1-eps1, 1+eps1, len(parent1[:-1]))
        shifts = np.random.normal(0, eps2, len(parent1[:-1]))#*self.scale

        gamma = 2.38/(2*len(indexes[::2])*len(parent1[:-1][mutate]))**0.5#*self.scale

        jumping_nodes = np.random.randint(0, 10)

        if jumping_nodes == 9:

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

       # jump = np.zeros(len(self.crossover))
       # nid = np.zeros(len(self.crossover))
        #nid[crossover_index] += 1
        accepted = np.zeros(len(parent1[:-1]))

        for ind, param in enumerate(self.fit_parameters.keys()):

            if (child[ind] < self.fit_parameters[param][1][0]) | (child[ind] > self.fit_parameters[param][1][1]):

                #progress[ind] = (pop[indexes[0]][ind]-parent1[ind])/2
                #child[ind] = parent1[ind]+progress[ind]

                child[ind] = parent1[ind]
                #progress[ind] = 0
                #mutate[ind] = False

                #return parent1, accepted#, jump, nid

        objective = self.objective_function(child[:-1])

        casino = np.random.uniform(0, 1)
        probability = np.exp((-objective*self.betas[ind1] + parent1[-1]*self.betas[ind1]))

        if probability > casino:

            child[-1] = objective
            #jump[crossover_index] += np.sum(progress[mutate]**2/var_pop[:-1][mutate])
            accepted[mutate] += 1

            return child, accepted#, jump, nid

        else:

            return parent1, accepted#, jump, nid

    def swap_temperatures(self, population):

        pop = np.copy(population)
        number_of_swap = int(len(population)/5)

        if (number_of_swap % 2) == 0:

            pass

        else:

            number_of_swap += 1

        choices = np.random.choice(len(population),number_of_swap, replace=False)

        for ind in np.arange(0, number_of_swap, 2):

            index = choices[ind]
            index2 = choices[ind+1]

            MH_temp = population[index2,-1]*self.betas[index]+population[index,-1]*self.betas[index2]
            MH_temp -= population[index,-1]*self.betas[index]+population[index2,-1]*self.betas[index2]

            casino = np.random.uniform(0, 1)
            probability = np.exp((-MH_temp))

            if probability > casino:

                child1 = np.copy(population[index2])
                child2 = np.copy(population[index])

                self.swap[index] += 1
                self.swap[index2] += 1

            else:

                child1 = np.copy(population[index])
                child2 = np.copy(population[index2])

            pop[index] = child1
            pop[index2] = child2

        return np.array(pop)

    def fit(self, computational_pool=None):

        start_time = python_time.time()

        #n_crossover = len(self.fit_parameters.keys())
        n_crossover = 3
        self.crossover = np.arange(1,n_crossover+1)/n_crossover
        self.prob_crossover = np.ones(n_crossover)/n_crossover
        #self.scale = np.ones(len(self.fit_parameters.keys()))

        initial_population = []
        number_of_walkers = int(self.DEMC_population_size*len(self.fit_parameters))
        import scipy.stats as ss
        sampler = ss.qmc.LatinHypercube(d=len(self.fit_parameters))
        self.betas = np.logspace(-3, 0, number_of_walkers)
        #self.betas = np.linspace(0,1, number_of_walkers)
        self.swap = np.zeros(number_of_walkers)

        for i in range(number_of_walkers):

            individual = sampler.random(n=1)[0]

            for ind,j in enumerate(self.fit_parameters.keys()):

                individual[ind] = individual[ind]*(self.fit_parameters[j][1][1]-self.fit_parameters[j][1][0])+self.fit_parameters[j][1][0]

            individual = np.array(individual)
            individual = np.r_[individual]

            objective = self.objective_function(individual)
            individual = np.r_[individual,objective]
            initial_population.append(individual)

        initial_population = np.array(initial_population)

        all_population = []
        all_population.append(initial_population)
        all_acceptance = []

        loop_population = np.copy(initial_population)

        #Jumps = np.ones(len(self.crossover))

       # N_id = np.ones(len(self.crossover))

        for loop in tqdm(range(self.max_iteration)):

            indexes = [(i, loop_population,np.var(loop_population, axis=0)) for i in range(len(loop_population))]

            if computational_pool is not None:

                new_step = np.array(computational_pool.starmap(self.new_individual, indexes))

                loop_population = np.vstack(new_step[:,0])
                acceptance = np.vstack(new_step[:, 1])
                #jumps = new_step[:,2]
               # n_id = new_step[:,3]
                loop_population = self.swap_temperatures(loop_population)

            else:

                var_pop = np.var(loop_population, axis=0)
                loop_population = []
                acceptance = []
                #jumps = []
                #n_id = []

                for j, ind in enumerate(indexes):

                    new_step = self.new_individual(ind[0], ind[1],var_pop)
                    loop_population.append(new_step[0])
                    acceptance.append(new_step[1])
                    #jumps.append(new_step[2])
                    #n_id.append(new_step[3])

                loop_population = np.array(loop_population)
                acceptance = np.array(acceptance)
                loop_population = self.swap_temperatures(loop_population)

                #jumps = np.array(jumps)
                #n_id = np.array(n_id)

            #if loop<0.1*self.max_iteration:

            #    jumps = np.sum(jumps,axis=0)
            #    mask = jumps == 0
            #    jumps[mask] = np.max(jumps)
            #    Jumps += jumps

            #    n_id = np.sum(n_id, axis=0)
            #    mask = n_id == 0
            #    n_id[mask] = 1
            #    N_id += n_id

            #    pCR = Jumps/N_id
            #    self.prob_crossover = pCR/np.sum(pCR)

            all_population.append(loop_population)
            all_acceptance.append(acceptance)

            #print(accepted,np.min(loop_population[:,-1]))
            #import pdb;
            #pdb.set_trace()

            #mask = accepted<0.1
            #self.scale[mask] /=2

            #mask = accepted > 0.4
            #self.scale[mask] *= 2

        self.population = np.array(all_population)
        self.acceptance = np.array(all_acceptance)
        DEMC_population = np.copy(self.population)

        print(self.swap/self.max_iteration)
        computation_time = python_time.time() - start_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')

        best_model_index = np.where(DEMC_population[:, :, -1] == DEMC_population[:, :, -1].min())
        fit_results = DEMC_population[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], :-1]
        fit_log_likelihood = DEMC_population[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0],-1]

        print('best_model:', fit_results, '-ln(likelihood)', fit_log_likelihood)

        self.fit_results = {'best_model': fit_results, '-ln(likelihood)': fit_log_likelihood,
                            'DEMC_population': DEMC_population, 'fit_time': computation_time}
