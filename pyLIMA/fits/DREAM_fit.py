import sys
import time as python_time

import numpy as np
from pyLIMA.fits.ML_fit import MLfit
from tqdm import tqdm
from pyLIMA.priors import parameters_priors


class DREAMfit(MLfit):
    """
    Under Construction
    """
    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DEMC_population_size=10,
                 max_iteration=10000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry,
                         telescopes_fluxes_method=telescopes_fluxes_method)

        self.population = []  # to be recognize by all process during parallelization
        self.DEMC_population_size = DEMC_population_size  # Times number of dimensions!
        self.max_iteration = max_iteration
        self.priors = parameters_priors.default_parameters_priors(self.fit_parameters)

    def fit_type(self):
        return "Differential Evolution Markov Chain"

    def objective_function(self, fit_process_parameters):

        objective = self.standard_objective_function(fit_process_parameters)
        return objective


    ### From scipy.DE
    def unscale_parameters(self, trial):
        """Scale from a number between 0 and 1 to parameters."""
        # trial either has shape (N, ) or (L, N), where L is the number of
        # solutions being scaled

        unscaled = self.scale_arg1 + (trial - 0.5) * self.scale_arg2

        return unscaled

    def scale_parameters(self, parameters):
        """Scale from parameters to a number between 0 and 1."""

        scaled = (parameters - self.scale_arg1) / self.scale_arg2 + 0.5

        return scaled

    def new_individual(self, parent0, parent1, parent2, parent3):

        # np.random.seed(random.randint(0, 100000))

        # number_of_parents = np.random.randint(5, 10) * 2

        # indexes = np.random.choice(len(pop), number_of_parents, replace=False)
        # print(parent1[-1],indexes)
        # try:

        #    index1 = np.random.choice(len(pop), number_of_parents, replace=False)

        # except:

        #    index1 = np.random.choice(len(pop), number_of_parents)

        # index2 = np.random.choice(len(pop[0]), number_of_parents, replace=False)
        # crossover_index = np.random.choice(len(self.crossover), 1,
        # p=self.prob_crossover)
        # crossover = self.crossover[crossover_index]
        crossover = np.random.uniform(0.0, 1.0, len(parent0[:-1]))

        mutate = np.random.uniform(0, 1, len(parent0[:-1])) < crossover
        # breakpoint()
        # mutate = np.random.uniform(0, 1, len(parent1[:-1])) < -1
        if True not in mutate:

            rand = np.random.randint(0, len(mutate))
            mutate[rand] = True

        # mutation = np.random.uniform(-2, 2, len(parent1[:-1]))
        eps1 = 10 ** -3
        eps2 = 10 ** -7

        mutation = np.random.uniform(1 - eps1, 1 + eps1, len(parent0[:-1]))
        shifts = np.random.normal(0, eps2, len(parent0[:-1]))  # *self.scale

        gamma = 2.38 / (2 * len(parent0[:-1][mutate])) ** 0.5  # *self.scale

        # gamma = 2.38 / (2 * len(index1[::2]) * len(parent1[:-1][mutate])) ** 0.5
        jumping_nodes = np.random.randint(0, 10)

        # if jumping_nodes == 5:

        # mutation = np.ones(len(parent1[:-1]))
        #    gamma = 1

        mutation *= gamma

        # progress1 = np.sum([pop[i] for i in indexes[::2]], axis=0)
        # progress2 = np.sum([pop[i] for i in indexes[1::2]], axis=0)
        # progress1 = np.sum([pop[i][j] for i in index1[::2] for j in index2[::2]],
        # axis=0)
        # progress2 = np.sum([pop[i][j] for i in index1[1::2] for j in index2[1::2]],
        # axis=0)

        progress1 = parent1
        progress2 = parent2

        progress = (progress1[:-1] - progress2[:-1]) * mutation

        if jumping_nodes == 9:
            #
            # snooker

            dz = parent0 - parent3
            # progress = dz[:-1]
            zp1 = np.dot(parent1, dz.T)
            zp2 = np.dot(parent2, dz.T)
            progress = np.random.uniform(1.2, 2.2) * (zp1 - zp2) * dz / np.dot(dz, dz.T)
            progress = progress[:-1]
        # breakpoint()

        if np.all(np.isfinite(progress)):

            pass

        else:

            progress = np.ones(len(parent0[:-1]))
        # print(gamma,progress)

        child = np.copy(parent0)

        child[:-1][mutate] += progress[mutate]
        child[:-1] += shifts

        # jump = np.zeros(len(self.crossover))
        # nid = np.zeros(len(self.crossover))
        # nid[crossover_index] += 1
        accepted = np.zeros(len(parent0[:-1]))

        for ind, param in enumerate(self.fit_parameters.keys()):

            if (child[ind] < 0) | (child[ind] > 1):
                progress[ind] = (parent1[ind] - parent0[ind]) / 2
                child[ind] = parent0[ind] + progress[ind]

                # child[ind] = parent1[ind]
                # progress[ind] = 0
                # mutate[ind] = False

                # return parent0, accepted#, jump, nid

        objective = self.objective_function(self.unscale_parameters(child[:-1]))
        # breakpoint()
        casino = np.random.uniform(0, 1)
        probability = np.exp((-objective + parent0[-1]))

        if probability > casino:

            child[-1] = objective
            # jump[crossover_index] += np.sum(progress[mutate]**2/var_pop[:-1][mutate])
            accepted[mutate] += 1
            # self.all.append(child)
            return child, accepted  # , jump, nid

        else:
            child[-1] = objective
            # self.all.append(child)
            return parent0, accepted  # , jump, nid

    def swap_temperatures(self, population):

        pop = np.copy(population)
        number_of_swap = int(len(population) / 5)

        if (number_of_swap % 2) == 0:

            pass

        else:

            number_of_swap += 1

        choices = np.random.choice(len(population), number_of_swap, replace=False)

        for ind in np.arange(0, number_of_swap, 2):

            index = choices[ind]
            index2 = choices[ind + 1]

            MH_temp = population[index2, -1] * self.betas[index] + population[
                index, -1] * self.betas[index2]
            MH_temp -= population[index, -1] * self.betas[index] + population[
                index2, -1] * self.betas[index2]

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

    def fit(self, initial_population=[], computational_pool=None):

        start_time = python_time.time()
        bounds_min = [self.fit_parameters[key][1][0] for key in
                      self.fit_parameters.keys()]
        bounds_max = [self.fit_parameters[key][1][1] for key in
                      self.fit_parameters.keys()]

        self.scale_arg1 = 0.5 * (np.array(bounds_min) + np.array(bounds_max))
        self.scale_arg2 = np.fabs(np.array(bounds_min) - np.array(bounds_max))

        # n_crossover = len(self.fit_parameters.keys())
        n_crossover = 3
        self.crossover = np.arange(1, n_crossover + 1) / n_crossover
        self.prob_crossover = np.ones(n_crossover) / n_crossover
        self.scale = 1
        number_of_walkers = int(
            np.round(self.DEMC_population_size * len(self.fit_parameters)))
        self.number_of_walkers = number_of_walkers
        self.swap = np.zeros(number_of_walkers)
        Z = []

        if initial_population == []:

            import scipy.stats as ss
            sampler = ss.qmc.LatinHypercube(d=len(self.fit_parameters))
            # self.betas = np.logspace(-3, 0, number_of_walkers)
            # self.betas = np.linspace(0,1, number_of_walkers)
            for j in range(len(self.fit_parameters) * 5):
                for i in range(number_of_walkers):

                    individual = sampler.random(n=1)[0]

                    for ind, j in enumerate(self.fit_parameters.keys()):
                        individual[ind] = individual[ind] * (
                                self.fit_parameters[j][1][1] -
                                self.fit_parameters[j][1][0]) + \
                                          self.fit_parameters[j][1][0]

                    individual = np.array(individual)
                    individual = np.r_[individual]

                    objective = self.objective_function(individual)
                    individual = np.r_[self.scale_parameters(individual), objective]
                    Z.append(individual.tolist())


        else:

            Z = [self.scale_parameters(i[:-1]).tolist() + [i[-1]] for i in
                 + initial_population]

        initial_population = np.array(Z[-number_of_walkers:])

        all_population = []
        all_population.append(initial_population)
        all_acceptance = []

        loop_population = np.copy(initial_population)
        self.all = Z.copy()
        # Jumps = np.ones(len(self.crossover))
        Z_prime = np.array(Z)
        # N_id = np.ones(len(self.crossover))
        for loop in tqdm(range(self.max_iteration)):

            parent_indexes = np.random.choice(len(Z_prime), 3 * number_of_walkers,
                                              replace=False)
            parents1 = Z_prime[parent_indexes[::2]]
            parents2 = Z_prime[parent_indexes[1::2]]
            parents3 = Z_prime[parent_indexes[2::2]]

            indexes = [(loop_population[i], parents1[i], parents2[i], parents3[i]) for i
                       in range(len(loop_population))]

            if computational_pool is not None:
                # breakpoint()

                new_step = computational_pool.starmap(self.new_individual, indexes)
                loop_population = np.array([i[0] for i in new_step])
                acceptance = np.array([i[1] for i in new_step])
                # jumps = new_step[:,2]
            # n_id = new_step[:,3]
            # loop_population = self.swap_temperatures(loop_population)

            else:

                # var_pop = np.var(loop_population, axis=0)
                loop_population = []
                acceptance = []
                # jumps = []
                # n_id = []

                for j, ind in enumerate(indexes):
                    new_step = self.new_individual(ind[0], ind[1], ind[2], ind[3])
                    loop_population.append(new_step[0])
                    acceptance.append(new_step[1])
                    # jumps.append(new_step[2])
                    # n_id.append(new_step[3])

                loop_population = np.array(loop_population)
                acceptance = np.array(acceptance)
                # loop_population = self.swap_temperatures(loop_population)

                # jumps = np.array(jumps)
                # n_id = np.array(n_id)

            # if loop<0.1*self.max_iteration:

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
            # breakpoint()

            all_population.append(loop_population)
            all_acceptance.append(acceptance)
            #breakpoint()

            if loop % 10 == 0:
                Z += loop_population.tolist()
                Z_prime = np.array(Z)

            # if loop%1000==0:
            #    accepted = np.mean([np.any(i==1,axis=1).sum() for i in
            #    all_acceptance[-1000:]])/len(loop_population)

            #    if (accepted>0.9):# and (self.scale<100):

            #        self.scale *= 2

            #    if (accepted<0.1):# and (self.scale>0.01):

            #        self.scale /= 2
            # breakpoint()

            print(loop, self.scale, np.array(all_population)[:, :, -1].min())

        self.population = np.array(all_population)
        self.population[:, :, :-1] = self.unscale_parameters(self.population[:, :, :-1])
        self.acceptance = np.array(all_acceptance)
        DEMC_population = np.copy(self.population)
        self.Z = Z_prime
        breakpoint()
        print(self.swap / self.max_iteration)
        computation_time = python_time.time() - start_time
        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')

        best_model_index = np.where(
            DEMC_population[:, :, -1] == DEMC_population[:, :, -1].min())
        fit_results = DEMC_population[np.unique(best_model_index[0])[0],
                      np.unique(best_model_index[1])[0], :-1]
        fit_log_likelihood = DEMC_population[
            np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], -1]

        print('best_model:', fit_results, self.loss_function, fit_log_likelihood)

        self.fit_results = {'best_model': fit_results,
                            self.loss_function: fit_log_likelihood,
                            'DEMC_population': DEMC_population,
                            'fit_time': computation_time}

    def samples_to_plot(self):

        chains = self.fit_results['DEMC_population']
        samples = chains.reshape(-1, chains.shape[2])
        samples_to_plot = samples[int(len(samples) / 2):]

        return samples_to_plot
