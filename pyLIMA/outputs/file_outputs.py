import datetime
import json

import numpy as np


def json_output(fit_object,json_name='./pyLIMA_fit.json'):
    dictionnary = {}

    dictionnary['Event'] = {'Name':fit_object.model.event.name}
    dictionnary['Model'] = {'Type':fit_object.model.model_type()}
    dictionnary['FitParameters'] = {'Parameters':fit_object.fit_parameters}
    dictionnary['FitPriors'] = {'Parameters':fit_object.priors_parameters}
    dictionnary['FitResults'] = {}

    distributions_to_output = {}

    for key, value in fit_object.fit_results.items():

        if isinstance(value, np.ndarray):

            tosave = value.tolist()
        else:

            tosave = value

        if key == 'fit_object':

            pass

        else:

            try:

                if (key == 'MCMC_chains'):

                    header =  [key for key in fit_object.fit_parameters.keys()]
                    distributions_to_output[key] = [tosave,header]

                elif (key == 'MCMC_chains_with_fluxes'):

                    header =  [key for key in fit_object.priors_parameters.keys()] +[
                        'priors',
                                                                 fit_object.loss_function ]
                    distributions_to_output[key] = [tosave,header]

                elif (key == 'DE_population'):

                    header =  [key for key in fit_object.priors_parameters.keys()]

                    distributions_to_output[key] = [tosave,header]

                else:

                    dictionnary['FitResults'][key] = tosave

            except TypeError:

                pass


    with open(json_name, 'w') as outfile:
        json.dump(dictionnary,outfile,indent=4)


    for key in distributions_to_output.keys():

        dist_name = json_name[:-5]+'_'+key+'.npz'
        np.savez(dist_name,pyLIMA_distribution=distributions_to_output[key][0],
                 header=distributions_to_output[key][1]+[
                     fit_object.loss_function, 'priors'])

   # with open(dist_name, 'w') as outfile:
   #     json.dump(distributions_to_output,outfile,indent=4)


# def json_output(array_parameters, parameters_name, filename='parameters',
#                 output_directory='./'):
#     fit_results = {}
#
#     for index, key in enumerate(parameters_name):
#         value = array_parameters[:, index]
#         param_dic = {}
#         param_dic['value'] = value
#         param_dic['comment'] = ''
#         param_dic['format'] = 'float'
#         param_dic['unit'] = ''
#         fit_results[key] = param_dic
#
#     with open(output_directory + filename + '_.json', 'w') as outfile:
#         json.dump(fit_results, outfile)


def numpy_output(array_parameters, filename='parameters', output_directory='./'):
    np.save(output_directory + filename, array_parameters)


def latex_output(array_parameters, parameters_name, filename='parameters',
                 output_directory='./'):
    """Function to output a LaTeX format table of the fit parameters"""


#    file_path = os.path.join(output_directory, filename+'.tex')

#    t = open(file_path, 'w')

#    t.write('\\begin{table}[h!]\n')
#    t.write('\\centering\n')

#    t.write('\\begin{tabular}{lll}\n')
#    t.write('\\hline\n')
#    t.write('\\hline\n')


#    t.write('Parameters&Value')
#    t.write('\\hline\n')

#    mcmc_chains = fit.MCMC_chains
#    best_model_index = np.argmax(mcmc_chains[:, -1])

#    for index, key in enumerate(fit.model.model_dictionnary):
#        best_param = mcmc_chains[best_model_index, index]
#        percent_34 = np.percentile(mcmc_chains[:, index], 16)
#        percent_50 = np.percentile(mcmc_chains[:, index], 50)
#        percent_84 = np.percentile(mcmc_chains[:, index], 84)

#        t.write(#            key + '&' + str(best_param) + '&[' + str(percent_34) +
#        ',' + str(percent_50) + ',' + str(
#                percent_84) + ']\\\\\n')

#    t.write('Chi2&' + str(-2 * mcmc_chains[best_model_index, -1]) + '&0\\\\\n')


#    t.write('Parameters&Value&Errors')
#    t.write('\\hline\n')

#    for index, key in enumerate(fit.model.model_dictionnary):
#        t.write(key + '&' + str(fit.fit_results[index]) + '&' + str(
#        fit.fit_covariance.diagonal()[index] ** 0.5) + '\\\\\n')

#        t.write('Chi2&' + str(fit.fit_results[-1]) + '&0\\\\\n')#

#    t.write('\\hline\n')
#    t.write('\\end{tabular}\n')
#    t.write('\\end{table}\n')

#    t.close()

def pdf_output(fit, output_directory):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_directory + fit.event.name + '.pdf') as pdf:
        figure_1 = fit.outputs.figure_lightcurve
        pdf.savefig(figure_1)

        figure_2 = fit.outputs.figure_geometry
        pdf.savefig(figure_2)

        if 'figure_distributions' in fit.outputs._fields:
            figure_3 = fit.outputs.figure_distributions
            pdf.savefig(figure_3)
        pdf_details = pdf.infodict()
        pdf_details['Title'] = fit.event.name + '_pyLIMA'
        pdf_details['Author'] = 'Produced by pyLIMA'
        pdf_details['Subject'] = 'A microlensing fit'

        pdf_details['CreationDate'] = datetime.today()
