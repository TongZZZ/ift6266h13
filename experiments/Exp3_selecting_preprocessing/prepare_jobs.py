yamlTemplate = """
 !obj:pylearn2.train.Train {
    dataset: &train !obj:contest_dataset.ContestDataset {
        which_set: 'train',
        start: 0,
        stop: 3500,
        preprocessor: !pkl: "%(preprocess)s",
        fit_preprocessor: False,
        fit_test_preprocessor: %(fitTest)s,
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [ !obj:pylearn2.models.mlp.RectifiedLinear {
                     layer_name: 'h0',
                     dim: %(dimHO)i,
                     sparse_init: 15,
                 }, !obj:pylearn2.models.mlp.Softmax {
                     layer_name: 'y',
                     n_classes: 7,
                     irange: 0.
                 }
                ],
        nvis: 2304,
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        monitoring_dataset:
            {
                'train' : *train,
                'test'  : !obj:contest_dataset.ContestDataset {
                    which_set: 'train',
                    start: 3500,
                    stop: 4100,
                    preprocessor: !pkl: "%(preprocess)s",
                    fit_preprocessor: False,
                    fit_test_preprocessor: %(fitTest)s,
                }
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
            }, !obj:pylearn2.models.mlp.WeightDecay {
                coeffs: [ .00110, .00110 ]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "test_y_misclass",
            prop_decrease: 0.,
            N: 30
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'test_y_misclass',
             save_path: "mlp_%(dimHO)i_%(preprocess)s_%(fitTest)s.pkl"
        }
    ],
}
 """


# Read the location of the pylearn2 from the command line arguments
import sys
import os
if len(sys.argv) == 2:
    pylearnPath = sys.argv[1]
    pylearnTrainScript = "pylearn2/scripts/train.py"
    fullPathToTrainScript = os.path.join(pylearnPath, pylearnTrainScript)
else:
    raise Exception("You need to pass the path to your pylearn2 folder as an argument to this script")


nbHiddenUnitsList = [200, 300, 400, 500, 700, 1000, 1500]
preprocessorList = ["ZCA_preprocess.pkl",
                    "GCN_preprocess.pkl",
                    "DC_GCN_preprocess.pkl",
                    "DC_GCN_ZCA_preprocess.pkl",
                    "DC_GCN_STD_preprocess.pkl",
                    "STD_preprocess.pkl"]

# Generate dictionnaries of parameters for insertion in the YAML string
paramDicts = []
for nbHiddenUnits in nbHiddenUnitsList:
    for preprocessor in preprocessorList:
        paramDicts.append({'dimHO':nbHiddenUnits,
                           'preprocess' : preprocessor,
                           'fitTest' : 'True'})
        paramDicts.append({'dimHO':nbHiddenUnits,
                           'preprocess' : preprocessor,
                           'fitTest' : 'False'})
                           

# Generate the yaml files to launch the jobs
yamlFilesNames = []
for paramDict in paramDicts:
    # Generate the name and content of the yamlfile
    yamlStr = yamlTemplate % paramDict
    yamlFilename = "mlp_%(dimHO)i_%(preprocess)s_%(fitTest)s.yaml" % paramDict
    yamlFilesNames.append(yamlFilename)
    
    # Output the yaml file
    f = open(yamlFilename, 'w')
    f.write(yamlStr)
    f.close()
    
# Generate the bash file to launch the jobs on the cluster
commandTemplate = "PYTHONPATH=~/Documents/PYTHON/LIBRARIES/Cutting_edge_pylearn2/:$PYTHONPATH jobdispatch --server --condor python %s %s"
bashCommands = ["#!/bin/sh"]
bashFileName = "launch_exp3_jobs.sh"

for yamlFilename in yamlFilesNames:
    bashCommands.append(commandTemplate % (fullPathToTrainScript, yamlFilename))

f = open(bashFileName, 'w')
f.write("\n".join(bashCommands))
f.close()
    
    
    
    
    
    
    
    
        
                    

 
 
 
