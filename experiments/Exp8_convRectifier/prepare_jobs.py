yamlTemplate = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:contestTransformerDataset.TransformerDataset {
        raw : !obj:contest_dataset.ContestDataset {
            which_set: 'train',
            start: 0,
            stop: 3456,
            preprocessor : !obj:pylearn2.datasets.preprocessing.Standardize {},
            fit_preprocessor: True,
            fit_test_preprocessor: True,
},
        transformer : !obj:transformer.TransformationPipeline {
            input_space: !obj:pylearn2.space.Conv2DSpace {
                shape: [48, 48],
                num_channels: 1,
         },
            transformations: [
                !obj:transformer.Occlusion {},
                !obj:transformer.HalfFace {},
                !obj:transformer.Translation {},
                !obj:transformer.Scaling {},
                !obj:transformer.Rotation {},
                !obj:transformer.Flipping {}
        ] },
        space_preserving : True,
},
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 64,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [48, 48],
            num_channels: 1
     },
        layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h0',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [3, 3],
                     max_kernel_norm: 1.9365,
                     border_mode: 'full'
                 }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h1',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [3, 3],
                     max_kernel_norm: 1.9365,
                     border_mode: 'full'
                 }, !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 7,
                     istdev: .05
                 }
                ],
        dropout_include_probs: [ %(dropout1)s, %(dropout2)s, 1 ],
        dropout_input_include_prob: 1.,
        dropout_input_scale: 1.,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 64,
        learning_rate: 1e-1,
        monitoring_dataset:
            {
                'train' : *train,
                'valid' : !obj:contest_dataset.ContestDataset {
                              which_set: 'train',
                              start: 3456,
                              stop: 4160,
                              preprocessor: !obj:pylearn2.datasets.preprocessing.Standardize { },
                              fit_preprocessor: True
                          }
            },
        cost: !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
         },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass",
            prop_decrease: 0.,
            N: 10
        }
    },
    extensions:
        [ !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "mlp_RELU_%(dropout1)s_%(dropout2)s_best.pkl"
        }, !obj:pylearn2.training_algorithms.sgd.OneOverEpoch {
                start: 15,
        }
    ],
    save_path: "mlp_RELU_%(dropout1)s_%(dropout2)s.pkl",
    save_freq: 1
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

dropoutFactors = ["1.", ".9", ".8", ".7", ".6", ".5", ".4", ]

# Generate dictionnaries of parameters for insertion in the YAML string
paramDicts = []
for dropout1 in dropoutFactors:
    for dropout2 in dropoutFactors:
        paramDicts.append({'dropout1': dropout1,
                           'dropout2': dropout2})
                           

# Generate the yaml files to launch the jobs
yamlFilesNames = []
for paramDict in paramDicts:
    # Generate the name and content of the yamlfile
    yamlStr = yamlTemplate % paramDict
    yamlFilename = "mlp_RELU_%(dropout1)s_%(dropout2)s.yaml" % paramDict
    yamlFilesNames.append(yamlFilename)
    
    # Output the yaml file
    f = open(yamlFilename, 'w')
    f.write(yamlStr)
    f.close()
    
# Generate the bash file to launch the jobs on the cluster
commandTemplate = "jobdispatch --server --condor python %s %s"
bashCommands = ["#!/bin/sh"]
bashFileName = "launch_exp9_jobs.sh"

for yamlFilename in yamlFilesNames:
    bashCommands.append(commandTemplate % (fullPathToTrainScript, yamlFilename))

f = open(bashFileName, 'w')
f.write("\n".join(bashCommands))
f.close()
    
    
    
    
    
    
    
    
        
                    

 
 
 
