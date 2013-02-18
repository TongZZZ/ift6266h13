import os
import cmd
import sys
import cPickle
import collections
from pylearn2.utils import serial

class Menu(cmd.Cmd):
    
    def __init__(self, modelNames, modelAttributes, attributeList):
        self.modelNames = modelNames
        self.modelAttributes = modelAttributes
        self.attributeList = attributeList
        cmd.Cmd.__init__(self)
    
    def do_list_models(self, arg):
        """ list_models
        Display the list of loaded models"""
        print ""
        self.printSectionTitle("List of the available models")
        for name in self.modelNames:
            print name
        print ""
            
    def do_list_attributes(self, arg):
        """ list_attributes
        Display the list of attributes by which the models can be listed"""
        print ""
        self.printSectionTitle("List of the available model attributes")
        for att in self.attributeList:
            print att  
        print ""
        
    def do_sort_models(self,attribute):
        """ sort_models <attribute>
        Display the list of loaded models, sorted by the specified attribute
        """
        
        if attribute not in self.attributeList:
            print ""
            print "None of the model have this attribute"
            print ""
            return 
        
        # Two lists of the models. The first one contains the names of the
        # models possessing the required attribute, sorted by this attribute.
        # The second one contains the the names of the models that do not
        # possess the required attribute.
        modelsWithAttribute = []
        modelsWithoutAttribute = []
       
        for i in range(len(self.modelNames)):
            
            if attribute in self.modelAttributes[i]:
                modelsWithAttribute.append((self.modelAttributes[i][attribute],
                                            self.modelNames[i]))
            else:
                modelsWithoutAttribute.append(self.modelNames[i])
                
        modelsWithAttribute.sort()
        
        # Output the model lists
        print ""
        if len(modelsWithAttribute) > 0 :
            self.printSectionTitle("Models possessing the attribute '%s'" %
                                   str(attribute))
            for (value,name) in modelsWithAttribute:
                print "%s : %f" % (name, float(value))
        
        if len(modelsWithoutAttribute) > 0 :
            self.printSectionTitle("Models NOT possessing the attribute '%s'" %
                                   str(attribute))
            for name in modelsWithoutAttribute:
                print name  
        print ""
                 
        
    def do_EOF(self, line):
        return True
        
    def postloop(self):
        print
        
    def printSectionTitle(self, str):
        print str
        print '=' * len(str)


print "----------------------------------------------------------------------"


# Extract the base folder to use to load the models
baseFolder =  "./"
if len(sys.args) >= 2:
else:
baseFolder =  "./"

# Find all the model files in subfolders of the current folder
print "Exploring subfolders to find model files"
modelFilenames = []
for path, subdirs, files in os.walk(baseFolder):
    for name in files:
        if name[-4:] == ".pkl":
            fullPath = os.path.join(path, name)
            modelFilenames.append(fullPath)
print "Exploration complete, %i models have been found" % len(modelFilenames)


print "----------------------------------------------------------------------"


# Load the models that have been found and extract the caracteristics for each
print "Loading the models"
modelProperties = [None,] * len(modelFilenames)
for i in range(len(modelFilenames)):
    
    # Load the model
    model = serial.load(modelFilenames[i])
    monitor = model.monitor
    channels = monitor.channels
    
    # Extract the model caracteristics and store them in a dictionnary
    modelProperties[i] = collections.OrderedDict()
    
    modelProperties[i]['epochs seen'] = monitor._epochs_seen
    modelProperties[i]['time trained'] = max(channels[key].time_record[-1] for key in channels)
    
    for key in sorted(channels.keys()):
        modelProperties[i][key] = channels[key].val_record[-1]
        
    print "Model %s was loaded" % modelFilenames[i]    
print "Models loaded"


# Compute the union of the attributes of all the models
attributeList = []
for attributes in modelProperties:
    for att in attributes:
        if att not in attributeList:
            attributeList.append(att) 


print "----------------------------------------------------------------------"


menu = Menu(modelFilenames, modelProperties, attributeList)
print "To exit the system at any time, press Ctrl-d"
print "If you need assistance, use the 'help' command"
menu.cmdloop()
