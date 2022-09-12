"""Factories

TransformerBasedClassificationFactory from small_text.

"""
from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.transformers.classifiers.classification import TransformerBasedClassification

class TransformerBasedClassificationFactory(AbstractClassifierFactory):

    def __init__(self, transformer_model, num_classes, num_epochs, kwargs={}):
        self.transformer_model = transformer_model
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.num_epochs = num_epochs


    def new(self):
        return TransformerBasedClassification(self.transformer_model,
                                              self.num_classes,
                                              self.num_epochs,
                                              **self.kwargs)
