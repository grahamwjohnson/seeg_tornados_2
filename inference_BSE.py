'''
@author: grahamwjohnson
April 2025

This script is intended to be used with a fully trained BSE pulled from GitHub using torch.hub in a tagged release.
It will run all of your preprocessed data through the BSE to get embeddings.
This is intended to be used for retrospective data anylyses on your dataset. 

IMPORTANT: Data must be preprocessed using the provided preprocessing pipeline. 
Specifically, histogram equalization is used and the BSE will not function properly without proper input distribution of data. 

'''