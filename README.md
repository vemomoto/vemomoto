# VeMoMoTo - Vector Movement Modelling Tools 

"Vector movement modelling tools" is a collection of python packages aiming to model the movement of invasive species or disease vectors through road networks. 

 - The package hybrid_vector_model provides the core functionality for the model. 
 - The package lopaths contains an algorithm to identify locally optimal routes in road networks. These routes are in particular a necessary requirement for the hybrid vector model. 
 - The package ci_rvm contains an algorithm to identify profile likelihood confidence intervals. This algorithm is in particular used to assess the hybrid vector model
 - The package vemomoto_core provides helpful tools required by the other packages

Please refer to the [documentation page][DOC] for further details.

The packages are licensed under the [LGPL-v3][LGPL]. Please report any bugs in the 
[bug tracker][BUG]. You are welcome to adjust the provided
files to your needs and to improve the code. It is encouraged that you share any 
improvements on the github page if possible.

[DOC]: https://vemomoto.github.io
[LGPL]: https://opensource.org/licenses/lgpl-3.0.html
[BUG]: https://github.com/vemomoto/vemomoto/issues