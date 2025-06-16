# Variational Problems – TFG Repository

This repository contains all the source code and simulation materials used in the Bachelor's Thesis on variational problems and their application to image processing and classical mechanics.

## Folder Structure

Each main folder corresponds to a specific part of the project. Most folders contain a `results/` subfolder, where the output figures and data from the simulations are stored. These results are the ones included and referenced throughout the thesis document.

If any simulation requires an input image or dataset, it is included directly in the corresponding main folder.

### `Brachistochrone/`
Code and plots for the classical brachistochrone problem and the code to obtain an evolution of the cycloid. Solves the variational formulation to compute the curve of fastest descent.

### `Catenary/`
Simulates the catenary curve based on energy minimization. Includes visualizations and comparison with the analytical solution.

### `Chan_vese_1/`
Initial implementation of the Chan–Vese active contour model, focused on detecting simple geometric shapes to validate the basic functionality of the algorithm. This version also compares different regularized delta functions, helping to identify the most suitable one for later simulations. It uses level set methods and smooth Heaviside approximations.

### `Chan_vese_2/`
An imporved version of the Chan_Vese_1 model that focuses on refining parameter selection and improving the numerical stability of the segmentation. This implementation includes more realistic test cases and emphasizes robustness in contour evolution by testing its efficiency on images with multiple shapes. It serves as a bridge between the initial tests and the final version.

### `Chan_vese_3/`
The final version applies the Chan–Vese model to real medical images, focusing on practical segmentation performance. It includes clearer visual outputs and was used to generate the figures presented in the thesis.

### `Morphological_Chan_Vese/`
An alternative segmentation method using morphological operations instead of PDEs, and based on example of `scikit-image` repositories. Useful for comparing results and performance with our Chan–Vese model implementation.

---

## Notes

- Output results for each simulation are stored in the `results/` subfolders inside each main directory.
- If a simulation requires input images, they are provided in the same folder as the corresponding code.
- All figures included in the thesis have been generated using the code in this repository.
- Some implementations were inspired by or adapted from publicly available segmentation libraries and academic resources, especially for the Chan–Vese model and morphological operations. Proper references are acknowledged in the thesis document.
- The Chan–Vese implementations were partly adapted from educational examples and open-source contributions, including segmentation routines from `scikit-image` and academic GitHub repositories.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- OpenCV (for image processing)
- [Optional] R (used for analytical curve simulations of the cycloid)

## License
This repository is provided for academic and research purposes related to the TFG submitted at Universidad Carlos III de Madrid.

## Contact
For any questions or feedback, please contact the author via GitHub.

