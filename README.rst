Contents
^^^^^^^^^^^^^^^^

* ``./Select_dataset`` contains the method code for selecting the profile set
* ``./TauDL_Create`` contains the main deep learning training code of the article and the corresponding training functions
  (There is no dataset because some data requires permission, while others can be downloaded by yourself)
  (ARMS needs to seek the consent of weng et al. Reference: Advanced Radiative Transfer Modeling System (ARMS) A New-Generation Satellite Observation Operator Developed for Numerical Weather Prediction and Remote Sensing  Applications )
* ``./interface``
    Contains the C++ wrapper code and header files required to bridge the Fortran-based ODPSDL core with the PyTorch LibTorch backend. 

Dependencies
^^^^^^^^^^^^^^^^
To run the training code and inference interfaces, ensure the following environment is set up:

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Python                    | 3.13.5                        |
+---------------------------+-------------------------------+
| PyTorch                   | 2.7.1                         |
+---------------------------+-------------------------------+
| LibTorch (C++ Distribution)| 2.7.1 (Matching PyTorch)     |
+---------------------------+-------------------------------+

Compilation
^^^^^^^^^^^^^^^^
This project implements a robust **Fortran-C++-Python** interoperability layer to integrate deep learning models into the legacy Fortran-based ARMS system. The technical workflow is as follows:

1. **Model Export (Python -> TorchScript)**
   Trained PyTorch models are exported to **TorchScript** format (``.pt`` files). This serializes the model architecture and weights into a language-agnostic representation that can be loaded by the C++ LibTorch API without requiring a Python interpreter at runtime.

2. **C++ Wrapper Layer (The Bridge)**
   A dedicated C++ interface (``dl_interface.cpp``) acts as the middleware:
   * **Loading**: Initializes the LibTorch runtime and loads the ``.pt`` model.
   * **Tensor View**: Accepts raw memory pointers from Fortran and constructs ``torch::Tensor`` objects using ``torch::from_blob()``. This achieves **zero-copy** data exchange, critical for performance.
   * **Forward Pass**: Executes ``module.forward(inputs)`` to predict transmittance.
   * **Adjoint/Jacobian Mode**: For data assimilation, inputs are tagged with ``requires_grad(true)``. After the forward pass, ``.backward()`` is invoked to compute gradients via automatic differentiation. The resulting gradient tensors are extracted and passed back to Fortran.

3. **Fortran Integration (ISO_C_BINDING)**
   The ARMS Fortran code utilizes the ``ISO_C_BINDING`` module to:
   * Define C-interoperable interfaces for the C++ functions.
   * Pass atmospheric profile arrays (Temperature, Humidity, Pressure, etc.) by reference (pointers) to the C++ layer.
   * Receive the computed transmittance and Jacobian matrices directly into Fortran arrays for subsequent radiative transfer calculations.
