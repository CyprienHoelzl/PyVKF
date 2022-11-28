# PyVKF
Python implementation of second generation Vold-Kalman Order Filter.

Vold-Kalman Filter was introduced by Håvard Vold and Jan Leuridan in 1993. VKF extracts non-stationary periodic components from a signal using a known phasor. The solution involves solving a sparse linear system which in Python is optimally performed with spsolve. The second generation VKF, implemented here can extract multiple orders in parallel [2]. A matlab version of the same script, originally written by Maarten van der Seijs is the base of PyVKF [3].

VKF is adapted for filtering signals from rotary equipments components such as bearings and axles, where the frequency components are known and the signal is corrupted by noise. PyVKF has been successfully used to filter signals from Axle Box Accelerometers on Railway vehicles for filtering effects originating from vehicle and track. 
The VKF is proposed for decomposing the signal into periodic wheel and track related excitation--response pair functions. We demonstrate that these components are in turn correlated to operational conditions, such as wheel out-of-roundness and the rail type.

# How to run
* Create a new environment with python 3.7
* Install the requirements using:

```
(VKF)$ pip install -r requirements.txt
```

* Run the demo: [VKF Demo on synthetic data](VoldKalmanFilter.py).

# Issues or Questions
If you have comments, you can write an issue at GitHub so that everyone can read it along with my response. Please don't view it as a way to report bugs only. 

# References
[1] Vold, H. and Leuridan, J. (1993), High resolution order tracking at extreme slew rates, using Kalman tracking filters. Technical Report 931288, Society of Automotive Engineers.

[2] Tuma, J. (2005), Setting the passband width in the Vold-Kalman order tracking filter. Proceedings of the International Congress on Sound and Vibration, Lisbon, Portugal.

[3] Maarten van der Seijs (2020). Second generation Vold-Kalman Order Filtering (https://www.mathworks.com/matlabcentral/fileexchange/36277-second-generation-vold-kalman-order-filtering), MATLAB Central File Exchange

