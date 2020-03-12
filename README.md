# PyVKF
Python implementation of second generation Vold-Kalman Order Filter.

Vold-Kalman Filter was introduced by Håvard Vold and Jan Leuridan in 1993. VKF extracts non-stationary periodic components from a signal using a known phasor. The solution involves solving a sparse linear system which in Python is optimally performed with spsolve. The second generation VKF, implemented here can extract multiple orders in parallel [2].

![alt text](https://github.com/CyprienHoelzl/PyVKF/blob/master/VKF_Example.png)

# References
[1] Vold, H. and Leuridan, J. (1993), High resolution order tracking at extreme slew rates, using Kalman tracking filters. Technical Report 931288, Society of Automotive Engineers.

[2] Tuma, J. (2005), Setting the passband width in the Vold-Kalman order tracking filter. Proceedings of the International Congress on Sound and Vibration, Lisbon, Portugal.
