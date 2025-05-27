# Motivation for this project
In October 2022, during my computational physics class, I was introduced to a Python module called Visual Python, also known as VPython. 
This powerful module opened my eyes to the possibilities of simulating physical theories on my computer. Unsurprisingly, I was excited when I was told that I could choose any physics-related topic for my final project. 
It didn’t take long for me to decide to work on one of the most fascinating and bizarre objects in the universe: a black hole. 
I went to my professor’s office to discuss my interest in simulating a non-rotating black hole—a topic I had only encountered in movies and books like A Brief History of Time. This project is a continuation
of years of interest.
# The physics of black holes.
The program I decided to work on simulates gravitational lensing by a non-rotating, aka Schwarzschild, black hole using a simplified weak-field approximation of General Relativity. We will briefly 
see what these names are and how they fit in Python.
# Gravitational Lensing & Einstein’s Deflection Formula
One of the most Impresive effects of black holes on space is Gravitational lensing. Gravitational lensing is a result of Einstein’s General Relativity, where massive objects (such as black holes) 
curve the spacetime around them, causing light to bend. The amount of deflection that a black hole causes on a light can be considered using the angle $\alpha$- also called deflection. The deflection 
angle for light passing near a Schwarzschild black hole is given by:

$$ \alpha = \frac{4GM}{c^2r} $$

where:  $ G$ is the gravitational constant, $ M$ is the mass of the black hole, $ c$ is the speed of light, $ r$ is the impact parameter (the closest distance light gets to the black hole). 

In our simulation, we approximate this deflection for a pixel at position $(x, y)$ as:

$$\alpha = \frac{4M}{r}$$

and adjust the pixel’s coordinates accordingly to simulate the bending of light.
# Python Implementation
The core idea is that each pixel in the background image represents a photon that gets deflected by gravity. Instead of following the full geodesic path of 
light(which if we did will add complexity to our computer program),
we approximate the deflection by shifting pixels based on the lensing formula.

By creating a grid of coordinates where each pixel's position is represented by (x, y) we can calculate the distance $r$ from the black hole center.
# Limitations of the project and further works
Our simulation needs a couple of additional mathematical and computational work to consider more black hole parameter's. This code uses weak lensing assumptions which means we only considered:

1) Small deflections: The true light path is curved, but we approximate it with a shift.

2) A case where Light doesn't orbit the black hole: In reality, photons close to the event horizon can undergo strong bending or even orbit multiple times.

3) No relativistic effects: We ignore redshift, and frame-dragging (which would be present in a rotating Kerr black hole).

For a more mathematically realistic simulation, we would need ray tracing methods, solving the full geodesic equation:

$$ \frac{d^2x^\mu}{d\tau^2}+\Gamma^\mu_{\alpha \beta}\frac{dx^\alpha}{d\tau}\frac{x^\beta}{d\tau}=0$$

where $ \Gamma^\mu_{\alpha \beta}$ are Christoffel symbols from General Relativity.