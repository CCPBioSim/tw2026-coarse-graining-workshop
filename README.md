# Coarse-graining-workshop-2026

This workshop source repository contains the build recipe for a docker container derived from the CCPBioSim JupyterHub image. This container adds the necessary software packages and notebook content to form a deployable course container.

Here we present a practical introduction to the construction, simulation and analysis of coarse-grained (CG) membrane protein system.

We have separated the content into tutorial notebooks:

- Preparing a protein with Martini 3
- Building the system (with a membrane)
- Preparing for production simulation
- Analysis of the simulation
- Conversion to atomistic resolution

The user should work sequentially through each numbered directory within this repository, starting from tutorial_1

This tutorial is inspired by the previous Coarse-graining workshop, run by Robert Clark and Iain Peter Shand Smith, in combination with those found on the [Martini tutorial section](https://cgmartini.nl/docs/tutorials/Martini3/tutorials.html). 

The work that inspired the system studied in these tutorials can be found here:

**Hexokinase-I directly binds to a charged membrane-buried glutamate of mitochondrial VDAC1 and VDAC2**. 
_Sebastian Bieker, Michael Timme, Nils Woge, Dina G. Hassan, Chelsea M. Brown, Siewert J. Marrink, Manuel N. Melo & Joost C. M. Holthuis_
[https://doi.org/10.1038/s42003-025-07551-9](https://doi.org/10.1038/s42003-025-07551-9)

How to Use
---
This training course is deployed on the CCPBioSim website via our cloud infrastructure, however you can deploy it on your own machine with docker.

Pull the container from our repository::

docker pull ghcr.io/ccpbiosim/coarse-graining-workshop:latest
In our containers, we are using the JupyterHub default port 8888, so you should forward this port when deploying locally::

docker run -p 8888:8888 ghcr.io/ccpbiosim/coarse-graining-workshop:latest

Authors
---
Workshop Content Authors:

- [Chelsea M. Brown](https://orcid.org/0000-0003-2006-5015)

Contact
---
Please direct all questions and feedback to Chelsea M. Brown (chelsea.m.brown[at]warwick.ac.uk)
