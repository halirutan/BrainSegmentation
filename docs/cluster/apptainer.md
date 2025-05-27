# Building an Apptainer on MPCDF

Apptainer (formerly known as Singularity) is a container platform specifically designed for scientific computing and
High-Performance Computing (HPC) environments. Unlike traditional containers, Apptainer is built with security and
compatibility with research computing systems in mind. It allows scientists and researchers to package their entire
computational environment, including the operating system, software, libraries, and data, into a single, portable
container that can run consistently across different computing systems.

Building an Apptainer can be tricky, sometimes painful.
However, in general you can think of it as setting up a new machine.
The recipy of setting up your machine is stored in a `.def` text file which defines step-by-step what you will 
install and set up in your Apptainer.
The `.def` file has several `%name` sections that define different stages and the behavior when
building and running your Apptainer.


1. You decide which operating system you want to use. In scientific computing, we usually use Linux. That is the
   _base container_ your Apptainer will build on.
2. What you will get is a fresh Linux installation. Now you can install the software you need, and you can use the
   distribution's package manager. So on an Ubuntu, you use `apt get`.
3. After your system is set up, you can download and install additional things like Python environments or code from
   GitHub repositories.

In the end, your goal is to have a fully self-contained Apptainer that is able to run the computations you need.
The container itself is just a `.sif` file and starting or running your new _machine_ is as easy as calling
`apptainer` with your `.sif` file.

## Building the BrainSegmentation Apptainer

To build a container that contains conda, the BrainSegmentation code, SynthSeg, CHARM Segmentation, and different
conda environments for each, you can use the following on the MPCDF cluster:

```shell
git clone https://github.com/halirutan/BrainSegmentation.git

module purge
module load apptainer

apptainer build mpcdf_raven.sif BrainSegmentation/apptainer/mpcdf_raven.def
```

The `apptainer` directory also contains an example SLURM script with an accompanying shell script to run a test.
However, you will need to adapt the paths in the scripts to your needs.

