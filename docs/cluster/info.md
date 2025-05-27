# MPCDF High-Performance Cluster

The Max Planck Computing and Data Facility (MPCDF) provides high-performance computing (HPC) resources to support 
computational and data-driven research within the Max Planck Society.
These resources are designed for large-scale simulations, complex data analyses, and other intensive scientific workloads.

MPCDF operates several powerful computing systems. The flagship system, **Raven**, includes over 1,500 compute nodes
based on Intel Xeon processors, many of which are equipped with large memory and high-speed interconnects.
It also features GPU-accelerated nodes with NVIDIA A100 GPUs, ideal for machine learning and parallel computations.
Another system, **Viper**, offers AMD EPYC-based nodes with high core counts and large memory configurations,
optimized for both general-purpose and memory-intensive applications.

Users access the systems remotely via secure SSH connections.
Work is typically done through a shared Linux environment with user-configurable software modules.
Job scheduling is managed using the Slurm workload manager, allowing users to submit computational tasks that
are distributed across the available resources.
The systems support batch processing as well as interactive sessions for development and debugging.

Important Links:

- [Request Access](https://selfservice.mpcdf.mpg.de/index.php?r=site%2Flogin)
- [Raven Documentation](https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html)
- [Viper CPU Documentation](https://docs.mpcdf.mpg.de/doc/computing/viper-user-guide.html)
- [Viper GPU Documentation](https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html)

# FAQ

In general, please check the [MPCDF FAQ](https://docs.mpcdf.mpg.de/faq/index.html).
Here are some specific questions that we found useful.

## What software is available on the cluster?

MPCDF uses as _module_ system that allows you to load software packages on demand.
Please [read here for more information](https://docs.mpcdf.mpg.de/doc/computing/software/environment-modules.html).

## Do I need to type my password every time I'm logging in with SSH

No. On Linux and macOS, you can use an SSH ControlMaster configuration that allows you to reuse your SSH
connection for several hours. This setup also lets you directly log into the raven and viper nodes without having to
go through the gateway machines.
[Read here](https://docs.mpcdf.mpg.de/faq/connecting.html#how-can-i-avoid-having-to-type-my-password-repeatedly)
      
## Can I test computations interactively?

Yes, to some degree.
The login nodes like `raven-i` or `viper-i` are suitable for interactive use, but they are not suitable for running
computations because they are too weak, are used by too many people, and don't contain GPUs.
However, you can specify dedicated [SLURM partitions] and `srun` to get a real interactive session for a limited 
amount of time.

!!! note "Note"
    Different partitions have different restrictions for the resources. E.g., an interactive GPU session on Raven
    cannot exceed 15 minutes.

Here are the commands for various partitions but also read further for info for
[Raven](https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html#interactive-debug-runs),
[Viper GPU](https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html#interactive-debug-runs),
and [Viper CPU](https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html#interactive-debug-runs):

Raven interactive CPU session for 30 minutes:

```shell
srun --verbose -p interactive --time=00:30:00 --pty bash
```

Raven interactive GPU session with an A100 GPU for 15 minutes:

```shell
srun --verbose --gres=gpu:a100:1 -p gpudev --time=15 --pty bash
```

## How do I find out about the properties of SLURM partitions?

All SLURM commands start with an `s`.
First, you can list all available partitions in a nicely formatted table like this:

```shell
sinfo -o "%20P %5D %10c %10m %25f %10G"
```

On Raven, the output will look like the following, and it will show you the partition names and their properties:

```shell
PARTITION            NODES CPUS       MEMORY     AVAIL_FEATURES            GRES      
interactive*         2     144        512000     login,icelake             (null)    
general              144   144        240000     icelake,fhi,cpu           (null)    
general              1378  144        240000     icelake,cpu               (null)    
general              157   144        500000     icelake,gpu               gpu:a100:4
general              32    144        500000     icelake,gpu-bw            gpu:a100:4
general              4     144        2048000    icelake,hugemem           (null)    
general              64    144        500000     icelake,largemem          (null)    
small                144   144        240000     icelake,fhi,cpu           (null)    
small                1378  144        240000     icelake,cpu               (null)    
gpu                  157   144        500000     icelake,gpu               gpu:a100:4
gpu                  32    144        500000     icelake,gpu-bw            gpu:a100:4
gpu1                 157   144        500000     icelake,gpu               gpu:a100:4
gpu1                 32    144        500000     icelake,gpu-bw            gpu:a100:4
rvs                  2     144        240000     icelake,cpu               (null)    
rvs                  2     144        500000     icelake,gpu               gpu:a100:4
gpudev               1     144        500000     icelake,gpu               gpu:a100:4
```

Now, to inspect a specific partition in detail, you can use the `scontrol` command.
Since we were looking for the partition `gpudev` before, let's use this as an example:

```shell
scontrol show partition gpudev     
```

```shell
PartitionName=gpudev
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO CpuBind=cores  QoS=N/A
   DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=1 MaxTime=00:15:00 MinNodes=1 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   NodeSets=dev
   Nodes=ravg1002
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=144 TotalNodes=1 SelectTypeParameters=NONE
   JobDefaults=DefCpuPerGPU=36
   DefMemPerNode=125000 MaxMemPerNode=UNLIMITED
   TRES=cpu=144,mem=500000M,node=1,billing=144,gres/gpu=4,gres/gpu:a100=4
```
Note the `MaxTime=00:15:00` property.

## Can I share/access data of other users?

Yes. We usually share project data in `/ptmp/myuser` using `setfacl` and `getfacl`.
Please [read here for more information](https://docs.mpcdf.mpg.de/faq/hpc_systems.html#how-can-i-grant-other-users-access-to-my-files-how-do-i-use-acls).

## SLURM: What's the point in specifying memory, cores, and time requirements?

Short answer: The **less** you specify, the **quicker** your job gets scheduled.
On a system like the HPC, many users compete for recourses.
The SLURM scheduler tries to find a spot for your job that is as close to your requirements as possible.
If it can fit your job into a spot, because, e.g., another user doesn't need a whole node and your job only needs a
bit of computational power, then it will do so.
On the other hand, if you specify a lot of memory or a lot of cores, then the scheduler might need to put you at the
back of the queue.
Therefore, always specify the **least** you need.

# What's the difference between Raven and Viper?

One answer is: Raven is older and Viper is newer.
The other answer is: Raven has Intel CPUs and NVidia A100 GPUs, while Viper has AMD CPUs and GPUs.
From a practical point of view, the Viper system is interesting because the AMD architecture offers _shared memory_
between CPU and GPU cores (AMD calls these things APU), and it has 128GB per node.
When you have a GPU accelerated program and you are fine with 40GB NVidia A100 cards, then use Raven.
Otherwise, it is worth trying Viper.