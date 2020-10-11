# AWS Setup Instructions #

For performance testing, you will need to run it on a VM instance on Amazon Web Services (AWS). We've already sent you student coupons that you can use for billing purposes. Here are the steps for how to get setup for running on AWS.

NOTE: For those working in teams, it might be desirable for both students to use the same virtual machine. To do so, only one of you should first create the VM instance following the instructions below. Then, follow the instructions at https://cloud.google.com/compute/docs/access/granting-access-to-resources to grant access to your partner.

NOTE #2: __Please don't forget to SHUT DOWN your instances when you're done with your work for the day!  The GPU-enabled cloud VM instances you create for this assignment cost approximately $0.50 per hour.  Leaving it on accidentally for a day could quickly eat up your $50 per student quota for the assignment.__

NOTE #3: __Update your GPU quota (all regions) under IAM & admin ASAP, as new projects need approval for GPU quota in GCP which could block you from starting.__ See https://cloud.google.com/compute/docs/gpus/#restrictions for more details.
   
### Creating a VM with a GPU ###
      
1. Now you're ready to create a VM instance. Click on the button that says `Launch Instances`. Choose the `Ubuntu Server 20.04 LTS (HVM), SSD Volume Type` AMI:
![AMI Selection](handout/choose_ami.png?raw=true)

2. Choose the `p2.xlarge` GPU Instance Type and then click `4. Add Storage` on the top bar: 
![GPU instance](handout/choose_instance.png?raw=true)

3. Change the size of the `Root` volume to 64 GB to accomodate the packages we will need to install to make the instance functional for the assignment:
![Storage](handout/choose_storage.png?raw=true)

5. AWS will ask you to select a key pair. Click the first dropdown and choose `Create a new key pair` and give it whatever name you'd like. This will download a keyfile to your computer called `<key_name>.pem` which you will use to login to the VM instance you are about to create. Finally, click `Launch Instances`.
![Key Pair](handout/new_key_pair.png?raw=true)

__Note: `p2.xlarge` instances cost $0.90 / hour, so leaving one running for a whole day will consume $21.60 worth of your AWS coupon.__

4. Now that you've created your VM, you should be able to __SSH__ into it. You need the public IP address to SSH into it, which you can find on the instance page by clicking the `View Instances` button on the current page and then the instance ID for your created instance (note, it may take a moment for the instance to startup and be assigned an IP address):
![IP Address](handout/ip_address.png?raw=true)
Once you have the IP address, you can login to the instance by running this command:
~~~~
ssh -i path/to/key_name.pem ubuntu@<public_ip_address>
~~~~

5. Once you SSH into your VM instance, you'll want to install whatever software you need to make the machine a useful development environment for you.  For example we recommend:
~~~~
sudo apt update
sudo apt install make g++ freeglut3-dev # Required
sudo apt install vim 
~~~~

### Installing CUDA ###    

6. Now you need to download the CUDA 10 runtime from NVIDIA. SSH into your AWS instance and run the following:

~~~~
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
~~~~
 
5. `nvcc` is the NVIDIA CUDA compiler. The default install locates CUDA binaries in `/usr/local/cuda/bin/`, so you'll want to add this directory to your path.  For example, from a bash shell that would be:

~~~~
export PATH=$PATH:/usr/local/cuda/bin
~~~~

In general we recommend that you perform this `$PATH` update on login, so you can add this line to the end of your `.bashrc` file.  Don't forget to `source .bashrc` if you want to have this modification take effect without logging out and back in to the instance.

At this point CUDA should be installed and you should be able to run the `nvidia-smi` command to make sure everything is setup correctly.  The result of the command should indicate that your VM has one NVIDIA K80 GPU.

~~~~
ubuntu@ip-172-31-62-219:~$ nvidia-smi
Sat Oct 10 22:42:30 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.23.05    Driver Version: 455.23.05    CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   44C    P8    28W / 149W |      0MiB / 11441MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
~~~~

If you're confused about any of the steps, having problems with setting up your account or have any additional questions, reach us out on Piazza!
  
__Again, please don't forget to STOP your instances when you're done with your work for the day!__