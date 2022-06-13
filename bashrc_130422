# .bashrc
# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# do this to manually reset the path cleanly, selection manually, should be correct for the HPI cluster
# did this on 16th November, 2021
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/hpi/fs00/scratch/lena.schwertmann/Programs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/hpi/fs00/scratch/lena.schwertmann/Programs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/hpi/fs00/scratch/lena.schwertmann/Programs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/hpi/fs00/scratch/lena.schwertmann/Programs/anaconda3/bin:$PATH"
    fi
fi

unset __conda_setup
# <<< conda initialize <<<

# now manually add the conda path because I had problems with this before (16th November, 2021)
# make sure to use PATH=$PATH:conda_path, such that conda_path is appended to $PATH and does not replace it accidentally
export PATH=$PATH:/hpi/fs00/scratch/lena.schwertmann/Programs/anaconda3/bin
export PATH=$PATH:/hpi/fs00/scratch/lena.schwertmann/Programs/anaconda3/condabin

export PYTHONPATH=$PYTHONPATH:/hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis

####### BASH ALIASES  #######

# general utility commands to make working with files nicer
alias ls="ls -lah"  # 8.3.22

# commands for getting the most information out of slurm commands
alias sq="squeue -O JobID:8,UserName,Name:12,StateCompact:5,NodeList:12,ReqNodes:12,tres-per-job:14,NumCPUs:5,MinMemory:12,StartTime:22,EndTime:22,SubmitTime:22,TimeLeft:12,ReasonList" # 8.3.2022
alias sqm="squeue -u lena.schwertmann -O JobID:8,UserName,Name:18,StateCompact:5,NodeList:12,ReqNodes:12,tres-per-job:14,NumCPUs:5,MinMemory:12,StartTime:22,EndTime:22,SubmitTime:22,TimeLeft:12,ReasonList"  # 16.3.2022
alias si="sinfo -O NodeList,StateLong:10,TimeStamp,Reason,Time:12,Gres,GresUsed:30,CPUsState,Memory:10,AllocMem:10,FreeMem:10 -n ac922-[01-02],dgxa100-01,a6k5-01,ic922-01,node-[01-32]" # 14.3.2022
alias sc="sacct --format=JobID,JobName,NodeList,State,Exit,Submit,Start,End,Timelimit,Elapsed" # 21.3.2022

####### WORKING WITH THE HPI CLUSTER #######

# always start the terminal in this folder with this environment
cd /hpi/fs00/scratch/lena.schwertmann/pycharm_master_thesis/
conda activate master_thesis_130422
