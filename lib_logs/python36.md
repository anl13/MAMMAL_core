apt-get update \
&& apt-get install -y software-properties-common curl \
&& add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update \
&& apt-get install -y python3.6 python3.6-venv


or install from source.
Download source and uncompress it. run
`./configure
make
make install`
Before configure, you may run
`apt install libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.5 libgdm-dev libdb4o-cil-dev libpcap-dev` to install some libs for pip to work well.
