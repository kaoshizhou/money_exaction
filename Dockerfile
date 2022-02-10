### 根据自己需求改动 ###
FROM ccr.ccs.tencentyun.com/deepwisdom/dw_ubuntu1804_pytorch180_py37_x86_64_455.45_cuda11.1:v1.0
#####################

##### 固定指令，勿动 #####
RUN mkdir -p /app/publicdata/pretrain && mkdir -p /app/tianji
WORKDIR /app/tianji
# 将at_template文件夹下的全部文件移动至顶层工作目录
COPY at_template/ /app/tianji/
# 安装at相关的依赖
RUN pip3 install -r at_requirements.txt -i https://mirror.baidu.com/pypi/simple
RUN pip3 install drpc[all]==0.2.0 -i https://pypi.deepwisdomai.com/root/dev
########################

### 根据自己需求改动 ###
# 将当前repo全部代码放入镜像
COPY . /app/tianji/
# 删除原at_template文件夹，避免冗余
RUN rm -rf /app/tianji/at_template
# 安装task相关的依赖
RUN pip3 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
#####################

##### 固定指令，勿动 #####
EXPOSE 8080
########################