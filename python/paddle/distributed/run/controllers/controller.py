# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import signal

from paddle.distributed.run.job import Job
from paddle.distributed.run.job import Pod
from paddle.distributed.run.job import Container

from .master import Master

import time


class ControleMode:
    COLLECTIVE = "collective"
    PS = "ps"


class ControllerBase(object):
    def __init__(self, ctx):
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGABRT, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.ctx = ctx

        self.master = Master.factory(self.ctx)

        self.job = Job()
        self.pod = Pod()

        self.build_job()
        self.build_pod()

        self.join_server = None

    def signal_handler(self, sigint, frame):
        self.ctx.logger.info("Termiating with signal {}".format(sigint))
        self.sigint = sigint
        self.ctx.running = False
        self.stop(sigint)
        time.sleep(1)
        sys.exit(sigint)

    def tach(self):
        '''
        for i in range(50):
            print([i]*(i+1))
            self.pod.logs()
            time.sleep(1)
        '''

        self.pod.join()
        self.master.stop()

    def run(self):
        self.ctx.logger.debug("Run pod {}".format(self.pod))

        if not self.pod.containers:
            raise "No containers in the pod"

        self.ctx.logger.debug(self.pod.containers[0])
        self.pod.deploy()

    def stop(self, sigint=None):
        self.master.stop()
        self.pod.stop(sigint)


'''
Controller API for customization
'''


class Controller(ControllerBase):
    def build_job(self):
        self.job.replicas = self.ctx.args.np or 1
        self.job.id = self.ctx.args.id

    def build_pod(self):
        raise NotImplementedError

    def _get_entrypoint(self):
        entrypoint = [sys.executable, "-u", self.ctx.args.training_script]
        entrypoint.extend(self.ctx.args.training_script_args)
        return entrypoint

    def build_container(self,
                        entrypoint=None,
                        envs={},
                        use_ctx_env=True,
                        out=None,
                        err=None):
        c = Container()
        c.entrypoint = entrypoint or self._get_entrypoint()
        c.env = self.ctx.get_envs() if use_ctx_env else {}
        c.out = os.path.join(self.ctx.args.log_dir, out) if out else None
        c.err = os.path.join(self.ctx.args.log_dir, err) if err else c.out
        c.update_env(envs)
        return c

    def add_container(self,
                      container=None,
                      entrypoint=None,
                      envs={},
                      is_init=False,
                      log_file=None):
        if not container:
            container = self.build_container(
                entrypoint=entrypoint, envs=envs, out=log_file, err=log_file)

        if is_init:
            self.pod.init_containers.append(container)
        else:
            self.pod.containers.append(container)

    def pod_replicas(self):
        if self.ctx.args.nproc_per_node:
            return int(self.ctx.args.nproc_per_node)
        else:
            return self.ctx.node.device.count

    def save_log(self, info):
        f = os.path.join(self.ctx.args.log_dir, '{}.log'.format(self.pod.name))
        try:
            with open(f, 'a') as fd:
                fd.write(str(info))
        except Exception as e:
            self.ctx.logger.error("save log failed because {}".format(e))
